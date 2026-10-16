"""Tests for Structured Security and Audit Logging (Part 6).

Covers:
- Valid single-line JSON formatting and required schema fields.
- UTC ISO-8601 timestamp ending in 'Z'.
- Complete redaction of production API keys (sk-rnt- followed by 48 hex chars) to [REDACTED_API_KEY] with no suffix.
- Redaction of test/fixture keys (sk-ran-...).
- Redaction of Bearer tokens.
- Structured database URL parsing and password masking across supported schemes and encoded passwords.
- Generic secret and password redaction.
- Absence of prompt and response contents from captured logs.
- CRLF and control character injection neutralization.
- Sanitized exception messages and tracebacks without local variables.
- Concurrent request-ID isolation and cleanup in finally blocks across async tasks and exceptions (asyncio & trio).
- Handler configuration idempotency without duplicating records or clearing pytest handlers.
- Safe static JSON formatter fallback on unexpected formatting errors.
- Correct structured events and allowlisted metadata across auth, rate limiting, billing, inference, and telemetry.
- Absence of user_id, api_key_id, and other PII in default audit metadata.
"""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
import json
import logging
import re
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import anyio
from httpx import ASGITransport, AsyncClient
import pytest

from app.auth import (
    AuthenticatedPrincipal,
    authenticate_api_key,
    clear_auth_cache,
    hash_api_key,
    require_valid_api_key,
)
from app.billing import (
    InFlightSpendTracker,
    require_billing_eligible_principal,
    verify_minimum_input_balance,
)
from app.config import Environment, Settings
from app.logging import (
    RedactingFilter,
    StructuredJsonFormatter,
    _ALLOWED_EVENTS,
    _ALLOWED_METADATA_FIELDS,
    configure_logging,
    get_current_request_id,
    log_event,
    reset_current_request_id,
    sanitize_database_url,
    sanitize_text,
    set_current_request_id,
)
from app.main import create_app
from app.rate_limit import (
    InMemoryRateLimiter,
    resolve_rate_limit_identity,
)
from app.telemetry import record_inference_telemetry
from app.vertex import BaseModelProvider, PredictionResult


# ── Helper Fixtures ───────────────────────────────────────────────


def _make_auth_row(
    raw_key: str,
    user_id: str = "usr_audit_test",
    api_key_id: str = "key_audit_test",
    wallet_balance_usd: Decimal = Decimal("50.000000"),
    user_is_banned: bool = False,
    expires_at: datetime | None = None,
) -> dict[str, Any]:
    return {
        "api_key_id": api_key_id,
        "user_id": user_id,
        "key_hash": hash_api_key(raw_key),
        "key_prefix": raw_key[:7],
        "key_name": "Test Key",
        "spend_limit_usd": Decimal("100.0000"),
        "expires_at": expires_at,
        "user_is_banned": user_is_banned,
        "user_email_verified": True,
        "wallet_balance_usd": wallet_balance_usd,
    }


def _make_mock_app(
    mock_auth_rows: list[dict[str, Any]] | None = None,
    model_provider: BaseModelProvider | None = None,
    rate_limit_rpm: int = 60,
    rate_limit_enabled: bool = False,
):
    mock_session = AsyncMock()

    async def _execute_side_effect(statement, params=None):
        stmt_str = str(statement)
        mock_res = MagicMock()
        if "mp_active_api_keys_v" in stmt_str or "SELECT" in stmt_str:
            mock_res.fetchall.return_value = mock_auth_rows if mock_auth_rows is not None else []
            return mock_res
        elif "mp_inference_logs" in stmt_str or "INSERT" in stmt_str:
            mock_res.fetchall.return_value = []
            return mock_res
        mock_res.fetchall.return_value = []
        return mock_res

    mock_session.execute = AsyncMock(side_effect=_execute_side_effect)

    settings = Settings(
        rantera_env="development",
        gcp_project_id="test-project",
        gcp_region="us-central1",
        vertex_endpoint_id="1234567890",
        database_url="postgresql+asyncpg://user:pass@localhost:5432/test_db",
        rate_limit_enabled=rate_limit_enabled,
        rate_limit_rpm=rate_limit_rpm,
        auth_cache_ttl_seconds=0,
    )

    class MockProvider(BaseModelProvider):
        async def predict(self, model, messages, temperature=0.7, max_tokens=1000):
            return PredictionResult(
                content="Mocked response",
                tokens_in=10,
                tokens_out=15,
                latency_ms=25,
                finish_reason="stop",
            )

    app = create_app(settings=settings, model_provider=model_provider or MockProvider())
    app.state.db_engine = MagicMock()

    class _MockContextManager:
        async def __aenter__(self):
            return mock_session

        async def __aexit__(self, *args):
            pass

    app.state.db_session_factory = lambda: _MockContextManager()
    return app, mock_session


# ── 1. Single-Line JSON Schema & Required Fields ──────────────────


def test_structured_json_formatter_produces_valid_single_line_json():
    """Formatter must produce a single-line JSON string containing all required fields."""
    formatter = StructuredJsonFormatter(environment="test")
    record = logging.LogRecord(
        name="app.test",
        level=logging.INFO,
        pathname="test.py",
        lineno=10,
        msg="Service initialized successfully",
        args=(),
        exc_info=None,
    )
    setattr(record, "event", "service.startup")
    setattr(record, "metadata", {"environment": "test", "pool_size": 5})

    token = set_current_request_id("req_test_12345")
    try:
        output = formatter.format(record)
    finally:
        reset_current_request_id(token)

    # Must be exactly one physical line
    assert "\n" not in output
    assert "\r" not in output

    parsed = json.loads(output)
    required_fields = {"timestamp", "level", "logger", "event", "message", "environment", "request_id"}
    for f in required_fields:
        assert f in parsed, f"Missing required field: {f}"

    assert parsed["level"] == "INFO"
    assert parsed["logger"] == "app.test"
    assert parsed["event"] == "service.startup"
    assert parsed["message"] == "Service initialized successfully"
    assert parsed["environment"] == "test"
    assert parsed["request_id"] == "req_test_12345"
    assert parsed["metadata"] == {"environment": "test", "pool_size": 5}

    # Timestamp must end in 'Z' and parse as ISO-8601
    assert parsed["timestamp"].endswith("Z")
    dt = datetime.fromisoformat(parsed["timestamp"].replace("Z", "+00:00"))
    assert dt is not None


# ── 2. Production API Key Redaction (sk-rnt- + 48 hex) ───────────


def test_production_api_key_full_redaction_no_suffix():
    """Real sk-rnt- keys (48 hex characters) must be completely replaced with [REDACTED_API_KEY] without suffix."""
    raw_key = "sk-rnt-0123456789abcdef0123456789abcdef0123456789abcdef"
    assert len(raw_key) == 7 + 48

    text_with_key = f"Client attempted authentication with key {raw_key} from internal network"
    sanitized = sanitize_text(text_with_key)

    assert "[REDACTED_API_KEY]" in sanitized
    assert raw_key not in sanitized
    assert "sk-rnt-" not in sanitized
    assert sanitized == "Client attempted authentication with key [REDACTED_API_KEY] from internal network"


def test_fixture_api_key_redaction():
    """Fixture keys matching sk-ran-... must also be redacted to [REDACTED_API_KEY]."""
    test_key = "sk-ran-super-secret-key-12345"
    text = f"Using fixture key: {test_key}"
    sanitized = sanitize_text(text)

    assert "[REDACTED_API_KEY]" in sanitized
    assert test_key not in sanitized
    assert sanitized == "Using fixture key: [REDACTED_API_KEY]"


# ── 3. Bearer Token Redaction ────────────────────────────────────


def test_bearer_token_redaction():
    """Bearer tokens in strings must be redacted to Bearer [REDACTED]."""
    text = "Authorization header: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.e30.secret"
    sanitized = sanitize_text(text)

    assert "Bearer [REDACTED]" in sanitized
    assert "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9" not in sanitized


# ── 4. Structured Database URL Credential Redaction ──────────────


def test_database_url_redaction_variants():
    """Database URLs must have passwords masked across all supported schemes and encoded passwords."""
    variants = [
        # Standard postgresql+asyncpg
        (
            "postgresql+asyncpg://admin:supersecret@db.internal:5432/rantera_prod",
            "postgresql+asyncpg://admin:***@db.internal:5432/rantera_prod",
        ),
        # postgresql scheme
        (
            "postgresql://app_user:mypassword@localhost:5432/testdb?sslmode=disable",
            "postgresql://app_user:***@localhost:5432/testdb?sslmode=disable",
        ),
        # postgres scheme
        (
            "postgres://rantera:secret99@10.0.0.1:5432/rantera",
            "postgres://rantera:***@10.0.0.1:5432/rantera",
        ),
        # URL with percent-encoded special characters in password
        (
            "postgresql+asyncpg://rantera_app:p%40ss%23word%21@db.internal:5432/rantera_db?ssl=require",
            "postgresql+asyncpg://rantera_app:***@db.internal:5432/rantera_db?ssl=require",
        ),
        # URL without password
        (
            "postgresql+asyncpg://readonly@db.internal:5432/rantera_db",
            "postgresql+asyncpg://readonly@db.internal:5432/rantera_db",
        ),
        # SQLite in-memory / file
        (
            "sqlite+aiosqlite:///:memory:",
            "sqlite+aiosqlite:///:memory:",
        ),
        (
            "sqlite+aiosqlite:///app.db",
            "sqlite+aiosqlite:///app.db",
        ),
    ]

    for raw, expected in variants:
        # Test direct structured function
        assert sanitize_database_url(raw) == expected
        # Test raw text sanitization inside a log message
        log_message = f"Connecting to database at {raw} ..."
        sanitized_msg = sanitize_text(log_message)
        assert expected in sanitized_msg
        assert "supersecret" not in sanitized_msg
        assert "p%40ss%23word%21" not in sanitized_msg


def test_database_url_query_parameters_and_fragment_sanitization():
    """Database URLs must redact sensitive query parameters and strip fragments while preserving safe parameters."""
    test_cases = [
        # PostgreSQL URL with credentials in authority, query parameters, and fragment
        (
            "postgresql+asyncpg://admin:pass@db.internal:5432/rantera_db?sslmode=require&password=secret_in_query&sslkey=/path/to/key&connect_timeout=10#frag_secret",
            "postgresql+asyncpg://admin:***@db.internal:5432/rantera_db?sslmode=require&password=***&sslkey=***&connect_timeout=10",
        ),
        # Case variants, tokens, credentials, and private keys
        (
            "postgresql://localhost:5432/db?API%5FKEY=my_key&Token=tok_val&PRIVATE_KEY=pk_val&application_name=Rantera%20App",
            "postgresql://localhost:5432/db?API_KEY=***&Token=***&PRIVATE_KEY=***&application_name=Rantera%20App",
        ),
        # SSL private key variants (sslkey, sslpassword, sslpass) vs harmless sslmode
        (
            "postgresql://app_user@db:5432/prod?sslmode=verify-full&sslpassword=passphrase&sslpass=pass2&sslrootcert=/etc/ssl/ca.crt",
            "postgresql://app_user@db:5432/prod?sslmode=verify-full&sslpassword=***&sslpass=***&sslrootcert=/etc/ssl/ca.crt",
        ),
        # SQLite with query parameters and fragment
        (
            "sqlite+aiosqlite:///app.db?mode=ro&secret=dbsecret#leaked_frag",
            "sqlite+aiosqlite:///app.db?mode=ro&secret=***",
        ),
        # SQLite memory with harmless query
        (
            "sqlite+aiosqlite:///:memory:?cache=shared",
            "sqlite+aiosqlite:///:memory:?cache=shared",
        ),
    ]

    for raw, expected in test_cases:
        sanitized = sanitize_database_url(raw)
        assert sanitized == expected
        # Verify no sensitive values or fragments leak
        assert "pass@" not in sanitized
        assert "secret_in_query" not in sanitized
        assert "/path/to/key" not in sanitized
        assert "frag_secret" not in sanitized
        assert "leaked_frag" not in sanitized
        assert "#" not in sanitized

        # Verify raw text sanitization inside a log message
        log_msg = f"Connecting to {raw} now."
        sanitized_log = sanitize_text(log_msg)
        assert expected in sanitized_log
        assert "secret_in_query" not in sanitized_log
        assert "frag_secret" not in sanitized_log


def test_database_url_parsing_failure_never_leaks_original():
    """When URL parsing fails, sanitize_database_url must fail closed and return [DATABASE_URL_REDACTED]."""
    # Malformed IPv6 URL with credentials in authority
    malformed_with_authority = "postgresql://user:pass@[invalid-ipv6/db"
    assert sanitize_database_url(malformed_with_authority) == "[DATABASE_URL_REDACTED]"

    # Malformed URL with credentials only in query parameters
    malformed_with_query = "postgresql://[invalid-ipv6/?secret=leaked"
    assert sanitize_database_url(malformed_with_query) == "[DATABASE_URL_REDACTED]"

    # Malformed URL with both authority credentials and sensitive query parameters
    malformed_with_both = "postgresql://user:pass@[invalid-ipv6/db?token=secret"
    sanitized_both = sanitize_database_url(malformed_with_both)
    assert sanitized_both == "[DATABASE_URL_REDACTED]"
    assert "secret" not in sanitized_both
    assert "pass@" not in sanitized_both

    # Non-URL garbage string
    assert sanitize_database_url("not_a_valid_url") == "[DATABASE_URL_REDACTED]"


# ── 5. Provider / Cloud Credential & Password Redaction ──────────


def test_generic_password_and_secret_redaction():
    """Generic key-value secret assignments must be masked with ***."""
    samples = [
        ("password=MySuperSecretPassword123", "password: ***"),
        ('client_secret: "topsecretclientsecret"', "client_secret: ***"),
        ("password: 'plain_password'", "password: ***"),
    ]

    for raw, expected_part in samples:
        sanitized = sanitize_text(raw)
        assert expected_part in sanitized
        assert "MySuperSecretPassword123" not in sanitized
        assert "topsecretclientsecret" not in sanitized


def test_multi_word_and_quoted_secrets_redaction():
    """Multi-word secrets with spaces and quotes must be completely redacted."""
    samples = [
        ('password="my secret pass"', "password: ***"),
        ("token='two words here'", "token: ***"),
        ('client_secret="a long client secret phrase"', "client_secret: ***"),
        ('PASSWORD="UPPERCASE SECRET"', "PASSWORD: ***"),
        ('pwd="short secret"', "pwd: ***"),
        ('api_key="secret-api-key-with-spaces in it"', "api_key: ***"),
        # Escaped quotes inside strings
        ('password="my \\"secret\\" pass"', "password: ***"),
        ("token='two \\'words\\' here'", "token: ***"),
    ]

    for raw, expected in samples:
        sanitized = sanitize_text(raw)
        assert sanitized == expected
        assert "my secret pass" not in sanitized
        assert "two words here" not in sanitized
        assert "a long client secret phrase" not in sanitized
        assert "UPPERCASE SECRET" not in sanitized
        assert "secret-api-key" not in sanitized


def test_secret_redaction_preserves_surrounding_fields():
    """Secret redaction must preserve surrounding non-secret fields and whitespace."""
    log_line = 'status=ok, password="my secret pass", retries=3, token=\'tok en\', next_field=active'
    sanitized = sanitize_text(log_line)

    assert "my secret pass" not in sanitized
    assert "tok en" not in sanitized
    assert "status=ok, " in sanitized
    assert ", retries=3, " in sanitized
    assert ", next_field=active" in sanitized
    assert sanitized == "status=ok, password: ***, retries=3, token: ***, next_field=active"


def test_secret_redaction_crlf_and_injection_attempts():
    """Secret redaction must safely handle CRLF, newlines, and forged log lines."""
    # Unquoted secret followed by CRLF
    unquoted_crlf = "password=secret123\r\n[2026-10-14T00:00:00Z] forged line"
    sanitized_unquoted = sanitize_text(unquoted_crlf)
    assert "secret123" not in sanitized_unquoted
    assert "password: ***\r\n[2026-10-14T00:00:00Z] forged line" == sanitized_unquoted

    # Quoted secret with escaped CRLF
    escaped_crlf = 'password="secret\\r\\ninjection" next_field=safe'
    sanitized_escaped = sanitize_text(escaped_crlf)
    assert "secret" not in sanitized_escaped
    assert sanitized_escaped == "password: *** next_field=safe"

    # Already-redacted values must not be double-redacted
    assert sanitize_text("password: ***") == "password: ***"
    assert sanitize_text("password: [REDACTED]") == "password: [REDACTED]"
    assert sanitize_text("password: [REDACTED_API_KEY]") == "password: [REDACTED_API_KEY]"


# ── 6. CRLF and Control Character Injection ───────────────────────


def test_crlf_and_control_character_injection_neutralized():
    """Injected newlines and carriage returns must be safely escaped by JSON formatting."""
    formatter = StructuredJsonFormatter(environment="test")
    record = logging.LogRecord(
        name="app.test",
        level=logging.INFO,
        pathname="test.py",
        lineno=10,
        msg="Injected line\r\n{\"fake\":\"json_record\",\"level\":\"CRITICAL\"}\nNormal line continued",
        args=(),
        exc_info=None,
    )

    output = formatter.format(record)
    # Output must strictly remain a single physical line
    assert "\n" not in output
    assert "\r" not in output

    parsed = json.loads(output)
    assert parsed["level"] == "INFO"
    assert "fake" not in parsed  # Not parsed as a separate object


# ── 7. Sanitized Exception Output & Tracebacks Without Locals ─────


def test_exception_sanitization_without_local_variables():
    """Exceptions must be sanitized and tracebacks must not include local variable values."""
    formatter = StructuredJsonFormatter(environment="test")
    sensitive_local_var = "sk-rnt-0123456789abcdef0123456789abcdef0123456789abcdef"

    try:
        raise ValueError("Connection failed to postgresql://user:secretpass@localhost:5432/db")
    except ValueError as exc:
        record = logging.LogRecord(
            name="app.test",
            level=logging.ERROR,
            pathname="test.py",
            lineno=10,
            msg="Exception caught",
            args=(),
            exc_info=(type(exc), exc, exc.__traceback__),
        )

    output = formatter.format(record)
    parsed = json.loads(output)

    assert "exception" in parsed
    exc_payload = parsed["exception"]
    assert exc_payload["type"] == "ValueError"
    # Exception message must be sanitized
    assert "secretpass" not in exc_payload["message"]
    assert "postgresql://user:***@localhost:5432/db" in exc_payload["message"]
    # Local variable from test frame must not be leaked
    assert sensitive_local_var not in output
    assert "sensitive_local_var" not in exc_payload["traceback"]


# ── 8. Concurrent Request-ID Isolation & Cleanup in finally ──────


@pytest.mark.anyio
async def test_concurrent_request_id_isolation():
    """Concurrent async tasks must have isolated request IDs that are cleaned up in finally."""
    task_results: list[tuple[str, str, str]] = []

    async def worker(task_id: str, client_req_id: str, should_fail: bool):
        token = set_current_request_id(client_req_id)
        before_sleep = get_current_request_id()
        try:
            await anyio.sleep(0.01)
            during_sleep = get_current_request_id()
            if should_fail:
                raise RuntimeError("Simulated failure")
            task_results.append((task_id, before_sleep, during_sleep))
        except RuntimeError:
            task_results.append((task_id, before_sleep, during_sleep))
        finally:
            reset_current_request_id(token)

    async with anyio.create_task_group() as tg:
        tg.start_soon(worker, "t1", "req_concurrent_aaa", False)
        tg.start_soon(worker, "t2", "req_concurrent_bbb", True)
        tg.start_soon(worker, "t3", "req_concurrent_ccc", False)

    # After completion, current request ID in caller must be reset to empty
    assert get_current_request_id() == ""

    # All 3 tasks must have seen only their own isolated request ID
    assert len(task_results) == 3
    for task_id, before, during in task_results:
        assert before == during
        if task_id == "t1":
            assert before == "req_concurrent_aaa"
        elif task_id == "t2":
            assert before == "req_concurrent_bbb"
        elif task_id == "t3":
            assert before == "req_concurrent_ccc"


# ── 9. Handler Configuration Idempotency ──────────────────────────


def test_configure_logging_is_idempotent():
    """configure_logging must be idempotent and preserve existing handlers without duplicating records."""
    app_logger = logging.getLogger("app")

    settings = Settings(rantera_env=Environment.DEVELOPMENT, log_level="DEBUG")
    configure_logging(settings)
    configured_handler_count = len(app_logger.handlers)

    # Calling configure_logging multiple times must not add duplicate handlers
    configure_logging(settings)
    configure_logging(settings)
    assert len(app_logger.handlers) == configured_handler_count


# ── 10. Safe Formatter Fallback ───────────────────────────────────


def test_safe_formatter_fallback_on_formatting_error(monkeypatch):
    """When _format_record encounters an unexpected exception, safe static JSON is emitted."""
    formatter = StructuredJsonFormatter(environment="test")
    record = logging.LogRecord(
        name="app.test",
        level=logging.ERROR,
        pathname="test.py",
        lineno=10,
        msg="Normal message",
        args=(),
        exc_info=None,
    )

    # Force _format_record to raise an error
    def broken_format(rec):
        raise TypeError("Simulated formatting crash")

    monkeypatch.setattr(formatter, "_format_record", broken_format)

    output = formatter.format(record)
    assert "\n" not in output
    parsed = json.loads(output)
    assert parsed["level"] == "ERROR"
    assert parsed["event"] == "system.error"
    assert parsed["message"] == "Log formatting failed safely"
    assert parsed["environment"] == "test"


# ── 11. Allowlisted Events & Metadata Filter ─────────────────────


def test_arbitrary_events_and_metadata_are_filtered():
    """Unapproved event names and arbitrary objects in metadata must be rejected or dropped."""
    formatter = StructuredJsonFormatter(environment="test")
    record = logging.LogRecord(
        name="app.test",
        level=logging.INFO,
        pathname="test.py",
        lineno=10,
        msg="Testing filtering",
        args=(),
        exc_info=None,
    )
    # Set an unapproved event name
    setattr(record, "event", "unapproved.arbitrary.event")
    # Set metadata with allowlisted, unallowlisted, and arbitrary complex objects
    setattr(
        record,
        "metadata",
        {
            "method": "POST",  # allowlisted
            "status_code": 200,  # allowlisted
            "arbitrary_dump": {"user_password": "secret", "prompt": "sensitive"},  # unapproved object
            "raw_headers": ["Authorization", "Bearer sk-rnt-..."],  # unapproved object
            "cost_usd": Decimal("0.00015"),  # allowlisted Decimal
        },
    )

    output = formatter.format(record)
    parsed = json.loads(output)

    # Event name must fall back to system.info
    assert parsed["event"] == "system.info"

    # Only allowlisted primitive metadata should remain
    metadata = parsed.get("metadata", {})
    assert metadata["method"] == "POST"
    assert metadata["status_code"] == 200
    assert metadata["cost_usd"] == "0.00015"
    assert "arbitrary_dump" not in metadata
    assert "raw_headers" not in metadata


# ── 12. Absence of Prompts and Model Responses in Logs ────────────


@pytest.mark.anyio
async def test_prompts_and_model_responses_never_logged(caplog):
    """Chat completions execution must never log prompt messages or model responses."""
    distinctive_prompt = "SUPER_SECRET_PROMPT_STRING_ABC123"
    distinctive_response = "SUPER_SECRET_RESPONSE_STRING_XYZ789"

    class MockProvider(BaseModelProvider):
        async def predict(self, model, messages, temperature=0.7, max_tokens=1000):
            return PredictionResult(
                content=distinctive_response,
                tokens_in=10,
                tokens_out=15,
                latency_ms=50,
                finish_reason="stop",
            )

    raw_key = "sk-ran-testloggingchat"
    row = _make_auth_row(raw_key)
    app, _ = _make_mock_app([row], model_provider=MockProvider())

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        with caplog.at_level(logging.DEBUG):
            response = await client.post(
                "/v1/chat/completions",
                headers={"X-API-Key": raw_key},
                json={
                    "model": "rantera-base-v1",
                    "messages": [{"role": "user", "content": distinctive_prompt}],
                },
            )

    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == distinctive_response

    # Verify that neither prompt nor response string exists in any captured log record
    for rec in caplog.records:
        formatted = rec.getMessage()
        assert distinctive_prompt not in formatted
        assert distinctive_response not in formatted
        if hasattr(rec, "metadata"):
            assert distinctive_prompt not in str(rec.metadata)
            assert distinctive_response not in str(rec.metadata)


# ── 13. Absence of PII by Default in Structured Events ────────────


@pytest.mark.anyio
async def test_pii_absence_in_default_structured_events(caplog):
    """Logs must not record user_id, api_key_id, or client IP by default."""
    raw_key = "sk-ran-testloggingpii"
    user_id = "sensitive_user_id_999"
    api_key_id = "sensitive_api_key_id_888"
    row = _make_auth_row(raw_key, user_id=user_id, api_key_id=api_key_id)
    app, _ = _make_mock_app([row])

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        with caplog.at_level(logging.INFO):
            resp = await client.get("/v1/models", headers={"X-API-Key": raw_key})

    assert resp.status_code == 200

    for rec in caplog.records:
        metadata = getattr(rec, "metadata", {})
        assert "user_id" not in metadata
        assert "api_key_id" not in metadata
        assert "client_ip" not in metadata
        assert user_id not in str(metadata)
        assert api_key_id not in str(metadata)


# ── 14. Structured Events for Auth, Rate Limit, Billing, Telemetry


@pytest.mark.anyio
async def test_structured_events_taxonomy_and_lifecycle(caplog):
    """Verify specific allowlisted events are emitted across key lifecycle paths."""
    # 1. Unauthenticated request -> auth.failed.missing_key
    app, _ = _make_mock_app([])
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        with caplog.at_level(logging.WARNING):
            resp = await client.get("/v1/models")
        assert resp.status_code == 401
        assert any(getattr(r, "event", None) == "auth.failed.missing_key" for r in caplog.records)

    caplog.clear()

    # 2. Rate limit rejection -> rate_limit.rejected
    raw_key = "sk-ran-testlimit1"
    row = _make_auth_row(raw_key)
    app_rl, _ = _make_mock_app([row], rate_limit_rpm=1, rate_limit_enabled=True)
    async with AsyncClient(transport=ASGITransport(app=app_rl), base_url="http://test") as client:
        r1 = await client.get("/v1/models", headers={"X-API-Key": raw_key})
        assert r1.status_code == 200
        with caplog.at_level(logging.WARNING):
            r2 = await client.get("/v1/models", headers={"X-API-Key": raw_key})
        assert r2.status_code == 429
        assert any(getattr(r, "event", None) == "rate_limit.rejected" for r in caplog.records)

    caplog.clear()

    # 3. Billing rejection: depleted wallet -> billing.rejected.depleted_wallet
    raw_key_zero = "sk-ran-testzero"
    row_zero = _make_auth_row(raw_key_zero, wallet_balance_usd=Decimal("0.0"))
    app_zero, _ = _make_mock_app([row_zero])
    async with AsyncClient(transport=ASGITransport(app=app_zero), base_url="http://test") as client:
        with caplog.at_level(logging.WARNING):
            resp_zero = await client.post(
                "/v1/chat/completions",
                headers={"X-API-Key": raw_key_zero},
                json={"model": "rantera-base-v1", "messages": [{"role": "user", "content": "hi"}]},
            )
        assert resp_zero.status_code == 402
        assert any(getattr(r, "event", None) == "billing.rejected.depleted_wallet" for r in caplog.records)

    caplog.clear()

    # 4. Telemetry write without session -> telemetry.session_none
    principal = AuthenticatedPrincipal(
        user_id="user_telem",
        api_key_id="key_telem",
        channel="api",
        wallet_balance=Decimal("10.0"),
    )
    with caplog.at_level(logging.ERROR):
        res = await record_inference_telemetry(
            session=None,
            request_id="req_telem_test",
            principal=principal,
            model_id="rantera-base-v1",
            tokens_in=5,
            tokens_out=5,
            cost_usd=Decimal("0.0001"),
            latency_ms=20,
        )
    assert res is False
    assert any(getattr(r, "event", None) == "telemetry.session_none" for r in caplog.records)
