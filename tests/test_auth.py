"""Tests for API-key authentication and caller identity resolution (Part 3).

Verifies:
- SHA-256 hash computation and constant-time verification.
- Rejection of missing, empty, and whitespace-only API keys with 401.
- Rejection of unregistered / invalid API keys with 401.
- Rejection of hash mismatch in database row with 401.
- Rejection of expired API keys with 401 ("API key expired.").
- Acceptance of unexpired API keys and keys with null expiry.
- Rejection of banned user accounts with 403 ("Account suspended.").
- Fail-closed behavior on unconfigured database (503).
- Fail-closed behavior on database connection/query exceptions (503).
- Parameterized SQL execution targeting mp_active_api_keys_v only.
- Strict isolation from core website tables (User, ApiKey, Wallet).
- Construction of typed AuthenticatedPrincipal with billing metrics.
- Complete disregard of client-supplied user IDs.
- Secret hygiene: plaintext API keys and credentials are never logged or leaked.
- Public health endpoints remain fully accessible without credentials.
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from httpx import ASGITransport, AsyncClient

from app.auth import (
    AuthenticatedPrincipal,
    authenticate_api_key,
    clear_auth_cache,
    hash_api_key,
    verify_key_hash,
)
from app.config import Settings
from app.main import create_app
from app.models import _REGISTRY


# ── Fixtures & Helpers ─────────────────────────────────────────


@pytest.fixture(autouse=True)
def _clean_cache():
    """Clear the in-memory authentication cache before each test."""
    clear_auth_cache()
    yield
    clear_auth_cache()


def _dev_app_unconfigured():
    """App with no database configured."""
    settings = Settings(
        rantera_env="development",
        gcp_project_id=None,
        gcp_region=None,
        vertex_endpoint_id=None,
        database_url=None,
    )
    return create_app(settings=settings)


def _make_mock_auth_app(
    mock_rows: list[dict[str, Any]] | None = None,
    execute_side_effect: Exception | None = None,
    auth_cache_ttl_seconds: int = 0,
) -> tuple[Any, AsyncMock]:
    """Create a FastAPI test app with a mocked database session."""
    mock_session = AsyncMock()
    mock_result = MagicMock()
    if execute_side_effect is not None:
        mock_session.execute.side_effect = execute_side_effect
    else:
        mock_result.fetchall.return_value = mock_rows if mock_rows is not None else []
        mock_session.execute.return_value = mock_result

    settings = Settings(
        rantera_env="development",
        gcp_project_id="test-project",
        gcp_region="us-central1",
        vertex_endpoint_id="1234567890",
        database_url="postgresql+asyncpg://user:pass@localhost:5432/test_db",
        auth_cache_ttl_seconds=auth_cache_ttl_seconds,
    )
    app = create_app(settings=settings)
    app.state.db_engine = MagicMock()
    # Pre-warm catalog with static registry so auth route calls to /v1/models do not invoke the auth mock session
    if hasattr(app.state, "model_catalog_service"):
        app.state.model_catalog_service._cache = dict(_REGISTRY)
        app.state.model_catalog_service._has_ever_loaded = True
        app.state.model_catalog_service._last_refresh_mono = time.monotonic()

    class _MockContextManager:
        async def __aenter__(self):
            return mock_session

        async def __aexit__(self, *args):
            pass

    app.state.db_session_factory = lambda: _MockContextManager()
    return app, mock_session


def _make_valid_row(
    raw_key: str,
    user_id: str = "usr_test_123",
    api_key_id: str = "key_test_abc",
    user_is_banned: bool = False,
    expires_at: datetime | None = None,
    wallet_balance_usd: Decimal = Decimal("50.000000"),
    spend_limit_usd: Decimal | None = Decimal("100.0000"),
) -> dict[str, Any]:
    """Construct a mock row matching the mp_active_api_keys_v projection."""
    return {
        "api_key_id": api_key_id,
        "user_id": user_id,
        "key_hash": hash_api_key(raw_key),
        "key_prefix": raw_key[:7] if len(raw_key) >= 7 else "sk-ran-",
        "key_name": "Default Test Key",
        "spend_limit_usd": spend_limit_usd,
        "expires_at": expires_at,
        "user_is_banned": user_is_banned,
        "user_email_verified": True,
        "wallet_balance_usd": wallet_balance_usd,
    }


# ── 1. Hash Utilities ──────────────────────────────────────────


def test_hash_api_key_deterministic():
    """Same input produces the identical 64-char hex SHA-256 hash."""
    h1 = hash_api_key("test-key-123")
    h2 = hash_api_key("test-key-123")
    assert h1 == h2
    assert len(h1) == 64


def test_hash_api_key_different_inputs():
    """Different inputs produce distinct hashes."""
    h1 = hash_api_key("key-a")
    h2 = hash_api_key("key-b")
    assert h1 != h2


def test_verify_key_hash_correct():
    """verify_key_hash returns True for a matching key in constant time."""
    raw = "my-secret-key"
    stored = hash_api_key(raw)
    assert verify_key_hash(raw, stored) is True


def test_verify_key_hash_incorrect():
    """verify_key_hash returns False for a non-matching key."""
    stored = hash_api_key("correct-key")
    assert verify_key_hash("wrong-key", stored) is False


# ── 2. Missing, Empty & Malformed API Keys (401) ───────────────


@pytest.mark.anyio
async def test_models_endpoint_no_key_returns_401():
    """GET /v1/models without X-API-Key header returns 401 Unauthorized."""
    app = _dev_app_unconfigured()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/v1/models")
    assert resp.status_code == 401
    body = resp.json()
    assert body["error"] == "Unauthorized"
    assert "Missing API key" in body["message"]
    assert "request_id" in body


@pytest.mark.anyio
async def test_models_endpoint_empty_key_returns_401():
    """An empty X-API-Key header is rejected with 401 Unauthorized."""
    app = _dev_app_unconfigured()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get(
            "/v1/models",
            headers={"X-API-Key": ""},
        )
    assert resp.status_code == 401
    assert "Missing API key" in resp.json()["message"]


@pytest.mark.anyio
async def test_models_endpoint_whitespace_key_returns_401():
    """A whitespace-only X-API-Key header is rejected with 401 Unauthorized."""
    app = _dev_app_unconfigured()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get(
            "/v1/models",
            headers={"X-API-Key": "    "},
        )
    assert resp.status_code == 401
    assert "Missing API key" in resp.json()["message"]


# ── 3. Valid API Key Authentication & Principal Resolution ────


@pytest.mark.anyio
async def test_valid_api_key_authenticates_successfully():
    """Valid API key returns 200 with registered model list."""
    test_key = "sk-ran-valid-live-key-456"
    mock_row = _make_valid_row(
        raw_key=test_key,
        user_id="usr_987654",
        api_key_id="key_123456",
        wallet_balance_usd=Decimal("75.500000"),
        spend_limit_usd=Decimal("500.0000"),
    )
    app, mock_session = _make_mock_auth_app(mock_rows=[mock_row])

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get(
            "/v1/models",
            headers={"X-API-Key": test_key},
        )

    assert resp.status_code == 200
    body = resp.json()
    assert "models" in body
    assert any(m["id"] == "rantera-base-v1" for m in body["models"])
    mock_session.execute.assert_awaited_once()


@pytest.mark.anyio
async def test_authenticated_principal_data_types():
    """authenticate_api_key returns a strictly typed AuthenticatedPrincipal."""
    test_key = "sk-ran-principal-type-test"
    mock_row = _make_valid_row(
        raw_key=test_key,
        user_id="usr_principal_1",
        api_key_id="key_principal_1",
        wallet_balance_usd=Decimal("123.456789"),
        spend_limit_usd=Decimal("1000.0000"),
    )
    mock_request = MagicMock()
    mock_request.url.path = "/v1/models"
    mock_request.app.state.settings.auth_cache_ttl_seconds = 0
    mock_request.state = MagicMock()

    mock_session = AsyncMock()
    mock_result = MagicMock()
    mock_result.fetchall.return_value = [mock_row]
    mock_session.execute.return_value = mock_result

    principal = await authenticate_api_key(
        raw_key=test_key,
        request=mock_request,
        session=mock_session,
    )

    assert isinstance(principal, AuthenticatedPrincipal)
    assert principal.user_id == "usr_principal_1"
    assert principal.api_key_id == "key_principal_1"
    assert principal.channel == "api"
    assert isinstance(principal.wallet_balance, Decimal)
    assert principal.wallet_balance == Decimal("123.456789")
    assert isinstance(principal.spend_limit, Decimal)
    assert principal.spend_limit == Decimal("1000.0000")
    assert principal.user_email_verified is True
    assert mock_request.state.principal is principal


# ── 4. Invalid API Key & Hash Mismatch (401) ──────────────────


@pytest.mark.anyio
async def test_unregistered_api_key_returns_401():
    """API key not present in mp_active_api_keys_v returns 401 Unauthorized."""
    app, mock_session = _make_mock_auth_app(mock_rows=[])

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get(
            "/v1/models",
            headers={"X-API-Key": "sk-ran-nonexistent-key"},
        )

    assert resp.status_code == 401
    body = resp.json()
    assert body["error"] == "Unauthorized"
    assert body["message"] == "Invalid API key."


@pytest.mark.anyio
async def test_hash_mismatch_in_row_returns_401():
    """If row key_hash does not match computed hash, reject with 401."""
    test_key = "sk-ran-attacker-key"
    mock_row = _make_valid_row(raw_key="different-key")
    # Force row to be returned despite hash mismatch
    app, _ = _make_mock_auth_app(mock_rows=[mock_row])

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get(
            "/v1/models",
            headers={"X-API-Key": test_key},
        )

    assert resp.status_code == 401
    assert resp.json()["message"] == "Invalid API key."


# ── 5. Expiration Handling (401 vs 200) ───────────────────────


@pytest.mark.anyio
async def test_expired_api_key_returns_401():
    """API key with past expires_at returns 401 Unauthorized ('API key expired.')."""
    test_key = "sk-ran-expired-key"
    past_date = datetime.now(timezone.utc) - timedelta(days=2)
    mock_row = _make_valid_row(raw_key=test_key, expires_at=past_date)
    app, _ = _make_mock_auth_app(mock_rows=[mock_row])

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get(
            "/v1/models",
            headers={"X-API-Key": test_key},
        )

    assert resp.status_code == 401
    body = resp.json()
    assert body["error"] == "Unauthorized"
    assert body["message"] == "API key expired."


@pytest.mark.anyio
async def test_unexpired_api_key_succeeds():
    """API key with future expires_at authenticates successfully."""
    test_key = "sk-ran-future-key"
    future_date = datetime.now(timezone.utc) + timedelta(days=30)
    mock_row = _make_valid_row(raw_key=test_key, expires_at=future_date)
    app, _ = _make_mock_auth_app(mock_rows=[mock_row])

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get(
            "/v1/models",
            headers={"X-API-Key": test_key},
        )

    assert resp.status_code == 200


@pytest.mark.anyio
async def test_api_key_without_expiration_succeeds():
    """API key with expires_at=None never expires and authenticates successfully."""
    test_key = "sk-ran-infinite-key"
    mock_row = _make_valid_row(raw_key=test_key, expires_at=None)
    app, _ = _make_mock_auth_app(mock_rows=[mock_row])

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get(
            "/v1/models",
            headers={"X-API-Key": test_key},
        )

    assert resp.status_code == 200


# ── 6. Banned User Accounts (403 Forbidden) ───────────────────


@pytest.mark.anyio
async def test_banned_user_returns_403_forbidden():
    """User account with user_is_banned=True returns 403 Forbidden ('Account suspended.')."""
    test_key = "sk-ran-banned-user-key"
    mock_row = _make_valid_row(raw_key=test_key, user_is_banned=True)
    app, _ = _make_mock_auth_app(mock_rows=[mock_row])

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get(
            "/v1/models",
            headers={"X-API-Key": test_key},
        )

    assert resp.status_code == 403
    body = resp.json()
    assert body["error"] == "Forbidden"
    assert body["message"] == "Account suspended."


# ── 7. Fail-Closed on Database & Infrastructure Errors (503) ──


@pytest.mark.anyio
async def test_unconfigured_database_returns_503():
    """When database is unconfigured (db_engine is None), auth fails closed with 503."""
    app = _dev_app_unconfigured()

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get(
            "/v1/models",
            headers={"X-API-Key": "sk-ran-valid-format-key"},
        )

    assert resp.status_code == 503
    body = resp.json()
    assert body["error"] == "Service Unavailable"
    assert "unavailable" in body["message"].lower()


@pytest.mark.anyio
async def test_database_query_exception_fails_closed_with_503():
    """Database exception during auth fails closed with 503 without leaking details."""
    app, _ = _make_mock_auth_app(
        execute_side_effect=ConnectionResetError("PostgreSQL connection severed")
    )

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get(
            "/v1/models",
            headers={"X-API-Key": "sk-ran-some-key"},
        )

    assert resp.status_code == 503
    body = resp.json()
    assert body["error"] == "Service Unavailable"
    assert "temporarily unavailable" in body["message"].lower()
    # Must NOT leak exception details
    assert "PostgreSQL connection severed" not in resp.text
    assert "ConnectionResetError" not in resp.text


@pytest.mark.anyio
async def test_database_timeout_fails_closed_with_503():
    """Database query timeout fails closed with 503."""
    app, _ = _make_mock_auth_app(
        execute_side_effect=TimeoutError("Statement timed out after 3.0s")
    )

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get(
            "/v1/models",
            headers={"X-API-Key": "sk-ran-timeout-key"},
        )

    assert resp.status_code == 503
    body = resp.json()
    assert body["error"] == "Service Unavailable"
    assert "timed out" not in resp.text


# ── 8. Query Parameterization & SQL Injection Defense ─────────


@pytest.mark.anyio
async def test_sql_injection_in_api_key_is_hashed_and_bound():
    """SQL injection payloads in X-API-Key are hashed and bound as literal parameters."""
    injection_key = "' OR '1'='1'; DROP TABLE mp_model_catalog; --"
    expected_hash = hash_api_key(injection_key)

    mock_row = _make_valid_row(raw_key=injection_key)
    app, mock_session = _make_mock_auth_app(mock_rows=[mock_row])

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get(
            "/v1/models",
            headers={"X-API-Key": injection_key},
        )

    assert resp.status_code == 200
    mock_session.execute.assert_awaited_once()
    call_args = mock_session.execute.call_args
    # First arg is TextClause
    statement_str = str(call_args[0][0])
    assert "mp_active_api_keys_v" in statement_str
    assert "DROP TABLE" not in statement_str
    assert ":key_hash" in statement_str
    # Second arg is bound parameter dict
    assert call_args[0][1]["key_hash"] == expected_hash


@pytest.mark.anyio
def test_auth_query_targets_view_only():
    """The authentication query must target mp_active_api_keys_v and no physical website tables."""
    from app.auth import _AUTH_QUERY
    from app.db import validate_query_safety

    # Must pass query safety validation
    validate_query_safety(_AUTH_QUERY)

    # Must not contain references to forbidden physical tables
    for forbidden in ['"User"', '"ApiKey"', '"Wallet"', '"Transaction"', '"UsageRecord"']:
        assert forbidden not in _AUTH_QUERY


def test_auth_query_column_names_match_approved_contract():
    """Verify that _AUTH_QUERY explicitly selects all 10 columns defined in the approved mp_active_api_keys_v contract."""
    import re
    from app.auth import _AUTH_QUERY

    # Expected column projections according to mp_active_api_keys_v
    expected_columns = [
        "api_key_id",
        "user_id",
        "key_hash",
        "key_prefix",
        "key_name",
        "spend_limit_usd",
        "expires_at",
        "user_is_banned",
        "user_email_verified",
        "wallet_balance_usd",
    ]

    for col in expected_columns:
        assert re.search(rf"\b{col}\b", _AUTH_QUERY), f"Expected column '{col}' missing from _AUTH_QUERY"

    # Query must target the security barrier view
    assert re.search(r"\bFROM\s+mp_active_api_keys_v\b", _AUTH_QUERY, re.IGNORECASE)
    # Query must use parameterized placeholder :key_hash
    assert re.search(r"\bWHERE\s+key_hash\s*=\s*:key_hash\b", _AUTH_QUERY, re.IGNORECASE)
    assert re.search(r"\bLIMIT\s+1\b", _AUTH_QUERY, re.IGNORECASE)


def test_authenticated_principal_aliases_and_properties():
    """Verify that AuthenticatedPrincipal provides alias properties for both naming conventions."""
    principal = AuthenticatedPrincipal(
        user_id="usr_123",
        api_key_id="key_abc",
        key_name="Work Key",
        wallet_balance=Decimal("42.50"),
        spend_limit=Decimal("100.00"),
        user_email_verified=True,
    )
    # Primary fields
    assert principal.key_name == "Work Key"
    assert principal.wallet_balance == Decimal("42.50")
    assert principal.spend_limit == Decimal("100.00")
    assert principal.user_email_verified is True

    # Alias properties
    assert principal.name == "Work Key"
    assert principal.wallet_balance_usd == Decimal("42.50")
    assert principal.spend_limit_usd == Decimal("100.00")
    assert principal.is_banned is False
    assert principal.email_verified is True


@pytest.mark.anyio
async def test_authenticate_api_key_fallback_column_names():
    """Verify authenticate_api_key can extract fields when rows use alternative field names (e.g. name, is_banned)."""
    raw_key = "sk-ran-test-fallback-fields"
    kh = hash_api_key(raw_key)

    alt_row = {
        "api_key_id": "key_fallback_1",
        "user_id": "usr_fallback_1",
        "key_hash": kh,
        "key_prefix": "sk-ran-",
        "name": "Alternative Name",
        "is_banned": False,
        "email_verified": True,
        "wallet_balance": Decimal("75.00"),
        "spend_limit": Decimal("150.00"),
        "expires_at": None,
    }

    mock_request = MagicMock()
    mock_request.url.path = "/v1/models"
    mock_request.app.state.settings.auth_cache_ttl_seconds = 0
    mock_request.state = MagicMock()

    mock_session = AsyncMock()
    mock_result = MagicMock()
    mock_result.fetchall.return_value = [alt_row]
    mock_session.execute.return_value = mock_result

    principal = await authenticate_api_key(raw_key, request=mock_request, session=mock_session)
    assert principal.user_id == "usr_fallback_1"
    assert principal.key_name == "Alternative Name"
    assert principal.name == "Alternative Name"
    assert principal.wallet_balance == Decimal("75.00")
    assert principal.wallet_balance_usd == Decimal("75.00")
    assert principal.spend_limit == Decimal("150.00")
    assert principal.spend_limit_usd == Decimal("150.00")
    assert principal.user_email_verified is True


# ── 9. Client-Supplied User ID Ignored ────────────────────────


@pytest.mark.anyio
async def test_client_supplied_user_id_is_strictly_ignored():
    """Headers attempting to spoof user identity are ignored; identity comes from DB."""
    test_key = "sk-ran-impersonation-test"
    db_user_id = "usr_real_database_owner_789"
    mock_row = _make_valid_row(raw_key=test_key, user_id=db_user_id)
    app, _ = _make_mock_auth_app(mock_rows=[mock_row])

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get(
            "/v1/models",
            headers={
                "X-API-Key": test_key,
                "X-User-ID": "usr_attacker_spoofed_123",
                "X-Client-User-ID": "admin",
            },
        )

    assert resp.status_code == 200
    # Authenticated principal on request state must reflect database identity
    # (Tested directly via authenticate_api_key below)
    mock_request = MagicMock()
    mock_request.url.path = "/v1/models"
    mock_request.app.state.settings.auth_cache_ttl_seconds = 0
    mock_request.headers = {"X-User-ID": "usr_attacker_spoofed_123"}
    mock_request.state = MagicMock()

    mock_session = AsyncMock()
    mock_result = MagicMock()
    mock_result.fetchall.return_value = [mock_row]
    mock_session.execute.return_value = mock_result

    principal = await authenticate_api_key(test_key, request=mock_request, session=mock_session)
    assert principal.user_id == db_user_id
    assert principal.user_id != "usr_attacker_spoofed_123"


# ── 10. Secret Redaction in Logs & Error Traces ────────────────


@pytest.mark.anyio
async def test_auth_logs_redact_plaintext_api_key(caplog):
    """Plaintext API keys must never appear in log records."""
    caplog.set_level(logging.DEBUG)
    test_key = "sk-ran-super-secret-production-key-999"
    app, _ = _make_mock_auth_app(mock_rows=[])

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        await client.get(
            "/v1/models",
            headers={"X-API-Key": test_key},
        )

    # Check that plaintext key is nowhere in captured logs
    for record in caplog.records:
        assert test_key not in record.getMessage()


@pytest.mark.anyio
async def test_auth_error_response_does_not_leak_key():
    """401/403/503 responses must never echo back the submitted API key."""
    test_key = "sk-ran-canary-key-to-detect-leak"
    app = _dev_app_unconfigured()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get(
            "/v1/models",
            headers={"X-API-Key": test_key},
        )
    assert test_key not in resp.text


# ── 11. In-Memory TTL Caching ──────────────────────────────────


@pytest.mark.anyio
async def test_in_memory_auth_cache_hits_and_invalidation():
    """With cache enabled, consecutive requests hit cache; clearing cache re-queries DB."""
    test_key = "sk-ran-cached-key"
    mock_row = _make_valid_row(raw_key=test_key)
    app, mock_session = _make_mock_auth_app(
        mock_rows=[mock_row],
        auth_cache_ttl_seconds=60,
    )

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # First request: cache miss -> queries DB
        r1 = await client.get("/v1/models", headers={"X-API-Key": test_key})
        assert r1.status_code == 200
        assert mock_session.execute.await_count == 1

        # Second request: cache hit -> does NOT query DB
        r2 = await client.get("/v1/models", headers={"X-API-Key": test_key})
        assert r2.status_code == 200
        assert mock_session.execute.await_count == 1

        # Clear cache: next request must query DB again
        clear_auth_cache()
        r3 = await client.get("/v1/models", headers={"X-API-Key": test_key})
        assert r3.status_code == 200
        assert mock_session.execute.await_count == 2


@pytest.mark.anyio
async def test_auth_cache_clamps_to_imminent_key_expiry():
    """Cache TTL is clamped to key expiration so cached credentials cannot bypass key expiry."""
    import time
    from app.auth import _AUTH_CACHE

    test_key = "sk-ran-expiring-soon"
    expiry_time = datetime.now(timezone.utc) + timedelta(seconds=1)
    mock_row = _make_valid_row(raw_key=test_key, expires_at=expiry_time)

    app, mock_session = _make_mock_auth_app(
        mock_rows=[mock_row],
        auth_cache_ttl_seconds=60,
    )

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # First request: loads into cache
        r1 = await client.get("/v1/models", headers={"X-API-Key": test_key})
        assert r1.status_code == 200

        # Verify entry in cache was clamped to <= 2 seconds from now
        kh = hash_api_key(test_key)
        assert kh in _AUTH_CACHE
        _, expire_mono = _AUTH_CACHE[kh]
        assert expire_mono <= time.monotonic() + 3.0


def test_auth_cache_bounded_size_and_eviction():
    """Verify that _AUTH_CACHE does not exceed _AUTH_CACHE_MAX_SIZE (1000 entries)."""
    from app.auth import _AUTH_CACHE, _AUTH_CACHE_MAX_SIZE, _set_cached_principal

    clear_auth_cache()
    sample_principal = AuthenticatedPrincipal(user_id="usr_bulk")

    # Insert 1050 entries with a long TTL
    for i in range(1050):
        _set_cached_principal(f"hash_{i}", sample_principal, ttl_seconds=300.0)

    # Size must not exceed maximum capacity
    assert len(_AUTH_CACHE) <= _AUTH_CACHE_MAX_SIZE


# ── 12. Public Endpoints & Method Verification ─────────────────


@pytest.mark.anyio
async def test_health_live_no_auth_required():
    """GET /health/live requires no authentication."""
    app = _dev_app_unconfigured()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/health/live")
    assert resp.status_code == 200
    assert resp.json()["status"] == "alive"


@pytest.mark.anyio
async def test_health_ready_no_auth_required():
    """GET /health/ready requires no authentication."""
    app = _dev_app_unconfigured()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/health/ready")
    assert resp.status_code == 503
    assert resp.status_code != 401


@pytest.mark.anyio
async def test_models_endpoint_post_returns_405():
    """POST /v1/models returns 405 Method Not Allowed."""
    app = _dev_app_unconfigured()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/v1/models")
    assert resp.status_code == 405


# ── 13. Dual-Header Authentication (Bearer & X-API-Key) & Conflicts ────


@pytest.mark.anyio
async def test_models_endpoint_bearer_header_success():
    """Valid API key provided via Authorization: Bearer <key> succeeds."""
    test_key = "sk-ran-bearer-valid-001"
    mock_row = _make_valid_row(raw_key=test_key, user_id="usr_bearer_1")
    app, mock_session = _make_mock_auth_app(mock_rows=[mock_row])

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get(
            "/v1/models",
            headers={"Authorization": f"Bearer {test_key}"},
        )

    assert resp.status_code == 200
    assert "models" in resp.json()
    mock_session.execute.assert_awaited_once()


@pytest.mark.anyio
async def test_models_endpoint_bearer_header_case_insensitive():
    """Authorization header scheme 'bearer' (lowercase) is accepted."""
    test_key = "sk-ran-bearer-case-002"
    mock_row = _make_valid_row(raw_key=test_key, user_id="usr_bearer_2")
    app, _ = _make_mock_auth_app(mock_rows=[mock_row])

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get(
            "/v1/models",
            headers={"Authorization": f"bearer {test_key}"},
        )

    assert resp.status_code == 200


@pytest.mark.anyio
async def test_models_endpoint_both_headers_identical_success():
    """When both X-API-Key and Authorization: Bearer provide identical keys, authentication succeeds."""
    test_key = "sk-ran-dual-identical-003"
    mock_row = _make_valid_row(raw_key=test_key, user_id="usr_dual_3")
    app, _ = _make_mock_auth_app(mock_rows=[mock_row])

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get(
            "/v1/models",
            headers={
                "X-API-Key": test_key,
                "Authorization": f"Bearer {test_key}",
            },
        )

    assert resp.status_code == 200


@pytest.mark.anyio
async def test_models_endpoint_conflicting_headers_rejected():
    """When X-API-Key and Authorization: Bearer provide different keys, request is rejected safely with 400."""
    key_a = "sk-ran-key-alpha"
    key_b = "sk-ran-key-beta"
    mock_row = _make_valid_row(raw_key=key_a)
    app, mock_session = _make_mock_auth_app(mock_rows=[mock_row])

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get(
            "/v1/models",
            headers={
                "X-API-Key": key_a,
                "Authorization": f"Bearer {key_b}",
            },
        )

    assert resp.status_code == 400
    body = resp.json()
    assert body["error"] == "Bad Request"
    assert "Conflicting authentication credentials" in body["message"]
    # Database query should never execute for conflicting headers
    mock_session.execute.assert_not_awaited()


@pytest.mark.anyio
async def test_models_endpoint_conflicting_schemes_rejected():
    """When X-API-Key is provided alongside a non-bearer Authorization header, request is rejected with 400."""
    key_a = "sk-ran-key-alpha"
    app, mock_session = _make_mock_auth_app(mock_rows=[])

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get(
            "/v1/models",
            headers={
                "X-API-Key": key_a,
                "Authorization": "Basic dXNlcjpwYXNzd29yZA==",
            },
        )

    assert resp.status_code == 400
    assert "Conflicting authentication credentials" in resp.json()["message"]
    mock_session.execute.assert_not_awaited()


@pytest.mark.anyio
async def test_models_endpoint_bearer_empty_token_returns_401():
    """Authorization header with Bearer but no token is rejected with 401."""
    app = _dev_app_unconfigured()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get(
            "/v1/models",
            headers={"Authorization": "Bearer"},
        )
    assert resp.status_code == 401
    assert "Missing API key" in resp.json()["message"]

    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp_ws = await client.get(
            "/v1/models",
            headers={"Authorization": "Bearer    "},
        )
    assert resp_ws.status_code == 401
    assert "Missing API key" in resp_ws.json()["message"]


@pytest.mark.anyio
async def test_models_endpoint_unsupported_auth_scheme_returns_401():
    """Non-bearer Authorization header without X-API-Key is rejected with 401."""
    app = _dev_app_unconfigured()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get(
            "/v1/models",
            headers={"Authorization": "Basic dXNlcjpwYXNzd29yZA=="},
        )
    assert resp.status_code == 401
    assert "Invalid authorization scheme" in resp.json()["message"]


@pytest.mark.anyio
async def test_models_endpoint_bearer_invalid_key_returns_401():
    """Unregistered key provided via Bearer token is rejected with 401."""
    app, _ = _make_mock_auth_app(mock_rows=[])
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get(
            "/v1/models",
            headers={"Authorization": "Bearer sk-ran-unregistered-bearer-key"},
        )
    assert resp.status_code == 401
    assert "Invalid API key" in resp.json()["message"]


@pytest.mark.anyio
async def test_models_endpoint_bearer_expired_key_returns_401():
    """Expired key provided via Bearer token is rejected with 401."""
    test_key = "sk-ran-expired-bearer-key"
    past = datetime.now(timezone.utc) - timedelta(hours=1)
    mock_row = _make_valid_row(raw_key=test_key, expires_at=past)
    app, _ = _make_mock_auth_app(mock_rows=[mock_row])

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get(
            "/v1/models",
            headers={"Authorization": f"Bearer {test_key}"},
        )
    assert resp.status_code == 401
    assert resp.json()["message"] == "API key expired."


@pytest.mark.anyio
async def test_models_endpoint_bearer_banned_user_returns_403():
    """Banned user key provided via Bearer token is rejected with 403 Forbidden."""
    test_key = "sk-ran-banned-bearer-key"
    mock_row = _make_valid_row(raw_key=test_key, user_is_banned=True)
    app, _ = _make_mock_auth_app(mock_rows=[mock_row])

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get(
            "/v1/models",
            headers={"Authorization": f"Bearer {test_key}"},
        )
    assert resp.status_code == 403
    assert resp.json()["message"] == "Account suspended."
