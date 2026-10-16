"""Focused integration and contract tests for POST /v1/chat/completions (Part 4)."""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from httpx import ASGITransport, AsyncClient

from app.auth import clear_auth_cache, hash_api_key
from app.config import Settings
from app.main import create_app
from app.models import RegisteredModel, _REGISTRY
from app.schemas import ChatMessage
from app.vertex import BaseModelProvider, MockModelProvider, PredictionResult


@pytest.fixture(autouse=True)
def _clean_cache():
    clear_auth_cache()
    yield
    clear_auth_cache()


def _make_mock_app(
    mock_auth_rows: list[dict[str, Any]] | None = None,
    telemetry_side_effect: Exception | None = None,
    model_provider: BaseModelProvider | None = None,
) -> tuple[Any, AsyncMock]:
    """Create a test application instance with mocked database session and auth row."""
    mock_session = AsyncMock()

    def _execute_side_effect(statement, params=None):
        stmt_str = str(statement)
        mock_res = MagicMock()
        if "mp_active_api_keys_v" in stmt_str or "SELECT" in stmt_str:
            mock_res.fetchall.return_value = mock_auth_rows if mock_auth_rows is not None else []
            return mock_res
        elif "mp_inference_logs" in stmt_str or "INSERT" in stmt_str:
            if telemetry_side_effect:
                raise telemetry_side_effect
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
        auth_cache_ttl_seconds=0,
    )
    app = create_app(settings=settings, model_provider=model_provider or MockModelProvider(latency_ms=5))
    app.state.db_engine = MagicMock()

    class _MockContextManager:
        async def __aenter__(self):
            return mock_session

        async def __aexit__(self, *args):
            pass

    app.state.db_session_factory = lambda: _MockContextManager()
    return app, mock_session


def _make_auth_row(
    raw_key: str,
    user_id: str = "usr_chat_test",
    api_key_id: str = "key_chat_test",
    wallet_balance_usd: Decimal = Decimal("50.000000"),
    user_is_banned: bool = False,
    expires_at: datetime | None = None,
) -> dict[str, Any]:
    return {
        "api_key_id": api_key_id,
        "user_id": user_id,
        "key_hash": hash_api_key(raw_key),
        "key_prefix": raw_key[:7],
        "key_name": "Test Chat Key",
        "spend_limit_usd": Decimal("100.0000"),
        "expires_at": expires_at,
        "user_is_banned": user_is_banned,
        "user_email_verified": True,
        "wallet_balance_usd": wallet_balance_usd,
    }


# ── 1. Authentication & Pre-Inference Billing Gate Tests ──────


@pytest.mark.anyio
async def test_chat_completions_requires_auth():
    """POST /v1/chat/completions without credentials returns 401."""
    app, _ = _make_mock_app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": "rantera-base-v1",
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )
    assert resp.status_code == 401
    assert "Missing API key" in resp.json()["message"]


@pytest.mark.anyio
async def test_chat_completions_zero_balance_rejected():
    """POST /v1/chat/completions with wallet balance 0.0 returns HTTP 402."""
    raw_key = "sk-ran-testkeyzero"
    row = _make_auth_row(raw_key, wallet_balance_usd=Decimal("0.0"))
    app, _ = _make_mock_app([row])

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            "/v1/chat/completions",
            headers={"X-API-Key": raw_key},
            json={
                "model": "rantera-base-v1",
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )
    assert resp.status_code == 402
    assert "Insufficient wallet balance" in resp.json()["message"]
    assert resp.json()["error"] == "Payment Required"


@pytest.mark.anyio
async def test_chat_completions_negative_balance_rejected():
    """POST /v1/chat/completions with negative wallet balance returns HTTP 402."""
    raw_key = "sk-ran-testkeyneg"
    row = _make_auth_row(raw_key, wallet_balance_usd=Decimal("-1.500000"))
    app, _ = _make_mock_app([row])

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            "/v1/chat/completions",
            headers={"X-API-Key": raw_key},
            json={
                "model": "rantera-base-v1",
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )
    assert resp.status_code == 402
    assert "Insufficient wallet balance" in resp.json()["message"]


@pytest.mark.anyio
async def test_chat_completions_insufficient_for_estimated_prompt_cost():
    """When wallet balance is positive but lower than estimated minimum input cost, return 402."""
    raw_key = "sk-ran-testkeylow"
    # Set a tiny balance of $0.000010
    row = _make_auth_row(raw_key, wallet_balance_usd=Decimal("0.000010"))
    app, _ = _make_mock_app([row])

    # A 5,000 character prompt (~1250 tokens @ $0.003/1k = ~$0.003750)
    large_prompt = "Explain quantum computing in comprehensive detail. " * 100
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            "/v1/chat/completions",
            headers={"X-API-Key": raw_key},
            json={
                "model": "rantera-base-v1",
                "messages": [{"role": "user", "content": large_prompt}],
            },
        )
    assert resp.status_code == 402
    assert "Insufficient wallet balance for estimated request cost" in resp.json()["message"]


# ── 2. Request Validation and Limits Tests ────────────────────


@pytest.mark.anyio
async def test_chat_completions_rejects_streaming():
    """Streaming is currently unsupported and must return HTTP 400."""
    raw_key = "sk-ran-testkeystream"
    row = _make_auth_row(raw_key)
    app, _ = _make_mock_app([row])

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            "/v1/chat/completions",
            headers={"X-API-Key": raw_key},
            json={
                "model": "rantera-base-v1",
                "messages": [{"role": "user", "content": "Hello"}],
                "stream": True,
            },
        )
    assert resp.status_code == 400
    assert "Streaming is not currently supported" in resp.json()["message"]


@pytest.mark.anyio
async def test_chat_completions_unknown_model_rejected():
    """Unregistered model IDs return HTTP 404."""
    raw_key = "sk-ran-testkeymodel"
    row = _make_auth_row(raw_key)
    app, _ = _make_mock_app([row])

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            "/v1/chat/completions",
            headers={"X-API-Key": raw_key},
            json={
                "model": "nonexistent-model-v99",
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )
    assert resp.status_code == 404
    assert "not registered" in resp.json()["message"]


@pytest.mark.anyio
async def test_chat_completions_prompt_exceeds_context_window():
    """Prompt exceeding 8192 tokens returns HTTP 400."""
    raw_key = "sk-ran-testkeycontext"
    row = _make_auth_row(raw_key, wallet_balance_usd=Decimal("500.000000"))
    app, _ = _make_mock_app([row])

    # ~40,000 characters =~ 10,000 tokens (> 8192)
    giant_prompt = "overflow " * 6000
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            "/v1/chat/completions",
            headers={"X-API-Key": raw_key},
            json={
                "model": "rantera-base-v1",
                "messages": [{"role": "user", "content": giant_prompt}],
            },
        )
    assert resp.status_code == 400
    assert "exceeds model context window" in resp.json()["message"]


@pytest.mark.anyio
async def test_chat_completions_schema_rejects_extra_fields():
    """Extra fields such as client-supplied endpoint IDs are rejected with HTTP 422."""
    raw_key = "sk-ran-testkeyextra"
    row = _make_auth_row(raw_key)
    app, _ = _make_mock_app([row])

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            "/v1/chat/completions",
            headers={"X-API-Key": raw_key},
            json={
                "model": "rantera-base-v1",
                "messages": [{"role": "user", "content": "Hello"}],
                "vertex_endpoint_id": "malicious_override",
            },
        )
    assert resp.status_code == 422


# ── 3. Successful Inference & Telemetry Contract Tests ─────────


@pytest.mark.anyio
async def test_chat_completions_successful_generation():
    """Valid request executes mock prediction, calculates Decimal cost, and returns response."""
    raw_key = "sk-ran-testkeysuccess"
    row = _make_auth_row(raw_key, user_id="usr_chat_1", api_key_id="key_chat_1")
    app, mock_session = _make_mock_app([row])

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            "/v1/chat/completions",
            headers={"X-API-Key": raw_key},
            json={
                "model": "rantera-base-v1",
                "messages": [{"role": "user", "content": "What is machine learning?"}],
                "temperature": 0.5,
            },
        )

    assert resp.status_code == 200
    data = resp.json()

    assert data["id"].startswith("chatcmpl_")
    assert data["object"] == "chat.completion"
    assert data["model"] == "rantera-base-v1"
    assert "X-Request-ID" in resp.headers

    # Choice validation
    assert len(data["choices"]) == 1
    choice = data["choices"][0]
    assert choice["index"] == 0
    assert choice["finish_reason"] == "stop"
    assert choice["message"]["role"] == "assistant"
    assert "Rantera [rantera-base-v1]" in choice["message"]["content"]

    # Usage validation
    usage = data["usage"]
    assert usage["prompt_tokens"] > 0
    assert usage["completion_tokens"] > 0
    assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]
    assert Decimal(usage["cost_usd"]) > Decimal("0.000000")

    # Verify telemetry insert was executed
    insert_calls = [
        call for call in mock_session.execute.call_args_list
        if "INSERT INTO mp_inference_logs" in str(call[0][0])
    ]
    assert len(insert_calls) == 1
    params = insert_calls[0][0][1]

    assert params["user_id"] == "usr_chat_1"
    assert params["api_key_id"] == "key_chat_1"
    assert params["model_id"] == "rantera-base-v1"
    assert params["channel"] == "api"
    assert params["request_status"] == "SUCCESS"
    assert params["settlement_status"] == "PENDING"
    assert params["settlement_attempts"] == 0
    assert params["cost_usd"] == Decimal(usage["cost_usd"])
    assert params["tokens_in"] == usage["prompt_tokens"]
    assert params["tokens_out"] == usage["completion_tokens"]


@pytest.mark.anyio
async def test_chat_completions_max_tokens_truncation():
    """When max_tokens is constrained, output is truncated and finish_reason is 'length'."""
    raw_key = "sk-ran-testkeymaxtok"
    row = _make_auth_row(raw_key)
    app, _ = _make_mock_app([row])

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            "/v1/chat/completions",
            headers={"X-API-Key": raw_key},
            json={
                "model": "rantera-base-v1",
                "messages": [{"role": "user", "content": "Tell me a story"}],
                "max_tokens": 4,
            },
        )
    assert resp.status_code == 200
    data = resp.json()
    assert data["choices"][0]["finish_reason"] == "length"
    assert data["usage"]["completion_tokens"] == 4


# ── 4. Telemetry Fault Tolerance Tests ─────────────────────────


@pytest.mark.anyio
async def test_chat_completions_telemetry_failure_fails_open(caplog):
    """When telemetry DB insertion fails post-inference, the client response is preserved."""
    raw_key = "sk-ran-testkeyfailopen"
    row = _make_auth_row(raw_key)
    app, _ = _make_mock_app(
        mock_auth_rows=[row],
        telemetry_side_effect=RuntimeError("Deadlock detected during telemetry insert"),
    )

    with caplog.at_level(logging.ERROR):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post(
                "/v1/chat/completions",
                headers={"X-API-Key": raw_key},
                json={
                    "model": "rantera-base-v1",
                    "messages": [{"role": "user", "content": "Explain fault tolerance"}],
                },
            )

    # User response MUST be preserved (HTTP 200)
    assert resp.status_code == 200
    data = resp.json()
    assert data["choices"][0]["message"]["content"] != ""

    # Structured EMERGENCY log must be emitted
    assert any("EMERGENCY" in rec.message for rec in caplog.records)


# ── 5. Upstream Provider Failure Tests ─────────────────────────


@pytest.mark.anyio
async def test_chat_completions_upstream_provider_failure():
    """When model provider fails unexpectedly, return HTTP 502 Bad Gateway."""
    raw_key = "sk-ran-testkeyupstream"
    row = _make_auth_row(raw_key)

    class FailingProvider:
        async def predict(self, *args, **kwargs):
            raise ConnectionError("Upstream RPC timeout")

    app, _ = _make_mock_app([row], model_provider=FailingProvider())

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            "/v1/chat/completions",
            headers={"X-API-Key": raw_key},
            json={
                "model": "rantera-base-v1",
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )
    assert resp.status_code == 502
    assert "Upstream model prediction service failed" in resp.json()["message"]


@pytest.mark.anyio
async def test_chat_completions_production_telemetry_failure_fails_closed():
    """In production mode, if telemetry cannot be persisted, fail closed with HTTP 503."""
    raw_key = "sk-ran-testkeyprodtelemetry"
    row = _make_auth_row(raw_key)
    app, _ = _make_mock_app(
        mock_auth_rows=[row],
        telemetry_side_effect=RuntimeError("Primary DB down"),
    )
    # Configure production settings with fail-closed telemetry and valid cached catalog
    app.state.model_catalog_service._cache = dict(_REGISTRY)
    app.state.model_catalog_service._has_ever_loaded = True
    app.state.model_catalog_service._last_refresh_mono = time.monotonic()
    app.state.settings.rantera_env = "production"
    app.state.settings.telemetry_fail_closed = True

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            "/v1/chat/completions",
            headers={"X-API-Key": raw_key},
            json={
                "model": "rantera-base-v1",
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )
    assert resp.status_code == 503
    assert "Billing telemetry service unavailable" in resp.json()["message"]


# ── 6. Conservative Cost, Spend Limits & Concurrency Tests ─────


@pytest.mark.anyio
async def test_chat_completions_rejected_when_balance_insufficient_for_max_output_tokens():
    """Wallet covers input tokens but is insufficient for conservative max output tokens -> 402."""
    raw_key = "sk-ran-testkeylowoutput"
    # Balance is $0.005000 (can afford ~1600 input tokens, but cannot afford 4096 output tokens @ $0.061440)
    row = _make_auth_row(raw_key, wallet_balance_usd=Decimal("0.005000"))
    app, _ = _make_mock_app([row])

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        # Case 1: Unspecified max_tokens defaults to 4096 output tokens ($0.061440 max cost) -> rejected
        resp_default = await client.post(
            "/v1/chat/completions",
            headers={"X-API-Key": raw_key},
            json={
                "model": "rantera-base-v1",
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )
        assert resp_default.status_code == 402
        assert "Insufficient wallet balance for estimated request cost" in resp_default.json()["message"]

        # Case 2: Constrained max_tokens=50 (max cost ~$0.000780 <= $0.005000) -> succeeds!
        resp_constrained = await client.post(
            "/v1/chat/completions",
            headers={"X-API-Key": raw_key},
            json={
                "model": "rantera-base-v1",
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 50,
            },
        )
        assert resp_constrained.status_code == 200


@pytest.mark.anyio
async def test_chat_completions_rejected_when_request_exceeds_api_key_spend_limit():
    """Wallet has abundant balance, but request cost exceeds API key configured spend limit -> 402."""
    raw_key = "sk-ran-testkeyspendlimit"
    # Wallet balance is $100.0, but API key spend limit is $0.05
    row = _make_auth_row(raw_key, wallet_balance_usd=Decimal("100.000000"))
    row["spend_limit_usd"] = Decimal("0.0500")
    app, _ = _make_mock_app([row])

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        # Case 1: Unspecified max_tokens (~$0.061740 > $0.050000) -> rejected with 402
        resp_over = await client.post(
            "/v1/chat/completions",
            headers={"X-API-Key": raw_key},
            json={
                "model": "rantera-base-v1",
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )
        assert resp_over.status_code == 402
        assert "Spend limit reached" in resp_over.json()["message"]

        # Case 2: Small max_tokens=20 (~$0.000330 <= $0.050000) -> succeeds!
        resp_within = await client.post(
            "/v1/chat/completions",
            headers={"X-API-Key": raw_key},
            json={
                "model": "rantera-base-v1",
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 20,
            },
        )
        assert resp_within.status_code == 200


@pytest.mark.anyio
async def test_chat_completions_reconciles_and_releases_in_flight_spend():
    """Successful chat completion reconciles reservation and releases in-flight hold."""
    raw_key = "sk-ran-testkeyreconcile"
    user_id = "usr_chat_reconcile"
    row = _make_auth_row(raw_key, user_id=user_id, wallet_balance_usd=Decimal("50.000000"))
    app, _ = _make_mock_app([row])

    tracker = app.state.in_flight_spend_tracker
    assert await tracker.get_in_flight_spend(user_id) == Decimal("0.0")

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            "/v1/chat/completions",
            headers={"X-API-Key": raw_key},
            json={
                "model": "rantera-base-v1",
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )
        assert resp.status_code == 200

    # In-flight spend must be completely released back to 0.0 post-inference
    assert await tracker.get_in_flight_spend(user_id) == Decimal("0.0")


@pytest.mark.anyio
async def test_chat_completions_provider_failure_releases_in_flight_spend():
    """Provider failure releases in-flight reservation cleanly without leaking holds."""
    raw_key = "sk-ran-testkeyfailrelease"
    user_id = "usr_chat_failrelease"
    row = _make_auth_row(raw_key, user_id=user_id, wallet_balance_usd=Decimal("50.000000"))

    class ExplodingProvider:
        async def predict(self, *args, **kwargs):
            raise RuntimeError("Upstream Vertex GPU failure")

    app, _ = _make_mock_app([row], model_provider=ExplodingProvider())
    tracker = app.state.in_flight_spend_tracker

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            "/v1/chat/completions",
            headers={"X-API-Key": raw_key},
            json={
                "model": "rantera-base-v1",
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )
        assert resp.status_code == 502

    # In-flight spend must be released even after 502 provider failure
    assert await tracker.get_in_flight_spend(user_id) == Decimal("0.0")
