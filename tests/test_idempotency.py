"""Unit and integration tests for Part 3: Request Idempotency & Concurrency Safety."""

from __future__ import annotations

from decimal import Decimal
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import anyio
import pytest
from httpx import ASGITransport, AsyncClient

from app.auth import AuthenticatedPrincipal, clear_auth_cache, hash_api_key
from app.billing import InFlightSpendTracker
from app.config import Settings
from app.idempotency import (
    IdempotencyTracker,
    clear_idempotency_tracker,
    get_idempotency_tracker,
)
from app.main import create_app
from app.models import RegisteredModel
from app.schemas import ChatCompletionResponse, ChatMessage, TokenUsage
from app.vertex import BaseModelProvider, PredictionResult


@pytest.fixture(autouse=True)
def _clean_state():
    clear_auth_cache()
    clear_idempotency_tracker()
    yield
    clear_auth_cache()
    clear_idempotency_tracker()


class _ControlledModelProvider(BaseModelProvider):
    """Model provider with call counting, artificial delay, and failure injection."""

    def __init__(self, delay_seconds: float = 0.0, should_fail: bool = False) -> None:
        self.delay_seconds = delay_seconds
        self.should_fail = should_fail
        self.call_count = 0

    async def predict(
        self,
        model: RegisteredModel,
        messages: list[ChatMessage],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> PredictionResult:
        self.call_count += 1
        if self.delay_seconds > 0:
            await anyio.sleep(self.delay_seconds)
        if self.should_fail:
            raise RuntimeError("Injected upstream provider failure")
        return PredictionResult(
            content=f"Hello! Call #{self.call_count}",
            tokens_in=10,
            tokens_out=6,
            latency_ms=25,
            finish_reason="stop",
        )


def _make_auth_row(
    raw_key: str,
    user_id: str = "usr_idem_test",
    api_key_id: str = "key_idem_test",
    wallet_balance_usd: Decimal = Decimal("50.000000"),
) -> dict[str, Any]:
    return {
        "api_key_id": api_key_id,
        "user_id": user_id,
        "key_hash": hash_api_key(raw_key),
        "key_prefix": raw_key[:7],
        "key_name": "Test Key",
        "spend_limit_usd": Decimal("100.0000"),
        "channel": "api",
        "key_is_active": True,
        "expires_at": None,
        "user_is_banned": False,
        "wallet_balance_usd": wallet_balance_usd,
        "wallet_currency": "USD",
        "wallet_is_frozen": False,
    }


def _make_app(
    auth_rows: list[dict[str, Any]],
    model_provider: BaseModelProvider | None = None,
    idempotency_tracker: IdempotencyTracker | None = None,
    db_inference_logs_records: list[dict[str, Any]] | None = None,
) -> tuple[Any, AsyncMock, BaseModelProvider]:
    mock_session = AsyncMock()

    def _execute_side_effect(statement, params=None):
        stmt_str = str(statement)
        mock_res = MagicMock()
        if "mp_active_api_keys_v" in stmt_str:
            if params and "key_hash" in params:
                matching = [r for r in auth_rows if r.get("key_hash") == params["key_hash"]]
                mock_res.fetchall.return_value = matching
            else:
                mock_res.fetchall.return_value = auth_rows
            return mock_res
        elif "mp_inference_logs" in stmt_str:
            if "SELECT" in stmt_str and db_inference_logs_records is not None:
                mock_res.fetchall.return_value = db_inference_logs_records
                return mock_res
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
    provider = model_provider or _ControlledModelProvider()
    app = create_app(
        settings=settings,
        model_provider=provider,
        idempotency_tracker=idempotency_tracker,
    )
    app.state.db_engine = MagicMock()

    class _MockContextManager:
        async def __aenter__(self):
            return mock_session

        async def __aexit__(self, *args):
            pass

    app.state.db_session_factory = lambda: _MockContextManager()
    return app, mock_session, provider


# ── Requirement 1 & 5: Completed Duplicate Requests ─────────────────────────


@pytest.mark.anyio
async def test_completed_duplicate_replays_response_without_second_inference():
    """Duplicate request with same X-Request-ID returns cached response and never calls provider twice."""
    raw_key = "sk-ran-test-idempotency-001"
    auth_rows = [_make_auth_row(raw_key=raw_key, user_id="usr_idem_1")]
    app, _, provider = _make_app(auth_rows=auth_rows)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        req_headers = {
            "Authorization": f"Bearer {raw_key}",
            "X-Request-ID": "req_idem_unique_001",
        }
        body = {
            "model": "rantera-base-v1",
            "messages": [{"role": "user", "content": "What is the capital of France?"}],
            "max_tokens": 50,
        }

        # 1. Initial request -> executes inference
        res1 = await client.post("/v1/chat/completions", headers=req_headers, json=body)
        assert res1.status_code == 200
        data1 = res1.json()
        assert data1["choices"][0]["message"]["content"] == "Hello! Call #1"
        assert provider.call_count == 1

        # 2. Duplicate request with identical X-Request-ID -> served from cache
        res2 = await client.post("/v1/chat/completions", headers=req_headers, json=body)
        assert res2.status_code == 200
        data2 = res2.json()

        # Exact response replay preserved
        assert data2["id"] == data1["id"]
        assert data2["choices"][0]["message"]["content"] == "Hello! Call #1"
        assert data2["usage"] == data1["usage"]

        # Crucial check: Provider predict was NOT called a second time
        assert provider.call_count == 1


# ── Requirement 7: Spend Reservation & Lifecycle Interaction ─────────────────


@pytest.mark.anyio
async def test_duplicate_requests_do_not_reserve_or_charge_spend_twice():
    """Verifies that in-flight spend tracker holds 0 active spend after duplicate replay."""
    raw_key = "sk-ran-test-idempotency-spend-001"
    auth_rows = [_make_auth_row(raw_key=raw_key, user_id="usr_idem_spend")]
    app, _, provider = _make_app(auth_rows=auth_rows)

    tracker: InFlightSpendTracker = app.state.in_flight_spend_tracker

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        req_headers = {
            "Authorization": f"Bearer {raw_key}",
            "X-Request-ID": "req_idem_spend_001",
        }
        body = {
            "model": "rantera-base-v1",
            "messages": [{"role": "user", "content": "Count to three"}],
        }

        # Verify initial in-flight spend is zero
        assert await tracker.get_in_flight_spend("usr_idem_spend") == Decimal("0.0")

        # Initial request
        res1 = await client.post("/v1/chat/completions", headers=req_headers, json=body)
        assert res1.status_code == 200

        # After completion, reservation was reconciled and released
        assert await tracker.get_in_flight_spend("usr_idem_spend") == Decimal("0.0")

        # Duplicate replay
        res2 = await client.post("/v1/chat/completions", headers=req_headers, json=body)
        assert res2.status_code == 200

        # In-flight spend remains strictly zero (not reserved or held on replay)
        assert await tracker.get_in_flight_spend("usr_idem_spend") == Decimal("0.0")


# ── Requirement 2: Cross-Principal Request-ID Isolation ──────────────────────


@pytest.mark.anyio
async def test_different_principal_reusing_request_id_is_rejected_409():
    """Caller B using Caller A's X-Request-ID receives 409 Conflict without seeing Caller A's data."""
    raw_key_a = "sk-ran-caller-a-secret"
    raw_key_b = "sk-ran-caller-b-secret"

    auth_rows = [
        _make_auth_row(raw_key=raw_key_a, user_id="usr_caller_a", api_key_id="key_a"),
        _make_auth_row(raw_key=raw_key_b, user_id="usr_caller_b", api_key_id="key_b"),
    ]
    app, _, provider = _make_app(auth_rows=auth_rows)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        shared_request_id = "req_shared_identifier_999"

        # 1. Caller A executes request
        res_a = await client.post(
            "/v1/chat/completions",
            headers={"Authorization": f"Bearer {raw_key_a}", "X-Request-ID": shared_request_id},
            json={"model": "rantera-base-v1", "messages": [{"role": "user", "content": "Secret prompt from A"}]},
        )
        assert res_a.status_code == 200
        assert provider.call_count == 1

        # 2. Caller B attempts to use Caller A's request ID
        res_b = await client.post(
            "/v1/chat/completions",
            headers={"Authorization": f"Bearer {raw_key_b}", "X-Request-ID": shared_request_id},
            json={"model": "rantera-base-v1", "messages": [{"role": "user", "content": "Attempting to inspect A"}]},
        )

        # Must reject with 409 Conflict
        assert res_b.status_code == 409
        error_msg = res_b.json().get("message", res_b.json().get("detail", ""))
        assert "already in use by another principal" in error_msg

        # Caller A's response content must NOT be leaked
        assert "Secret prompt" not in res_b.text
        assert "Call #1" not in res_b.text

        # Provider must NOT have been called for Caller B
        assert provider.call_count == 1


# ── Requirement 3: Simultaneous Duplicate Requests ──────────────────────────


@pytest.mark.anyio
async def test_simultaneous_duplicate_requests_safely_handled():
    """Concurrent in-flight duplicate requests are detected; second receives 409 Conflict."""
    raw_key = "sk-ran-test-simultaneous-001"
    auth_rows = [_make_auth_row(raw_key=raw_key, user_id="usr_simultaneous")]

    # Provider introduces 0.15s delay so second request arrives while first is in-flight
    provider = _ControlledModelProvider(delay_seconds=0.15)
    app, _, _ = _make_app(auth_rows=auth_rows, model_provider=provider)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        req_id = "req_concurrent_atomic_test_001"
        headers = {
            "Authorization": f"Bearer {raw_key}",
            "X-Request-ID": req_id,
        }
        body = {
            "model": "rantera-base-v1",
            "messages": [{"role": "user", "content": "Slow inference query"}],
        }

        # Launch two requests concurrently using anyio TaskGroup (compatible with asyncio and trio)
        results: list[Any] = [None, None]

        async def _do_req(idx: int):
            results[idx] = await client.post("/v1/chat/completions", headers=headers, json=body)

        async with anyio.create_task_group() as tg:
            tg.start_soon(_do_req, 0)
            tg.start_soon(_do_req, 1)

        res1, res2 = results[0], results[1]

        # One must succeed (200 OK) and one must be rejected (409 Conflict: in progress)
        statuses = sorted([res1.status_code, res2.status_code])
        assert statuses == [200, 409]

        conflict_res = res1 if res1.status_code == 409 else res2
        conflict_msg = conflict_res.json().get("message", conflict_res.json().get("detail", ""))
        assert "currently in progress" in conflict_msg

        # Provider was called only once across both concurrent attempts
        assert provider.call_count == 1


# ── Requirement 6: Duplicate of Failed Request ──────────────────────────────


@pytest.mark.anyio
async def test_failed_request_duplicate_rejected_with_conflict():
    """If a request fails (502 Bad Gateway), retrying with the same request ID rejects with 409."""
    raw_key = "sk-ran-test-failed-req-001"
    auth_rows = [_make_auth_row(raw_key=raw_key, user_id="usr_fail_test")]

    # Provider will fail with RuntimeError -> translated to 502 Bad Gateway
    provider = _ControlledModelProvider(should_fail=True)
    app, _, _ = _make_app(auth_rows=auth_rows, model_provider=provider)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        req_id = "req_failing_inference_001"
        headers = {
            "Authorization": f"Bearer {raw_key}",
            "X-Request-ID": req_id,
        }
        body = {
            "model": "rantera-base-v1",
            "messages": [{"role": "user", "content": "Fail query"}],
        }

        # 1. Initial request fails with 502
        res1 = await client.post("/v1/chat/completions", headers=headers, json=body)
        assert res1.status_code == 502
        assert provider.call_count == 1

        # 2. Retrying with the same request ID is rejected with 409 Conflict
        res2 = await client.post("/v1/chat/completions", headers=headers, json=body)
        assert res2.status_code == 409
        error_msg = res2.json().get("message", res2.json().get("detail", ""))
        assert "failed" in error_msg
        assert "Please retry with a new request ID" in error_msg

        # Provider was NOT called again
        assert provider.call_count == 1


# ── Requirement 4 & 5: Database-Backed Cross-Instance Deduplication ─────────


@pytest.mark.anyio
async def test_cross_instance_duplicate_detected_in_database_returns_409():
    """When in-memory cache is empty but database contains completed SUCCESS log, returns 409."""
    raw_key = "sk-ran-test-cross-instance-001"
    auth_rows = [_make_auth_row(raw_key=raw_key, user_id="usr_cross_inst")]

    req_id = "req_cross_instance_executed_001"

    # Simulate existing database record written by another Cloud Run instance
    db_inference_record = {
        "request_id": req_id,
        "user_id": "usr_cross_inst",
        "api_key_id": "key_cross_inst",
        "model_id": "rantera-base-v1",
        "request_status": "SUCCESS",
        "tokens_in": 15,
        "tokens_out": 8,
        "cost_usd": Decimal("0.000150"),
        "error_code": None,
    }

    # Fresh tracker (no in-memory cache)
    fresh_tracker = IdempotencyTracker()
    app, _, provider = _make_app(
        auth_rows=auth_rows,
        idempotency_tracker=fresh_tracker,
        db_inference_logs_records=[db_inference_record],
    )

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        res = await client.post(
            "/v1/chat/completions",
            headers={"Authorization": f"Bearer {raw_key}", "X-Request-ID": req_id},
            json={"model": "rantera-base-v1", "messages": [{"role": "user", "content": "Hello"}]},
        )

        # Requirement 5: Return clear conflict response when full replay is unsupported by schema
        assert res.status_code == 409
        msg = res.json().get("message", res.json().get("detail", ""))
        assert "already been executed" in msg
        assert "Full response replay is not persisted" in msg

        # Crucial check: Provider was NOT called
        assert provider.call_count == 0


@pytest.mark.anyio
async def test_cross_instance_principal_mismatch_detected_in_database_returns_409():
    """Database record owned by Caller A rejects Caller B's request with 409."""
    raw_key_b = "sk-ran-test-caller-b-key"
    auth_rows = [_make_auth_row(raw_key=raw_key_b, user_id="usr_cross_b")]

    req_id = "req_cross_instance_owned_by_a"

    # Existing DB record belongs to usr_cross_a
    db_inference_record = {
        "request_id": req_id,
        "user_id": "usr_cross_a",  # Different user!
        "api_key_id": "key_a",
        "model_id": "rantera-base-v1",
        "request_status": "SUCCESS",
        "tokens_in": 10,
        "tokens_out": 5,
        "cost_usd": Decimal("0.000100"),
        "error_code": None,
    }

    fresh_tracker = IdempotencyTracker()
    app, _, provider = _make_app(
        auth_rows=auth_rows,
        idempotency_tracker=fresh_tracker,
        db_inference_logs_records=[db_inference_record],
    )

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        res = await client.post(
            "/v1/chat/completions",
            headers={"Authorization": f"Bearer {raw_key_b}", "X-Request-ID": req_id},
            json={"model": "rantera-base-v1", "messages": [{"role": "user", "content": "Attempting hijack"}]},
        )

        assert res.status_code == 409
        msg = res.json().get("message", res.json().get("detail", ""))
        assert "already in use by another principal" in msg
        assert provider.call_count == 0


# ── Unit Tests for IdempotencyTracker Internals ──────────────────────────────


@pytest.mark.anyio
async def test_tracker_ttl_expiration():
    """Tracker prunes expired cache entries after TTL."""
    tracker = IdempotencyTracker(cache_ttl_seconds=0.05)
    principal = AuthenticatedPrincipal(
        user_id="usr_ttl",
        api_key_id="key_ttl",
        key_name="TTL Key",
        wallet_balance=Decimal("50.0"),
        spend_limit=None,
        channel="api",
    )

    resp = ChatCompletionResponse(
        id="chatcmpl_test",
        object="chat.completion",
        created=123456789,
        model="rantera-base-v1",
        choices=[],
        usage=TokenUsage(
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
            cost_usd=Decimal("0.000105"),
        ),
    )

    # 1. Claim and complete
    await tracker.check_and_claim_request("req_ttl_1", principal)
    await tracker.complete_request("req_ttl_1", principal, resp)

    # Immediate replay -> returns cached response
    hit = await tracker.check_and_claim_request("req_ttl_1", principal)
    assert hit is not None
    assert hit.id == "chatcmpl_test"

    # Wait for TTL to expire
    await anyio.sleep(0.06)

    # After expiration, prune occurs and request ID is claimed anew
    miss = await tracker.check_and_claim_request("req_ttl_1", principal)
    assert miss is None
    # Now in progress again
    assert "req_ttl_1" in tracker._in_progress
