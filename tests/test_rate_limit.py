"""Comprehensive tests for Part 5.1: Local rate limiting and abuse prevention.

Verifies:
- Token-bucket threshold boundaries and burst exhaustion.
- Continuous refill arithmetic with injectable deterministic monotonic clock (no sleeps).
- Concurrency safety under concurrent anyio tasks.
- Separate-key and identity isolation (key:<api_key_id> vs user:<user_id>).
- Rejection of invalid, empty, or whitespace identities.
- Malformed configuration validation.
- Memory bounding, anti-quota-reset invariants, and capacity saturation fail-closed behavior.
- Cancellation safety during rate checks.
- HTTP header contracts on both 200 OK and 429 Too Many Requests (Retry-After, X-RateLimit-*).
- Strict pipeline ordering: rejection before wallet checks, catalog lookup, spend reservation, and provider calls.
- Production safety: failure to start when rate limiting is disabled without explicit bypass flag.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import anyio
import pytest
from httpx import ASGITransport, AsyncClient
from pydantic import ValidationError

from app.auth import AuthenticatedPrincipal, clear_auth_cache, hash_api_key
from app.config import Settings
from app.main import create_app
from app.rate_limit import (
    BaseRateLimiter,
    InMemoryRateLimiter,
    RateLimitResult,
    require_rate_limited_principal,
    resolve_rate_limit_identity,
)
from app.vertex import MockModelProvider


# ── Deterministic Clock Helpers ───────────────────────────────


class MockClock:
    """Deterministic monotonic clock for test execution without real-time sleeps."""

    def __init__(self, start: float = 1000.0) -> None:
        self.time = start

    def __call__(self) -> float:
        return self.time

    def advance(self, seconds: float) -> None:
        self.time += seconds


class MockWallClock:
    """Deterministic wall clock returning epoch timestamps."""

    def __init__(self, start: float = 1700000000.0) -> None:
        self.time = start

    def __call__(self) -> float:
        return self.time

    def advance(self, seconds: float) -> None:
        self.time += seconds


@pytest.fixture(autouse=True)
def _clean_cache():
    clear_auth_cache()
    yield
    clear_auth_cache()


def _make_auth_row(
    raw_key: str,
    user_id: str = "usr_ratelimit_test",
    api_key_id: str = "key_ratelimit_test",
    wallet_balance_usd: Decimal = Decimal("50.000000"),
    user_is_banned: bool = False,
    expires_at: datetime | None = None,
) -> dict[str, Any]:
    return {
        "api_key_id": api_key_id,
        "user_id": user_id,
        "key_hash": hash_api_key(raw_key),
        "key_prefix": raw_key[:7],
        "key_name": "Test Rate Limit Key",
        "spend_limit_usd": Decimal("100.0000"),
        "expires_at": expires_at,
        "user_is_banned": user_is_banned,
        "user_email_verified": True,
        "wallet_balance_usd": wallet_balance_usd,
    }


def _make_mock_app(
    mock_auth_rows: list[dict[str, Any]] | None = None,
    settings: Settings | None = None,
    rate_limiter: BaseRateLimiter | None = None,
    predict_mock: AsyncMock | None = None,
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
            mock_res.fetchall.return_value = []
            return mock_res
        mock_res.fetchall.return_value = []
        return mock_res

    mock_session.execute = AsyncMock(side_effect=_execute_side_effect)

    if settings is None:
        settings = Settings(
            rantera_env="development",
            gcp_project_id="test-project",
            gcp_region="us-central1",
            vertex_endpoint_id="1234567890",
            database_url="postgresql+asyncpg://user:pass@localhost:5432/test_db",
            auth_cache_ttl_seconds=0,
            rate_limit_rpm=60,
        )

    provider = MockModelProvider(latency_ms=1)
    if predict_mock is not None:
        provider.predict = predict_mock

    app = create_app(
        settings=settings,
        model_provider=provider,
        rate_limiter=rate_limiter,
    )
    app.state.db_engine = MagicMock()

    class _MockContextManager:
        async def __aenter__(self):
            return mock_session

        async def __aexit__(self, *args):
            pass

    app.state.db_session_factory = lambda: _MockContextManager()
    return app, mock_session


# ── 1. Threshold Boundaries & Refill Behavior ────────────────


@pytest.mark.anyio
async def test_rate_limiter_first_request_success():
    """Initial check on fresh bucket consumes 1 token and returns remaining=rpm-1."""
    mono_clock = MockClock(1000.0)
    wall_clock = MockWallClock(1700000000.0)
    limiter = InMemoryRateLimiter(rpm=60, clock=mono_clock, wall_clock=wall_clock)

    res = await limiter.check("key:test_key_1")
    assert res.allowed is True
    assert res.limit == 60
    assert res.remaining == 59
    assert res.reset_seconds == 1
    assert res.reset_at == 1700000001
    assert res.retry_after == 0


@pytest.mark.anyio
async def test_rate_limiter_burst_exhaustion_and_rejection():
    """Consuming full burst capacity allows exactly rpm requests; (rpm+1) is rejected."""
    mono_clock = MockClock(1000.0)
    wall_clock = MockWallClock(1700000000.0)
    limiter = InMemoryRateLimiter(rpm=10, clock=mono_clock, wall_clock=wall_clock)

    # First 10 requests must all succeed
    for i in range(10):
        res = await limiter.check("key:burst_test")
        assert res.allowed is True
        assert res.remaining == 9 - i

    # 11th request must be rejected
    rejected = await limiter.check("key:burst_test")
    assert rejected.allowed is False
    assert rejected.remaining == 0
    assert rejected.retry_after >= 1
    assert rejected.reset_seconds == 60
    assert rejected.reset_at == 1700000060


@pytest.mark.anyio
async def test_rate_limiter_continuous_refill_deterministic():
    """Advancing monotonic clock by fractional and full intervals refilled deterministically."""
    mono_clock = MockClock(1000.0)
    wall_clock = MockWallClock(1700000000.0)
    # 60 RPM = 1.0 token / second
    limiter = InMemoryRateLimiter(rpm=60, clock=mono_clock, wall_clock=wall_clock)

    # Deplete all 60 tokens
    for _ in range(60):
        res = await limiter.check("key:refill_test")
        assert res.allowed is True

    # Immediate next check fails
    res_rejected = await limiter.check("key:refill_test")
    assert res_rejected.allowed is False
    assert res_rejected.retry_after == 1

    # Advance clock by 0.5s (0.5 token refilled, not enough for 1.0 token)
    mono_clock.advance(0.5)
    wall_clock.advance(0.5)
    res_half = await limiter.check("key:refill_test")
    assert res_half.allowed is False
    assert res_half.remaining == 0
    assert res_half.retry_after == 1

    # Advance clock by another 0.5s (total 1.0 token refilled)
    mono_clock.advance(0.5)
    wall_clock.advance(0.5)
    res_one = await limiter.check("key:refill_test")
    assert res_one.allowed is True
    assert res_one.remaining == 0  # 1.0 available, 1.0 consumed

    # Advance clock by 120s (refills fully to capacity, does not overfill)
    mono_clock.advance(120.0)
    wall_clock.advance(120.0)
    res_full = await limiter.check("key:refill_test")
    assert res_full.allowed is True
    assert res_full.remaining == 59  # 60.0 refilled, 1.0 consumed -> 59


# ── 2. Concurrency Safety ────────────────────────────────────


@pytest.mark.anyio
async def test_rate_limiter_concurrency_safe():
    """Under concurrent async requests, exactly burst tokens are granted and surplus are rejected."""
    limiter = InMemoryRateLimiter(rpm=25, burst=25)
    success_count = 0
    reject_count = 0

    async def _worker():
        nonlocal success_count, reject_count
        res = await limiter.check("key:concurrent_key")
        if res.allowed:
            success_count += 1
        else:
            reject_count += 1

    async with anyio.create_task_group() as tg:
        for _ in range(50):
            tg.start_soon(_worker)

    assert success_count == 25
    assert reject_count == 25


# ── 3. Separate-Key & Identity Isolation ─────────────────────


@pytest.mark.anyio
async def test_rate_limiter_separate_key_isolation():
    """Exhausting quota on key A does not affect key B."""
    limiter = InMemoryRateLimiter(rpm=5)

    # Exhaust key A
    for _ in range(5):
        assert (await limiter.check("key:key_a")).allowed is True
    assert (await limiter.check("key:key_a")).allowed is False

    # Key B has independent quota
    res_b = await limiter.check("key:key_b")
    assert res_b.allowed is True
    assert res_b.remaining == 4


def test_resolve_rate_limit_identity_hierarchy():
    """Primary identity is key:<api_key_id>; fallback is user:<user_id> only if api_key_id is absent."""
    p_with_key = AuthenticatedPrincipal(
        user_id="usr_001",
        api_key_id="key_999",
    )
    assert resolve_rate_limit_identity(p_with_key) == "key:key_999"

    p_without_key = AuthenticatedPrincipal(
        user_id="usr_002",
        api_key_id=None,
    )
    assert resolve_rate_limit_identity(p_without_key) == "user:usr_002"

    p_with_whitespace_key = AuthenticatedPrincipal(
        user_id="usr_003",
        api_key_id="   ",
    )
    assert resolve_rate_limit_identity(p_with_whitespace_key) == "user:usr_003"


def test_resolve_rate_limit_identity_invalid_cases():
    """Rejects invalid types and empty/whitespace identities."""
    with pytest.raises(TypeError, match="Expected AuthenticatedPrincipal"):
        resolve_rate_limit_identity(None)  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="Expected AuthenticatedPrincipal"):
        resolve_rate_limit_identity("usr_123")  # type: ignore[arg-type]

    p_empty = AuthenticatedPrincipal(
        user_id="   ",
        api_key_id="   ",
    )
    with pytest.raises(ValueError, match="neither a valid api_key_id nor user_id"):
        resolve_rate_limit_identity(p_empty)


@pytest.mark.anyio
async def test_rate_limiter_rejects_empty_or_whitespace_key():
    """limiter.check rejects empty or whitespace keys."""
    limiter = InMemoryRateLimiter(rpm=60)
    with pytest.raises(ValueError, match="cannot be empty or whitespace"):
        await limiter.check("")

    with pytest.raises(ValueError, match="cannot be empty or whitespace"):
        await limiter.check("   \t  ")


# ── 4. Malformed Configuration Validation ────────────────────


def test_rate_limiter_init_bounds():
    """InMemoryRateLimiter rejects invalid initialization arguments."""
    with pytest.raises(ValueError, match="rpm must be >= 1"):
        InMemoryRateLimiter(rpm=0)

    with pytest.raises(ValueError, match="rpm must be >= 1"):
        InMemoryRateLimiter(rpm=-10)

    with pytest.raises(ValueError, match="burst must be >= 1"):
        InMemoryRateLimiter(rpm=60, burst=0)

    with pytest.raises(ValueError, match="max_keys must be >= 1"):
        InMemoryRateLimiter(rpm=60, max_keys=0)

    with pytest.raises(ValueError, match="idle_prune_seconds must be > 0"):
        InMemoryRateLimiter(rpm=60, idle_prune_seconds=0)

    with pytest.raises(ValueError, match="idle_prune_seconds must be > 0"):
        InMemoryRateLimiter(rpm=60, idle_prune_seconds=-5.0)


def test_settings_rate_limit_rpm_bounds():
    """Settings validates rate_limit_rpm bounds (ge=1, le=10000)."""
    # Valid values
    s1 = Settings(rate_limit_rpm=1)
    assert s1.rate_limit_rpm == 1
    s2 = Settings(rate_limit_rpm=10000)
    assert s2.rate_limit_rpm == 10000

    # Bounds violations
    with pytest.raises(ValidationError):
        Settings(rate_limit_rpm=0)

    with pytest.raises(ValidationError):
        Settings(rate_limit_rpm=-1)

    with pytest.raises(ValidationError):
        Settings(rate_limit_rpm=10001)


# ── 5. Memory Bounds, Anti-Quota-Reset & Capacity Saturation ─


@pytest.mark.anyio
async def test_rate_limiter_anti_quota_reset_pruning_policy():
    """Pruning safely removes ONLY fully-refilled, idle buckets; depleted buckets are never evicted."""
    clock = MockClock(1000.0)
    limiter = InMemoryRateLimiter(
        rpm=60,
        idle_prune_seconds=60.0,
        clock=clock,
    )

    # Key 1 consumes 1 token at t=1000 (remaining: 59)
    await limiter.check("key:1")
    # Key 2 consumes 60 tokens at t=1000 (depleted, remaining: 0)
    for _ in range(60):
        await limiter.check("key:2")

    assert await limiter.get_bucket_count() == 2

    # Advance clock by 30s (t=1030)
    # Key 1 has refilled to 60 tokens, but idle_time=30 < 60s -> NOT eligible for pruning
    # Key 2 has refilled to 30 tokens < 60 -> NOT eligible
    clock.advance(30.0)
    pruned = await limiter.prune()
    assert pruned == 0
    assert await limiter.get_bucket_count() == 2

    # Advance clock by another 35s (t=1065)
    # Key 1: refilled (60), idle_time=65 >= 60s -> PRUNED!
    # Key 2: refilled to 60 tokens (elapsed 65s), idle_time=65 >= 60s -> PRUNED!
    clock.advance(35.0)
    pruned = await limiter.prune()
    assert pruned == 2
    assert await limiter.get_bucket_count() == 0


@pytest.mark.anyio
async def test_rate_limiter_depleted_active_keys_never_evicted():
    """An actively requested key that is depleted is NEVER evicted to grant free quota."""
    clock = MockClock(1000.0)
    limiter = InMemoryRateLimiter(
        rpm=60,
        max_keys=2,
        idle_prune_seconds=60.0,
        clock=clock,
    )

    # Exhaust key:victim
    for _ in range(60):
        await limiter.check("key:victim")

    assert (await limiter.check("key:victim")).allowed is False

    # Keep key:victim active by sending requests every 20 seconds
    for _ in range(5):
        clock.advance(20.0)
        # Consumes the 20 refilled tokens
        for _ in range(20):
            await limiter.check("key:victim")
        # Quota remains depleted
        assert (await limiter.check("key:victim")).allowed is False
        # Pruning attempt must NOT evict key:victim
        pruned = await limiter.prune()
        assert pruned == 0

    assert await limiter.get_bucket_count() == 1


@pytest.mark.anyio
async def test_rate_limiter_capacity_saturation_fails_closed():
    """When max_keys is reached and existing keys are active/depleted, new keys fail closed."""
    clock = MockClock(1000.0)
    limiter = InMemoryRateLimiter(
        rpm=60,
        max_keys=2,
        idle_prune_seconds=60.0,
        clock=clock,
    )

    # Fill capacity with Key A and Key B
    await limiter.check("key:A")
    await limiter.check("key:B")
    assert await limiter.get_bucket_count() == 2

    # Key C arrives while Key A and Key B are active (idle < 60s)
    # Limiter cannot prune A or B, so it fails closed deterministically for C
    res_c = await limiter.check("key:C")
    assert res_c.allowed is False
    assert res_c.remaining == 0
    assert res_c.retry_after == 60
    assert await limiter.get_bucket_count() == 2  # Max keys respected

    # Existing tracked keys still function normally within their quota
    res_a = await limiter.check("key:A")
    assert res_a.allowed is True
    assert res_a.remaining == 58


# ── 6. Cancellation Safety ───────────────────────────────────


@pytest.mark.anyio
async def test_rate_limiter_cancellation_safety():
    """Cancelling during check leaves rate limiter state intact and usable."""
    limiter = InMemoryRateLimiter(rpm=60)

    with anyio.CancelScope() as cs:
        cs.cancel()
        await limiter.check("key:cancel_test")

    # Subsequent check proceeds normally without deadlock
    res = await limiter.check("key:cancel_test")
    assert res.allowed is True


# ── 7. HTTP Headers Contract (200 OK and 429 Too Many Requests)


@pytest.mark.anyio
async def test_chat_completions_returns_rate_limit_headers_on_success():
    """Successful 200 response returns X-RateLimit-Limit, Remaining, Reset, and X-Request-ID."""
    raw_key = "sk-ran-testkeyheaders"
    row = _make_auth_row(raw_key)
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

    assert resp.status_code == 200
    assert "X-Request-ID" in resp.headers
    assert resp.headers["X-RateLimit-Limit"] == "60"
    assert resp.headers["X-RateLimit-Remaining"] == "59"
    assert "X-RateLimit-Reset" in resp.headers
    assert int(resp.headers["X-RateLimit-Reset"]) > 0


@pytest.mark.anyio
async def test_chat_completions_returns_429_with_all_required_headers():
    """When rate limit is exceeded, HTTP 429 preserves Retry-After, Limit, Remaining=0, Reset."""
    raw_key = "sk-ran-testkey429"
    row = _make_auth_row(raw_key)
    mono_clock = MockClock(1000.0)
    wall_clock = MockWallClock(1700000000.0)
    limiter = InMemoryRateLimiter(rpm=2, clock=mono_clock, wall_clock=wall_clock)

    app, _ = _make_mock_app([row], rate_limiter=limiter)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        # Request 1: succeeds (remaining=1)
        r1 = await client.post(
            "/v1/chat/completions",
            headers={"X-API-Key": raw_key},
            json={"model": "rantera-base-v1", "messages": [{"role": "user", "content": "1"}]},
        )
        assert r1.status_code == 200
        assert r1.headers["X-RateLimit-Remaining"] == "1"

        # Request 2: succeeds (remaining=0)
        r2 = await client.post(
            "/v1/chat/completions",
            headers={"X-API-Key": raw_key},
            json={"model": "rantera-base-v1", "messages": [{"role": "user", "content": "2"}]},
        )
        assert r2.status_code == 200
        assert r2.headers["X-RateLimit-Remaining"] == "0"

        # Request 3: rejected with HTTP 429
        r3 = await client.post(
            "/v1/chat/completions",
            headers={"X-API-Key": raw_key},
            json={"model": "rantera-base-v1", "messages": [{"role": "user", "content": "3"}]},
        )
        assert r3.status_code == 429
        data = r3.json()
        assert data["error"] == "Too Many Requests"
        assert "Rate limit exceeded" in data["message"]
        assert "request_id" in data

        # Validate headers on 429 response
        assert "Retry-After" in r3.headers
        assert int(r3.headers["Retry-After"]) >= 1
        assert r3.headers["X-RateLimit-Limit"] == "2"
        assert r3.headers["X-RateLimit-Remaining"] == "0"
        assert "X-RateLimit-Reset" in r3.headers
        assert "X-Request-ID" in r3.headers


@pytest.mark.anyio
async def test_models_endpoint_has_rate_limit_headers():
    """GET /v1/models is also protected by rate limiting and returns headers."""
    raw_key = "sk-ran-testkeymodelsrl"
    row = _make_auth_row(raw_key)
    app, _ = _make_mock_app([row])

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/v1/models", headers={"X-API-Key": raw_key})

    assert resp.status_code == 200
    assert resp.headers["X-RateLimit-Limit"] == "60"
    assert resp.headers["X-RateLimit-Remaining"] == "59"
    assert "X-RateLimit-Reset" in resp.headers


# ── 8. Rejection Precedes Billing, Catalog & Inference ────────


@pytest.mark.anyio
async def test_rate_limit_rejection_precedes_wallet_check():
    """Rate limit rejection (429) precedes wallet balance check (402).

    A principal with wallet balance 0.0 who is rate limited receives HTTP 429, NOT 402.
    """
    raw_key = "sk-ran-testkeyzerowallet"
    # User has 0.0 balance (normally triggers 402)
    row = _make_auth_row(raw_key, wallet_balance_usd=Decimal("0.0"))
    limiter = InMemoryRateLimiter(rpm=1)
    # Pre-exhaust the limiter for this key
    await limiter.check(f"key:{row['api_key_id']}")

    app, _ = _make_mock_app([row], rate_limiter=limiter)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            "/v1/chat/completions",
            headers={"X-API-Key": raw_key},
            json={
                "model": "rantera-base-v1",
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )

    # Must be 429 Too Many Requests, NOT 402 Payment Required
    assert resp.status_code == 429
    assert resp.json()["error"] == "Too Many Requests"


@pytest.mark.anyio
async def test_rate_limit_rejection_precedes_provider_and_telemetry():
    """When a caller is rate limited, upstream provider is not invoked and no telemetry is recorded."""
    raw_key = "sk-ran-testkeypreceed"
    row = _make_auth_row(raw_key)
    limiter = InMemoryRateLimiter(rpm=1)
    await limiter.check(f"key:{row['api_key_id']}")

    predict_mock = AsyncMock()
    app, mock_session = _make_mock_app(
        [row],
        rate_limiter=limiter,
        predict_mock=predict_mock,
    )

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            "/v1/chat/completions",
            headers={"X-API-Key": raw_key},
            json={
                "model": "rantera-base-v1",
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )

    assert resp.status_code == 429
    # Provider predict MUST NOT be called
    predict_mock.assert_not_called()
    # Telemetry INSERT MUST NOT be executed
    insert_calls = [
        call for call in mock_session.execute.call_args_list
        if "INSERT INTO mp_inference_logs" in str(call[0][0])
    ]
    assert len(insert_calls) == 0


# ── 9. Production Safety Decisions ───────────────────────────


def test_production_readiness_rejects_disabled_rate_limit_without_bypass():
    """Production configuration must fail closed if rate limiting is disabled without bypass flag."""
    settings = Settings(
        rantera_env="production",
        gcp_project_id="prod-project",
        gcp_region="us-central1",
        vertex_endpoint_id="123456",
        rate_limit_enabled=False,
        rate_limit_allow_production_bypass=False,
    )
    missing = settings.validate_production_readiness()
    assert "RATE_LIMIT_ENABLED" in missing

    # create_app must fail with RuntimeError
    with pytest.raises(RuntimeError, match="RATE_LIMIT_ENABLED"):
        create_app(settings=settings)


def test_production_allows_rate_limit_disable_with_explicit_bypass(caplog):
    """Production allows disabling rate limiting ONLY when explicit bypass flag is set."""
    settings = Settings(
        rantera_env="production",
        gcp_project_id="prod-project",
        gcp_region="us-central1",
        vertex_endpoint_id="123456",
        rate_limit_enabled=False,
        rate_limit_allow_production_bypass=True,
    )
    missing = settings.validate_production_readiness()
    assert "RATE_LIMIT_ENABLED" not in missing
    assert missing == []

    with caplog.at_level(logging.WARNING):
        app = create_app(settings=settings)

    assert app is not None
    assert any("OPERATIONAL OVERRIDE" in rec.message for rec in caplog.records)


@pytest.mark.anyio
async def test_development_rate_limit_disabled_passes_unlimited():
    """In development, setting rate_limit_enabled=False bypasses rate limiting."""
    raw_key = "sk-ran-devdisabled"
    row = _make_auth_row(raw_key)
    settings = Settings(
        rantera_env="development",
        gcp_project_id="test-project",
        gcp_region="us-central1",
        vertex_endpoint_id="123456",
        database_url="postgresql+asyncpg://user:pass@localhost:5432/test_db",
        rate_limit_rpm=1,
        rate_limit_enabled=False,
    )
    app, _ = _make_mock_app([row], settings=settings)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        # Multiple requests succeed even though rate_limit_rpm=1
        for _ in range(5):
            resp = await client.post(
                "/v1/chat/completions",
                headers={"X-API-Key": raw_key},
                json={"model": "rantera-base-v1", "messages": [{"role": "user", "content": "Hi"}]},
            )
            assert resp.status_code == 200


# ── 10. Focused Regression Tests (Part 5.1.1) ──────────────────


@pytest.mark.anyio
async def test_unauthenticated_chat_completions_no_rate_limit_headers():
    """Unauthenticated POST /v1/chat/completions returns 401 with X-Request-ID and without X-RateLimit-* headers."""
    app, _ = _make_mock_app()

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        # Request with missing API key header
        resp_missing = await client.post(
            "/v1/chat/completions",
            json={"model": "rantera-base-v1", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert resp_missing.status_code == 401
        assert "X-Request-ID" in resp_missing.headers
        assert "X-RateLimit-Limit" not in resp_missing.headers
        assert "X-RateLimit-Remaining" not in resp_missing.headers
        assert "X-RateLimit-Reset" not in resp_missing.headers
        assert "Retry-After" not in resp_missing.headers

        # Request with invalid API key
        resp_invalid = await client.post(
            "/v1/chat/completions",
            headers={"X-API-Key": "sk-ran-invalid-key-999"},
            json={"model": "rantera-base-v1", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert resp_invalid.status_code == 401
        assert "X-Request-ID" in resp_invalid.headers
        assert "X-RateLimit-Limit" not in resp_invalid.headers
        assert "X-RateLimit-Remaining" not in resp_invalid.headers
        assert "X-RateLimit-Reset" not in resp_invalid.headers
        assert "Retry-After" not in resp_invalid.headers


@pytest.mark.anyio
async def test_malformed_principal_empty_identity_returns_403_and_blocks_pipeline():
    """Authenticated principal with whitespace-only identity returns 403 and never invokes billing or provider."""
    raw_key = "sk-ran-testmalformedid"
    # Row contains whitespace user_id and whitespace api_key_id
    row = _make_auth_row(raw_key, user_id="   ", api_key_id="   ")
    predict_mock = AsyncMock()
    app, mock_session = _make_mock_app([row], predict_mock=predict_mock)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            "/v1/chat/completions",
            headers={"X-API-Key": raw_key},
            json={"model": "rantera-base-v1", "messages": [{"role": "user", "content": "hi"}]},
        )

    assert resp.status_code == 403
    data = resp.json()
    assert data["error"] == "Forbidden"
    assert "Unable to establish caller identity" in data["message"]
    assert "X-Request-ID" in resp.headers

    # Upstream model provider predict MUST NOT be called
    predict_mock.assert_not_called()

    # Telemetry insertion MUST NOT be called
    insert_calls = [
        call for call in mock_session.execute.call_args_list
        if "INSERT INTO mp_inference_logs" in str(call[0][0])
    ]
    assert len(insert_calls) == 0


@pytest.mark.anyio
async def test_models_endpoint_exhaustion_returns_429_with_all_headers():
    """GET /v1/models returns HTTP 429 with all rate-limit headers when its quota is exhausted."""
    raw_key = "sk-ran-testmodels429"
    row = _make_auth_row(raw_key)
    mono_clock = MockClock(1000.0)
    wall_clock = MockWallClock(1700000000.0)
    limiter = InMemoryRateLimiter(rpm=2, clock=mono_clock, wall_clock=wall_clock)
    app, _ = _make_mock_app([row], rate_limiter=limiter)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        # Request 1: succeeds (remaining=1)
        r1 = await client.get("/v1/models", headers={"X-API-Key": raw_key})
        assert r1.status_code == 200
        assert r1.headers["X-RateLimit-Remaining"] == "1"

        # Request 2: succeeds (remaining=0)
        r2 = await client.get("/v1/models", headers={"X-API-Key": raw_key})
        assert r2.status_code == 200
        assert r2.headers["X-RateLimit-Remaining"] == "0"

        # Request 3: rejected with HTTP 429
        r3 = await client.get("/v1/models", headers={"X-API-Key": raw_key})
        assert r3.status_code == 429
        assert r3.json()["error"] == "Too Many Requests"
        assert "Rate limit exceeded" in r3.json()["message"]

        # Required headers on 429
        assert "Retry-After" in r3.headers
        assert int(r3.headers["Retry-After"]) >= 1
        assert r3.headers["X-RateLimit-Limit"] == "2"
        assert r3.headers["X-RateLimit-Remaining"] == "0"
        assert "X-RateLimit-Reset" in r3.headers
        assert int(r3.headers["X-RateLimit-Reset"]) >= 1700000000
        assert "X-Request-ID" in r3.headers


@pytest.mark.anyio
async def test_exhausted_caller_recovers_to_200_after_retry_window_without_sleep():
    """An exhausted caller recovers to HTTP 200 after retry window passes using injected monotonic clock."""
    raw_key = "sk-ran-testrecovery"
    row = _make_auth_row(raw_key)
    mono_clock = MockClock(1000.0)
    wall_clock = MockWallClock(1700000000.0)
    # 1 request per minute (refill rate = 1 token per 60s)
    limiter = InMemoryRateLimiter(rpm=1, clock=mono_clock, wall_clock=wall_clock)
    app, _ = _make_mock_app([row], rate_limiter=limiter)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        # 1. First request exhausts quota
        r1 = await client.post(
            "/v1/chat/completions",
            headers={"X-API-Key": raw_key},
            json={"model": "rantera-base-v1", "messages": [{"role": "user", "content": "hello"}]},
        )
        assert r1.status_code == 200
        assert r1.headers["X-RateLimit-Remaining"] == "0"

        # 2. Second immediate request is rejected with 429
        r2 = await client.post(
            "/v1/chat/completions",
            headers={"X-API-Key": raw_key},
            json={"model": "rantera-base-v1", "messages": [{"role": "user", "content": "retry too soon"}]},
        )
        assert r2.status_code == 429
        retry_after = int(r2.headers["Retry-After"])
        assert retry_after == 60

        # 3. Advance fake clock by partial window (30s) -> still rejected
        mono_clock.advance(30.0)
        wall_clock.advance(30.0)
        r_mid = await client.post(
            "/v1/chat/completions",
            headers={"X-API-Key": raw_key},
            json={"model": "rantera-base-v1", "messages": [{"role": "user", "content": "halfway"}]},
        )
        assert r_mid.status_code == 429
        assert int(r_mid.headers["Retry-After"]) == 30

        # 4. Advance fake clock by remaining 30s (total 60s passed) -> recovered to 200 OK!
        mono_clock.advance(30.0)
        wall_clock.advance(30.0)
        r_recovered = await client.post(
            "/v1/chat/completions",
            headers={"X-API-Key": raw_key},
            json={"model": "rantera-base-v1", "messages": [{"role": "user", "content": "recovered"}]},
        )
        assert r_recovered.status_code == 200
        assert r_recovered.headers["X-RateLimit-Remaining"] == "0"
        assert "X-Request-ID" in r_recovered.headers
