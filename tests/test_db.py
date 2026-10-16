"""Unit tests for PostgreSQL integration layer.

Covers:
- Missing database configuration & fail-closed behavior
- Engine initialization, connection pooling settings, and clean disposal via lifespan
- Database connection failure and timeout handling
- Readiness reporting degraded when database cannot be reached
- Liveness probe remaining completely independent of database availability
- Parameterized query helper execution & SQL injection prevention
- Prohibition of DDL execution and direct queries to core website tables
- Credential masking in settings and logs

All tests run in strict isolation using mocks and do NOT connect to any database.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException
from httpx import ASGITransport, AsyncClient
from pydantic import SecretStr, ValidationError
from sqlalchemy.ext.asyncio import AsyncEngine

from app.config import Settings
from app.db import (
    check_database_readiness,
    create_db_engine,
    execute_parameterized_query,
    get_db_session,
    normalize_database_url,
    validate_query_safety,
)
from app.main import create_app


# ── Fixtures & Helpers ─────────────────────────────────────────


def _make_settings(**overrides: Any) -> Settings:
    defaults = {
        "rantera_env": "development",
        "gcp_project_id": "test-project",
        "gcp_region": "us-central1",
        "vertex_endpoint_id": "1234567890",
    }
    defaults.update(overrides)
    return Settings(**defaults)  # type: ignore[arg-type]


# ── 1. Missing Database Configuration & Fail-Closed ────────────


def test_missing_database_configuration():
    """When DATABASE_URL is not set, engine creation fails and settings reflect unconfigured state."""
    settings = _make_settings(database_url=None)
    assert not settings.database_configured
    assert settings.validate_database_configuration() == ["DATABASE_URL"]

    with pytest.raises(ValueError, match="database_url is not configured"):
        create_db_engine(settings)


@pytest.mark.anyio
async def test_get_db_session_fails_closed_when_engine_missing():
    """get_db_session raises HTTP 503 when app.state.db_engine is None."""
    mock_request = MagicMock()
    mock_request.app.state.db_engine = None

    with pytest.raises(HTTPException) as exc_info:
        # Evaluate generator
        gen = get_db_session(mock_request)
        await anext(gen)

    assert exc_info.value.status_code == 503
    assert "Database service is unavailable" in exc_info.value.detail


@pytest.mark.anyio
async def test_readiness_check_returns_false_for_none_engine():
    """check_database_readiness returns False immediately if engine is None."""
    result = await check_database_readiness(None)
    assert result is False


# ── 2. Engine Initialization, Pooling & Lifespan Disposal ─────


@pytest.mark.anyio
async def test_lifespan_initializes_and_disposes_engine():
    """Engine is initialized during lifespan startup and disposed during shutdown."""
    settings = _make_settings(
        database_url="postgresql+asyncpg://test_user:test_pass@localhost:5432/test_db",
        db_pool_size=10,
        db_max_overflow=20,
    )

    mock_engine = AsyncMock()
    mock_engine.dispose = AsyncMock()

    mock_session = AsyncMock()
    mock_session.__aenter__.return_value = mock_session
    mock_session.__aexit__.return_value = None
    mock_factory = lambda: mock_session

    with (
        patch("app.main.create_db_engine", return_value=mock_engine) as mock_create,
        patch("app.main.get_session_factory", return_value=mock_factory),
    ):
        app = create_app(settings=settings)

        # Execute lifespan
        async with app.router.lifespan_context(app):
            mock_create.assert_called_once_with(settings)
            assert app.state.db_engine is mock_engine
            # Engine not yet disposed during active lifespan
            mock_engine.dispose.assert_not_called()

        # Engine must be cleanly disposed on lifespan exit
        mock_engine.dispose.assert_awaited_once()


def test_create_db_engine_configures_pooling_parameters():
    """create_db_engine passes pool size, overflow, timeouts, and pre-ping."""
    settings = _make_settings(
        database_url="postgresql+asyncpg://test_user:test_pass@localhost:5432/test_db",
        db_pool_size=15,
        db_max_overflow=25,
        db_pool_timeout_seconds=12.0,
        db_pool_recycle_seconds=1200,
        db_statement_timeout_ms=4000,
        db_connect_timeout_seconds=6.0,
    )

    with patch("app.db.create_async_engine") as mock_create_async:
        with patch("app.db._HAS_ASYNC_SQLALCHEMY", True):
            create_db_engine(settings)

            mock_create_async.assert_called_once()
            _, kwargs = mock_create_async.call_args
            assert kwargs["pool_size"] == 15
            assert kwargs["max_overflow"] == 25
            assert kwargs["pool_timeout"] == 12.0
            assert kwargs["pool_recycle"] == 1200
            assert kwargs["pool_pre_ping"] is True
            assert kwargs["connect_args"]["command_timeout"] == 4.0
            assert kwargs["connect_args"]["timeout"] == 6.0


# ── 3. Database Connection Failure and Timeout Behavior ───────


@pytest.mark.anyio
async def test_readiness_probe_handles_connection_failure():
    """check_database_readiness returns False on connection error without raising."""
    mock_engine = MagicMock()
    mock_engine.connect.side_effect = ConnectionRefusedError("Connection refused")

    result = await check_database_readiness(mock_engine, timeout_seconds=1.0)
    assert result is False


@pytest.mark.anyio
async def test_readiness_probe_timeout_behavior():
    """check_database_readiness returns False when query times out."""
    mock_conn = AsyncMock()

    # Simulate query hanging indefinitely
    async def slow_execute(*args: Any, **kwargs: Any) -> Any:
        await asyncio.sleep(1.0)
        return MagicMock(scalar=lambda: 1)

    mock_conn.execute.side_effect = slow_execute
    mock_engine = MagicMock()
    mock_engine.connect.return_value.__aenter__.return_value = mock_conn

    # Use a short timeout
    result = await check_database_readiness(mock_engine, timeout_seconds=0.05)
    assert result is False


# ── 4. Readiness Reports Degraded / Available ─────────────────


@pytest.mark.anyio
async def test_health_ready_reports_degraded_when_database_unreachable():
    """GET /health/ready reports 'degraded' when database check fails."""
    settings = _make_settings(
        database_url="postgresql+asyncpg://user:pass@localhost:5432/test_db",
    )
    app = create_app(settings=settings)

    mock_engine = MagicMock()
    mock_engine.connect.side_effect = Exception("DB connection error")
    app.state.db_engine = mock_engine

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/health/ready")

    assert resp.status_code == 503
    body = resp.json()
    assert body["status"] == "degraded"
    assert body["database_configured"] is True
    assert body["database_ready"] is False
    assert "Database connectivity unavailable" in body["details"]
    # Ensure no credentials leaked in details
    assert "pass" not in body["details"]
    assert "user" not in body["details"]


@pytest.mark.anyio
async def test_health_ready_reports_ready_when_database_connected():
    """GET /health/ready reports 'ready' when database probe returns 1."""
    settings = _make_settings(
        database_url="postgresql+asyncpg://user:pass@localhost:5432/test_db",
    )
    app = create_app(settings=settings)

    mock_conn = AsyncMock()
    mock_result = MagicMock()
    mock_result.scalar.return_value = 1
    mock_conn.execute.return_value = mock_result

    mock_engine = MagicMock()
    mock_engine.connect.return_value.__aenter__.return_value = mock_conn
    app.state.db_engine = mock_engine

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/health/ready")

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ready"
    assert body["database_configured"] is True
    assert body["database_ready"] is True
    assert "Database connected." in body["details"]


# ── 5. Liveness Remains Independent of Database Availability ──


@pytest.mark.anyio
async def test_health_live_independent_of_database():
    """GET /health/live returns 'alive' even when database engine is missing or failing."""
    settings = _make_settings(
        database_url="postgresql+asyncpg://user:pass@localhost:5432/test_db",
    )
    app = create_app(settings=settings)

    # Database is completely broken/unavailable
    mock_engine = MagicMock()
    mock_engine.connect.side_effect = ConnectionResetError("Broken pipe")
    app.state.db_engine = mock_engine

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        live_resp = await client.get("/health/live")
        ready_resp = await client.get("/health/ready")

    # Liveness MUST succeed
    assert live_resp.status_code == 200
    assert live_resp.json()["status"] == "alive"

    # Readiness reflects degraded state
    assert ready_resp.status_code == 503
    assert ready_resp.json()["status"] == "degraded"


# ── 6. Parameterized Query Helper & SQL Injection Prevention ───


@pytest.mark.anyio
async def test_parameterized_query_helper_executes_with_bindings():
    """execute_parameterized_query passes statement and parameters as bound values."""
    mock_session = AsyncMock()
    mock_result = MagicMock()
    mock_result.fetchall.return_value = [("row1",), ("row2",)]
    mock_session.execute.return_value = mock_result

    statement = "SELECT api_key_id FROM mp_active_api_keys_v WHERE key_hash = :hash"
    params = {"hash": "abc123def456"}

    rows = await execute_parameterized_query(mock_session, statement, params)

    assert len(rows) == 2
    mock_session.execute.assert_awaited_once()
    call_args = mock_session.execute.call_args
    # First argument is a SQLAlchemy TextClause
    assert str(call_args[0][0]) == statement
    # Second argument is the bound parameters dictionary
    assert call_args[0][1] == params


@pytest.mark.anyio
async def test_sql_injection_values_passed_as_literals():
    """Malicious input in parameters is bound safely rather than interpolated."""
    mock_session = AsyncMock()
    mock_result = MagicMock()
    mock_result.fetchall.return_value = []
    mock_session.execute.return_value = mock_result

    statement = "SELECT model_id FROM mp_model_catalog WHERE model_id = :model_id"
    # Classic SQL injection strings
    malicious_inputs = [
        "' OR '1'='1",
        "'; DROP TABLE mp_model_catalog; --",
        "admin'--",
    ]

    for attack in malicious_inputs:
        params = {"model_id": attack}
        await execute_parameterized_query(mock_session, statement, params)
        call_args = mock_session.execute.call_args
        # The statement itself is unchanged and unconcatenated
        assert str(call_args[0][0]) == statement
        # The payload remains isolated within the parameters dictionary
        assert call_args[0][1]["model_id"] == attack


# ── 7. Prohibition of DDL & Core Website Table Access ──────────


def test_ddl_operations_are_strictly_forbidden():
    """validate_query_safety raises ValueError on DDL commands."""
    disallowed_statements = [
        "CREATE TABLE test (id INT)",
        "ALTER TABLE mp_model_catalog ADD COLUMN new_col VARCHAR",
        "DROP TABLE mp_inference_logs",
        "TRUNCATE mp_inference_logs",
        "GRANT SELECT ON mp_model_catalog TO other_role",
        "REVOKE SELECT ON mp_model_catalog FROM other_role",
    ]

    for stmt in disallowed_statements:
        with pytest.raises(ValueError, match="DDL and privilege operations.*strictly forbidden"):
            validate_query_safety(stmt)


def test_direct_access_to_core_website_tables_is_forbidden():
    """validate_query_safety raises PermissionError when querying core website tables."""
    forbidden_queries = [
        'SELECT * FROM "User"',
        'SELECT * FROM User WHERE id = :id',
        'SELECT email FROM "ApiKey"',
        'SELECT balanceUsd FROM "Wallet"',
        'SELECT * FROM "Transaction"',
        'SELECT * FROM "UsageRecord"',
        'SELECT * FROM "Conversation"',
        'SELECT * FROM "Message"',
    ]

    for query in forbidden_queries:
        with pytest.raises(PermissionError, match="Direct access to website table.*forbidden"):
            validate_query_safety(query)


def test_approved_views_and_tables_are_permitted():
    """Approved views and model platform tables pass query safety validation."""
    approved_queries = [
        "SELECT * FROM mp_active_api_keys_v WHERE key_hash = :hash",
        "SELECT * FROM mp_active_users_v WHERE user_id = :user_id",
        "SELECT * FROM mp_model_catalog WHERE is_active = true",
        "SELECT * FROM mp_inference_logs WHERE request_id = :req_id",
        "INSERT INTO mp_inference_logs (request_id, model_id) VALUES (:req_id, :model_id)",
        "UPDATE mp_inference_logs SET settlement_status = 'SETTLED' WHERE request_id = :req_id",
    ]

    for query in approved_queries:
        # Should not raise any exception
        validate_query_safety(query)


# ── 8. Secret Hygiene & Configuration Safety ───────────────────


def test_database_url_credentials_masked_in_repr_and_str():
    """Database URL and passwords must not be exposed in settings repr or str."""
    settings = Settings(
        database_url="postgresql+asyncpg://service_user:super_secret_pw@10.0.0.1:5432/rantera_db",
    )
    r = repr(settings)
    s = str(settings)

    for output in (r, s):
        assert "super_secret_pw" not in output
        assert "service_user" not in output
        assert "10.0.0.1" not in output
        assert "database_configured=True" in output


def test_database_url_placeholder_rejection():
    """Placeholders for database_url are rejected and treated as None."""
    placeholders = [
        "<pending-database-url>",
        "change-me",
        "changeme",
        "todo",
        "fixme",
        "none",
        "null",
        "",
        "   ",
    ]

    for p in placeholders:
        settings = Settings(database_url=p)
        assert settings.database_url is None
        assert not settings.database_configured


def test_database_url_scheme_validation():
    """Only postgresql+asyncpg:// (and test schemes) are accepted."""
    # Valid
    s1 = Settings(database_url="postgresql+asyncpg://user:pass@host/db")
    assert s1.database_url is not None

    # Auto-normalized from postgresql://
    s2 = Settings(database_url="postgresql://user:pass@host/db")
    assert s2.database_url is not None
    assert s2.database_url.get_secret_value().startswith("postgresql+asyncpg://")

    # Invalid scheme
    with pytest.raises(ValueError, match="Invalid database URL scheme"):
        Settings(database_url="mysql+aiomysql://user:pass@host/db")


# ── 9. Real AsyncEngine Construction Without Live Connection ─


@pytest.mark.anyio
async def test_real_async_engine_construction_without_live_connection():
    """create_db_engine constructs a real AsyncEngine with asyncpg driver without connecting."""
    settings = _make_settings(
        database_url="postgresql://test_user:super_secret@127.0.0.1:5432/test_db",
        db_pool_size=5,
        db_max_overflow=10,
        db_pool_timeout_seconds=8.0,
        db_pool_recycle_seconds=900,
        db_statement_timeout_ms=2500,
        db_connect_timeout_seconds=4.0,
    )

    # Must create real AsyncEngine without attempting network connection
    engine = create_db_engine(settings)
    try:
        assert isinstance(engine, AsyncEngine)
        assert engine.url.drivername == "postgresql+asyncpg"
        assert engine.url.host == "127.0.0.1"
        assert engine.url.port == 5432
        assert engine.url.database == "test_db"
        assert engine.url.username == "test_user"

        # Safe representation: secret is masked
        assert "super_secret" not in str(engine.url)
        assert "***" in str(engine.url)

        # Connection pool starts lazy with 0 active connections
        assert engine.pool.checkedin() == 0
        assert engine.pool.checkedout() == 0
    finally:
        await engine.dispose()


# ── 10. PostgreSQL URL Normalization ──────────────────────────


def test_normalize_database_url():
    """normalize_database_url handles standard PostgreSQL prefixes and preserves others."""
    # postgresql:// -> postgresql+asyncpg://
    assert (
        normalize_database_url("postgresql://user:pass@host:5432/db")
        == "postgresql+asyncpg://user:pass@host:5432/db"
    )
    # postgres:// -> postgresql+asyncpg://
    assert (
        normalize_database_url("postgres://user:pass@host:5432/db")
        == "postgresql+asyncpg://user:pass@host:5432/db"
    )
    # Already normalized
    assert (
        normalize_database_url("postgresql+asyncpg://user:pass@host:5432/db")
        == "postgresql+asyncpg://user:pass@host:5432/db"
    )
    # SQLite test scheme remains unchanged
    assert (
        normalize_database_url("sqlite+aiosqlite:///:memory:")
        == "sqlite+aiosqlite:///:memory:"
    )
    # Whitespace is stripped
    assert (
        normalize_database_url("  postgresql://user:pass@host:5432/db  ")
        == "postgresql+asyncpg://user:pass@host:5432/db"
    )
    # sslmode=require -> ssl=require for asyncpg compatibility
    assert (
        normalize_database_url("postgresql://user:pass@host:5432/db?sslmode=require")
        == "postgresql+asyncpg://user:pass@host:5432/db?ssl=require"
    )
    # sslmode translated alongside other query parameters
    assert (
        normalize_database_url("postgresql+asyncpg://user:pass@host:5432/db?sslmode=prefer&connect_timeout=10")
        == "postgresql+asyncpg://user:pass@host:5432/db?ssl=prefer&connect_timeout=10"
    )
    # Already specified ssl=require remains unchanged
    assert (
        normalize_database_url("postgresql+asyncpg://user:pass@host:5432/db?ssl=require")
        == "postgresql+asyncpg://user:pass@host:5432/db?ssl=require"
    )
    # sslmode=disable -> ssl=disable
    assert (
        normalize_database_url("postgresql+asyncpg://user:pass@host:5432/db?sslmode=disable")
        == "postgresql+asyncpg://user:pass@host:5432/db?ssl=disable"
    )


# ── 11. Connection Pool Parameter Bounds & Validation ─────────


def test_invalid_pool_configuration_bounds():
    """Settings rejects out-of-bounds database connection pool parameters."""
    base = {
        "database_url": "postgresql+asyncpg://user:pass@host/db",
    }

    # db_pool_size < 1 or > 50
    with pytest.raises(ValidationError):
        Settings(**base, db_pool_size=0)
    with pytest.raises(ValidationError):
        Settings(**base, db_pool_size=51)

    # db_max_overflow < 0 or > 50
    with pytest.raises(ValidationError):
        Settings(**base, db_max_overflow=-1)
    with pytest.raises(ValidationError):
        Settings(**base, db_max_overflow=51)

    # db_pool_timeout_seconds < 1.0
    with pytest.raises(ValidationError):
        Settings(**base, db_pool_timeout_seconds=0.5)

    # db_pool_recycle_seconds < 60
    with pytest.raises(ValidationError):
        Settings(**base, db_pool_recycle_seconds=30)

    # db_statement_timeout_ms < 500 or > 30000
    with pytest.raises(ValidationError):
        Settings(**base, db_statement_timeout_ms=100)
    with pytest.raises(ValidationError):
        Settings(**base, db_statement_timeout_ms=40000)

    # db_connect_timeout_seconds < 1.0 or > 30.0
    with pytest.raises(ValidationError):
        Settings(**base, db_connect_timeout_seconds=0.5)
    with pytest.raises(ValidationError):
        Settings(**base, db_connect_timeout_seconds=40.0)


# ── 12. Session Acquisition and Cleanup Lifecycle ─────────────


@pytest.mark.anyio
async def test_session_acquisition_and_cleanup_on_success():
    """get_db_session yields an active session and exits context cleanly on completion."""
    mock_session = AsyncMock()
    mock_session.__aenter__.return_value = mock_session
    mock_session.__aexit__.return_value = None

    mock_request = MagicMock()
    mock_request.app.state.db_engine = MagicMock()
    mock_request.app.state.db_session_factory = lambda: mock_session

    gen = get_db_session(mock_request)
    session = await anext(gen)
    assert session is mock_session
    mock_session.__aenter__.assert_awaited_once()
    mock_session.__aexit__.assert_not_called()

    # Finish generator normally
    with pytest.raises(StopAsyncIteration):
        await anext(gen)

    mock_session.__aexit__.assert_awaited_once()


@pytest.mark.anyio
async def test_session_cleanup_and_rollback_on_failure():
    """get_db_session rolls back and exits context when an exception is raised."""
    mock_session = AsyncMock()
    mock_session.rollback = AsyncMock()
    mock_session.__aenter__.return_value = mock_session
    mock_session.__aexit__.return_value = None

    mock_request = MagicMock()
    mock_request.app.state.db_engine = MagicMock()
    mock_request.app.state.db_session_factory = lambda: mock_session

    gen = get_db_session(mock_request)
    session = await anext(gen)
    assert session is mock_session

    with pytest.raises(RuntimeError, match="Simulated handler error"):
        await gen.athrow(RuntimeError("Simulated handler error"))

    mock_session.rollback.assert_awaited_once()
    mock_session.__aexit__.assert_awaited_once()


@pytest.mark.anyio
async def test_session_cleanup_and_rollback_on_cancellation():
    """get_db_session rolls back and exits context on asyncio.CancelledError."""
    mock_session = AsyncMock()
    mock_session.rollback = AsyncMock()
    mock_session.__aenter__.return_value = mock_session
    mock_session.__aexit__.return_value = None

    mock_request = MagicMock()
    mock_request.app.state.db_engine = MagicMock()
    mock_request.app.state.db_session_factory = lambda: mock_session

    gen = get_db_session(mock_request)
    session = await anext(gen)
    assert session is mock_session

    with pytest.raises(asyncio.CancelledError):
        await gen.athrow(asyncio.CancelledError())

    mock_session.rollback.assert_awaited_once()
    mock_session.__aexit__.assert_awaited_once()


# ── 13. Lifespan Startup & Teardown Error Resilience ───────────


@pytest.mark.anyio
async def test_lifespan_engine_initialization_failure_is_non_fatal():
    """Lifespan handles engine creation failure gracefully without crashing the app."""
    settings = _make_settings(
        database_url="postgresql://user:pass@localhost:5432/test_db",
    )
    with patch("app.main.create_db_engine", side_effect=RuntimeError("Engine creation failed")):
        app = create_app(settings=settings)
        async with app.router.lifespan_context(app):
            assert app.state.db_engine is None
            assert app.state.db_session_factory is None

            # App continues running; readiness reports degraded
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.get("/health/ready")
                assert resp.status_code == 503
                assert resp.json()["status"] == "degraded"


@pytest.mark.anyio
async def test_lifespan_engine_disposal_error_is_non_fatal():
    """Lifespan handles engine disposal failure on shutdown without raising."""
    settings = _make_settings(
        database_url="postgresql://user:pass@localhost:5432/test_db",
    )
    mock_engine = AsyncMock()
    mock_engine.dispose.side_effect = RuntimeError("Dispose failure")

    mock_session = AsyncMock()
    mock_session.__aenter__.return_value = mock_session
    mock_session.__aexit__.return_value = None
    mock_factory = lambda: mock_session

    with (
        patch("app.main.create_db_engine", return_value=mock_engine),
        patch("app.main.get_session_factory", return_value=mock_factory),
    ):
        app = create_app(settings=settings)
        async with app.router.lifespan_context(app):
            assert app.state.db_engine is mock_engine

        # Lifespan exits cleanly despite dispose error
        mock_engine.dispose.assert_awaited_once()


# ── 14. Readiness Edge Cases & Response Consistency ───────────


@pytest.mark.anyio
async def test_readiness_probe_returns_false_when_scalar_not_one():
    """check_database_readiness returns False if SELECT 1 returns something other than 1."""
    mock_conn = AsyncMock()
    mock_result = MagicMock()
    mock_result.scalar.return_value = 0
    mock_conn.execute.return_value = mock_result

    mock_engine = MagicMock()
    mock_engine.connect.return_value.__aenter__.return_value = mock_conn

    result = await check_database_readiness(mock_engine)
    assert result is False


@pytest.mark.anyio
async def test_readiness_consistency_unconfigured_and_configured_states():
    """Readiness probe behavior is consistent across configured, degraded, and unconfigured states."""
    # 1. Unconfigured state in development
    dev_unconfigured = _make_settings(database_url=None)
    app1 = create_app(settings=dev_unconfigured)
    async with app1.router.lifespan_context(app1):
        assert app1.state.db_engine is None
        transport = ASGITransport(app=app1)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            r1 = await client.get("/health/ready")
            data1 = r1.json()
            assert data1["status"] == "ready"
            assert data1["database_configured"] is False
            assert data1.get("database_ready") is None

    # 2. Unconfigured state in production
    prod_unconfigured = _make_settings(
        rantera_env="production",
        database_url=None,
    )
    app2 = create_app(settings=prod_unconfigured)
    async with app2.router.lifespan_context(app2):
        transport = ASGITransport(app=app2)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            r2 = await client.get("/health/ready")
            assert r2.status_code == 503
            data2 = r2.json()
            assert data2["status"] == "degraded"
            assert data2.get("database_configured") is None
            assert data2.get("database_ready") is None

    # 3. Configured but degraded state in production
    prod_configured_degraded = _make_settings(
        rantera_env="production",
        database_url="postgresql://user:pass@localhost:5432/test_db",
    )
    mock_broken_engine = MagicMock()
    mock_broken_engine.connect.side_effect = ConnectionRefusedError("DB down")

    mock_session = AsyncMock()
    mock_session.__aenter__.return_value = mock_session
    mock_session.__aexit__.return_value = None
    mock_factory = lambda: mock_session

    with (
        patch("app.main.create_db_engine", return_value=mock_broken_engine),
        patch("app.main.get_session_factory", return_value=mock_factory),
    ):
        app3 = create_app(settings=prod_configured_degraded)
        async with app3.router.lifespan_context(app3):
            transport = ASGITransport(app=app3)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                r3 = await client.get("/health/ready")
                data3 = r3.json()
                assert data3["status"] == "degraded"
                assert data3.get("database_configured") is None
                assert data3.get("database_ready") is None
                assert data3["details"] == "Service not fully configured."
