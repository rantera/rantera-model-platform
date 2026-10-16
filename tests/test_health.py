"""Tests for health endpoints."""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from app.config import Settings
from app.main import create_app


def _dev_settings(**overrides: object) -> Settings:
    """Create test settings with development defaults."""
    defaults = {
        "_env_file": None,
        "rantera_env": "development",
        "gcp_project_id": None,
        "gcp_region": None,
        "vertex_endpoint_id": None,
    }
    defaults.update(overrides)
    return Settings(**defaults)  # type: ignore[arg-type]


def _configured_settings() -> Settings:
    """Create test settings with Vertex AI configured."""
    return Settings(
        _env_file=None,
        rantera_env="development",
        gcp_project_id="test-project",
        gcp_region="us-central1",
        vertex_endpoint_id="1234567890",
    )


@pytest.fixture
def unconfigured_app():
    """App with no Vertex AI configuration."""
    return create_app(settings=_dev_settings())


@pytest.fixture
def configured_app():
    """App with Vertex AI configuration present."""
    return create_app(settings=_configured_settings())


# ── Liveness ─────────────────────────────────────────────────


@pytest.mark.anyio
async def test_health_live_returns_alive(unconfigured_app):
    """GET /health/live returns 200 with status 'alive'."""
    transport = ASGITransport(app=unconfigured_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/health/live")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "alive"


@pytest.mark.anyio
async def test_health_live_has_request_id(unconfigured_app):
    """Every response includes an X-Request-ID header."""
    transport = ASGITransport(app=unconfigured_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/health/live")
    assert "x-request-id" in resp.headers
    assert len(resp.headers["x-request-id"]) > 0


# ── Readiness (unconfigured) ────────────────────────────────


@pytest.mark.anyio
async def test_health_ready_unconfigured(unconfigured_app):
    """GET /health/ready reports 'degraded' with HTTP 503 when Vertex is not configured."""
    transport = ASGITransport(app=unconfigured_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/health/ready")
    assert resp.status_code == 503
    body = resp.json()
    assert body["status"] == "degraded"
    assert body["vertex_configured"] is False
    assert "Missing configuration" in body["details"]


# ── Readiness (configured) ──────────────────────────────────


@pytest.mark.anyio
async def test_health_ready_configured(configured_app):
    """GET /health/ready reports 'ready' when Vertex AI is configured."""
    transport = ASGITransport(app=configured_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/health/ready")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ready"
    assert body["vertex_configured"] is True
    # Must NOT claim upstream is verified.
    assert "not verified" in body["details"].lower() or "reachability" in body["details"].lower()


# ── Health endpoints expose no secrets ───────────────────────


@pytest.mark.anyio
async def test_health_endpoints_no_secrets(configured_app):
    """Health responses must not leak project IDs or endpoint IDs."""
    transport = ASGITransport(app=configured_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        live = await client.get("/health/live")
        ready = await client.get("/health/ready")

    for resp in (live, ready):
        text = resp.text
        assert "test-project" not in text
        assert "1234567890" not in text
        assert "us-central1" not in text


# ── Non-development readiness behavior ────────────────────────


@pytest.mark.anyio
async def test_health_ready_non_development_configured():
    """In non-dev, GET /health/ready returns 200 ready and must not disclose internal details."""
    import time
    from unittest.mock import AsyncMock, MagicMock

    settings = Settings(
        rantera_env="production",
        gcp_project_id="test-project",
        gcp_region="us-central1",
        vertex_endpoint_id="1234567890",
        database_url="postgresql+asyncpg://user:pass@localhost:5432/test_db",
    )
    app = create_app(settings=settings)

    # Mock database connectivity returning 1
    mock_conn = AsyncMock()
    mock_result = MagicMock()
    mock_result.scalar.return_value = 1
    mock_conn.execute.return_value = mock_result
    mock_engine = MagicMock()
    mock_engine.connect.return_value.__aenter__.return_value = mock_conn
    app.state.db_engine = mock_engine

    # Ensure catalog is loaded
    catalog_service = app.state.model_catalog_service
    catalog_service._cache = {"rantera-base-v1": MagicMock()}
    catalog_service._has_ever_loaded = True
    catalog_service._last_refresh_mono = time.monotonic()

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/health/ready")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ready"
    # Internal details must NOT be exposed in non-development
    assert "vertex_configured" not in body
    assert "database_configured" not in body
    assert "database_ready" not in body
    assert "Vertex AI" not in body["details"]
    assert "reachability not verified" in body["details"].lower()


@pytest.mark.anyio
async def test_health_ready_non_development_degraded():
    """In non-dev, degraded state returns HTTP 503 and must not disclose missing configuration variable names."""
    from app.config import Environment

    dev_settings = Settings(
        rantera_env="development",
        gcp_project_id=None,
        gcp_region=None,
        vertex_endpoint_id=None,
    )
    app = create_app(settings=dev_settings)
    # Simulate a non-development environment where configuration is incomplete
    app.state.settings.rantera_env = Environment.PRODUCTION

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/health/ready")
    assert resp.status_code == 503
    body = resp.json()
    assert body["status"] == "degraded"
    assert "vertex_configured" not in body
    assert body["details"] == "Service not fully configured."
    # Never leak internal variable names in response body
    assert "GCP_PROJECT_ID" not in body["details"]
    assert "Missing configuration" not in body["details"]


@pytest.mark.anyio
async def test_health_ready_production_missing_database_returns_503():
    """In production, missing database_url must return 503 degraded."""
    settings = Settings(
        rantera_env="production",
        gcp_project_id="test-project",
        gcp_region="us-central1",
        vertex_endpoint_id="1234567890",
        database_url=None,
    )
    app = create_app(settings=settings)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/health/ready")
    assert resp.status_code == 503
    body = resp.json()
    assert body["status"] == "degraded"
    assert "database_configured" not in body


@pytest.mark.anyio
async def test_health_ready_production_failed_db_connectivity_returns_503():
    """In production, database connectivity failure must return 503 degraded."""
    from unittest.mock import MagicMock

    settings = Settings(
        rantera_env="production",
        gcp_project_id="test-project",
        gcp_region="us-central1",
        vertex_endpoint_id="1234567890",
        database_url="postgresql+asyncpg://user:pass@localhost:5432/test_db",
    )
    app = create_app(settings=settings)
    mock_engine = MagicMock()
    mock_engine.connect.side_effect = ConnectionRefusedError("DB connection refused")
    app.state.db_engine = mock_engine

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/health/ready")
    assert resp.status_code == 503
    body = resp.json()
    assert body["status"] == "degraded"


@pytest.mark.anyio
async def test_health_ready_production_catalog_unavailable_returns_503():
    """In production, successful SELECT 1 alone does not imply readiness if model catalog is unavailable."""
    from unittest.mock import AsyncMock, MagicMock

    settings = Settings(
        rantera_env="production",
        gcp_project_id="test-project",
        gcp_region="us-central1",
        vertex_endpoint_id="1234567890",
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

    # Ensure model catalog has not loaded any models
    catalog_service = app.state.model_catalog_service
    catalog_service._cache = {}
    catalog_service._has_ever_loaded = False

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/health/ready")
    assert resp.status_code == 503
    body = resp.json()
    assert body["status"] == "degraded"


@pytest.mark.anyio
async def test_health_live_independent_of_database_and_vertex():
    """GET /health/live remains 200 alive even when DB and Vertex are unconfigured or failing."""
    from unittest.mock import MagicMock

    settings = Settings(
        rantera_env="development",
        gcp_project_id=None,
        gcp_region=None,
        vertex_endpoint_id=None,
        database_url="postgresql+asyncpg://user:pass@localhost:5432/test_db",
    )
    app = create_app(settings=settings)
    mock_engine = MagicMock()
    mock_engine.connect.side_effect = ConnectionRefusedError("DB down")
    app.state.db_engine = mock_engine

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        live_resp = await client.get("/health/live")
        ready_resp = await client.get("/health/ready")

    assert live_resp.status_code == 200
    assert live_resp.json()["status"] == "alive"
    assert ready_resp.status_code == 503
    assert ready_resp.json()["status"] == "degraded"


# ── Production Hardening Tests (Part 6) ──────────────────────


@pytest.mark.anyio
async def test_api_docs_policy_development_enabled():
    """In development mode, /docs and /openapi.json are accessible."""
    app = create_app(settings=_dev_settings())
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        docs_resp = await client.get("/docs")
        openapi_resp = await client.get("/openapi.json")

    assert docs_resp.status_code == 200
    assert openapi_resp.status_code == 200


@pytest.mark.anyio
async def test_api_docs_policy_production_disabled():
    """In production mode, /docs and /openapi.json are disabled (404)."""
    settings = Settings(
        rantera_env="production",
        gcp_project_id="test-project",
        gcp_region="us-central1",
        vertex_endpoint_id="1234567890",
    )
    app = create_app(settings=settings)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        docs_resp = await client.get("/docs")
        openapi_resp = await client.get("/openapi.json")

    assert docs_resp.status_code == 404
    assert openapi_resp.status_code == 404


@pytest.mark.anyio
async def test_security_headers_present_in_development():
    """Every response in development mode includes nosniff and DENY security headers."""
    app = create_app(settings=_dev_settings())
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/health/live")

    assert resp.status_code == 200
    assert resp.headers.get("x-content-type-options") == "nosniff"
    assert resp.headers.get("x-frame-options") == "DENY"
    # Strict-Transport-Security is not forced in development
    assert "strict-transport-security" not in resp.headers


@pytest.mark.anyio
async def test_security_headers_present_in_production():
    """Every response in production includes nosniff, DENY, and Strict-Transport-Security."""
    settings = Settings(
        rantera_env="production",
        gcp_project_id="test-project",
        gcp_region="us-central1",
        vertex_endpoint_id="1234567890",
    )
    app = create_app(settings=settings)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/health/live")

    assert resp.status_code == 200
    assert resp.headers.get("x-content-type-options") == "nosniff"
    assert resp.headers.get("x-frame-options") == "DENY"
    assert "max-age=31536000" in resp.headers.get("strict-transport-security", "")


@pytest.mark.anyio
async def test_request_body_size_limit_exceeded_returns_413():
    """Requests with Content-Length exceeding 1MB (1,048,576 bytes) return HTTP 413."""
    app = create_app(settings=_dev_settings())
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Simulate payload exceeding 1MB via Content-Length header
        resp = await client.post(
            "/v1/chat/completions",
            headers={
                "Content-Length": str(1_048_576 + 100),
                "Content-Type": "application/json",
            },
            content=b"{}",
        )

    assert resp.status_code == 413
    body = resp.json()
    assert body["error"] == "Request Entity Too Large"
    assert "exceeds maximum allowable limit" in body["message"]
    assert resp.headers.get("x-content-type-options") == "nosniff"
    assert resp.headers.get("x-frame-options") == "DENY"
