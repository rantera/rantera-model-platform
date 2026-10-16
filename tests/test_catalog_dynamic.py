"""Focused unit tests for dynamic model catalog caching, freshness boundaries, and production fail-closed rules."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException

from app.config import Settings
from app.models import ModelCatalogService, RegisteredModel


def _dev_settings() -> Settings:
    return Settings(
        rantera_env="development",
        gcp_project_id=None,
        gcp_region=None,
        vertex_endpoint_id=None,
    )


def _prod_settings() -> Settings:
    return Settings(
        rantera_env="production",
        gcp_project_id="prod-project",
        gcp_region="us-central1",
        vertex_endpoint_id="1234567890",
        database_url="postgresql+asyncpg://prod_user:prod_pass@localhost:5432/prod_db",
    )


def _mock_row(
    model_id: str = "rantera-base-v1",
    display_name: str = "Rantera Base v1 (Dynamic)",
    description: str = "Dynamic catalog test model",
    provider_model_id: str | None = None,
    context_window: int = 16384,
    max_output_tokens: int = 8192,
    input_cost: str = "0.002500",
    output_cost: str = "0.012500",
    is_active: bool = True,
) -> dict:
    return {
        "model_id": model_id,
        "display_name": display_name,
        "description": description,
        "provider_model_id": provider_model_id,
        "context_window": context_window,
        "max_output_tokens": max_output_tokens,
        "input_cost_per_1k": Decimal(input_cost),
        "output_cost_per_1k": Decimal(output_cost),
        "is_active": is_active,
    }


def _make_mock_session_factory(rows: list[dict] | None = None, side_effect: Exception | None = None):
    mock_session = AsyncMock()
    mock_result = MagicMock()
    if side_effect:
        mock_session.execute.side_effect = side_effect
    else:
        mock_result.fetchall.return_value = rows if rows is not None else []
        mock_session.execute.return_value = mock_result

    class _ContextManager:
        async def __aenter__(self):
            return mock_session

        async def __aexit__(self, *args):
            pass

    return lambda: _ContextManager(), mock_session


# ── Dynamic Loading and Caching Tests ────────────────────────


@pytest.mark.anyio
async def test_catalog_loads_from_database_rows():
    """Verify that ModelCatalogService parses and caches model specs from database query."""
    row = _mock_row(
        model_id="rantera-base-v1",
        display_name="Rantera Dynamic v1",
        provider_model_id="/tmp/afriqueqwen",
        context_window=16384,
        input_cost="0.002500",
    )
    session_factory, _ = _make_mock_session_factory([row])
    service = ModelCatalogService(cache_ttl_seconds=60.0)

    model = await service.get_model("rantera-base-v1", session_factory=session_factory, settings=_dev_settings())
    assert model is not None
    assert model.model_id == "rantera-base-v1"
    assert model.display_name == "Rantera Dynamic v1"
    assert model.provider_model_id == "/tmp/afriqueqwen"
    assert model.context_window == 16384
    assert model.input_cost_per_1k == Decimal("0.002500")


@pytest.mark.anyio
async def test_catalog_cached_within_ttl():
    """Within cache_ttl_seconds, do not re-execute database query."""
    row = _mock_row()
    session_factory, mock_session = _make_mock_session_factory([row])
    service = ModelCatalogService(cache_ttl_seconds=60.0)

    # First call queries DB
    await service.get_model("rantera-base-v1", session_factory=session_factory, settings=_dev_settings())
    assert mock_session.execute.call_count == 1

    # Second call within TTL does not query DB again
    await service.get_model("rantera-base-v1", session_factory=session_factory, settings=_dev_settings())
    assert mock_session.execute.call_count == 1


# ── Staleness & Fail-Closed Boundary Tests ────────────────────


@pytest.mark.anyio
async def test_catalog_serves_stale_cache_under_max_threshold():
    """When DB refresh fails but cache age <= max_stale_seconds, serve from cache with warning."""
    row = _mock_row(display_name="Stale But Valid")
    session_factory, _ = _make_mock_session_factory([row])
    service = ModelCatalogService(cache_ttl_seconds=1.0, max_stale_seconds=300.0)

    # Initial successful load
    await service.get_model("rantera-base-v1", session_factory=session_factory, settings=_prod_settings())

    # Simulate TTL expiration (age = 2s > ttl 1s, but < max_stale 300s)
    service._last_refresh_mono -= 2.0

    # DB now fails
    failing_factory, _ = _make_mock_session_factory(side_effect=RuntimeError("Database unreachable"))

    # Must still serve from cache because age (2s) <= max_stale (300s)
    model = await service.get_model("rantera-base-v1", session_factory=failing_factory, settings=_prod_settings())
    assert model is not None
    assert model.display_name == "Stale But Valid"


@pytest.mark.anyio
async def test_catalog_fails_closed_when_staleness_exceeds_threshold():
    """When DB refresh fails and cache age > max_stale_seconds, fail closed with HTTP 503."""
    row = _mock_row()
    session_factory, _ = _make_mock_session_factory([row])
    service = ModelCatalogService(cache_ttl_seconds=1.0, max_stale_seconds=5.0)

    # Initial load
    await service.get_model("rantera-base-v1", session_factory=session_factory, settings=_prod_settings())

    # Simulate age exceeding max_stale threshold (e.g. 10s > 5s)
    service._last_refresh_mono -= 10.0

    failing_factory, _ = _make_mock_session_factory(side_effect=RuntimeError("Database down"))

    with pytest.raises(HTTPException) as exc_info:
        await service.get_model("rantera-base-v1", session_factory=failing_factory, settings=_prod_settings())

    assert exc_info.value.status_code == 503
    assert "stale" in exc_info.value.detail.lower()


# ── Production vs Development Fallback Invariants ─────────────


@pytest.mark.anyio
async def test_catalog_production_never_falls_back_to_static_registry():
    """In production, if catalog has never loaded from database, fail closed with HTTP 503."""
    service = ModelCatalogService()
    # No session factory / database unreachable
    with pytest.raises(HTTPException) as exc_info:
        await service.get_model("rantera-base-v1", session_factory=None, settings=_prod_settings())

    assert exc_info.value.status_code == 503
    assert "unavailable" in exc_info.value.detail.lower()


@pytest.mark.anyio
async def test_catalog_development_allows_static_fallback():
    """In development, unconfigured database allows safe fallback to static registry for offline tests."""
    service = ModelCatalogService()
    model = await service.get_model("rantera-base-v1", session_factory=None, settings=_dev_settings())
    assert model is not None
    assert model.model_id == "rantera-base-v1"


@pytest.mark.anyio
async def test_dynamic_catalog_provider_model_id_survives_to_vertex_payload():
    """Verify that provider_model_id loaded dynamically from DB reaches the Vertex request payload."""
    from app.schemas import ChatMessage
    from app.vertex import VertexModelProvider

    row = _mock_row(
        model_id="rantera-base-v1",
        display_name="Rantera Base v1",
        provider_model_id="/tmp/afriqueqwen",
    )
    session_factory, _ = _make_mock_session_factory([row])
    service = ModelCatalogService()

    model = await service.get_model("rantera-base-v1", session_factory=session_factory, settings=_prod_settings())
    assert model is not None
    assert model.provider_model_id == "/tmp/afriqueqwen"

    # Verify VertexModelProvider uses provider_model_id in payload
    provider = VertexModelProvider(settings=_prod_settings())
    payload = provider._build_payload(
        model=model,
        messages=[ChatMessage(role="user", content="Translate this text.")],
    )
    assert payload["model"] == "/tmp/afriqueqwen"


@pytest.mark.anyio
async def test_dynamic_catalog_without_provider_model_id_falls_back_to_model_id():
    """Verify that when provider_model_id is NULL/None in DB, payload model falls back to model_id."""
    from app.schemas import ChatMessage
    from app.vertex import VertexModelProvider

    row = _mock_row(
        model_id="rantera-base-v1",
        display_name="Rantera Base v1",
        provider_model_id=None,
    )
    session_factory, _ = _make_mock_session_factory([row])
    service = ModelCatalogService()

    model = await service.get_model("rantera-base-v1", session_factory=session_factory, settings=_prod_settings())
    assert model is not None
    assert model.provider_model_id is None

    provider = VertexModelProvider(settings=_prod_settings())
    payload = provider._build_payload(
        model=model,
        messages=[ChatMessage(role="user", content="Hello")],
    )
    assert payload["model"] == "rantera-base-v1"
