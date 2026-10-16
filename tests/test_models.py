"""Tests for the model registry."""

from __future__ import annotations

from app.config import Settings
from app.models import get_model, get_model_ids, list_models, to_settlement_model_id


def _dev_settings(**overrides: object) -> Settings:
    defaults = {
        "rantera_env": "development",
        "gcp_project_id": None,
        "gcp_region": None,
        "vertex_endpoint_id": None,
    }
    defaults.update(overrides)
    return Settings(**defaults)  # type: ignore[arg-type]


# ── Registry contents ────────────────────────────────────────


def test_rantera_base_v1_is_registered():
    """The rantera-base-v1 model must be in the registry."""
    model = get_model("rantera-base-v1")
    assert model is not None
    assert model.model_id == "rantera-base-v1"
    assert len(model.description) > 0


def test_unknown_model_returns_none():
    """Querying an unregistered model ID returns None."""
    assert get_model("nonexistent-model") is None


def test_get_model_ids_includes_base():
    """get_model_ids returns a sorted list containing rantera-base-v1."""
    ids = get_model_ids()
    assert "rantera-base-v1" in ids
    assert ids == sorted(ids)


# ── Public model info does not expose internals ──────────────


def test_list_models_no_endpoint_ids():
    """list_models output must not contain endpoint IDs or project IDs."""
    settings = _dev_settings(
        gcp_project_id="secret-project",
        gcp_region="us-central1",
        vertex_endpoint_id="987654321",
    )
    models = list_models(settings)
    for m in models:
        dumped = m.model_dump()
        text = str(dumped)
        assert "secret-project" not in text
        assert "987654321" not in text
        assert "us-central1" not in text


def test_list_models_status_pending_when_unconfigured():
    """Models should report pending_configuration when Vertex is not set up."""
    settings = _dev_settings()
    models = list_models(settings)
    base = [m for m in models if m.id == "rantera-base-v1"]
    assert len(base) == 1
    assert base[0].status == "pending_configuration"


def test_list_models_status_available_when_configured():
    """Models should report available when Vertex is configured."""
    settings = _dev_settings(
        gcp_project_id="test-project",
        gcp_region="region",
        vertex_endpoint_id="12345",
    )
    models = list_models(settings)
    base = [m for m in models if m.id == "rantera-base-v1"]
    assert len(base) == 1
    assert base[0].status == "available"


# ── Immutability ─────────────────────────────────────────────


def test_registry_is_immutable():
    """Module-level _REGISTRY must reject item assignment and deletion."""
    import pytest
    from app.models import _REGISTRY, RegisteredModel

    with pytest.raises(TypeError):
        _REGISTRY["rantera-custom-v2"] = RegisteredModel(  # type: ignore[index]
            model_id="rantera-custom-v2",
            description="Mutation attempt",
            requires_vertex_config=False,
        )

    with pytest.raises(TypeError):
        del _REGISTRY["rantera-base-v1"]  # type: ignore[arg-type]


# ── Model Configuration & Settlement Mapping (Part 5) ────────


def test_rantera_base_v1_served_model_name():
    """Verify that rantera-base-v1 in the registry maps to the confirmed served model name /tmp/afriqueqwen."""
    model = get_model("rantera-base-v1")
    assert model is not None
    assert model.provider_model_id == "/tmp/afriqueqwen"
    assert model.display_name == "Rantera Base v1"


def test_rantera_base_v1_settlement_mapping():
    """Verify that rantera-base-v1 maps deterministically to the website UsageRecord ModelId enum RANTERA_BASE_V1."""
    model = get_model("rantera-base-v1")
    assert model is not None
    assert model.settlement_model_id == "RANTERA_BASE_V1"
    assert to_settlement_model_id("rantera-base-v1") == "RANTERA_BASE_V1"
    assert to_settlement_model_id("  rantera-base-v1  ") == "RANTERA_BASE_V1"


def test_rantera_base_v1_token_limits_and_pricing():
    """Verify context window, max output tokens, and approved pricing rates."""
    from decimal import Decimal

    model = get_model("rantera-base-v1")
    assert model is not None
    assert model.context_window == 8192
    assert model.max_output_tokens == 4096
    assert model.input_cost_per_1k == Decimal("0.003000")
    assert model.output_cost_per_1k == Decimal("0.015000")


def test_vertex_provider_payload_uses_registered_rantera_base_v1():
    """Verify Vertex provider payload uses the registered model's provider_model_id."""
    from app.schemas import ChatMessage
    from app.vertex import VertexModelProvider

    settings = Settings(
        rantera_env="production",
        gcp_project_id="203108330907",
        gcp_region="us-central1",
        vertex_endpoint_id="6728110112210157568",
    )
    provider = VertexModelProvider(settings=settings)
    model = get_model("rantera-base-v1")
    assert model is not None

    messages = [ChatMessage(role="user", content="Hello")]
    payload = provider._build_payload(model=model, messages=messages)
    assert payload["model"] == "/tmp/afriqueqwen"


def test_historical_vertex_endpoint_url_construction():
    """Verify endpoint URL construction with historical GCP project number, us-central1 region, and endpoint ID."""
    from app.vertex import VertexModelProvider

    settings = Settings(
        rantera_env="production",
        gcp_project_id="203108330907",
        gcp_region="us-central1",
        vertex_endpoint_id="6728110112210157568",
    )
    provider = VertexModelProvider(settings=settings)
    url = provider._build_endpoint_url()
    expected = (
        "https://us-central1-aiplatform.googleapis.com/v1/"
        "projects/203108330907/locations/us-central1/"
        "endpoints/6728110112210157568:rawPredict"
    )
    assert url == expected
