"""Unit tests for async Vertex AI :rawPredict provider with mocked HTTP and authentication."""

from __future__ import annotations

import json
from decimal import Decimal
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fastapi import HTTPException

from app.config import Settings
from app.models import RegisteredModel
from app.schemas import ChatMessage
from app.vertex import (
    MockModelProvider,
    PredictionResult,
    VertexModelProvider,
    get_model_provider,
)


def _make_prod_settings(**overrides: Any) -> Settings:
    defaults = {
        "rantera_env": "production",
        "gcp_project_id": "203108330907",
        "gcp_region": "us-central1",
        "vertex_endpoint_id": "6728110112210157568",
        "rate_limit_enabled": True,
    }
    defaults.update(overrides)
    return Settings(**defaults)  # type: ignore[arg-type]


def _make_mock_credentials(token: str = "test-bearer-token-12345", valid: bool = True) -> MagicMock:
    creds = MagicMock()
    creds.token = token
    creds.valid = valid
    creds.refresh = MagicMock()
    return creds


def _make_sample_model(
    model_id: str = "rantera-base-v1",
    provider_model_id: str = "afriqueqwen-vllm-v4",
) -> RegisteredModel:
    return RegisteredModel(
        model_id=model_id,
        description="Rantera base model",
        display_name="Rantera Base v1",
        requires_vertex_config=True,
        input_cost_per_1k=Decimal("0.003000"),
        output_cost_per_1k=Decimal("0.015000"),
        context_window=8192,
        max_output_tokens=4096,
        is_active=True,
    )


# ── 1. Endpoint URL and Payload Construction ──────────────────


def test_endpoint_url_construction():
    """Verify target URL is constructed using the :rawPredict protocol."""
    settings = _make_prod_settings(
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


def test_payload_construction_matches_openai_schema():
    """Verify payload is constructed in OpenAI Chat Completions format required by vLLM."""
    settings = _make_prod_settings()
    provider = VertexModelProvider(settings=settings)
    model = _make_sample_model()
    messages = [
        ChatMessage(role="system", content="You are a helpful assistant."),
        ChatMessage(role="user", content="Translate to Swahili."),
    ]

    payload = provider._build_payload(
        model=model,
        messages=messages,
        temperature=0.5,
        max_tokens=2048,
    )

    assert payload["model"] == "rantera-base-v1"
    assert len(payload["messages"]) == 2
    assert payload["messages"][0] == {"role": "system", "content": "You are a helpful assistant."}
    assert payload["messages"][1] == {"role": "user", "content": "Translate to Swahili."}
    assert payload["temperature"] == 0.5
    assert payload["max_tokens"] == 2048
    assert payload["stream"] is False


def test_payload_construction_omits_max_tokens_when_none():
    """When max_tokens is None, max_tokens field is omitted from payload."""
    settings = _make_prod_settings()
    provider = VertexModelProvider(settings=settings)
    model = _make_sample_model()
    messages = [ChatMessage(role="user", content="Hello")]

    payload = provider._build_payload(model=model, messages=messages, max_tokens=None)
    assert "max_tokens" not in payload


def test_payload_construction_uses_provider_model_id():
    """Verify provider_model_id overrides logical model_id in the upstream vLLM payload."""
    settings = _make_prod_settings()
    provider = VertexModelProvider(settings=settings)
    model = RegisteredModel(
        model_id="rantera-base-v1",
        provider_model_id="/tmp/afriqueqwen",
        description="Rantera base model",
    )
    messages = [ChatMessage(role="user", content="Translate to Swahili.")]

    payload = provider._build_payload(
        model=model,
        messages=messages,
    )

    assert payload["model"] == "/tmp/afriqueqwen"


# ── 2. Successful Prediction and Response Parsing ─────────────


@pytest.mark.anyio
async def test_predict_success_with_mocked_client():
    """Verify successful prediction parses OpenAI vLLM response into PredictionResult."""
    settings = _make_prod_settings()
    mock_creds = _make_mock_credentials("secret-token-xyz")

    vllm_response = {
        "id": "chatcmpl-test-001",
        "object": "chat.completion",
        "created": 1729000000,
        "model": "afriqueqwen-vllm-v4",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Jambo! Karibu sana.",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 18,
            "completion_tokens": 8,
            "total_tokens": 26,
        },
    }

    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.status_code = 200
    mock_resp.json.return_value = vllm_response
    mock_client.post.return_value = mock_resp

    provider = VertexModelProvider(
        settings=settings,
        client=mock_client,
        credentials=mock_creds,
    )

    model = _make_sample_model()
    messages = [ChatMessage(role="user", content="Hello in Swahili")]

    result = await provider.predict(
        model=model,
        messages=messages,
        temperature=0.7,
        max_tokens=100,
    )

    assert isinstance(result, PredictionResult)
    assert result.content == "Jambo! Karibu sana."
    assert result.tokens_in == 18
    assert result.tokens_out == 8
    assert result.finish_reason == "stop"
    assert result.latency_ms >= 1

    # Verify request headers and URL
    mock_client.post.assert_awaited_once()
    call_args, call_kwargs = mock_client.post.call_args
    assert call_args[0].endswith(":rawPredict")
    assert call_kwargs["headers"]["Authorization"] == "Bearer secret-token-xyz"
    assert call_kwargs["headers"]["Content-Type"] == "application/json"


# ── 3. Upstream Error Handling and Fail-Closed Behavior ───────


@pytest.mark.anyio
async def test_predict_upstream_500_raises_502_bad_gateway():
    """Upstream 5xx from Vertex AI raises HTTP 502 without leaking internals."""
    settings = _make_prod_settings()
    mock_creds = _make_mock_credentials()

    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.status_code = 500
    mock_resp.text = "Internal Server Error in Container"
    mock_client.post.return_value = mock_resp

    provider = VertexModelProvider(
        settings=settings,
        client=mock_client,
        credentials=mock_creds,
    )
    model = _make_sample_model()

    with pytest.raises(HTTPException) as exc_info:
        await provider.predict(model=model, messages=[ChatMessage(role="user", content="Hi")])

    assert exc_info.value.status_code == 502
    assert "Upstream model prediction service failed" in exc_info.value.detail
    # Must NOT leak internal upstream details or endpoint IDs to client
    assert "6728110112210157568" not in exc_info.value.detail
    assert "secret" not in exc_info.value.detail


@pytest.mark.anyio
async def test_predict_upstream_429_quota_raises_502():
    """Upstream 429 quota exhaustion from Vertex AI maps safely to HTTP 502."""
    settings = _make_prod_settings()
    mock_creds = _make_mock_credentials()

    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.status_code = 429
    mock_resp.text = "Quota exceeded"
    mock_client.post.return_value = mock_resp

    provider = VertexModelProvider(
        settings=settings,
        client=mock_client,
        credentials=mock_creds,
    )
    model = _make_sample_model()

    with pytest.raises(HTTPException) as exc_info:
        await provider.predict(model=model, messages=[ChatMessage(role="user", content="Hi")])

    assert exc_info.value.status_code == 502
    assert "Upstream model prediction service failed" in exc_info.value.detail


@pytest.mark.anyio
async def test_predict_timeout_raises_504_gateway_timeout():
    """Upstream timeout cleanly maps to HTTP 504 Gateway Timeout."""
    settings = _make_prod_settings()
    mock_creds = _make_mock_credentials()

    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_client.post.side_effect = httpx.ReadTimeout("Connection timed out waiting for tokens")

    provider = VertexModelProvider(
        settings=settings,
        client=mock_client,
        credentials=mock_creds,
    )
    model = _make_sample_model()

    with pytest.raises(HTTPException) as exc_info:
        await provider.predict(model=model, messages=[ChatMessage(role="user", content="Hi")])

    assert exc_info.value.status_code == 504
    assert "timed out" in exc_info.value.detail


@pytest.mark.anyio
async def test_predict_network_error_raises_502():
    """Network connection failure maps safely to HTTP 502 Bad Gateway."""
    settings = _make_prod_settings()
    mock_creds = _make_mock_credentials()

    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_client.post.side_effect = httpx.ConnectError("Connection refused by endpoint")

    provider = VertexModelProvider(
        settings=settings,
        client=mock_client,
        credentials=mock_creds,
    )
    model = _make_sample_model()

    with pytest.raises(HTTPException) as exc_info:
        await provider.predict(model=model, messages=[ChatMessage(role="user", content="Hi")])

    assert exc_info.value.status_code == 502
    assert "Upstream model prediction service failed" in exc_info.value.detail


@pytest.mark.anyio
async def test_predict_malformed_json_response_raises_502():
    """Invalid JSON or missing choices array raises HTTP 502 with format error."""
    settings = _make_prod_settings()
    mock_creds = _make_mock_credentials()

    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"invalid_structure": True}
    mock_client.post.return_value = mock_resp

    provider = VertexModelProvider(
        settings=settings,
        client=mock_client,
        credentials=mock_creds,
    )
    model = _make_sample_model()

    with pytest.raises(HTTPException) as exc_info:
        await provider.predict(model=model, messages=[ChatMessage(role="user", content="Hi")])

    assert exc_info.value.status_code == 502
    assert "invalid response format" in exc_info.value.detail


# ── 4. Configuration and Environment Validation ──────────────


@pytest.mark.anyio
async def test_predict_unconfigured_vertex_fails_closed_503():
    """If Vertex AI settings are missing, predict immediately fails closed with HTTP 503."""
    settings = Settings(
        rantera_env="development",
        gcp_project_id=None,
        gcp_region=None,
        vertex_endpoint_id=None,
    )
    provider = VertexModelProvider(settings=settings)
    model = _make_sample_model()

    with pytest.raises(HTTPException) as exc_info:
        await provider.predict(model=model, messages=[ChatMessage(role="user", content="Hi")])

    assert exc_info.value.status_code == 503
    assert "not configured" in exc_info.value.detail


@pytest.mark.anyio
async def test_numeric_gcp_project_number_accepted():
    """Verify numeric GCP project number 203108330907 passes validation and builds URL."""
    settings = Settings(
        gcp_project_id="203108330907",
        gcp_region="us-central1",
        vertex_endpoint_id="6728110112210157568",
    )
    assert settings.gcp_project_id == "203108330907"
    provider = VertexModelProvider(settings=settings)
    assert "projects/203108330907/" in provider._build_endpoint_url()


@pytest.mark.anyio
async def test_predict_null_content_parsed_safely_as_empty_string():
    """Verify null content in upstream choice is safely normalized to empty string, not 'None'."""
    settings = _make_prod_settings()
    mock_creds = _make_mock_credentials()

    vllm_response = {
        "id": "chatcmpl-test-null",
        "object": "chat.completion",
        "created": 1729000000,
        "model": "afriqueqwen-vllm-v4",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 0,
            "total_tokens": 10,
        },
    }

    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.status_code = 200
    mock_resp.json.return_value = vllm_response
    mock_client.post.return_value = mock_resp

    provider = VertexModelProvider(
        settings=settings,
        client=mock_client,
        credentials=mock_creds,
    )

    model = _make_sample_model()
    messages = [ChatMessage(role="user", content="Test null")]

    result = await provider.predict(model=model, messages=messages)

    assert result.content == ""
    assert result.content != "None"
    assert result.tokens_in == 10
    assert result.tokens_out == 0


def test_get_model_provider_production_never_falls_back_to_mock():
    """Production mode must NEVER select MockModelProvider, even if Vertex config is missing."""
    settings = Settings(
        rantera_env="production",
        gcp_project_id=None,
        gcp_region=None,
        vertex_endpoint_id=None,
    )
    assert settings.is_production is True
    assert settings.vertex_configured is False

    provider = get_model_provider(settings)
    assert isinstance(provider, VertexModelProvider)
    assert not isinstance(provider, MockModelProvider)


def test_get_model_provider_development_uses_mock_when_unconfigured():
    """Development mode returns MockModelProvider when Vertex settings are unconfigured."""
    settings = Settings(
        rantera_env="development",
        gcp_project_id=None,
        gcp_region=None,
        vertex_endpoint_id=None,
    )
    assert settings.is_development is True
    assert settings.vertex_configured is False

    provider = get_model_provider(settings)
    assert isinstance(provider, MockModelProvider)
