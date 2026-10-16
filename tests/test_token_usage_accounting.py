"""Unit and integration tests for Part 4: Safe Token Usage Accounting."""

from __future__ import annotations

from decimal import Decimal
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from fastapi import HTTPException
from httpx import ASGITransport, AsyncClient
from pydantic import ValidationError

from app.auth import clear_auth_cache, hash_api_key
from app.billing import InFlightSpendTracker
from app.config import Settings
from app.idempotency import clear_idempotency_tracker
from app.main import create_app
from app.models import RegisteredModel, calculate_inference_cost
from app.schemas import ChatMessage
from app.vertex import BaseModelProvider, PredictionResult, VertexModelProvider


@pytest.fixture(autouse=True)
def _clean_state():
    clear_auth_cache()
    clear_idempotency_tracker()
    yield
    clear_auth_cache()
    clear_idempotency_tracker()


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


def _make_mock_credentials() -> MagicMock:
    creds = MagicMock()
    creds.token = "mock-bearer-token-12345"
    creds.valid = True
    creds.refresh = MagicMock()
    return creds


def _make_sample_model() -> RegisteredModel:
    return RegisteredModel(
        model_id="rantera-base-v1",
        description="Rantera base model",
        display_name="Rantera Base v1",
        requires_vertex_config=True,
        input_cost_per_1k=Decimal("0.003000"),
        output_cost_per_1k=Decimal("0.015000"),
        context_window=8192,
        max_output_tokens=4096,
        is_active=True,
    )


def _make_vertex_provider(response_json: dict[str, Any], status_code: int = 200) -> VertexModelProvider:
    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.status_code = status_code
    mock_resp.json.return_value = response_json
    mock_client.post.return_value = mock_resp

    return VertexModelProvider(
        settings=_make_prod_settings(),
        client=mock_client,
        credentials=_make_mock_credentials(),
    )


# ── 1. VertexModelProvider: Valid Usage Parsing ──────────────────────────────


@pytest.mark.anyio
async def test_vertex_predict_valid_usage_parsed():
    """Valid upstream usage metadata is reliably parsed into PredictionResult."""
    vllm_resp = {
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Habari za asubuhi!"},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 14,
            "completion_tokens": 6,
            "total_tokens": 20,
        },
    }
    provider = _make_vertex_provider(vllm_resp)
    model = _make_sample_model()
    messages = [ChatMessage(role="user", content="Good morning in Swahili")]

    result = await provider.predict(model=model, messages=messages)

    assert result.content == "Habari za asubuhi!"
    assert result.tokens_in == 14
    assert result.tokens_out == 6
    assert result.finish_reason == "stop"


# ── 2. Missing Usage Metadata Never Defaulted to Zero ─────────────────────────


@pytest.mark.anyio
async def test_vertex_predict_missing_usage_raises_502():
    """Vertex response omitting the 'usage' object raises HTTP 502 (never defaults to 0)."""
    vllm_resp = {
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Response without usage"},
                "finish_reason": "stop",
            }
        ],
        # "usage" key completely missing
    }
    provider = _make_vertex_provider(vllm_resp)
    model = _make_sample_model()
    messages = [ChatMessage(role="user", content="Hello")]

    with pytest.raises(HTTPException) as exc_info:
        await provider.predict(model=model, messages=messages)

    assert exc_info.value.status_code == 502
    assert "invalid response format" in exc_info.value.detail


@pytest.mark.anyio
async def test_vertex_predict_null_usage_raises_502():
    """Vertex response with 'usage': None raises HTTP 502."""
    vllm_resp = {
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": "Hello"}, "finish_reason": "stop"}
        ],
        "usage": None,
    }
    provider = _make_vertex_provider(vllm_resp)
    with pytest.raises(HTTPException) as exc_info:
        await provider.predict(model=_make_sample_model(), messages=[ChatMessage(role="user", content="Hi")])
    assert exc_info.value.status_code == 502


@pytest.mark.anyio
async def test_vertex_predict_empty_usage_dict_raises_502():
    """Vertex response with 'usage': {} (missing token fields) raises HTTP 502."""
    vllm_resp = {
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": "Hello"}, "finish_reason": "stop"}
        ],
        "usage": {},
    }
    provider = _make_vertex_provider(vllm_resp)
    with pytest.raises(HTTPException) as exc_info:
        await provider.predict(model=_make_sample_model(), messages=[ChatMessage(role="user", content="Hi")])
    assert exc_info.value.status_code == 502


# ── 3. Malformed and Incomplete Usage Metadata ───────────────────────────────


@pytest.mark.anyio
async def test_vertex_predict_usage_not_a_dict_raises_502():
    """Vertex response with 'usage' as list or string raises HTTP 502."""
    vllm_resp = {
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hi"}}],
        "usage": "prompt_tokens: 10, completion_tokens: 5",
    }
    provider = _make_vertex_provider(vllm_resp)
    with pytest.raises(HTTPException) as exc_info:
        await provider.predict(model=_make_sample_model(), messages=[ChatMessage(role="user", content="Hi")])
    assert exc_info.value.status_code == 502


@pytest.mark.anyio
async def test_vertex_predict_missing_prompt_tokens_raises_502():
    """Usage missing 'prompt_tokens' raises HTTP 502."""
    vllm_resp = {
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hi"}}],
        "usage": {"completion_tokens": 5},
    }
    provider = _make_vertex_provider(vllm_resp)
    with pytest.raises(HTTPException) as exc_info:
        await provider.predict(model=_make_sample_model(), messages=[ChatMessage(role="user", content="Hi")])
    assert exc_info.value.status_code == 502


@pytest.mark.anyio
async def test_vertex_predict_missing_completion_tokens_raises_502():
    """Usage missing 'completion_tokens' raises HTTP 502."""
    vllm_resp = {
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hi"}}],
        "usage": {"prompt_tokens": 10},
    }
    provider = _make_vertex_provider(vllm_resp)
    with pytest.raises(HTTPException) as exc_info:
        await provider.predict(model=_make_sample_model(), messages=[ChatMessage(role="user", content="Hi")])
    assert exc_info.value.status_code == 502


@pytest.mark.anyio
async def test_vertex_predict_boolean_tokens_raises_502():
    """Usage containing booleans for prompt_tokens or completion_tokens raises HTTP 502."""
    vllm_resp = {
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hi"}}],
        "usage": {"prompt_tokens": True, "completion_tokens": 10},
    }
    provider = _make_vertex_provider(vllm_resp)
    with pytest.raises(HTTPException) as exc_info:
        await provider.predict(model=_make_sample_model(), messages=[ChatMessage(role="user", content="Hi")])
    assert exc_info.value.status_code == 502


@pytest.mark.anyio
async def test_vertex_predict_non_integer_tokens_raises_502():
    """Usage containing strings or non-numeric tokens raises HTTP 502."""
    vllm_resp = {
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hi"}}],
        "usage": {"prompt_tokens": "unmetered", "completion_tokens": 10},
    }
    provider = _make_vertex_provider(vllm_resp)
    with pytest.raises(HTTPException) as exc_info:
        await provider.predict(model=_make_sample_model(), messages=[ChatMessage(role="user", content="Hi")])
    assert exc_info.value.status_code == 502


@pytest.mark.anyio
async def test_vertex_predict_negative_tokens_raises_502():
    """Negative token counts in usage raise HTTP 502."""
    vllm_resp = {
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hi"}}],
        "usage": {"prompt_tokens": -1, "completion_tokens": 10},
    }
    provider = _make_vertex_provider(vllm_resp)
    with pytest.raises(HTTPException) as exc_info:
        await provider.predict(model=_make_sample_model(), messages=[ChatMessage(role="user", content="Hi")])
    assert exc_info.value.status_code == 502


# ── 4. Prompt Tokens and Content Consistency Checks ──────────────────────────


@pytest.mark.anyio
async def test_vertex_predict_zero_prompt_tokens_rejected_502():
    """Zero prompt_tokens reported for chat completion is rejected (prevents unmetered inference)."""
    vllm_resp = {
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hi"}}],
        "usage": {"prompt_tokens": 0, "completion_tokens": 5},
    }
    provider = _make_vertex_provider(vllm_resp)
    with pytest.raises(HTTPException) as exc_info:
        await provider.predict(model=_make_sample_model(), messages=[ChatMessage(role="user", content="Hi")])
    assert exc_info.value.status_code == 502


@pytest.mark.anyio
async def test_vertex_predict_inconsistent_zero_completion_tokens_rejected_502():
    """Non-empty generated text with zero completion_tokens is rejected as inconsistent accounting."""
    vllm_resp = {
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "Generated non-empty output"}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 0},
    }
    provider = _make_vertex_provider(vllm_resp)
    with pytest.raises(HTTPException) as exc_info:
        await provider.predict(model=_make_sample_model(), messages=[ChatMessage(role="user", content="Hi")])
    assert exc_info.value.status_code == 502


# ── 5. PredictionResult Consistency Validator ────────────────────────────────


def test_prediction_result_rejects_non_empty_content_with_zero_completion_tokens():
    """PredictionResult validator rejects non-empty content when tokens_out is zero."""
    with pytest.raises(ValidationError, match="Inconsistent token accounting"):
        PredictionResult(
            content="Valid non-empty generated text",
            tokens_in=10,
            tokens_out=0,
            latency_ms=25,
            finish_reason="stop",
        )


def test_prediction_result_accepts_empty_content_with_zero_completion_tokens():
    """PredictionResult allows zero completion tokens when content is empty string."""
    res = PredictionResult(
        content="",
        tokens_in=10,
        tokens_out=0,
        latency_ms=25,
        finish_reason="stop",
    )
    assert res.content == ""
    assert res.tokens_out == 0


# ── 6. Gateway Integration: Reservations & Idempotency Interaction ───────────


class _FlawedUsageProvider(BaseModelProvider):
    """Test provider returning corrupted or zero token usage."""

    def __init__(self, mode: str = "zero_prompt") -> None:
        self.mode = mode
        self.call_count = 0

    async def predict(self, model: RegisteredModel, messages: list[ChatMessage], **kwargs):
        self.call_count += 1
        if self.mode == "zero_prompt":
            return PredictionResult(content="Hello", tokens_in=0, tokens_out=5, latency_ms=10)
        elif self.mode == "zero_completion":
            # Direct instantiation with empty content bypassed to simulate inconsistent output
            # By passing whitespace content that strips to empty for Pydantic but has characters
            return PredictionResult(content="", tokens_in=10, tokens_out=0, latency_ms=10)
        return PredictionResult(content="Hello", tokens_in=10, tokens_out=5, latency_ms=10)


def _make_auth_row(raw_key: str, user_id: str = "usr_usage_test") -> dict[str, Any]:
    return {
        "api_key_id": "key_usage_test",
        "user_id": user_id,
        "key_hash": hash_api_key(raw_key),
        "key_prefix": raw_key[:7],
        "key_name": "Test Key",
        "spend_limit_usd": Decimal("100.0000"),
        "channel": "api",
        "key_is_active": True,
        "expires_at": None,
        "user_is_banned": False,
        "wallet_balance_usd": Decimal("50.000000"),
        "wallet_currency": "USD",
        "wallet_is_frozen": False,
    }


def _make_test_gateway(provider: BaseModelProvider, auth_rows: list[dict[str, Any]]) -> tuple[Any, AsyncMock]:
    mock_session = AsyncMock()

    def _execute_side_effect(statement, params=None):
        stmt_str = str(statement)
        mock_res = MagicMock()
        if "mp_active_api_keys_v" in stmt_str:
            mock_res.fetchall.return_value = auth_rows
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
    app = create_app(settings=settings, model_provider=provider)
    app.state.db_engine = MagicMock()

    class _MockContextManager:
        async def __aenter__(self):
            return mock_session

        async def __aexit__(self, *args):
            pass

    app.state.db_session_factory = lambda: _MockContextManager()
    return app, mock_session


@pytest.mark.anyio
async def test_chat_endpoint_invalid_usage_releases_reservation_and_fails_idempotent():
    """When model provider returns invalid token accounting:
    1. Gateway rejects with HTTP 502 Bad Gateway.
    2. In-flight spend reservation is safely released back to 0.
    3. Failure is recorded in idempotency tracker so duplicate retries return 409 Conflict.
    4. Provider is NOT called a second time.
    """
    raw_key = "sk-ran-test-token-usage-001"
    auth_rows = [_make_auth_row(raw_key=raw_key, user_id="usr_usage_idem")]
    flawed_provider = _FlawedUsageProvider(mode="zero_prompt")
    app, _ = _make_test_gateway(flawed_provider, auth_rows)

    tracker: InFlightSpendTracker = app.state.in_flight_spend_tracker
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        req_headers = {
            "Authorization": f"Bearer {raw_key}",
            "X-Request-ID": "req_flawed_usage_001",
        }
        body = {
            "model": "rantera-base-v1",
            "messages": [{"role": "user", "content": "Testing invalid usage accounting"}],
        }

        # Initial request: provider returns tokens_in=0 -> rejected with 502
        res1 = await client.post("/v1/chat/completions", headers=req_headers, json=body)
        assert res1.status_code == 502
        assert "invalid token usage metadata" in res1.json().get("message", res1.json().get("detail", ""))
        assert flawed_provider.call_count == 1

        # Part 2 verification: in-flight spend reservation is fully released (0.0 USD)
        assert await tracker.get_in_flight_spend("usr_usage_idem") == Decimal("0.0")

        # Part 3 verification: duplicate submission with same request ID is rejected with 409 Conflict
        res2 = await client.post("/v1/chat/completions", headers=req_headers, json=body)
        assert res2.status_code == 409
        conflict_msg = res2.json().get("message", res2.json().get("detail", ""))
        assert "failed" in conflict_msg
        assert "Please retry with a new request ID" in conflict_msg

        # Provider must NOT have been called a second time
        assert flawed_provider.call_count == 1
