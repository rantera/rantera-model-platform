"""Tests for Pydantic schema validation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from decimal import Decimal

from app.schemas import (
    AuthenticatedPrincipal,
    ErrorResponse,
    HealthLiveResponse,
    HealthReadyResponse,
    ModelInfo,
    ModelListResponse,
)


# ── HealthLiveResponse ───────────────────────────────────────


def test_health_live_response_valid():
    resp = HealthLiveResponse(status="alive")
    assert resp.status == "alive"


def test_health_live_response_rejects_extra_fields():
    with pytest.raises(ValidationError):
        HealthLiveResponse(status="alive", secret="oops")  # type: ignore[call-arg]


def test_health_live_response_rejects_too_long_status():
    with pytest.raises(ValidationError):
        HealthLiveResponse(status="a" * 100)


# ── HealthReadyResponse ─────────────────────────────────────


def test_health_ready_response_valid():
    resp = HealthReadyResponse(
        status="ready",
        vertex_configured=True,
        details="All good.",
    )
    assert resp.vertex_configured is True


def test_health_ready_response_optional_vertex_configured():
    resp = HealthReadyResponse(
        status="ready",
        details="All good.",
    )
    assert resp.vertex_configured is None


def test_health_ready_response_rejects_extra():
    with pytest.raises(ValidationError):
        HealthReadyResponse(
            status="ready",
            vertex_configured=True,
            endpoint_id="leaked",  # type: ignore[call-arg]
        )


# ── ModelInfo ────────────────────────────────────────────────


def test_model_info_valid():
    info = ModelInfo(
        id="rantera-base-v1",
        description="Test model.",
        status="available",
    )
    assert info.id == "rantera-base-v1"


def test_model_info_rejects_extra_fields():
    """Clients cannot inject extra fields like endpoint_id."""
    with pytest.raises(ValidationError):
        ModelInfo(
            id="rantera-base-v1",
            description="Test.",
            status="available",
            endpoint_id="should-not-exist",  # type: ignore[call-arg]
        )


def test_model_info_rejects_empty_id():
    with pytest.raises(ValidationError):
        ModelInfo(id="", description="Test.", status="available")


def test_model_info_rejects_oversized_description():
    with pytest.raises(ValidationError):
        ModelInfo(id="test", description="x" * 600, status="available")


# ── ModelListResponse ────────────────────────────────────────


def test_model_list_response_valid():
    resp = ModelListResponse(
        models=[
            ModelInfo(id="test-model", description="A test.", status="available"),
        ]
    )
    assert len(resp.models) == 1


# ── ErrorResponse ────────────────────────────────────────────


def test_error_response_valid():
    err = ErrorResponse(
        error="Not Found",
        message="Resource not found.",
        request_id="abc123",
    )
    assert err.error == "Not Found"


def test_error_response_rejects_extra():
    with pytest.raises(ValidationError):
        ErrorResponse(
            error="Error",
            message="msg",
            stack_trace="should not be here",  # type: ignore[call-arg]
        )


# ── AuthenticatedPrincipal ───────────────────────────────────


def test_authenticated_principal_valid_defaults():
    principal = AuthenticatedPrincipal(user_id="user_123")
    assert principal.user_id == "user_123"
    assert principal.api_key_id is None
    assert principal.channel == "api"
    assert principal.wallet_balance == Decimal("0.0")
    assert principal.spend_limit is None
    assert principal.user_email_verified is True


def test_authenticated_principal_all_fields():
    principal = AuthenticatedPrincipal(
        user_id="user_456",
        api_key_id="key_789",
        channel="web",
        wallet_balance=Decimal("123.456700"),
        spend_limit=Decimal("500.0000"),
        key_prefix="sk-ran-",
        key_name="Test Key",
        user_email_verified=False,
    )
    assert principal.user_id == "user_456"
    assert principal.api_key_id == "key_789"
    assert principal.channel == "web"
    assert principal.wallet_balance == Decimal("123.456700")
    assert principal.spend_limit == Decimal("500.0000")
    assert principal.key_prefix == "sk-ran-"
    assert principal.key_name == "Test Key"
    assert principal.user_email_verified is False


def test_authenticated_principal_rejects_extra_fields():
    with pytest.raises(ValidationError):
        AuthenticatedPrincipal(
            user_id="user_123",
            password_hash="leaked",  # type: ignore[call-arg]
        )


def test_authenticated_principal_rejects_empty_user_id():
    with pytest.raises(ValidationError):
        AuthenticatedPrincipal(user_id="")


# ── Chat Completion Schemas (Part 4) ─────────────────────────

from app.schemas import (
    ChatChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    TokenUsage,
)


def test_chat_message_valid():
    msg = ChatMessage(role="user", content="Hello world")
    assert msg.role == "user"
    assert msg.content == "Hello world"


def test_chat_message_rejects_extra():
    with pytest.raises(ValidationError):
        ChatMessage(role="user", content="Hi", foo="bar")  # type: ignore[call-arg]


def test_chat_completion_request_valid_defaults():
    req = ChatCompletionRequest(
        model="rantera-base-v1",
        messages=[ChatMessage(role="user", content="Test")],
    )
    assert req.model == "rantera-base-v1"
    assert req.temperature == 0.7
    assert req.max_tokens is None
    assert req.stream is False


def test_chat_completion_request_rejects_extra_fields():
    with pytest.raises(ValidationError):
        ChatCompletionRequest(
            model="rantera-base-v1",
            messages=[ChatMessage(role="user", content="Test")],
            vertex_endpoint_id="12345",  # type: ignore[call-arg]
        )


def test_chat_completion_request_bounds():
    # temperature > 2.0 rejected
    with pytest.raises(ValidationError):
        ChatCompletionRequest(
            model="rantera-base-v1",
            messages=[ChatMessage(role="user", content="Test")],
            temperature=2.5,
        )
    # max_tokens > 4096 rejected
    with pytest.raises(ValidationError):
        ChatCompletionRequest(
            model="rantera-base-v1",
            messages=[ChatMessage(role="user", content="Test")],
            max_tokens=5000,
        )
    # empty messages rejected
    with pytest.raises(ValidationError):
        ChatCompletionRequest(
            model="rantera-base-v1",
            messages=[],
        )


def test_token_usage_valid():
    usage = TokenUsage(
        prompt_tokens=10,
        completion_tokens=20,
        total_tokens=30,
        cost_usd=Decimal("0.000330"),
    )
    assert usage.prompt_tokens == 10
    assert usage.cost_usd == Decimal("0.000330")


def test_token_usage_rejects_extra():
    with pytest.raises(ValidationError):
        TokenUsage(
            prompt_tokens=10,
            completion_tokens=20,
            total_tokens=30,
            cost_usd=Decimal("0.000330"),
            extra_field=True,  # type: ignore[call-arg]
        )


def test_chat_completion_response_valid():
    resp = ChatCompletionResponse(
        id="chatcmpl_abc123",
        created=1700000000,
        model="rantera-base-v1",
        choices=[
            ChatChoice(
                index=0,
                message=ChatMessage(role="assistant", content="Response"),
                finish_reason="stop",
            )
        ],
        usage=TokenUsage(
            prompt_tokens=5,
            completion_tokens=10,
            total_tokens=15,
            cost_usd=Decimal("0.000165"),
        ),
    )
    assert resp.id == "chatcmpl_abc123"
    assert resp.object == "chat.completion"
    assert len(resp.choices) == 1
    assert resp.choices[0].finish_reason == "stop"


def test_chat_completion_response_rejects_extra():
    with pytest.raises(ValidationError):
        ChatCompletionResponse(
            id="chatcmpl_abc123",
            created=1700000000,
            model="rantera-base-v1",
            choices=[],
            usage=TokenUsage(
                prompt_tokens=5,
                completion_tokens=10,
                total_tokens=15,
                cost_usd=Decimal("0.000165"),
            ),
            leaked_secret="token",  # type: ignore[call-arg]
        )


def test_chat_message_role_restriction():
    """ChatMessage must only accept supported roles: system, user, assistant."""
    # Valid roles
    for valid_role in ("system", "user", "assistant"):
        msg = ChatMessage(role=valid_role, content="Test")  # type: ignore[arg-type]
        assert msg.role == valid_role

    # Invalid roles must fail validation
    for invalid_role in ("admin", "root", "developer", "tool", "function", "bot", ""):
        with pytest.raises(ValidationError):
            ChatMessage(role=invalid_role, content="Test")  # type: ignore[arg-type]


def test_chat_message_content_length_limit():
    """ChatMessage content must enforce maximum length of 65536 characters."""
    # Exactly 65536 is accepted
    msg = ChatMessage(role="user", content="a" * 65536)
    assert len(msg.content) == 65536

    # 65537 is rejected
    with pytest.raises(ValidationError):
        ChatMessage(role="user", content="a" * 65537)


def test_chat_completion_request_message_count_limit():
    """ChatCompletionRequest must enforce maximum of 100 messages."""
    # Exactly 100 messages is accepted
    req = ChatCompletionRequest(
        model="rantera-base-v1",
        messages=[ChatMessage(role="user", content="Hi") for _ in range(100)],
    )
    assert len(req.messages) == 100

    # 101 messages is rejected
    with pytest.raises(ValidationError):
        ChatCompletionRequest(
            model="rantera-base-v1",
            messages=[ChatMessage(role="user", content="Hi") for _ in range(101)],
        )
