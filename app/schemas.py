"""Request and response schemas for the Rantera Model Platform.

Uses strict Pydantic validation with bounded field sizes.
Does not allow clients to specify infrastructure identifiers.

Only schemas needed for the current phase are defined here.
Chat/inference schemas will be added when the Vertex AI integration is implemented.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field

from app.auth import AuthenticatedPrincipal


# ── Health Responses ─────────────────────────────────────────


class HealthLiveResponse(BaseModel):
    """Response for the liveness probe."""

    model_config = ConfigDict(extra="forbid")

    status: str = Field(
        ...,
        description="Process liveness status.",
        max_length=32,
    )


class HealthReadyResponse(BaseModel):
    """Response for the readiness probe."""

    model_config = ConfigDict(extra="forbid")

    status: str = Field(
        ...,
        description="Application readiness status.",
        max_length=32,
    )
    vertex_configured: bool | None = Field(
        default=None,
        description="Whether Vertex AI connection settings are present (exposed in development only).",
    )
    database_configured: bool | None = Field(
        default=None,
        description="Whether database connection settings are present (exposed in development only).",
    )
    database_ready: bool | None = Field(
        default=None,
        description="Whether database connectivity is verified (exposed in development only).",
    )
    details: str = Field(
        default="",
        description="Human-readable readiness detail (never exposes secrets).",
        max_length=256,
    )


# ── Model Registry Responses ────────────────────────────────


class ModelInfo(BaseModel):
    """Public-facing model metadata.

    Never includes Vertex endpoint IDs, GCP project IDs, or
    infrastructure configuration.
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(
        ...,
        description="Logical model identifier.",
        min_length=1,
        max_length=128,
    )
    description: str = Field(
        ...,
        description="Human-readable model description.",
        max_length=512,
    )
    status: str = Field(
        ...,
        description="Model status: 'available' or 'pending_configuration'.",
        max_length=32,
    )


class ModelListResponse(BaseModel):
    """Response for listing available models."""

    model_config = ConfigDict(extra="forbid")

    models: list[ModelInfo] = Field(
        ...,
        description="List of registered models.",
        max_length=100,
    )


# ── Error Responses ──────────────────────────────────────────


class ErrorResponse(BaseModel):
    """Structured error response.

    Never contains stack traces, credentials, endpoint IDs,
    or internal infrastructure details.
    """

    model_config = ConfigDict(extra="forbid")

    error: str = Field(
        ...,
        description="Error type or category.",
        max_length=64,
    )
    message: str = Field(
        ...,
        description="Safe, user-facing error message.",
        max_length=512,
    )
    request_id: str = Field(
        default="",
        description="Unique request identifier for correlation.",
        max_length=64,
    )


# ── Chat Completion Schemas (Part 4) ─────────────────────────


class ChatMessage(BaseModel):
    """Chat message in a conversation."""

    model_config = ConfigDict(extra="forbid")

    role: Literal["system", "user", "assistant"] = Field(
        ...,
        description="Message role: system, user, or assistant.",
    )
    content: str = Field(
        ...,
        description="Message content text.",
        min_length=1,
        max_length=65536,
    )


class ChatCompletionRequest(BaseModel):
    """Request payload for model inference via chat completions."""

    model_config = ConfigDict(extra="forbid")

    model: str = Field(
        ...,
        description="Model ID to invoke (e.g. 'rantera-base-v1').",
        min_length=1,
        max_length=64,
    )
    messages: list[ChatMessage] = Field(
        ...,
        description="List of messages in the conversation history.",
        min_length=1,
        max_length=100,
    )
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Sampling temperature between 0.0 and 2.0.",
    )
    max_tokens: int | None = Field(
        default=None,
        ge=1,
        le=4096,
        description="Optional maximum output tokens.",
    )
    stream: bool = Field(
        default=False,
        description="Streaming output flag (currently non-streaming only).",
    )


class TokenUsage(BaseModel):
    """Token consumption and monetary cost for the request."""

    model_config = ConfigDict(extra="forbid")

    prompt_tokens: int = Field(..., ge=0, description="Tokens consumed by prompt.")
    completion_tokens: int = Field(..., ge=0, description="Tokens generated in completion.")
    total_tokens: int = Field(..., ge=0, description="Total tokens consumed.")
    cost_usd: Decimal = Field(..., ge=Decimal("0.000000"), description="Calculated cost in USD.")


class ChatChoice(BaseModel):
    """Single completion choice."""

    model_config = ConfigDict(extra="forbid")

    index: int = Field(default=0, ge=0)
    message: ChatMessage = Field(...)
    finish_reason: str = Field(default="stop", max_length=32)


class ChatCompletionResponse(BaseModel):
    """Standard chat completion response."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(..., description="Unique response identifier (e.g. 'chatcmpl-...').")
    object: str = Field(default="chat.completion", max_length=32)
    created: int = Field(..., ge=0, description="Unix timestamp of completion.")
    model: str = Field(..., description="Model ID invoked.")
    choices: list[ChatChoice] = Field(...)
    usage: TokenUsage = Field(...)
