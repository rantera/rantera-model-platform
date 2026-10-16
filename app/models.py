"""Model registry for the Rantera Model Platform.

Maps logical model IDs to internal configuration. The public model ID
never contains or reveals the actual Vertex AI endpoint ID.

Security invariants:
- Clients never see endpoint IDs, GCP project IDs, or region.
- The target endpoint is never accepted from a client request.
- No Google Cloud API calls are made from this module.
- No training, fine-tuning, or model deployment logic exists here.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from types import MappingProxyType
from typing import Optional

from app.config import Settings
from app.schemas import ModelInfo


@dataclass(frozen=True)
class RegisteredModel:
    """Internal model registration entry.

    Attributes:
        model_id: Public logical model identifier (e.g. 'rantera-base-v1').
        description: Human-readable description for API consumers.
        requires_vertex_config: Whether this model needs Vertex AI settings
            to be configured before it can serve predictions.
        display_name: Human-readable display name matching mp_model_catalog.
        context_window: Maximum total token limit (prompt + completion).
        max_output_tokens: Maximum allowed output tokens per request.
        input_cost_per_1k: Cost in USD per 1,000 prompt tokens.
        output_cost_per_1k: Cost in USD per 1,000 completion tokens.
        is_active: Whether the model is currently active for inference.
        provider_model_id: Internal model name accepted by the serving endpoint.
    """

    model_id: str
    description: str
    requires_vertex_config: bool = True
    display_name: str = "Rantera Base v1"
    context_window: int = 8192
    max_output_tokens: int = 4096
    input_cost_per_1k: Decimal = Decimal("0.003000")
    output_cost_per_1k: Decimal = Decimal("0.015000")
    is_active: bool = True
    provider_model_id: Optional[str] = None

    @property
    def settlement_model_id(self) -> str:
        """Uppercase PostgreSQL ModelId enum representation for website settlement.

        Maps kebab-case identifiers to uppercase snake-case enum values.
        E.g. 'rantera-base-v1' -> 'RANTERA_BASE_V1'
        """
        return to_settlement_model_id(self.model_id)


def to_settlement_model_id(model_id: str) -> str:
    """Map public kebab-case model ID to uppercase website settlement enum value.

    Example:
        to_settlement_model_id("rantera-base-v1") -> "RANTERA_BASE_V1"
    """
    return model_id.strip().upper().replace("-", "_")


# ── Static Registry ──────────────────────────────────────────
# Models are registered here. The Vertex endpoint mapping comes from
# server-side configuration (Settings), not from this registry.
# Protected against accidental mutation via MappingProxyType.

_REGISTRY: MappingProxyType[str, RegisteredModel] = MappingProxyType({
    "rantera-base-v1": RegisteredModel(
        model_id="rantera-base-v1",
        provider_model_id="/tmp/afriqueqwen",
        description="Rantera base model for internal evaluation and experimentation.",
        requires_vertex_config=True,
        display_name="Rantera Base v1",
        context_window=8192,
        max_output_tokens=4096,
        input_cost_per_1k=Decimal("0.003000"),
        output_cost_per_1k=Decimal("0.015000"),
        is_active=True,
    ),
})


import anyio
import logging
import time
from typing import Any

from fastapi import HTTPException, status
from sqlalchemy import text

from app.db import validate_query_safety

logger = logging.getLogger(__name__)


def calculate_inference_cost(
    model: RegisteredModel,
    tokens_in: int,
    tokens_out: int,
) -> Decimal:
    """Calculate inference cost in USD using exact 6-decimal arithmetic.

    Matches PostgreSQL DECIMAL(12, 6) precision:
    cost = (tokens_in * input_cost_per_1k + tokens_out * output_cost_per_1k) / 1000

    Raises:
        TypeError: If token counts are booleans or not integers.
        ValueError: If token counts are negative.
    """
    if isinstance(tokens_in, bool) or isinstance(tokens_out, bool):
        raise TypeError(
            f"Token counts must be integers, not booleans (got tokens_in={type(tokens_in).__name__}, tokens_out={type(tokens_out).__name__})."
        )
    if not isinstance(tokens_in, int) or not isinstance(tokens_out, int):
        raise TypeError(
            f"Token counts must be integers, got {type(tokens_in).__name__} and {type(tokens_out).__name__}."
        )
    if tokens_in < 0 or tokens_out < 0:
        raise ValueError(
            f"Token counts must be non-negative integers, got tokens_in={tokens_in}, tokens_out={tokens_out}."
        )

    cost = (
        Decimal(tokens_in) * model.input_cost_per_1k
        + Decimal(tokens_out) * model.output_cost_per_1k
    ) / Decimal("1000")
    return cost.quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP)


def estimate_input_cost(
    model: RegisteredModel,
    tokens_in: int,
) -> Decimal:
    """Calculate estimated minimum input cost in USD with 6-decimal precision.

    Raises:
        TypeError: If token count is a boolean or not an integer.
        ValueError: If token count is negative.
    """
    if isinstance(tokens_in, bool):
        raise TypeError(f"Token count must be an integer, not a boolean (got {type(tokens_in).__name__}).")
    if not isinstance(tokens_in, int):
        raise TypeError(f"Token count must be an integer, got {type(tokens_in).__name__}.")
    if tokens_in < 0:
        raise ValueError(f"Token count must be a non-negative integer, got tokens_in={tokens_in}.")

    cost = (Decimal(tokens_in) * model.input_cost_per_1k) / Decimal("1000")
    return cost.quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP)


def calculate_max_request_cost(
    model: RegisteredModel,
    estimated_tokens_in: int,
    max_tokens: Optional[int] = None,
) -> Decimal:
    """Calculate conservative maximum request cost in USD with 6-decimal precision.

    Matches PostgreSQL DECIMAL(12, 6) precision:
    - Assumes prompt consumes estimated_tokens_in.
    - Assumes completion consumes up to requested max_tokens (or model.max_output_tokens if not specified).
    - Caps completion tokens by the model's remaining context window.

    Raises:
        TypeError: If token counts are booleans or not integers.
        ValueError: If token counts are negative.
    """
    if isinstance(estimated_tokens_in, bool) or (max_tokens is not None and isinstance(max_tokens, bool)):
        raise TypeError("Token counts must be integers, not booleans.")
    if not isinstance(estimated_tokens_in, int):
        raise TypeError(f"estimated_tokens_in must be an integer, got {type(estimated_tokens_in).__name__}.")
    if estimated_tokens_in < 0:
        raise ValueError(f"estimated_tokens_in must be a non-negative integer, got {estimated_tokens_in}.")

    requested_output = max_tokens if max_tokens is not None else model.max_output_tokens
    if not isinstance(requested_output, int):
        raise TypeError(f"max_tokens must be an integer, got {type(requested_output).__name__}.")
    if requested_output < 0:
        raise ValueError(f"max_tokens must be a non-negative integer, got {requested_output}.")

    effective_max_output = min(requested_output, model.max_output_tokens)
    context_remaining = max(0, model.context_window - estimated_tokens_in)
    effective_max_output = min(effective_max_output, context_remaining)

    return calculate_inference_cost(model, estimated_tokens_in, effective_max_output)


# Parameterized SELECT statement matching mp_model_catalog schema contract
_SELECT_MODEL_CATALOG_QUERY = """
SELECT
    model_id,
    display_name,
    description,
    provider_model_id,
    context_window,
    max_output_tokens,
    input_cost_per_1k,
    output_cost_per_1k,
    is_active
FROM mp_model_catalog
WHERE is_active = true
"""


class ModelCatalogService:
    """Dynamic model catalog manager with TTL caching and fail-closed security.

    Invariants:
    - Production/Staging MUST NEVER silently fall back to the static development registry.
    - If a trusted catalog has never been loaded in production, the service fails closed (503).
    - If a cached catalog exceeds max_stale_seconds without a successful database refresh,
      the service fails closed (503) to prevent serving outdated pricing or deactivated models.
    - Only in development mode is fallback to _STATIC_REGISTRY permitted for offline testability.
    """

    def __init__(
        self,
        cache_ttl_seconds: float = 60.0,
        max_stale_seconds: float = 300.0,
    ) -> None:
        self.cache_ttl_seconds = cache_ttl_seconds
        self.max_stale_seconds = max_stale_seconds
        self._lock = anyio.Lock()
        self._cache: dict[str, RegisteredModel] = {}
        self._last_refresh_mono: float = 0.0
        self._has_ever_loaded: bool = False

    async def ensure_catalog_fresh(
        self,
        session_factory: Any = None,
        settings: Settings | None = None,
    ) -> None:
        """Refresh catalog from database if TTL expired, enforcing freshness boundaries."""
        now_mono = time.monotonic()
        is_dev = settings.is_development if settings else True

        if self._has_ever_loaded and (now_mono - self._last_refresh_mono) < self.cache_ttl_seconds:
            return

        async with self._lock:
            # Re-check under lock to avoid duplicate queries
            now_mono = time.monotonic()
            if self._has_ever_loaded and (now_mono - self._last_refresh_mono) < self.cache_ttl_seconds:
                return

            refreshed = False
            if session_factory is not None:
                try:
                    async with session_factory() as session:
                        validate_query_safety(_SELECT_MODEL_CATALOG_QUERY)
                        result = await session.execute(text(_SELECT_MODEL_CATALOG_QUERY))
                        rows = result.fetchall()
                        new_cache: dict[str, RegisteredModel] = {}
                        for row in rows:
                            row_dict = row._mapping if hasattr(row, "_mapping") else dict(row)
                            m_id = str(row_dict["model_id"])
                            new_cache[m_id] = RegisteredModel(
                                model_id=m_id,
                                description=str(row_dict["description"]),
                                display_name=str(row_dict.get("display_name", m_id)),
                                provider_model_id=row_dict.get("provider_model_id"),
                                context_window=int(row_dict.get("context_window", 8192)),
                                max_output_tokens=int(row_dict.get("max_output_tokens", 4096)),
                                input_cost_per_1k=Decimal(str(row_dict.get("input_cost_per_1k", "0.003000"))),
                                output_cost_per_1k=Decimal(str(row_dict.get("output_cost_per_1k", "0.015000"))),
                                is_active=bool(row_dict.get("is_active", True)),
                                requires_vertex_config=True,
                            )
                        if new_cache:
                            self._cache = new_cache
                            self._last_refresh_mono = time.monotonic()
                            self._has_ever_loaded = True
                            refreshed = True
                except Exception as exc:
                    logger.warning(
                        "Failed to refresh model catalog from mp_model_catalog: %s",
                        type(exc).__name__,
                    )

            if refreshed:
                return

            # Refresh failed or was unavailable:
            current_age = (time.monotonic() - self._last_refresh_mono) if self._has_ever_loaded else float("inf")

            if not self._has_ever_loaded:
                if not is_dev:
                    logger.error(
                        "Production catalog failure: trusted model catalog was never loaded from database. Failing closed."
                    )
                    raise HTTPException(
                        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                        detail="Model catalog is unavailable. Cannot verify model availability or pricing.",
                    )
                else:
                    logger.warning("Development mode: falling back to static model registry.")
                    self._cache = dict(_REGISTRY)
                    self._has_ever_loaded = True
                    self._last_refresh_mono = time.monotonic()
                    return

            if current_age <= self.max_stale_seconds:
                logger.warning(
                    "Model catalog refresh failed; serving from cached catalog (age=%.1fs, max_stale=%.1fs)",
                    current_age,
                    self.max_stale_seconds,
                )
                return
            else:
                logger.error(
                    "Model catalog staleness exceeded maximum allowable threshold (age=%.1fs > %.1fs). Failing closed.",
                    current_age,
                    self.max_stale_seconds,
                )
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="Model catalog is stale; unable to verify pricing or model status.",
                )

    async def get_model(
        self,
        model_id: str,
        session_factory: Any = None,
        settings: Settings | None = None,
    ) -> Optional[RegisteredModel]:
        """Resolve a registered model with freshness verification."""
        await self.ensure_catalog_fresh(session_factory=session_factory, settings=settings)
        return self._cache.get(model_id)

    async def list_models(
        self,
        session_factory: Any = None,
        settings: Settings | None = None,
    ) -> list[ModelInfo]:
        """List active registered models with freshness verification."""
        await self.ensure_catalog_fresh(session_factory=session_factory, settings=settings)
        result: list[ModelInfo] = []
        vertex_ok = settings.vertex_configured if settings else False
        for model in self._cache.values():
            if model.requires_vertex_config:
                stat = "available" if vertex_ok else "pending_configuration"
            else:
                stat = "available"
            result.append(
                ModelInfo(
                    id=model.model_id,
                    description=model.description,
                    status=stat,
                )
            )
        return result


def list_models(settings: Settings) -> list[ModelInfo]:
    """Return public model metadata from static registry (synchronous backward compatibility)."""
    result: list[ModelInfo] = []
    for model in _REGISTRY.values():
        if model.requires_vertex_config:
            stat = "available" if settings.vertex_configured else "pending_configuration"
        else:
            stat = "available"
        result.append(
            ModelInfo(
                id=model.model_id,
                description=model.description,
                status=stat,
            )
        )
    return result


def get_model(model_id: str) -> Optional[RegisteredModel]:
    """Look up a registered model by public ID from static registry (synchronous backward compatibility)."""
    return _REGISTRY.get(model_id)


def get_model_ids() -> list[str]:
    """Return all registered model IDs from static registry."""
    return sorted(_REGISTRY.keys())
