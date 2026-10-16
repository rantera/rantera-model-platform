"""FastAPI application for the Rantera Model Platform.

Provides:
- GET /health/live   — process liveness (public, no auth)
- GET /health/ready  — configuration readiness & database probe (public, no auth)
- GET /v1/models     — model registry (protected, requires authentication)

Security invariants:
- No permissive CORS enabled by default.
- No public debug endpoints.
- Tracebacks, credentials, project IDs, and endpoint IDs are never
  exposed to clients.
- Forwarded headers are not trusted from arbitrary clients.
- Request IDs are attached to every response.
- Database lifecycle: engine is initialized and disposed via lifespan.
- Database readiness check fails closed without leaking credentials or stack traces.
- Process liveness remains independent of database reachability.
"""

from __future__ import annotations

import base64
from decimal import Decimal
import hmac
import logging
import re
import uuid
import time
from contextlib import asynccontextmanager
from typing import Annotated, AsyncGenerator, Optional

from fastapi import Depends, FastAPI, HTTPException, Request, Response, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.auth import AuthenticatedPrincipal, require_valid_api_key
from app.logging import (
    configure_logging,
    log_event,
    reset_current_request_id,
    set_current_request_id,
)
from app.rate_limit import (
    BaseRateLimiter,
    InMemoryRateLimiter,
    require_rate_limited_principal,
)
from app.billing import (
    InFlightSpendTracker,
    require_billing_eligible_principal,
    track_in_flight_spend,
    verify_minimum_input_balance,
    verify_request_budget,
)
from app.config import Environment, Settings
from app.db import check_database_readiness, create_db_engine, get_session_factory
from app.models import (
    ModelCatalogService,
    calculate_inference_cost,
    calculate_max_request_cost,
    estimate_input_cost,
    get_model,
    list_models,
)
from app.schemas import (
    ChatChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    ErrorResponse,
    HealthLiveResponse,
    HealthReadyResponse,
    ModelListResponse,
    TokenUsage,
)
from app.telemetry import (
    BaseTelemetryPublisher,
    PubSubTelemetryPublisher,
    TelemetryEvent,
    TelemetryPersistenceError,
    ingest_telemetry_event,
    record_durable_inference_telemetry,
    record_inference_telemetry,
    validate_telemetry_event,
)
from app.vertex import BaseModelProvider, get_model_provider
from app.idempotency import (
    IdempotencyTracker,
    clear_idempotency_tracker,
    get_idempotency_tracker,
)

logger = logging.getLogger(__name__)


def _create_settings() -> Settings:
    """Create application settings.

    Separated for testability — tests can override this.
    """
    return Settings()


def create_app(
    settings: Settings | None = None,
    model_provider: BaseModelProvider | None = None,
    rate_limiter: BaseRateLimiter | None = None,
    telemetry_queue_publisher: BaseTelemetryPublisher | None = None,
    idempotency_tracker: IdempotencyTracker | None = None,
) -> FastAPI:
    """Application factory for the Rantera Model Platform.

    Args:
        settings: Optional pre-built settings (used in tests).
            If None, settings are loaded from the environment.
        model_provider: Optional pre-configured model provider (used in tests).
        rate_limiter: Optional rate limiter instance.
        telemetry_queue_publisher: Optional durable telemetry queue publisher.

    Returns:
        Configured FastAPI application instance.
    """
    if settings is None:
        settings = _create_settings()

    # Configure structured JSON audit logging
    configure_logging(settings)

    # Fail closed in non-development if required settings are missing.
    missing = settings.validate_production_readiness()
    if settings.is_production and missing:
        raise RuntimeError(
            f"Cannot start in {settings.rantera_env.value} mode: "
            f"missing required configuration: {', '.join(missing)}. "
            "Set the corresponding environment variables before deploying."
        )
    elif missing:
        log_event(
            logger,
            logging.WARNING,
            "system.warning",
            f"Running in {settings.rantera_env.value} mode with missing configuration: {', '.join(missing)}.",
            metadata={"environment": settings.rantera_env.value},
        )

    if not settings.rate_limit_enabled:
        if settings.is_production and settings.rate_limit_allow_production_bypass:
            log_event(
                logger,
                logging.WARNING,
                "system.warning",
                f"OPERATIONAL OVERRIDE: Rate limiting is disabled in {settings.rantera_env.value} mode via RATE_LIMIT_ALLOW_PRODUCTION_BYPASS.",
                metadata={"environment": settings.rantera_env.value},
            )
        else:
            log_event(
                logger,
                logging.INFO,
                "system.info",
                f"Rate limiting is disabled ({settings.rantera_env.value} mode).",
                metadata={"environment": settings.rantera_env.value},
            )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
        log_event(
            logger,
            logging.INFO,
            "service.startup",
            "Rantera Model Platform starting",
            metadata={
                "environment": settings.rantera_env.value,
                "vertex_configured": settings.vertex_configured,
                "database_configured": settings.database_configured,
            },
        )
        # Initialize database engine if database_url is provided
        if settings.database_configured:
            try:
                engine = create_db_engine(settings)
                app.state.db_engine = engine
                app.state.db_session_factory = get_session_factory(engine)
                log_event(
                    logger,
                    logging.INFO,
                    "database.pool_initialized",
                    "Database connection pool initialized",
                    metadata={
                        "pool_size": settings.db_pool_size,
                        "max_overflow": settings.db_max_overflow,
                    },
                )
            except Exception as exc:
                log_event(
                    logger,
                    logging.ERROR,
                    "system.error",
                    f"Failed to initialize database engine: {type(exc).__name__}",
                    metadata={"error_type": type(exc).__name__},
                )
                app.state.db_engine = None
                app.state.db_session_factory = None
        else:
            app.state.db_engine = None
            app.state.db_session_factory = None
            log_event(
                logger,
                logging.INFO,
                "system.info",
                "Database URL not configured; running without database connection",
            )

        # Attempt initial catalog load if database is configured
        if getattr(app.state, "db_session_factory", None) is not None:
            try:
                await app.state.model_catalog_service.ensure_catalog_fresh(
                    session_factory=app.state.db_session_factory,
                    settings=settings,
                )
            except Exception as exc:
                log_event(
                    logger,
                    logging.WARNING,
                    "catalog.refresh.failed",
                    f"Initial catalog refresh encountered error: {type(exc).__name__}",
                    metadata={"error_type": type(exc).__name__},
                )

        yield

        log_event(
            logger,
            logging.INFO,
            "service.shutdown",
            "Rantera Model Platform shutting down",
            metadata={"environment": settings.rantera_env.value},
        )
        if getattr(app.state, "db_engine", None) is not None:
            log_event(
                logger,
                logging.INFO,
                "system.info",
                "Disposing database connection pool",
            )
            try:
                await app.state.db_engine.dispose()
                log_event(
                    logger,
                    logging.INFO,
                    "database.pool_disposed",
                    "Database connection pool disposed",
                )
            except Exception as exc:
                log_event(
                    logger,
                    logging.WARNING,
                    "system.warning",
                    f"Error disposing database connection pool: {type(exc).__name__}",
                    metadata={"error_type": type(exc).__name__},
                )

        publisher = getattr(app.state, "telemetry_queue_publisher", None)
        if publisher is not None:
            try:
                await publisher.close()
            except Exception as pub_exc:
                log_event(
                    logger,
                    logging.WARNING,
                    "system.warning",
                    f"Error closing telemetry publisher: {type(pub_exc).__name__}",
                )

    MAX_REQUEST_BODY_BYTES = 1_048_576  # 1MB maximum allowable request payload

    app = FastAPI(
        title="Rantera Model Platform",
        description="Secure internal API for Rantera foundation models.",
        version="0.1.0",
        docs_url="/docs" if not settings.is_production else None,
        redoc_url=None,
        openapi_url="/openapi.json" if not settings.is_production else None,
        lifespan=lifespan,
    )

    # Store settings, model provider, and services on app state for access in route handlers.
    app.state.settings = settings
    app.state.model_provider = model_provider or get_model_provider(settings)
    app.state.rate_limiter = rate_limiter or InMemoryRateLimiter(
        rpm=settings.rate_limit_rpm,
        burst=settings.rate_limit_rpm,
    )
    app.state.in_flight_spend_tracker = InFlightSpendTracker()
    app.state.idempotency_tracker = idempotency_tracker or IdempotencyTracker()
    app.state.model_catalog_service = ModelCatalogService()
    if telemetry_queue_publisher is not None:
        app.state.telemetry_queue_publisher = telemetry_queue_publisher
    elif settings.telemetry_queue_enabled and settings.gcp_project_id:
        app.state.telemetry_queue_publisher = PubSubTelemetryPublisher(
            project_id=settings.gcp_project_id,
            topic=settings.pubsub_topic_telemetry or "mp-inference-telemetry",
            dead_letter_topic=settings.pubsub_dead_letter_topic,
        )
    else:
        app.state.telemetry_queue_publisher = None

    # ── Middleware: Request ID & Security Headers ─────────────

    _VALID_REQUEST_ID_REGEX = re.compile(r"^[a-zA-Z0-9_\-\.:]{1,64}$")

    @app.middleware("http")
    async def add_request_id(request: Request, call_next) -> Response:  # type: ignore[no-untyped-def]
        """Attach a unique request ID, rate limit headers, and security headers to every response.

        Preserves a valid inbound X-Request-ID (1-64 alphanumeric/hyphen/underscore/dot/colon chars);
        otherwise generates a cryptographically random UUID to prevent log/header injection.
        Enforces maximum request body payload limits and attaches nosniff, DENY, and HSTS headers.
        """
        raw_inbound = (
            request.headers.get("X-Request-ID")
            or request.headers.get("X-Request-Id")
            or request.headers.get("x-request-id")
        )
        if raw_inbound and _VALID_REQUEST_ID_REGEX.match(raw_inbound.strip()):
            request_id = raw_inbound.strip()
        else:
            request_id = f"req_{uuid.uuid4().hex}"

        request.state.request_id = request_id
        token = set_current_request_id(request_id)

        # Enforce maximum request body size limit to protect against unbounded payload DoS
        content_length_header = (
            request.headers.get("Content-Length")
            or request.headers.get("content-length")
        )
        if content_length_header:
            try:
                content_length = int(content_length_header)
                if content_length > MAX_REQUEST_BODY_BYTES:
                    log_event(
                        logger,
                        logging.WARNING,
                        "http.request.payload_too_large",
                        f"Request payload size {content_length} bytes exceeds limit of {MAX_REQUEST_BODY_BYTES} bytes",
                        metadata={"content_length": content_length, "max_allowed": MAX_REQUEST_BODY_BYTES},
                        request_id=request_id,
                    )
                    headers: dict[str, str] = {
                        "X-Request-ID": request_id,
                        "X-Content-Type-Options": "nosniff",
                        "X-Frame-Options": "DENY",
                    }
                    if settings.is_production:
                        headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
                    return JSONResponse(
                        status_code=getattr(status, "HTTP_413_CONTENT_TOO_LARGE", 413),
                        content=ErrorResponse(
                            error="Request Entity Too Large",
                            message=f"Request body exceeds maximum allowable limit of {MAX_REQUEST_BODY_BYTES} bytes.",
                            request_id=request_id,
                        ).model_dump(),
                        headers=headers,
                    )
            except ValueError:
                pass

        start_time = time.monotonic()
        try:
            response: Response = await call_next(request)
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Content-Type-Options"] = "nosniff"
            response.headers["X-Frame-Options"] = "DENY"
            if settings.is_production:
                response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"

            rate_limit_headers = getattr(request.state, "rate_limit_headers", None)
            if rate_limit_headers:
                for k, v in rate_limit_headers.items():
                    if k not in response.headers:
                        response.headers[k] = str(v)
            duration_ms = round((time.monotonic() - start_time) * 1000, 2)
            log_event(
                logger,
                logging.INFO,
                "http.request.completed",
                f"{request.method} {request.url.path} completed with {response.status_code}",
                metadata={
                    "method": request.method,
                    "path": request.url.path,
                    "status_code": response.status_code,
                    "duration_ms": duration_ms,
                },
            )
            return response
        finally:
            reset_current_request_id(token)

    # ── Exception Handlers ───────────────────────────────────

    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(
        request: Request, exc: StarletteHTTPException
    ) -> JSONResponse:
        """Return structured error responses without leaking internals."""
        request_id = getattr(request.state, "request_id", "")
        headers = dict(exc.headers) if getattr(exc, "headers", None) else {}
        return JSONResponse(
            status_code=exc.status_code,
            headers=headers if headers else None,
            content=ErrorResponse(
                error=_status_phrase(exc.status_code),
                message=str(exc.detail) if exc.detail else "An error occurred.",
                request_id=request_id,
            ).model_dump(),
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        """Return a safe 422 response without exposing internal details."""
        request_id = getattr(request.state, "request_id", "")
        # Provide field-level errors but not stack traces.
        safe_errors = []
        for error in exc.errors():
            safe_errors.append({
                "field": " → ".join(str(loc) for loc in error.get("loc", [])),
                "message": error.get("msg", "Validation error"),
                "type": error.get("type", "unknown"),
            })
        return JSONResponse(
            status_code=422,
            content=ErrorResponse(
                error="Validation Error",
                message=f"{len(safe_errors)} validation error(s) in the request.",
                request_id=request_id,
            ).model_dump()
            | {"details": safe_errors},
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(
        request: Request, exc: Exception
    ) -> JSONResponse:
        """Catch-all: log the real error, return a safe response."""
        request_id = getattr(request.state, "request_id", "")
        log_event(
            logger,
            logging.ERROR,
            "system.error",
            f"Unhandled exception during {request.method} {request.url.path}",
            metadata={
                "method": request.method,
                "path": request.url.path,
                "error_type": type(exc).__name__,
            },
            exc_info=True,
        )
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                error="Internal Server Error",
                message="An unexpected error occurred. Please try again later.",
                request_id=request_id,
            ).model_dump(),
        )

    # ── Health Endpoints (public, no auth) ───────────────────

    @app.get(
        "/health/live",
        response_model=HealthLiveResponse,
        tags=["health"],
        summary="Liveness probe",
    )
    async def health_live() -> HealthLiveResponse:
        """Process liveness check.

        Returns 200 if the process is running. Does not check
        upstream dependencies or database connectivity.
        """
        return HealthLiveResponse(status="alive")

    @app.get(
        "/health/ready",
        response_model=HealthReadyResponse,
        response_model_exclude_none=True,
        tags=["health"],
        summary="Readiness probe",
    )
    async def health_ready(response: Response) -> HealthReadyResponse:
        """Application readiness check.

        Reports whether server-side configuration is present and verifies
        database connectivity and model catalog availability.
        Returns HTTP 200 when ready, or HTTP 503 when degraded or unready.
        Does NOT verify that the Vertex AI endpoint is reachable.
        """
        configured = settings.vertex_configured
        missing = settings.validate_production_readiness()

        # Database readiness check
        db_engine = getattr(app.state, "db_engine", None)
        db_configured = settings.database_configured
        db_ready = False
        if db_configured and db_engine is not None:
            db_ready = await check_database_readiness(db_engine, timeout_seconds=15.0)

        # Model catalog availability check
        catalog_service: ModelCatalogService = getattr(
            app.state, "model_catalog_service", None
        )
        catalog_ready = False
        if catalog_service is not None:
            if catalog_service._has_ever_loaded and len(catalog_service._cache) > 0:
                age = time.monotonic() - catalog_service._last_refresh_mono
                catalog_ready = age <= catalog_service.max_stale_seconds
            elif getattr(app.state, "db_session_factory", None) is not None:
                try:
                    await catalog_service.ensure_catalog_fresh(
                        session_factory=app.state.db_session_factory,
                        settings=settings,
                    )
                    catalog_ready = catalog_service._has_ever_loaded and len(catalog_service._cache) > 0
                except Exception:
                    catalog_ready = False
            elif settings.is_development:
                try:
                    await catalog_service.ensure_catalog_fresh(
                        session_factory=None,
                        settings=settings,
                    )
                    catalog_ready = catalog_service._has_ever_loaded and len(catalog_service._cache) > 0
                except Exception:
                    catalog_ready = False

        # In production, require Vertex AI configuration, database configuration,
        # verified database reachability, and an available model catalog.
        if settings.is_production:
            is_ready = configured and db_configured and db_ready and catalog_ready
        else:
            is_ready = configured and (db_ready if db_configured else True) and catalog_ready

        if not is_ready:
            response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
            logger.warning(
                "Readiness probe: service not ready. Vertex configured: %s, DB configured: %s, DB ready: %s, Catalog ready: %s",
                configured,
                db_configured,
                db_ready,
                catalog_ready,
            )
        else:
            response.status_code = status.HTTP_200_OK
            logger.debug("Readiness probe: application configuration complete and ready.")

        # In non-development environments, do not expose internal configuration
        # details, variable names, or internal booleans in the public response.
        if settings.is_development:
            details_parts: list[str] = []
            if configured:
                details_parts.append("Vertex AI configuration present. Upstream reachability not verified.")
            else:
                details_parts.append(f"Missing configuration: {', '.join(missing)}.")

            if db_configured:
                details_parts.append("Database connected." if db_ready else "Database connectivity unavailable or timed out.")

            if not catalog_ready:
                details_parts.append("Model catalog unavailable.")

            return HealthReadyResponse(
                status="ready" if is_ready else "degraded",
                vertex_configured=configured,
                database_configured=db_configured,
                database_ready=db_ready if db_configured else None,
                details=" ".join(details_parts),
            )

        return HealthReadyResponse(
            status="ready" if is_ready else "degraded",
            vertex_configured=None,
            database_configured=None,
            database_ready=None,
            details=(
                "Application ready. Upstream reachability not verified."
                if is_ready
                else "Service not fully configured."
            ),
        )

    # ── Protected Endpoints ──────────────────────────────────

    @app.get(
        "/v1/models",
        response_model=ModelListResponse,
        tags=["models"],
        summary="List available models",
        dependencies=[Depends(require_rate_limited_principal)],
    )
    async def list_available_models() -> ModelListResponse:
        """Return the public model registry.

        Protected by API-key authentication. Model entries never
        contain Vertex endpoint IDs or infrastructure details.
        """
        catalog_service: ModelCatalogService = getattr(
            app.state, "model_catalog_service", None
        ) or ModelCatalogService()
        session_factory = getattr(app.state, "db_session_factory", None)
        models = await catalog_service.list_models(session_factory=session_factory, settings=settings)
        return ModelListResponse(models=models)

    @app.post(
        "/v1/chat/completions",
        response_model=ChatCompletionResponse,
        tags=["chat"],
        summary="Create a model completion",
    )
    async def create_chat_completion(
        request: Request,
        body: ChatCompletionRequest,
        principal: Annotated[AuthenticatedPrincipal, Depends(require_billing_eligible_principal)],
    ) -> ChatCompletionResponse:
        """Create model completion via chat interface.

        Security & Pre-Inference Invariants:
        - Authenticated via require_billing_eligible_principal (valid key, unexpired, unbanned).
        - Enforces positive wallet balance (> 0.0 USD). Rejects with HTTP 402 if depleted.
        - Evaluates estimated minimum prompt cost against wallet balance.
        - Concurrency dampening: in-memory InFlightSpendTracker prevents local intra-process bursts.
          (Notice: Single-process mitigation only; does not provide cross-instance protection).
        - Model lookup: Validates model against dynamic catalog service with freshness boundaries.
        - Telemetry: Records usage to mp_inference_logs with settlement_status='PENDING'.
        - Fault tolerance: If telemetry DB insert fails post-inference, the client response
          is preserved and an emergency error is logged (temporary documented non-durable policy).
        """
        request_id = getattr(request.state, "request_id", "") or f"req_{uuid.uuid4().hex[:16]}"

        # 0. Idempotency Check & Claim: Scoped strictly to authenticated principal.
        # Checks in-memory (in-progress, completed replay, failed) and cross-instance DB logs.
        # Idempotency checks precede Part 2's spend reservation so duplicates never reserve,
        # charge, or release funds.
        idempotency_tracker: IdempotencyTracker = getattr(
            app.state, "idempotency_tracker", None
        ) or get_idempotency_tracker()
        session_factory = getattr(app.state, "db_session_factory", None)

        cached_response = await idempotency_tracker.check_and_claim_request(
            request_id=request_id,
            principal=principal,
            session_factory=session_factory,
        )
        if cached_response is not None:
            return cached_response

        try:
            # 1. Reject streaming (currently unsupported)
            if body.stream:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Streaming is not currently supported. Set stream: false.",
                )

            # 2. Resolve model from dynamic catalog service (with freshness boundaries & production fail-closed)
            catalog_service: ModelCatalogService = getattr(
                app.state, "model_catalog_service", None
            ) or ModelCatalogService()
            model = await catalog_service.get_model(
                body.model, session_factory=session_factory, settings=settings
            )
            if not model:
                log_event(
                    logger,
                    logging.WARNING,
                    "inference.failed.bad_request",
                    f"Model '{body.model}' is not registered or not available.",
                    metadata={"model_id": body.model, "path": request.url.path},
                )
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Model '{body.model}' is not registered or not available.",
                )

            if not model.is_active:
                log_event(
                    logger,
                    logging.WARNING,
                    "inference.failed.bad_request",
                    f"Model '{body.model}' is currently inactive.",
                    metadata={"model_id": body.model, "path": request.url.path},
                )
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Model '{body.model}' is currently inactive.",
                )

            # 3. Estimate input token count for context window check & minimum input cost
            total_prompt_chars = sum(len(m.content) for m in body.messages)
            estimated_tokens_in = max(1, (total_prompt_chars + 3) // 4 + len(body.messages) * 4)

            if estimated_tokens_in > model.context_window:
                log_event(
                    logger,
                    logging.WARNING,
                    "inference.failed.bad_request",
                    f"Prompt length ({estimated_tokens_in} tokens) exceeds context window ({model.context_window})",
                    metadata={"model_id": model.model_id, "prompt_tokens": estimated_tokens_in, "limit": model.context_window},
                )
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Prompt length ({estimated_tokens_in} tokens) exceeds model context window of {model.context_window} tokens.",
                )

            # 4. Conservative maximum request cost calculation & budget / spend-limit verification
            max_request_cost = calculate_max_request_cost(
                model=model,
                estimated_tokens_in=estimated_tokens_in,
                max_tokens=body.max_tokens,
            )
            verify_request_budget(
                principal=principal,
                max_cost=max_request_cost,
                request=request,
            )
            in_flight_tracker: InFlightSpendTracker = getattr(
                app.state, "in_flight_spend_tracker", None
            ) or InFlightSpendTracker()

            async with track_in_flight_spend(
                tracker=in_flight_tracker,
                user_id=principal.user_id,
                balance=principal.wallet_balance,
                max_cost=max_request_cost,
                api_key_id=principal.api_key_id,
                spend_limit=principal.spend_limit,
                reservation_id=request_id,
            ) as reservation:
                # 5. Acquire provider
                provider: BaseModelProvider = getattr(
                    app.state, "model_provider", None
                ) or get_model_provider(settings)

                # 6. Execute prediction
                try:
                    prediction = await provider.predict(
                        model=model,
                        messages=body.messages,
                        temperature=body.temperature,
                        max_tokens=body.max_tokens,
                    )
                except HTTPException:
                    raise
                except Exception as exc:
                    log_event(
                        logger,
                        logging.ERROR,
                        "inference.failed.provider_error",
                        f"Model provider prediction error: {type(exc).__name__}",
                        metadata={
                            "model_id": model.model_id,
                            "error_type": type(exc).__name__,
                        },
                    )
                    raise HTTPException(
                        status_code=status.HTTP_502_BAD_GATEWAY,
                        detail="Upstream model prediction service failed.",
                    )

                # 7. Validate usage accounting integrity and calculate exact cost
                if (
                    isinstance(prediction.tokens_in, bool)
                    or isinstance(prediction.tokens_out, bool)
                    or not isinstance(prediction.tokens_in, int)
                    or not isinstance(prediction.tokens_out, int)
                    or prediction.tokens_in <= 0
                    or prediction.tokens_out < 0
                    or (bool(prediction.content.strip()) and prediction.tokens_out == 0)
                ):
                    log_event(
                        logger,
                        logging.CRITICAL,
                        "inference.failed.invalid_usage_accounting",
                        f"Invalid token usage metadata from model provider: tokens_in={prediction.tokens_in}, tokens_out={prediction.tokens_out}",
                        metadata={
                            "model_id": model.model_id,
                            "tokens_in": str(prediction.tokens_in),
                            "tokens_out": str(prediction.tokens_out),
                        },
                        request_id=request_id,
                    )
                    raise HTTPException(
                        status_code=status.HTTP_502_BAD_GATEWAY,
                        detail="Upstream model prediction returned invalid token usage metadata.",
                    )

                cost_usd = calculate_inference_cost(model, prediction.tokens_in, prediction.tokens_out)
                if (model.input_cost_per_1k > Decimal("0.0") or model.output_cost_per_1k > Decimal("0.0")) and cost_usd <= Decimal("0.0"):
                    log_event(
                        logger,
                        logging.CRITICAL,
                        "inference.failed.zero_cost_accounting",
                        f"Calculated zero cost for billable inference: tokens_in={prediction.tokens_in}, tokens_out={prediction.tokens_out}",
                        metadata={"model_id": model.model_id, "cost_usd": str(cost_usd)},
                        request_id=request_id,
                    )
                    raise HTTPException(
                        status_code=status.HTTP_502_BAD_GATEWAY,
                        detail="Upstream model prediction returned invalid zero-cost usage metadata.",
                    )

                await reservation.reconcile(actual_cost=cost_usd)

                # 8. Record durable telemetry in mp_inference_logs (with queue fallback and production fail-closed policy)
                queue_publisher = getattr(app.state, "telemetry_queue_publisher", None)
                fail_closed = settings.telemetry_fail_closed if settings.is_production else False

                try:
                    if session_factory is not None:
                        async with session_factory() as sess:
                            await record_durable_inference_telemetry(
                                session=sess,
                                request_id=request_id,
                                principal=principal,
                                model_id=model.model_id,
                                tokens_in=prediction.tokens_in,
                                tokens_out=prediction.tokens_out,
                                cost_usd=cost_usd,
                                latency_ms=prediction.latency_ms,
                                request_status="SUCCESS",
                                channel=principal.channel,
                                queue_publisher=queue_publisher,
                                fail_closed=fail_closed,
                            )
                    else:
                        await record_durable_inference_telemetry(
                            session=None,
                            request_id=request_id,
                            principal=principal,
                            model_id=model.model_id,
                            tokens_in=prediction.tokens_in,
                            tokens_out=prediction.tokens_out,
                            cost_usd=cost_usd,
                            latency_ms=prediction.latency_ms,
                            request_status="SUCCESS",
                            channel=principal.channel,
                            queue_publisher=queue_publisher,
                            fail_closed=fail_closed,
                        )
                except TelemetryPersistenceError as exc:
                    log_event(
                        logger,
                        logging.CRITICAL,
                        "inference.failed.telemetry_persistence",
                        f"Inference aborted post-execution: durable telemetry failed to commit ({exc})",
                        metadata={"request_id": request_id, "model_id": model.model_id},
                        request_id=request_id,
                    )
                    raise HTTPException(
                        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                        detail="Billing telemetry service unavailable. Request aborted to prevent unbilled inference.",
                    )

                log_event(
                    logger,
                    logging.INFO,
                    "inference.completed",
                    f"Inference completed for {model.model_id}",
                    metadata={
                        "model_id": model.model_id,
                        "prompt_tokens": prediction.tokens_in,
                        "completion_tokens": prediction.tokens_out,
                        "total_tokens": prediction.tokens_in + prediction.tokens_out,
                        "cost_usd": str(cost_usd),
                        "latency_ms": prediction.latency_ms,
                    },
                )

                # 9. Format response
                completion_id = f"chatcmpl_{uuid.uuid4().hex[:24]}"
                response = ChatCompletionResponse(
                    id=completion_id,
                    object="chat.completion",
                    created=int(time.time()),
                    model=model.model_id,
                    choices=[
                        ChatChoice(
                            index=0,
                            message=ChatMessage(
                                role="assistant",
                                content=prediction.content,
                            ),
                            finish_reason=prediction.finish_reason,
                        )
                    ],
                    usage=TokenUsage(
                        prompt_tokens=prediction.tokens_in,
                        completion_tokens=prediction.tokens_out,
                        total_tokens=prediction.tokens_in + prediction.tokens_out,
                        cost_usd=cost_usd,
                    ),
                )
                # Mark request as completed in idempotency tracker
                await idempotency_tracker.complete_request(
                    request_id=request_id,
                    principal=principal,
                    response=response,
                )
                return response

        except HTTPException as http_exc:
            if http_exc.status_code >= 500:
                await idempotency_tracker.fail_request(
                    request_id=request_id,
                    principal=principal,
                    error_detail=str(http_exc.detail),
                )
            else:
                await idempotency_tracker.clear_request(request_id)
            raise
        except Exception as exc:
            await idempotency_tracker.fail_request(
                request_id=request_id,
                principal=principal,
                error_detail=type(exc).__name__,
            )
            raise

    # ── Telemetry Push Replay Endpoint ──────────────────────────

    @app.post(
        "/v1/telemetry/pubsub/push",
        tags=["telemetry"],
        summary="Ingest queued telemetry events from Google Cloud Pub/Sub push subscription",
    )
    async def ingest_pubsub_push_telemetry(
        request: Request,
    ) -> JSONResponse:
        """Ingest queued telemetry messages from Pub/Sub push subscriptions into mp_inference_logs.

        Security:
        - If internal_service_key is configured, requires matching X-Internal-Service-Key header
          or token query parameter.
        - Malformed payloads are acknowledged with 200 (after logging error) to prevent infinite poison loops.
        - Database failures return HTTP 503 so Pub/Sub retries delivery with exponential backoff.
        - Ingested records have settlement_status='PENDING' and settlement_attempts=0 for pickup by the settlement cron.
        """
        # 1. Query string authentication is strictly forbidden to prevent secret leakage in URL access logs.
        if request.query_params.get("token"):
            log_event(
                logger,
                logging.WARNING,
                "telemetry.push_unauthorized",
                "Pub/Sub push webhook rejected: query string tokens are strictly forbidden",
            )
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Query string authentication is forbidden. Use Authorization: Bearer <token> or X-Internal-Service-Key.",
            )

        raw_auth = request.headers.get("Authorization") or request.headers.get("authorization") or ""
        bearer_token = raw_auth[7:].strip() if raw_auth.lower().startswith("bearer ") else None
        internal_key_header = request.headers.get("X-Internal-Service-Key")

        # Safely detect and reject conflicting authentication credentials
        if raw_auth.strip() and internal_key_header and internal_key_header.strip():
            log_event(
                logger,
                logging.WARNING,
                "telemetry.push_unauthorized",
                "Pub/Sub push webhook rejected: conflicting authentication credentials provided",
            )
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Conflicting authentication credentials provided.",
            )

        authenticated = False

        # Mode A: Google Pub/Sub OIDC Token Authentication (Bearer token)
        if raw_auth.strip():
            if not raw_auth.lower().startswith("bearer ") or not bearer_token:
                log_event(
                    logger,
                    logging.WARNING,
                    "telemetry.push_unauthorized",
                    "Pub/Sub push webhook rejected: invalid authorization scheme",
                )
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid authorization scheme. Provide a Bearer token or X-Internal-Service-Key.",
                )

            expected_aud = (settings.pubsub_push_audience or "").strip()
            expected_sa = (settings.pubsub_push_service_account_email or "").strip()

            if not expected_aud or not expected_sa:
                log_event(
                    logger,
                    logging.WARNING,
                    "telemetry.push_unconfigured",
                    "Pub/Sub push webhook rejected: OIDC audience or service account email not configured",
                )
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Pub/Sub push OIDC authentication not configured on server.",
                )

            try:
                from google.oauth2 import id_token
                from google.auth.transport import requests as google_requests

                claims = id_token.verify_oauth2_token(
                    bearer_token,
                    google_requests.Request(),
                    audience=expected_aud,
                )

                if not isinstance(claims, dict):
                    raise ValueError("OIDC token claims must be a dictionary")

                issuer = claims.get("iss", "")
                if issuer not in ("accounts.google.com", "https://accounts.google.com"):
                    log_event(
                        logger,
                        logging.WARNING,
                        "telemetry.push_unauthorized",
                        "Pub/Sub push OIDC rejected: untrusted token issuer",
                    )
                    raise HTTPException(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        detail="Invalid token issuer.",
                    )

                email = claims.get("email")
                if not email or str(email).strip().lower() != expected_sa.lower():
                    log_event(
                        logger,
                        logging.WARNING,
                        "telemetry.push_unauthorized",
                        "Pub/Sub push OIDC rejected: unauthorized service account identity",
                    )
                    raise HTTPException(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        detail="Unauthorized Pub/Sub push service account.",
                    )

                if claims.get("email_verified") is False:
                    log_event(
                        logger,
                        logging.WARNING,
                        "telemetry.push_unauthorized",
                        "Pub/Sub push OIDC rejected: unverified service account email",
                    )
                    raise HTTPException(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        detail="Unauthorized Pub/Sub push service account.",
                    )

                authenticated = True
            except HTTPException:
                raise
            except Exception as token_err:
                log_event(
                    logger,
                    logging.WARNING,
                    "telemetry.push_unauthorized",
                    f"Pub/Sub push OIDC verification failed: {type(token_err).__name__}",
                )
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid OIDC bearer token.",
                )

        # Mode B: Internal Service Key Header (X-Internal-Service-Key)
        elif internal_key_header:
            expected_key = (
                settings.internal_service_key.get_secret_value().strip()
                if settings.internal_service_key
                else ""
            )
            if not expected_key:
                log_event(
                    logger,
                    logging.WARNING,
                    "telemetry.push_unauthorized",
                    "Pub/Sub push webhook rejected: internal service key unconfigured on server",
                )
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Authentication credentials not configured.",
                )

            cleaned_header = internal_key_header.strip()
            if not cleaned_header or not hmac.compare_digest(cleaned_header, expected_key):
                log_event(
                    logger,
                    logging.WARNING,
                    "telemetry.push_unauthorized",
                    "Pub/Sub push webhook rejected: invalid internal service key",
                )
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid internal service credentials.",
                )
            authenticated = True

        if not authenticated:
            log_event(
                logger,
                logging.WARNING,
                "telemetry.push_unauthorized",
                "Pub/Sub push webhook rejected: missing authentication credentials",
            )
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Missing authentication credentials for Pub/Sub push webhook.",
            )

        try:
            body = await request.json()
        except Exception:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid JSON body.")

        message = body.get("message", {})
        b64_data = message.get("data")
        if not b64_data:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Missing message.data in Pub/Sub payload.")

        try:
            raw_bytes = base64.b64decode(b64_data)
            event = TelemetryEvent.from_json(raw_bytes.decode("utf-8"))
            validate_telemetry_event(event)
        except Exception as exc:
            log_event(
                logger,
                logging.WARNING,
                "telemetry.push_malformed",
                f"Malformed Pub/Sub push payload: {exc}",
            )
            # Return 200 to acknowledge and discard poisoned/unparseable messages from the queue
            return JSONResponse(status_code=200, content={"status": "dead_letter", "error": "malformed_payload"})

        session_factory = getattr(app.state, "db_session_factory", None)
        try:
            if session_factory is not None:
                async with session_factory() as session:
                    success = await ingest_telemetry_event(session, event)
            else:
                success = await ingest_telemetry_event(None, event)
        except ValueError as val_err:
            log_event(
                logger,
                logging.ERROR,
                "telemetry.push_conflict",
                f"Conflicting payload for existing request_id {event.request_id}: {val_err}",
            )
            return JSONResponse(status_code=200, content={"status": "dead_letter", "error": str(val_err)})

        if not success:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Database unavailable for telemetry ingestion. Retry later.",
            )

        return JSONResponse(status_code=200, content={"status": "ingested", "request_id": event.request_id})

    return app


def _status_phrase(status_code: int) -> str:
    """Map HTTP status codes to human-readable phrases."""
    phrases = {
        400: "Bad Request",
        401: "Unauthorized",
        402: "Payment Required",
        403: "Forbidden",
        404: "Not Found",
        405: "Method Not Allowed",
        422: "Validation Error",
        429: "Too Many Requests",
        500: "Internal Server Error",
        502: "Bad Gateway",
        503: "Service Unavailable",
    }
    return phrases.get(status_code, "Error")


# ── Module-level app instance for uvicorn ────────────────────
# Usage: uvicorn app.main:app --reload
app = create_app()
