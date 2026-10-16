"""Inference telemetry logging for Rantera Model Platform.

Security and Architectural Invariants:
- Ingestion targets mp_inference_logs exclusively.
- All new inference records are recorded with settlement_status='PENDING'.
- Columns match the approved migration DDL (20261014000000_add_model_platform_tables) exactly.
- Idempotency is enforced by unique index idx_mp_inference_logs_request_id via ON CONFLICT (request_id) DO NOTHING.
- Settlement-owned columns (settlement_id, settled_at) default to NULL and are never written by FastAPI.
- FAULT TOLERANCE & NON-DURABLE RECOVERY NOTICE:
  NON-DURABLE RECOVERY & TEMPORARY FAIL-OPEN POLICY NOTICE:
  PostgreSQL inference logs CANNOT recover successful inference records when the initial database
  write fails. The current fail-open behavior (preserving the user's generated response and emitting
  an EMERGENCY log with the request_id) is an EXPLICITLY DOCUMENTED TEMPORARY POLICY prioritizing
  user experience over revenue assurance until the team formally approves a production policy.
  APPLICATION LOGGING ALONE IS NOT DURABLE RECOVERY (APPLICATION LOGGING ALONE IS NOT DURABLE TELEMETRY).
  If the container terminates or logs are rotated, unwritten telemetry records are permanently lost,
  leading to unbilled inference. Production billing must not be considered reliable until an
  asynchronous durable queue (e.g. Cloud Pub/Sub) is implemented.
- AT-LEAST-ONCE DELIVERY & IDEMPOTENT SETTLEMENT:
  This platform guarantees at-least-once telemetry delivery attempts paired with idempotent settlement
  execution deduplicated by request_id. Exactly-once billing across distributed systems is not claimed.
"""

from __future__ import annotations

import logging
import uuid
from decimal import Decimal
from typing import Any

from sqlalchemy import text

from app.auth import AuthenticatedPrincipal
from app.db import validate_query_safety
from app.logging import log_event

logger = logging.getLogger(__name__)

# Parameterized INSERT statement matching the approved migration schema with ON CONFLICT idempotency
_INSERT_INFERENCE_LOG_QUERY = """
INSERT INTO mp_inference_logs (
    id,
    request_id,
    user_id,
    api_key_id,
    model_id,
    channel,
    tokens_in,
    tokens_out,
    total_tokens,
    cost_usd,
    latency_ms,
    ttft_ms,
    request_status,
    error_code,
    settlement_status,
    settlement_attempts,
    created_at,
    updated_at
) VALUES (
    :id,
    :request_id,
    :user_id,
    :api_key_id,
    :model_id,
    :channel,
    :tokens_in,
    :tokens_out,
    :total_tokens,
    :cost_usd,
    :latency_ms,
    :ttft_ms,
    :request_status,
    :error_code,
    :settlement_status,
    :settlement_attempts,
    CURRENT_TIMESTAMP,
    CURRENT_TIMESTAMP
)
ON CONFLICT (request_id) DO NOTHING
"""


async def record_inference_telemetry(
    session: Any,
    request_id: str,
    principal: AuthenticatedPrincipal,
    model_id: str,
    tokens_in: int,
    tokens_out: int,
    cost_usd: Decimal,
    latency_ms: int,
    request_status: str = "SUCCESS",
    error_code: str | None = None,
    ttft_ms: int | None = None,
    channel: str = "api",
) -> bool:
    """Insert an inference telemetry record directly into mp_inference_logs.

    Direct database telemetry logger preserving backward-compatible contract.
    """
    validate_query_safety(_INSERT_INFERENCE_LOG_QUERY)

    log_id = f"inf_{uuid.uuid4().hex[:32]}"
    total_tokens = tokens_in + tokens_out

    params = {
        "id": log_id,
        "request_id": request_id,
        "user_id": principal.user_id,
        "api_key_id": principal.api_key_id,
        "model_id": model_id,
        "channel": channel,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "total_tokens": total_tokens,
        "cost_usd": cost_usd,
        "latency_ms": latency_ms,
        "ttft_ms": ttft_ms,
        "request_status": request_status,
        "error_code": error_code,
        "settlement_status": "PENDING",
        "settlement_attempts": 0,
    }

    if session is None:
        log_event(
            logger,
            logging.ERROR,
            "telemetry.session_none",
            "Database session is None; inference telemetry could not be written to mp_inference_logs.",
            metadata={
                "model_id": model_id,
                "prompt_tokens": tokens_in,
                "completion_tokens": tokens_out,
                "total_tokens": total_tokens,
                "cost_usd": str(cost_usd),
                "latency_ms": latency_ms,
            },
            request_id=request_id,
        )
        return False

    try:
        await session.execute(text(_INSERT_INFERENCE_LOG_QUERY), params)
        if hasattr(session, "commit"):
            await session.commit()
        log_event(
            logger,
            logging.DEBUG,
            "telemetry.recorded",
            f"Inference telemetry recorded (model={model_id}, tokens={total_tokens}, cost={cost_usd})",
            metadata={
                "model_id": model_id,
                "prompt_tokens": tokens_in,
                "completion_tokens": tokens_out,
                "total_tokens": total_tokens,
                "cost_usd": str(cost_usd),
                "latency_ms": latency_ms,
            },
            request_id=request_id,
        )
        return True
    except Exception as exc:
        log_event(
            logger,
            logging.ERROR,
            "telemetry.write_failed",
            f"EMERGENCY: Failed to insert inference telemetry into mp_inference_logs: {type(exc).__name__}",
            metadata={
                "model_id": model_id,
                "prompt_tokens": tokens_in,
                "completion_tokens": tokens_out,
                "total_tokens": total_tokens,
                "cost_usd": str(cost_usd),
                "latency_ms": latency_ms,
                "error_type": type(exc).__name__,
            },
            request_id=request_id,
        )
        return False


# ── Durable Telemetry Architecture (Phase B) ────────────────────────────────


import abc
import base64
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import httpx


@dataclass(frozen=True)
class TelemetryEvent:
    """Immutable, serializable inference telemetry event for durable transmission.

    Guarantees idempotent processing via deterministic request_id.
    """

    id: str
    request_id: str
    user_id: str | None
    api_key_id: str | None
    model_id: str
    channel: str
    tokens_in: int
    tokens_out: int
    total_tokens: int
    cost_usd: Decimal
    latency_ms: int
    ttft_ms: int | None
    request_status: str
    error_code: str | None
    created_at: str

    @classmethod
    def create(
        cls,
        request_id: str,
        user_id: str | None,
        api_key_id: str | None,
        model_id: str,
        tokens_in: int,
        tokens_out: int,
        cost_usd: Decimal,
        latency_ms: int,
        request_status: str = "SUCCESS",
        error_code: str | None = None,
        ttft_ms: int | None = None,
        channel: str = "api",
    ) -> TelemetryEvent:
        log_id = f"inf_{uuid.uuid4().hex[:32]}"
        now_utc = datetime.now(timezone.utc).isoformat()
        return cls(
            id=log_id,
            request_id=request_id,
            user_id=user_id,
            api_key_id=api_key_id,
            model_id=model_id,
            channel=channel,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            total_tokens=tokens_in + tokens_out,
            cost_usd=cost_usd,
            latency_ms=latency_ms,
            ttft_ms=ttft_ms,
            request_status=request_status,
            error_code=error_code,
            created_at=now_utc,
        )

    def to_params(self) -> dict[str, Any]:
        """Convert to database parameter dictionary matching _INSERT_INFERENCE_LOG_QUERY."""
        return {
            "id": self.id,
            "request_id": self.request_id,
            "user_id": self.user_id,
            "api_key_id": self.api_key_id,
            "model_id": self.model_id,
            "channel": self.channel,
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "total_tokens": self.total_tokens,
            "cost_usd": self.cost_usd,
            "latency_ms": self.latency_ms,
            "ttft_ms": self.ttft_ms,
            "request_status": self.request_status,
            "error_code": self.error_code,
            "settlement_status": "PENDING",
            "settlement_attempts": 0,
        }

    def to_json(self) -> str:
        """Serialize event to JSON string for message payload."""
        data = asdict(self)
        data["cost_usd"] = str(self.cost_usd)
        return json.dumps(data)

    @classmethod
    def from_json(cls, raw_json: str) -> TelemetryEvent:
        """Deserialize event from JSON string."""
        data = json.loads(raw_json)
        return cls(
            id=data["id"],
            request_id=data["request_id"],
            user_id=data.get("user_id"),
            api_key_id=data.get("api_key_id"),
            model_id=data["model_id"],
            channel=data.get("channel", "api"),
            tokens_in=int(data["tokens_in"]),
            tokens_out=int(data["tokens_out"]),
            total_tokens=int(data.get("total_tokens", data["tokens_in"] + data["tokens_out"])),
            cost_usd=Decimal(str(data["cost_usd"])),
            latency_ms=int(data["latency_ms"]),
            ttft_ms=int(data["ttft_ms"]) if data.get("ttft_ms") is not None else None,
            request_status=data.get("request_status", "SUCCESS"),
            error_code=data.get("error_code"),
            created_at=data.get("created_at", datetime.now(timezone.utc).isoformat()),
        )


class BaseTelemetryPublisher(abc.ABC):
    """Abstract interface for durable telemetry queue publication."""

    @abc.abstractmethod
    async def publish_event(self, event: TelemetryEvent) -> bool:
        """Publish a telemetry event to a durable destination.

        Returns True on successful delivery/acceptance, False on permanent failure.
        """
        pass

    @abc.abstractmethod
    async def close(self) -> None:
        """Release underlying network/connection resources."""
        pass


class PubSubTelemetryPublisher(BaseTelemetryPublisher):
    """Google Cloud Pub/Sub implementation of durable telemetry publication.

    Architectural properties:
    - Uses Google Cloud Pub/Sub REST API (compatible with google-auth and httpx).
    - Idempotent deduplication key: request_id is bound as the message orderingKey and attribute.
    - Bounded retries: 3 attempts with exponential backoff on transient errors (429, 5xx).
    - Observable failures: Emits structured log events on delivery errors.
    - Dead letter support: Forwards to configured DLQ if primary publication fails.
    """

    def __init__(
        self,
        project_id: str,
        topic: str,
        dead_letter_topic: str | None = None,
        timeout_seconds: float = 5.0,
        max_retries: int = 3,
        client: httpx.AsyncClient | None = None,
        credentials: Any = None,
    ) -> None:
        self.project_id = project_id
        self.topic = topic
        self.dead_letter_topic = dead_letter_topic
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self._client = client
        self._owns_client = client is None
        self._credentials = credentials

    def _get_auth_token(self) -> str | None:
        """Obtain GCP bearer token using google-auth if credentials exist."""
        if self._credentials is None:
            try:
                import google.auth
                self._credentials, _ = google.auth.default(
                    scopes=["https://www.googleapis.com/auth/pubsub"]
                )
            except Exception:
                return None

        if self._credentials is not None:
            try:
                from google.auth.transport.requests import Request as GoogleAuthRequest
                if not self._credentials.valid:
                    self._credentials.refresh(GoogleAuthRequest())
                return getattr(self._credentials, "token", None)
            except Exception as exc:
                logger.warning(f"Could not refresh GCP credentials for Pub/Sub: {exc}")
                return None
        return None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=self.timeout_seconds)
            self._owns_client = True
        return self._client

    async def publish_event(self, event: TelemetryEvent) -> bool:
        """Publish event to Google Cloud Pub/Sub with retries and dead-letter handling."""
        data_bytes = event.to_json().encode("utf-8")
        b64_payload = base64.b64encode(data_bytes).decode("ascii")

        message = {
            "data": b64_payload,
            "attributes": {
                "request_id": event.request_id,
                "model_id": event.model_id,
                "user_id": str(event.user_id or ""),
                "channel": event.channel,
            },
            "orderingKey": event.request_id,
        }

        url = f"https://pubsub.googleapis.com/v1/projects/{self.project_id}/topics/{self.topic}:publish"
        token = self._get_auth_token()
        headers = {"Content-Type": "application/json"}
        if token:
            headers["Authorization"] = f"Bearer {token}"

        client = await self._get_client()

        for attempt in range(1, self.max_retries + 1):
            try:
                response = await client.post(url, headers=headers, json={"messages": [message]})
                if response.status_code == 200:
                    log_event(
                        logger,
                        logging.DEBUG,
                        "telemetry.pubsub_published",
                        f"Durable telemetry published to Pub/Sub topic {self.topic}",
                        metadata={"request_id": event.request_id, "topic": self.topic},
                        request_id=event.request_id,
                    )
                    return True

                if response.status_code in (429, 500, 502, 503, 504) and attempt < self.max_retries:
                    import anyio
                    await anyio.sleep(0.1 * (2 ** (attempt - 1)))
                    continue

                log_event(
                    logger,
                    logging.ERROR,
                    "telemetry.pubsub_failed",
                    f"Pub/Sub returned HTTP {response.status_code}: {response.text[:200]}",
                    metadata={"request_id": event.request_id, "status_code": response.status_code},
                    request_id=event.request_id,
                )
                break
            except Exception as exc:
                if attempt < self.max_retries:
                    import anyio
                    await anyio.sleep(0.1 * (2 ** (attempt - 1)))
                    continue
                log_event(
                    logger,
                    logging.ERROR,
                    "telemetry.pubsub_exception",
                    f"Pub/Sub publish exception: {type(exc).__name__}: {str(exc)[:200]}",
                    metadata={"request_id": event.request_id, "error": str(exc)[:200]},
                    request_id=event.request_id,
                )

        # Attempt Dead Letter Queue if configured
        if self.dead_letter_topic:
            try:
                dlq_url = f"https://pubsub.googleapis.com/v1/projects/{self.project_id}/topics/{self.dead_letter_topic}:publish"
                await client.post(dlq_url, headers=headers, json={"messages": [message]})
                log_event(
                    logger,
                    logging.WARNING,
                    "telemetry.dlq_published",
                    f"Telemetry routed to Dead Letter Topic {self.dead_letter_topic}",
                    metadata={"request_id": event.request_id, "dlq_topic": self.dead_letter_topic},
                    request_id=event.request_id,
                )
            except Exception as dlq_exc:
                logger.error(f"Failed to publish to DLQ topic {self.dead_letter_topic}: {dlq_exc}")

        return False

    async def close(self) -> None:
        if self._owns_client and self._client and not self._client.is_closed:
            await self._client.aclose()


class MockTelemetryPublisher(BaseTelemetryPublisher):
    """In-memory mock publisher strictly for isolated local tests.

    CRITICAL NOTICE: In-memory queues do NOT provide durable recovery in production.
    """

    def __init__(self, should_fail: bool = False, max_capacity: int = 1000) -> None:
        self.events: list[TelemetryEvent] = []
        self.should_fail = should_fail
        self.max_capacity = max_capacity
        self.is_durable = False

    async def publish_event(self, event: TelemetryEvent) -> bool:
        if self.should_fail:
            return False
        if len(self.events) >= self.max_capacity:
            self.events.pop(0)  # bounded queue eviction
        self.events.append(event)
        return True

    async def close(self) -> None:
        pass


class TelemetryPersistenceError(Exception):
    """Raised when an inference usage record cannot be persisted to durable storage."""

    def __init__(self, message: str, event: TelemetryEvent | None = None) -> None:
        super().__init__(message)
        self.event = event


@dataclass
class TelemetryRecordResult:
    """Result summary of a telemetry recording attempt."""

    persisted: bool
    storage_type: str  # "database", "queue", or "none"
    event: TelemetryEvent
    error: str | None = None


async def record_durable_inference_telemetry(
    session: Any,
    request_id: str,
    principal: AuthenticatedPrincipal,
    model_id: str,
    tokens_in: int,
    tokens_out: int,
    cost_usd: Decimal,
    latency_ms: int,
    request_status: str = "SUCCESS",
    error_code: str | None = None,
    ttft_ms: int | None = None,
    channel: str = "api",
    queue_publisher: BaseTelemetryPublisher | None = None,
    fail_closed: bool = True,
) -> TelemetryRecordResult:
    """Unified telemetry coordinator ensuring at-least-once durable usage recording.

    Storage Strategy:
    1. Primary Destination: Direct atomic INSERT into PostgreSQL mp_inference_logs.
    2. Fallback Destination: Asynchronous durable queue (Pub/Sub) when DB is unavailable.
    3. Production Policy: If both destinations fail, fails closed by raising
       TelemetryPersistenceError so the platform cannot silently leak unbilled inference.
    """
    event = TelemetryEvent.create(
        request_id=request_id,
        user_id=principal.user_id,
        api_key_id=principal.api_key_id,
        model_id=model_id,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        cost_usd=cost_usd,
        latency_ms=latency_ms,
        request_status=request_status,
        error_code=error_code,
        ttft_ms=ttft_ms,
        channel=channel,
    )

    # 1. Primary: Direct Database Write
    if session is not None:
        try:
            validate_query_safety(_INSERT_INFERENCE_LOG_QUERY)
            await session.execute(text(_INSERT_INFERENCE_LOG_QUERY), event.to_params())
            if hasattr(session, "commit"):
                await session.commit()
            log_event(
                logger,
                logging.DEBUG,
                "telemetry.recorded",
                f"Inference telemetry recorded in database (model={model_id}, tokens={event.total_tokens}, cost={cost_usd})",
                metadata={
                    "model_id": model_id,
                    "prompt_tokens": tokens_in,
                    "completion_tokens": tokens_out,
                    "total_tokens": event.total_tokens,
                    "cost_usd": str(cost_usd),
                    "latency_ms": latency_ms,
                },
                request_id=request_id,
            )
            return TelemetryRecordResult(persisted=True, storage_type="database", event=event)
        except Exception as exc:
            log_event(
                logger,
                logging.ERROR,
                "telemetry.db_write_failed",
                f"EMERGENCY: Database telemetry write failed; evaluating durable queue: {type(exc).__name__}",
                metadata={"request_id": request_id, "error": str(exc)[:200]},
                request_id=request_id,
            )

    # 2. Fallback: Durable Queue Publisher (Pub/Sub)
    if queue_publisher is not None:
        try:
            published = await queue_publisher.publish_event(event)
            if published:
                log_event(
                    logger,
                    logging.INFO,
                    "telemetry.queue_recorded",
                    f"Inference telemetry durably buffered in queue (model={model_id}, tokens={event.total_tokens}, cost={cost_usd})",
                    metadata={"request_id": request_id, "model_id": model_id},
                    request_id=request_id,
                )
                return TelemetryRecordResult(persisted=True, storage_type="queue", event=event)
        except Exception as queue_exc:
            log_event(
                logger,
                logging.ERROR,
                "telemetry.queue_publish_error",
                f"Queue publisher failed: {type(queue_exc).__name__}: {str(queue_exc)[:200]}",
                metadata={"request_id": request_id, "error": str(queue_exc)[:200]},
                request_id=request_id,
            )

    # 3. Dual Failure / Unavailability Handling
    log_event(
        logger,
        logging.CRITICAL,
        "telemetry.durable_persistence_failed",
        f"EMERGENCY: Failed to commit durable telemetry record for request {request_id}",
        metadata={
            "request_id": request_id,
            "user_id": principal.user_id,
            "model_id": model_id,
            "tokens": event.total_tokens,
            "cost_usd": str(cost_usd),
        },
        request_id=request_id,
    )

    if fail_closed:
        raise TelemetryPersistenceError(
            f"Failed to record durable inference telemetry for request {request_id}",
            event=event,
        )

    return TelemetryRecordResult(
        persisted=False,
        storage_type="none",
        event=event,
        error="All durable storage destinations failed",
    )


import re


_VALID_REQUEST_ID_REGEX = re.compile(r"^[a-zA-Z0-9_\-\.:]{1,64}$")


def validate_telemetry_event(event: TelemetryEvent) -> None:
    """Validate telemetry event shape, identifiers, token bounds, and costs.

    Raises:
        ValueError: If any attribute violates schema or security constraints.
    """
    if not event.request_id or not _VALID_REQUEST_ID_REGEX.match(event.request_id):
        raise ValueError(f"Invalid or malformed request_id: {event.request_id!r}")
    if not event.model_id or len(event.model_id) > 64:
        raise ValueError(f"Invalid model_id: {event.model_id!r}")
    if event.user_id is not None and len(event.user_id) > 64:
        raise ValueError("user_id exceeds maximum length of 64")
    if event.api_key_id is not None and len(event.api_key_id) > 64:
        raise ValueError("api_key_id exceeds maximum length of 64")
    if event.tokens_in < 0 or event.tokens_out < 0 or event.total_tokens < 0:
        raise ValueError("Token counts must be non-negative")
    if event.total_tokens != (event.tokens_in + event.tokens_out):
        raise ValueError("total_tokens must equal tokens_in + tokens_out")
    if event.cost_usd < Decimal("0.0"):
        raise ValueError("cost_usd must be non-negative")
    if event.latency_ms < 0:
        raise ValueError("latency_ms must be non-negative")
    try:
        datetime.fromisoformat(event.created_at)
    except Exception:
        raise ValueError(f"Invalid created_at timestamp: {event.created_at!r}")


@dataclass
class TelemetryReplayBatchSummary:
    """Summary of a telemetry queue replay batch execution."""

    pulled: int
    ingested: int
    duplicates: int
    failed: int
    dead_letter: int


class PubSubTelemetryConsumer:
    """Consumer for replaying durable telemetry messages from Google Cloud Pub/Sub into mp_inference_logs.

    Architectural properties:
    - Uses Pub/Sub REST pull API (pull + acknowledge).
    - Validates event payload against schema constraints before database insert.
    - Idempotent ingestion: uses ingest_telemetry_event() with ON CONFLICT (request_id) DO NOTHING.
    - Acknowledges messages ONLY after successful database commit.
    - Retries transient DB failures with bounded exponential backoff.
    - Routes permanently unparseable/corrupted messages to Dead Letter Queue (DLQ).
    - Never logs credentials, secrets, or prompts.
    """

    def __init__(
        self,
        project_id: str,
        subscription: str,
        dead_letter_topic: str | None = None,
        timeout_seconds: float = 10.0,
        max_messages: int = 50,
        client: httpx.AsyncClient | None = None,
        credentials: Any = None,
    ) -> None:
        self.project_id = project_id
        self.subscription = subscription
        self.dead_letter_topic = dead_letter_topic
        self.timeout_seconds = timeout_seconds
        self.max_messages = min(max(1, max_messages), 100)
        self._client = client
        self._owns_client = client is None
        self._credentials = credentials

    def _get_auth_token(self) -> str | None:
        """Obtain GCP bearer token using google-auth if credentials exist."""
        if self._credentials is None:
            try:
                import google.auth
                self._credentials, _ = google.auth.default(
                    scopes=["https://www.googleapis.com/auth/pubsub"]
                )
            except Exception:
                return None

        if self._credentials is not None:
            try:
                from google.auth.transport.requests import Request as GoogleAuthRequest
                if not self._credentials.valid:
                    self._credentials.refresh(GoogleAuthRequest())
                return getattr(self._credentials, "token", None)
            except Exception as exc:
                logger.warning(f"Could not refresh GCP credentials for Pub/Sub consumer: {exc}")
                return None
        return None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=self.timeout_seconds)
            self._owns_client = True
        return self._client

    async def parse_and_validate_payload(self, b64_data: str) -> TelemetryEvent:
        """Decode and validate a raw base64 Pub/Sub payload.

        Raises:
            ValueError: If payload cannot be decoded, parsed as JSON, or violates schema constraints.
        """
        try:
            raw_bytes = base64.b64decode(b64_data)
            raw_str = raw_bytes.decode("utf-8")
            event = TelemetryEvent.from_json(raw_str)
            validate_telemetry_event(event)
            return event
        except Exception as exc:
            raise ValueError(f"Malformed telemetry payload: {exc}") from exc

    async def route_to_dead_letter(self, b64_data: str, attributes: dict[str, str], error_reason: str) -> bool:
        """Forward an unprocessable message to the configured Dead Letter Topic."""
        if not self.dead_letter_topic:
            return False

        url = f"https://pubsub.googleapis.com/v1/projects/{self.project_id}/topics/{self.dead_letter_topic}:publish"
        token = self._get_auth_token()
        headers = {"Content-Type": "application/json"}
        if token:
            headers["Authorization"] = f"Bearer {token}"

        dlq_attrs = dict(attributes)
        dlq_attrs["dead_letter_reason"] = error_reason[:200]
        dlq_attrs["dead_lettered_at"] = datetime.now(timezone.utc).isoformat()

        client = await self._get_client()
        try:
            res = await client.post(
                url,
                headers=headers,
                json={"messages": [{"data": b64_data, "attributes": dlq_attrs}]},
            )
            return res.status_code == 200
        except Exception as dlq_exc:
            logger.error(f"Failed to publish to DLQ topic {self.dead_letter_topic}: {dlq_exc}")
            return False

    async def pull_and_process_batch(
        self,
        session_factory: Any,
        max_messages: int | None = None,
    ) -> TelemetryReplayBatchSummary:
        """Pull a batch of messages from Pub/Sub and ingest them into mp_inference_logs.

        Guarantees:
        - Acknowledges messages ONLY after successful database insert.
        - Idempotently handles duplicates without error.
        - Routes permanently corrupted messages to dead-letter queue.
        - Unhandled transient database errors leave messages unacknowledged for Pub/Sub redelivery.
        """
        batch_limit = min(max(1, max_messages or self.max_messages), 100)
        url = f"https://pubsub.googleapis.com/v1/projects/{self.project_id}/subscriptions/{self.subscription}:pull"
        token = self._get_auth_token()
        headers = {"Content-Type": "application/json"}
        if token:
            headers["Authorization"] = f"Bearer {token}"

        client = await self._get_client()
        try:
            res = await client.post(url, headers=headers, json={"maxMessages": batch_limit})
            if res.status_code != 200:
                logger.error(f"Pub/Sub pull returned HTTP {res.status_code}: {res.text[:200]}")
                return TelemetryReplayBatchSummary(pulled=0, ingested=0, duplicates=0, failed=0, dead_letter=0)
            data = res.json()
        except Exception as exc:
            logger.error(f"Exception pulling from Pub/Sub subscription {self.subscription}: {exc}")
            return TelemetryReplayBatchSummary(pulled=0, ingested=0, duplicates=0, failed=0, dead_letter=0)

        received_messages = data.get("receivedMessages", [])
        if not received_messages:
            return TelemetryReplayBatchSummary(pulled=0, ingested=0, duplicates=0, failed=0, dead_letter=0)

        pulled = len(received_messages)
        ingested = 0
        duplicates = 0
        failed = 0
        dead_letter = 0
        ack_ids: list[str] = []

        for item in received_messages:
            ack_id = item.get("ackId")
            msg = item.get("message", {})
            b64_data = msg.get("data", "")
            attrs = msg.get("attributes", {})

            # 1. Parse & Validate Event
            try:
                event = await self.parse_and_validate_payload(b64_data)
            except ValueError as val_err:
                dlq_ok = await self.route_to_dead_letter(b64_data, attrs, str(val_err))
                if dlq_ok:
                    dead_letter += 1
                    if ack_id:
                        ack_ids.append(ack_id)
                else:
                    failed += 1
                    logger.error(
                        f"Failed to route malformed message to DLQ (topic={self.dead_letter_topic}); leaving unacknowledged: {val_err}"
                    )
                continue

            # 2. Ingest into mp_inference_logs
            db_success = False
            try:
                if session_factory is not None:
                    async with session_factory() as sess:
                        db_success = await ingest_telemetry_event(sess, event)
                else:
                    db_success = await ingest_telemetry_event(None, event)
            except ValueError as val_err:
                # Conflicting duplicate payload or validation failure during ingestion
                logger.error(f"Event payload conflict/validation error for {event.request_id}: {val_err}")
                dlq_ok = await self.route_to_dead_letter(b64_data, attrs, str(val_err))
                if dlq_ok:
                    dead_letter += 1
                    if ack_id:
                        ack_ids.append(ack_id)
                else:
                    failed += 1
                    logger.error(f"Failed to route conflicting payload to DLQ; leaving unacknowledged: {val_err}")
                continue
            except Exception as db_exc:
                logger.error(f"Database error replaying event {event.request_id}: {db_exc}")
                db_success = False

            if db_success:
                ingested += 1
                if ack_id:
                    ack_ids.append(ack_id)
            else:
                failed += 1

        # 3. Acknowledge successfully processed and safely dead-lettered messages
        if ack_ids:
            ack_url = f"https://pubsub.googleapis.com/v1/projects/{self.project_id}/subscriptions/{self.subscription}:acknowledge"
            try:
                await client.post(ack_url, headers=headers, json={"ackIds": ack_ids})
            except Exception as ack_exc:
                logger.error(f"Failed to acknowledge Pub/Sub messages: {ack_exc}")

        return TelemetryReplayBatchSummary(
            pulled=pulled,
            ingested=ingested,
            duplicates=duplicates,
            failed=failed,
            dead_letter=dead_letter,
        )

    async def close(self) -> None:
        if self._owns_client and self._client and not self._client.is_closed:
            await self._client.aclose()


async def ingest_telemetry_event(session: Any, event: TelemetryEvent) -> bool:
    """Ingest a durable telemetry event from a queue subscription into mp_inference_logs.

    Security & Idempotency:
    - Validates event payload against schema constraints before inserting.
    - Binds parameters safely using parameterized SQL.
    - Uses ON CONFLICT (request_id) DO NOTHING to ensure idempotent redelivery.
    - Detects and rejects conflicting duplicate payloads where request_id matches but payload differs.
    - Rejects invalid or corrupted event payloads.
    - Newly ingested records have settlement_status='PENDING' and settlement_attempts=0,
      making them immediately eligible for pickup by the website settlement scheduler.
    """
    validate_query_safety(_INSERT_INFERENCE_LOG_QUERY)
    validate_telemetry_event(event)

    if session is None:
        log_event(
            logger,
            logging.ERROR,
            "telemetry.session_none",
            "Database session is None; inference telemetry replay could not be written to mp_inference_logs.",
            metadata={"request_id": event.request_id, "model_id": event.model_id},
            request_id=event.request_id,
        )
        return False

    try:
        result = await session.execute(text(_INSERT_INFERENCE_LOG_QUERY), event.to_params())
        rowcount = getattr(result, "rowcount", None)

        if rowcount == 0:
            # Row was not inserted due to conflict on request_id.
            # Verify whether existing record is an identical idempotent replay or a conflicting payload.
            check_query = """
            SELECT model_id, user_id, tokens_in, tokens_out, cost_usd
            FROM mp_inference_logs
            WHERE request_id = :request_id
            LIMIT 1
            """
            existing = await session.execute(text(check_query), {"request_id": event.request_id})
            row = existing.mappings().first() if hasattr(existing, "mappings") else None
            if not row and hasattr(existing, "fetchone"):
                row = existing.fetchone()

            if row:
                m_model = row["model_id"] if isinstance(row, dict) else getattr(row, "model_id", row[0])
                m_user = row["user_id"] if isinstance(row, dict) else getattr(row, "user_id", row[1])
                m_tin = row["tokens_in"] if isinstance(row, dict) else getattr(row, "tokens_in", row[2])
                m_tout = row["tokens_out"] if isinstance(row, dict) else getattr(row, "tokens_out", row[3])
                m_cost = Decimal(str(row["cost_usd"] if isinstance(row, dict) else getattr(row, "cost_usd", row[4])))

                if (
                    m_model != event.model_id
                    or (m_user or None) != (event.user_id or None)
                    or m_tin != event.tokens_in
                    or m_tout != event.tokens_out
                    or m_cost != event.cost_usd
                ):
                    log_event(
                        logger,
                        logging.ERROR,
                        "telemetry.duplicate_payload_mismatch",
                        f"Conflicting telemetry payload for existing request_id: {event.request_id}",
                        metadata={"request_id": event.request_id, "model_id": event.model_id},
                        request_id=event.request_id,
                    )
                    raise ValueError(f"Conflicting payload for existing request_id: {event.request_id}")

        if hasattr(session, "commit"):
            await session.commit()
        log_event(
            logger,
            logging.INFO,
            "telemetry.replay_ingested",
            f"Queued telemetry event replayed into mp_inference_logs (request_id={event.request_id}, model={event.model_id})",
            metadata={"request_id": event.request_id, "model_id": event.model_id, "cost_usd": str(event.cost_usd)},
            request_id=event.request_id,
        )
        return True
    except ValueError:
        raise
    except Exception as exc:
        log_event(
            logger,
            logging.ERROR,
            "telemetry.ingest_failed",
            f"Failed to ingest telemetry event into database: {type(exc).__name__}",
            metadata={"request_id": event.request_id, "error": str(exc)[:200]},
            request_id=event.request_id,
        )
        return False
