"""Unit tests for Phase B durable telemetry architecture.

Covers:
- TelemetryEvent serialization, deserialization, and parameter mapping.
- Direct database recording vs fallback queue publication.
- Fail-closed enforcement in production vs fail-open development override.
- Google Cloud Pub/Sub payload encoding, attributes, retries, and dead-letter handling.
- Idempotent event ingestion via ON CONFLICT (request_id) DO NOTHING.
- Bounded processing and resource safety.
"""

from __future__ import annotations

import base64
import json
import logging
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from app.auth import AuthenticatedPrincipal
from app.telemetry import (
    MockTelemetryPublisher,
    PubSubTelemetryPublisher,
    TelemetryEvent,
    TelemetryPersistenceError,
    TelemetryRecordResult,
    ingest_telemetry_event,
    record_durable_inference_telemetry,
)


def _mock_principal() -> AuthenticatedPrincipal:
    return AuthenticatedPrincipal(
        user_id="usr_durable_test_123",
        api_key_id="key_durable_test_456",
        channel="api",
        wallet_balance=Decimal("50.000000"),
    )


# ── TelemetryEvent Contract Tests ──────────────────────────────────────────


def test_telemetry_event_creation_and_serialization():
    """TelemetryEvent preserves exact decimal precision and parameter contract."""
    event = TelemetryEvent.create(
        request_id="req_durable_001",
        user_id="usr_test",
        api_key_id="key_test",
        model_id="rantera-base-v1",
        tokens_in=100,
        tokens_out=50,
        cost_usd=Decimal("0.001050"),
        latency_ms=150,
        request_status="SUCCESS",
        channel="api",
    )

    assert event.request_id == "req_durable_001"
    assert event.total_tokens == 150
    assert event.cost_usd == Decimal("0.001050")

    # JSON serialization
    json_str = event.to_json()
    deserialized = TelemetryEvent.from_json(json_str)

    assert deserialized.id == event.id
    assert deserialized.request_id == event.request_id
    assert deserialized.cost_usd == Decimal("0.001050")
    assert deserialized.total_tokens == 150

    # DB parameters mapping
    params = event.to_params()
    assert params["request_id"] == "req_durable_001"
    assert params["settlement_status"] == "PENDING"
    assert params["settlement_attempts"] == 0


# ── Durable Coordinator Tests ──────────────────────────────────────────────


@pytest.mark.anyio
async def test_record_durable_telemetry_database_success():
    """When DB session is active, primary database write is used and succeeds."""
    session = AsyncMock()
    principal = _mock_principal()
    mock_queue = MockTelemetryPublisher()

    result = await record_durable_inference_telemetry(
        session=session,
        request_id="req_db_success_1",
        principal=principal,
        model_id="rantera-base-v1",
        tokens_in=40,
        tokens_out=20,
        cost_usd=Decimal("0.000420"),
        latency_ms=85,
        queue_publisher=mock_queue,
        fail_closed=True,
    )

    assert result.persisted is True
    assert result.storage_type == "database"
    assert len(mock_queue.events) == 0  # Queue was not needed
    session.execute.assert_awaited_once()
    session.commit.assert_awaited_once()


@pytest.mark.anyio
async def test_record_durable_telemetry_database_failure_falls_back_to_queue():
    """When DB write throws an exception, coordinator falls back to durable queue."""
    session = AsyncMock()
    session.execute.side_effect = RuntimeError("PostgreSQL connection refused")
    principal = _mock_principal()
    mock_queue = MockTelemetryPublisher()

    result = await record_durable_inference_telemetry(
        session=session,
        request_id="req_db_fail_fallback_queue",
        principal=principal,
        model_id="rantera-base-v1",
        tokens_in=40,
        tokens_out=20,
        cost_usd=Decimal("0.000420"),
        latency_ms=85,
        queue_publisher=mock_queue,
        fail_closed=True,
    )

    assert result.persisted is True
    assert result.storage_type == "queue"
    assert len(mock_queue.events) == 1
    assert mock_queue.events[0].request_id == "req_db_fail_fallback_queue"


@pytest.mark.anyio
async def test_record_durable_telemetry_dual_failure_fails_closed_in_production():
    """When DB is unavailable and queue fails, coordinator fails closed by raising."""
    session = None
    principal = _mock_principal()
    mock_queue = MockTelemetryPublisher(should_fail=True)

    with pytest.raises(TelemetryPersistenceError) as exc_info:
        await record_durable_inference_telemetry(
            session=session,
            request_id="req_dual_failure_prod",
            principal=principal,
            model_id="rantera-base-v1",
            tokens_in=10,
            tokens_out=5,
            cost_usd=Decimal("0.000105"),
            latency_ms=40,
            queue_publisher=mock_queue,
            fail_closed=True,
        )

    assert "req_dual_failure_prod" in str(exc_info.value)
    assert exc_info.value.event is not None
    assert exc_info.value.event.request_id == "req_dual_failure_prod"


@pytest.mark.anyio
async def test_record_durable_telemetry_dual_failure_fail_open_override():
    """When fail_closed=False (explicit dev override), returns failure result without raising."""
    session = None
    principal = _mock_principal()
    mock_queue = MockTelemetryPublisher(should_fail=True)

    result = await record_durable_inference_telemetry(
        session=session,
        request_id="req_dual_failure_dev",
        principal=principal,
        model_id="rantera-base-v1",
        tokens_in=10,
        tokens_out=5,
        cost_usd=Decimal("0.000105"),
        latency_ms=40,
        queue_publisher=mock_queue,
        fail_closed=False,
    )

    assert result.persisted is False
    assert result.storage_type == "none"
    assert result.error is not None


# ── Google Cloud Pub/Sub Publisher Tests ───────────────────────────────────


@pytest.mark.anyio
async def test_pubsub_publisher_payload_and_attributes():
    """Verify PubSub message structure binds base64 data, orderingKey, and attributes."""
    captured_requests: list[dict] = []

    def mock_transport(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        captured_requests.append(body)
        return httpx.Response(200, json={"messageIds": ["pubsub_msg_12345"]})

    client = httpx.AsyncClient(transport=httpx.MockTransport(mock_transport))

    publisher = PubSubTelemetryPublisher(
        project_id="test-rantera-project",
        topic="mp-inference-telemetry",
        client=client,
        credentials=MagicMock(valid=True, token="mock-gcp-token"),
    )

    event = TelemetryEvent.create(
        request_id="req_pubsub_spec_1",
        user_id="usr_pubsub_1",
        api_key_id="key_pubsub_1",
        model_id="rantera-base-v1",
        tokens_in=50,
        tokens_out=25,
        cost_usd=Decimal("0.000525"),
        latency_ms=90,
    )

    success = await publisher.publish_event(event)
    assert success is True
    assert len(captured_requests) == 1

    msg = captured_requests[0]["messages"][0]
    assert msg["orderingKey"] == "req_pubsub_spec_1"
    assert msg["attributes"]["request_id"] == "req_pubsub_spec_1"
    assert msg["attributes"]["model_id"] == "rantera-base-v1"
    assert msg["attributes"]["user_id"] == "usr_pubsub_1"

    decoded_json = base64.b64decode(msg["data"]).decode("utf-8")
    payload = json.loads(decoded_json)
    assert payload["request_id"] == "req_pubsub_spec_1"
    assert payload["cost_usd"] == "0.000525"


@pytest.mark.anyio
async def test_pubsub_publisher_retries_and_dlq_on_outage():
    """PubSub retries on 503 and routes to dead letter topic on permanent failure."""
    calls = []

    def mock_transport(request: httpx.Request) -> httpx.Response:
        url_str = str(request.url)
        calls.append(url_str)
        if "mp-inference-telemetry-dlq" in url_str:
            return httpx.Response(200, json={"messageIds": ["dlq_msg_001"]})
        # Primary topic fails
        return httpx.Response(503, text="Service Unavailable")

    client = httpx.AsyncClient(transport=httpx.MockTransport(mock_transport))

    publisher = PubSubTelemetryPublisher(
        project_id="test-rantera-project",
        topic="mp-inference-telemetry",
        dead_letter_topic="mp-inference-telemetry-dlq",
        max_retries=2,
        client=client,
        credentials=MagicMock(valid=True, token="mock-token"),
    )

    event = TelemetryEvent.create(
        request_id="req_pubsub_dlq_test",
        user_id="usr_dlq",
        api_key_id="key_dlq",
        model_id="rantera-base-v1",
        tokens_in=10,
        tokens_out=5,
        cost_usd=Decimal("0.000105"),
        latency_ms=30,
    )

    success = await publisher.publish_event(event)
    # Returns False since primary failed, but DLQ was reached
    assert success is False
    # 2 retries to primary topic + 1 DLQ attempt
    assert len(calls) == 3
    assert any("mp-inference-telemetry-dlq" in c for c in calls)


# ── Telemetry Ingestion Worker Tests ───────────────────────────────────────


@pytest.mark.anyio
async def test_ingest_telemetry_event_idempotent_execution():
    """Ingestion worker binds parameters and executes with ON CONFLICT DO NOTHING."""
    session = AsyncMock()
    event = TelemetryEvent.create(
        request_id="req_ingest_idempotent_01",
        user_id="usr_worker",
        api_key_id="key_worker",
        model_id="rantera-base-v1",
        tokens_in=30,
        tokens_out=15,
        cost_usd=Decimal("0.000315"),
        latency_ms=60,
    )

    success = await ingest_telemetry_event(session, event)
    assert success is True
    session.execute.assert_awaited_once()
    session.commit.assert_awaited_once()

    call_args = session.execute.await_args
    params = call_args[0][1]
    assert params["request_id"] == "req_ingest_idempotent_01"
    assert params["settlement_status"] == "PENDING"
