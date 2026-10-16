"""Unit tests for telemetry consumer/replay, request ID preservation, and push webhook."""

import base64
import json
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from app.models import RegisteredModel
from app.schemas import ChatMessage
from app.telemetry import (
    PubSubTelemetryConsumer,
    TelemetryEvent,
    TelemetryReplayBatchSummary,
    ingest_telemetry_event,
    validate_telemetry_event,
)
from app.vertex import PredictionResult


class _DummyModelProvider:
    async def predict(self, model: RegisteredModel, messages: list[ChatMessage], **kwargs):
        return PredictionResult(
            content="dummy response",
            tokens_in=10,
            tokens_out=5,
            latency_ms=15,
        )


# ── Event Validation Tests ──────────────────────────────────────────────────


def test_validate_telemetry_event_valid():
    """Valid event passes schema and bound constraints."""
    event = TelemetryEvent.create(
        request_id="req_valid_12345",
        user_id="usr_123",
        api_key_id="key_123",
        model_id="rantera-base-v1",
        tokens_in=10,
        tokens_out=5,
        cost_usd=Decimal("0.000105"),
        latency_ms=25,
    )
    validate_telemetry_event(event)


def test_validate_telemetry_event_invalid_request_id():
    """Malformed request_id with newline injection or invalid chars raises ValueError."""
    event = TelemetryEvent.create(
        request_id="req_invalid\r\ninjected: header",
        user_id="usr_123",
        api_key_id="key_123",
        model_id="rantera-base-v1",
        tokens_in=10,
        tokens_out=5,
        cost_usd=Decimal("0.000105"),
        latency_ms=25,
    )
    with pytest.raises(ValueError, match="Invalid or malformed request_id"):
        validate_telemetry_event(event)


def test_validate_telemetry_event_negative_tokens():
    """Negative token count raises ValueError."""
    event = TelemetryEvent(
        id="inf_123",
        request_id="req_123",
        user_id="usr_123",
        api_key_id=None,
        model_id="rantera-base-v1",
        channel="api",
        tokens_in=-5,
        tokens_out=10,
        total_tokens=5,
        cost_usd=Decimal("0.000100"),
        latency_ms=20,
        ttft_ms=None,
        request_status="SUCCESS",
        error_code=None,
        created_at="2026-10-15T12:00:00Z",
    )
    with pytest.raises(ValueError, match="Token counts must be non-negative"):
        validate_telemetry_event(event)


# ── PubSubTelemetryConsumer Tests ───────────────────────────────────────────


@pytest.mark.anyio
async def test_consumer_pull_and_process_batch_success():
    """Consumer pulls messages from Pub/Sub, ingests to DB, and acknowledges messages."""
    event = TelemetryEvent.create(
        request_id="req_pull_01",
        user_id="usr_batch",
        api_key_id=None,
        model_id="rantera-base-v1",
        tokens_in=20,
        tokens_out=10,
        cost_usd=Decimal("0.000210"),
        latency_ms=40,
    )
    b64_payload = base64.b64encode(event.to_json().encode("utf-8")).decode("ascii")

    mock_pull_response = {
        "receivedMessages": [
            {
                "ackId": "ack_token_123",
                "message": {
                    "data": b64_payload,
                    "attributes": {"request_id": "req_pull_01"},
                },
            }
        ]
    }

    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_client.is_closed = False

    async def mock_post(url, **kwargs):
        if ":pull" in url:
            return httpx.Response(200, json=mock_pull_response)
        if ":acknowledge" in url:
            assert kwargs.get("json", {}).get("ackIds") == ["ack_token_123"]
            return httpx.Response(200, json={})
        return httpx.Response(404)

    mock_client.post = AsyncMock(side_effect=mock_post)

    mock_session = AsyncMock()
    mock_session_factory = MagicMock()
    mock_session_factory.return_value.__aenter__.return_value = mock_session

    consumer = PubSubTelemetryConsumer(
        project_id="test-project",
        subscription="mp-inference-telemetry-sub",
        client=mock_client,
    )

    summary = await consumer.pull_and_process_batch(mock_session_factory, max_messages=10)

    assert summary.pulled == 1
    assert summary.ingested == 1
    assert summary.failed == 0
    assert summary.dead_letter == 0
    mock_session.execute.assert_awaited_once()
    mock_session.commit.assert_awaited_once()


@pytest.mark.anyio
async def test_consumer_corrupted_payload_routes_to_dlq_and_acks():
    """Corrupted/poison payloads route to DLQ topic and are acknowledged to unblock queue."""
    mock_pull_response = {
        "receivedMessages": [
            {
                "ackId": "ack_poison_999",
                "message": {
                    "data": base64.b64encode(b"not-valid-json").decode("ascii"),
                    "attributes": {"malformed": "true"},
                },
            }
        ]
    }

    dlq_calls = []
    ack_calls = []

    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_client.is_closed = False

    async def mock_post(url, **kwargs):
        if ":pull" in url:
            return httpx.Response(200, json=mock_pull_response)
        if "mp-telemetry-dlq" in url:
            dlq_calls.append(kwargs)
            return httpx.Response(200, json={})
        if ":acknowledge" in url:
            ack_calls.append(kwargs)
            return httpx.Response(200, json={})
        return httpx.Response(404)

    mock_client.post = AsyncMock(side_effect=mock_post)
    mock_session_factory = MagicMock()

    consumer = PubSubTelemetryConsumer(
        project_id="test-project",
        subscription="mp-inference-telemetry-sub",
        dead_letter_topic="mp-telemetry-dlq",
        client=mock_client,
    )

    summary = await consumer.pull_and_process_batch(mock_session_factory)

    assert summary.pulled == 1
    assert summary.ingested == 0
    assert summary.dead_letter == 1
    assert len(dlq_calls) == 1
    assert len(ack_calls) == 1
    assert ack_calls[0].get("json", {}).get("ackIds") == ["ack_poison_999"]


# ── Request ID Inbound Preservation Tests ────────────────────────────────────


def test_request_id_inbound_preserved():
    """Valid inbound X-Request-Id header is preserved across response and state."""
    settings = Settings(
        gcp_project_id=None,
        gcp_region=None,
        vertex_endpoint_id=None,
    )
    app = create_app(settings=settings)
    client = TestClient(app)

    custom_id = "req_custom_upstream_trace_12345"
    response = client.get("/health/live", headers={"X-Request-Id": custom_id})

    assert response.status_code == 200
    assert response.headers.get("X-Request-ID") == custom_id


def test_request_id_malformed_inbound_replaced():
    """Inbound request ID containing spaces or control characters is sanitized and replaced with fresh ID."""
    settings = Settings(
        gcp_project_id=None,
        gcp_region=None,
        vertex_endpoint_id=None,
    )
    app = create_app(settings=settings)
    client = TestClient(app)

    malformed_id = "req_bad\r\ninjection: attempt"
    response = client.get("/health/live", headers={"X-Request-Id": malformed_id})

    assert response.status_code == 200
    returned_id = response.headers.get("X-Request-ID")
    assert returned_id != malformed_id
    assert returned_id.startswith("req_")
    assert "\r" not in returned_id


def test_request_id_missing_inbound_generates_new():
    """When no inbound X-Request-Id is sent, gateway generates a fresh req_ identifier."""
    settings = Settings(
        gcp_project_id=None,
        gcp_region=None,
        vertex_endpoint_id=None,
    )
    app = create_app(settings=settings)
    client = TestClient(app)

    response = client.get("/health/live")
    assert response.status_code == 200
    returned_id = response.headers.get("X-Request-ID")
    assert returned_id is not None
    assert returned_id.startswith("req_")


# ── Pub/Sub Push Webhook Endpoint Tests ─────────────────────────────────────


def test_pubsub_push_endpoint_success():
    """Push endpoint decodes valid Pub/Sub message and ingests to database."""
    settings = Settings(
        gcp_project_id=None,
        gcp_region=None,
        vertex_endpoint_id=None,
        internal_service_key="secret-service-key-999",
    )
    app = create_app(settings=settings)

    event = TelemetryEvent.create(
        request_id="req_push_test_001",
        user_id="usr_push",
        api_key_id=None,
        model_id="rantera-base-v1",
        tokens_in=15,
        tokens_out=8,
        cost_usd=Decimal("0.000165"),
        latency_ms=50,
    )
    b64_payload = base64.b64encode(event.to_json().encode("utf-8")).decode("ascii")

    payload = {
        "message": {
            "data": b64_payload,
            "attributes": {"request_id": "req_push_test_001"},
            "messageId": "msg_123",
        },
        "subscription": "projects/test/subscriptions/sub",
    }

    mock_session = AsyncMock()
    mock_factory = MagicMock()
    mock_factory.return_value.__aenter__.return_value = mock_session
    app.state.db_session_factory = mock_factory

    client = TestClient(app)
    response = client.post(
        "/v1/telemetry/pubsub/push",
        headers={"X-Internal-Service-Key": "secret-service-key-999"},
        json=payload,
    )

    assert response.status_code == 200
    assert response.json() == {"status": "ingested", "request_id": "req_push_test_001"}
    mock_session.execute.assert_awaited_once()


def test_pubsub_push_endpoint_unauthorized():
    """Push endpoint rejects request when credentials are missing or incorrect."""
    settings = Settings(
        gcp_project_id=None,
        gcp_region=None,
        vertex_endpoint_id=None,
        internal_service_key="correct-key-12345",
    )
    app = create_app(settings=settings)
    client = TestClient(app)

    response = client.post(
        "/v1/telemetry/pubsub/push",
        headers={"X-Internal-Service-Key": "wrong-key"},
        json={"message": {"data": "dGVzdA=="}},
    )
    assert response.status_code == 401


def test_pubsub_push_endpoint_forbids_query_param_tokens():
    """Push endpoint strictly rejects query string authentication."""
    settings = Settings(
        gcp_project_id=None,
        gcp_region=None,
        vertex_endpoint_id=None,
        internal_service_key="secret-key",
    )
    app = create_app(settings=settings)
    client = TestClient(app)

    response = client.post(
        "/v1/telemetry/pubsub/push?token=secret-key",
        json={"message": {"data": "dGVzdA=="}},
    )
    assert response.status_code == 401
    assert "forbidden" in response.json()["message"].lower()


@patch("google.oauth2.id_token.verify_oauth2_token")
def test_pubsub_push_endpoint_google_oidc_success(mock_verify):
    """Push endpoint accepts verified Google OIDC bearer token matching audience and service account."""
    mock_verify.return_value = {
        "iss": "https://accounts.google.com",
        "email": "pubsub-invoker@rantera-prod.iam.gserviceaccount.com",
        "aud": "https://gateway.rantera.ai/v1/telemetry/pubsub/push",
    }
    settings = Settings(
        gcp_project_id=None,
        gcp_region=None,
        vertex_endpoint_id=None,
        pubsub_push_audience="https://gateway.rantera.ai/v1/telemetry/pubsub/push",
        pubsub_push_service_account_email="pubsub-invoker@rantera-prod.iam.gserviceaccount.com",
    )
    app = create_app(settings=settings)

    event = TelemetryEvent.create(
        request_id="req_push_oidc_001",
        user_id="usr_oidc",
        api_key_id=None,
        model_id="rantera-base-v1",
        tokens_in=10,
        tokens_out=5,
        cost_usd=Decimal("0.000105"),
        latency_ms=40,
    )
    b64_payload = base64.b64encode(event.to_json().encode("utf-8")).decode("ascii")

    mock_session = AsyncMock()
    mock_factory = MagicMock()
    mock_factory.return_value.__aenter__.return_value = mock_session
    app.state.db_session_factory = mock_factory

    client = TestClient(app)
    response = client.post(
        "/v1/telemetry/pubsub/push",
        headers={"Authorization": "Bearer mock-google-oidc-jwt-token"},
        json={"message": {"data": b64_payload}},
    )
    assert response.status_code == 200
    assert response.json() == {"status": "ingested", "request_id": "req_push_oidc_001"}
    mock_verify.assert_called_once()


@patch("google.oauth2.id_token.verify_oauth2_token")
def test_pubsub_push_endpoint_google_oidc_unauthorized_service_account(mock_verify):
    """Push endpoint rejects token from unauthorized service account email."""
    mock_verify.return_value = {
        "iss": "https://accounts.google.com",
        "email": "unauthorized-hacker@evil.com",
        "aud": "https://gateway.rantera.ai/v1/telemetry/pubsub/push",
    }
    settings = Settings(
        gcp_project_id=None,
        gcp_region=None,
        vertex_endpoint_id=None,
        pubsub_push_audience="https://gateway.rantera.ai/v1/telemetry/pubsub/push",
        pubsub_push_service_account_email="pubsub-invoker@rantera-prod.iam.gserviceaccount.com",
    )
    app = create_app(settings=settings)
    client = TestClient(app)

    response = client.post(
        "/v1/telemetry/pubsub/push",
        headers={"Authorization": "Bearer mock-evil-token"},
        json={"message": {"data": "dGVzdA=="}},
    )
    assert response.status_code == 401
    assert "Unauthorized Pub/Sub push service account" in response.json()["message"]


@patch("google.oauth2.id_token.verify_oauth2_token")
def test_pubsub_push_endpoint_google_oidc_wrong_audience(mock_verify):
    """Push endpoint rejects token when audience verification fails in google-auth library."""
    mock_verify.side_effect = ValueError("Audience does not match")
    settings = Settings(
        gcp_project_id=None,
        gcp_region=None,
        vertex_endpoint_id=None,
        pubsub_push_audience="https://gateway.rantera.ai/v1/telemetry/pubsub/push",
        pubsub_push_service_account_email="pubsub-invoker@rantera-prod.iam.gserviceaccount.com",
    )
    app = create_app(settings=settings)
    client = TestClient(app)

    response = client.post(
        "/v1/telemetry/pubsub/push",
        headers={"Authorization": "Bearer token-wrong-aud"},
        json={"message": {"data": "dGVzdA=="}},
    )
    assert response.status_code == 401
    assert "Invalid OIDC bearer token" in response.json()["message"]


def test_pubsub_push_endpoint_google_oidc_missing_service_account_email():
    """Push endpoint fails closed (401) when audience is configured but service account email is missing."""
    settings = Settings(
        gcp_project_id=None,
        gcp_region=None,
        vertex_endpoint_id=None,
        pubsub_push_audience="https://gateway.rantera.ai/v1/telemetry/pubsub/push",
        pubsub_push_service_account_email=None,
    )
    app = create_app(settings=settings)
    client = TestClient(app)

    response = client.post(
        "/v1/telemetry/pubsub/push",
        headers={"Authorization": "Bearer any-token"},
        json={"message": {"data": "dGVzdA=="}},
    )
    assert response.status_code == 401
    assert "not configured" in response.json()["message"]


def test_pubsub_push_endpoint_google_oidc_missing_audience():
    """Push endpoint fails closed (401) when service account email is configured but audience is missing."""
    settings = Settings(
        gcp_project_id=None,
        gcp_region=None,
        vertex_endpoint_id=None,
        pubsub_push_audience=None,
        pubsub_push_service_account_email="pubsub-invoker@rantera-prod.iam.gserviceaccount.com",
    )
    app = create_app(settings=settings)
    client = TestClient(app)

    response = client.post(
        "/v1/telemetry/pubsub/push",
        headers={"Authorization": "Bearer any-token"},
        json={"message": {"data": "dGVzdA=="}},
    )
    assert response.status_code == 401
    assert "not configured" in response.json()["message"]


@patch("google.oauth2.id_token.verify_oauth2_token")
def test_pubsub_push_endpoint_google_oidc_expired_token(mock_verify):
    """Push endpoint rejects token when token expiration check fails."""
    mock_verify.side_effect = ValueError("Token used too late (expired)")
    settings = Settings(
        gcp_project_id=None,
        gcp_region=None,
        vertex_endpoint_id=None,
        pubsub_push_audience="https://gateway.rantera.ai/v1/telemetry/pubsub/push",
        pubsub_push_service_account_email="pubsub-invoker@rantera-prod.iam.gserviceaccount.com",
    )
    app = create_app(settings=settings)
    client = TestClient(app)

    response = client.post(
        "/v1/telemetry/pubsub/push",
        headers={"Authorization": "Bearer expired-token"},
        json={"message": {"data": "dGVzdA=="}},
    )
    assert response.status_code == 401
    assert "Invalid OIDC bearer token" in response.json()["message"]


@patch("google.oauth2.id_token.verify_oauth2_token")
def test_pubsub_push_endpoint_google_oidc_invalid_token_signature(mock_verify):
    """Push endpoint rejects token when signature validation fails."""
    mock_verify.side_effect = ValueError("Invalid token signature")
    settings = Settings(
        gcp_project_id=None,
        gcp_region=None,
        vertex_endpoint_id=None,
        pubsub_push_audience="https://gateway.rantera.ai/v1/telemetry/pubsub/push",
        pubsub_push_service_account_email="pubsub-invoker@rantera-prod.iam.gserviceaccount.com",
    )
    app = create_app(settings=settings)
    client = TestClient(app)

    response = client.post(
        "/v1/telemetry/pubsub/push",
        headers={"Authorization": "Bearer corrupted-token"},
        json={"message": {"data": "dGVzdA=="}},
    )
    assert response.status_code == 401
    assert "Invalid OIDC bearer token" in response.json()["message"]


@patch("google.oauth2.id_token.verify_oauth2_token")
def test_pubsub_push_endpoint_google_oidc_untrusted_issuer(mock_verify):
    """Push endpoint rejects token when issuer is not accounts.google.com."""
    mock_verify.return_value = {
        "iss": "https://untrusted-issuer.example.com",
        "email": "pubsub-invoker@rantera-prod.iam.gserviceaccount.com",
        "aud": "https://gateway.rantera.ai/v1/telemetry/pubsub/push",
    }
    settings = Settings(
        gcp_project_id=None,
        gcp_region=None,
        vertex_endpoint_id=None,
        pubsub_push_audience="https://gateway.rantera.ai/v1/telemetry/pubsub/push",
        pubsub_push_service_account_email="pubsub-invoker@rantera-prod.iam.gserviceaccount.com",
    )
    app = create_app(settings=settings)
    client = TestClient(app)

    response = client.post(
        "/v1/telemetry/pubsub/push",
        headers={"Authorization": "Bearer untrusted-issuer-token"},
        json={"message": {"data": "dGVzdA=="}},
    )
    assert response.status_code == 401
    assert "Invalid token issuer" in response.json()["message"]


@patch("google.oauth2.id_token.verify_oauth2_token")
def test_pubsub_push_endpoint_google_oidc_missing_email_claim(mock_verify):
    """Push endpoint rejects token when email claim is completely missing."""
    mock_verify.return_value = {
        "iss": "https://accounts.google.com",
        "aud": "https://gateway.rantera.ai/v1/telemetry/pubsub/push",
    }
    settings = Settings(
        gcp_project_id=None,
        gcp_region=None,
        vertex_endpoint_id=None,
        pubsub_push_audience="https://gateway.rantera.ai/v1/telemetry/pubsub/push",
        pubsub_push_service_account_email="pubsub-invoker@rantera-prod.iam.gserviceaccount.com",
    )
    app = create_app(settings=settings)
    client = TestClient(app)

    response = client.post(
        "/v1/telemetry/pubsub/push",
        headers={"Authorization": "Bearer no-email-token"},
        json={"message": {"data": "dGVzdA=="}},
    )
    assert response.status_code == 401
    assert "Unauthorized Pub/Sub push service account" in response.json()["message"]


@patch("google.oauth2.id_token.verify_oauth2_token")
def test_pubsub_push_endpoint_google_oidc_unverified_email(mock_verify):
    """Push endpoint rejects token when email_verified is explicitly false."""
    mock_verify.return_value = {
        "iss": "https://accounts.google.com",
        "email": "pubsub-invoker@rantera-prod.iam.gserviceaccount.com",
        "email_verified": False,
        "aud": "https://gateway.rantera.ai/v1/telemetry/pubsub/push",
    }
    settings = Settings(
        gcp_project_id=None,
        gcp_region=None,
        vertex_endpoint_id=None,
        pubsub_push_audience="https://gateway.rantera.ai/v1/telemetry/pubsub/push",
        pubsub_push_service_account_email="pubsub-invoker@rantera-prod.iam.gserviceaccount.com",
    )
    app = create_app(settings=settings)
    client = TestClient(app)

    response = client.post(
        "/v1/telemetry/pubsub/push",
        headers={"Authorization": "Bearer unverified-email-token"},
        json={"message": {"data": "dGVzdA=="}},
    )
    assert response.status_code == 401
    assert "Unauthorized Pub/Sub push service account" in response.json()["message"]


def test_pubsub_push_endpoint_missing_credentials():
    """Push endpoint rejects request when no authentication headers are provided."""
    settings = Settings(
        gcp_project_id=None,
        gcp_region=None,
        vertex_endpoint_id=None,
        pubsub_push_audience="https://gateway.rantera.ai/v1/telemetry/pubsub/push",
        pubsub_push_service_account_email="pubsub-invoker@rantera-prod.iam.gserviceaccount.com",
    )
    app = create_app(settings=settings)
    client = TestClient(app)

    response = client.post(
        "/v1/telemetry/pubsub/push",
        json={"message": {"data": "dGVzdA=="}},
    )
    assert response.status_code == 401
    assert "Missing authentication credentials" in response.json()["message"]


def test_pubsub_push_endpoint_invalid_authorization_scheme():
    """Push endpoint rejects non-Bearer authorization headers."""
    settings = Settings(
        gcp_project_id=None,
        gcp_region=None,
        vertex_endpoint_id=None,
        pubsub_push_audience="https://gateway.rantera.ai/v1/telemetry/pubsub/push",
        pubsub_push_service_account_email="pubsub-invoker@rantera-prod.iam.gserviceaccount.com",
    )
    app = create_app(settings=settings)
    client = TestClient(app)

    response = client.post(
        "/v1/telemetry/pubsub/push",
        headers={"Authorization": "Basic dXNlcjpwYXNzd29yZA=="},
        json={"message": {"data": "dGVzdA=="}},
    )
    assert response.status_code == 401
    assert "Invalid authorization scheme" in response.json()["message"]


def test_pubsub_push_endpoint_conflicting_credentials():
    """Push endpoint rejects request when both Bearer token and X-Internal-Service-Key are provided."""
    settings = Settings(
        gcp_project_id=None,
        gcp_region=None,
        vertex_endpoint_id=None,
        internal_service_key="secret-service-key-999",
        pubsub_push_audience="https://gateway.rantera.ai/v1/telemetry/pubsub/push",
        pubsub_push_service_account_email="pubsub-invoker@rantera-prod.iam.gserviceaccount.com",
    )
    app = create_app(settings=settings)
    client = TestClient(app)

    response = client.post(
        "/v1/telemetry/pubsub/push",
        headers={
            "Authorization": "Bearer some-token",
            "X-Internal-Service-Key": "secret-service-key-999",
        },
        json={"message": {"data": "dGVzdA=="}},
    )
    assert response.status_code == 400
    assert "Conflicting authentication credentials" in response.json()["message"]


def test_pubsub_push_endpoint_fails_closed_when_key_unconfigured():
    """Push endpoint fails closed (401) when neither OIDC nor internal service key is configured."""
    settings = Settings(
        gcp_project_id=None,
        gcp_region=None,
        vertex_endpoint_id=None,
        internal_service_key=None,
        pubsub_push_audience=None,
        pubsub_push_service_account_email=None,
    )
    app = create_app(settings=settings)
    client = TestClient(app)

    # 1. Bearer token when OIDC unconfigured
    response = client.post(
        "/v1/telemetry/pubsub/push",
        headers={"Authorization": "Bearer any-token"},
        json={"message": {"data": "dGVzdA=="}},
    )
    assert response.status_code == 401
    assert "not configured" in response.json()["message"]

    # 2. X-Internal-Service-Key when internal key unconfigured
    response_key = client.post(
        "/v1/telemetry/pubsub/push",
        headers={"X-Internal-Service-Key": "any-key"},
        json={"message": {"data": "dGVzdA=="}},
    )
    assert response_key.status_code == 401
    assert "not configured" in response_key.json()["message"]


@pytest.mark.anyio
async def test_ingest_telemetry_event_conflicting_duplicate_payload_raises():
    """Ingest rejects conflicting duplicate payload with same request_id but different token count."""
    event = TelemetryEvent.create(
        request_id="req_conflict_001",
        user_id="usr_1",
        api_key_id=None,
        model_id="rantera-base-v1",
        tokens_in=100,
        tokens_out=200,
        cost_usd=Decimal("0.003300"),
        latency_ms=100,
    )

    mock_session = AsyncMock()
    # Simulate rowcount = 0 (conflict on request_id)
    execute_insert_result = MagicMock()
    execute_insert_result.rowcount = 0

    # Simulate existing row having DIFFERENT tokens_in (e.g. 10 instead of 100)
    existing_row = {
        "model_id": "rantera-base-v1",
        "user_id": "usr_1",
        "tokens_in": 10,  # DIFFERENT!
        "tokens_out": 200,
        "cost_usd": Decimal("0.003300"),
    }
    execute_select_result = MagicMock()
    execute_select_result.mappings.return_value.first.return_value = existing_row

    mock_session.execute.side_effect = [execute_insert_result, execute_select_result]

    with pytest.raises(ValueError, match="Conflicting payload for existing request_id"):
        await ingest_telemetry_event(mock_session, event)


@pytest.mark.anyio
async def test_consumer_failed_dlq_publish_leaves_unacknowledged():
    """Consumer does NOT acknowledge message if dead-letter queue routing fails."""
    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_client.is_closed = False
    mock_pull_res = MagicMock()
    mock_pull_res.status_code = 200
    mock_pull_res.json.return_value = {
        "receivedMessages": [
            {
                "ackId": "ack_corrupt_message",
                "message": {"data": "bm90LWpzb24=", "attributes": {}},
            }
        ]
    }
    # Pull succeeds, but DLQ publish returns 500 error!
    mock_dlq_res = MagicMock()
    mock_dlq_res.status_code = 500

    mock_client.post.side_effect = [mock_pull_res, mock_dlq_res]

    consumer = PubSubTelemetryConsumer(
        project_id="test-proj",
        subscription="sub-1",
        dead_letter_topic="dlq-topic",
        client=mock_client,
    )

    summary = await consumer.pull_and_process_batch(session_factory=None)
    assert summary.pulled == 1
    assert summary.dead_letter == 0
    assert summary.failed == 1

    # Client was called for pull (1) and DLQ publish (2), but NEVER for acknowledge!
    assert mock_client.post.await_count == 2
    for call in mock_client.post.await_args_list:
        assert "acknowledge" not in call.args[0]
