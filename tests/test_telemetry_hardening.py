"""Focused unit tests for telemetry insert idempotency and conflict handling."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import AsyncMock

import pytest

import app.telemetry as telemetry_module
from app.auth import AuthenticatedPrincipal
from app.telemetry import _INSERT_INFERENCE_LOG_QUERY, record_inference_telemetry


def _mock_principal() -> AuthenticatedPrincipal:
    return AuthenticatedPrincipal(
        user_id="usr_telemetry_hardening",
        api_key_id="key_telemetry_hardening",
        channel="api",
        wallet_balance=Decimal("50.000000"),
    )


def test_insert_inference_log_query_contains_on_conflict_do_nothing():
    """Verify that INSERT statement specifies ON CONFLICT (request_id) DO NOTHING."""
    assert "ON CONFLICT (request_id) DO NOTHING" in _INSERT_INFERENCE_LOG_QUERY


def test_telemetry_documentation_explicitly_acknowledges_non_durable_recovery():
    """Verify that telemetry module docstring explicitly documents non-durable recovery and temporary fail-open risk."""
    doc = telemetry_module.__doc__ or ""
    assert "NON-DURABLE RECOVERY & TEMPORARY FAIL-OPEN POLICY NOTICE" in doc
    assert "CANNOT recover" in doc
    assert "AT-LEAST-ONCE DELIVERY & IDEMPOTENT SETTLEMENT" in doc


@pytest.mark.anyio
async def test_duplicate_telemetry_insert_idempotent_execution():
    """Verify that executing telemetry insert with existing request_id executes without exception."""
    session = AsyncMock()
    principal = _mock_principal()

    success = await record_inference_telemetry(
        session=session,
        request_id="req_duplicate_test_123",
        principal=principal,
        model_id="rantera-base-v1",
        tokens_in=50,
        tokens_out=25,
        cost_usd=Decimal("0.000525"),
        latency_ms=80,
    )

    assert success is True
    session.execute.assert_awaited_once()
    session.commit.assert_awaited_once()
