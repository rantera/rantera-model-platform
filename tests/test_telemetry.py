"""Focused unit tests for inference telemetry logging into mp_inference_logs."""

from __future__ import annotations

import logging
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

import pytest

import app.telemetry as telemetry_module
from app.auth import AuthenticatedPrincipal
from app.db import validate_query_safety
from app.telemetry import (
    _INSERT_INFERENCE_LOG_QUERY,
    record_inference_telemetry,
)


def _mock_principal() -> AuthenticatedPrincipal:
    return AuthenticatedPrincipal(
        user_id="usr_telemetry_test",
        api_key_id="key_telemetry_test",
        channel="api",
        wallet_balance=Decimal("25.000000"),
    )


# ── Query Structure and Safety Tests ─────────────────────────


def test_insert_inference_log_query_passes_safety_validation():
    """Telemetry query must pass validate_query_safety (no forbidden website tables, no DDL)."""
    validate_query_safety(_INSERT_INFERENCE_LOG_QUERY)


def test_insert_inference_log_query_targets_mp_inference_logs():
    """Query targets mp_inference_logs and sets settlement_status to :settlement_status."""
    assert "INSERT INTO mp_inference_logs" in _INSERT_INFERENCE_LOG_QUERY
    assert ":settlement_status" in _INSERT_INFERENCE_LOG_QUERY
    assert ":settlement_attempts" in _INSERT_INFERENCE_LOG_QUERY


def test_insert_inference_log_query_omits_settlement_owned_columns():
    """FastAPI never writes settlement_id or settled_at during initial inference logging."""
    # Columns in the INSERT statement
    insert_cols_section = _INSERT_INFERENCE_LOG_QUERY.split("VALUES")[0]
    assert "settlement_id" not in insert_cols_section
    assert "settled_at" not in insert_cols_section


# ── Execution and Parameter Binding Tests ─────────────────────


@pytest.mark.anyio
async def test_record_inference_telemetry_success():
    """Verify successful telemetry recording commits and binds all required fields."""
    session = AsyncMock()
    principal = _mock_principal()

    success = await record_inference_telemetry(
        session=session,
        request_id="req_test_12345",
        principal=principal,
        model_id="rantera-base-v1",
        tokens_in=100,
        tokens_out=50,
        cost_usd=Decimal("0.001050"),
        latency_ms=120,
        request_status="SUCCESS",
    )

    assert success is True
    session.execute.assert_awaited_once()
    session.commit.assert_awaited_once()

    call_args = session.execute.await_args
    params = call_args[0][1]

    assert params["request_id"] == "req_test_12345"
    assert params["user_id"] == "usr_telemetry_test"
    assert params["api_key_id"] == "key_telemetry_test"
    assert params["model_id"] == "rantera-base-v1"
    assert params["channel"] == "api"
    assert params["tokens_in"] == 100
    assert params["tokens_out"] == 50
    assert params["total_tokens"] == 150
    assert params["cost_usd"] == Decimal("0.001050")
    assert params["latency_ms"] == 120
    assert params["request_status"] == "SUCCESS"
    assert params["settlement_status"] == "PENDING"
    assert params["settlement_attempts"] == 0
    assert params["id"].startswith("inf_")


# ── Fault Tolerance and Logging Tests ────────────────────────


@pytest.mark.anyio
async def test_record_inference_telemetry_session_none_logs_emergency(caplog):
    """When DB session is None, telemetry returns False and logs an ERROR event without misleading EMERGENCY prefix."""
    principal = _mock_principal()

    with caplog.at_level(logging.ERROR):
        success = await record_inference_telemetry(
            session=None,
            request_id="req_no_session_999",
            principal=principal,
            model_id="rantera-base-v1",
            tokens_in=20,
            tokens_out=10,
            cost_usd=Decimal("0.000210"),
            latency_ms=45,
        )

    assert success is False
    assert any(
        rec.levelno == logging.ERROR
        and getattr(rec, "event", None) == "telemetry.session_none"
        and "req_no_session_999" in getattr(rec, "extra", {}).get("request_id", str(rec.__dict__))
        for rec in caplog.records
    )
    assert not any("EMERGENCY" in rec.message for rec in caplog.records)


@pytest.mark.anyio
async def test_record_inference_telemetry_database_exception_fails_open(caplog):
    """When DB execution throws an error, return False and do not raise."""
    session = AsyncMock()
    session.execute.side_effect = RuntimeError("Database connection reset by peer")
    principal = _mock_principal()

    with caplog.at_level(logging.ERROR):
        success = await record_inference_telemetry(
            session=session,
            request_id="req_db_error_888",
            principal=principal,
            model_id="rantera-base-v1",
            tokens_in=30,
            tokens_out=15,
            cost_usd=Decimal("0.000315"),
            latency_ms=60,
        )

    assert success is False
    assert any("EMERGENCY: Failed to insert inference telemetry" in rec.message for rec in caplog.records)


def test_telemetry_documentation_reflects_non_durable_recovery():
    """Verify that telemetry module docstring explicitly documents non-durable recovery."""
    doc = telemetry_module.__doc__ or ""
    assert "FAULT TOLERANCE & NON-DURABLE RECOVERY NOTICE" in doc
    assert "APPLICATION LOGGING ALONE IS NOT DURABLE RECOVERY" in doc
    assert "PENDING" in doc
