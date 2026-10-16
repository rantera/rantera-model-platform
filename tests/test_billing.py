"""Focused unit tests for pre-inference billing eligibility and wallet checks."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException

import app.billing as billing_module
from app.auth import AuthenticatedPrincipal
from app.billing import (
    require_billing_eligible_principal,
    verify_minimum_input_balance,
    verify_request_budget,
)


def _mock_request(path: str = "/v1/chat/completions") -> MagicMock:
    req = MagicMock()
    req.url.path = path
    return req


# ── Wallet Balance Pre-Check Tests ────────────────────────────


@pytest.mark.anyio
async def test_require_billing_eligible_principal_positive_balance():
    """Principal with positive balance passes the billing eligibility gate."""
    principal = AuthenticatedPrincipal(
        user_id="usr_valid_123",
        wallet_balance=Decimal("10.500000"),
    )
    req = _mock_request()

    res = await require_billing_eligible_principal(request=req, principal=principal)
    assert res is principal
    assert res.wallet_balance == Decimal("10.500000")


@pytest.mark.anyio
async def test_require_billing_eligible_principal_zero_balance_rejected():
    """Principal with exactly zero balance is rejected with HTTP 402."""
    principal = AuthenticatedPrincipal(
        user_id="usr_zero_123",
        wallet_balance=Decimal("0.0"),
    )
    req = _mock_request()

    with pytest.raises(HTTPException) as exc_info:
        await require_billing_eligible_principal(request=req, principal=principal)

    assert exc_info.value.status_code == 402
    assert "Insufficient wallet balance" in exc_info.value.detail


@pytest.mark.anyio
async def test_require_billing_eligible_principal_negative_balance_rejected():
    """Principal with negative balance is rejected with HTTP 402."""
    principal = AuthenticatedPrincipal(
        user_id="usr_debtor_123",
        wallet_balance=Decimal("-0.050000"),
    )
    req = _mock_request()

    with pytest.raises(HTTPException) as exc_info:
        await require_billing_eligible_principal(request=req, principal=principal)

    assert exc_info.value.status_code == 402
    assert "Insufficient wallet balance" in exc_info.value.detail


# ── Estimated Input Cost vs Wallet Balance Tests ──────────────


def test_verify_minimum_input_balance_covers_prompt():
    """Balance greater than or equal to estimated prompt cost passes."""
    principal = AuthenticatedPrincipal(
        user_id="usr_valid_456",
        wallet_balance=Decimal("0.050000"),
    )
    req = _mock_request()

    # Required cost is less than balance
    verify_minimum_input_balance(
        principal=principal,
        estimated_input_cost=Decimal("0.003000"),
        request=req,
    )

    # Required cost is exactly equal to balance
    verify_minimum_input_balance(
        principal=principal,
        estimated_input_cost=Decimal("0.050000"),
        request=req,
    )


def test_verify_minimum_input_balance_exceeds_available_balance():
    """When estimated minimum prompt cost exceeds wallet balance, fail with HTTP 402."""
    principal = AuthenticatedPrincipal(
        user_id="usr_valid_789",
        wallet_balance=Decimal("0.001000"),
    )
    req = _mock_request()

    with pytest.raises(HTTPException) as exc_info:
        verify_minimum_input_balance(
            principal=principal,
            estimated_input_cost=Decimal("0.003000"),
            request=req,
        )

    assert exc_info.value.status_code == 402
    assert "Insufficient wallet balance for estimated request cost" in exc_info.value.detail


def test_verify_minimum_input_balance_zero_cost_passes():
    """Zero estimated input cost does not trigger rejection."""
    principal = AuthenticatedPrincipal(
        user_id="usr_valid_000",
        wallet_balance=Decimal("0.000010"),
    )
    req = _mock_request()

    verify_minimum_input_balance(
        principal=principal,
        estimated_input_cost=Decimal("0.0"),
        request=req,
    )


# ── Conservative Maximum Request Budget & Spend Limit Tests ────


def test_verify_request_budget_sufficient_balance_and_limit():
    """Request passes when balance and spend limit cover conservative max request cost."""
    principal = AuthenticatedPrincipal(
        user_id="usr_valid_budget_1",
        api_key_id="key_valid_budget_1",
        wallet_balance=Decimal("10.000000"),
        spend_limit=Decimal("5.0000"),
    )
    req = _mock_request()

    # Max cost $0.061740 is well below $10.00 balance and $5.00 spend limit
    verify_request_budget(
        principal=principal,
        max_cost=Decimal("0.061740"),
        request=req,
    )


def test_verify_request_budget_insufficient_for_max_output_tokens():
    """When wallet balance is sufficient for prompt but insufficient for requested output tokens, reject 402."""
    # User has $0.001000 balance (enough for a $0.000300 prompt, but not for $0.061440 completion)
    principal = AuthenticatedPrincipal(
        user_id="usr_low_budget_2",
        wallet_balance=Decimal("0.001000"),
    )
    req = _mock_request()

    with pytest.raises(HTTPException) as exc_info:
        verify_request_budget(
            principal=principal,
            max_cost=Decimal("0.061740"),
            request=req,
        )

    assert exc_info.value.status_code == 402
    assert "Insufficient wallet balance for estimated request cost" in exc_info.value.detail


def test_verify_request_budget_exceeds_api_key_spend_limit():
    """When max request cost exceeds the API key's configured spend limit, reject with 402."""
    principal = AuthenticatedPrincipal(
        user_id="usr_valid_user_3",
        api_key_id="key_limited_3",
        wallet_balance=Decimal("100.000000"),
        spend_limit=Decimal("0.0500"),  # Spend limit is $0.05
    )
    req = _mock_request()

    # Max request cost is $0.061740 > $0.050000
    with pytest.raises(HTTPException) as exc_info:
        verify_request_budget(
            principal=principal,
            max_cost=Decimal("0.061740"),
            request=req,
        )

    assert exc_info.value.status_code == 402
    assert "Spend limit reached" in exc_info.value.detail


def test_verify_request_budget_zero_cost_passes():
    """Zero max request cost passes without deducting or blocking."""
    principal = AuthenticatedPrincipal(
        user_id="usr_valid_user_4",
        wallet_balance=Decimal("0.000010"),
        spend_limit=Decimal("0.0000"),
    )
    req = _mock_request()

    verify_request_budget(
        principal=principal,
        max_cost=Decimal("0.0"),
        request=req,
    )


# ── Architectural Documentation Invariants ───────────────────


def test_billing_documentation_reflects_non_atomic_and_spend_limit_decisions():
    """Verify that billing module explicitly documents non-atomic concurrency and spend limits."""
    doc = billing_module.__doc__ or ""
    assert "NON-ATOMIC PRE-CHECK NOTICE" in doc
    assert "concurrent overspending" in doc
    assert "SPEND-LIMIT POLICY NOTICE" in doc
    assert "cumulative" in doc.lower()
