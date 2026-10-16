"""Pre-inference billing eligibility and wallet verification.

Security and Architectural Invariants:
- The Model Platform gateway operates under strict read-only access to billing data.
- Read-only queries against mp_active_api_keys_v project wallet_balance_usd and spend_limit_usd.
- Callers with non-positive wallet balances (balance <= 0.0) are rejected with HTTP 402.
- Callers whose wallet balance is insufficient for the conservative maximum request cost
  (including estimated prompt tokens and requested maximum output tokens) are rejected with HTTP 402.
- API key spend limits (ApiKey.spendLimitUsd) are verified prior to inference dispatch
  and tracked during concurrent in-flight execution.
- NON-ATOMIC PRE-CHECK NOTICE:
  This pre-inference eligibility check and in-memory tracker operate within a single
  process instance. It dampens local concurrent overspending, but CANNOT prevent
  concurrent overspending across multiple separate application instances (e.g. Cloud Run
  containers) without shared atomic database holds or a distributed coordination store.
- SPEND-LIMIT POLICY NOTICE:
  ApiKey.spendLimitUsd is an unmaterialized ceiling in the database. Because cumulative
  per-key spend is not tracked in the database schema across services, cumulative
  cross-service spend-limit enforcement is explicitly NOT implemented here pending an
  approved website schema and settlement contract. Single-request and concurrent in-flight
  spend limits are strictly enforced locally.
- Direct wallet balance mutations, transaction records, and deductions are strictly
  forbidden in this service and belong exclusively to the website billing domain.
"""

from __future__ import annotations

import logging
import uuid
from contextlib import asynccontextmanager
from decimal import Decimal
from typing import Annotated, AsyncGenerator

import anyio
from fastapi import Depends, HTTPException, Request, status

from app.auth import AuthenticatedPrincipal
from app.logging import log_event
from app.rate_limit import require_rate_limited_principal

logger = logging.getLogger(__name__)


async def require_billing_eligible_principal(
    request: Request,
    principal: Annotated[AuthenticatedPrincipal, Depends(require_rate_limited_principal)],
) -> AuthenticatedPrincipal:
    """FastAPI dependency that enforces positive wallet balance prior to inference.

    Fails closed with HTTP 402 Payment Required if:
    - principal.wallet_balance is less than or equal to 0.0 USD.

    Args:
        request: The incoming FastAPI request.
        principal: The authenticated principal resolved by require_valid_api_key.

    Returns:
        The validated AuthenticatedPrincipal.

    Raises:
        HTTPException:
            - 402 if wallet balance is non-positive.
    """
    if principal.wallet_balance <= Decimal("0.0"):
        log_event(
            logger,
            logging.WARNING,
            "billing.rejected.depleted_wallet",
            "Billing check failed: non-positive wallet balance",
            metadata={
                "path": request.url.path,
                "method": request.method,
                "balance_usd": str(principal.wallet_balance),
            },
        )
        raise HTTPException(
            status_code=status.HTTP_402_PAYMENT_REQUIRED,
            detail="Insufficient wallet balance. Please fund your account to continue.",
        )

    return principal


def verify_minimum_input_balance(
    principal: AuthenticatedPrincipal,
    estimated_input_cost: Decimal,
    request: Request,
) -> None:
    """Verify that available wallet balance covers the estimated minimum input cost.

    Non-atomic notice:
    This check verifies whether current balance can cover prompt ingestion.
    It does not guarantee against concurrent request exhaustion or output token cost.

    Args:
        principal: The authenticated caller.
        estimated_input_cost: Minimum cost calculated for prompt tokens.
        request: The incoming FastAPI request.

    Raises:
        HTTPException:
            - 402 if wallet balance is insufficient for estimated prompt cost.
    """
    if estimated_input_cost > Decimal("0.0") and principal.wallet_balance < estimated_input_cost:
        log_event(
            logger,
            logging.WARNING,
            "billing.rejected.insufficient_for_prompt",
            "Billing check failed: balance below estimated input cost",
            metadata={
                "path": request.url.path,
                "method": request.method,
                "balance_usd": str(principal.wallet_balance),
                "estimated_cost_usd": str(estimated_input_cost),
            },
        )
        raise HTTPException(
            status_code=status.HTTP_402_PAYMENT_REQUIRED,
            detail="Insufficient wallet balance for estimated request cost. Please fund your account to continue.",
        )


def verify_request_budget(
    principal: AuthenticatedPrincipal,
    max_cost: Decimal,
    request: Request | None = None,
) -> None:
    """Verify that available wallet balance and API key spend limit cover maximum request cost.

    Checks:
    1. Wallet balance must be greater than or equal to max_cost.
    2. If the API key has a configured spend_limit, max_cost must not exceed it.

    Args:
        principal: The authenticated caller.
        max_cost: Conservative maximum cost calculated for prompt + max output tokens.
        request: Optional incoming FastAPI request for structured logging.

    Raises:
        HTTPException:
            - 402 if wallet balance is insufficient for max_cost.
            - 402 if max_cost exceeds the API key's configured spend_limit.
    """
    path = request.url.path if request else "/v1/chat/completions"
    method = request.method if request else "POST"

    if max_cost > Decimal("0.0") and principal.wallet_balance < max_cost:
        log_event(
            logger,
            logging.WARNING,
            "billing.rejected.insufficient_for_request",
            "Billing check failed: balance below estimated maximum request cost",
            metadata={
                "path": path,
                "method": method,
                "balance_usd": str(principal.wallet_balance),
                "max_cost_usd": str(max_cost),
            },
        )
        raise HTTPException(
            status_code=status.HTTP_402_PAYMENT_REQUIRED,
            detail="Insufficient wallet balance for estimated request cost. Please fund your account to continue.",
        )

    if principal.spend_limit is not None and max_cost > principal.spend_limit:
        log_event(
            logger,
            logging.WARNING,
            "billing.rejected.spend_limit_exceeded",
            "Billing check failed: request cost exceeds API key spend limit",
            metadata={
                "path": path,
                "method": method,
                "spend_limit_usd": str(principal.spend_limit),
                "max_cost_usd": str(max_cost),
                "api_key_id": principal.api_key_id or "",
            },
        )
        raise HTTPException(
            status_code=status.HTTP_402_PAYMENT_REQUIRED,
            detail="Spend limit reached for this API key. Request cost exceeds configured spend limit.",
        )


class SpendReservation:
    """Handle representing an active in-flight cost reservation."""

    def __init__(
        self,
        reservation_id: str,
        user_id: str,
        reserved_cost: Decimal,
        api_key_id: str | None = None,
        tracker: InFlightSpendTracker | None = None,
    ) -> None:
        self.reservation_id = reservation_id
        self.user_id = user_id
        self.api_key_id = api_key_id
        self.reserved_cost = reserved_cost
        self.current_hold = reserved_cost
        self.actual_cost: Decimal | None = None
        self.status = "ACTIVE"  # "ACTIVE", "RECONCILED", "RELEASED"
        self._tracker = tracker

    async def reconcile(self, actual_cost: Decimal) -> None:
        """Reconcile reservation against actual inference cost."""
        if self._tracker is not None:
            await self._tracker.reconcile_reservation(self, actual_cost)
        else:
            self.status = "RECONCILED"
            self.actual_cost = actual_cost
            self.current_hold = actual_cost

    async def release(self) -> None:
        """Release reservation safely and idempotently."""
        if self._tracker is not None:
            await self._tracker.release_reservation(self)
        else:
            self.status = "RELEASED"
            self.current_hold = Decimal("0.0")


class InFlightSpendTracker:
    """Local, single-process in-flight spend tracker with reconciliation and spend-limit protection.

    Operational Boundary and Scope:
    - STRICTLY SINGLE-PROCESS: This tracker operates entirely in-memory within a single
      Python worker. It does NOT provide distributed, cross-container, or production
      wallet protection on multi-instance Cloud Run deployments.
    - RECONCILIATION & SAFE RELEASE: Reserves conservative maximum request cost, reconciles
      actual cost post-inference to immediately release unspent budget, and guarantees
      idempotent, single-release execution on all error and cancellation paths.
    - WALLET & SPEND-LIMIT CHECKS: Prevents concurrent requests from reserving or spending
      the same available budget or exceeding API key spend limits.
    - NO INDEPENDENT CHARGING: This class does NOT mutate database wallet balances or
      execute financial transactions. Authoritative wallet balance deductions belong to
      the website billing settlement domain.
    """

    def __init__(self) -> None:
        self._lock = anyio.Lock()
        self._user_in_flight: dict[str, Decimal] = {}
        self._key_in_flight: dict[str, Decimal] = {}
        self._reservations: dict[str, SpendReservation] = {}

    @property
    def _in_flight(self) -> dict[str, Decimal]:
        """Backward compatibility alias for _user_in_flight."""
        return self._user_in_flight

    async def reserve(
        self,
        user_id: str,
        balance: Decimal,
        estimated_cost: Decimal,
        *,
        api_key_id: str | None = None,
        spend_limit: Decimal | None = None,
        reservation_id: str | None = None,
    ) -> SpendReservation:
        """Reserve cost against user balance and optional API key spend limit.

        Raises:
            HTTPException: 402 if balance minus current in-flight spend is insufficient.
            HTTPException: 402 if API key in-flight spend plus cost exceeds spend limit.
        """
        if estimated_cost < Decimal("0.0"):
            raise ValueError("Cost to reserve cannot be negative.")

        res_id = reservation_id or f"res_{uuid.uuid4().hex}"

        async with self._lock:
            # 1. Check wallet balance concurrency
            current_user = self._user_in_flight.get(user_id, Decimal("0.0"))
            effective_available = balance - current_user
            if effective_available < estimated_cost:
                log_event(
                    logger,
                    logging.WARNING,
                    "billing.rejected.concurrent_in_flight",
                    "Local concurrency check failed: in-flight spend exceeds balance",
                    metadata={
                        "user_id": user_id,
                        "balance_usd": str(balance),
                        "in_flight_usd": str(current_user),
                        "estimated_cost_usd": str(estimated_cost),
                    },
                )
                raise HTTPException(
                    status_code=status.HTTP_402_PAYMENT_REQUIRED,
                    detail="Insufficient wallet balance for concurrent requests. Please fund your account to continue.",
                )

            # 2. Check API key spend limit concurrency
            if api_key_id and spend_limit is not None:
                current_key = self._key_in_flight.get(api_key_id, Decimal("0.0"))
                if current_key + estimated_cost > spend_limit:
                    log_event(
                        logger,
                        logging.WARNING,
                        "billing.rejected.concurrent_spend_limit",
                        "Local concurrency check failed: in-flight spend exceeds key spend limit",
                        metadata={
                            "api_key_id": api_key_id,
                            "spend_limit_usd": str(spend_limit),
                            "key_in_flight_usd": str(current_key),
                            "estimated_cost_usd": str(estimated_cost),
                        },
                    )
                    raise HTTPException(
                        status_code=status.HTTP_402_PAYMENT_REQUIRED,
                        detail="Spend limit reached for this API key. Concurrent requests exceed configured spend limit.",
                    )

            # Record reservation
            self._user_in_flight[user_id] = current_user + estimated_cost
            if api_key_id:
                current_key = self._key_in_flight.get(api_key_id, Decimal("0.0"))
                self._key_in_flight[api_key_id] = current_key + estimated_cost

            reservation = SpendReservation(
                reservation_id=res_id,
                user_id=user_id,
                reserved_cost=estimated_cost,
                api_key_id=api_key_id,
                tracker=self,
            )
            self._reservations[res_id] = reservation
            return reservation

    async def reconcile_reservation(self, reservation: SpendReservation, actual_cost: Decimal) -> None:
        """Reconcile reserved cost to actual cost, immediately returning unspent funds."""
        async with self._lock:
            if reservation.status != "ACTIVE":
                return  # Idempotent: already reconciled or released

            if actual_cost < Decimal("0.0"):
                raise ValueError("Actual cost cannot be negative.")

            delta = reservation.current_hold - actual_cost

            # Adjust user in-flight spend
            curr_user = self._user_in_flight.get(reservation.user_id, Decimal("0.0"))
            new_user = curr_user - delta
            if new_user <= Decimal("0.0"):
                self._user_in_flight.pop(reservation.user_id, None)
            else:
                self._user_in_flight[reservation.user_id] = new_user

            # Adjust key in-flight spend if applicable
            if reservation.api_key_id:
                curr_key = self._key_in_flight.get(reservation.api_key_id, Decimal("0.0"))
                new_key = curr_key - delta
                if new_key <= Decimal("0.0"):
                    self._key_in_flight.pop(reservation.api_key_id, None)
                else:
                    self._key_in_flight[reservation.api_key_id] = new_key

            reservation.actual_cost = actual_cost
            reservation.current_hold = actual_cost
            reservation.status = "RECONCILED"

    async def release_reservation(self, reservation: SpendReservation) -> None:
        """Release reservation safely and idempotently."""
        async with self._lock:
            if reservation.status == "RELEASED":
                return  # Idempotent: already released, do not release twice

            amount = reservation.current_hold
            if amount > Decimal("0.0"):
                curr_user = self._user_in_flight.get(reservation.user_id, Decimal("0.0"))
                rem_user = curr_user - amount
                if rem_user <= Decimal("0.0"):
                    self._user_in_flight.pop(reservation.user_id, None)
                else:
                    self._user_in_flight[reservation.user_id] = rem_user

                if reservation.api_key_id:
                    curr_key = self._key_in_flight.get(reservation.api_key_id, Decimal("0.0"))
                    rem_key = curr_key - amount
                    if rem_key <= Decimal("0.0"):
                        self._key_in_flight.pop(reservation.api_key_id, None)
                    else:
                        self._key_in_flight[reservation.api_key_id] = rem_key

            reservation.current_hold = Decimal("0.0")
            reservation.status = "RELEASED"
            self._reservations.pop(reservation.reservation_id, None)

    async def release(
        self,
        user_id: str,
        estimated_cost: Decimal,
        api_key_id: str | None = None,
    ) -> None:
        """Legacy release method for backward compatibility."""
        async with self._lock:
            current = self._user_in_flight.get(user_id, Decimal("0.0"))
            remaining = current - estimated_cost
            if remaining <= Decimal("0.0"):
                self._user_in_flight.pop(user_id, None)
            else:
                self._user_in_flight[user_id] = remaining

            if api_key_id:
                curr_key = self._key_in_flight.get(api_key_id, Decimal("0.0"))
                rem_key = curr_key - estimated_cost
                if rem_key <= Decimal("0.0"):
                    self._key_in_flight.pop(api_key_id, None)
                else:
                    self._key_in_flight[api_key_id] = rem_key

    async def get_in_flight_spend(self, user_id: str) -> Decimal:
        """Return the current active in-flight spend tracked for a user."""
        async with self._lock:
            return self._user_in_flight.get(user_id, Decimal("0.0"))

    async def get_key_in_flight_spend(self, api_key_id: str) -> Decimal:
        """Return the current active in-flight spend tracked for an API key."""
        async with self._lock:
            return self._key_in_flight.get(api_key_id, Decimal("0.0"))


@asynccontextmanager
async def track_in_flight_spend(
    tracker: InFlightSpendTracker,
    user_id: str,
    balance: Decimal,
    estimated_cost: Decimal | None = None,
    *,
    max_cost: Decimal | None = None,
    api_key_id: str | None = None,
    spend_limit: Decimal | None = None,
    reservation_id: str | None = None,
) -> AsyncGenerator[SpendReservation, None]:
    """Async context manager for cancellation-safe in-flight spend tracking and reconciliation.

    Guarantees:
    - Reserves conservative request cost (prompt + max output tokens) prior to inference.
    - Enforces both user wallet balance and API key spend limit concurrency bounds.
    - Yields a SpendReservation handle supporting early reconciliation of unspent tokens.
    - Releases reservation exactly once in finally block.
    - Shields release from cancellation so task cancellation does not leak reservations.
    - Preserves and propagates any original exception without suppression.
    """
    cost_to_reserve = max_cost if max_cost is not None else (estimated_cost or Decimal("0.0"))

    if cost_to_reserve <= Decimal("0.0"):
        dummy = SpendReservation(
            reservation_id=reservation_id or "res_zero",
            user_id=user_id,
            reserved_cost=Decimal("0.0"),
            api_key_id=api_key_id,
            tracker=None,
        )
        dummy.status = "RELEASED"
        yield dummy
        return

    reservation = await tracker.reserve(
        user_id=user_id,
        balance=balance,
        estimated_cost=cost_to_reserve,
        api_key_id=api_key_id,
        spend_limit=spend_limit,
        reservation_id=reservation_id,
    )
    try:
        yield reservation
    finally:
        # Shield cleanup from cancellation so task cancellation does not abort lock acquisition or release
        with anyio.CancelScope(shield=True):
            await reservation.release()
