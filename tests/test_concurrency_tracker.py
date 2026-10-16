"""Focused unit tests for local single-process in-flight spend tracking and cancellation safety."""

from __future__ import annotations

from decimal import Decimal

import anyio
import pytest
from fastapi import HTTPException

from app.billing import InFlightSpendTracker, track_in_flight_spend


@pytest.mark.anyio
async def test_in_flight_spend_reserves_and_releases():
    """Verify single-process tracker increments and decrements in-flight spend."""
    tracker = InFlightSpendTracker()
    user_id = "usr_concurrent_1"
    balance = Decimal("1.000000")
    cost = Decimal("0.300000")

    await tracker.reserve(user_id=user_id, balance=balance, estimated_cost=cost)
    assert await tracker.get_in_flight_spend(user_id) == cost

    await tracker.release(user_id=user_id, estimated_cost=cost)
    assert await tracker.get_in_flight_spend(user_id) == Decimal("0.0")


@pytest.mark.anyio
async def test_in_flight_spend_blocks_concurrent_exhaustion():
    """When concurrent requests exceed effective available balance, reject with 402."""
    tracker = InFlightSpendTracker()
    user_id = "usr_concurrent_2"
    balance = Decimal("0.050000")
    req_cost = Decimal("0.030000")

    # First request: 0.05 - 0.0 >= 0.03 -> OK
    await tracker.reserve(user_id=user_id, balance=balance, estimated_cost=req_cost)

    # Second concurrent request: 0.05 - 0.03 = 0.02 < 0.03 -> 402
    with pytest.raises(HTTPException) as exc_info:
        await tracker.reserve(user_id=user_id, balance=balance, estimated_cost=req_cost)

    assert exc_info.value.status_code == 402
    assert "concurrent requests" in exc_info.value.detail

    # Clean up first request
    await tracker.release(user_id=user_id, estimated_cost=req_cost)
    assert await tracker.get_in_flight_spend(user_id) == Decimal("0.0")


@pytest.mark.anyio
async def test_track_in_flight_spend_context_manager_releases_on_success():
    """Context manager reserves upon entry and releases on exit."""
    tracker = InFlightSpendTracker()
    user_id = "usr_concurrent_3"
    balance = Decimal("10.000000")
    cost = Decimal("0.500000")

    async with track_in_flight_spend(tracker=tracker, user_id=user_id, balance=balance, estimated_cost=cost):
        assert await tracker.get_in_flight_spend(user_id) == cost

    assert await tracker.get_in_flight_spend(user_id) == Decimal("0.0")


@pytest.mark.anyio
async def test_track_in_flight_spend_context_manager_releases_on_exception():
    """Context manager releases reservation even if an unexpected exception occurs."""
    tracker = InFlightSpendTracker()
    user_id = "usr_concurrent_4"
    balance = Decimal("10.000000")
    cost = Decimal("0.500000")

    with pytest.raises(RuntimeError, match="Test failure inside inference block"):
        async with track_in_flight_spend(tracker=tracker, user_id=user_id, balance=balance, estimated_cost=cost):
            assert await tracker.get_in_flight_spend(user_id) == cost
            raise RuntimeError("Test failure inside inference block")

    # Must be released after exception
    assert await tracker.get_in_flight_spend(user_id) == Decimal("0.0")


@pytest.mark.anyio
async def test_track_in_flight_spend_cancellation_safe():
    """Simulate task cancellation and verify release occurs via shielded cancel scope."""
    tracker = InFlightSpendTracker()
    user_id = "usr_concurrent_5"
    balance = Decimal("10.000000")
    cost = Decimal("0.500000")

    with pytest.raises(TimeoutError):
        with anyio.fail_after(0.02):
            async with track_in_flight_spend(tracker=tracker, user_id=user_id, balance=balance, estimated_cost=cost):
                assert await tracker.get_in_flight_spend(user_id) == cost
                await anyio.sleep(10)

    # Shielded release must have cleared in-flight spend
    assert await tracker.get_in_flight_spend(user_id) == Decimal("0.0")


@pytest.mark.anyio
async def test_in_flight_spend_blocks_concurrent_api_key_spend_limit():
    """When concurrent requests exceed API key configured spend limit, reject with 402."""
    tracker = InFlightSpendTracker()
    user_id = "usr_key_concurrent_1"
    api_key_id = "key_concurrent_limit_1"
    balance = Decimal("100.000000")  # User has plenty of balance
    spend_limit = Decimal("0.1000")  # Key is limited to $0.10
    cost = Decimal("0.060000")

    # First request: 0.0 + 0.06 <= 0.10 -> OK
    res1 = await tracker.reserve(
        user_id=user_id,
        balance=balance,
        estimated_cost=cost,
        api_key_id=api_key_id,
        spend_limit=spend_limit,
    )
    assert await tracker.get_key_in_flight_spend(api_key_id) == cost

    # Second concurrent request on same key: 0.06 + 0.06 = 0.12 > 0.10 -> 402
    with pytest.raises(HTTPException) as exc_info:
        await tracker.reserve(
            user_id=user_id,
            balance=balance,
            estimated_cost=cost,
            api_key_id=api_key_id,
            spend_limit=spend_limit,
        )

    assert exc_info.value.status_code == 402
    assert "Spend limit reached" in exc_info.value.detail

    # Clean up first request
    await res1.release()
    assert await tracker.get_key_in_flight_spend(api_key_id) == Decimal("0.0")
    assert await tracker.get_in_flight_spend(user_id) == Decimal("0.0")


@pytest.mark.anyio
async def test_reservation_reconciliation_frees_unspent_budget():
    """Reconciling reserved cost to actual cost immediately frees unspent budget for waiting requests."""
    tracker = InFlightSpendTracker()
    user_id = "usr_reconcile_1"
    balance = Decimal("0.100000")  # $0.10 balance
    reserved_cost = Decimal("0.061740")

    # Request 1 reserves conservative max cost
    res1 = await tracker.reserve(
        user_id=user_id,
        balance=balance,
        estimated_cost=reserved_cost,
    )
    assert await tracker.get_in_flight_spend(user_id) == reserved_cost

    # Request 2 arrives: balance $0.10 - $0.061740 = $0.038260 < $0.061740 -> blocked!
    with pytest.raises(HTTPException) as exc_info:
        await tracker.reserve(
            user_id=user_id,
            balance=balance,
            estimated_cost=reserved_cost,
        )
    assert exc_info.value.status_code == 402

    # Request 1 completes inference: actual tokens cost only $0.002000!
    actual_cost = Decimal("0.002000")
    await res1.reconcile(actual_cost=actual_cost)

    # In-flight hold is now reduced to actual cost ($0.002000)
    assert await tracker.get_in_flight_spend(user_id) == actual_cost

    # Request 2 retries: balance $0.10 - $0.002000 = $0.098000 >= $0.061740 -> now succeeds!
    res2 = await tracker.reserve(
        user_id=user_id,
        balance=balance,
        estimated_cost=reserved_cost,
    )
    assert await tracker.get_in_flight_spend(user_id) == actual_cost + reserved_cost

    # Clean up both
    await res1.release()
    assert await tracker.get_in_flight_spend(user_id) == reserved_cost
    await res2.release()
    assert await tracker.get_in_flight_spend(user_id) == Decimal("0.0")


@pytest.mark.anyio
async def test_track_in_flight_spend_releases_on_provider_failure():
    """Verify context manager releases reservation on provider exception."""
    tracker = InFlightSpendTracker()
    user_id = "usr_fail_1"
    api_key_id = "key_fail_1"
    balance = Decimal("10.000000")
    cost = Decimal("0.061740")

    with pytest.raises(ConnectionError, match="Vertex connection timeout"):
        async with track_in_flight_spend(
            tracker=tracker,
            user_id=user_id,
            balance=balance,
            max_cost=cost,
            api_key_id=api_key_id,
            spend_limit=Decimal("1.0000"),
        ) as reservation:
            assert await tracker.get_in_flight_spend(user_id) == cost
            assert await tracker.get_key_in_flight_spend(api_key_id) == cost
            assert reservation.status == "ACTIVE"
            raise ConnectionError("Vertex connection timeout")

    # Entire reservation safely released on failure
    assert await tracker.get_in_flight_spend(user_id) == Decimal("0.0")
    assert await tracker.get_key_in_flight_spend(api_key_id) == Decimal("0.0")


@pytest.mark.anyio
async def test_reservation_idempotent_double_release():
    """Calling release multiple times is idempotent and does not corrupt tracker balance."""
    tracker = InFlightSpendTracker()
    user_id = "usr_double_release"
    api_key_id = "key_double_release"
    cost = Decimal("0.050000")

    reservation = await tracker.reserve(
        user_id=user_id,
        balance=Decimal("1.000000"),
        estimated_cost=cost,
        api_key_id=api_key_id,
    )
    assert await tracker.get_in_flight_spend(user_id) == cost

    # First release
    await reservation.release()
    assert await tracker.get_in_flight_spend(user_id) == Decimal("0.0")
    assert await tracker.get_key_in_flight_spend(api_key_id) == Decimal("0.0")

    # Second release (must be safe no-op)
    await reservation.release()
    assert await tracker.get_in_flight_spend(user_id) == Decimal("0.0")
    assert await tracker.get_key_in_flight_spend(api_key_id) == Decimal("0.0")


@pytest.mark.anyio
async def test_reservation_reconcile_after_release_is_noop():
    """Reconciling a reservation that was already released is an idempotent no-op."""
    tracker = InFlightSpendTracker()
    user_id = "usr_reconcile_after_rel"
    cost = Decimal("0.050000")

    reservation = await tracker.reserve(
        user_id=user_id,
        balance=Decimal("1.000000"),
        estimated_cost=cost,
    )
    await reservation.release()
    assert await tracker.get_in_flight_spend(user_id) == Decimal("0.0")

    # Calling reconcile after release should not resurrect or alter state
    await reservation.reconcile(actual_cost=Decimal("0.010000"))
    assert await tracker.get_in_flight_spend(user_id) == Decimal("0.0")
