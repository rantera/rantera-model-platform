"""Focused unit tests for model pricing and exact Decimal arithmetic."""

from __future__ import annotations

from decimal import Decimal

from app.models import (
    RegisteredModel,
    calculate_inference_cost,
    calculate_max_request_cost,
    estimate_input_cost,
    get_model,
)


def test_rantera_base_v1_pricing_and_limits():
    """Verify registered model attributes match catalog contract."""
    model = get_model("rantera-base-v1")
    assert model is not None
    assert model.display_name == "Rantera Base v1"
    assert model.context_window == 8192
    assert model.max_output_tokens == 4096
    assert model.input_cost_per_1k == Decimal("0.003000")
    assert model.output_cost_per_1k == Decimal("0.015000")
    assert model.is_active is True


def test_calculate_inference_cost_zero_tokens():
    """Zero tokens in and out should result in exactly 0.000000 USD."""
    model = get_model("rantera-base-v1")
    assert model is not None

    cost = calculate_inference_cost(model, tokens_in=0, tokens_out=0)
    assert cost == Decimal("0.000000")


def test_calculate_inference_cost_standard_rates():
    """Verify standard rates: $0.003/1k input, $0.015/1k output."""
    model = get_model("rantera-base-v1")
    assert model is not None

    # 1000 in, 0 out = 0.003000
    cost_in_only = calculate_inference_cost(model, tokens_in=1000, tokens_out=0)
    assert cost_in_only == Decimal("0.003000")

    # 0 in, 1000 out = 0.015000
    cost_out_only = calculate_inference_cost(model, tokens_in=0, tokens_out=1000)
    assert cost_out_only == Decimal("0.015000")

    # 1000 in, 1000 out = 0.018000
    cost_both = calculate_inference_cost(model, tokens_in=1000, tokens_out=1000)
    assert cost_both == Decimal("0.018000")

    # 500 in, 200 out: (500*0.003 + 200*0.015)/1000 = (1.5 + 3.0)/1000 = 0.004500
    cost_mixed = calculate_inference_cost(model, tokens_in=500, tokens_out=200)
    assert cost_mixed == Decimal("0.004500")


def test_calculate_inference_cost_rounding_half_up():
    """Verify 6-decimal rounding follows ROUND_HALF_UP."""
    model = RegisteredModel(
        model_id="test-custom-rates",
        description="Model with fractional cost for testing rounding",
        input_cost_per_1k=Decimal("0.001000"),
        output_cost_per_1k=Decimal("0.001000"),
    )

    # 1 token at $0.001 per 1000 = 0.000001 USD
    cost_1 = calculate_inference_cost(model, tokens_in=1, tokens_out=0)
    assert cost_1 == Decimal("0.000001")

    # Test half-up rounding: custom model where 1 token = 0.0000005 USD
    half_model = RegisteredModel(
        model_id="test-half-rounding",
        description="Model producing half-cent decimals",
        input_cost_per_1k=Decimal("0.000500"),
        output_cost_per_1k=Decimal("0.000500"),
    )
    # 1 token * 0.000500 / 1000 = 0.0000005 -> rounds up to 0.000001
    cost_half = calculate_inference_cost(half_model, tokens_in=1, tokens_out=0)
    assert cost_half == Decimal("0.000001")


def test_estimate_input_cost():
    """Verify estimate_input_cost calculates prompt cost accurately."""
    model = get_model("rantera-base-v1")
    assert model is not None

    # 1000 input tokens = 0.003000 USD
    est = estimate_input_cost(model, tokens_in=1000)
    assert est == Decimal("0.003000")

    # 250 input tokens = 0.000750 USD
    est_250 = estimate_input_cost(model, tokens_in=250)
    assert est_250 == Decimal("0.000750")

    # 0 tokens = 0.000000 USD
    est_0 = estimate_input_cost(model, tokens_in=0)
    assert est_0 == Decimal("0.000000")


def test_calculate_max_request_cost_standard():
    """Verify conservative max cost includes estimated input and max output tokens."""
    model = get_model("rantera-base-v1")
    assert model is not None

    # Unspecified max_tokens defaults to model.max_output_tokens (4096)
    # 1000 in * $0.003/1k + 4096 out * $0.015/1k = 0.003000 + 0.061440 = 0.064440
    cost_default = calculate_max_request_cost(model, estimated_tokens_in=1000)
    assert cost_default == Decimal("0.064440")

    # Explicit max_tokens = 500
    # 1000 in * $0.003/1k + 500 out * $0.015/1k = 0.003000 + 0.007500 = 0.010500
    cost_explicit = calculate_max_request_cost(model, estimated_tokens_in=1000, max_tokens=500)
    assert cost_explicit == Decimal("0.010500")

    # 0 input tokens, 100 output tokens
    cost_zero_in = calculate_max_request_cost(model, estimated_tokens_in=0, max_tokens=100)
    assert cost_zero_in == Decimal("0.001500")


def test_calculate_max_request_cost_context_window_cap():
    """Verify output tokens are capped by remaining context window."""
    model = get_model("rantera-base-v1")
    assert model is not None
    assert model.context_window == 8192

    # 7000 prompt tokens leaves 1192 tokens in context window
    # Even if max_tokens is unspecified (4096) or requested as 2000, output is capped at 1192
    # 7000 * 0.003/1000 + 1192 * 0.015/1000 = 0.021000 + 0.017880 = 0.038880
    cost_capped = calculate_max_request_cost(model, estimated_tokens_in=7000, max_tokens=4096)
    assert cost_capped == Decimal("0.038880")

    cost_capped_default = calculate_max_request_cost(model, estimated_tokens_in=7000, max_tokens=None)
    assert cost_capped_default == Decimal("0.038880")


def test_calculate_max_request_cost_validation():
    """Verify calculate_max_request_cost rejects booleans and negative values."""
    import pytest
    model = get_model("rantera-base-v1")
    assert model is not None

    with pytest.raises(TypeError):
        calculate_max_request_cost(model, estimated_tokens_in=True)  # type: ignore[arg-type]

    with pytest.raises(TypeError):
        calculate_max_request_cost(model, estimated_tokens_in=100, max_tokens=False)  # type: ignore[arg-type]

    with pytest.raises(ValueError):
        calculate_max_request_cost(model, estimated_tokens_in=-5)

    with pytest.raises(ValueError):
        calculate_max_request_cost(model, estimated_tokens_in=100, max_tokens=-10)
