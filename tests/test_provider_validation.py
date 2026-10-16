"""Focused unit tests for provider usage validation and token count integrity."""

from __future__ import annotations

from decimal import Decimal
import pytest
from pydantic import ValidationError

from app.models import RegisteredModel, calculate_inference_cost, estimate_input_cost
from app.vertex import PredictionResult


def test_prediction_result_accepts_valid_integers():
    """PredictionResult accepts strictly non-negative integer tokens and latency."""
    res = PredictionResult(
        content="Hello world",
        tokens_in=10,
        tokens_out=25,
        latency_ms=120,
        finish_reason="stop",
    )
    assert res.tokens_in == 10
    assert res.tokens_out == 25
    assert res.latency_ms == 120
    assert res.finish_reason == "stop"


def test_prediction_result_accepts_zero_tokens():
    """PredictionResult does not arbitrarily reject legitimate zero-token responses."""
    res = PredictionResult(
        content="",
        tokens_in=0,
        tokens_out=0,
        latency_ms=15,
        finish_reason="stop",
    )
    assert res.tokens_in == 0
    assert res.tokens_out == 0


def test_prediction_result_rejects_negative_input_tokens():
    """tokens_in < 0 must raise ValidationError."""
    with pytest.raises(ValidationError):
        PredictionResult(
            content="Error test",
            tokens_in=-1,
            tokens_out=10,
            latency_ms=50,
        )


def test_prediction_result_rejects_negative_output_tokens():
    """tokens_out < 0 must raise ValidationError."""
    with pytest.raises(ValidationError):
        PredictionResult(
            content="Error test",
            tokens_in=10,
            tokens_out=-5,
            latency_ms=50,
        )


def test_prediction_result_rejects_negative_latency():
    """latency_ms < 0 must raise ValidationError."""
    with pytest.raises(ValidationError):
        PredictionResult(
            content="Error test",
            tokens_in=10,
            tokens_out=10,
            latency_ms=-1,
        )


def test_prediction_result_rejects_float_tokens():
    """Floats must be rejected for integer token counts."""
    with pytest.raises(ValidationError):
        PredictionResult(
            content="Error test",
            tokens_in=12.5,  # type: ignore[arg-type]
            tokens_out=10,
            latency_ms=50,
        )


def test_calculate_inference_cost_rejects_negative_or_invalid_tokens():
    """calculate_inference_cost must raise ValueError or TypeError on invalid token types."""
    model = RegisteredModel(
        model_id="test-model",
        description="test",
        input_cost_per_1k=Decimal("0.003000"),
        output_cost_per_1k=Decimal("0.015000"),
    )

    with pytest.raises(ValueError, match="non-negative"):
        calculate_inference_cost(model, tokens_in=-5, tokens_out=10)

    with pytest.raises(ValueError, match="non-negative"):
        calculate_inference_cost(model, tokens_in=10, tokens_out=-1)

    with pytest.raises(TypeError, match="integers"):
        calculate_inference_cost(model, tokens_in="10", tokens_out=5)  # type: ignore[arg-type]

    # Explicitly verify booleans are rejected despite bool being a subclass of int
    with pytest.raises(TypeError, match="not booleans"):
        calculate_inference_cost(model, tokens_in=True, tokens_out=10)  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="not booleans"):
        calculate_inference_cost(model, tokens_in=False, tokens_out=10)  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="not booleans"):
        calculate_inference_cost(model, tokens_in=10, tokens_out=True)  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="not booleans"):
        calculate_inference_cost(model, tokens_in=10, tokens_out=False)  # type: ignore[arg-type]


def test_estimate_input_cost_rejects_negative_or_invalid_tokens():
    """estimate_input_cost must raise ValueError or TypeError on invalid tokens."""
    model = RegisteredModel(
        model_id="test-model",
        description="test",
        input_cost_per_1k=Decimal("0.003000"),
        output_cost_per_1k=Decimal("0.015000"),
    )

    with pytest.raises(ValueError, match="non-negative"):
        estimate_input_cost(model, tokens_in=-1)

    with pytest.raises(TypeError, match="integer"):
        estimate_input_cost(model, tokens_in=10.5)  # type: ignore[arg-type]

    # Explicitly verify booleans are rejected despite bool being a subclass of int
    with pytest.raises(TypeError, match="not a boolean"):
        estimate_input_cost(model, tokens_in=True)  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="not a boolean"):
        estimate_input_cost(model, tokens_in=False)  # type: ignore[arg-type]


def test_prediction_result_rejects_booleans():
    """PredictionResult must strictly reject booleans for tokens_in and tokens_out."""
    with pytest.raises(ValidationError):
        PredictionResult(
            content="Error test",
            tokens_in=True,  # type: ignore[arg-type]
            tokens_out=10,
            latency_ms=50,
        )

    with pytest.raises(ValidationError):
        PredictionResult(
            content="Error test",
            tokens_in=10,
            tokens_out=False,  # type: ignore[arg-type]
            latency_ms=50,
        )
