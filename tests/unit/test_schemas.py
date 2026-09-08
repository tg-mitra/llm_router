import pytest

from llm_router.exceptions import ModelValidationError
from llm_router.schemas import (
    ClassificationResult,
    ModelCapabilities,
    ModelDefinition,
    ModelPerformance,
    ModelPricing,
)


def make_model(**overrides):
    defaults = dict(
        id="model_a",
        provider="provider_a",
        display_name="Model A",
        categories=("simple",),
        pricing=ModelPricing(input_per_1k_tokens=0.01, output_per_1k_tokens=0.02),
        performance=ModelPerformance(expected_latency_ms=500),
        capabilities=ModelCapabilities(context_window=32000, tool_calling=True),
        priority=1,
        enabled=True,
    )
    defaults.update(overrides)
    return ModelDefinition(**defaults)


class TestClassificationResult:
    def test_is_confident_true_when_no_reason(self):
        result = ClassificationResult(category="coding", confidence=0.9, scores={"coding": 0.9})
        assert result.is_confident

    def test_is_confident_false_when_reason_set(self):
        result = ClassificationResult(
            category="unknown", confidence=0.4, scores={}, reason="classification_below_threshold"
        )
        assert not result.is_confident


class TestModelPricing:
    def test_negative_input_price_rejected(self):
        with pytest.raises(ModelValidationError):
            ModelPricing(input_per_1k_tokens=-1, output_per_1k_tokens=0.01)

    def test_negative_output_price_rejected(self):
        with pytest.raises(ModelValidationError):
            ModelPricing(input_per_1k_tokens=0.01, output_per_1k_tokens=-1)


class TestModelPerformance:
    def test_non_positive_latency_rejected(self):
        with pytest.raises(ModelValidationError):
            ModelPerformance(expected_latency_ms=0)


class TestModelCapabilities:
    def test_non_positive_context_window_rejected(self):
        with pytest.raises(ModelValidationError):
            ModelCapabilities(context_window=0)


class TestModelDefinition:
    def test_valid_model_constructs(self):
        model = make_model()
        assert model.id == "model_a"
        assert model.supports("simple")
        assert not model.supports("coding")

    def test_empty_id_rejected(self):
        with pytest.raises(ModelValidationError):
            make_model(id="")

    def test_empty_provider_rejected(self):
        with pytest.raises(ModelValidationError):
            make_model(provider="")

    def test_no_categories_rejected(self):
        with pytest.raises(ModelValidationError):
            make_model(categories=())

    def test_negative_priority_rejected(self):
        with pytest.raises(ModelValidationError):
            make_model(priority=-1)
