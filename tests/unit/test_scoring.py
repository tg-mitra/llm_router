import pytest

from llm_router.config import SelectionWeights
from llm_router.schemas import ModelCapabilities, ModelDefinition, ModelPerformance, ModelPricing
from llm_router.scoring import (
    compute_capability_scores,
    compute_cost_scores,
    compute_latency_scores,
    compute_priority_first_scores,
    compute_priority_scores,
    compute_weighted_scores,
    estimate_cost,
)


def make_model(
    model_id,
    priority=1,
    input_price=0.01,
    output_price=0.02,
    latency_ms=500,
    context_window=8000,
    tool_calling=False,
    structured_output=False,
):
    return ModelDefinition(
        id=model_id,
        provider="provider_a",
        display_name=model_id,
        categories=("simple",),
        pricing=ModelPricing(input_per_1k_tokens=input_price, output_per_1k_tokens=output_price),
        performance=ModelPerformance(expected_latency_ms=latency_ms),
        capabilities=ModelCapabilities(
            context_window=context_window,
            tool_calling=tool_calling,
            structured_output=structured_output,
        ),
        priority=priority,
    )


class TestPriorityScores:
    def test_lower_priority_scores_higher(self):
        models = [make_model("a", priority=1), make_model("b", priority=5)]
        scores = compute_priority_scores(models)
        assert scores["a"] > scores["b"]

    def test_equal_priority_scores_equal(self):
        models = [make_model("a", priority=2), make_model("b", priority=2)]
        scores = compute_priority_scores(models)
        assert scores["a"] == scores["b"] == 1.0


class TestCostScores:
    def test_cheaper_model_scores_higher(self):
        models = [make_model("a", input_price=0.01, output_price=0.02), make_model("b", input_price=0.5, output_price=0.5)]
        scores = compute_cost_scores(models)
        assert scores["a"] > scores["b"]


class TestLatencyScores:
    def test_faster_model_scores_higher(self):
        models = [make_model("a", latency_ms=200), make_model("b", latency_ms=2000)]
        scores = compute_latency_scores(models)
        assert scores["a"] > scores["b"]


class TestCapabilityScores:
    def test_larger_context_scores_higher(self):
        models = [make_model("a", context_window=200000), make_model("b", context_window=4000)]
        scores = compute_capability_scores(models)
        assert scores["a"] > scores["b"]

    def test_feature_flags_increase_score(self):
        models = [
            make_model("a", tool_calling=True, structured_output=True),
            make_model("b", tool_calling=False, structured_output=False),
        ]
        scores = compute_capability_scores(models)
        assert scores["a"] > scores["b"]


class TestWeightedScores:
    def test_ranked_descending_by_final_score(self):
        models = [
            make_model("cheap_slow", priority=1, input_price=0.001, output_price=0.001, latency_ms=3000),
            make_model("pricey_fast", priority=1, input_price=1.0, output_price=1.0, latency_ms=100),
        ]
        weights = SelectionWeights(priority=0.0, cost=1.0, latency=0.0, capability=0.0)
        ranked = compute_weighted_scores(models, weights)
        assert ranked[0].model_id == "cheap_slow"
        assert ranked[0].final_score >= ranked[1].final_score

    def test_empty_models_returns_empty(self):
        weights = SelectionWeights()
        assert compute_weighted_scores([], weights) == []

    def test_final_score_is_normalized_weighted_sum(self):
        models = [make_model("a", priority=1), make_model("b", priority=10)]
        weights = SelectionWeights(priority=1.0, cost=0.0, latency=0.0, capability=0.0)
        ranked = compute_weighted_scores(models, weights)
        by_id = {c.model_id: c for c in ranked}
        assert by_id["a"].final_score == pytest.approx(1.0)
        assert by_id["b"].final_score == pytest.approx(0.0)


class TestPriorityFirstScores:
    def test_orders_strictly_by_priority_regardless_of_other_factors(self):
        models = [
            make_model("expensive_low_priority_number", priority=1, input_price=10.0, output_price=10.0),
            make_model("cheap_high_priority_number", priority=5, input_price=0.001, output_price=0.001),
        ]
        ranked = compute_priority_first_scores(models)
        assert [c.model_id for c in ranked] == [
            "expensive_low_priority_number",
            "cheap_high_priority_number",
        ]

    def test_empty_models_returns_empty(self):
        assert compute_priority_first_scores([]) == []


class TestEstimateCost:
    def test_basic_calculation(self):
        model = make_model("a", input_price=0.01, output_price=0.02)
        cost = estimate_cost(model, input_tokens=1000, output_tokens=500)
        assert cost == pytest.approx(0.01 * 1 + 0.02 * 0.5)

    def test_zero_tokens_costs_zero(self):
        model = make_model("a")
        assert estimate_cost(model, input_tokens=0, output_tokens=0) == 0.0

    def test_negative_tokens_raise(self):
        model = make_model("a")
        with pytest.raises(ValueError):
            estimate_cost(model, input_tokens=-1, output_tokens=0)
