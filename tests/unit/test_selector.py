import pytest

from llm_router.config import SelectionConfig, SelectionWeights
from llm_router.exceptions import NoEligibleModelsError, UnsupportedCategoryError
from llm_router.schemas import ModelCapabilities, ModelDefinition, ModelPerformance, ModelPricing
from llm_router.selector import Selector

CATEGORIES = ("simple", "coding", "reasoning")


def make_model(
    model_id,
    categories,
    priority=1,
    enabled=True,
    context_window=8000,
    tool_calling=False,
):
    return ModelDefinition(
        id=model_id,
        provider="provider_a",
        display_name=model_id,
        categories=tuple(categories),
        pricing=ModelPricing(input_per_1k_tokens=0.01, output_per_1k_tokens=0.02),
        performance=ModelPerformance(expected_latency_ms=500),
        capabilities=ModelCapabilities(context_window=context_window, tool_calling=tool_calling),
        priority=priority,
        enabled=enabled,
    )


def make_selector(policy="priority", required_capabilities=None):
    config = SelectionConfig(
        policy=policy,
        weights=SelectionWeights(),
        required_capabilities=required_capabilities or {},
    )
    return Selector(config, CATEGORIES)


class TestFilterCandidates:
    def test_unsupported_category_raises(self):
        selector = make_selector()
        with pytest.raises(UnsupportedCategoryError):
            selector.filter_candidates("not_a_category", [])

    def test_excludes_disabled_models(self):
        selector = make_selector()
        models = [make_model("a", ["simple"], enabled=False)]
        assert selector.filter_candidates("simple", models) == []

    def test_excludes_non_matching_category(self):
        selector = make_selector()
        models = [make_model("a", ["coding"])]
        assert selector.filter_candidates("simple", models) == []

    def test_includes_matching_enabled_model(self):
        selector = make_selector()
        models = [make_model("a", ["simple"])]
        result = selector.filter_candidates("simple", models)
        assert [m.id for m in result] == ["a"]

    def test_required_min_context_window(self):
        selector = make_selector(required_capabilities={"min_context_window": 10000})
        models = [make_model("small", ["simple"], context_window=4000), make_model("big", ["simple"], context_window=32000)]
        result = selector.filter_candidates("simple", models)
        assert [m.id for m in result] == ["big"]

    def test_required_tool_calling(self):
        selector = make_selector(required_capabilities={"tool_calling": True})
        models = [
            make_model("no_tools", ["simple"], tool_calling=False),
            make_model("has_tools", ["simple"], tool_calling=True),
        ]
        result = selector.filter_candidates("simple", models)
        assert [m.id for m in result] == ["has_tools"]

    def test_unknown_required_capability_key_ignored(self):
        selector = make_selector(required_capabilities={"some_unknown_flag": True})
        models = [make_model("a", ["simple"])]
        result = selector.filter_candidates("simple", models)
        assert [m.id for m in result] == ["a"]


class TestRank:
    def test_no_candidates_returns_empty(self):
        selector = make_selector()
        assert selector.rank("simple", []) == []

    def test_priority_policy_orders_by_priority(self):
        selector = make_selector(policy="priority")
        models = [make_model("low", ["simple"], priority=5), make_model("high", ["simple"], priority=1)]
        ranked = selector.rank("simple", models)
        assert [c.model_id for c in ranked] == ["high", "low"]

    def test_weighted_policy_returns_all_candidates(self):
        selector = make_selector(policy="weighted")
        models = [make_model("a", ["simple"]), make_model("b", ["simple"])]
        ranked = selector.rank("simple", models)
        assert {c.model_id for c in ranked} == {"a", "b"}


class TestSelect:
    def test_raises_when_no_eligible_models(self):
        selector = make_selector()
        with pytest.raises(NoEligibleModelsError):
            selector.select("simple", [])

    def test_returns_best_candidate(self):
        selector = make_selector(policy="priority")
        models = [make_model("low", ["simple"], priority=5), make_model("high", ["simple"], priority=1)]
        best = selector.select("simple", models)
        assert best.model_id == "high"
