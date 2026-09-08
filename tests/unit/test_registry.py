from llm_router.registry import ModelRegistry
from llm_router.schemas import ModelCapabilities, ModelDefinition, ModelPerformance, ModelPricing


def make_model(model_id, categories, enabled=True, priority=1):
    return ModelDefinition(
        id=model_id,
        provider="provider_a",
        display_name=model_id,
        categories=tuple(categories),
        pricing=ModelPricing(input_per_1k_tokens=0.01, output_per_1k_tokens=0.02),
        performance=ModelPerformance(expected_latency_ms=500),
        capabilities=ModelCapabilities(context_window=8000),
        priority=priority,
        enabled=enabled,
    )


class TestModelRegistry:
    def test_get_models_returns_all(self):
        models = [make_model("a", ["simple"]), make_model("b", ["coding"], enabled=False)]
        registry = ModelRegistry(models)
        assert len(registry.get_models()) == 2
        assert len(registry) == 2

    def test_get_enabled_models_excludes_disabled(self):
        models = [make_model("a", ["simple"]), make_model("b", ["coding"], enabled=False)]
        registry = ModelRegistry(models)
        enabled = registry.get_enabled_models()
        assert [m.id for m in enabled] == ["a"]

    def test_filter_by_category(self):
        models = [
            make_model("a", ["simple", "summarization"]),
            make_model("b", ["coding"]),
            make_model("c", ["simple"], enabled=False),
        ]
        registry = ModelRegistry(models)
        matches = registry.filter_by_category("simple")
        assert [m.id for m in matches] == ["a"]

    def test_get_model_found_and_missing(self):
        registry = ModelRegistry([make_model("a", ["simple"])])
        assert registry.get_model("a") is not None
        assert registry.get_model("missing") is None

    def test_iteration(self):
        models = [make_model("a", ["simple"]), make_model("b", ["coding"])]
        registry = ModelRegistry(models)
        assert [m.id for m in registry] == ["a", "b"]
