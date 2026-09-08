"""End-to-end pipeline test: prompt -> embedding -> classifier -> selection -> model.

Uses a deterministic FakeEmbedder (no network / model download) combined with
a real, freshly-fitted sklearn LogisticRegression so the classifier stage is
exercised for real, not stubbed out.
"""

from __future__ import annotations

from sklearn.linear_model import LogisticRegression

from llm_router.classifier import TaskClassifier
from llm_router.config import (
    ClassifierConfig,
    EmbeddingConfig,
    LoggingConfig,
    RouterConfig,
    SelectionConfig,
    SelectionWeights,
)
from llm_router.registry import ModelRegistry
from llm_router.router import Router
from llm_router.schemas import ModelCapabilities, ModelDefinition, ModelPerformance, ModelPricing
from llm_router.selector import Selector

from .._fakes import FakeEmbedder

CATEGORIES = ("simple", "coding", "reasoning")

TRAIN_TEXTS = [
    "what is 2 plus 2",
    "what is the capital of france",
    "how many days in a week",
    "write a python function to sort a list",
    "debug this null pointer exception",
    "fix this sql query",
    "explain the trade-offs between two architectures",
    "compare renting versus buying a home",
    "analyze the causes of this system failure",
]
TRAIN_LABELS = [
    "simple",
    "simple",
    "simple",
    "coding",
    "coding",
    "coding",
    "reasoning",
    "reasoning",
    "reasoning",
]


def _fit_classifier(threshold: float = 0.5) -> TaskClassifier:
    embedder = FakeEmbedder(dim=32)
    X = embedder.encode_batch(TRAIN_TEXTS)
    clf = LogisticRegression(max_iter=1000, random_state=0)
    clf.fit(X, TRAIN_LABELS)
    return TaskClassifier(
        embedder=embedder,
        sklearn_model=clf,
        classes=list(clf.classes_),
        confidence_threshold=threshold,
        max_prompt_length=4000,
    )


def _make_model(model_id, categories, priority=1, enabled=True):
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


def _make_router_config(policy="priority", log_prompts=False) -> RouterConfig:
    return RouterConfig(
        embedding=EmbeddingConfig(model="fake/model"),
        classifier=ClassifierConfig(artifact="unused.joblib"),
        categories=CATEGORIES,
        selection=SelectionConfig(policy=policy, weights=SelectionWeights()),
        models=(),
        logging=LoggingConfig(log_prompts=log_prompts),
        config_path=None,
    )


class TestFullPipeline:
    def test_confident_prompt_selects_a_model(self):
        classifier = _fit_classifier(threshold=0.3)
        registry = ModelRegistry(
            [_make_model("fast", ["simple"], priority=1), _make_model("coder", ["coding"], priority=1)]
        )
        selector = Selector(_make_router_config().selection, CATEGORIES)
        router = Router(_make_router_config(), classifier, registry, selector)

        result = router.route("what is 2 plus 2")

        assert result.category == "simple"
        assert result.reason == "selected"
        assert result.model_id == "fast"
        assert result.selected
        assert len(result.candidates) == 1

    def test_low_confidence_prompt_returns_unknown(self):
        classifier = _fit_classifier(threshold=0.999)
        registry = ModelRegistry([_make_model("fast", ["simple"])])
        selector = Selector(_make_router_config().selection, CATEGORIES)
        router = Router(_make_router_config(), classifier, registry, selector)

        result = router.route("what is 3 plus 3")

        assert result.category == "unknown"
        assert result.reason == "classification_below_threshold"
        assert result.model_id is None
        assert not result.selected
        assert result.candidates == []

    def test_no_eligible_models_for_confident_category(self):
        classifier = _fit_classifier(threshold=0.3)
        # No model at all supports "coding".
        registry = ModelRegistry([_make_model("fast", ["simple"])])
        selector = Selector(_make_router_config().selection, CATEGORIES)
        router = Router(_make_router_config(), classifier, registry, selector)

        result = router.route("debug this null pointer exception")

        assert result.category == "coding"
        assert result.model_id is None
        assert result.reason == "no_eligible_models"

    def test_disabled_model_is_excluded(self):
        classifier = _fit_classifier(threshold=0.3)
        registry = ModelRegistry([_make_model("fast", ["simple"], enabled=False)])
        selector = Selector(_make_router_config().selection, CATEGORIES)
        router = Router(_make_router_config(), classifier, registry, selector)

        result = router.route("what is 3 plus 3")

        assert result.model_id is None
        assert result.reason == "no_eligible_models"

    def test_weighted_policy_ranks_all_candidates(self):
        classifier = _fit_classifier(threshold=0.3)
        registry = ModelRegistry(
            [
                _make_model("cheap", ["simple"], priority=2),
                _make_model("premium", ["simple"], priority=1),
            ]
        )
        config = _make_router_config(policy="weighted")
        selector = Selector(config.selection, CATEGORIES)
        router = Router(config, classifier, registry, selector)

        result = router.route("what is 2 plus 2")

        assert result.reason == "selected"
        assert len(result.candidates) == 2
        assert result.candidates[0].final_score >= result.candidates[1].final_score

    def test_privacy_safe_logging_does_not_raise(self, caplog):
        classifier = _fit_classifier(threshold=0.3)
        registry = ModelRegistry([_make_model("fast", ["simple"])])
        config = _make_router_config(log_prompts=False)
        selector = Selector(config.selection, CATEGORIES)
        router = Router(config, classifier, registry, selector)

        with caplog.at_level("INFO"):
            router.route("what is 3 plus 3")

        assert "what is 3 plus 3" not in caplog.text
