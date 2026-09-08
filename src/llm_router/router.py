"""Public library facade: prompt -> classification -> model selection."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Union

from .classifier import TaskClassifier
from .config import RouterConfig, load_config
from .embeddings import Embedder, SentenceTransformerEmbedder
from .registry import ModelRegistry
from .schemas import RoutingResult
from .selector import Selector

logger = logging.getLogger(__name__)

REASON_SELECTED = "selected"
REASON_NO_ELIGIBLE_MODELS = "no_eligible_models"


def _resolve_artifact_path(artifact: str, config_path: Path | None) -> Path:
    path = Path(artifact)
    if path.is_absolute() or config_path is None:
        return path
    return (config_path.parent / path).resolve()


class Router:
    """High-level entry point: classify a prompt and select the best model for it."""

    def __init__(
        self,
        config: RouterConfig,
        classifier: TaskClassifier,
        registry: ModelRegistry,
        selector: Selector,
    ) -> None:
        self.config = config
        self.classifier = classifier
        self.registry = registry
        self.selector = selector

    @classmethod
    def from_config(
        cls,
        config_path: Union[str, Path],
        embedder: Embedder | None = None,
    ) -> "Router":
        """Build a fully-wired Router from a YAML configuration file.

        Args:
            config_path: path to a `config.yml` following the schema in
                `config.example.yml`.
            embedder: optional pre-built embedder to reuse (e.g. to share a
                loaded sentence-transformers model across multiple routers).
                When omitted, one is built from `embedding.model`.
        """
        config = load_config(config_path)
        logger.info("loaded router configuration from %s", config.config_path)

        if embedder is None:
            embedder = SentenceTransformerEmbedder(
                model_name=config.embedding.model,
                cache_folder=config.embedding.cache_folder,
                device=config.embedding.device,
            )

        artifact_path = _resolve_artifact_path(config.classifier.artifact, config.config_path)
        classifier = TaskClassifier.from_artifact(
            artifact_path,
            embedder=embedder,
            confidence_threshold=config.classifier.confidence_threshold,
            max_prompt_length=config.classifier.max_prompt_length,
        )
        logger.info("loaded classifier artifact from %s", artifact_path)

        registry = ModelRegistry(config.models)
        selector = Selector(config.selection, config.categories)

        return cls(config=config, classifier=classifier, registry=registry, selector=selector)

    def route(self, prompt: str) -> RoutingResult:
        """Classify `prompt` and select the best available model for it.

        Never raises for "no confident category" or "no eligible model"
        outcomes -- both are reported via `RoutingResult.reason` with
        `model_id=None` so callers can handle them without exception
        handling on the hot path. Invalid input (`InvalidPromptError`) and
        misconfiguration (`UnsupportedCategoryError`) still raise, since
        those are actionable bugs rather than routine runtime outcomes.
        """
        classification = self.classifier.predict(prompt)

        if self.config.logging.log_prompts:
            logger.info("routing prompt=%r category=%s confidence=%.4f",
                        prompt, classification.category, classification.confidence)
        else:
            logger.info(
                "routing prompt(len=%d) category=%s confidence=%.4f",
                len(prompt),
                classification.category,
                classification.confidence,
            )

        if classification.reason is not None:
            return RoutingResult(
                category=classification.category,
                confidence=classification.confidence,
                model_id=None,
                reason=classification.reason,
                candidates=[],
            )

        ranked = self.selector.rank(classification.category, self.registry.get_models())
        if not ranked:
            logger.warning("no eligible models for category '%s'", classification.category)
            return RoutingResult(
                category=classification.category,
                confidence=classification.confidence,
                model_id=None,
                reason=REASON_NO_ELIGIBLE_MODELS,
                candidates=[],
            )

        return RoutingResult(
            category=classification.category,
            confidence=classification.confidence,
            model_id=ranked[0].model_id,
            reason=REASON_SELECTED,
            candidates=ranked,
        )
