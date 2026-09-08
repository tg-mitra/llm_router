"""Prompt embedding + supervised task classification."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Union

import joblib
import numpy as np

from .embeddings import Embedder
from .exceptions import ClassifierArtifactError, InvalidPromptError
from .schemas import ClassificationResult

logger = logging.getLogger(__name__)

UNKNOWN_CATEGORY = "unknown"
LOW_CONFIDENCE_REASON = "classification_below_threshold"

_REQUIRED_ARTIFACT_KEYS = ("classifier", "classes")


class TaskClassifier:
    """Classifies a prompt into a configured task category.

    Combines an :class:`~llm_router.embeddings.Embedder` with a trained
    scikit-learn classifier. The classifier never encodes knowledge of
    specific LLM model names -- only task categories (requirement 4.1).
    """

    def __init__(
        self,
        embedder: Embedder,
        sklearn_model: Any,
        classes: list[str],
        confidence_threshold: float = 0.70,
        max_prompt_length: int = 4000,
    ) -> None:
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between 0.0 and 1.0")
        if max_prompt_length <= 0:
            raise ValueError("max_prompt_length must be > 0")

        self._embedder = embedder
        self._model = sklearn_model
        self._classes = list(classes)
        self.confidence_threshold = confidence_threshold
        self.max_prompt_length = max_prompt_length

    @property
    def classes(self) -> list[str]:
        return list(self._classes)

    @classmethod
    def from_artifact(
        cls,
        artifact_path: Union[str, Path],
        embedder: Embedder,
        confidence_threshold: float = 0.70,
        max_prompt_length: int = 4000,
    ) -> "TaskClassifier":
        """Load a classifier from a joblib artifact produced by `training/train.py`."""

        path = Path(artifact_path)
        if not path.is_file():
            raise ClassifierArtifactError(f"classifier artifact not found: {path}")

        try:
            payload = joblib.load(path)
        except Exception as exc:
            raise ClassifierArtifactError(
                f"failed to load classifier artifact '{path}': {exc}"
            ) from exc

        if not isinstance(payload, dict):
            raise ClassifierArtifactError(
                f"classifier artifact '{path}' has an unexpected format "
                "(expected a dict with 'classifier' and 'classes' keys)"
            )

        missing = [k for k in _REQUIRED_ARTIFACT_KEYS if k not in payload]
        if missing:
            raise ClassifierArtifactError(
                f"classifier artifact '{path}' is missing required keys: {missing}"
            )

        return cls(
            embedder=embedder,
            sklearn_model=payload["classifier"],
            classes=list(payload["classes"]),
            confidence_threshold=confidence_threshold,
            max_prompt_length=max_prompt_length,
        )

    def validate_prompt(self, prompt: str) -> str:
        if not isinstance(prompt, str):
            raise InvalidPromptError(f"prompt must be a string, got {type(prompt).__name__}")
        trimmed = prompt.strip()
        if not trimmed:
            raise InvalidPromptError("prompt must not be empty")
        if len(trimmed) > self.max_prompt_length:
            raise InvalidPromptError(
                f"prompt length {len(trimmed)} exceeds max_prompt_length "
                f"({self.max_prompt_length})"
            )
        return trimmed

    def predict_proba(self, prompt: str) -> dict[str, float]:
        """Return the full category -> probability distribution for a prompt."""

        trimmed = self.validate_prompt(prompt)
        embedding = self._embedder.encode(trimmed)
        probabilities = self._model.predict_proba(np.asarray([embedding]))[0]
        return {label: float(score) for label, score in zip(self._classes, probabilities)}

    def predict(self, prompt: str) -> ClassificationResult:
        """Classify a prompt, applying the configured confidence threshold."""

        scores = self.predict_proba(prompt)
        best_category = max(scores, key=scores.get)
        best_confidence = scores[best_category]

        if best_confidence < self.confidence_threshold:
            logger.info(
                "classification below threshold: best=%s confidence=%.4f threshold=%.4f",
                best_category,
                best_confidence,
                self.confidence_threshold,
            )
            return ClassificationResult(
                category=UNKNOWN_CATEGORY,
                confidence=best_confidence,
                scores=scores,
                reason=LOW_CONFIDENCE_REASON,
            )

        logger.info(
            "classification result: category=%s confidence=%.4f", best_category, best_confidence
        )
        return ClassificationResult(
            category=best_category,
            confidence=best_confidence,
            scores=scores,
            reason=None,
        )
