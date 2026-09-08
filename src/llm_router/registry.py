"""Model registry: static metadata about available LLMs.

The registry answers "what models exist and what are they capable of" -- it
never tracks live health/availability state (requirement 4.2 / 8).
"""

from __future__ import annotations

import logging
from typing import Iterable

from .schemas import ModelDefinition

logger = logging.getLogger(__name__)


class ModelRegistry:
    """Read-only collection of :class:`ModelDefinition` entries."""

    def __init__(self, models: Iterable[ModelDefinition]) -> None:
        self._models = tuple(models)
        logger.debug("model registry initialized with %d models", len(self._models))

    def get_models(self) -> list[ModelDefinition]:
        """Return every registered model, including disabled ones."""
        return list(self._models)

    def get_enabled_models(self) -> list[ModelDefinition]:
        return [m for m in self._models if m.enabled]

    def get_model(self, model_id: str) -> ModelDefinition | None:
        for model in self._models:
            if model.id == model_id:
                return model
        return None

    def filter_by_category(self, category: str) -> list[ModelDefinition]:
        """Return enabled models that support the given category."""
        return [m for m in self.get_enabled_models() if m.supports(category)]

    def __len__(self) -> int:
        return len(self._models)

    def __iter__(self):
        return iter(self._models)
