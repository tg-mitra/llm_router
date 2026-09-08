"""Candidate filtering and ranking (requirement 9)."""

from __future__ import annotations

import logging
from typing import Any, Iterable

from .config import SelectionConfig
from .exceptions import NoEligibleModelsError, UnsupportedCategoryError
from .schemas import CandidateScore, ModelDefinition
from .scoring import compute_priority_first_scores, compute_weighted_scores

logger = logging.getLogger(__name__)


def _meets_capability_constraints(
    model: ModelDefinition, required_capabilities: dict[str, Any]
) -> bool:
    for key, required_value in required_capabilities.items():
        if key == "min_context_window":
            if model.capabilities.context_window < required_value:
                return False
        elif key in ("tool_calling", "structured_output"):
            if required_value and not getattr(model.capabilities, key):
                return False
        else:
            logger.warning("ignoring unknown required_capabilities key: %s", key)
    return True


class Selector:
    """Filters candidate models and ranks them per the configured policy."""

    def __init__(
        self,
        selection_config: SelectionConfig,
        categories: Iterable[str],
    ) -> None:
        self._config = selection_config
        self._categories = tuple(categories)

    def filter_candidates(
        self, category: str, models: Iterable[ModelDefinition]
    ) -> list[ModelDefinition]:
        """Return enabled, category-compatible, capability-eligible models.

        Raises:
            UnsupportedCategoryError: if `category` is not part of the configured category set.
        """
        if category not in self._categories:
            raise UnsupportedCategoryError(
                f"category '{category}' is not in the configured category set "
                f"{list(self._categories)}"
            )

        candidates = [
            m
            for m in models
            if m.enabled
            and m.supports(category)
            and _meets_capability_constraints(m, self._config.required_capabilities)
        ]
        logger.info("category '%s': %d candidate(s) after filtering", category, len(candidates))
        return candidates

    def rank(self, category: str, models: Iterable[ModelDefinition]) -> list[CandidateScore]:
        """Filter then rank candidates according to the configured selection policy."""
        candidates = self.filter_candidates(category, models)
        if not candidates:
            return []

        if self._config.policy == "weighted":
            return compute_weighted_scores(candidates, self._config.weights)
        return compute_priority_first_scores(candidates)

    def select(self, category: str, models: Iterable[ModelDefinition]) -> CandidateScore:
        """Return the single best-ranked candidate.

        Raises:
            UnsupportedCategoryError: if `category` is not configured.
            NoEligibleModelsError: if no model passes filtering for this category.
        """
        ranked = self.rank(category, models)
        if not ranked:
            raise NoEligibleModelsError(f"no eligible models found for category '{category}'")
        best = ranked[0]
        logger.info("selected model '%s' for category '%s'", best.model_id, category)
        return best
