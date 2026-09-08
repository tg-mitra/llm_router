"""Scoring utilities for ranking candidate models.

Implements the normalized weighted score described in requirement 9.3:

    final_score = priority_weight * priority_score
                + cost_weight     * cost_score
                + latency_weight  * latency_score
                + capability_weight * capability_score

All sub-scores are normalized to [0, 1] (higher is always better) across the
current candidate set, then combined using the configured weights.
"""

from __future__ import annotations

from typing import Sequence

from .config import SelectionWeights
from .schemas import CandidateScore, ModelDefinition


def _normalize(values: Sequence[float], invert: bool) -> list[float]:
    """Min-max normalize `values` to [0, 1].

    When `invert` is True, the smallest raw value maps to 1.0 (used for
    "lower is better" metrics like cost and latency). If every value is
    identical, every candidate scores 1.0 -- there is nothing to
    differentiate them on, so it should not be treated as a penalty.
    """
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi == lo:
        return [1.0 for _ in values]
    normalized = [(v - lo) / (hi - lo) for v in values]
    if invert:
        normalized = [1.0 - n for n in normalized]
    return normalized


def compute_priority_scores(models: Sequence[ModelDefinition]) -> dict[str, float]:
    """Lower `priority` value is better (priority 1 outranks priority 2)."""
    raw = [m.priority for m in models]
    normalized = _normalize(raw, invert=True)
    return {m.id: score for m, score in zip(models, normalized)}


def compute_cost_scores(models: Sequence[ModelDefinition]) -> dict[str, float]:
    """Lower blended input/output price is better."""
    raw = [
        m.pricing.input_per_1k_tokens + m.pricing.output_per_1k_tokens for m in models
    ]
    normalized = _normalize(raw, invert=True)
    return {m.id: score for m, score in zip(models, normalized)}


def compute_latency_scores(models: Sequence[ModelDefinition]) -> dict[str, float]:
    """Lower expected latency is better."""
    raw = [m.performance.expected_latency_ms for m in models]
    normalized = _normalize(raw, invert=True)
    return {m.id: score for m, score in zip(models, normalized)}


def compute_capability_scores(models: Sequence[ModelDefinition]) -> dict[str, float]:
    """Larger context window and richer feature support is better.

    Combines a normalized context-window score with a flat bonus for
    tool-calling and structured-output support.
    """
    context_values = [m.capabilities.context_window for m in models]
    context_scores = _normalize(context_values, invert=False)

    result: dict[str, float] = {}
    for model, context_score in zip(models, context_scores):
        feature_bonus = 0.0
        feature_bonus += 0.5 if model.capabilities.tool_calling else 0.0
        feature_bonus += 0.5 if model.capabilities.structured_output else 0.0
        result[model.id] = (0.5 * context_score) + (0.5 * feature_bonus)
    return result


def compute_weighted_scores(
    models: Sequence[ModelDefinition], weights: SelectionWeights
) -> list[CandidateScore]:
    """Rank candidates using the configured weighted scoring policy."""
    if not models:
        return []

    priority_scores = compute_priority_scores(models)
    cost_scores = compute_cost_scores(models)
    latency_scores = compute_latency_scores(models)
    capability_scores = compute_capability_scores(models)

    weight_sum = weights.priority + weights.cost + weights.latency + weights.capability

    candidates = []
    for model in models:
        p = priority_scores[model.id]
        c = cost_scores[model.id]
        lat = latency_scores[model.id]
        cap = capability_scores[model.id]
        final = (
            weights.priority * p
            + weights.cost * c
            + weights.latency * lat
            + weights.capability * cap
        ) / weight_sum

        candidates.append(
            CandidateScore(
                model_id=model.id,
                final_score=final,
                priority_score=p,
                cost_score=c,
                latency_score=lat,
                capability_score=cap,
                model=model,
            )
        )

    candidates.sort(key=lambda c: (-c.final_score, c.model.priority, c.model_id))
    return candidates


def compute_priority_first_scores(models: Sequence[ModelDefinition]) -> list[CandidateScore]:
    """Deterministic priority-first ranking (requirement 9.3, first-implementation policy).

    Candidates are ordered purely by ascending `priority` (ties broken by
    model id). Sub-scores are still computed and reported for
    explainability, but do not affect ordering.
    """
    if not models:
        return []

    priority_scores = compute_priority_scores(models)
    cost_scores = compute_cost_scores(models)
    latency_scores = compute_latency_scores(models)
    capability_scores = compute_capability_scores(models)

    candidates = [
        CandidateScore(
            model_id=model.id,
            final_score=priority_scores[model.id],
            priority_score=priority_scores[model.id],
            cost_score=cost_scores[model.id],
            latency_score=latency_scores[model.id],
            capability_score=capability_scores[model.id],
            model=model,
        )
        for model in models
    ]
    candidates.sort(key=lambda c: (c.model.priority, c.model_id))
    return candidates


def estimate_cost(model: ModelDefinition, input_tokens: int, output_tokens: int) -> float:
    """Estimate the price of a request using configured (not live) pricing.

    This is a configuration-derived estimate only. Actual billed usage must
    come from the host application or LLM provider (requirement 9.4).
    """
    if input_tokens < 0 or output_tokens < 0:
        raise ValueError("input_tokens and output_tokens must be >= 0")

    input_cost = (input_tokens / 1000.0) * model.pricing.input_per_1k_tokens
    output_cost = (output_tokens / 1000.0) * model.pricing.output_per_1k_tokens
    return input_cost + output_cost
