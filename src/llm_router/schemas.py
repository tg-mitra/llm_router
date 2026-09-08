"""Typed data structures shared across the llm_router package."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .exceptions import ModelValidationError


@dataclass(frozen=True)
class ClassificationResult:
    """Result of classifying a prompt into a task category."""

    category: str
    confidence: float
    scores: dict[str, float]
    reason: Optional[str] = None

    @property
    def is_confident(self) -> bool:
        return self.reason is None


@dataclass(frozen=True)
class ModelPricing:
    input_per_1k_tokens: float
    output_per_1k_tokens: float

    def __post_init__(self) -> None:
        if self.input_per_1k_tokens < 0:
            raise ModelValidationError("pricing.input_per_1k_tokens must be >= 0")
        if self.output_per_1k_tokens < 0:
            raise ModelValidationError("pricing.output_per_1k_tokens must be >= 0")


@dataclass(frozen=True)
class ModelPerformance:
    expected_latency_ms: float

    def __post_init__(self) -> None:
        if self.expected_latency_ms <= 0:
            raise ModelValidationError("performance.expected_latency_ms must be > 0")


@dataclass(frozen=True)
class ModelCapabilities:
    context_window: int
    tool_calling: bool = False
    structured_output: bool = False

    def __post_init__(self) -> None:
        if self.context_window <= 0:
            raise ModelValidationError("capabilities.context_window must be > 0")


@dataclass(frozen=True)
class ModelDefinition:
    """Static metadata describing an LLM available for routing."""

    id: str
    provider: str
    display_name: str
    categories: tuple[str, ...]
    pricing: ModelPricing
    performance: ModelPerformance
    capabilities: ModelCapabilities
    priority: int = 100
    enabled: bool = True

    def __post_init__(self) -> None:
        if not self.id or not self.id.strip():
            raise ModelValidationError("model.id must be a non-empty string")
        if not self.provider or not self.provider.strip():
            raise ModelValidationError(f"model '{self.id}': provider must be a non-empty string")
        if not self.categories:
            raise ModelValidationError(f"model '{self.id}': must declare at least one category")
        if self.priority < 0:
            raise ModelValidationError(f"model '{self.id}': priority must be >= 0")

    def supports(self, category: str) -> bool:
        return category in self.categories


@dataclass(frozen=True)
class CandidateScore:
    """Ranking breakdown for a single candidate model."""

    model_id: str
    final_score: float
    priority_score: float
    cost_score: float
    latency_score: float
    capability_score: float
    model: ModelDefinition


@dataclass(frozen=True)
class RoutingResult:
    """Outcome of routing a prompt to a selected model."""

    category: str
    confidence: float
    model_id: Optional[str]
    reason: str
    candidates: list[CandidateScore] = field(default_factory=list)

    @property
    def selected(self) -> bool:
        return self.model_id is not None
