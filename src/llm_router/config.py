"""YAML configuration loading and validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

import yaml

from .exceptions import ConfigurationError, ModelValidationError
from .schemas import ModelCapabilities, ModelDefinition, ModelPerformance, ModelPricing

VALID_POLICIES = ("priority", "weighted")
DEFAULT_CATEGORIES = ("simple", "coding", "reasoning", "security", "summarization")


@dataclass(frozen=True)
class EmbeddingConfig:
    model: str
    cache_folder: Optional[str] = None
    device: Optional[str] = None


@dataclass(frozen=True)
class ClassifierConfig:
    artifact: str
    confidence_threshold: float = 0.70
    max_prompt_length: int = 4000

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ConfigurationError(
                "classifier.confidence_threshold must be between 0.0 and 1.0"
            )
        if self.max_prompt_length <= 0:
            raise ConfigurationError("classifier.max_prompt_length must be > 0")


@dataclass(frozen=True)
class SelectionWeights:
    priority: float = 0.40
    cost: float = 0.30
    latency: float = 0.20
    capability: float = 0.10

    def __post_init__(self) -> None:
        for name, value in (
            ("priority", self.priority),
            ("cost", self.cost),
            ("latency", self.latency),
            ("capability", self.capability),
        ):
            if value < 0:
                raise ConfigurationError(f"selection.weights.{name} must be >= 0")
        if sum((self.priority, self.cost, self.latency, self.capability)) <= 0:
            raise ConfigurationError("selection.weights must not all be zero")


@dataclass(frozen=True)
class SelectionConfig:
    policy: str = "priority"
    weights: SelectionWeights = field(default_factory=SelectionWeights)
    required_capabilities: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.policy not in VALID_POLICIES:
            raise ConfigurationError(
                f"selection.policy must be one of {VALID_POLICIES}, got '{self.policy}'"
            )


@dataclass(frozen=True)
class LoggingConfig:
    level: str = "INFO"
    log_prompts: bool = False


@dataclass(frozen=True)
class RouterConfig:
    embedding: EmbeddingConfig
    classifier: ClassifierConfig
    categories: tuple[str, ...]
    selection: SelectionConfig
    models: tuple[ModelDefinition, ...]
    logging: LoggingConfig
    config_path: Optional[Path] = None


def _require_dict(value: Any, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ConfigurationError(f"'{name}' must be a mapping")
    return value


def _build_model_definition(raw: dict[str, Any]) -> ModelDefinition:
    if not isinstance(raw, dict):
        raise ModelValidationError("each entry in 'models' must be a mapping")

    try:
        pricing_raw = _require_dict(raw.get("pricing"), "models[].pricing")
        performance_raw = _require_dict(raw.get("performance"), "models[].performance")
        capabilities_raw = _require_dict(raw.get("capabilities"), "models[].capabilities")

        pricing = ModelPricing(
            input_per_1k_tokens=float(pricing_raw.get("input_per_1k_tokens", 0.0)),
            output_per_1k_tokens=float(pricing_raw.get("output_per_1k_tokens", 0.0)),
        )
        performance = ModelPerformance(
            expected_latency_ms=float(performance_raw.get("expected_latency_ms", 1.0)),
        )
        capabilities = ModelCapabilities(
            context_window=int(capabilities_raw.get("context_window", 1)),
            tool_calling=bool(capabilities_raw.get("tool_calling", False)),
            structured_output=bool(capabilities_raw.get("structured_output", False)),
        )
        categories = raw.get("categories") or []
        if not isinstance(categories, list):
            raise ModelValidationError(
                f"model '{raw.get('id')}': categories must be a list"
            )

        return ModelDefinition(
            id=str(raw.get("id", "")),
            provider=str(raw.get("provider", "")),
            display_name=str(raw.get("display_name", raw.get("id", ""))),
            categories=tuple(str(c) for c in categories),
            pricing=pricing,
            performance=performance,
            capabilities=capabilities,
            priority=int(raw.get("priority", 100)),
            enabled=bool(raw.get("enabled", True)),
        )
    except (TypeError, ValueError) as exc:
        raise ModelValidationError(
            f"model '{raw.get('id', '<unknown>')}': invalid field value ({exc})"
        ) from exc


def _validate_models(models: list[ModelDefinition], categories: tuple[str, ...]) -> None:
    seen_ids: set[str] = set()
    for model in models:
        if model.id in seen_ids:
            raise ModelValidationError(f"duplicate model id '{model.id}' in registry")
        seen_ids.add(model.id)
        unknown = [c for c in model.categories if c not in categories]
        if unknown:
            raise ModelValidationError(
                f"model '{model.id}' references unknown categories {unknown}; "
                f"configured categories are {list(categories)}"
            )


def load_config(path: Union[str, Path]) -> RouterConfig:
    """Load and validate a router configuration file.

    Raises:
        ConfigurationError: if the file is missing, malformed, or fails schema validation.
        ModelValidationError: if a model registry entry fails validation.
    """
    config_path = Path(path)
    if not config_path.is_file():
        raise ConfigurationError(f"configuration file not found: {config_path}")

    try:
        raw_text = config_path.read_text(encoding="utf-8")
        raw = yaml.safe_load(raw_text)
    except yaml.YAMLError as exc:
        raise ConfigurationError(f"failed to parse YAML config '{config_path}': {exc}") from exc

    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ConfigurationError(f"top-level config in '{config_path}' must be a mapping")

    return build_config(raw, config_path=config_path)


def build_config(raw: dict[str, Any], config_path: Optional[Path] = None) -> RouterConfig:
    """Build a validated RouterConfig from an already-parsed dict."""

    embedding_raw = _require_dict(raw.get("embedding"), "embedding")
    if "model" not in embedding_raw:
        raise ConfigurationError("embedding.model is required")
    embedding = EmbeddingConfig(
        model=str(embedding_raw["model"]),
        cache_folder=embedding_raw.get("cache_folder"),
        device=embedding_raw.get("device"),
    )

    classifier_raw = _require_dict(raw.get("classifier"), "classifier")
    if "artifact" not in classifier_raw:
        raise ConfigurationError("classifier.artifact is required")
    classifier = ClassifierConfig(
        artifact=str(classifier_raw["artifact"]),
        confidence_threshold=float(classifier_raw.get("confidence_threshold", 0.70)),
        max_prompt_length=int(classifier_raw.get("max_prompt_length", 4000)),
    )

    categories_raw = raw["categories"] if "categories" in raw else list(DEFAULT_CATEGORIES)
    if not isinstance(categories_raw, list) or not categories_raw:
        raise ConfigurationError("'categories' must be a non-empty list")
    categories = tuple(str(c) for c in categories_raw)
    if len(set(categories)) != len(categories):
        raise ConfigurationError("'categories' must not contain duplicates")

    selection_raw = _require_dict(raw.get("selection"), "selection")
    weights_raw = _require_dict(selection_raw.get("weights"), "selection.weights")
    selection = SelectionConfig(
        policy=str(selection_raw.get("policy", "priority")),
        weights=SelectionWeights(
            priority=float(weights_raw.get("priority", 0.40)),
            cost=float(weights_raw.get("cost", 0.30)),
            latency=float(weights_raw.get("latency", 0.20)),
            capability=float(weights_raw.get("capability", 0.10)),
        ),
        required_capabilities=_require_dict(
            selection_raw.get("required_capabilities"), "selection.required_capabilities"
        ),
    )

    logging_raw = _require_dict(raw.get("logging"), "logging")
    logging_config = LoggingConfig(
        level=str(logging_raw.get("level", "INFO")),
        log_prompts=bool(logging_raw.get("log_prompts", False)),
    )

    models_raw = raw.get("models") or []
    if not isinstance(models_raw, list):
        raise ConfigurationError("'models' must be a list")
    models = tuple(_build_model_definition(m) for m in models_raw)
    _validate_models(list(models), categories)

    return RouterConfig(
        embedding=embedding,
        classifier=classifier,
        categories=categories,
        selection=selection,
        models=models,
        logging=logging_config,
        config_path=config_path,
    )
