"""llm_router: prompt classification and LLM model selection for agentic AI systems."""

from .classifier import TaskClassifier
from .config import RouterConfig, load_config
from .embeddings import Embedder, SentenceTransformerEmbedder
from .exceptions import (
    ClassifierArtifactError,
    ConfigurationError,
    EmbeddingError,
    InvalidPromptError,
    LLMRouterError,
    ModelValidationError,
    NoEligibleModelsError,
    UnsupportedCategoryError,
)
from .registry import ModelRegistry
from .router import Router
from .schemas import (
    CandidateScore,
    ClassificationResult,
    ModelCapabilities,
    ModelDefinition,
    ModelPerformance,
    ModelPricing,
    RoutingResult,
)
from .scoring import estimate_cost
from .selector import Selector
from .version import __version__

__all__ = [
    "__version__",
    "Router",
    "RouterConfig",
    "load_config",
    "TaskClassifier",
    "Embedder",
    "SentenceTransformerEmbedder",
    "ModelRegistry",
    "Selector",
    "estimate_cost",
    "ClassificationResult",
    "ModelDefinition",
    "ModelPricing",
    "ModelPerformance",
    "ModelCapabilities",
    "CandidateScore",
    "RoutingResult",
    "LLMRouterError",
    "ConfigurationError",
    "InvalidPromptError",
    "ClassifierArtifactError",
    "EmbeddingError",
    "UnsupportedCategoryError",
    "NoEligibleModelsError",
    "ModelValidationError",
]
