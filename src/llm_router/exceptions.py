"""Typed exceptions for the llm_router package."""

from __future__ import annotations


class LLMRouterError(Exception):
    """Base class for all llm_router errors."""


class ConfigurationError(LLMRouterError):
    """Raised when configuration is missing, malformed, or fails validation."""


class InvalidPromptError(LLMRouterError):
    """Raised when a prompt fails input validation."""


class ClassifierArtifactError(LLMRouterError):
    """Raised when a classifier artifact cannot be located or loaded."""


class EmbeddingError(LLMRouterError):
    """Raised when the embedding model fails to load or encode text."""


class UnsupportedCategoryError(LLMRouterError):
    """Raised when a category is not part of the configured category set."""


class NoEligibleModelsError(LLMRouterError):
    """Raised when no registry model is eligible for the requested category."""


class ModelValidationError(LLMRouterError):
    """Raised when model registry metadata fails validation."""
