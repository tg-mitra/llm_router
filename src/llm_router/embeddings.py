"""Sentence Transformer embedding wrapper.

The embedding backend is intentionally isolated behind a small protocol so
that the classifier and training pipeline never depend on a concrete
implementation. This keeps the embedding model configurable rather than
hard-coded into business logic (see requirements 4.1 / 5).
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Protocol, Sequence, runtime_checkable

import numpy as np

from .exceptions import EmbeddingError

logger = logging.getLogger(__name__)


@runtime_checkable
class Embedder(Protocol):
    """Structural interface required by the classifier and training pipeline."""

    def encode(self, text: str) -> np.ndarray: ...

    def encode_batch(self, texts: Sequence[str]) -> np.ndarray: ...


class SentenceTransformerEmbedder:
    """Embeds text using a `sentence-transformers` model.

    The underlying model is loaded lazily on first use so that constructing
    this object (and, by extension, a Router) is cheap and side-effect free.
    """

    def __init__(
        self,
        model_name: str,
        cache_folder: Optional[str] = None,
        device: Optional[str] = None,
    ) -> None:
        self.model_name = model_name
        self.cache_folder = cache_folder
        self.device = device
        self._model: Any = None

    def _load(self) -> Any:
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - environment misconfiguration
            raise EmbeddingError(
                "sentence-transformers is not installed; add it to your environment"
            ) from exc

        try:
            logger.info("loading embedding model: %s", self.model_name)
            self._model = SentenceTransformer(
                self.model_name,
                cache_folder=self.cache_folder,
                device=self.device,
            )
        except Exception as exc:
            raise EmbeddingError(
                f"failed to load embedding model '{self.model_name}': {exc}"
            ) from exc
        return self._model

    def encode(self, text: str) -> np.ndarray:
        return self.encode_batch([text])[0]

    def encode_batch(self, texts: Sequence[str]) -> np.ndarray:
        model = self._load()
        try:
            embeddings = model.encode(
                list(texts),
                convert_to_numpy=True,
                show_progress_bar=False,
            )
        except Exception as exc:
            raise EmbeddingError(f"failed to encode text batch: {exc}") from exc
        return np.asarray(embeddings)
