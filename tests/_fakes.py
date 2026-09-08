"""Deterministic test doubles shared across unit and integration tests.

Keeping these out of the real sentence-transformers / sklearn training loop
makes the test suite fast and network-independent.
"""

from __future__ import annotations

import hashlib
from typing import Sequence

import numpy as np


class FakeEmbedder:
    """Deterministic, hash-based embedder -- no model download required."""

    def __init__(self, dim: int = 16) -> None:
        self.dim = dim

    def _vector(self, text: str) -> np.ndarray:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        raw = np.frombuffer((digest * ((self.dim // len(digest)) + 1))[: self.dim], dtype=np.uint8)
        return raw.astype(np.float64) / 255.0

    def encode(self, text: str) -> np.ndarray:
        return self._vector(text)

    def encode_batch(self, texts: Sequence[str]) -> np.ndarray:
        return np.asarray([self._vector(t) for t in texts])


class FakeSklearnModel:
    """Stub classifier with a fixed, configurable predict_proba output."""

    def __init__(self, classes: list[str], fixed_probabilities: list[float]) -> None:
        assert len(classes) == len(fixed_probabilities)
        self.classes_ = list(classes)
        self._probs = list(fixed_probabilities)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return np.asarray([self._probs for _ in range(len(X))])
