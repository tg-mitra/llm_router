import builtins

import numpy as np
import pytest

from llm_router.embeddings import Embedder, SentenceTransformerEmbedder
from llm_router.exceptions import EmbeddingError

from .._fakes import FakeEmbedder


class TestEmbedderProtocol:
    def test_fake_embedder_satisfies_protocol(self):
        assert isinstance(FakeEmbedder(), Embedder)

    def test_encode_batch_shape(self):
        embedder = FakeEmbedder(dim=8)
        batch = embedder.encode_batch(["hello", "world"])
        assert batch.shape == (2, 8)

    def test_encode_single_matches_batch_row(self):
        embedder = FakeEmbedder(dim=8)
        single = embedder.encode("hello")
        batch = embedder.encode_batch(["hello"])
        assert np.allclose(single, batch[0])

    def test_deterministic(self):
        embedder = FakeEmbedder()
        assert np.array_equal(embedder.encode("same text"), embedder.encode("same text"))


class TestSentenceTransformerEmbedder:
    def test_lazy_load_does_not_run_on_construction(self, monkeypatch):
        def fail_import(name, *args, **kwargs):
            if name == "sentence_transformers":
                raise ImportError("should not be imported yet")
            return real_import(name, *args, **kwargs)

        real_import = builtins.__import__
        monkeypatch.setattr(builtins, "__import__", fail_import)

        # Construction must not import sentence_transformers.
        SentenceTransformerEmbedder(model_name="fake/model")

    def test_import_failure_raises_embedding_error(self, monkeypatch):
        def fail_import(name, *args, **kwargs):
            if name == "sentence_transformers":
                raise ImportError("simulated missing dependency")
            return real_import(name, *args, **kwargs)

        real_import = builtins.__import__
        monkeypatch.setattr(builtins, "__import__", fail_import)

        embedder = SentenceTransformerEmbedder(model_name="fake/model")
        with pytest.raises(EmbeddingError):
            embedder.encode("hello")

    def test_model_load_failure_wrapped_as_embedding_error(self, monkeypatch):
        import sys
        import types

        fake_module = types.ModuleType("sentence_transformers")

        class BoomSentenceTransformer:
            def __init__(self, *args, **kwargs):
                raise RuntimeError("boom")

        fake_module.SentenceTransformer = BoomSentenceTransformer
        monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)

        embedder = SentenceTransformerEmbedder(model_name="fake/model")
        with pytest.raises(EmbeddingError):
            embedder.encode("hello")
