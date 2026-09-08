import pytest

from llm_router.config import build_config, load_config
from llm_router.exceptions import ConfigurationError, ModelValidationError

VALID_RAW = {
    "embedding": {"model": "sentence-transformers/all-MiniLM-L6-v2"},
    "classifier": {"artifact": "model/classifier.joblib", "confidence_threshold": 0.7},
    "categories": ["simple", "coding"],
    "selection": {
        "policy": "weighted",
        "weights": {"priority": 0.4, "cost": 0.3, "latency": 0.2, "capability": 0.1},
    },
    "models": [
        {
            "id": "model_a",
            "provider": "provider_a",
            "display_name": "Model A",
            "categories": ["simple"],
            "pricing": {"input_per_1k_tokens": 0.01, "output_per_1k_tokens": 0.02},
            "performance": {"expected_latency_ms": 500},
            "capabilities": {"context_window": 32000, "tool_calling": True},
            "priority": 1,
            "enabled": True,
        }
    ],
}


class TestBuildConfig:
    def test_valid_config_parses(self):
        config = build_config(VALID_RAW)
        assert config.embedding.model == "sentence-transformers/all-MiniLM-L6-v2"
        assert config.classifier.confidence_threshold == 0.7
        assert config.categories == ("simple", "coding")
        assert config.selection.policy == "weighted"
        assert len(config.models) == 1
        assert config.models[0].id == "model_a"

    def test_missing_embedding_model_raises(self):
        raw = {**VALID_RAW, "embedding": {}}
        with pytest.raises(ConfigurationError):
            build_config(raw)

    def test_missing_classifier_artifact_raises(self):
        raw = {**VALID_RAW, "classifier": {}}
        with pytest.raises(ConfigurationError):
            build_config(raw)

    def test_invalid_confidence_threshold_raises(self):
        raw = {**VALID_RAW, "classifier": {"artifact": "a.joblib", "confidence_threshold": 1.5}}
        with pytest.raises(ConfigurationError):
            build_config(raw)

    def test_empty_categories_raises(self):
        raw = {**VALID_RAW, "categories": []}
        with pytest.raises(ConfigurationError):
            build_config(raw)

    def test_duplicate_categories_raises(self):
        raw = {**VALID_RAW, "categories": ["simple", "simple"]}
        with pytest.raises(ConfigurationError):
            build_config(raw)

    def test_invalid_policy_raises(self):
        raw = {**VALID_RAW, "selection": {"policy": "random"}}
        with pytest.raises(ConfigurationError):
            build_config(raw)

    def test_all_zero_weights_raises(self):
        raw = {
            **VALID_RAW,
            "selection": {
                "policy": "weighted",
                "weights": {"priority": 0, "cost": 0, "latency": 0, "capability": 0},
            },
        }
        with pytest.raises(ConfigurationError):
            build_config(raw)

    def test_duplicate_model_id_raises(self):
        raw = {**VALID_RAW, "models": [VALID_RAW["models"][0], VALID_RAW["models"][0]]}
        with pytest.raises(ModelValidationError):
            build_config(raw)

    def test_model_unknown_category_raises(self):
        bad_model = {**VALID_RAW["models"][0], "categories": ["not_a_real_category"]}
        raw = {**VALID_RAW, "models": [bad_model]}
        with pytest.raises(ModelValidationError):
            build_config(raw)

    def test_models_defaults_to_empty_list(self):
        raw = {**VALID_RAW, "models": []}
        config = build_config(raw)
        assert config.models == ()


class TestLoadConfig:
    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(ConfigurationError):
            load_config(tmp_path / "does_not_exist.yml")

    def test_malformed_yaml_raises(self, tmp_path):
        bad_file = tmp_path / "bad.yml"
        bad_file.write_text("embedding: [unterminated", encoding="utf-8")
        with pytest.raises(ConfigurationError):
            load_config(bad_file)

    def test_valid_yaml_file_loads(self, tmp_path):
        import yaml

        config_file = tmp_path / "config.yml"
        config_file.write_text(yaml.safe_dump(VALID_RAW), encoding="utf-8")
        config = load_config(config_file)
        assert config.config_path == config_file
        assert config.models[0].id == "model_a"
