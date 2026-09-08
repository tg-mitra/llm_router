import joblib
import pytest

from llm_router.classifier import LOW_CONFIDENCE_REASON, UNKNOWN_CATEGORY, TaskClassifier
from llm_router.exceptions import ClassifierArtifactError, InvalidPromptError

from .._fakes import FakeEmbedder, FakeSklearnModel

CLASSES = ["coding", "simple"]


def make_classifier(probabilities, threshold=0.7, max_len=4000):
    model = FakeSklearnModel(classes=CLASSES, fixed_probabilities=probabilities)
    return TaskClassifier(
        embedder=FakeEmbedder(),
        sklearn_model=model,
        classes=CLASSES,
        confidence_threshold=threshold,
        max_prompt_length=max_len,
    )


class TestPredictProba:
    def test_returns_full_distribution(self):
        clf = make_classifier([0.9, 0.1])
        scores = clf.predict_proba("write a function")
        assert scores == {"coding": 0.9, "simple": 0.1}


class TestPredict:
    def test_confident_prediction(self):
        clf = make_classifier([0.9, 0.1], threshold=0.7)
        result = clf.predict("write a function")
        assert result.category == "coding"
        assert result.confidence == pytest.approx(0.9)
        assert result.reason is None
        assert result.is_confident

    def test_low_confidence_returns_unknown(self):
        clf = make_classifier([0.55, 0.45], threshold=0.7)
        result = clf.predict("ambiguous prompt")
        assert result.category == UNKNOWN_CATEGORY
        assert result.reason == LOW_CONFIDENCE_REASON
        assert not result.is_confident
        assert result.confidence == pytest.approx(0.55)

    def test_boundary_confidence_is_accepted(self):
        clf = make_classifier([0.70, 0.30], threshold=0.70)
        result = clf.predict("exact threshold")
        assert result.category == "coding"
        assert result.reason is None


class TestValidatePrompt:
    def test_non_string_rejected(self):
        clf = make_classifier([0.9, 0.1])
        with pytest.raises(InvalidPromptError):
            clf.validate_prompt(123)  # type: ignore[arg-type]

    def test_empty_string_rejected(self):
        clf = make_classifier([0.9, 0.1])
        with pytest.raises(InvalidPromptError):
            clf.validate_prompt("   ")

    def test_too_long_rejected(self):
        clf = make_classifier([0.9, 0.1], max_len=10)
        with pytest.raises(InvalidPromptError):
            clf.validate_prompt("this prompt is definitely too long")

    def test_trims_whitespace(self):
        clf = make_classifier([0.9, 0.1])
        assert clf.validate_prompt("  hello  ") == "hello"


class TestFromArtifact:
    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(ClassifierArtifactError):
            TaskClassifier.from_artifact(
                tmp_path / "missing.joblib", embedder=FakeEmbedder()
            )

    def test_non_dict_payload_raises(self, tmp_path):
        path = tmp_path / "bad.joblib"
        joblib.dump(["not", "a", "dict"], path)
        with pytest.raises(ClassifierArtifactError):
            TaskClassifier.from_artifact(path, embedder=FakeEmbedder())

    def test_missing_required_keys_raises(self, tmp_path):
        path = tmp_path / "bad.joblib"
        joblib.dump({"classifier": FakeSklearnModel(CLASSES, [0.5, 0.5])}, path)
        with pytest.raises(ClassifierArtifactError):
            TaskClassifier.from_artifact(path, embedder=FakeEmbedder())

    def test_valid_artifact_round_trips(self, tmp_path):
        path = tmp_path / "clf.joblib"
        model = FakeSklearnModel(CLASSES, [0.8, 0.2])
        joblib.dump(
            {"classifier": model, "classes": CLASSES, "embedding_model": "fake/model"}, path
        )
        clf = TaskClassifier.from_artifact(path, embedder=FakeEmbedder(), confidence_threshold=0.5)
        result = clf.predict("write some code")
        assert result.category == "coding"
        assert clf.classes == CLASSES
