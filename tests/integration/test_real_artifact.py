"""Regression test against the real trained artifact (requirement 15).

Loads the actual `model/classifier.joblib` produced by `training/train.py`
via `Router.from_config`, and checks it against a small fixed set of
representative prompts per category (`data/regression_dataset.jsonl`).

Skipped automatically if the artifact has not been trained yet -- run
`python training/train.py` first to produce it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from llm_router import Router

MAIN_DIR = Path(__file__).resolve().parent.parent.parent
ARTIFACT_PATH = MAIN_DIR / "model" / "classifier.joblib"
CONFIG_PATH = MAIN_DIR / "config.example.yml"
REGRESSION_DATASET = MAIN_DIR / "data" / "regression_dataset.jsonl"

pytestmark = pytest.mark.skipif(
    not ARTIFACT_PATH.is_file(),
    reason="classifier artifact not found; run training/train.py first",
)


@pytest.fixture(scope="module")
def router() -> Router:
    return Router.from_config(CONFIG_PATH)


def _load_regression_examples() -> list[dict]:
    examples = []
    with REGRESSION_DATASET.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


class TestRegressionDataset:
    def test_regression_dataset_exists_and_is_non_empty(self):
        examples = _load_regression_examples()
        assert len(examples) > 0

    def test_classifier_accuracy_on_regression_set_meets_floor(self, router: Router):
        examples = _load_regression_examples()
        correct = 0
        for example in examples:
            result = router.classifier.predict(example["text"])
            if result.category == example["label"]:
                correct += 1
        accuracy = correct / len(examples)
        # Regression floor, not a release acceptance bar -- catches gross
        # classifier regressions between releases without being flaky on
        # individual borderline prompts.
        assert accuracy >= 0.5, f"regression-set accuracy dropped to {accuracy:.2f}"

    def test_router_route_returns_structured_result_for_each_prompt(self, router: Router):
        examples = _load_regression_examples()
        for example in examples:
            result = router.route(example["text"])
            assert result.category in router.config.categories or result.category == "unknown"
            assert 0.0 <= result.confidence <= 1.0
