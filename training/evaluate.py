"""Evaluation metrics for the trained task classifier (requirement 7.4)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)


@dataclass
class EvaluationReport:
    accuracy: float
    macro_f1: float
    labels: list[str]
    per_class: dict[str, dict[str, float]] = field(default_factory=dict)
    confusion_matrix: list[list[int]] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "accuracy": self.accuracy,
            "macro_f1": self.macro_f1,
            "labels": self.labels,
            "per_class": self.per_class,
            "confusion_matrix": self.confusion_matrix,
        }


def evaluate_predictions(
    y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str]
) -> EvaluationReport:
    """Compute accuracy, macro F1, per-class precision/recall, and confusion matrix."""
    labels = list(labels)

    accuracy = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )

    per_class = {
        label: {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i, label in enumerate(labels)
    }

    cm = confusion_matrix(y_true, y_pred, labels=labels).tolist()

    return EvaluationReport(
        accuracy=accuracy,
        macro_f1=macro_f1,
        labels=labels,
        per_class=per_class,
        confusion_matrix=cm,
    )
