"""Training dataset validation (requirement 7.2).

This module is only used by the offline training pipeline -- it is not part
of the runtime `llm_router` package and has no dependency on it.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence, Union


class DatasetValidationError(Exception):
    """Raised when a training dataset fails validation."""


@dataclass
class Example:
    text: str
    label: str


@dataclass
class ValidationReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return not self.errors

    def raise_if_invalid(self) -> None:
        if not self.is_valid:
            raise DatasetValidationError(
                "dataset failed validation:\n" + "\n".join(f"  - {e}" for e in self.errors)
            )


def load_jsonl(path: Union[str, Path]) -> list[Example]:
    """Load a JSONL dataset of {"text": ..., "label": ...} records."""
    file_path = Path(path)
    if not file_path.is_file():
        raise DatasetValidationError(f"dataset file not found: {file_path}")

    examples: list[Example] = []
    with file_path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DatasetValidationError(
                    f"{file_path}:{line_no}: invalid JSON ({exc})"
                ) from exc
            text = record.get("text")
            label = record.get("label")
            examples.append(Example(text=text, label=label))
    return examples


def validate_examples(
    examples: Sequence[Example],
    known_labels: Iterable[str],
    min_examples_per_label: int = 5,
    max_imbalance_ratio: float = 5.0,
) -> ValidationReport:
    """Validate structural correctness and class balance of a dataset.

    Checks: empty text, missing/unknown labels, duplicate (text) entries
    (with conflicting-label duplicates treated as errors), and severe class
    imbalance across labels.
    """
    report = ValidationReport()
    known = set(known_labels)

    seen_text_to_labels: dict[str, set[str]] = {}
    label_counts: Counter[str] = Counter()

    for idx, example in enumerate(examples):
        if not isinstance(example.text, str) or not example.text.strip():
            report.errors.append(f"example[{idx}]: empty or non-string 'text'")
            continue

        if not isinstance(example.label, str) or not example.label.strip():
            report.errors.append(f"example[{idx}]: missing or non-string 'label'")
            continue

        if example.label not in known:
            report.errors.append(
                f"example[{idx}]: unknown label '{example.label}' "
                f"(expected one of {sorted(known)})"
            )
            continue

        normalized_text = example.text.strip()
        seen_text_to_labels.setdefault(normalized_text, set()).add(example.label)
        label_counts[example.label] += 1

    for text, labels in seen_text_to_labels.items():
        if len(labels) > 1:
            report.errors.append(
                f"conflicting labels {sorted(labels)} for duplicate text: {text!r}"
            )

    duplicate_count = sum(1 for _ in seen_text_to_labels)
    total_valid = sum(label_counts.values())
    if total_valid and (total_valid - duplicate_count) > 0:
        report.warnings.append(
            f"{total_valid - duplicate_count} duplicate example(s) found (same text, same label)"
        )

    for label in known:
        count = label_counts.get(label, 0)
        if count == 0:
            report.errors.append(f"label '{label}' has no training examples")
        elif count < min_examples_per_label:
            report.warnings.append(
                f"label '{label}' has only {count} example(s), "
                f"below the recommended minimum of {min_examples_per_label}"
            )

    if label_counts:
        max_count = max(label_counts.values())
        min_count = min(label_counts.values())
        if min_count > 0 and (max_count / min_count) > max_imbalance_ratio:
            report.warnings.append(
                f"severe class imbalance: max class count {max_count} is more than "
                f"{max_imbalance_ratio}x the min class count {min_count}"
            )

    return report


def check_split_leakage(
    *splits: Sequence[Example], names: Sequence[str] | None = None
) -> ValidationReport:
    """Detect exact-text overlap between two or more dataset splits (e.g. train/test)."""
    report = ValidationReport()
    split_names = list(names) if names else [f"split_{i}" for i in range(len(splits))]

    text_sets = [
        {e.text.strip() for e in split if isinstance(e.text, str)} for split in splits
    ]
    for i in range(len(text_sets)):
        for j in range(i + 1, len(text_sets)):
            overlap = text_sets[i] & text_sets[j]
            if overlap:
                report.errors.append(
                    f"train/test leakage: {len(overlap)} example(s) shared between "
                    f"'{split_names[i]}' and '{split_names[j]}'"
                )
    return report
