"""Training pipeline: dataset -> embeddings -> LogisticRegression classifier.

Usage:
    python training/train.py \
        --dataset data/training_data.jsonl \
        --output-dir model \
        --categories simple coding reasoning security summarization

Produces, in --output-dir:
    classifier.joblib      (sklearn model + label list + embedding model name, for TaskClassifier.from_artifact)
    metadata.json          (Hugging Face-portable metadata, requirement 16)
    label_mapping.json
    training_config.json
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from dotenv import load_dotenv
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

TRAINING_DIR = Path(__file__).resolve().parent
MAIN_DIR = TRAINING_DIR.parent
sys.path.insert(0, str(MAIN_DIR / "src"))

# Loads HF_TOKEN (see .env.example) so huggingface_hub authenticates when
# downloading the embedding model, avoiding the unauthenticated rate limit.
# Training-only concern -- the runtime llm_router package never touches .env.
load_dotenv(MAIN_DIR / ".env")

from dataset_validation import (  # noqa: E402
    Example,
    check_split_leakage,
    load_jsonl,
    validate_examples,
)
from evaluate import evaluate_predictions  # noqa: E402

from llm_router.embeddings import SentenceTransformerEmbedder  # noqa: E402

DEFAULT_CATEGORIES = ["simple", "coding", "reasoning", "security", "summarization"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the llm_router task classifier.")
    parser.add_argument("--dataset", type=Path, default=MAIN_DIR / "data" / "training_data.jsonl")
    parser.add_argument("--output-dir", type=Path, default=MAIN_DIR / "model")
    parser.add_argument("--categories", nargs="+", default=DEFAULT_CATEGORIES)
    parser.add_argument(
        "--embedding-model", type=str, default="sentence-transformers/all-MiniLM-L6-v2"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument(
        "--min-accuracy",
        type=float,
        default=0.75,
        help="Minimum acceptable test-set accuracy (release acceptance criterion, requirement 7.5).",
    )
    parser.add_argument(
        "--min-macro-f1",
        type=float,
        default=0.70,
        help="Minimum acceptable test-set macro F1 (release acceptance criterion, requirement 7.5).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Save the artifact even if acceptance criteria are not met.",
    )
    return parser.parse_args()


def _dataset_version(path: Path) -> str:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return digest[:16]


def _dependency_versions(packages: list[str]) -> dict[str, str]:
    versions = {}
    for pkg in packages:
        try:
            versions[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            versions[pkg] = "unknown"
    return versions


def main() -> int:
    args = parse_args()
    categories = list(args.categories)

    print(f"loading dataset from {args.dataset}")
    examples = load_jsonl(args.dataset)

    report = validate_examples(examples, known_labels=categories)
    for warning in report.warnings:
        print(f"[warning] {warning}")
    if not report.is_valid:
        print("dataset validation failed:")
        for error in report.errors:
            print(f"  - {error}")
        return 1

    valid_examples = [e for e in examples if e.label in categories and e.text.strip()]
    texts = [e.text.strip() for e in valid_examples]
    labels = [e.label for e in valid_examples]

    # train/val/test stratified split (requirement 7.3), seed recorded in metadata.
    train_texts, temp_texts, train_labels, temp_labels = train_test_split(
        texts,
        labels,
        test_size=(args.test_size + args.val_size),
        random_state=args.seed,
        stratify=labels,
    )
    relative_test_size = args.test_size / (args.test_size + args.val_size)
    val_texts, test_texts, val_labels, test_labels = train_test_split(
        temp_texts,
        temp_labels,
        test_size=relative_test_size,
        random_state=args.seed,
        stratify=temp_labels,
    )

    leakage_report = check_split_leakage(
        [Example(t, l) for t, l in zip(train_texts, train_labels)],
        [Example(t, l) for t, l in zip(val_texts, val_labels)],
        [Example(t, l) for t, l in zip(test_texts, test_labels)],
        names=["train", "val", "test"],
    )
    if not leakage_report.is_valid:
        print("train/val/test leakage detected:")
        for error in leakage_report.errors:
            print(f"  - {error}")
        return 1

    print(
        f"split sizes: train={len(train_texts)} val={len(val_texts)} test={len(test_texts)}"
    )

    print(f"loading embedding model: {args.embedding_model}")
    embedder = SentenceTransformerEmbedder(model_name=args.embedding_model)

    print("encoding splits...")
    X_train = embedder.encode_batch(train_texts)
    X_val = embedder.encode_batch(val_texts)
    X_test = embedder.encode_batch(test_texts)

    sorted_categories = sorted(categories)

    print("fitting LogisticRegression classifier...")
    clf = LogisticRegression(
        max_iter=args.max_iter,
        random_state=args.seed,
        class_weight="balanced",
    )
    clf.fit(X_train, train_labels)

    # sklearn orders predict_proba columns by clf.classes_, not our category list.
    class_order = list(clf.classes_)

    val_pred = clf.predict(X_val)
    val_report = evaluate_predictions(val_labels, val_pred, labels=sorted_categories)
    print(f"validation: accuracy={val_report.accuracy:.4f} macro_f1={val_report.macro_f1:.4f}")

    test_pred = clf.predict(X_test)
    test_report = evaluate_predictions(test_labels, test_pred, labels=sorted_categories)
    print(f"test:       accuracy={test_report.accuracy:.4f} macro_f1={test_report.macro_f1:.4f}")

    meets_acceptance = (
        test_report.accuracy >= args.min_accuracy and test_report.macro_f1 >= args.min_macro_f1
    )
    status = "PASS" if meets_acceptance else "FAIL"
    print(
        f"acceptance criteria [{status}]: "
        f"accuracy>={args.min_accuracy} (got {test_report.accuracy:.4f}), "
        f"macro_f1>={args.min_macro_f1} (got {test_report.macro_f1:.4f})"
    )

    if not meets_acceptance and not args.force:
        print("refusing to save artifact (use --force to override).")
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)

    artifact_path = args.output_dir / "classifier.joblib"
    joblib.dump(
        {
            "classifier": clf,
            "classes": class_order,
            "embedding_model": args.embedding_model,
        },
        artifact_path,
    )
    print(f"saved classifier artifact to {artifact_path}")

    label_mapping = {label: idx for idx, label in enumerate(class_order)}
    (args.output_dir / "label_mapping.json").write_text(
        json.dumps(label_mapping, indent=2), encoding="utf-8"
    )

    training_config = {
        "dataset": str(args.dataset),
        "categories": sorted_categories,
        "seed": args.seed,
        "test_size": args.test_size,
        "val_size": args.val_size,
        "embedding_model": args.embedding_model,
        "classifier": "sklearn.linear_model.LogisticRegression",
        "classifier_params": {
            "max_iter": args.max_iter,
            "class_weight": "balanced",
            "random_state": args.seed,
        },
        "split_sizes": {
            "train": len(train_texts),
            "val": len(val_texts),
            "test": len(test_texts),
        },
    }
    (args.output_dir / "training_config.json").write_text(
        json.dumps(training_config, indent=2), encoding="utf-8"
    )

    metadata = {
        "classifier_type": "sklearn.linear_model.LogisticRegression",
        "embedding_model": args.embedding_model,
        "label_set": class_order,
        "training_dataset_version": _dataset_version(args.dataset),
        "training_timestamp": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "dependency_versions": _dependency_versions(
            ["scikit-learn", "sentence-transformers", "numpy", "joblib", "torch"]
        ),
        "random_seed": args.seed,
        "acceptance_criteria": {
            "min_accuracy": args.min_accuracy,
            "min_macro_f1": args.min_macro_f1,
            "met": meets_acceptance,
        },
        "evaluation": {
            "validation": val_report.to_dict(),
            "test": test_report.to_dict(),
        },
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(f"saved metadata.json, label_mapping.json, training_config.json to {args.output_dir}")

    return 0 if meets_acceptance else 1


if __name__ == "__main__":
    raise SystemExit(main())
