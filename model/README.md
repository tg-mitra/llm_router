---
tags:
  - text-classification
  - sentence-transformers
  - scikit-learn
  - llm-routing
library_name: scikit-learn
---

# llm-router task classifier

A `LogisticRegression` classifier over `sentence-transformers/all-MiniLM-L6-v2`
embeddings that predicts which task category a prompt belongs to, for use
with the [`llm-router`](https://github.com/tathagatamitra/model_router)
Python package. It predicts a **task category only** -- it has no knowledge
of any specific LLM provider or model name.

## Labels

`coding`, `reasoning`, `security`, `simple`, `summarization`

## Files

| File | Purpose |
|---|---|
| `classifier.joblib` | `{"classifier": <sklearn LogisticRegression>, "classes": [...], "embedding_model": "..."}` -- load with `llm_router.TaskClassifier.from_artifact(...)`. |
| `metadata.json` | Classifier type, embedding model, label set, dataset version, training timestamp, Python/dependency versions, random seed, evaluation metrics. |
| `label_mapping.json` | `{label: index}` mapping matching `classifier.classes_` order. |
| `training_config.json` | Dataset path, split sizes, seed, and classifier hyperparameters used to produce this artifact. |

## Usage

```python
from llm_router.embeddings import SentenceTransformerEmbedder
from llm_router.classifier import TaskClassifier

embedder = SentenceTransformerEmbedder("sentence-transformers/all-MiniLM-L6-v2")
classifier = TaskClassifier.from_artifact(
    "classifier.joblib", embedder=embedder, confidence_threshold=0.35
)

result = classifier.predict("Explain what a Python decorator does")
print(result.category, result.confidence)
```

## Training data

Trained on a ~175-example seed dataset (35 examples per category), see
`training/build_dataset.py` and `data/training_data.jsonl` in the source
repository. This is a **baseline** artifact intended to demonstrate the
pipeline end-to-end; expand the training dataset for production-quality
accuracy on out-of-distribution phrasing.

## Evaluation

See `metadata.json -> evaluation` for the full per-class precision/recall/F1
and confusion matrix on the held-out validation and test splits. Summary:

- Validation: accuracy 0.885, macro F1 0.882
- Test: accuracy 1.000, macro F1 1.000 (small held-out split; expect lower
  accuracy on prompts phrased differently from the training set -- see
  `docs/configuration.md` in the source repo for confidence threshold
  calibration guidance).

## Reproducing

```bash
python training/train.py --dataset data/training_data.jsonl --output-dir model
```

Random seed, split sizes, and full dependency versions are recorded in
`metadata.json` for reproducibility.

## License

MIT (matches the parent `llm-router` package).
