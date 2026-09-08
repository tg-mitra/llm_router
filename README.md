# llm-router

A standalone Python library that classifies a prompt into a task category
and recommends the best-fit LLM from a configured model registry.

It answers one question: **given a prompt and a list of available models,
which model is the best fit for this task?** It does not call any LLM, does
not know about API keys, and does not monitor whether a model is currently
online -- see [docs/architecture.md](docs/architecture.md) for the full
scope boundary.

```text
Prompt -> Sentence Transformer -> Embedding -> Task Classifier
       -> Category + Confidence -> Registry Filter -> Ranked Candidates
       -> Selected Model
```

## Getting started (end-to-end)

Everything below runs from inside `model_router/`. See the sections further down
for details on each step.

```bash
# 1. Create and activate a virtual environment local to model_router/
python -m venv .venv
source .venv/bin/activate        # Windows (PowerShell): .venv\Scripts\Activate.ps1

# 2. Install dependencies + the package itself
pip install -r requirements.txt
pip install -e .

# 3. Optional: set HF_TOKEN for faster/authenticated embedding-model downloads
cp .env.example .env             # then edit .env

# 4. Train the classifier (produces model/classifier.joblib + metadata)
python training/train.py

# 5. Copy the example config (edit the models: registry for your real deployment)
cp config.example.yml config.yml

# 6. Run the test suite
python -m pytest

# 7. Smoke-test the router directly
python -c "from llm_router import Router; r = Router.from_config('config.yml'); print(r.route('Write a function to reverse a string'))"
```

Without activating the venv, prefix each command with its interpreter
instead: `.venv/bin/python`/`.venv/bin/pip` (macOS/Linux) or
`.venv\Scripts\python.exe`/`.venv\Scripts\pip.exe` (Windows).

## Installation

Requires Python 3.10+. Create a virtual environment inside `model_router/` (this is
what the commands below and `.gitignore` assume -- `.venv` here is local to
this package, not shared with anything else in the repository):

```bash
# macOS / Linux
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .

# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
```

`requirements.txt` pins everything needed to train the classifier and run
the test suite. `pip install -e .` installs the `llm_router` package itself
(editable, so source edits take effect immediately).

Optional extra: `.[huggingface]` pulls in `huggingface_hub` for loading
artifacts from a HF repo instead of a local path.

The commands in the rest of this README assume an activated venv (plain
`python`, `pytest`, etc.). Without activating, call the venv's interpreter
directly instead: `.venv/bin/python` (macOS/Linux) or
`.venv\Scripts\python.exe` (Windows).

## Quickstart

```python
from llm_router import Router

router = Router.from_config("config.yml")  # copy config.example.yml first

result = router.route("Analyze this Python code and identify race conditions")

print(result.category)     # "coding"
print(result.confidence)   # 0.62
print(result.model_id)     # "code-specialist"
print(result.reason)       # "selected"
for candidate in result.candidates:
    print(candidate.model_id, candidate.final_score)
```

`router.route()` never raises for "not confident enough" or "no eligible
model" -- both are reported via `result.reason` with `result.model_id is
None`. It does raise typed exceptions (`llm_router.InvalidPromptError`,
`ConfigurationError`, etc.) for actionable misuse -- see
[docs/architecture.md](docs/architecture.md#error-handling-philosophy).

### Lower-level API

```python
from llm_router import ModelRegistry, Selector, TaskClassifier, SentenceTransformerEmbedder

embedder = SentenceTransformerEmbedder("sentence-transformers/all-MiniLM-L6-v2")
classifier = TaskClassifier.from_artifact("model/classifier.joblib", embedder=embedder)

result = classifier.predict("Explain what a Python decorator does")
proba = classifier.predict_proba("Explain what a Python decorator does")
```

### Cost estimation

```python
from llm_router import estimate_cost

model = router.registry.get_model("code-specialist")
cost = estimate_cost(model, input_tokens=1000, output_tokens=500)
```

This is a configuration-derived estimate only -- actual billed usage must
come from your LLM provider.

## Configuration

Copy `config.example.yml` to `config.yml` and edit the `models:` registry
for your deployment. Full field reference:
[docs/configuration.md](docs/configuration.md). Never put API keys in this
file.

## Training the classifier

A baseline classifier artifact is checked into `model/` (trained on the
~175-example seed dataset in `data/training_data.jsonl`). To retrain:

Optional: copy `.env.example` to `.env` and set `HF_TOKEN` (a Hugging Face
"Read" token) to avoid unauthenticated-request rate limits when downloading
the embedding model. `.env` is gitignored and only ever read by
`training/train.py` -- the runtime `llm_router` package never touches it.

```bash
python training/train.py \
    --dataset data/training_data.jsonl \
    --output-dir model \
    --categories simple coding reasoning security summarization
```

This validates the dataset (empty/duplicate/missing/unknown labels, class
imbalance, train/val/test leakage), fits a `LogisticRegression` classifier
on `sentence-transformers` embeddings, evaluates it, and writes:

- `model/classifier.joblib` -- the artifact `TaskClassifier.from_artifact()` loads.
- `model/metadata.json` -- classifier type, embedding model, label set, dataset version, seed, dependency versions, evaluation metrics (Hugging Face-portable, requirement 16).
- `model/label_mapping.json`, `model/training_config.json`.

The script exits non-zero (and refuses to save, unless `--force` is
passed) if test-set accuracy/macro-F1 fall below `--min-accuracy` /
`--min-macro-f1` (defaults: 0.75 / 0.70) -- these are CLI flags, not
hardcoded constants, per the "don't embed arbitrary release thresholds in
code" requirement.

To grow the dataset, add more `{"text": ..., "label": ...}` lines to
`data/training_data.jsonl` (or a copy) -- see
`training/build_dataset.py` for how the seed set was generated, and
`training/dataset_validation.py` for the validation rules a new dataset
must pass.

## Hugging Face distribution

`model/classifier.joblib` + `model/metadata.json` + `model/label_mapping.json`
+ `model/training_config.json` are designed to be uploaded as-is to a
Hugging Face model repository alongside a model card. `metadata.json`
records everything needed to reproduce or audit the artifact: classifier
type, embedding model, label set, dataset version (content hash), training
timestamp, Python version, dependency versions, random seed, and
evaluation metrics.

## Testing

```bash
python -m pytest
```

- `tests/unit/` -- config parsing, schema validation, embedding wrapper,
  classifier thresholding, registry filtering, scoring, selection policies,
  dataset validation, evaluation metrics.
- `tests/integration/test_pipeline.py` -- full prompt -> embedding ->
  classifier -> selection -> model flow using a deterministic fake embedder
  (no network required).
- `tests/integration/test_real_artifact.py` -- regression check against the
  real trained artifact and `data/regression_dataset.jsonl` (requirement
  15). Skipped automatically if `model/classifier.joblib` doesn't exist yet.

## Project structure

```text
model_router/
├── pyproject.toml
├── config.example.yml
├── src/llm_router/       # the library
├── training/             # offline training pipeline (train.py, evaluate.py, dataset_validation.py)
├── data/                 # training_data.jsonl, regression_dataset.jsonl
├── model/                # classifier.joblib + metadata (trained artifact)
├── tests/{unit,integration}/
└── docs/                 # architecture.md, configuration.md
```

## Explicitly out of scope

Model availability/uptime monitoring, active health checks, circuit
breakers, retry/failover, an API serving layer (FastAPI etc.), direct LLM
invocation, agent orchestration, and secrets/API-key management all belong
to the host application or a separate health-monitor package -- see
`health_monitor_requirements.MD` at the repository root.
