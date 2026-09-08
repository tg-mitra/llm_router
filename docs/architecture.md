# Architecture

## Pipeline

```text
                    User Prompt
                         |
                         v
              +---------------------+
              | Sentence Transformer|   src/llm_router/embeddings.py
              +----------+----------+
                         |
                         v
                    Embedding (np.ndarray)
                         |
                         v
              +---------------------+
              | Task Classifier      |  src/llm_router/classifier.py
              | sklearn model        |
              +----------+----------+
                         |
                         v
              Task + Confidence Score       (ClassificationResult)
                         |
                         v
              +---------------------+
              | Model Registry       |  src/llm_router/registry.py
              +----------+----------+
                         |
                         v
              Category/Capability Filter    src/llm_router/selector.py
                         |
                         v
              Candidate Model Ranking       src/llm_router/scoring.py
                         |
                         v
                  Selected Model             (RoutingResult)
```

`Router.route(prompt)` (`src/llm_router/router.py`) wires the above stages
together behind a single call.

## Module boundaries

| Module | Responsibility | Depends on |
|---|---|---|
| `embeddings.py` | Wraps `sentence-transformers`. Lazy-loaded, provider of `Embedder` protocol. | sentence-transformers |
| `classifier.py` | Prompt validation, embedding -> sklearn `predict_proba`, confidence thresholding. | `embeddings`, `schemas`, `exceptions` |
| `config.py` | YAML parsing and validation into typed dataclasses. | PyYAML, `schemas` |
| `schemas.py` | Frozen dataclasses shared across the package (`ModelDefinition`, `ClassificationResult`, `RoutingResult`, ...). | none (leaf module) |
| `registry.py` | Read-only collection of `ModelDefinition`, static metadata only -- **no health/availability state**. | `schemas` |
| `scoring.py` | Pure functions: normalize + combine priority/cost/latency/capability into scores. | `schemas`, `config` |
| `selector.py` | Filters candidates (enabled, category, capability gates) then ranks via `scoring`. | `scoring`, `schemas`, `config`, `exceptions` |
| `router.py` | Public facade: `Router.from_config(...)`, `Router.route(prompt)`. | all of the above |

Data flows one direction: `router -> classifier -> embeddings`, and
`router -> selector -> scoring`. Nothing in `classifier.py`, `embeddings.py`,
`registry.py`, `scoring.py`, or `selector.py` imports from `router.py` --
each layer is independently testable and importable.

## Why the classifier never sees model names

The classifier is trained to predict a **task category** (`simple`,
`coding`, `reasoning`, `security`, `summarization`, ...), never a specific
model id. This is what lets the model registry (`config.yml`) change --
models added, removed, repriced -- without ever retraining or redeploying
the classifier artifact. The two lifecycles are fully decoupled:

- Classifier artifact version: `training_dataset_version` + `random_seed` in `model/metadata.json`.
- Model registry version: whatever the host application's `config.yml` deployment process tracks.

## Why capability and availability are separate concerns

`ModelRegistry` and `Selector` only ever reason about **static** metadata:
declared categories, declared capabilities (context window, tool calling,
structured output), configured price, and configured expected latency. None
of this package makes network calls, pings endpoints, or tracks uptime.
Whether a model is *currently* reachable is the responsibility of a
separate health-monitor component (see `health_monitor_requirements.MD` in
the repository root) that the host application can consult before or after
calling `Router.route()`. This package answers "which model is the best
static fit," not "which model is currently alive."

## Determinism

Given the same prompt, the same classifier artifact, the same
`config.yml`, and the same selection policy, `Router.route()` always
returns the same `RoutingResult`. Nothing in the routing path uses
wall-clock time, randomness, or external I/O beyond the one-time embedding
model load. This is what makes the weighted scoring formula in
`scoring.py` explainable: every sub-score (`priority_score`, `cost_score`,
`latency_score`, `capability_score`) is returned on `CandidateScore` for
every ranked candidate, not just the winner.

## Error handling philosophy

Two different kinds of "no answer" are handled differently:

- **Routine runtime outcomes** -- low classifier confidence, or a
  confidently-classified category with no eligible model -- are reported
  through `RoutingResult.reason` with `model_id=None`. `Router.route()`
  never raises for these; a host application calling it in a hot path
  should not need a try/except for "the classifier wasn't sure."
- **Actionable bugs** -- invalid prompt input, malformed configuration, a
  missing classifier artifact, invalid model metadata, or a category the
  classifier predicts that isn't in the configured category set -- raise
  typed exceptions from `exceptions.py`. These indicate something a
  developer or operator needs to fix, not a normal request outcome.
