# Configuration Reference

`llm_router` is configured entirely through one YAML file (see
`config.example.yml`). Secrets (API keys, credentials) must never be placed
in this file -- it contains no fields for them by design.

## `embedding`

```yaml
embedding:
  model: sentence-transformers/all-MiniLM-L6-v2
  cache_folder: null   # optional, passed to SentenceTransformer(cache_folder=...)
  device: null          # optional, e.g. "cpu" or "cuda"
```

`model` is any `sentence-transformers`-compatible model name or local path.
It must match the `embedding_model` recorded in the classifier artifact's
`metadata.json` -- the router does not currently re-validate this at load
time, so a mismatch will silently produce garbage classifications rather
than an error. Keep the two in sync.

## `classifier`

```yaml
classifier:
  artifact: model/classifier.joblib   # resolved relative to the config file's directory if relative
  confidence_threshold: 0.35
  max_prompt_length: 4000
```

- `artifact`: path to a joblib artifact produced by `training/train.py`
  (a dict with `classifier`, `classes`, and `embedding_model` keys).
- `confidence_threshold`: the minimum top-1 softmax probability required to
  return a confident category. Below this, `Router.route()` returns
  `category="unknown"`, `model_id=None`,
  `reason="classification_below_threshold"`.
- `max_prompt_length`: prompts longer than this (after trimming whitespace)
  raise `InvalidPromptError`.

### Calibrating `confidence_threshold`

There is no universally correct value -- it depends on the number of
categories, dataset size, and classifier. With 5 balanced categories and a
few dozen examples per category (the bundled baseline dataset), correct
top-1 predictions commonly land in the 0.4-0.65 range, not the 0.9+ range
you might expect from a binary classifier. Setting the threshold too high
means every request gets classified `unknown` regardless of accuracy.

To recalibrate: run `training/train.py`, inspect
`model/metadata.json -> evaluation.test.per_class`, and pick a threshold
using a validation script that sweeps thresholds against held-out data and
picks the point where precision on "confident" predictions meets your
tolerance for misrouting. As the training dataset grows, softmax
probabilities typically sharpen and the threshold can be raised.

## `categories`

```yaml
categories:
  - simple
  - coding
  - reasoning
  - security
  - summarization
```

The full set of task categories the classifier can output. Every model's
`categories` list (see below) must be a subset of this list --
`ModelValidationError` is raised otherwise. Add new categories here and
retrain the classifier with matching labels in the training dataset; no
core package code needs to change (requirement 4.5).

## `selection`

```yaml
selection:
  policy: weighted   # or "priority"
  weights:
    priority: 0.40
    cost: 0.30
    latency: 0.20
    capability: 0.10
  required_capabilities: {}
```

- `policy: priority` -- deterministic: candidates are ordered purely by
  ascending `priority` (a model with `priority: 1` always outranks
  `priority: 2`, regardless of cost/latency/capability). Ties broken by
  model id.
- `policy: weighted` -- each of priority/cost/latency/capability is
  min-max normalized to `[0, 1]` across the current candidate set (higher
  is always better), then combined via the configured weights and
  renormalized by their sum. See `src/llm_router/scoring.py`.
- `weights`: need not sum to 1 -- they're normalized internally -- but
  must not all be zero.
- `required_capabilities`: an optional static gate applied *before*
  ranking. Supported keys:
  - `tool_calling: true` -- excludes models without tool-calling support.
  - `structured_output: true` -- excludes models without structured-output support.
  - `min_context_window: <int>` -- excludes models with a smaller context window.
  Unknown keys are ignored (logged at `WARNING`), not treated as errors,
  so this section can be extended without breaking older configs.

## `logging`

```yaml
logging:
  level: INFO
  log_prompts: false
```

When `log_prompts` is `false` (the default and recommended production
setting), the router logs prompt *length* and classification outcome, but
never the prompt text itself. Set `log_prompts: true` only in trusted
development environments.

## `models`

Each entry is validated into a `ModelDefinition`:

```yaml
models:
  - id: fast-general            # required, unique across the registry
    provider: provider_a        # required, free-form string (informational)
    display_name: Fast General Model
    categories: [simple, summarization]   # must be a subset of top-level `categories`
    pricing:
      input_per_1k_tokens: 0.01
      output_per_1k_tokens: 0.02
    performance:
      expected_latency_ms: 400  # must be > 0
    capabilities:
      context_window: 32000     # must be > 0
      tool_calling: true
      structured_output: true
    priority: 1                 # lower is better/higher-ranked; must be >= 0
    enabled: true                # disabled models are never selected
```

This registry describes **capability, not availability**. There is no
health/uptime field here by design -- pair this package with a separate
health-monitor component if you need to exclude models that are currently
down (see `health_monitor_requirements.MD`).
