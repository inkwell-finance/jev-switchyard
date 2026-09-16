# TypeSafe Jev classifier experiment

This prototype replaces Switchyard's routing-time LLM with TypeSafe's Jev model. Completion calls
still go to the same strong and weak models. The adapter presents an OpenAI Chat Completions surface
because that is already a supported Switchyard classifier target.

## How it works

Switchyard's capability classifier normally asks a generative model for four JSON fields. The
adapter maps that judgment to two TypeSafe questions evaluated together:

1. A `Choice` selects the capability rule that best describes the task's hardest requirement.
2. A `Noul` returns the probability that the efficient agent completes the whole task.

The adapter derives `capability_boundary` from the selected rule, uses that rule's description as
the `crux`, and returns the Noul value as `p_solve`. It emits the JSON contract Switchyard already
validates. Current Switchyard routing uses the rule, boundary, and solve probability.

This is a good fit for Jev's interface. It asks for two narrow decisions and composes them in code.
It also avoids asking Jev to generate text, which the model does not support.

Before an experiment, `model_profiles.py` caches the strong and efficient models' Artificial
Analysis indices and pricing from OpenRouter's documented Benchmarks API. The adapter gives Jev the
efficient model's exact profile as prior evidence. The benchmark values do not replace the
task-specific judgment or the end-to-end experiment results.

This path uses the existing `OPENROUTER_API_KEY`; it does not require an Artificial Analysis key.
Artificial Analysis prohibits automated scraping in its website terms, so the experiment does not
scrape its pages. The cached response preserves OpenRouter's source, timestamp, version, citation,
and source URL. Generated snapshots live below the ignored results directory.

## Run a smoke test

The adapter has no third-party runtime dependencies:

```bash
export TYPESAFE_API_KEY="..."
python benchmark/typesafe/adapter.py --host 127.0.0.1 --port 8090
```

Run one live adapter request without starting Switchyard:

```bash
python -m benchmark.typesafe.model_profiles \
  --output benchmark/typesafe/results/model-profiles.json
python -m benchmark.typesafe.smoke \
  --model-profiles benchmark/typesafe/results/model-profiles.json
```

Force a refresh of only the model evidence cache with:

```bash
python -m benchmark.typesafe.model_profiles \
  --output benchmark/typesafe/results/model-profiles.json \
  --max-age-hours 0
```

The TypeSafe organization must have available API credits. An exhausted organization returns HTTP
402, which the adapter surfaces as an upstream failure.

The checked-in server config expects the adapter at `http://typesafe-adapter:8090`, its Docker
network alias in the experiment runner. For a local Switchyard server, copy the config and change
that base URL to `http://127.0.0.1:8090/v1`, or use the checked-in `-local.toml` variant.

## Run the controlled comparison

Prepare the Harbor dataset and patch described in `benchmark/README.md`, then set both keys:

```bash
export TYPESAFE_API_KEY="..."
export OPENROUTER_API_KEY="..."
bash benchmark/typesafe/run_experiment.sh
```

The default 20-task subset runs these conditions in order:

- `strong`: Claude Opus 4.7 for every call, establishing the quality ceiling.
- `weak`: Kimi K2.7 Code for every call, establishing the efficient-model floor.
- `gemini`: existing Switchyard capability routing with Gemini 3.5 Flash.
- `jev`: the identical Switchyard route, thresholds, strong model, and weak model with Jev as the
  classifier.

Set `TASK_LIST_FILE`, `HARBOR_PATH`, `RESULTS_DIR`, `AGENT`, `N_CONCURRENT`, or `MAX_RETRIES` to
override defaults. Start with one or two tasks to validate credentials and connectivity. Use the
full Terminal-Bench 2.1 task list and at least three repeats per task for a decision-quality result.
The current runner performs one pass, so repeats should use separate result roots and identical task
lists.

## What to compare

The primary end-to-end metric is task solve rate. Report it beside:

- total provider cost and cost per solved task;
- total wall time and task latency;
- classifier p50 and p99 latency;
- classifier input tokens, failures, and fail-open count;
- strong/weak completion-call share;
- agreement between Jev and the existing classifier;
- paired task outcomes and a bootstrap confidence interval across tasks.

Do not compare only classifier latency. A faster classifier that sends more solvable work to the
strong model can cost more overall, while an aggressive weak-model route can reduce cost by lowering
task success. Keep the completion models, task order, agent version, reasoning effort, retry policy,
book mode, and Switchyard thresholds fixed.

For dollar cost, use provider invoice data when possible. Otherwise snapshot each model's input,
cached-input, and output prices at experiment time and apply them to the saved token counts. Jev is
currently advertised at $0.042 per million input tokens with free output tokens, but pricing is
time-sensitive and should be recorded with the run.

The comparison helper accepts a price snapshot:

```json
{
  "jev-latest": {
    "input_per_million": 0.042,
    "output_per_million": 0.0
  },
  "provider/model": {
    "input_per_million": 1.0,
    "cached_input_per_million": 0.1,
    "output_per_million": 5.0
  }
}
```

Include every completion and classifier model shown in `routing_stats_final.json`, then run:

```bash
python benchmark/typesafe/compare.py RESULTS_DIR --prices prices-at-run-time.json
```

## Validation layers

Use two complementary evaluations:

1. Router-only replay. Feed the same task openings to Gemini and Jev. Measure decision agreement,
   latency, failures, and calibration against empirical weak-model success from repeated runs.
2. End-to-end agent runs. Compare fixed strong, fixed weak, Gemini-routed, and Jev-routed outcomes.
   The fixed weak arm estimates whether routing decisions actually separate tasks by
   difficulty.

The 20-task subset is a smoke test. It is too small for a reliable quality claim.

## Current limits

- This adapter implements Switchyard's built-in capability-classifier contract only.
- TypeSafe questions in one request are independent. `p_solve` does not condition on the selected
  rule; both questions see the same task and capability card.
- The adapter does not support streaming because Switchyard's classifier call is buffered.
- The adapter fails with HTTP 502 on malformed TypeSafe output. Switchyard then applies its existing
  classifier fail-open behavior.

Primary references:

- https://docs.typesafe.ai/introduction
- https://docs.typesafe.ai/primitives
- https://docs.typesafe.ai/primitives/noul
- https://docs.typesafe.ai/confidence
- https://openrouter.ai/docs/api/api-reference/benchmarks/list-benchmarks
- https://artificialanalysis.ai/terms-of-use
- https://github.com/NVIDIA-NeMo/Switchyard
- https://github.com/NVIDIA-NeMo/Switchyard/blob/main/benchmark/README.md
