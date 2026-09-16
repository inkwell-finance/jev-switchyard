#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compare fixed, LLM-classifier, and Jev-classifier Harbor runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def one_run(condition: Path) -> Path:
    manifests = sorted(condition.glob("*/run_manifest.json"))
    if len(manifests) != 1:
        raise ValueError(f"expected one run under {condition}, found {len(manifests)}")
    return manifests[0].parent


def metric_means(result: dict[str, Any]) -> str:
    evals = result.get("stats", {}).get("evals", {})
    values = []
    for eval_name, evaluation in evals.items():
        for index, metric in enumerate(evaluation.get("metrics", [])):
            if "mean" not in metric:
                continue
            name = metric.get("name") or metric.get("metric") or f"metric_{index}"
            values.append(f"{eval_name}.{name}={metric['mean']:.4f}")
    return ", ".join(values) or "unavailable"


def classifier_summary(stats: dict[str, Any]) -> str:
    classifier = stats.get("classifier", {})
    models = classifier.get("models", {})
    if not models:
        return "—"
    parts = []
    for model, values in models.items():
        latency = values.get("model_call_latency", {})
        parts.append(
            f"{model}: calls={values.get('calls', 0)}, errors={values.get('errors', 0)}, "
            f"tokens={values.get('total_tokens', 0)}, p50={latency.get('p50_ms', 0):.2f}ms"
        )
    return "; ".join(parts)


def route_mix(stats: dict[str, Any]) -> str:
    models = stats.get("models", {})
    return ", ".join(
        f"{model}={values.get('calls', 0)}" for model, values in sorted(models.items())
    ) or "unavailable"


def model_cost(values: dict[str, Any], price: dict[str, Any]) -> float:
    prompt = int(values.get("prompt_tokens", 0))
    cached = int(values.get("cached_tokens", 0))
    completion = int(values.get("completion_tokens", 0))
    uncached = max(prompt - cached, 0)
    cached_rate = price.get("cached_input_per_million", price["input_per_million"])
    return (
        uncached * float(price["input_per_million"])
        + cached * float(cached_rate)
        + completion * float(price["output_per_million"])
    ) / 1_000_000


def estimated_cost(stats: dict[str, Any], prices: dict[str, Any] | None) -> str:
    if prices is None:
        return "—"
    total = 0.0
    seen = False
    for section in (stats.get("models", {}), stats.get("classifier", {}).get("models", {})):
        for model, values in section.items():
            if model not in prices:
                return f"missing price: {model}"
            total += model_cost(values, prices[model])
            seen = True
    return f"${total:.4f}" if seen else "unavailable"


def load(condition: Path, prices: dict[str, Any] | None = None) -> dict[str, str]:
    run = one_run(condition)
    result_path = run / "harbor_result.json"
    if not result_path.is_file():
        raise ValueError(f"missing {result_path}")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    stats_path = run / "routing_stats_final.json"
    stats = json.loads(stats_path.read_text(encoding="utf-8")) if stats_path.is_file() else {}
    overhead = stats.get("routing_overhead", {})
    return {
        "score": metric_means(result),
        "mix": route_mix(stats),
        "classifier": classifier_summary(stats),
        "overhead": f"{overhead.get('p50_ms', 0):.2f} / {overhead.get('p99_ms', 0):.2f}",
        "cost": estimated_cost(stats, prices),
        "run": str(run),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument(
        "--prices",
        type=Path,
        help="JSON model price map with input_per_million and output_per_million",
    )
    args = parser.parse_args()
    try:
        prices = json.loads(args.prices.read_text(encoding="utf-8")) if args.prices else None
        rows = {
            name: load(args.results / name, prices)
            for name in ("strong", "weak", "gemini", "jev")
        }
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        parser.error(str(error))

    print(f"{'condition':<10} {'benchmark metric':<38} {'route p50/p99 ms':>19} {'cost':>18}")
    for name, row in rows.items():
        print(f"{name:<10} {row['score']:<38} {row['overhead']:>19} {row['cost']:>18}")
    for name, row in rows.items():
        print(f"\n{name} serving calls: {row['mix']}")
        print(f"{name} classifier: {row['classifier']}")
        print(f"{name} artifacts: {row['run']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
