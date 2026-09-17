#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 Inkwell Finance, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cache OpenRouter benchmark evidence for the TypeSafe routing experiment."""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

BENCHMARKS_URL = "https://openrouter.ai/api/v1/benchmarks"
DEFAULT_MODELS = {
    "anthropic/claude-opus-4.7": "anthropic/claude-4.7-opus-20260416",
    "moonshotai/kimi-k2.7-code": "moonshotai/kimi-k2.7-code-20260612",
}


class ProfileError(Exception):
    """Benchmark evidence could not be fetched or normalized."""


def parse_model_mapping(value: str) -> tuple[str, str]:
    """Parse ROUTABLE_ID=BENCHMARK_PERMASLUG from the command line."""
    routable_id, separator, permaslug = value.partition("=")
    if not separator or not routable_id.strip() or not permaslug.strip():
        raise argparse.ArgumentTypeError("model must be ROUTABLE_ID=BENCHMARK_PERMASLUG")
    return routable_id.strip(), permaslug.strip()


def fetch_benchmarks(api_key: str, timeout: float = 30.0) -> dict[str, Any]:
    """Fetch Artificial Analysis indices through OpenRouter's documented API."""
    query = urllib.parse.urlencode(
        {"source": "artificial-analysis", "max_results": 500}
    )
    request = urllib.request.Request(
        f"{BENCHMARKS_URL}?{query}",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json",
            "User-Agent": "jev-switchyard-benchmark/0.1",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        raise ProfileError(f"OpenRouter returned HTTP {error.code}: {body[:500]}") from error
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
        raise ProfileError(f"OpenRouter benchmark request failed: {error}") from error


def build_snapshot(
    response: dict[str, Any], model_mapping: dict[str, str]
) -> dict[str, Any]:
    """Select exact benchmark rows and preserve their attribution metadata."""
    rows = response.get("data")
    meta = response.get("meta")
    if not isinstance(rows, list) or not isinstance(meta, dict):
        raise ProfileError("OpenRouter response must contain data and meta")
    by_permaslug = {
        row.get("model_permaslug"): row for row in rows if isinstance(row, dict)
    }
    models = {}
    for routable_id, permaslug in model_mapping.items():
        row = by_permaslug.get(permaslug)
        if row is None:
            raise ProfileError(f"OpenRouter benchmark row not found for {permaslug}")
        models[routable_id] = {
            "display_name": row.get("display_name"),
            "benchmark_model_permaslug": permaslug,
            "intelligence_index": row.get("intelligence_index"),
            "coding_index": row.get("coding_index"),
            "agentic_index": row.get("agentic_index"),
            "pricing_per_token_usd": row.get("pricing"),
        }
    return {
        "schema_version": 1,
        "fetched_at": int(time.time()),
        "source": {
            key: meta.get(key)
            for key in ("source", "source_url", "as_of", "version", "citation")
        },
        "models": models,
    }


def read_fresh_snapshot(path: Path, max_age_hours: float) -> dict[str, Any] | None:
    """Return a recent valid snapshot, if one exists."""
    if not path.is_file() or max_age_hours <= 0:
        return None
    try:
        snapshot = json.loads(path.read_text(encoding="utf-8"))
        fetched_at = float(snapshot["fetched_at"])
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None
    if time.time() - fetched_at > max_age_hours * 3600:
        return None
    return snapshot


def snapshot_matches(
    snapshot: dict[str, Any], model_mapping: dict[str, str]
) -> bool:
    """Check that a cache contains the exact requested benchmark identities."""
    models = snapshot.get("models")
    if not isinstance(models, dict) or set(models) != set(model_mapping):
        return False
    return all(
        isinstance(models.get(model_id), dict)
        and models[model_id].get("benchmark_model_permaslug") == permaslug
        for model_id, permaslug in model_mapping.items()
    )


def write_snapshot(path: Path, snapshot: dict[str, Any]) -> None:
    """Atomically replace the cached snapshot."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(snapshot, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--model",
        action="append",
        type=parse_model_mapping,
        metavar="ROUTABLE_ID=BENCHMARK_PERMASLUG",
        help="exact OpenRouter benchmark identity; may be repeated",
    )
    parser.add_argument("--max-age-hours", type=float, default=24.0)
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args()
    if args.max_age_hours < 0:
        parser.error("--max-age-hours must be nonnegative")

    mapping = dict(args.model) if args.model else DEFAULT_MODELS
    cached = read_fresh_snapshot(args.output, args.max_age_hours)
    if cached is not None and snapshot_matches(cached, mapping):
        print(f"Using cached model profiles from {args.output}")
        return 0

    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        parser.error("OPENROUTER_API_KEY is required")
    try:
        snapshot = build_snapshot(fetch_benchmarks(api_key, args.timeout), mapping)
        write_snapshot(args.output, snapshot)
    except (OSError, ProfileError) as error:
        parser.error(str(error))
    print(f"Wrote {len(snapshot['models'])} model profiles to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
