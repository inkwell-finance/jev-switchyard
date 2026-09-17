# SPDX-FileCopyrightText: Copyright (c) 2026 Inkwell Finance, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import pytest

from benchmark.typesafe.adapter import (
    AdapterError,
    classifier_verdict,
    load_model_profile,
    openai_response,
    task_state,
    typesafe_payload,
)
from benchmark.typesafe.compare import (
    classifier_summary,
    estimated_cost,
    load,
    metric_means,
)
from benchmark.typesafe.model_profiles import (
    ProfileError,
    build_snapshot,
    parse_model_mapping,
    snapshot_matches,
)


def request() -> dict:
    return {
        "model": "jev-latest",
        "messages": [
            {"role": "system", "content": "Switchyard classifier prompt"},
            {"role": "user", "content": "Fix the failing parser tests."},
        ],
    }


def typesafe_response(rule: str = "SUP-2", p_solve: float = 0.81) -> dict:
    return {
        "model": "jev-latest",
        "answers": {
            "primary_rule": {
                "type": "choice",
                "choice": rule,
                "probabilities": {rule: 0.9},
                "confidence": 0.8,
            },
            "p_solve": {"type": "noul", "noul": p_solve},
        },
        "usage": {"input_tokens": 123, "output_tokens": 7},
    }


def test_payload_uses_task_messages_and_two_atomic_questions() -> None:
    profile = {"model_id": "moonshotai/kimi-k2.7-code", "coding_index": 60.8}
    payload = typesafe_payload(request(), "jev-latest", profile)

    assert payload["state"]["task"] == [
        {"role": "user", "content": "Fix the failing parser tests."}
    ]
    assert set(payload["questions"]) == {"primary_rule", "p_solve"}
    assert payload["questions"]["primary_rule"]["type"] == "choice"
    assert payload["questions"]["p_solve"]["type"] == "noul"
    assert payload["state"]["efficient_model_profile"] == profile


def test_verdict_derives_a_consistent_boundary() -> None:
    verdict, evidence = classifier_verdict(typesafe_response())

    assert verdict == {
        "crux": (
            "All required inputs are available, the target environment can be inspected, and "
            "correctness can be verified end-to-end without inaccessible external state."
        ),
        "primary_rule": "SUP-2",
        "capability_boundary": "supported",
        "p_solve": 0.81,
    }
    assert evidence["rule_confidence"] == 0.8


def test_openai_response_preserves_typesafe_usage() -> None:
    response, _ = openai_response(request(), typesafe_response())

    assert response["usage"] == {
        "prompt_tokens": 123,
        "completion_tokens": 7,
        "total_tokens": 130,
    }
    assert '"p_solve":0.81' in response["choices"][0]["message"]["content"]


@pytest.mark.parametrize(
    ("response", "message"),
    [
        (typesafe_response(rule="unknown"), "unknown rule"),
        (typesafe_response(p_solve=1.1), "out-of-range"),
        ({"answers": {}}, "invalid TypeSafe response"),
    ],
)
def test_invalid_typesafe_answers_fail_closed(response: dict, message: str) -> None:
    with pytest.raises(AdapterError, match=message):
        classifier_verdict(response)


def test_request_requires_a_task_message() -> None:
    with pytest.raises(AdapterError, match="no non-system task messages"):
        task_state({"messages": [{"role": "system", "content": "prompt"}]})


def test_comparison_reads_harbor_and_switchyard_artifacts() -> None:
    fixture = Path(__file__).parent / "fixtures" / "typesafe_compare" / "jev"

    row = load(fixture)

    assert row["score"] == "tb.reward=0.7500"
    assert row["mix"] == "strong=1, weak=3"
    assert "jev-latest: calls=4" in row["classifier"]
    assert row["overhead"] == "110.00 / 220.00"


def test_comparison_helpers_handle_missing_optional_data() -> None:
    assert metric_means({}) == "unavailable"
    assert classifier_summary({}) == "—"


def test_cost_uses_cached_and_uncached_rates() -> None:
    stats = {
        "models": {
            "model": {
                "prompt_tokens": 1_000_000,
                "cached_tokens": 250_000,
                "completion_tokens": 100_000,
            }
        },
        "classifier": {
            "models": {
                "jev-latest": {
                    "prompt_tokens": 10_000,
                    "completion_tokens": 1_000,
                }
            }
        },
    }
    prices = {
        "model": {
            "input_per_million": 2,
            "cached_input_per_million": 0.2,
            "output_per_million": 10,
        },
        "jev-latest": {"input_per_million": 0.042, "output_per_million": 0},
    }

    assert estimated_cost(stats, prices) == "$2.5504"


def test_model_profile_snapshot_selects_exact_benchmark_id() -> None:
    response = {
        "meta": {
            "source": "artificial-analysis",
            "source_url": "https://artificialanalysis.ai/",
            "as_of": "2026-09-16",
            "version": "4.3",
            "citation": "Artificial Analysis via OpenRouter",
        },
        "data": [
            {
                "source": "artificial-analysis",
                "model_permaslug": "moonshotai/kimi-k2.7-code-20260612",
                "display_name": "Kimi K2.7 Code",
                "intelligence_index": 26.3,
                "coding_index": 60.8,
                "agentic_index": 22.5,
                "pricing": {"prompt": "0.00000095", "completion": "0.000004"},
            }
        ],
    }
    snapshot = build_snapshot(
        response,
        {"moonshotai/kimi-k2.7-code": "moonshotai/kimi-k2.7-code-20260612"},
    )
    fixture = Path(__file__).parent / "fixtures" / "typesafe_model_profiles.json"

    profile = load_model_profile(fixture, "moonshotai/kimi-k2.7-code")

    assert snapshot["models"]["moonshotai/kimi-k2.7-code"]["coding_index"] == 60.8
    assert snapshot_matches(
        snapshot,
        {"moonshotai/kimi-k2.7-code": "moonshotai/kimi-k2.7-code-20260612"},
    )
    assert not snapshot_matches(snapshot, {"other/model": "other/model-20260101"})
    assert profile["coding_index"] == 60.8
    assert profile["benchmark_source"]["citation"] == "Artificial Analysis via OpenRouter"


def test_model_profile_helpers_reject_ambiguous_or_missing_models() -> None:
    assert parse_model_mapping("route/model=benchmark/model-20260101") == (
        "route/model",
        "benchmark/model-20260101",
    )
    with pytest.raises(ProfileError, match="not found"):
        build_snapshot(
            {"meta": {}, "data": []},
            {"route/model": "missing/model"},
        )
