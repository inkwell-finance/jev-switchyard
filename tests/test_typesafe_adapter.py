# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import pytest

from benchmark.typesafe.adapter import (
    AdapterError,
    classifier_verdict,
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
    payload = typesafe_payload(request(), "jev-latest")

    assert payload["state"]["task"] == [
        {"role": "user", "content": "Fix the failing parser tests."}
    ]
    assert set(payload["questions"]) == {"primary_rule", "p_solve"}
    assert payload["questions"]["primary_rule"]["type"] == "choice"
    assert payload["questions"]["p_solve"]["type"] == "noul"


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
