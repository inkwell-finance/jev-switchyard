#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Expose TypeSafe Jev as an OpenAI-compatible Switchyard classifier target."""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.request
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

TYPESAFE_URL = "https://api.typesafe.ai/v1/systemone"

RULES = {
    "SUP-1": "The task provides a complete output contract and a deterministic local validator that covers the material requirements.",
    "SUP-2": "All required inputs are available, the target environment can be inspected, and correctness can be verified end-to-end without inaccessible external state.",
    "SUP-3": "Mathematical behavior, interfaces, shapes, data types, tolerances, and performance requirements are explicit and exercised by a representative harness.",
    "SUP-4": "The required mechanism is identified, the relevant search space is bounded, and the success condition is executable.",
    "SUP-5": "Reconstruction or behavioral reproduction is constrained by an executable reference, parser, format specification, or checker strong enough to distinguish correct from plausible output.",
    "UNC-1": "Multiple reasonable interpretations of preprocessing, representation, indexing, naming, or output placement produce different results, and neither instructions nor a validator resolve the choice.",
    "UNC-2": "Success requires finding every relevant item across heterogeneous inputs or environment state, but the task does not define the search boundary or provide a completeness check.",
    "LIM-1": "Correctness depends primarily on extracting precise information from noisy visual, temporal, or rendered media without a machine-checkable extraction or replay mechanism.",
    "LIM-2": "Success depends on reproducing undocumented reference behavior, hidden intermediate state, or an unknown configuration, and small deviations fail despite satisfying the visible specification.",
    "none": "No listed rule materially describes the hardest requirement.",
}

BOUNDARY = {
    "SUP-1": "supported",
    "SUP-2": "supported",
    "SUP-3": "supported",
    "SUP-4": "supported",
    "SUP-5": "supported",
    "UNC-1": "uncertain",
    "UNC-2": "uncertain",
    "LIM-1": "unsupported",
    "LIM-2": "unsupported",
    "none": "unmatched",
}

SUCCESS_QUESTION = (
    "On one fresh run under the actual harness, tools, and budget, will the efficient "
    "coding agent complete the whole task correctly as judged by the final verifier? "
    "Estimate the probability of whole-task success. Use only `task`, "
    "`efficient_agent_capability_card`, and `efficient_model_profile` when that profile is "
    "present. Benchmark scores are broad prior evidence, not a task-specific success rate. "
    "Do not assume hidden state, tools, validators, documentation, access, or future work "
    "habits. Missing information should limit extreme estimates but is not evidence for "
    "exactly 0.5."
)


class AdapterError(Exception):
    """A request cannot be converted into a valid classifier response."""


def load_model_profile(path: Path, model_id: str) -> dict[str, Any]:
    """Load one exact model profile and its benchmark provenance."""
    try:
        snapshot = json.loads(path.read_text(encoding="utf-8"))
        profile = snapshot["models"][model_id]
        source = snapshot["source"]
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as error:
        raise AdapterError(f"invalid model profile snapshot: {error}") from error
    if not isinstance(profile, dict) or not isinstance(source, dict):
        raise AdapterError("invalid model profile snapshot: profile and source must be objects")
    return {"model_id": model_id, **profile, "benchmark_source": source}


def _text_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return json.dumps(content, ensure_ascii=False)
    parts = []
    for item in content:
        if isinstance(item, dict) and item.get("type") == "text":
            parts.append(str(item.get("text", "")))
        else:
            parts.append(json.dumps(item, ensure_ascii=False, sort_keys=True))
    return "\n".join(part for part in parts if part)


def task_state(
    openai_request: dict[str, Any], efficient_model_profile: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Keep the classifier-visible conversation and the qualitative capability card."""
    messages = openai_request.get("messages")
    if not isinstance(messages, list):
        raise AdapterError("messages must be a list")
    task = []
    for message in messages:
        if not isinstance(message, dict):
            raise AdapterError("each message must be an object")
        role = message.get("role")
        if role == "system":
            continue
        task.append({"role": role, "content": _text_content(message.get("content", ""))})
    if not task:
        raise AdapterError("classifier request has no non-system task messages")
    state = {
        "task": task,
        "efficient_agent_capability_card": RULES,
    }
    if efficient_model_profile is not None:
        state["efficient_model_profile"] = efficient_model_profile
    return state


def typesafe_payload(
    openai_request: dict[str, Any],
    model: str,
    efficient_model_profile: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Translate a Switchyard capability-classifier request to two atomic Jev questions."""
    return {
        "model": model,
        "state": task_state(openai_request, efficient_model_profile),
        "questions": {
            "primary_rule": {
                "type": "choice",
                "instructions": (
                    "Which single rule in `efficient_agent_capability_card` best describes "
                    "the hardest material requirement (the crux) for completing `task`?"
                ),
                "criteria": RULES,
            },
            "p_solve": {
                "type": "noul",
                "instructions": SUCCESS_QUESTION,
                "criteria": {
                    "true": "The efficient agent completes the whole task correctly.",
                    "false": "Any other outcome, including partial or unverifiable completion.",
                },
            },
        },
    }


def classifier_verdict(typesafe_response: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Turn typed Jev answers into Switchyard's validated capability verdict."""
    try:
        answers = typesafe_response["answers"]
        rule_answer = answers["primary_rule"]
        primary_rule = rule_answer["choice"]
        p_solve = float(answers["p_solve"]["noul"])
    except (KeyError, TypeError, ValueError) as error:
        raise AdapterError(f"invalid TypeSafe response: {error}") from error
    if primary_rule not in BOUNDARY:
        raise AdapterError(f"TypeSafe returned unknown rule {primary_rule!r}")
    if not 0.0 <= p_solve <= 1.0:
        raise AdapterError(f"TypeSafe returned out-of-range p_solve {p_solve}")
    evidence = {
        "rule_probabilities": rule_answer.get("probabilities", {}),
        "rule_confidence": rule_answer.get("confidence"),
    }
    return {
        "crux": RULES[primary_rule],
        "primary_rule": primary_rule,
        "capability_boundary": BOUNDARY[primary_rule],
        "p_solve": p_solve,
    }, evidence


def openai_response(
    request: dict[str, Any], typesafe_response: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build the buffered Chat Completions shape consumed by Switchyard."""
    verdict, evidence = classifier_verdict(typesafe_response)
    usage = typesafe_response.get("usage", {})
    prompt_tokens = int(usage.get("input_tokens", 0))
    completion_tokens = int(usage.get("output_tokens", 0))
    model = str(typesafe_response.get("model") or request.get("model") or "jev-latest")
    response = {
        "id": f"chatcmpl-typesafe-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": json.dumps(verdict, separators=(",", ":")),
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }
    return response, {"verdict": verdict, **evidence, "usage": usage}


class TypeSafeUpstream:
    def __init__(
        self,
        api_key: str,
        url: str,
        model: str,
        timeout: float,
        max_retries: int,
        efficient_model_profile: dict[str, Any] | None = None,
    ) -> None:
        self.api_key = api_key
        self.url = url
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self.efficient_model_profile = efficient_model_profile

    def classify(self, request: dict[str, Any]) -> dict[str, Any]:
        payload = json.dumps(
            typesafe_payload(request, self.model, self.efficient_model_profile)
        ).encode("utf-8")
        upstream = urllib.request.Request(
            self.url,
            data=payload,
            method="POST",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )
        for attempt in range(self.max_retries + 1):
            try:
                with urllib.request.urlopen(upstream, timeout=self.timeout) as response:
                    return json.loads(response.read())
            except urllib.error.HTTPError as error:
                body = error.read().decode("utf-8", errors="replace")
                if error.code not in (429, 529) or attempt == self.max_retries:
                    raise AdapterError(
                        f"TypeSafe returned HTTP {error.code}: {body[:1000]}"
                    ) from error
                retry_after = error.headers.get("Retry-After")
                try:
                    delay = min(float(retry_after), 10.0) if retry_after else 0.25 * 2**attempt
                except ValueError:
                    delay = 0.25 * 2**attempt
                time.sleep(delay)
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
                raise AdapterError(f"TypeSafe request failed: {error}") from error
        raise AdapterError("TypeSafe retry loop exited without a response")


class AdapterServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(
        self,
        address: tuple[str, int],
        upstream: TypeSafeUpstream,
        log_path: Path | None,
    ) -> None:
        super().__init__(address, AdapterHandler)
        self.upstream = upstream
        self.log_path = log_path

    def log_event(self, event: dict[str, Any]) -> None:
        if self.log_path is None:
            return
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as output:
            output.write(json.dumps(event, sort_keys=True) + "\n")


class AdapterHandler(BaseHTTPRequestHandler):
    server: AdapterServer

    def log_message(self, format: str, *args: Any) -> None:
        print(f"{self.address_string()} - {format % args}")

    def _json(self, status: int, body: dict[str, Any]) -> None:
        encoded = json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def do_GET(self) -> None:
        if self.path == "/health":
            self._json(HTTPStatus.OK, {"status": "ok"})
        elif self.path == "/v1/models":
            model = self.server.upstream.model
            self._json(
                HTTPStatus.OK,
                {"object": "list", "data": [{"id": model, "object": "model"}]},
            )
        else:
            self._json(HTTPStatus.NOT_FOUND, {"error": {"message": "not found"}})

    def do_POST(self) -> None:
        if self.path != "/v1/chat/completions":
            self._json(HTTPStatus.NOT_FOUND, {"error": {"message": "not found"}})
            return
        started = time.perf_counter()
        try:
            length = int(self.headers.get("Content-Length", "0"))
            request = json.loads(self.rfile.read(length))
            if request.get("stream"):
                raise AdapterError("streaming classifier requests are not supported")
            upstream_response = self.server.upstream.classify(request)
            response, event = openai_response(request, upstream_response)
            event.update(
                {
                    "ok": True,
                    "latency_ms": round((time.perf_counter() - started) * 1000, 3),
                    "timestamp": time.time(),
                }
            )
            self.server.log_event(event)
            self._json(HTTPStatus.OK, response)
        except (AdapterError, json.JSONDecodeError, TypeError, ValueError) as error:
            self.server.log_event(
                {
                    "ok": False,
                    "error": str(error),
                    "latency_ms": round((time.perf_counter() - started) * 1000, 3),
                    "timestamp": time.time(),
                }
            )
            self._json(HTTPStatus.BAD_GATEWAY, {"error": {"message": str(error)}})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--model", default="jev-latest")
    parser.add_argument("--typesafe-url", default=TYPESAFE_URL)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--log", type=Path)
    parser.add_argument("--model-profiles", type=Path)
    parser.add_argument("--efficient-model", default="moonshotai/kimi-k2.7-code")
    args = parser.parse_args()
    if args.max_retries < 0:
        parser.error("--max-retries must be nonnegative")
    api_key = os.environ.get("TYPESAFE_API_KEY", "").strip()
    if not api_key:
        parser.error("TYPESAFE_API_KEY is required")
    try:
        efficient_model_profile = (
            load_model_profile(args.model_profiles, args.efficient_model)
            if args.model_profiles
            else None
        )
    except AdapterError as error:
        parser.error(str(error))
    server = AdapterServer(
        (args.host, args.port),
        TypeSafeUpstream(
            api_key,
            args.typesafe_url,
            args.model,
            args.timeout,
            args.max_retries,
            efficient_model_profile,
        ),
        args.log,
    )
    print(f"TypeSafe classifier adapter listening on http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
