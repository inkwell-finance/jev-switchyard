#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run one live request through the local adapter and TypeSafe API."""

from __future__ import annotations

import argparse
import json
import os
import threading
import urllib.error
import urllib.request

from benchmark.typesafe.adapter import AdapterServer, TypeSafeUpstream


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task",
        default=(
            "Fix a parser bug in this repository. Run the existing tests and preserve the "
            "public API."
        ),
    )
    parser.add_argument("--model", default="jev-latest")
    args = parser.parse_args()
    api_key = os.environ.get("TYPESAFE_API_KEY", "").strip()
    if not api_key:
        parser.error("TYPESAFE_API_KEY is required")

    server = AdapterServer(
        ("127.0.0.1", 0),
        TypeSafeUpstream(
            api_key,
            "https://api.typesafe.ai/v1/systemone",
            args.model,
            30.0,
            2,
        ),
        None,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        request = {
            "model": args.model,
            "stream": False,
            "messages": [
                {"role": "system", "content": "Switchyard capability classifier"},
                {"role": "user", "content": args.task},
            ],
        }
        body = json.dumps(request).encode("utf-8")
        url = f"http://127.0.0.1:{server.server_port}/v1/chat/completions"
        try:
            with urllib.request.urlopen(
                urllib.request.Request(
                    url,
                    data=body,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                ),
                timeout=40,
            ) as response:
                result = json.loads(response.read())
        except urllib.error.HTTPError as error:
            body = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"adapter returned HTTP {error.code}: {body}") from error
        verdict = json.loads(result["choices"][0]["message"]["content"])
        print(
            json.dumps(
                {
                    "model": result["model"],
                    "primary_rule": verdict["primary_rule"],
                    "capability_boundary": verdict["capability_boundary"],
                    "p_solve": verdict["p_solve"],
                    "usage": result["usage"],
                },
                indent=2,
                sort_keys=True,
            )
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
