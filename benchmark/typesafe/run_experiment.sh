#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
HARBOR_PATH="${HARBOR_PATH:-${ROOT}/benchmark/datasets/openthoughts-tblite-closed-book}"
RESULTS_DIR="${RESULTS_DIR:-${ROOT}/benchmark/typesafe/results/$(date -u +%Y%m%dT%H%M%SZ)}"
TASK_LIST_FILE="${TASK_LIST_FILE:-${ROOT}/benchmark/tb_lite_subset_20.txt}"
AGENT="${AGENT:-codex}"
N_CONCURRENT="${N_CONCURRENT:-8}"
MAX_RETRIES="${MAX_RETRIES:-2}"
REASONING_EFFORT="${REASONING_EFFORT:-xhigh}"
ADAPTER_IMAGE="${ADAPTER_IMAGE:-python:3.12-slim}"
RUN_ID="typesafe-$RANDOM-$$"
NETWORK="switchyard-${RUN_ID}"
ADAPTER_CONTAINER="typesafe-adapter-${RUN_ID}"

die() {
    echo "ERROR: $*" >&2
    exit 1
}

[[ -n "${TYPESAFE_API_KEY:-}" ]] || die "TYPESAFE_API_KEY is required"
[[ -n "${OPENROUTER_API_KEY:-}" ]] || die "OPENROUTER_API_KEY is required"
[[ -d "${HARBOR_PATH}" ]] || die "Harbor dataset not found: ${HARBOR_PATH}"
[[ -f "${TASK_LIST_FILE}" ]] || die "Task list not found: ${TASK_LIST_FILE}"

mkdir -p "${RESULTS_DIR}"/{strong,weak,gemini,jev}
python "${ROOT}/benchmark/typesafe/model_profiles.py" \
    --output "${RESULTS_DIR}/model-profiles.json"
docker network create "${NETWORK}" >/dev/null

cleanup() {
    docker rm -f "${ADAPTER_CONTAINER}" >/dev/null 2>&1 || true
    docker network rm "${NETWORK}" >/dev/null 2>&1 || true
}
trap cleanup EXIT

docker run -d --rm \
    --name "${ADAPTER_CONTAINER}" \
    --network "${NETWORK}" \
    --network-alias typesafe-adapter \
    -e TYPESAFE_API_KEY \
    -v "${ROOT}:/workspace:ro" \
    -v "${RESULTS_DIR}:/results" \
    "${ADAPTER_IMAGE}" \
    python /workspace/benchmark/typesafe/adapter.py \
        --model-profiles /results/model-profiles.json \
        --efficient-model moonshotai/kimi-k2.7-code \
        --log /results/jev-classifier.jsonl >/dev/null

for _ in $(seq 1 30); do
    if docker exec "${ADAPTER_CONTAINER}" python -c \
        'import urllib.request; urllib.request.urlopen("http://127.0.0.1:8090/health", timeout=1)' \
        >/dev/null 2>&1; then
        break
    fi
    sleep 1
done
docker exec "${ADAPTER_CONTAINER}" python -c \
    'import urllib.request; urllib.request.urlopen("http://127.0.0.1:8090/health", timeout=1)' \
    >/dev/null || die "TypeSafe adapter did not become healthy"

COMMON=(
    --harbor-path "${HARBOR_PATH}"
    --agent "${AGENT}"
    --reasoning-effort "${REASONING_EFFORT}"
    --task-list-file "${TASK_LIST_FILE}"
    --n-concurrent "${N_CONCURRENT}"
    --max-retries "${MAX_RETRIES}"
    --foreground
)

SWITCHYARD_DOCKER_NETWORK="${NETWORK}" \
    bash "${ROOT}/benchmark/run-baseline.sh" \
    "${COMMON[@]}" \
    --output-dir "${RESULTS_DIR}/strong" \
    --server-config "${ROOT}/benchmark/server-configs/tb-lite-single-opus-4-7.toml" \
    --model tb-lite-single-opus-4-7

SWITCHYARD_DOCKER_NETWORK="${NETWORK}" \
    bash "${ROOT}/benchmark/run-baseline.sh" \
    "${COMMON[@]}" \
    --output-dir "${RESULTS_DIR}/weak" \
    --server-config "${ROOT}/benchmark/server-configs/tb-lite-single-kimi-k2-7-code.toml" \
    --model tb-lite-single-kimi-k2-7-code

SWITCHYARD_DOCKER_NETWORK="${NETWORK}" \
    bash "${ROOT}/benchmark/run-baseline.sh" \
    "${COMMON[@]}" \
    --output-dir "${RESULTS_DIR}/gemini" \
    --server-config "${ROOT}/benchmark/server-configs/tb-lite-llm-classifier-opus-kimi-gemini.toml" \
    --model switchyard

SWITCHYARD_DOCKER_NETWORK="${NETWORK}" \
    bash "${ROOT}/benchmark/run-baseline.sh" \
    "${COMMON[@]}" \
    --output-dir "${RESULTS_DIR}/jev" \
    --server-config "${ROOT}/benchmark/server-configs/tb-lite-typesafe-jev-opus-kimi.toml" \
    --model switchyard

python "${ROOT}/benchmark/typesafe/compare.py" "${RESULTS_DIR}"
