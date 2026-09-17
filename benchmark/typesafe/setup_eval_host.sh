#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 Inkwell Finance, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
HARBOR_PATH="${HARBOR_PATH:-${ROOT}/benchmark/datasets/openthoughts-tblite-closed-book}"
PATCH_FILE="${ROOT}/benchmark/patches/harbor-agent-patches.diff"
PROFILE_PATH="${ROOT}/benchmark/typesafe/results/model-profiles.json"

die() {
    echo "ERROR: $*" >&2
    exit 1
}

[[ "$(uname -s)" == "Linux" ]] || die "This setup script requires Linux"
for command in docker uv patch; do
    command -v "${command}" >/dev/null || die "${command} is required"
done
docker info >/dev/null 2>&1 || die "Docker is installed but the daemon is unavailable"
docker compose version >/dev/null 2>&1 || die "Docker Compose is required"
[[ -n "${TYPESAFE_API_KEY:-}" ]] || die "TYPESAFE_API_KEY is required"
[[ -n "${OPENROUTER_API_KEY:-}" ]] || die "OPENROUTER_API_KEY is required"
[[ -f "${PATCH_FILE}" ]] || die "Harbor patch not found: ${PATCH_FILE}"

cd "${ROOT}"
uv sync --python 3.12

HARBOR_SITE="$(
    uv run --no-sync python - <<'PY'
import sysconfig

print(sysconfig.get_paths()["purelib"])
PY
)"

if patch --dry-run --silent -R -d "${HARBOR_SITE}" -p1 < "${PATCH_FILE}"; then
    echo "Harbor patch is already applied"
elif patch --dry-run --silent -d "${HARBOR_SITE}" -p1 < "${PATCH_FILE}"; then
    patch -d "${HARBOR_SITE}" -p1 < "${PATCH_FILE}"
else
    die "Harbor patch does not apply cleanly to ${HARBOR_SITE}"
fi

if [[ -f "${HARBOR_PATH}/switchyard_dataset_manifest.json" ]]; then
    echo "Using existing Harbor dataset at ${HARBOR_PATH}"
else
    uv run --no-sync python benchmark/prepare_harbor_dataset.py \
        --output-dir "${HARBOR_PATH}" \
        --overwrite
fi

uv run --no-sync python -m benchmark.typesafe.model_profiles \
    --output "${PROFILE_PATH}"

uv run --no-sync harbor --help >/dev/null
echo "Evaluation host is ready"
echo "Run: bash benchmark/typesafe/run_pilot.sh"
