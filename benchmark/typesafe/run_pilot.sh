#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 Inkwell Finance, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

export TASK_LIST_FILE="${TASK_LIST_FILE:-${ROOT}/benchmark/typesafe/pilot_tasks_2.txt}"
export RESULTS_DIR="${RESULTS_DIR:-${ROOT}/benchmark/typesafe/results/pilot-$(date -u +%Y%m%dT%H%M%SZ)}"
export N_CONCURRENT="${N_CONCURRENT:-2}"
export MAX_RETRIES="${MAX_RETRIES:-1}"

exec bash "${ROOT}/benchmark/typesafe/run_experiment.sh"
