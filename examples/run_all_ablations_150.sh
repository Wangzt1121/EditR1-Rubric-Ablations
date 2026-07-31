#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODE=${1:-}

if [ -n "${MODE}" ] && [ "${MODE}" != "--preflight-only" ]; then
    echo "Usage: $0 [--preflight-only]" >&2
    exit 2
fi

for experiment in l1_balanced l3_prompt_identity_equal l3_identity_priority; do
    echo "[run_all] starting ${experiment}"
    bash "${SCRIPT_DIR}/run_ablation_150.sh" "${experiment}" ${MODE:+"${MODE}"}
done
