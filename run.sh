#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJECT_ROOT}"

if [ ! -f .env ]; then
    echo "[run] .env not found; using variables already exported in the environment."
fi

case "${1:-}" in
    "")
        exec bash examples/run_all_ablations_150.sh
        ;;
    --preflight-only)
        exec bash examples/run_all_ablations_150.sh --preflight-only
        ;;
    *)
        echo "Usage: bash run.sh [--preflight-only]" >&2
        exit 2
        ;;
esac
