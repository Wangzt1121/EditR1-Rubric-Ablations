#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

ENV_FILE=${GEMINI_API_ENV_FILE:-${PROJECT_ROOT}/.env}
if [ -f "${ENV_FILE}" ]; then
    set -a
    . "${ENV_FILE}"
    set +a
fi

if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
    echo "Usage: $0 {l1_balanced|l3_prompt_identity_equal|l3_identity_priority} [--preflight-only]" >&2
    exit 2
fi

EXPERIMENT=$1
MODE=${2:-}
if [ -n "${MODE}" ] && [ "${MODE}" != "--preflight-only" ]; then
    echo "ERROR: unknown mode ${MODE}" >&2
    exit 2
fi
case "${EXPERIMENT}" in
    l1_balanced)
        REWARD_MODE=l1
        RUBRIC_PATH=${PROJECT_ROOT}/rubric_variants/l1_only_balanced/unified_l1_only.yaml
        DEFAULT_PORT=29511
        export L1_REWARD_USE_RESPONSE_SCHEMA=${L1_REWARD_USE_RESPONSE_SCHEMA:-1}
        ;;
    l3_prompt_identity_equal)
        REWARD_MODE=l3
        RUBRIC_PATH=${PROJECT_ROOT}/rubric_variants/l3_prompt_identity_equal
        DEFAULT_PORT=29512
        ;;
    l3_identity_priority)
        REWARD_MODE=l3
        RUBRIC_PATH=${PROJECT_ROOT}/rubric_variants/l3_identity_priority
        DEFAULT_PORT=29513
        ;;
    *)
        echo "ERROR: unknown ablation ${EXPERIMENT}" >&2
        exit 2
        ;;
esac

export REWARD_BACKEND=${REWARD_BACKEND:-official_gemini_native}
export SINGLE_REWARD_MODE=${REWARD_MODE}
export SINGLE_REWARD_RUBRIC_YAML=${RUBRIC_PATH}
export NUM_EPOCHS=150
export SAVE_FREQ=${SAVE_FREQ:-25}
export LORA_LEARNING_RATE=${LORA_LEARNING_RATE:-5e-4}
export EVAL_FREQ=${EVAL_FREQ:-50}
export SKIP_EVAL=${SKIP_EVAL:-0}
export MAX_EVAL_BATCHES=${MAX_EVAL_BATCHES:-0}
export SINGLE_REWARD_FAIL_OPEN=${SINGLE_REWARD_FAIL_OPEN:-0}
export OUTPUT_ROOT=${OUTPUT_ROOT:-${PROJECT_ROOT}/outputs}
export LOGDIR=${OUTPUT_ROOT}/${EXPERIMENT}
export MASTER_PORT=${MASTER_PORT:-${DEFAULT_PORT}}
export PYTHON_BIN=${PYTHON_BIN:-python}

cd "${PROJECT_ROOT}"
"${PYTHON_BIN}" tools/validate_ablation_setup.py --variant "${EXPERIMENT}" --check-env

if [ "${MODE}" = "--preflight-only" ]; then
    echo "[ablation] preflight passed: ${EXPERIMENT}"
    exit 0
fi

echo "[ablation] name=${EXPERIMENT} steps=${NUM_EPOCHS} lr=${LORA_LEARNING_RATE} eval_every=${EVAL_FREQ} eval_batches=all mode=${SINGLE_REWARD_MODE} rubric=${SINGLE_REWARD_RUBRIC_YAML} logdir=${LOGDIR}"
exec bash examples/train_kontext_gemini.sh kontext_single_api_reward
