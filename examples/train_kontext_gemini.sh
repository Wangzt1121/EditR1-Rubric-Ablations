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

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTHON_BIN=${PYTHON_BIN:-python}
export REWARD_BACKEND=official_gemini_native
export SINGLE_REWARD_NATIVE_GEMINI=1
export GEMINI_NATIVE_API=1
export GEMINI_API_BASE_URL=${GEMINI_API_BASE_URL:-https://generativelanguage.googleapis.com/v1beta}
export SINGLE_REWARD_BASE_URL=${SINGLE_REWARD_BASE_URL:-${GEMINI_API_BASE_URL}}
export SINGLE_REWARD_MODEL=${SINGLE_REWARD_MODEL:-${GEMINI_REWARD_MODEL:-gemini-3.1-pro-preview}}
export SINGLE_REWARD_API_KEY=${SINGLE_REWARD_API_KEY:-${GEMINI_API_KEY:-${GOOGLE_API_KEY:-}}}

if [ -z "${SINGLE_REWARD_API_KEY}" ]; then
    echo "ERROR: GEMINI_API_KEY, GOOGLE_API_KEY, or SINGLE_REWARD_API_KEY is required." >&2
    exit 1
fi

export SINGLE_REWARD_MODE=${SINGLE_REWARD_MODE:-l3}
case "${SINGLE_REWARD_MODE}" in
    l1)
        export SINGLE_REWARD_HELPER_PATH=${PROJECT_ROOT}/reward_server/l1_only_training_reward.py
        export SINGLE_REWARD_RUBRIC_YAML=${SINGLE_REWARD_RUBRIC_YAML:-${PROJECT_ROOT}/rubric_variants/l1_only_balanced/unified_l1_only.yaml}
        ;;
    l3)
        export SINGLE_REWARD_HELPER_PATH=${PROJECT_ROOT}/reward_server/test_gemini_reward.py
        export SINGLE_REWARD_RUBRIC_YAML=${SINGLE_REWARD_RUBRIC_YAML:-${PROJECT_ROOT}/rubric_variants/l3_prompt_identity_equal}
        ;;
    *)
        echo "ERROR: SINGLE_REWARD_MODE must be l1 or l3." >&2
        exit 1
        ;;
esac

if [ "${USE_PROXY:-0}" = "1" ]; then
    if [ -z "${PROXY_ENV_FILE:-}" ] || [ ! -f "${PROXY_ENV_FILE}" ]; then
        echo "ERROR: USE_PROXY=1 requires an existing PROXY_ENV_FILE." >&2
        exit 1
    fi
    # shellcheck disable=SC1090
    . "${PROXY_ENV_FILE}"
fi

export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-true}
export WORLD_SIZE=${WORLD_SIZE:-1}
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export MASTER_PORT=${MASTER_PORT:-29501}
export RANK=${RANK:-0}
if [ -z "${NPROC_PER_NODE:-}" ]; then
    IFS=',' read -r -a VISIBLE_GPUS <<< "${CUDA_VISIBLE_DEVICES}"
    export NPROC_PER_NODE=${#VISIBLE_GPUS[@]}
fi

# Shared training configuration for all three controlled ablations.
export NUM_EPOCHS=${NUM_EPOCHS:-150}
export SAVE_FREQ=${SAVE_FREQ:-25}
export NUM_GROUPS_PER_EPOCH=${NUM_GROUPS_PER_EPOCH:-2}
export EVAL_FREQ=${EVAL_FREQ:-1}
export SKIP_EVAL=${SKIP_EVAL:-1}
export MAX_EVAL_BATCHES=${MAX_EVAL_BATCHES:-5}
export WANDB_MODE=${WANDB_MODE:-offline}

export DATASET_ROOT=${DATASET_ROOT:-data}
export PROMPT_METADATA_FILE=${PROMPT_METADATA_FILE:-${DATASET_ROOT}/train_metadata.jsonl}
export EVAL_PROMPT_METADATA_FILE=${EVAL_PROMPT_METADATA_FILE:-${DATASET_ROOT}/test_metadata.jsonl}
export EDIT_R1_TASK_PREFIXES=${EDIT_R1_TASK_PREFIXES:-}

export EDIT_R1_PAD_TO_SQUARE=${EDIT_R1_PAD_TO_SQUARE:-1}
export EDIT_R1_CANVAS_SIZE=${EDIT_R1_CANVAS_SIZE:-1024}
export TRAIN_RESOLUTION=${TRAIN_RESOLUTION:-1024}
export SAMPLE_NUM_STEPS=${SAMPLE_NUM_STEPS:-20}
export SAMPLE_EVAL_NUM_STEPS=${SAMPLE_EVAL_NUM_STEPS:-15}
export SAMPLE_GUIDANCE_SCALE=${SAMPLE_GUIDANCE_SCALE:-1.25}
export SAMPLE_NOISE_LEVEL=${SAMPLE_NOISE_LEVEL:-0.7}
export SAMPLE_DETERMINISTIC=${SAMPLE_DETERMINISTIC:-0}
export SAMPLE_SOLVER=${SAMPLE_SOLVER:-flow}
export EDIT_R1_EXPLICIT_CANDIDATE_SEEDS=${EDIT_R1_EXPLICIT_CANDIDATE_SEEDS:-1}

export PRETRAINED_MODEL=${PRETRAINED_MODEL:-black-forest-labs/FLUX.1-Kontext-dev}
export LORA_PATH=${LORA_PATH:-${TRAIN_LORA_PATH:-}}

export SINGLE_NUM_CANDIDATES=${SINGLE_NUM_CANDIDATES:-8}
export SINGLE_INITIAL_BSZ=${SINGLE_INITIAL_BSZ:-$([ "${NPROC_PER_NODE}" = "1" ] && echo 8 || echo 2)}
export SINGLE_REWARD_TIMEOUT=${SINGLE_REWARD_TIMEOUT:-240}
export SINGLE_REWARD_MAX_RETRIES=${SINGLE_REWARD_MAX_RETRIES:-8}
export SINGLE_REWARD_MAX_IMAGE_SIDE=${SINGLE_REWARD_MAX_IMAGE_SIDE:-1024}
export SINGLE_REWARD_JPEG_QUALITY=${SINGLE_REWARD_JPEG_QUALITY:-90}
export SINGLE_REWARD_WORKERS=${SINGLE_REWARD_WORKERS:-8}
export SINGLE_REWARD_DEADLINE_SECONDS=${SINGLE_REWARD_DEADLINE_SECONDS:-600}
export SINGLE_REWARD_FAIL_OPEN=${SINGLE_REWARD_FAIL_OPEN:-0}
export SINGLE_REWARD_API_LOCK=${SINGLE_REWARD_API_LOCK:-1}
export SINGLE_REWARD_API_CHANNELS=${SINGLE_REWARD_API_CHANNELS:-8}
export SINGLE_REWARD_DEBUG=${SINGLE_REWARD_DEBUG:-0}
export SINGLE_REWARD_DEBUG_LIMIT=${SINGLE_REWARD_DEBUG_LIMIT:-0}

CONFIG_NAME=${1:-kontext_single_api_reward}
cd "${PROJECT_ROOT}"

echo "[train_kontext] config=${CONFIG_NAME} mode=${SINGLE_REWARD_MODE} model=${SINGLE_REWARD_MODEL} rubric=${SINGLE_REWARD_RUBRIC_YAML} steps=${NUM_EPOCHS} gpus=${CUDA_VISIBLE_DEVICES}"

exec "${PYTHON_BIN}" -m torch.distributed.run \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --nnodes="${WORLD_SIZE}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    --node_rank="${RANK}" \
    scripts/train_nft_kontext.py --config "config/kontext_nft.py:${CONFIG_NAME}"
