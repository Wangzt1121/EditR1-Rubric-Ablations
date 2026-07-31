# Edit-R1 Rubric Ablations

This is a self-contained Edit-R1 release for three controlled 150-step rubric
ablations. After filling one private `.env` file (or exporting the same
variables), `bash run.sh` validates and runs all three experiments sequentially
with shared model, data, sampling, and Gemini settings.

| Run | Reward | Rubric | L1 weights (TF / LI / SP / VQ) |
| --- | --- | --- | --- |
| `l1_balanced` | Direct four-dimensional L1 scoring | `unified_l1_only.yaml` | 0.42 / 0.11 / 0.42 / 0.05 |
| `l3_prompt_identity_equal` | 22-facet bottom-up L3 scoring | 50 task YAML files | 0.42 / 0.11 / 0.42 / 0.05 |
| `l3_identity_priority` | 22-facet bottom-up L3 scoring | 50 task YAML files | 0.40 / 0.11 / 0.44 / 0.05 |

Each L3 judge response is a fixed vector of 22 integers in `[0, 9]`. Scores
are normalized by `s_i / 9`, aggregated through `L3 -> L2 -> L1`, and then
combined with the selected L1 weights. The task-specific failure-propagation
policy is included in the judge prompt before the final score vector is
returned. `N/A` is not permitted.

## What is included

- `config/`, `flow_grpo/`, and `scripts/`: required Kontext training runtime
  and entry point.
- `reward_server/`: only the direct-L1 and bottom-up-L3 Gemini helpers used by
  these runs.
- `rubric_variants/`: one L1 rubric and two complete 50-task L3 variants.
- `examples/`: validated launchers for one run or all three runs.
- `tools/validate_ablation_setup.py`: static rubric and environment checks.

Model weights, datasets, outputs, checkpoints, caches, and API keys are not
included.

## Setup

Use Python 3.10 and the same CUDA environment as Edit-R1. Install the package:

```bash
python -m pip install -e .
```

Create a private environment file:

```bash
cp .env.example .env
```

Set `PYTHON_BIN`, `GEMINI_API_KEY`, `PRETRAINED_MODEL`, `LORA_PATH`,
`PROMPT_METADATA_FILE`, and `EVAL_PROMPT_METADATA_FILE` in `.env`.
The three runs must use the same model, LoRA, metadata, GPU count, sampling
settings, and candidate count for a valid ablation.

## Validate

```bash
bash run.sh --preflight-only
```

This checks all 101 YAML files, the fixed 0--9 L3 contract, expected L1
weights, required paths, credentials, and GPU settings. It makes no API calls
and does not load model weights.

## Run the three ablations

Each launcher first validates its rubric and environment, then runs 150
optimizer steps with fail-closed Gemini scoring and a LoRA learning rate of
`5e-4`. All other model, data, sampling, and candidate settings are shared
through `.env`.

The three experiments are parallel variants on the same `main` branch:

| Experiment | Launch command | Reward and weights (TF / LI / SP / VQ) |
| --- | --- | --- |
| `l1_balanced` | `bash examples/run_l1_balanced_150.sh` | Direct L1, 0.42 / 0.11 / 0.42 / 0.05 |
| `l3_prompt_identity_equal` | `bash examples/run_l3_prompt_identity_equal_150.sh` | Bottom-up L3, 0.42 / 0.11 / 0.42 / 0.05 |
| `l3_identity_priority` | `bash examples/run_l3_identity_priority_150.sh` | Bottom-up L3, 0.40 / 0.11 / 0.44 / 0.05 |

Their rubric and output directories follow the same names:

```text
rubric_variants/l1_only_balanced/       -> ${OUTPUT_ROOT}/l1_balanced
rubric_variants/l3_prompt_identity_equal/ -> ${OUTPUT_ROOT}/l3_prompt_identity_equal
rubric_variants/l3_identity_priority/     -> ${OUTPUT_ROOT}/l3_identity_priority
```

Run all three sequentially with the same environment:

```bash
bash run.sh
```

Output is written to `OUTPUT_ROOT/l1_balanced`,
`OUTPUT_ROOT/l3_prompt_identity_equal`, and
`OUTPUT_ROOT/l3_identity_priority`.

## Step accounting

`config/kontext_nft.py` uses one gradient update per outer epoch for these
launchers. Therefore `NUM_EPOCHS=150` produces 150 optimizer/global steps.
Checkpoints are saved every 25 epochs by default, including step 150.

The launchers set `EVAL_FREQ=50`, `SKIP_EVAL=0`, and
`EVAL_SAMPLE_COUNT=50`. Evaluation therefore reuses the same first 50 samples
from the validation manifest after optimizer steps 50, 100, and 150. The limit
is global across all GPUs, not per GPU.

## Reproducibility and security

- The official Gemini native endpoint is used by default.
- API failure is fail-closed (`SINGLE_REWARD_FAIL_OPEN=0`) for paper runs.
- No launcher contains a key. Keys are read only from the environment or the
  ignored `.env` file.
- `tools/test_reward_contract.py` verifies the fixed 22-score schema,
  normalization, and the expected effect of identity-priority weighting without
  calling an API.
- Each output directory stores the selected rubric path, reward details,
  training metrics, candidate grids, and checkpoints produced by Edit-R1.
- Run the three variants sequentially unless separate GPUs, API quotas, and
  unique `MASTER_PORT` values are explicitly assigned.
