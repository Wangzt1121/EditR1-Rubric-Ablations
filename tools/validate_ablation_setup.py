#!/usr/bin/env python3
"""Validate rubric variants and optional runtime inputs before training."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_WEIGHTS = {
    "l3_prompt_identity_equal": {
        "Prompt Compliance": 0.42,
        "Localization and Anatomical Integration": 0.11,
        "Subject Preservation": 0.42,
        "Task Realism and Image Quality": 0.05,
    },
    "l3_identity_priority": {
        "Prompt Compliance": 0.40,
        "Localization and Anatomical Integration": 0.11,
        "Subject Preservation": 0.44,
        "Task Realism and Image Quality": 0.05,
    },
}


def import_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def assert_weights(actual: dict[str, float], expected: dict[str, float], source: Path) -> None:
    canonical_actual: dict[str, float] = {}
    for key, value in actual.items():
        if key.startswith("Prompt Compliance"):
            canonical_key = "Prompt Compliance"
        elif "Localization" in key or "Integration" in key:
            canonical_key = "Localization and Anatomical Integration"
        elif key.startswith("Subject Preservation"):
            canonical_key = "Subject Preservation"
        elif key.startswith("Task Realism and Image Quality"):
            canonical_key = "Task Realism and Image Quality"
        else:
            canonical_key = key
        canonical_actual[canonical_key] = float(value)

    missing = set(expected) - set(canonical_actual)
    if missing:
        raise ValueError(f"{source}: missing L1 weights {sorted(missing)}")
    for key, expected_value in expected.items():
        actual_value = canonical_actual[key]
        if abs(actual_value - expected_value) > 1e-9:
            raise ValueError(
                f"{source}: {key} weight is {actual_value}, expected {expected_value}"
            )


def validate_l1() -> dict[str, Any]:
    helper = import_module(
        ROOT / "reward_server" / "l1_only_training_reward.py",
        "edit_r1_release_l1_helper",
    )
    path = ROOT / "rubric_variants" / "l1_only_balanced" / "unified_l1_only.yaml"
    rubric = helper.load_rubric(str(path))
    expected = EXPECTED_WEIGHTS["l3_prompt_identity_equal"]
    actual = {name: spec.weight for name, spec in rubric.dimensions.items()}
    assert_weights(actual, expected, path)
    return {"files": 1, "weights": actual}


def validate_l3(variant: str) -> dict[str, Any]:
    scorer = import_module(
        ROOT / "reward_server" / "test_gemini.py",
        f"edit_r1_release_l3_scorer_{variant}",
    )
    root = ROOT / "rubric_variants" / variant
    paths = sorted(root.glob("*.yaml"))
    if len(paths) != 50:
        raise ValueError(f"{root}: expected 50 YAML files, found {len(paths)}")

    task_numbers = sorted(int(path.name.split("_", 1)[0]) for path in paths)
    if task_numbers != list(range(1, 51)):
        raise ValueError(f"{root}: expected task prefixes 01-50, got {task_numbers}")

    expected = EXPECTED_WEIGHTS[variant]
    active_counts = set()
    for path in paths:
        rubric = scorer.load_rubric(str(path))
        assert_weights(rubric.l1_weights, expected, path)
        active_counts.add(len(rubric.active_l3))
        if len(rubric.active_l3) != len(set(rubric.active_l3)):
            raise ValueError(f"{path}: duplicate active_l3 ids")
        if set(rubric.score_mapping) != {str(value) for value in range(10)}:
            raise ValueError(f"{path}: L3 score mapping must contain exactly integer labels 0-9")
        if any(
            abs(float(rubric.score_mapping[str(value)]) - value / 9.0) > 1e-9
            for value in range(10)
        ):
            raise ValueError(f"{path}: L3 scores must be normalized by score/9")
        failure_policy = rubric.global_scoring_policy.get("failure_propagation")
        if not isinstance(failure_policy, dict) or not failure_policy.get("failures"):
            raise ValueError(f"{path}: missing task-specific failure_propagation policy")
    return {
        "files": len(paths),
        "weights": expected,
        "active_l3_counts": sorted(active_counts),
        "score_scale": "integer_0_9_normalized_by_9",
        "failure_propagation": True,
    }


def require_path(name: str, allow_model_id: bool = False) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise ValueError(f"Missing required environment variable {name}")
    if allow_model_id and "/" in value and not value.startswith(("/", ".")):
        return value
    path = Path(value).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"{name} does not exist: {path}")
    return str(path.resolve())


def validate_environment() -> dict[str, Any]:
    key_names = ("GEMINI_API_KEY", "GOOGLE_API_KEY", "SINGLE_REWARD_API_KEY")
    if not any(os.getenv(name, "").strip() for name in key_names):
        raise ValueError("Missing Gemini API key in the environment")
    visible = [item for item in os.getenv("CUDA_VISIBLE_DEVICES", "0").split(",") if item.strip()]
    nproc = int(os.getenv("NPROC_PER_NODE", str(len(visible))))
    if nproc != len(visible):
        raise ValueError(
            f"NPROC_PER_NODE={nproc} does not match CUDA_VISIBLE_DEVICES={visible}"
        )
    return {
        "pretrained_model": require_path("PRETRAINED_MODEL", allow_model_id=True),
        "lora_path": require_path("LORA_PATH"),
        "train_metadata": require_path("PROMPT_METADATA_FILE"),
        "eval_metadata": require_path("EVAL_PROMPT_METADATA_FILE"),
        "visible_gpu_count": len(visible),
        "api_key_present": True,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variant",
        choices=("l1_balanced", "l3_prompt_identity_equal", "l3_identity_priority"),
    )
    parser.add_argument("--check-env", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report: dict[str, Any] = {}
    if args.variant in (None, "l1_balanced"):
        report["l1_balanced"] = validate_l1()
    if args.variant in (None, "l3_prompt_identity_equal"):
        report["l3_prompt_identity_equal"] = validate_l3("l3_prompt_identity_equal")
    if args.variant in (None, "l3_identity_priority"):
        report["l3_identity_priority"] = validate_l3("l3_identity_priority")
    if args.check_env:
        report["environment"] = validate_environment()

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        for name, details in report.items():
            print(f"[ok] {name}: {details}")


if __name__ == "__main__":
    main()
