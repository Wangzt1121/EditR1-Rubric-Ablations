#!/usr/bin/env python3
"""Local synthetic tests for the L3 reward contract; no model or API required."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_scorer():
    path = ROOT / "reward_server" / "test_gemini.py"
    spec = importlib.util.spec_from_file_location("edit_r1_contract_scorer", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import scorer: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def first_rubric(variant: str) -> Path:
    paths = sorted((ROOT / "rubric_variants" / variant).glob("*.yaml"))
    if len(paths) != 50:
        raise AssertionError(f"{variant}: expected 50 rubrics, found {len(paths)}")
    return paths[0]


def synthetic_scores(scorer, rubric) -> list[int]:
    facets = scorer.flatten_facets(rubric.taxonomy)
    values = {
        "Prompt Compliance": 9,
        "Localization and Anatomical Integration": 5,
        "Subject Preservation": 2,
        "Task Realism and Image Quality": 7,
    }
    return [values[facets[facet_id].l1] for facet_id in rubric.active_l3]


def aggregate(scorer, rubric, scores: list[int]) -> dict:
    output = scorer.normalize_judge_output({"scores": scores}, rubric.active_l3)
    scorer.validate_judge_output(output, rubric.active_l3)
    result = scorer.aggregate_bottom_up(rubric=rubric, judge_output=output)
    if not 0.0 <= result["reward"] <= 1.0:
        raise AssertionError(f"Reward outside [0, 1]: {result['reward']}")
    return result


def main() -> None:
    scorer = load_scorer()
    schema = scorer.build_response_schema([f"facet_{idx}" for idx in range(22)])
    score_schema = schema["properties"]["scores"]
    assert schema["required"] == ["scores"]
    assert score_schema["minItems"] == score_schema["maxItems"] == 22
    assert score_schema["items"] == {"type": "integer", "minimum": 0, "maximum": 9}

    equal = scorer.load_rubric(str(first_rubric("l3_prompt_identity_equal")))
    priority = scorer.load_rubric(str(first_rubric("l3_identity_priority")))
    scores = synthetic_scores(scorer, equal)
    equal_result = aggregate(scorer, equal, scores)
    priority_result = aggregate(scorer, priority, scores)

    assert len(equal_result["l3_scores"]) == 22
    assert set(equal_result["l3_raw_labels"].values()) <= set(scorer.SCORE_LABELS)
    assert priority_result["reward"] < equal_result["reward"], (
        "Identity-priority weighting should reduce this deliberately low-preservation "
        "synthetic candidate relative to equal prompt/identity weighting."
    )
    print(
        "[ok] L3 contract: 22 integer scores in [0,9], /9 normalization, "
        f"equal_reward={equal_result['reward']:.6f}, "
        f"identity_priority_reward={priority_result['reward']:.6f}"
    )


if __name__ == "__main__":
    main()
