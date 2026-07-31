#!/usr/bin/env python3
"""Repair duplicated cross-L1 facet identifiers in the release rubrics.

The source YAML files reused a small number of semantic names in two L1
groups. The original L3 loader rejects those files before training. This
one-time, idempotent migration gives the second role a distinct identifier
without changing its rubric text or weight.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import yaml


ROOT = Path(__file__).resolve().parents[1]

RENAMES: Dict[str, Dict[str, Dict[str, str]]] = {
    "13_男性粗大臂.yaml": {
        "Task Realism and Image Quality": {
            "arm_anatomy_pose_plausible": "arm_anatomy_pose_quality",
        },
    },
    "14_耳朵调整.yaml": {
        "Task Realism and Image Quality": {
            "inner_ear_structure_detail_match": "inner_ear_structure_detail_quality",
        },
    },
    "15_眼部美化.yaml": {
        "Task Realism and Image Quality": {
            "gaze_iris_pupil_preserved_eye": "gaze_iris_pupil_preservation_quality",
        },
    },
    "16_嘴唇.yaml": {
        "Task Realism and Image Quality": {
            "lip_texture_boundary_realistic": "lip_texture_boundary_quality",
        },
    },
    "17_发际线.yaml": {
        "Localization and Anatomical Integration": {
            "hairline_boundary_blend": "hairline_boundary_blend_localization",
        },
        "Task Realism and Image Quality": {
            "hairline_density_roots_babyhair_realistic": "hairline_density_roots_babyhair_quality",
        },
    },
}


def rename_mapping_key(mapping: dict, old: str, new: str) -> None:
    if new in mapping:
        return
    if old not in mapping:
        raise KeyError(f"Missing expected facet {old!r}")
    rebuilt = []
    for key, value in mapping.items():
        rebuilt.append((new if key == old else key, value))
    mapping.clear()
    mapping.update(rebuilt)


def flatten_taxonomy_ids(data: dict) -> list[str]:
    result: list[str] = []
    for l2_groups in data["taxonomy"].values():
        for facets in l2_groups.values():
            result.extend(str(key) for key in facets)
    return result


def repair(path: Path) -> bool:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    changed = False
    for l1_name, mappings in RENAMES[path.name].items():
        for old, new in mappings.items():
            for facets in data["taxonomy"][l1_name].values():
                if old in facets or new in facets:
                    before = tuple(facets)
                    rename_mapping_key(facets, old, new)
                    changed = changed or tuple(facets) != before

            weights = data.get("facet_weights_within_l1", {}).get(l1_name, {})
            if weights:
                before = tuple(weights)
                rename_mapping_key(weights, old, new)
                changed = changed or tuple(weights) != before

    canonical_active = flatten_taxonomy_ids(data)
    if data.get("active_l3") != canonical_active:
        data["active_l3"] = canonical_active
        changed = True

    if changed:
        version = str(data.get("version", ""))
        if not version.endswith("_unique_ids"):
            data["version"] = version + "_unique_ids"
        path.write_text(
            yaml.safe_dump(data, allow_unicode=True, sort_keys=False, width=120),
            encoding="utf-8",
        )
    return changed


def main() -> None:
    changed_paths = []
    for variant in ("l3_prompt_identity_equal", "l3_identity_priority"):
        root = ROOT / "rubric_variants" / variant
        for filename in RENAMES:
            path = root / filename
            if repair(path):
                changed_paths.append(path.relative_to(ROOT).as_posix())
    print(f"repaired={len(changed_paths)}")
    for path in changed_paths:
        print(path)


if __name__ == "__main__":
    main()
