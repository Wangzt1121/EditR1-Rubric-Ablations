#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
YAML-driven bottom-up GPT rubric scorer.

Purpose
-------
Test one source image + one prompt + N edited samples.

This script does NOT contain a built-in unified/default rubric.
You must provide your own task rubric via --rubric_yaml.

Pipeline
--------
source image + prompt + edited samples
    -> load YAML rubric
    -> Gemini scores exactly 22 active L3 facets with integers from 0 to 9
    -> Python maps labels to numeric values
    -> Python aggregates L3 -> L2 -> L1 -> overall reward
    -> Python applies soft preservation penalties
    -> outputs scores: List[float] and detailed JSON/CSV logs

Install
-------
pip install openai pillow pyyaml

Example
-------
export OPENAI_API_KEY="your_key"

python test_bottomup_rubric_yaml.py \
  --data_dir /nvmedata/workspace2/users/wzt/data \
  --rubric_yaml rubrics/hungarian_mustache_strict.yaml \
  --output_dir /nvmedata/workspace2/users/wzt/data/rubric_test_out

Or explicitly:

python test_bottomup_rubric_yaml.py \
  --source /nvmedata/workspace2/users/wzt/data/source.jpg \
  --prompt "Please add a Hungarian mustache to the character, making it as realistic as possible without blurring or smoothing. The added beard should not change the character's face shape, neck, or any other area outside the beard area." \
  --sample_dir /nvmedata/workspace2/users/wzt/data \
  --rubric_yaml rubrics/hungarian_mustache_strict.yaml \
  --output_dir /nvmedata/workspace2/users/wzt/data/rubric_test_out
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import base64
import colorsys
import csv
import hashlib
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps, ImageStat

try:
    import yaml
except ImportError as exc:
    raise ImportError("Missing dependency: pyyaml. Install with `pip install pyyaml`.") from exc


# ============================================================
# 1. Data structures
# ============================================================

SCORE_LABELS: Tuple[str, ...] = tuple(str(value) for value in range(10))
DEFAULT_SCORE_MAPPING: Dict[str, Optional[float]] = {
    label: int(label) / 9.0 for label in SCORE_LABELS
}


@dataclass(frozen=True)
class FacetSpec:
    facet_id: str
    l1: str
    l2: str
    title: str
    rubric: str
    anchors: Dict[str, str]


@dataclass(frozen=True)
class RubricSpec:
    task_key: str
    version: str
    judge_instructions: str
    taxonomy: Dict[str, Dict[str, Dict[str, FacetSpec]]]
    active_l3: List[str]
    score_mapping: Dict[str, Optional[float]]
    l1_weights: Dict[str, float]
    primary_edit_facets: List[str]
    penalty_groups: Dict[str, List[str]]
    global_scoring_policy: Dict[str, Any]
    source_path: str


# ============================================================
# 2. YAML loading and validation
# ============================================================

def load_yaml(path: str) -> Dict[str, Any]:
    yaml_path = Path(path)
    if not yaml_path.exists():
        raise FileNotFoundError(f"Rubric YAML not found: {path}")

    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Rubric YAML must be a mapping/object: {path}")

    return data


def normalize_openai_compatible_base_url(base_url: str) -> str:
    """Accept either a base URL or a full chat/responses endpoint.

    OpenAI-compatible clients append /chat/completions internally. Some relay
    dashboards show the full endpoint, e.g. https://host/v1/chat/completions;
    using that value directly would produce a duplicated path.
    """
    value = str(base_url or "").strip().rstrip("/")
    if not value:
        return ""
    lower = value.lower()
    for suffix in ("/chat/completions", "/responses"):
        if lower.endswith(suffix):
            value = value[: -len(suffix)].rstrip("/")
            break
    return value


def load_default_judge_env(path: str = "/home/student/.config/monet/judge.env") -> None:
    env_path = Path(path)
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'\"")
        if key and key not in os.environ:
            os.environ[key] = value




def parse_score_mapping(raw: Optional[Dict[str, Any]]) -> Dict[str, Optional[float]]:
    mapping: Dict[str, Optional[float]] = dict(DEFAULT_SCORE_MAPPING)

    if raw:
        for key, value in raw.items():
            k = str(key)
            if value is None:
                mapping[k] = None
            else:
                mapping[k] = float(value)

    required = set(SCORE_LABELS)
    missing = required - set(mapping.keys())
    if missing:
        raise ValueError(f"score_mapping missing keys: {sorted(missing)}")

    return mapping


def parse_taxonomy(raw_taxonomy: Any) -> Dict[str, Dict[str, Dict[str, FacetSpec]]]:
    """
    Expected YAML format:

    taxonomy:
      L1 name:
        L2 name:
          l3_id:
            title: ...
            rubric: ...

    Also supports:
          l3_id: "rubric text"
    """
    if not isinstance(raw_taxonomy, dict) or not raw_taxonomy:
        raise ValueError("taxonomy must be a non-empty mapping: L1 -> L2 -> L3")

    taxonomy: Dict[str, Dict[str, Dict[str, FacetSpec]]] = {}
    seen_facets: Dict[str, Tuple[str, str]] = {}

    for l1_name, l2_dict in raw_taxonomy.items():
        if not isinstance(l2_dict, dict) or not l2_dict:
            raise ValueError(f"taxonomy[{l1_name!r}] must be a non-empty mapping of L2 groups")

        l1 = str(l1_name)
        taxonomy[l1] = {}

        for l2_name, l3_dict in l2_dict.items():
            if not isinstance(l3_dict, dict) or not l3_dict:
                raise ValueError(f"taxonomy[{l1!r}][{l2_name!r}] must be a non-empty mapping of L3 facets")

            l2 = str(l2_name)
            taxonomy[l1][l2] = {}

            for facet_id_raw, spec_raw in l3_dict.items():
                facet_id = str(facet_id_raw).strip()
                if not facet_id:
                    raise ValueError(f"Empty facet_id under {l1}/{l2}")

                if facet_id in seen_facets:
                    prev_l1, prev_l2 = seen_facets[facet_id]
                    raise ValueError(
                        f"Duplicate facet_id {facet_id!r}: already defined under {prev_l1}/{prev_l2}, "
                        f"redefined under {l1}/{l2}"
                    )
                seen_facets[facet_id] = (l1, l2)

                if isinstance(spec_raw, dict):
                    title = str(spec_raw.get("title", facet_id)).strip()
                    # Newer 22-facet rubrics call the criterion text
                    # "definition"; older Edit-R1 files call it "rubric".
                    # Accept both without changing the original scorer path.
                    rubric = str(
                        spec_raw.get("rubric")
                        or spec_raw.get("definition")
                        or ""
                    ).strip()
                    if not rubric:
                        raise ValueError(
                            f"Facet {facet_id!r} missing rubric/definition text"
                        )
                    anchors_raw = spec_raw.get("anchors") or {}
                    anchors = {
                        str(key): str(value).strip()
                        for key, value in anchors_raw.items()
                    } if isinstance(anchors_raw, dict) else {}
                else:
                    title = facet_id
                    rubric = str(spec_raw).strip()
                    if not rubric:
                        raise ValueError(f"Facet {facet_id!r} has empty rubric text")
                    anchors = {}

                taxonomy[l1][l2][facet_id] = FacetSpec(
                    facet_id=facet_id,
                    l1=l1,
                    l2=l2,
                    title=title,
                    rubric=rubric,
                    anchors=anchors,
                )

    return taxonomy


def flatten_facets(taxonomy: Dict[str, Dict[str, Dict[str, FacetSpec]]]) -> Dict[str, FacetSpec]:
    out: Dict[str, FacetSpec] = {}
    for l2_dict in taxonomy.values():
        for l3_dict in l2_dict.values():
            out.update(l3_dict)
    return out


def active_l1_names(taxonomy: Dict[str, Dict[str, Dict[str, FacetSpec]]], active_l3: Iterable[str]) -> List[str]:
    active = set(active_l3)
    names = []
    for l1, l2_dict in taxonomy.items():
        has_active = any(facet_id in active for l3_dict in l2_dict.values() for facet_id in l3_dict.keys())
        if has_active:
            names.append(l1)
    return names


def parse_l1_weights(raw: Optional[Dict[str, Any]], active_l1: List[str]) -> Dict[str, float]:
    if raw:
        weights = {str(k): float(v) for k, v in raw.items()}
    else:
        # No built-in rubric weights. If YAML omits weights, use equal weights over active L1s.
        weights = {l1: 1.0 for l1 in active_l1}

    for l1 in active_l1:
        if l1 not in weights:
            raise ValueError(
                f"l1_weights missing active L1 {l1!r}. "
                f"Either add it to YAML l1_weights or remove all its active L3 facets."
            )

    for l1, w in weights.items():
        if w < 0:
            raise ValueError(f"l1_weights[{l1!r}] must be non-negative")

    active_weight_sum = sum(weights[l1] for l1 in active_l1)
    if active_weight_sum <= 0:
        raise ValueError("Sum of active L1 weights must be > 0")

    return weights



def dedupe_preserve_order(values: Iterable[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for value in values:
        item = str(value).strip()
        if not item or item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def parse_facet_id_list(raw: Any, active_l3: List[str], field_name: str) -> List[str]:
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ValueError(f"{field_name} must be a list of active L3 facet ids")

    active = set(active_l3)
    values: List[str] = []
    for idx, item in enumerate(raw):
        facet_id = str(item).strip()
        if not facet_id:
            continue
        if facet_id not in active:
            raise ValueError(f"{field_name}[{idx}]={facet_id!r} is not in active_l3")
        values.append(facet_id)
    return dedupe_preserve_order(values)


def parse_penalty_groups(raw: Any, active_l3: List[str]) -> Dict[str, List[str]]:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError("penalty_groups must be a mapping")

    groups: Dict[str, List[str]] = {}
    for group_name_raw, spec in raw.items():
        group_name = str(group_name_raw).strip()
        if not group_name:
            continue
        if isinstance(spec, dict):
            facet_items = spec.get("facet_ids", [])
        else:
            facet_items = spec
        groups[group_name] = parse_facet_id_list(facet_items, active_l3, f"penalty_groups.{group_name}.facet_ids")
    return groups


def facet_context_text(spec: FacetSpec) -> str:
    return " ".join([spec.facet_id, spec.title, spec.l1, spec.l2, spec.rubric]).lower()


def infer_primary_edit_facets(
    taxonomy: Dict[str, Dict[str, Dict[str, FacetSpec]]],
    active_l3: List[str],
) -> List[str]:
    active = set(active_l3)
    primary: List[str] = []
    for l1, l2_dict in taxonomy.items():
        l1_lower = str(l1).lower()
        is_non_target_l1 = (
            "non-target" in l1_lower
            or "non target" in l1_lower
            or "non_target" in l1_lower
        )
        is_preservation_or_consistency_l1 = (
            "preservation" in l1_lower
            or "consistency" in l1_lower
        )
        is_target_l1 = (
            not is_non_target_l1
            and (
                "target edit" in l1_lower
                or "prompt" in l1_lower
                or l1_lower.strip() in {"edit correctness", "target correctness", "prompt edit correctness"}
                or ("edit" in l1_lower and not is_preservation_or_consistency_l1)
            )
        )
        if not is_target_l1:
            continue
        for l3_dict in l2_dict.values():
            for facet_id in l3_dict:
                if facet_id in active:
                    primary.append(facet_id)

    if primary:
        return dedupe_preserve_order(primary)

    generic_priority = [
        "target_edit_applied",
        "primary_instruction_fulfilled",
        "target_region_correct",
        "all_requested_attributes_covered",
        "edit_strength_appropriate",
        "no_wrong_edit_type",
        "hair_edit_applied",
        "primary_hair_instruction_fulfilled",
        "facial_hair_added_or_edited",
        "prompt_facial_hair_category_correct",
    ]
    return [facet_id for facet_id in generic_priority if facet_id in active]



def infer_penalty_groups(
    taxonomy: Dict[str, Dict[str, Dict[str, FacetSpec]]],
    active_l3: List[str],
    primary_edit_facets: List[str],
) -> Dict[str, List[str]]:
    """Infer generic penalty groups from each task's own L1/L2/L3 names.

    The groups are intentionally task-agnostic: every task can define target
    edits differently, while preservation, color, geometry, localization, and
    quality are recognized from the facet names and hierarchy. Rubric prose is
    used only for explicit defect words, because many task facets contain broad
    words such as "natural" that should not make every facet a visual-quality
    penalty.
    """
    active = set(active_l3)
    primary = set(primary_edit_facets)
    all_facets = flatten_facets(taxonomy)

    groups: Dict[str, List[str]] = {
        "prompt_following": [facet_id for facet_id in primary_edit_facets if facet_id in active],
        "color_preservation": [],
        "geometry_framing_preservation": [],
        "identity_preservation": [],
        "non_target_preservation": [],
        "localization_blending": [],
        "visual_quality": [],
    }

    color_terms = (
        "color", "tone", "skin tone", "exposure", "lighting", "brightness",
        "contrast", "saturation", "white balance", "shadow", "highlight",
        "recolor", "filter", "photometric", "washed out", "tint", "relit",
    )
    geometry_terms = (
        "pose", "position", "scale", "framing", "camera", "angle", "distance",
        "head size", "subject framing", "body pose", "view", "zoom", "crop",
        "perspective", "translated", "re-centered", "proportion",
    )
    identity_short_terms = (
        "identity", "same person", "face shape", "facial features", "expression",
        "skin texture", "face skin", "face brightness", "over-smoothing",
        "beautification", "age impression", "gaze", "facial geometry",
    )
    non_target_short_terms = (
        "non-target", "non target", "non_target", "unchanged", "preserved",
        "clothing", "background", "accessories", "hair preserved", "makeup",
        "body pose", "scene", "lighting", "global color", "camera angle",
    )
    localization_terms = (
        "localization", "localized", "confined", "leakage", "boundary", "blend",
        "blending", "occlusion", "edge", "bleeding", "halo", "spill", "mask",
    )
    quality_short_terms = (
        "visual", "fidelity", "quality", "photoreal", "realistic", "plausible",
        "texture", "blur", "smoothing", "artifact", "duplicate", "broken",
        "distortion", "waxy", "plastic", "anatomy", "resolution", "sharpness",
    )
    explicit_quality_prose_terms = (
        "visible artifact", "artifacts", "blurred", "blurry", "over-smoothed",
        "waxy", "plastic", "distorted", "broken", "duplicated", "melted",
        "unnatural texture", "compression", "low resolution", "sharpness",
    )

    def contains_any(value: str, terms: Tuple[str, ...]) -> bool:
        return any(term in value for term in terms)

    for facet_id in active_l3:
        if facet_id in primary:
            continue
        spec = all_facets[facet_id]
        short_text = " ".join([spec.facet_id, spec.title, spec.l1, spec.l2]).lower()
        full_text = facet_context_text(spec)
        l1_lower = spec.l1.lower()
        l2_lower = spec.l2.lower()
        is_target_l1 = "target edit correctness" in l1_lower or "target" == l1_lower.strip()

        if not is_target_l1 and contains_any(full_text, color_terms):
            groups["color_preservation"].append(facet_id)
        if not is_target_l1 and contains_any(full_text, geometry_terms):
            groups["geometry_framing_preservation"].append(facet_id)
        if (
            ("identity" in l1_lower or contains_any(short_text, identity_short_terms))
            and facet_id not in groups["color_preservation"]
            and facet_id not in groups["geometry_framing_preservation"]
        ):
            groups["identity_preservation"].append(facet_id)
        if (
            ("non-target" in l1_lower or "non target" in l1_lower or contains_any(short_text, non_target_short_terms))
            and facet_id not in groups["color_preservation"]
            and facet_id not in groups["geometry_framing_preservation"]
        ):
            groups["non_target_preservation"].append(facet_id)
        if (
            "localization" in l1_lower
            or "blending" in l1_lower
            or "localization" in l2_lower
            or "blending" in l2_lower
            or contains_any(short_text, localization_terms)
            or contains_any(full_text, ("leakage", "bleeding", "halo", "hard mask", "boundary artifact"))
        ):
            groups["localization_blending"].append(facet_id)
        if (
            "visual" in l1_lower
            or "fidelity" in l1_lower
            or "quality" in l1_lower
            or contains_any(short_text, quality_short_terms)
            or contains_any(full_text, explicit_quality_prose_terms)
        ):
            groups["visual_quality"].append(facet_id)

    return {group: dedupe_preserve_order(facets) for group, facets in groups.items() if facets}

def load_rubric(path: str) -> RubricSpec:
    raw = load_yaml(path)

    for key in ["task_key", "taxonomy", "active_l3"]:
        if key not in raw:
            raise ValueError(f"Rubric YAML missing required key: {key}")

    task_key = str(raw["task_key"]).strip()
    version = str(raw.get("version", "unknown")).strip()
    judge_instructions = str(raw.get("judge_instructions", "")).strip()
    taxonomy = parse_taxonomy(raw["taxonomy"])
    all_facets = flatten_facets(taxonomy)

    active_l3_raw = raw["active_l3"]
    if not isinstance(active_l3_raw, list) or not active_l3_raw:
        raise ValueError("active_l3 must be a non-empty list")

    active_l3 = [str(x).strip() for x in active_l3_raw]
    if len(active_l3) != len(set(active_l3)):
        duplicates = sorted({x for x in active_l3 if active_l3.count(x) > 1})
        raise ValueError(f"active_l3 contains duplicates: {duplicates}")

    missing = [facet_id for facet_id in active_l3 if facet_id not in all_facets]
    if missing:
        raise ValueError(f"active_l3 contains facets not defined in taxonomy: {missing}")

    score_mapping = parse_score_mapping(raw.get("score_mapping"))
    l1_names = active_l1_names(taxonomy, active_l3)
    l1_weights = parse_l1_weights(raw.get("l1_weights"), l1_names)
    primary_edit_facets = parse_facet_id_list(raw.get("primary_edit_facets"), active_l3, "primary_edit_facets")
    if not primary_edit_facets:
        primary_edit_facets = infer_primary_edit_facets(taxonomy, active_l3)

    inferred_penalty_groups = infer_penalty_groups(taxonomy, active_l3, primary_edit_facets)
    penalty_groups = parse_penalty_groups(raw.get("penalty_groups"), active_l3)
    for group_name, facet_ids in inferred_penalty_groups.items():
        penalty_groups.setdefault(group_name, facet_ids)

    return RubricSpec(
        task_key=task_key,
        version=version,
        judge_instructions=judge_instructions,
        taxonomy=taxonomy,
        active_l3=active_l3,
        score_mapping=score_mapping,
        l1_weights=l1_weights,
        primary_edit_facets=primary_edit_facets,
        penalty_groups=penalty_groups,
        global_scoring_policy=(
            raw.get("global_scoring_policy")
            if isinstance(raw.get("global_scoring_policy"), dict)
            else {}
        ),
        source_path=str(Path(path)),
    )


def rubric_to_jsonable(rubric: RubricSpec) -> Dict[str, Any]:
    taxonomy_json: Dict[str, Dict[str, Dict[str, Dict[str, str]]]] = {}
    for l1, l2_dict in rubric.taxonomy.items():
        taxonomy_json[l1] = {}
        for l2, l3_dict in l2_dict.items():
            taxonomy_json[l1][l2] = {}
            for fid, spec in l3_dict.items():
                taxonomy_json[l1][l2][fid] = {
                    "title": spec.title,
                    "rubric": spec.rubric,
                    "anchors": spec.anchors,
                }

    return {
        "task_key": rubric.task_key,
        "version": rubric.version,
        "judge_instructions": rubric.judge_instructions,
        "source_path": rubric.source_path,
        "score_mapping": rubric.score_mapping,
        "l1_weights": rubric.l1_weights,
        "primary_edit_facets": rubric.primary_edit_facets,
        "penalty_groups": rubric.penalty_groups,
        "global_scoring_policy": rubric.global_scoring_policy,
        "taxonomy": taxonomy_json,
        "active_l3": rubric.active_l3,
    }


# ============================================================
# 3. Input utilities
# ============================================================

def _looks_like_metadata_key(key: str) -> bool:
    key = key.strip()
    if not key or len(key) > 80 or any(ch.isspace() for ch in key):
        return False
    return all(ch.isalnum() or ch in "_.-" for ch in key)


def parse_prompt_metadata(prompt_file: Path) -> Dict[str, str]:
    if not prompt_file.exists():
        return {}

    text = prompt_file.read_text(encoding="utf-8", errors="replace")
    metadata: Dict[str, str] = {}
    current_key: Optional[str] = None
    current_lines: List[str] = []

    def flush() -> None:
        nonlocal current_key, current_lines
        if current_key:
            metadata[current_key] = "\n".join(current_lines).strip()
        current_key = None
        current_lines = []

    for raw_line in text.splitlines():
        line = raw_line.rstrip()

        # prompt/requirement bodies often contain natural-language colons, e.g.
        # "main character: change the hairstyle...". Only compact field-like
        # keys such as prompt, requirement, candidate_01, reward_mean start a
        # new metadata field.
        if line.endswith(":") and _looks_like_metadata_key(line[:-1]):
            flush()
            current_key = line[:-1].strip()
            current_lines = []
            continue

        if ":" in line and not line.startswith((" ", "\t")):
            key, value = line.split(":", 1)
            if _looks_like_metadata_key(key):
                flush()
                metadata[key.strip()] = value.strip()
                continue

        if current_key:
            current_lines.append(line)

    flush()
    return metadata


def resolve_inputs(args: argparse.Namespace) -> argparse.Namespace:
    data_dir = Path(args.data_dir)
    prompt_path = Path(args.prompt_file) if args.prompt_file else data_dir / "prompt.txt"
    metadata = parse_prompt_metadata(prompt_path)

    if not args.source:
        args.source = str(data_dir / "source.jpg")

    if not args.prompt:
        args.prompt = metadata.get("prompt", "")
        if not args.prompt and prompt_path.exists():
            # Some datasets store prompt.txt as plain text instead of
            # key-value metadata. Treat the whole non-empty file as prompt.
            args.prompt = prompt_path.read_text(encoding="utf-8", errors="replace").strip()

    if not args.requirement:
        args.requirement = metadata.get("requirement", "")

    if not args.sample_dir and not args.samples:
        args.sample_dir = str(data_dir)

    if not args.prompt:
        raise ValueError(
            f"Prompt is empty. Please set --prompt or add prompt.txt under {data_dir}. "
            "prompt.txt may be either plain text or contain a prompt: field."
        )

    if not Path(args.source).exists():
        raise FileNotFoundError(f"Source image not found: {args.source}")

    return args


def collect_sample_paths(args: argparse.Namespace) -> List[str]:
    if args.samples:
        paths = [str(Path(p)) for p in args.samples]
    elif args.sample_dir:
        sample_dir = Path(args.sample_dir)
        if not sample_dir.exists():
            raise FileNotFoundError(f"Sample directory not found: {sample_dir}")

        exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        source_abs = Path(args.source).resolve()

        all_images = [
            p for p in sorted(sample_dir.iterdir())
            if p.is_file() and p.suffix.lower() in exts and p.resolve() != source_abs
        ]

        if args.candidate_prefix:
            prefixed = [p for p in all_images if p.name.startswith(args.candidate_prefix)]
            paths = [str(p) for p in prefixed]
        else:
            paths = [str(p) for p in all_images]
    else:
        raise ValueError("Please provide --samples or --sample_dir")

    if args.expected_samples > 0 and len(paths) != args.expected_samples:
        raise ValueError(
            f"Expected exactly {args.expected_samples} sample images, got {len(paths)}. "
            f"Use --expected_samples 0 to disable this check."
        )

    for path in paths:
        if not Path(path).exists():
            raise FileNotFoundError(f"Sample image not found: {path}")

    return paths


# ============================================================
# 4. Image and prompt construction
# ============================================================

def image_to_data_url(path: str, max_side: int = 1024, quality: int = 90) -> str:
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    image = Image.open(path_obj).convert("RGB")
    w, h = image.size
    scale = min(1.0, float(max_side) / float(max(w, h)))

    if scale < 1.0:
        image = image.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)

    import io
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=quality)
    encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"


def build_rubric_text(rubric: RubricSpec) -> str:
    active = set(rubric.active_l3)
    lines: List[str] = []

    for l1, l2_dict in rubric.taxonomy.items():
        l1_has_active = False

        for l2, l3_dict in l2_dict.items():
            active_items = [(fid, spec) for fid, spec in l3_dict.items() if fid in active]
            if not active_items:
                continue

            if not l1_has_active:
                lines.append(f"# L1: {l1}")
                l1_has_active = True

            lines.append(f"## L2: {l2}")
            for fid, spec in active_items:
                lines.append(f"- L3 `{fid}` ({spec.title}): {spec.rubric}")
                if spec.anchors:
                    anchors = "; ".join(
                        f"{score}={description}"
                        for score, description in sorted(
                            spec.anchors.items(),
                            key=lambda item: int(item[0]) if item[0].isdigit() else 999,
                        )
                    )
                    lines.append(f"  Anchors: {anchors}")

        if l1_has_active:
            lines.append("")

    return "\n".join(lines).strip()


def build_response_schema(active_l3: List[str]) -> Dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "scores": {
                "type": "array",
                "minItems": len(active_l3),
                "maxItems": len(active_l3),
                "items": {"type": "integer", "minimum": 0, "maximum": 9},
            },
        },
        "required": ["scores"],
    }


def extract_json_object_text(text: Any) -> str:
    """Extract a JSON object from model text, tolerating fenced JSON output."""
    if isinstance(text, list):
        parts: List[str] = []
        for item in text:
            if isinstance(item, dict):
                parts.append(str(item.get("text", "")))
            else:
                parts.append(str(item))
        text = "\n".join(parts)

    raw = str(text).strip()
    if raw.startswith("```"):
        raw = raw.strip("`").strip()
        if raw.lower().startswith("json"):
            raw = raw[4:].strip()

    start = raw.find("{")
    end = raw.rfind("}")
    if start >= 0 and end > start:
        return raw[start:end + 1]
    return raw


def parse_json_object(text: Any) -> Dict[str, Any]:
    """Parse the first valid JSON object from model output.

    Gemini sometimes returns a valid JSON object followed by another object or
    short note. Use the first valid object so training does not fail on harmless
    trailing content, while schema validation still catches real bad outputs.
    """
    extracted = extract_json_object_text(text)
    try:
        obj = json.loads(extracted)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError as first_err:
        raw = str(text or "").strip()
        decoder = json.JSONDecoder()
        fallback_obj: Optional[Dict[str, Any]] = None
        for start, ch in enumerate(raw):
            if ch != "{":
                continue
            try:
                obj, _ = decoder.raw_decode(raw[start:])
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                if "facet_scores" in obj:
                    return obj
                if fallback_obj is None:
                    fallback_obj = obj
        if fallback_obj is not None:
            return fallback_obj
        raise first_err
    raise ValueError("Model output did not contain a JSON object.")


def normalize_judge_output(output: Dict[str, Any], active_l3: List[str]) -> Dict[str, Any]:
    """Normalize the fixed 0--9 score vector to the legacy facet representation."""
    if not isinstance(output, dict):
        output = {}

    vector = output.get("scores", output.get("score_vector"))
    vector_items = vector if isinstance(vector, list) else []
    raw_items = output.get("facet_scores")
    raw_items = raw_items if isinstance(raw_items, list) else []
    by_id = {
        str(item.get("facet_id", "")).strip(): item
        for item in raw_items
        if isinstance(item, dict) and str(item.get("facet_id", "")).strip()
    }

    def normalize_score(value: Any) -> Optional[str]:
        if isinstance(value, bool):
            return None
        try:
            numeric = int(value)
        except (TypeError, ValueError):
            return None
        if str(value).strip() not in {str(numeric), f"{numeric}.0"}:
            return None
        return str(numeric) if 0 <= numeric <= 9 else None

    normalized_items: List[Dict[str, Any]] = []
    missing: List[str] = []
    for index, fid in enumerate(active_l3):
        if index < len(vector_items):
            raw_score = vector_items[index]
            item = {"facet_id": fid, "score": raw_score, "evidence": ""}
        else:
            item = dict(by_id.get(fid) or {})
        score = normalize_score(item.get("score")) if item else None
        if score is None:
            missing.append(fid)
            score = "0"
        normalized_items.append({
            "facet_id": fid,
            "score": score,
            "score_int": int(score),
            "evidence": str(item.get("evidence", "")) if item else "",
        })

    output["facet_scores"] = normalized_items
    output["scores"] = [item["score_int"] for item in normalized_items]
    tags = output.get("failure_tags")
    output["failure_tags"] = tags if isinstance(tags, list) else []
    if missing:
        output["failure_tags"].append("missing_or_invalid_l3_scores_counted_as_zero")
        output["missing_l3_facets"] = missing
    return output


def validate_judge_output(output: Dict[str, Any], active_l3: List[str]) -> None:
    expected = set(active_l3)
    items = output.get("facet_scores", [])
    got = [x.get("facet_id") for x in items]
    got_set = set(got)

    if got_set != expected:
        missing = sorted(expected - got_set)
        extra = sorted(got_set - expected)
        raise ValueError(f"Facet mismatch. missing={missing}, extra={extra}")

    if len(got) != len(got_set):
        raise ValueError(f"Duplicate facet_id in output: {got}")

    for item in items:
        if item.get("score") not in SCORE_LABELS:
            raise ValueError(f"Invalid score: {item}")


def mean_valid(values: Iterable[Optional[float]]) -> Optional[float]:
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def image_consistency_metrics(source_path: str, edited_path: str, side: int = 256) -> Dict[str, float]:
    def one(path: str) -> Dict[str, Any]:
        image = Image.open(path).convert("RGB").resize((side, side), Image.Resampling.LANCZOS)
        gray = image.convert("L")
        rgb_stat = ImageStat.Stat(image)
        gray_stat = ImageStat.Stat(gray)
        pixels = list(image.getdata())
        hsv_samples = []
        for r, g, b in pixels[::64]:
            _, s, v = colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)
            hsv_samples.append((s, v))
        saturation = sum(item[0] for item in hsv_samples) / max(1, len(hsv_samples)) * 255.0
        edge_strength = ImageStat.Stat(gray.filter(ImageFilter.FIND_EDGES)).mean[0]
        return {
            "rgb_mean": rgb_stat.mean,
            "brightness": gray_stat.mean[0],
            "contrast": gray_stat.stddev[0],
            "saturation": saturation,
            "edge_strength": edge_strength,
        }

    source = one(source_path)
    edited = one(edited_path)
    rgb_mean_l2 = math.sqrt(
        sum((edited["rgb_mean"][idx] - source["rgb_mean"][idx]) ** 2 for idx in range(3))
    )
    return {
        "rgb_mean_l2": float(rgb_mean_l2),
        "brightness_delta": float(edited["brightness"] - source["brightness"]),
        "contrast_delta": float(edited["contrast"] - source["contrast"]),
        "saturation_delta": float(edited["saturation"] - source["saturation"]),
        "edge_sharpness_delta": float(edited["edge_strength"] - source["edge_strength"]),
    }


def format_consistency_metrics(metrics: Dict[str, float]) -> str:
    return (
        f"rgb_mean_l2={metrics['rgb_mean_l2']:.2f}; "
        f"brightness_delta={metrics['brightness_delta']:.2f}; "
        f"contrast_delta={metrics['contrast_delta']:.2f}; "
        f"saturation_delta={metrics['saturation_delta']:.2f}; "
        f"edge_sharpness_delta={metrics['edge_sharpness_delta']:.2f}"
    )



def metric_color_consistency_penalty(metrics: Dict[str, float]) -> Tuple[float, float, List[str]]:
    """Continuous full-image photometric/detail penalty."""
    rgb = abs(float(metrics.get("rgb_mean_l2", 0.0)))
    brightness = abs(float(metrics.get("brightness_delta", 0.0)))
    contrast_delta = float(metrics.get("contrast_delta", 0.0))
    contrast = abs(contrast_delta)
    saturation = abs(float(metrics.get("saturation_delta", 0.0)))
    edge_delta = float(metrics.get("edge_sharpness_delta", 0.0))
    edge_loss = max(0.0, -edge_delta)

    penalty = 0.0
    reasons: List[str] = []

    def add(amount: float, reason: str) -> None:
        nonlocal penalty
        penalty += amount
        reasons.append(reason)

    if rgb >= 35:
        add(0.14, "large_rgb_mean_shift")
    elif rgb >= 25:
        add(0.10, "moderate_rgb_mean_shift")
    elif rgb >= 18:
        add(0.06, "mild_rgb_mean_shift")
    elif rgb >= 10:
        add(0.03, "small_rgb_mean_shift")

    if brightness >= 20:
        add(0.12, "large_brightness_shift")
    elif brightness >= 12:
        add(0.08, "moderate_brightness_shift")
    elif brightness >= 6:
        add(0.04, "mild_brightness_shift")

    if saturation >= 30:
        add(0.13, "large_saturation_shift")
    elif saturation >= 18:
        add(0.08, "moderate_saturation_shift")
    elif saturation >= 9:
        add(0.04, "mild_saturation_shift")

    if contrast >= 14:
        add(0.08, "large_contrast_shift")
    elif contrast >= 8:
        add(0.04, "moderate_contrast_shift")

    if edge_loss >= 5.0 or contrast_delta <= -14:
        add(0.12, "severe_over_smoothing_or_texture_loss")
    elif edge_loss >= 3.0 or contrast_delta <= -10:
        add(0.09, "large_over_smoothing_or_texture_loss")
    elif edge_loss >= 1.5 or contrast_delta <= -7:
        add(0.06, "moderate_over_smoothing_or_texture_loss")
    elif edge_loss >= 0.8 or contrast_delta <= -4.5:
        add(0.03, "mild_over_smoothing_or_texture_loss")

    penalty = min(penalty, 0.36)
    quality_score = _clamp_float(1.0 - penalty / 0.36, 0.0, 1.0)
    return penalty, quality_score, reasons

def _first_float_optional(mapping: Dict[str, Any], keys: List[str]) -> Optional[float]:
    for key in keys:
        if key in mapping and mapping.get(key) is not None:
            value = _float_or_none(mapping.get(key))
            if value is not None:
                return value
    return None


def _first_float(mapping: Dict[str, Any], keys: List[str], default: float = 0.0) -> float:
    value = _first_float_optional(mapping, keys)
    return float(default if value is None else value)


def _l1_by_keywords(l1_scores: Dict[str, Any], required: List[str], excluded: List[str] = []) -> Optional[float]:
    for name, value in l1_scores.items():
        lower = str(name).lower()
        if all(token.lower() in lower for token in required) and not any(token.lower() in lower for token in excluded):
            parsed = _float_or_none(value)
            if parsed is not None:
                return parsed
    return None


def _task_prompt_l1_score(l1_scores: Dict[str, Any]) -> float:
    direct = _first_float_optional(l1_scores, [
        "Prompt Facial-Hair Correctness",
        "Hair Edit Correctness",
        "Prompt Edit Correctness",
        "Edit Correctness",
    ])
    if direct is not None:
        return float(direct)
    keyword = _l1_by_keywords(l1_scores, ["correctness"], ["identity", "non-target", "localization", "visual"])
    return float(0.0 if keyword is None else keyword)


def _task_identity_l1_score(l1_scores: Dict[str, Any]) -> float:
    direct = _first_float_optional(l1_scores, [
        "Identity and Geometry Preservation",
        "Identity and Face Preservation",
        "Identity Preservation",
    ])
    if direct is not None:
        return float(direct)
    keyword = _l1_by_keywords(l1_scores, ["identity"])
    return float(0.0 if keyword is None else keyword)


def _task_non_target_l1_score(l1_scores: Dict[str, Any]) -> float:
    direct = _first_float_optional(l1_scores, [
        "Non-target Region Preservation",
        "Non-target Consistency",
        "Non-target Preservation",
    ])
    if direct is not None:
        return float(direct)
    keyword = _l1_by_keywords(l1_scores, ["non-target"])
    return float(0.0 if keyword is None else keyword)


def _task_localization_l1_score(l1_scores: Dict[str, Any], fallback: float) -> float:
    direct = _first_float_optional(l1_scores, ["Localization and Blending", "Localization"])
    if direct is not None:
        return float(direct)
    keyword = _l1_by_keywords(l1_scores, ["localization"])
    return float(fallback if keyword is None else keyword)


def _task_quality_l1_score(l1_scores: Dict[str, Any]) -> Optional[float]:
    direct = _first_float_optional(l1_scores, ["Visual Fidelity", "Image Quality", "Quality"])
    if direct is not None:
        return float(direct)
    return _l1_by_keywords(l1_scores, ["fidelity"])


def _task_color_l1_score(l1_scores: Dict[str, Any]) -> float:
    direct = _first_float_optional(l1_scores, ["Color Exposure and Image Consistency"])
    if direct is not None:
        return float(direct)
    non_target = _task_non_target_l1_score(l1_scores)
    quality = _task_quality_l1_score(l1_scores)
    if quality is None:
        return float(non_target)
    return float(min(non_target, quality))



def apply_soft_preservation_penalties(result: Dict[str, Any], metrics: Dict[str, float], rubric: RubricSpec) -> Dict[str, Any]:
    """Apply task-agnostic soft gates after normalized 0--9 rubric scoring.

    The bottom-up reward already reflects the full normalized facet score.
    This stage only adds a small extra correction for
    clear failures and objective color/detail drift. It intentionally avoids
    summing every partial facet across overlapping groups, because the same
    visible issue can appear in color, non-target, localization, and quality
    groups at once.
    """
    l3_scores = result.get("l3_scores", {}) or {}
    l3_labels = result.get("l3_raw_labels", {}) or {}
    penalties: List[Dict[str, Any]] = []
    group_scores: Dict[str, Optional[float]] = {}
    group_label_counts: Dict[str, Dict[str, int]] = {}
    group_penalty_values: Dict[str, float] = {}

    group_penalty_config = {
        # These values operate on fail/partial ratios, not raw counts.
        "prompt_following": {"fail": 0.34, "partial": 0.08, "max": 0.22},
        "color_preservation": {"fail": 0.30, "partial": 0.08, "max": 0.20},
        "geometry_framing_preservation": {"fail": 0.38, "partial": 0.10, "max": 0.24},
        "identity_preservation": {"fail": 0.34, "partial": 0.07, "max": 0.20},
        "non_target_preservation": {"fail": 0.28, "partial": 0.07, "max": 0.18},
        "localization_blending": {"fail": 0.24, "partial": 0.06, "max": 0.16},
        "visual_quality": {"fail": 0.26, "partial": 0.06, "max": 0.18},
    }

    for group_name, facet_ids in (rubric.penalty_groups or {}).items():
        active_ids = [facet_id for facet_id in facet_ids if facet_id in l3_scores]
        labels = [str(l3_labels.get(facet_id, "")) for facet_id in active_ids]
        numeric_labels = [int(label) for label in labels if label in SCORE_LABELS]
        scores = [l3_scores.get(facet_id) for facet_id in active_ids]
        group_scores[group_name] = mean_valid(scores)

        counts = {
            "fail": sum(1 for score in numeric_labels if score <= 3),
            "partial": sum(1 for score in numeric_labels if 4 <= score <= 6),
            "excellent": sum(1 for score in numeric_labels if score >= 7),
            "invalid": len(labels) - len(numeric_labels),
            "total": len(numeric_labels),
        }
        valid_count = max(1, counts["total"])
        counts["fail_ratio"] = counts["fail"] / valid_count
        counts["partial_ratio"] = counts["partial"] / valid_count
        group_label_counts[group_name] = counts

        config = group_penalty_config.get(group_name)
        if not config or not active_ids:
            continue
        amount = min(
            float(config["max"]),
            counts["fail_ratio"] * float(config["fail"]) + counts["partial_ratio"] * float(config["partial"]),
        )
        group_penalty_values[group_name] = amount
        if amount > 0:
            penalties.append({
                "reason": f"{group_name}_weak_facets",
                "amount": round(amount, 6),
                "source": "l3_group_ratio",
                "failures": counts["fail"],
                "partials": counts["partial"],
                "fail_ratio": round(counts["fail_ratio"], 6),
                "partial_ratio": round(counts["partial_ratio"], 6),
            })

    metric_penalty, metric_quality_score, metric_reasons = metric_color_consistency_penalty(metrics)
    if metric_penalty > 0:
        penalties.append({
            "reason": "objective_color_detail_drift",
            "amount": round(metric_penalty, 6),
            "source": "metric",
            "metric_reasons": metric_reasons,
        })

    base_reward = float(result["reward"])
    l1_scores = result.get("l1_scores", {}) or {}

    def fallback_group(name: str, fallback: Optional[float]) -> float:
        value = _float_or_none(group_scores.get(name))
        if value is not None:
            return float(value)
        parsed = _float_or_none(fallback)
        return 1.0 if parsed is None else float(parsed)

    prompt_score = fallback_group("prompt_following", _task_prompt_l1_score(l1_scores))
    localization_score = fallback_group("localization_blending", _task_localization_l1_score(l1_scores, prompt_score))
    color_score = fallback_group("color_preservation", _task_color_l1_score(l1_scores))
    geometry_score = fallback_group("geometry_framing_preservation", None)
    identity_score = fallback_group("identity_preservation", _task_identity_l1_score(l1_scores))
    non_target_score = fallback_group("non_target_preservation", _task_non_target_l1_score(l1_scores))
    quality_score = fallback_group("visual_quality", _task_quality_l1_score(l1_scores))

    edit_score = min(prompt_score, localization_score)
    preservation_score = min(identity_score, non_target_score, geometry_score)
    color_detail_score = min(color_score, metric_quality_score)
    quality_detail_score = min(quality_score, metric_quality_score)
    essential_bottleneck = min(edit_score, preservation_score, color_detail_score, quality_detail_score)

    prompt_penalty = group_penalty_values.get("prompt_following", 0.0)
    preservation_penalty = max(
        group_penalty_values.get("color_preservation", 0.0),
        group_penalty_values.get("geometry_framing_preservation", 0.0),
        group_penalty_values.get("identity_preservation", 0.0),
        group_penalty_values.get("non_target_preservation", 0.0),
    )
    quality_penalty = max(
        group_penalty_values.get("localization_blending", 0.0),
        group_penalty_values.get("visual_quality", 0.0),
    )

    total_penalty = min(
        0.34,
        prompt_penalty
        + 0.65 * preservation_penalty
        + 0.45 * quality_penalty
        + 0.45 * metric_penalty,
    )
    reward_after_penalty = max(0.0, base_reward - total_penalty)

    gate_factor = 0.82 + 0.18 * _clamp_float(essential_bottleneck, 0.0, 1.0)
    severe_multiplier = 1.0
    if group_label_counts.get("geometry_framing_preservation", {}).get("fail", 0) > 0:
        severe_multiplier *= 0.78
    elif group_label_counts.get("geometry_framing_preservation", {}).get("partial_ratio", 0.0) >= 0.35:
        severe_multiplier *= 0.94
    if group_label_counts.get("prompt_following", {}).get("fail_ratio", 0.0) >= 0.25:
        severe_multiplier *= 0.78
    if group_label_counts.get("color_preservation", {}).get("fail_ratio", 0.0) >= 0.25 or metric_penalty >= 0.24:
        severe_multiplier *= 0.82
    elif metric_penalty >= 0.16:
        severe_multiplier *= 0.92
    if group_label_counts.get("visual_quality", {}).get("fail_ratio", 0.0) >= 0.25:
        severe_multiplier *= 0.84

    final_reward = _clamp_float(reward_after_penalty * gate_factor * severe_multiplier, 0.0, 1.0)

    result["reward_raw_bottom_up"] = base_reward
    result["reward_before_soft_penalty"] = base_reward
    result["soft_penalty_total"] = total_penalty
    result["soft_penalties"] = penalties
    result["group_penalty_values"] = group_penalty_values
    result["reward_after_soft_penalty"] = reward_after_penalty
    result["penalty_group_scores"] = group_scores
    result["penalty_group_label_counts"] = group_label_counts
    result["metric_color_penalty"] = metric_penalty
    result["metric_color_quality_score"] = metric_quality_score
    result["metric_color_penalty_reasons"] = metric_reasons
    result["edit_group_score"] = edit_score
    result["preservation_group_score"] = preservation_score
    result["color_detail_group_score"] = color_detail_score
    result["quality_group_score"] = quality_detail_score
    result["essential_bottleneck"] = essential_bottleneck
    result["gate_factor"] = gate_factor
    result["severe_multiplier"] = severe_multiplier
    result["reward_after_soft_penalty_and_gate"] = final_reward
    result["reward_after_soft_penalty_before_floor"] = reward_after_penalty
    result["prompt_correctness_floor"] = 0.0
    result["reward_floor_applied"] = False
    result["three_pillar_color"] = color_detail_score
    result["three_pillar_preservation"] = preservation_score
    result["three_pillar_edit"] = edit_score
    result["three_pillar_bottleneck"] = essential_bottleneck
    result["three_pillar_gate_factor"] = gate_factor
    result["reward_after_three_pillar_gate"] = final_reward
    result["reward"] = final_reward
    return result

def _clamp_float(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


def _mean_numeric(values: Iterable[Any], default: float = 0.0) -> float:
    nums: List[float] = []
    for value in values:
        parsed = _float_or_none(value)
        if parsed is not None:
            nums.append(float(parsed))
    return float(default if not nums else sum(nums) / len(nums))


def _hash_unit(value: str) -> float:
    digest = hashlib.sha256(str(value).encode("utf-8", errors="replace")).hexdigest()
    return int(digest[:12], 16) / float(0xFFFFFFFFFFFF)


def apply_training_reward_calibration(
    result: Dict[str, Any],
    *,
    sample_path: str,
    sample_index: int,
    floor: float = 0.001,
    ceiling: float = 0.995,
) -> Dict[str, Any]:
    """Add a small deterministic tie-breaker to normalized 0--9 rewards.

    The judge produces auditable integer 0--9 facet scores. This post-processing
    keeps that raw score, then adds a small deterministic continuous component
    from preservation metrics and a hash tiebreaker. The goal is not to change
    obvious rankings, but to avoid identical/zero rewards that make pairwise RL
    sample construction brittle.
    """
    raw_reward = _clamp_float(float(result.get("reward", 0.0) or 0.0), 0.0, ceiling)
    l3_labels = result.get("l3_raw_labels", {}) or {}
    valid_labels = [str(label) for label in l3_labels.values() if str(label) in SCORE_LABELS]
    label_counts = {label: valid_labels.count(label) for label in SCORE_LABELS}
    valid_label_count = len(valid_labels)
    dominant_label = None
    dominant_ratio = 0.0
    if valid_label_count > 0:
        dominant_label = max(label_counts, key=lambda label: label_counts[label])
        dominant_ratio = label_counts[dominant_label] / float(valid_label_count)
    result["judge_label_counts"] = label_counts
    result["judge_label_valid_count"] = valid_label_count
    result["judge_label_dominant"] = dominant_label
    result["judge_label_dominant_ratio"] = dominant_ratio
    # A near-uniform label distribution is usually a judge calibration failure
    # for this dense L3 rubric. Real edits almost always have a mix of excellent,
    # partial, and sometimes failed preservation/target facets.
    result["judge_uniform_score_warning"] = bool(valid_label_count >= 12 and dominant_ratio >= 0.90)

    metrics = result.get("image_consistency_metrics", {}) or {}

    rgb = abs(float(metrics.get("rgb_mean_l2", 0.0) or 0.0))
    brightness = abs(float(metrics.get("brightness_delta", 0.0) or 0.0))
    contrast = abs(float(metrics.get("contrast_delta", 0.0) or 0.0))
    saturation = abs(float(metrics.get("saturation_delta", 0.0) or 0.0))
    edge_loss = max(0.0, -float(metrics.get("edge_sharpness_delta", 0.0) or 0.0))

    metric_penalty = (
        min(rgb / 80.0, 0.35)
        + min(brightness / 60.0, 0.20)
        + min(contrast / 45.0, 0.15)
        + min(saturation / 90.0, 0.15)
        + min(edge_loss / 20.0, 0.15)
    )
    metric_quality = _clamp_float(1.0 - metric_penalty, 0.0, 1.0)

    l1_mean = _mean_numeric((result.get("l1_scores") or {}).values(), default=raw_reward)
    l2_values: List[Any] = []
    for l2_dict in (result.get("l2_scores") or {}).values():
        if isinstance(l2_dict, dict):
            l2_values.extend(l2_dict.values())
    l2_mean = _mean_numeric(l2_values, default=l1_mean)
    pillar_mean = _mean_numeric([
        result.get("three_pillar_color"),
        result.get("three_pillar_preservation"),
        result.get("three_pillar_edit"),
        result.get("metric_color_quality_score"),
    ], default=raw_reward)
    hash_signal = _hash_unit(f"{sample_index}:{sample_path}:{raw_reward:.12f}")

    quality_signal = _clamp_float(
        0.36 * metric_quality
        + 0.24 * l1_mean
        + 0.18 * l2_mean
        + 0.17 * pillar_mean
        + 0.05 * hash_signal,
        0.0,
        1.0,
    )
    # Keep the calibration small. It spreads ties and rescues exact zero, while
    # preserving the coarse judge's main preference signal.
    delta_scale = 0.004 + 0.018 * (1.0 - min(abs(raw_reward - 0.5) * 2.0, 1.0))
    delta = delta_scale * (0.82 * quality_signal + 0.18 * hash_signal)
    calibrated = _clamp_float(raw_reward + delta, floor, ceiling)

    saturation_cap = None
    saturation_reason = None
    if result.get("judge_uniform_score_warning") and dominant_label is not None:
        if int(dominant_label) >= 8:
            # A near-uniform 8/9 vector is usually an uncalibrated judge response.
            saturation_cap = 0.80 + 0.05 * metric_quality
            saturation_reason = "dominant_all_excellent_labels"
    if saturation_cap is not None:
        saturation_cap = _clamp_float(float(saturation_cap), floor, ceiling)
        if calibrated > saturation_cap:
            calibrated = saturation_cap

    result["reward_saturation_cap"] = saturation_cap
    result["reward_saturation_cap_reason"] = saturation_reason

    result["reward_raw_discrete"] = raw_reward
    result["reward_for_training"] = calibrated
    result["reward_tiebreaker_signal"] = quality_signal
    result["reward_tiebreaker_hash"] = hash_signal
    result["reward_tiebreaker_delta"] = calibrated - raw_reward
    result["reward_floor"] = floor
    result["reward_ceiling"] = ceiling
    result["reward"] = calibrated
    return result


def enforce_unique_training_rewards(
    rows: List[Dict[str, Any]],
    *,
    min_gap: float = 0.001,
    floor: float = 0.001,
    ceiling: float = 0.995,
) -> None:
    """Make final rewards strictly unique for preference-pair construction."""
    order = sorted(
        range(len(rows)),
        key=lambda idx: (
            float(rows[idx].get("reward", 0.0) or 0.0),
            float(rows[idx].get("reward_tiebreaker_signal", 0.0) or 0.0),
            float(rows[idx].get("reward_tiebreaker_hash", 0.0) or 0.0),
            -int(rows[idx].get("sample_index", idx + 1) or (idx + 1)),
        ),
        reverse=True,
    )
    previous = ceiling + min_gap
    total = len(order)
    for rank, idx in enumerate(order, start=1):
        row = rows[idx]
        before = _clamp_float(float(row.get("reward", 0.0) or 0.0), floor, ceiling)
        target = min(before, previous - min_gap)
        if target < floor:
            target = floor + max(0, total - rank) * (min_gap / max(total, 1))
        target = _clamp_float(target, floor, ceiling)
        row["reward_before_unique_adjustment"] = before
        row["reward_unique_adjustment"] = target - before
        row["reward_unique_rank"] = rank
        row["reward"] = round(target, 6)
        row["reward_for_training"] = row["reward"]
        previous = target


def _fmt_score(value: Any) -> str:
    value = _float_or_none(value)
    return "n/a" if value is None else f"{value:.3f}"


def _collect_facet_issues(
    result: Dict[str, Any],
    facet_ids: List[str],
    *,
    max_items: int = 3,
) -> List[Dict[str, Any]]:
    l3_scores = result.get("l3_scores", {}) or {}
    l3_labels = result.get("l3_raw_labels", {}) or {}
    l3_evidence = result.get("l3_evidence", {}) or {}
    issues: List[Dict[str, Any]] = []
    for facet_id in facet_ids:
        label = str(l3_labels.get(facet_id, ""))
        if label not in {"0", "1"}:
            continue
        score = _float_or_none(l3_scores.get(facet_id))
        if score is None:
            continue
        evidence = str(l3_evidence.get(facet_id, "")).strip()
        issues.append({
            "facet_id": facet_id,
            "score": score,
            "label": l3_labels.get(facet_id),
            "evidence": evidence,
        })
    issues.sort(key=lambda item: (float(item.get("score") or 0.0), item["facet_id"]))
    return issues[:max_items]



def build_candidate_reason(result: Dict[str, Any], rubric: RubricSpec) -> Tuple[str, Dict[str, Any]]:
    """Build a compact, task-agnostic explanation for the final reward."""
    group_facets = rubric.penalty_groups or {}
    group_scores = result.get("penalty_group_scores", {}) or {}
    metric_quality = _float_or_none(result.get("metric_color_quality_score"))
    final_reward = _float_or_none(result.get("reward"))

    group_issues: Dict[str, List[Dict[str, Any]]] = {
        group_name: _collect_facet_issues(result, facet_ids, max_items=4)
        for group_name, facet_ids in group_facets.items()
    }

    def score_of(group_name: str) -> Optional[float]:
        return _float_or_none(group_scores.get(group_name))

    priority = [
        ("prompt_following", "Prompt following is the main problem."),
        ("geometry_framing_preservation", "Camera view, head/body scale, pose, or framing changed."),
        ("color_preservation", "Color, saturation, brightness, lighting, or source tone changed."),
        ("identity_preservation", "Identity or facial geometry preservation is weak."),
        ("non_target_preservation", "Non-target regions changed outside the requested edit."),
        ("visual_quality", "Visual quality, texture, smoothing, or artifacts are the main problem."),
        ("localization_blending", "Localization or blending around the edit is weak."),
    ]

    primary = "Prompt is mostly followed and preservation/quality are comparatively strong."
    issue_bucket: List[Dict[str, Any]] = []
    for group_name, message in priority:
        group_score = score_of(group_name)
        if group_score is not None and group_score < 0.78:
            primary = message
            issue_bucket = group_issues.get(group_name, [])
            break

    if metric_quality is not None and metric_quality < 0.78:
        primary = "Objective color/detail metrics indicate visible global drift or texture loss."
        issue_bucket = issue_bucket or []

    evidence_bits = []
    for item in issue_bucket[:2]:
        evidence = item.get("evidence") or item["facet_id"]
        evidence_bits.append(f"{item['facet_id']}={item.get('label')}: {evidence}")
    if not evidence_bits:
        penalties = result.get("soft_penalties", []) or []
        evidence_bits = [str(item.get("reason", "")) for item in penalties[:2] if item.get("reason")]

    details = {
        "group_scores": group_scores,
        "group_issues": group_issues,
        "soft_penalties": result.get("soft_penalties", [])[:8],
        "metric_color_penalty_reasons": result.get("metric_color_penalty_reasons", []),
    }

    reason = (
        f"{primary} "
        f"final={_fmt_score(final_reward)}, edit={_fmt_score(result.get('edit_group_score'))}, "
        f"preservation={_fmt_score(result.get('preservation_group_score'))}, "
        f"color/detail={_fmt_score(result.get('color_detail_group_score'))}, "
        f"quality={_fmt_score(result.get('quality_group_score'))}, "
        f"metric_quality={_fmt_score(metric_quality)}."
    )
    if evidence_bits:
        reason += " Evidence: " + " | ".join(evidence_bits[:2])
    return reason, details


# ============================================================
# 5. GPT API: L3 scoring only
# ============================================================

def call_gpt_l3_judge(
    *,
    client: Any,
    model: str,
    source_image_path: str,
    edited_image_path: str,
    prompt: str,
    rubric: RubricSpec,
    requirement: str = "",
    max_retries: int = 3,
    max_image_side: int = 1024,
    jpeg_quality: int = 90,
) -> Dict[str, Any]:
    active_l3 = rubric.active_l3
    rubric_text = build_rubric_text(rubric)
    schema = build_response_schema(active_l3)
    consistency_metrics = image_consistency_metrics(source_image_path, edited_image_path)

    source_url = image_to_data_url(source_image_path, max_side=max_image_side, quality=jpeg_quality)
    edited_url = image_to_data_url(edited_image_path, max_side=max_image_side, quality=jpeg_quality)

    system_prompt = (
        "You are a strict expert evaluator for fine-grained human image editing. "
        "You compare the source image and the edited image under the user's edit instruction. "
        "You must score only the listed L3 rubric facets. "
        "Do not output an overall score. "
        "Do not output L1 or L2 scores. "
        "Use only visible evidence in the images. "
        "Be strict about target-style mismatch, identity drift, non-target changes, global color shifts, "
        "edit leakage, blur, smoothing, and artifacts."
    )

    user_text = f"""
# Task
task_key: {rubric.task_key}
rubric_version: {rubric.version}

# Edit instruction
{prompt}

# Additional requirement
{requirement or "None"}

# Objective image consistency hints
These values compare the source image and edited image over the full image.
They are not the final score, but they should guide preservation facets.
Large absolute changes usually mean global color, brightness, contrast, or quality drift outside the target edit.
{format_consistency_metrics(consistency_metrics)}

# Scoring contract
Return exactly {len(active_l3)} integer scores in the listed active-L3 order.
Each score must be from 0 to 9; N/A is not allowed. Use the facet-specific
anchors below. Globally, 0--3 denotes a major failure, 4--5 partial or mediocre
satisfaction, 6 mostly correct with one clear issue, 7 strong but visibly
imperfect, 8 nearly exact, and 9 fully satisfied with no visible defect.

Judge facets independently. Do not let realism compensate for an incorrect
target edit, or let target compliance compensate for identity or non-target
drift. Apply every task-specific failure-propagation cap in the global policy
before returning the final score vector.

# Task-specific global scoring and failure-propagation policy
{json.dumps(rubric.global_scoring_policy, ensure_ascii=False, indent=2)}

# Rubric-specific judge instructions
{rubric.judge_instructions or "None"}

# Rubric source audit
rubric_source_path: {rubric.source_path}
rubric_task_key: {rubric.task_key}
rubric_version: {rubric.version}
rubric_checklist_sha256: {hashlib.sha256(rubric_text.encode("utf-8", errors="replace")).hexdigest()}

# Active L3 facet ids
{json.dumps(active_l3, ensure_ascii=False)}

# L1/L2/L3 rubric checklist actually used for this judgment
{rubric_text}

# Exact compact JSON schema
{json.dumps(schema, ensure_ascii=False)}

Return only JSON matching the schema. Do not output facet names, evidence,
explanations, markdown, L1/L2 values, an overall score, or failure reasoning.
""".strip()

    last_err: Optional[Exception] = None
    base_url_text = str(getattr(client, "base_url", "") or "").lower()
    use_chat_completions = model.lower().startswith("gemini") or "generativelanguage.googleapis.com" in base_url_text
    prefer_plain_chat = (
        "generativelanguage.googleapis.com" in base_url_text
        and os.getenv("GEMINI_OPENAI_PLAIN_JSON", "1").strip().lower() not in {"0", "false", "no", "n"}
    )

    for attempt in range(1, max_retries + 1):
        try:
            if use_chat_completions:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": user_text},
                            {"type": "text", "text": "Source image before editing:"},
                            {"type": "image_url", "image_url": {"url": source_url}},
                            {"type": "text", "text": "Edited image generated by the model:"},
                            {"type": "image_url", "image_url": {"url": edited_url}},
                        ],
                    },
                ]
                chat_kwargs = {
                    "model": model,
                    "temperature": 0,
                    "max_tokens": 4096,
                    "messages": messages,
                }
                if prefer_plain_chat:
                    response = client.chat.completions.create(**chat_kwargs)
                else:
                    try:
                        response = client.chat.completions.create(
                            **chat_kwargs,
                            response_format={
                                "type": "json_schema",
                                "json_schema": {
                                    "name": "bottomup_l3_rubric_scores",
                                    "strict": True,
                                    "schema": schema,
                                },
                            },
                        )
                    except Exception:
                        try:
                            response = client.chat.completions.create(
                                **chat_kwargs,
                                response_format={"type": "json_object"},
                            )
                        except Exception:
                            # Some Gemini-compatible relays do not implement the
                            # OpenAI response_format extension. The prompt still
                            # requires strict JSON, so fall back to plain chat.
                            response = client.chat.completions.create(**chat_kwargs)
                output_text = response.choices[0].message.content
            else:
                response = client.responses.create(
                    model=model,
                    temperature=0,
                    max_output_tokens=4096,
                    input=[
                        {"role": "system", "content": system_prompt},
                        {
                            "role": "user",
                            "content": [
                                {"type": "input_text", "text": user_text},
                                {"type": "input_text", "text": "Source image before editing:"},
                                {"type": "input_image", "image_url": source_url},
                                {"type": "input_text", "text": "Edited image generated by the model:"},
                                {"type": "input_image", "image_url": edited_url},
                            ],
                        },
                    ],
                    text={
                        "format": {
                            "type": "json_schema",
                            "name": "bottomup_l3_rubric_scores",
                            "strict": True,
                            "schema": schema,
                        }
                    },
                )
                output_text = response.output_text

            output = normalize_judge_output(parse_json_object(output_text), active_l3)
            validate_judge_output(output, active_l3)
            return output

        except Exception as err:
            last_err = err
            if attempt < max_retries:
                time.sleep(2 * attempt)
            else:
                raise RuntimeError(f"GPT judge failed after {max_retries} attempts: {last_err}") from err

    raise RuntimeError("Unexpected GPT judge failure")


# ============================================================
# 6. Python aggregation: L3 -> L2 -> L1 -> Overall
# ============================================================


def aggregate_bottom_up(
    *,
    rubric: RubricSpec,
    judge_output: Dict[str, Any],
) -> Dict[str, Any]:
    all_facets = flatten_facets(rubric.taxonomy)
    raw_by_id = {item["facet_id"]: item for item in judge_output["facet_scores"]}

    l3_scores: Dict[str, Optional[float]] = {}
    l3_raw_labels: Dict[str, str] = {}
    l3_evidence: Dict[str, str] = {}

    for facet_id in rubric.active_l3:
        item = raw_by_id[facet_id]
        label = item["score"]

        if label not in rubric.score_mapping:
            raise ValueError(f"Score label {label!r} not found in score_mapping")

        l3_raw_labels[facet_id] = label
        l3_scores[facet_id] = rubric.score_mapping[label]
        l3_evidence[facet_id] = item.get("evidence", "")

    tree_values: Dict[str, Dict[str, List[Optional[float]]]] = {}

    for facet_id, score in l3_scores.items():
        spec = all_facets[facet_id]
        tree_values.setdefault(spec.l1, {}).setdefault(spec.l2, []).append(score)

    l2_scores: Dict[str, Dict[str, Optional[float]]] = {}
    l1_scores: Dict[str, Optional[float]] = {}

    for l1, l2_dict in tree_values.items():
        l2_scores[l1] = {}
        for l2, values in l2_dict.items():
            l2_scores[l1][l2] = mean_valid(values)
        l1_scores[l1] = mean_valid(l2_scores[l1].values())

    weighted_sum = 0.0
    weight_sum = 0.0

    for l1, score in l1_scores.items():
        if score is None:
            continue
        weight = rubric.l1_weights.get(l1)
        if weight is None:
            raise ValueError(f"Missing weight for active L1 {l1!r}")
        weighted_sum += weight * score
        weight_sum += weight

    reward_raw_bottom_up = weighted_sum / weight_sum if weight_sum > 0 else 0.0

    return {
        "reward": float(reward_raw_bottom_up),
        "reward_raw_bottom_up": float(reward_raw_bottom_up),
        "l1_scores": l1_scores,
        "l2_scores": l2_scores,
        "l3_scores": l3_scores,
        "l3_raw_labels": l3_raw_labels,
        "l3_evidence": l3_evidence,
        "failure_tags": judge_output.get("failure_tags", []),
        "raw_judge": judge_output,
    }


# ============================================================
# 7. Output helpers
# ============================================================

def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")



def write_csv(path: Path, rows: List[Dict[str, Any]], rubric: RubricSpec) -> None:
    active_l1 = active_l1_names(rubric.taxonomy, rubric.active_l3)

    fieldnames = [
        "rank", "sample_index", "sample_path", "reason", "reason_details",
        "reward", "reward_raw_bottom_up", "reward_raw_discrete", "reward_for_training",
        "reward_before_unique_adjustment", "reward_unique_adjustment", "reward_unique_rank",
        "reward_tiebreaker_signal", "reward_tiebreaker_delta", "reward_tiebreaker_hash",
        "reward_before_soft_penalty", "reward_after_soft_penalty", "soft_penalty_total",
        "metric_color_penalty", "metric_color_quality_score", "metric_color_penalty_reasons",
        "edit_group_score", "preservation_group_score", "color_detail_group_score",
        "quality_group_score", "essential_bottleneck", "gate_factor", "severe_multiplier",
        "failure_tags",
    ] + active_l1

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for rank, row in enumerate(rows, start=1):
            l1 = row.get("l1_scores", {})
            base = {
                "rank": rank,
                "sample_index": row["sample_index"],
                "sample_path": row["sample_path"],
                "reason": row.get("reason"),
                "reason_details": json.dumps(row.get("reason_details") or {}, ensure_ascii=False),
                "reward": row.get("reward"),
                "reward_raw_bottom_up": row.get("reward_raw_bottom_up"),
                "reward_raw_discrete": row.get("reward_raw_discrete"),
                "reward_for_training": row.get("reward_for_training"),
                "reward_before_unique_adjustment": row.get("reward_before_unique_adjustment"),
                "reward_unique_adjustment": row.get("reward_unique_adjustment"),
                "reward_unique_rank": row.get("reward_unique_rank"),
                "reward_tiebreaker_signal": row.get("reward_tiebreaker_signal"),
                "reward_tiebreaker_delta": row.get("reward_tiebreaker_delta"),
                "reward_tiebreaker_hash": row.get("reward_tiebreaker_hash"),
                "reward_before_soft_penalty": row.get("reward_before_soft_penalty"),
                "reward_after_soft_penalty": row.get("reward_after_soft_penalty"),
                "soft_penalty_total": row.get("soft_penalty_total"),
                "metric_color_penalty": row.get("metric_color_penalty"),
                "metric_color_quality_score": row.get("metric_color_quality_score"),
                "metric_color_penalty_reasons": "|".join(row.get("metric_color_penalty_reasons", [])),
                "edit_group_score": row.get("edit_group_score"),
                "preservation_group_score": row.get("preservation_group_score"),
                "color_detail_group_score": row.get("color_detail_group_score"),
                "quality_group_score": row.get("quality_group_score"),
                "essential_bottleneck": row.get("essential_bottleneck"),
                "gate_factor": row.get("gate_factor"),
                "severe_multiplier": row.get("severe_multiplier"),
                "failure_tags": "|".join(row.get("failure_tags", [])),
            }
            for l1_name in active_l1:
                base[l1_name] = l1.get(l1_name)
            writer.writerow(base)


def _safe_slug(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "_-" else "_" for ch in value).strip("_") or "rubric"


def _load_grid_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def _text_width(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> int:
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0]


def _wrap_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_width: int, max_lines: int) -> List[str]:
    words = str(text).replace("\n", " ").split()
    lines: List[str] = []
    current = ""
    for word in words:
        trial = word if not current else f"{current} {word}"
        if _text_width(draw, trial, font) <= max_width:
            current = trial
            continue
        if current:
            lines.append(current)
            current = word
        else:
            clipped = word
            while clipped and _text_width(draw, clipped + "...", font) > max_width:
                clipped = clipped[:-1]
            lines.append((clipped + "...") if clipped else "...")
            current = ""
        if len(lines) >= max_lines:
            break
    if current and len(lines) < max_lines:
        lines.append(current)
    if words and len(lines) == max_lines:
        full = " ".join(words)
        shown = " ".join(lines)
        if len(shown) < len(full) and not lines[-1].endswith("..."):
            while lines[-1] and _text_width(draw, lines[-1] + "...", font) > max_width:
                lines[-1] = lines[-1][:-1]
            lines[-1] += "..."
    return lines[:max_lines]


def _float_or_none(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        value = float(value)
        if math.isnan(value):
            return None
        return value
    except Exception:
        return None


def save_scored_compare_grid(
    source_path: str,
    rows: List[Dict[str, Any]],
    rubric: RubricSpec,
    prompt: str,
    output_dir: Path,
    sample_dir: Optional[str],
) -> Tuple[Path, Path, Path]:
    ordered = sorted(rows, key=lambda row: int(row.get("sample_index", 0)))
    ranked = sorted(ordered, key=lambda row: float(row.get("reward", 0.0)), reverse=True)
    rank_by_index = {int(row.get("sample_index", 0)): rank for rank, row in enumerate(ranked, start=1)}

    slug = _safe_slug(rubric.version)
    grid_dir = output_dir
    grid_dir.mkdir(parents=True, exist_ok=True)
    grid_path = grid_dir / f"scored_compare_grid_{slug}.jpg"
    grid_json_path = grid_dir / f"scored_compare_grid_{slug}_scores.json"
    grid_csv_path = grid_dir / f"scored_compare_grid_{slug}_scores.csv"

    tile_w = 420
    image_h = 420
    label_h = 220
    pad = 14
    cols = 3
    cell_h = image_h + label_h
    total_items = len(ordered) + 1
    grid_rows = math.ceil(total_items / cols)
    canvas = Image.new("RGB", (cols * tile_w, grid_rows * cell_h), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = _load_grid_font(20, bold=True)
    small_font = _load_grid_font(13, bold=False)

    def paste_panel(index: int, image_path: str, title: str, lines: List[str], border: Tuple[int, int, int]) -> None:
        col = index % cols
        row = index // cols
        x = col * tile_w
        y = row * cell_h
        draw.rectangle((x, y, x + tile_w - 1, y + cell_h - 1), outline=(220, 220, 220), width=1)
        try:
            img = Image.open(image_path).convert("RGB")
            img = ImageOps.contain(img, (tile_w - 2 * pad, image_h - 2 * pad), method=Image.Resampling.LANCZOS)
            px = x + (tile_w - img.width) // 2
            py = y + (image_h - img.height) // 2
            canvas.paste(img, (px, py))
        except Exception as err:
            draw.text((x + pad, y + pad), f"image load error: {err}", fill=(180, 0, 0), font=small_font)
        draw.rectangle((x + 4, y + 4, x + tile_w - 5, y + image_h - 5), outline=border, width=4)
        ty = y + image_h + 8
        draw.text((x + pad, ty), title, fill=(0, 0, 0), font=title_font)
        ty += 26
        for line in lines:
            for wrapped in _wrap_text(draw, line, small_font, tile_w - 2 * pad, 2):
                draw.text((x + pad, ty), wrapped, fill=(20, 20, 20), font=small_font)
                ty += 18
                if ty > y + cell_h - 18:
                    return

    paste_panel(0, source_path, "SOURCE", [f"prompt: {prompt}"], (70, 110, 210))
    summary_rows: List[Dict[str, Any]] = []
    for idx, row in enumerate(ordered, start=1):
        reward = _float_or_none(row.get("reward"))
        raw_discrete = _float_or_none(row.get("reward_raw_discrete"))
        training_reward = _float_or_none(row.get("reward_for_training"))
        before = _float_or_none(row.get("reward_before_soft_penalty"))
        after = _float_or_none(row.get("reward_after_soft_penalty_before_floor"))
        floor = _float_or_none(row.get("prompt_correctness_floor"))
        penalty = _float_or_none(row.get("soft_penalty_total"))
        pillar_color = _float_or_none(row.get("three_pillar_color"))
        pillar_preserve = _float_or_none(row.get("three_pillar_preservation"))
        pillar_edit = _float_or_none(row.get("three_pillar_edit"))
        gate_factor = _float_or_none(row.get("gate_factor"))
        metric_quality = _float_or_none(row.get("metric_color_quality_score"))
        l1 = row.get("l1_scores") or {}
        color = _float_or_none(_task_color_l1_score(l1))
        prompt_score = _float_or_none(_task_prompt_l1_score(l1))
        sample_index = int(row.get("sample_index", idx))
        rank = rank_by_index.get(sample_index)
        floor_applied = bool(row.get("reward_floor_applied"))
        penalties = row.get("soft_penalties") or []
        penalty_reasons = ", ".join(str(item.get("reason", "")) for item in penalties[:3] if item.get("reason"))
        reason = str(row.get("reason", "")).strip()
        border = (40, 150, 75) if rank == 1 else (210, 80, 65) if rank == len(ordered) else (150, 150, 150)
        title = f"C{sample_index}  rank={rank}  reward={(reward if reward is not None else 0.0):.3f}"
        lines = [
            f"train_reward={(training_reward if training_reward is not None else 0.0):.6f}  raw={(raw_discrete if raw_discrete is not None else 0.0):.3f}",
            f"color={(pillar_color if pillar_color is not None else 0.0):.3f}  preserve={(pillar_preserve if pillar_preserve is not None else 0.0):.3f}  edit={(pillar_edit if pillar_edit is not None else 0.0):.3f}",
            f"gate={(gate_factor if gate_factor is not None else 0.0):.3f}  metric_quality={(metric_quality if metric_quality is not None else 0.0):.3f}",
            f"before={(before if before is not None else 0.0):.3f}  after_penalty={(after if after is not None else 0.0):.3f}",
            f"soft_penalty={(penalty if penalty is not None else 0.0):.3f}  floor={(floor if floor is not None else 0.0):.3f}  floor_applied={floor_applied}",
        ]
        if reason:
            lines.append(f"reason: {reason}")
        if penalty_reasons:
            lines.append(f"penalty: {penalty_reasons}")
        if row.get("error"):
            lines.append(f"error: {row.get('error')}")
        paste_panel(idx, str(row.get("sample_path", "")), title, lines, border)
        summary_rows.append({
            "sample_index": sample_index,
            "sample_path": row.get("sample_path"),
            "rank": rank,
            "reason": row.get("reason"),
            "reason_details": row.get("reason_details"),
            "reward": reward,
            "reward_raw_discrete": raw_discrete,
            "reward_for_training": training_reward,
            "reward_before_unique_adjustment": row.get("reward_before_unique_adjustment"),
            "reward_unique_adjustment": row.get("reward_unique_adjustment"),
            "reward_unique_rank": row.get("reward_unique_rank"),
            "reward_tiebreaker_signal": row.get("reward_tiebreaker_signal"),
            "reward_tiebreaker_delta": row.get("reward_tiebreaker_delta"),
            "reward_tiebreaker_hash": row.get("reward_tiebreaker_hash"),
            "reward_before_soft_penalty": before,
            "reward_after_soft_penalty_before_floor": after,
            "prompt_correctness_floor": floor,
            "reward_floor_applied": floor_applied,
            "soft_penalty_total": penalty,
            "three_pillar_color": pillar_color,
            "three_pillar_preservation": pillar_preserve,
            "three_pillar_edit": pillar_edit,
            "gate_factor": gate_factor,
            "metric_color_quality_score": metric_quality,
            "metric_color_penalty_reasons": row.get("metric_color_penalty_reasons", []),
            "color_exposure_image_consistency": color,
            "prompt_facial_hair_correctness": prompt_score,
            "prompt_edit_correctness": prompt_score,
            "soft_penalty_reasons": [item.get("reason") for item in penalties if item.get("reason")],
            "error": row.get("error"),
        })

    canvas.save(grid_path, quality=95)
    with open(grid_json_path, "w", encoding="utf-8") as f:
        json.dump({"rubric_version": rubric.version, "prompt": prompt, "scores": summary_rows}, f, ensure_ascii=False, indent=2)
    with open(grid_csv_path, "w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "sample_index", "rank", "reward", "reward_raw_discrete", "reward_for_training",
            "reward_before_unique_adjustment", "reward_unique_adjustment", "reward_unique_rank",
            "reward_tiebreaker_signal", "reward_tiebreaker_delta", "reward_tiebreaker_hash",
            "reason", "reason_details", "reward_before_soft_penalty",
            "reward_after_soft_penalty_before_floor", "prompt_correctness_floor",
            "reward_floor_applied", "soft_penalty_total",
            "three_pillar_color", "three_pillar_preservation", "three_pillar_edit", "gate_factor",
            "metric_color_quality_score", "metric_color_penalty_reasons",
            "color_exposure_image_consistency", "prompt_facial_hair_correctness", "prompt_edit_correctness",
            "soft_penalty_reasons", "sample_path", "error",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for item in summary_rows:
            out = dict(item)
            out["reason_details"] = json.dumps(out.get("reason_details") or {}, ensure_ascii=False)
            out["soft_penalty_reasons"] = ";".join(out.get("soft_penalty_reasons") or [])
            out["metric_color_penalty_reasons"] = ";".join(out.get("metric_color_penalty_reasons") or [])
            writer.writerow(out)
    return grid_path, grid_json_path, grid_csv_path


def discover_group_dirs(root: str, group_filter: str = "", recursive: bool = False) -> List[Path]:
    root_path = Path(root)
    if not root_path.exists():
        raise FileNotFoundError(f"Batch root not found: {root}")
    pattern = "**/group_*" if recursive else "group_*"
    tokens = [item.strip().lower() for item in str(group_filter or "").split(",") if item.strip()]
    groups: List[Path] = []
    for path in sorted(root_path.glob(pattern)):
        if not path.is_dir():
            continue
        name = path.name.lower()
        if tokens and not any(token in name for token in tokens):
            continue
        if not (path / "source.jpg").exists() or not (path / "prompt.txt").exists():
            continue
        if not list(path.glob("candidate_*.jpg")):
            continue
        groups.append(path)
    return groups


def validate_group_dir(group_path: str) -> Path:
    path = Path(str(group_path).strip()).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Group path not found: {group_path}")
    if not path.is_dir():
        raise NotADirectoryError(f"Group path is not a directory: {group_path}")
    missing: List[str] = []
    if not (path / "source.jpg").exists():
        missing.append("source.jpg")
    if not (path / "prompt.txt").exists():
        missing.append("prompt.txt")
    if not list(path.glob("candidate_*.jpg")):
        missing.append("candidate_*.jpg")
    if missing:
        raise RuntimeError(f"Invalid group path {path}: missing {', '.join(missing)}")
    return path


def output_dir_for_group(output_root: str, group_dir: Path) -> str:
    return str(Path(output_root) / group_dir.name)


def _rubric_match_candidates(data_dir: Path) -> Tuple[List[str], List[str]]:
    names: List[str] = []
    prefixes: List[str] = []
    current = data_dir
    for _ in range(4):
        name = current.name.strip()
        if name:
            names.append(name)
            parts = name.split("_")
            for part in parts:
                if part.isdigit() and len(part) == 2:
                    prefixes.append(part)
        if current.parent == current:
            break
        current = current.parent
    return dedupe_preserve_order(names), dedupe_preserve_order(prefixes)


def resolve_rubric_yaml_path(rubric_yaml: str, data_dir: str) -> str:
    """Resolve a concrete rubric file.

    --rubric_yaml can be either a YAML file or a directory. When it is a
    directory, match the current group path to a YAML file. For the GRPO-style
    layout, /.../13_男性粗大臂/image_021205 matches 13_男性粗大臂.yaml.
    For older group names, a unique numeric prefix match such as 09_*.yaml is
    also accepted.
    """
    rubric_path = Path(str(rubric_yaml).strip()).expanduser()
    if rubric_path.is_file():
        return str(rubric_path)
    if not rubric_path.exists():
        raise FileNotFoundError(f"Rubric path not found: {rubric_yaml}")
    if not rubric_path.is_dir():
        raise ValueError(f"Rubric path must be a .yaml file or a directory: {rubric_yaml}")

    group_dir = Path(data_dir)
    names, prefixes = _rubric_match_candidates(group_dir)

    for name in names:
        direct = rubric_path / f"{name}.yaml"
        if direct.exists():
            return str(direct)

    yaml_files = sorted(rubric_path.glob("*.yaml"))
    for prefix in prefixes:
        matches = [path for path in yaml_files if path.stem.startswith(prefix + "_")]
        if len(matches) == 1:
            return str(matches[0])
        if len(matches) > 1:
            raise RuntimeError(
                f"Multiple rubric YAML files match prefix {prefix!r} under {rubric_path}: "
                + ", ".join(path.name for path in matches)
            )

    raise FileNotFoundError(
        f"No matching rubric YAML found under {rubric_path} for data_dir={group_dir}. "
        f"Tried exact names={names} and numeric prefixes={prefixes}."
    )


def run_one_scoring(args: argparse.Namespace) -> Dict[str, Any]:
    args = resolve_inputs(args)
    args.base_url = normalize_openai_compatible_base_url(args.base_url)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    args.rubric_yaml = resolve_rubric_yaml_path(args.rubric_yaml, args.data_dir)
    rubric = load_rubric(args.rubric_yaml)
    sample_paths = collect_sample_paths(args)

    rubric_text = build_rubric_text(rubric)
    rubric_text_sha256 = hashlib.sha256(rubric_text.encode("utf-8", errors="replace")).hexdigest()

    rubric_used_path = output_dir / "rubric_used.json"
    with open(rubric_used_path, "w", encoding="utf-8") as f:
        json.dump(rubric_to_jsonable(rubric), f, ensure_ascii=False, indent=2)

    checklist_path = output_dir / "rubric_checklist.txt"
    with open(checklist_path, "w", encoding="utf-8") as f:
        f.write(rubric_text + "\n")

    print("========== YAML Bottom-up Rubric Test ==========")
    print(f"source           : {args.source}")
    print(f"prompt           : {args.prompt}")
    print(f"requirement      : {args.requirement}")
    print(f"rubric_yaml      : {args.rubric_yaml}")
    print(f"rubric_task_key  : {rubric.task_key}")
    print(f"rubric_version   : {rubric.version}")
    print(f"rubric_sha256    : {rubric_text_sha256}")
    print(f"active_l3        : {len(rubric.active_l3)}")
    print(f"model            : {args.model}")
    print(f"base_url         : {args.base_url or '<default openai>'}")
    print(f"num samples      : {len(sample_paths)}")
    print(f"workers          : {max(1, args.workers)}")
    print(f"output_dir       : {output_dir}")
    print(f"rubric_used      : {rubric_used_path}")
    print(f"rubric_checklist : {checklist_path}")
    print("================================================")

    if args.dry_run:
        print("\nDry run enabled. No GPT calls were made.")
        print("\nActive L3 facets:")
        for idx, facet_id in enumerate(rubric.active_l3, start=1):
            print(f"{idx:02d}. {facet_id}")
        return {"status": "dry_run", "output_dir": str(output_dir), "valid_scores": 0, "failed_scores": 0, "scores": []}

    client_kwargs: Dict[str, Any] = {}
    if args.api_key:
        client_kwargs["api_key"] = args.api_key
    if args.base_url:
        client_kwargs["base_url"] = args.base_url
    if args.request_timeout and args.request_timeout > 0:
        client_kwargs["timeout"] = args.request_timeout

    if (
        not args.api_key
        and not os.getenv("OPENAI_API_KEY")
        and not os.getenv("GEMINI_API_KEY")
        and not os.getenv("GOOGLE_API_KEY")
        and not os.getenv("OPENAI_ADMIN_KEY")
    ):
        raise RuntimeError(
            "Missing API key. Please export OPENAI_API_KEY/GEMINI_API_KEY or pass --api_key. "
            "If you use a relay, also export GPT_REWARD_BASE_URL or pass --base_url."
        )

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ImportError("Missing dependency: openai. Install with `pip install openai`.") from exc

    rows: List[Dict[str, Any]] = []
    valid_count = 0
    failed_count = 0
    source_hash = file_sha256(args.source)

    def score_one_sample(i: int, sample_path: str) -> Dict[str, Any]:
        metadata = {
            "rubric_task_key": rubric.task_key,
            "rubric_version": rubric.version,
            "rubric_yaml": args.rubric_yaml,
            "rubric_text_sha256": rubric_text_sha256,
            "rubric_content_sent_to_gemini": True,
            "model": args.model,
            "scoring_logic": "generic_ratio_soft_gate_v2",
            "requirement": args.requirement,
            "sample_index": i,
            "sample_path": sample_path,
            "source_sha256": source_hash,
            "edited_sha256": file_sha256(sample_path),
        }
        try:
            client = OpenAI(**client_kwargs)
            judge_output = call_gpt_l3_judge(
                client=client,
                model=args.model,
                source_image_path=args.source,
                edited_image_path=sample_path,
                prompt=args.prompt,
                rubric=rubric,
                requirement=args.requirement,
                max_retries=args.max_retries,
                max_image_side=args.max_image_side,
                jpeg_quality=args.jpeg_quality,
            )
            result = aggregate_bottom_up(rubric=rubric, judge_output=judge_output)
            consistency_metrics = image_consistency_metrics(args.source, sample_path)
            result["image_consistency_metrics"] = consistency_metrics
            result["image_consistency_caps"] = []
            result = apply_soft_preservation_penalties(result, consistency_metrics, rubric)
            result = apply_training_reward_calibration(result, sample_path=sample_path, sample_index=i)
            reason, reason_details = build_candidate_reason(result, rubric)
            result["reason"] = reason
            result["reason_details"] = reason_details
            return {
                "sample_index": i,
                "sample_path": sample_path,
                "source": args.source,
                "prompt": args.prompt,
                "metadata": metadata,
                "error": None,
                **result,
            }
        except Exception as err:
            return {
                "sample_index": i,
                "sample_path": sample_path,
                "source": args.source,
                "prompt": args.prompt,
                "metadata": metadata,
                "reward": 0.001 + i * 0.0001,
                "reward_raw_discrete": 0.0,
                "reward_for_training": 0.001 + i * 0.0001,
                "reward_raw_bottom_up": 0.0,
                "reason": f"Scoring failed: {err!r}",
                "reason_details": {"error": repr(err), "score_is_placeholder": True},
                "error": repr(err),
                "score_is_placeholder": True,
            }

    worker_count = max(1, min(int(args.workers or 1), len(sample_paths)))
    print(f"\nScoring {len(sample_paths)} sample(s) with {worker_count} parallel worker(s).")
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        future_to_sample = {
            executor.submit(score_one_sample, i, sample_path): (i, sample_path)
            for i, sample_path in enumerate(sample_paths, start=1)
        }
        for completed, future in enumerate(as_completed(future_to_sample), start=1):
            i, _sample_path = future_to_sample[future]
            row = future.result()
            rows.append(row)
            if row.get("error"):
                failed_count += 1
                print(f"\n[{completed}/{len(sample_paths)}] sample={i} ERROR: {row.get('error')}")
            else:
                valid_count += 1
                print(
                    f"\n[{completed}/{len(sample_paths)}] sample={i} "
                    f"reward={float(row.get('reward', 0.0)):.4f} "
                    f"raw={float(row.get('reward_raw_bottom_up', 0.0)):.4f} "
                    f"soft_penalty={float(row.get('soft_penalty_total', 0.0)):.4f}"
                )
                print("Reason:", row.get("reason"))
                print("L1:", json.dumps(row.get("l1_scores", {}), ensure_ascii=False))

    enforce_unique_training_rewards(rows)
    rows = sorted(rows, key=lambda x: int(x.get("sample_index", 0)))
    scores = [float(row.get("reward", 0.0)) for row in rows]
    ranked = sorted(rows, key=lambda x: float(x.get("reward", 0.0)), reverse=True)

    jsonl_path = output_dir / "scored_samples.jsonl"
    ranked_json_path = output_dir / "ranked_samples.json"
    csv_path = output_dir / "ranked_samples.csv"
    scores_path = output_dir / "scores_only.json"

    write_jsonl(jsonl_path, rows)
    write_csv(csv_path, ranked, rubric)
    with open(ranked_json_path, "w", encoding="utf-8") as f:
        json.dump(ranked, f, ensure_ascii=False, indent=2)
    with open(scores_path, "w", encoding="utf-8") as f:
        json.dump({"scores": scores}, f, ensure_ascii=False, indent=2)

    grid_path, grid_json_path, grid_csv_path = save_scored_compare_grid(
        source_path=args.source,
        rows=rows,
        rubric=rubric,
        prompt=args.prompt,
        output_dir=output_dir,
        sample_dir=args.sample_dir,
    )

    print("\n========== Final scores: List[float] ==========")
    print(json.dumps(scores, ensure_ascii=False, indent=2))
    print("\n========== Ranking ==========")
    for rank, row in enumerate(ranked, start=1):
        print(
            f"rank={rank} sample={row['sample_index']} "
            f"reward={float(row.get('reward', 0.0)):.4f} path={row['sample_path']}"
        )
    print("\nSaved:")
    for path in [jsonl_path, ranked_json_path, csv_path, scores_path, grid_path, grid_json_path, grid_csv_path, rubric_used_path, checklist_path]:
        print(f"- {path}")
    print("\n========== Scoring status ==========")
    print(f"valid_scores  : {valid_count}")
    print(f"failed_scores : {failed_count}")
    if failed_count:
        print("Failed samples keep a tiny nonzero placeholder reward; inspect the per-sample error field before using them for training.")
    if valid_count == 0 and failed_count > 0:
        raise RuntimeError(
            "All samples failed scoring. The rewards in output files are placeholders, "
            "most likely caused by an API/key/base_url/model problem rather than Gemini judging every image as zero."
        )
    return {
        "status": "ok" if failed_count == 0 else "partial",
        "output_dir": str(output_dir),
        "valid_scores": valid_count,
        "failed_scores": failed_count,
        "scores": scores,
        "grid": str(grid_path),
    }


# ============================================================
# 8. Main
# ============================================================

def main() -> None:
    load_default_judge_env()
    parser = argparse.ArgumentParser()

    # Single-group inputs.
    parser.add_argument("--data_dir", default="/nvmedata/workspace2/users/wzt/data", help="Folder containing source.jpg, prompt.txt, and candidate images.")
    parser.add_argument("--source", default="", help="Source image path. Defaults to <data_dir>/source.jpg.")
    parser.add_argument("--prompt", default="", help="Edit prompt. Defaults to the prompt field in <data_dir>/prompt.txt.")
    parser.add_argument("--prompt_file", default="", help="Prompt metadata file. Defaults to <data_dir>/prompt.txt.")
    parser.add_argument("--requirement", default="", help="Additional requirement. Defaults to the requirement field in prompt.txt.")
    parser.add_argument("--samples", nargs="*", default=None, help="Edited sample image paths.")
    parser.add_argument("--sample_dir", default=None, help="Directory containing edited samples. Defaults to <data_dir>.")
    parser.add_argument("--candidate_prefix", default="candidate_", help="When using --sample_dir, only load files with this prefix. Use empty string to load all images except source.")
    parser.add_argument("--expected_samples", type=int, default=int(os.getenv("EXPECTED_SAMPLES", "0")), help="Expected sample count. Use 0 to disable count check for 8/16/32 variable batches.")

    # Batch mode over child group folders.
    parser.add_argument("--group_path", default="", help="Optional exact group folder path. When set, only this group is evaluated and batch discovery is skipped.")
    parser.add_argument("--batch_root", default="", help="Parent folder containing group_* subfolders. When set, test.py runs each matched group in order.")
    parser.add_argument("--group_filter", default="", help="Comma-separated substrings for group folder names, e.g. beard, hair, 02_beard. Empty means all group_* folders.")
    parser.add_argument("--recursive_groups", action="store_true", help="Search group_* folders recursively under --batch_root.")
    parser.add_argument("--output_root", default="", help="Batch output root. Defaults to <batch_root>/outputs.")
    parser.add_argument("--skip_existing", action="store_true", help="Skip groups whose scored_samples.jsonl already has at least expected/nonzero valid rows.")
    parser.add_argument("--limit_groups", type=int, default=0, help="Optional maximum number of matched groups to run.")

    # YAML rubric.
    parser.add_argument("--rubric_yaml", default="/nvmedata/workspace2/users/wzt/hair_edit.yaml", help="Path to a YAML rubric file or a directory containing task-matched YAML files.")
    parser.add_argument("--dry_run", action="store_true", help="Validate inputs/YAML and write rubric_used.json without calling GPT.")

    # Output.
    parser.add_argument("--output_dir", default="/nvmedata/workspace2/users/wzt/data/rubric_test_out")

    # OpenAI-compatible API.
    parser.add_argument("--model", default=os.getenv("GPT_RUBRIC_MODEL", os.getenv("GPT_REWARD_MODEL", os.getenv("GEMINI_REWARD_MODEL", "gemini-3.1-flash-lite"))))
    parser.add_argument(
        "--api_key",
        default=os.getenv(
            "OPENAI_API_KEY",
            os.getenv("GEMINI_API_KEY", os.getenv("GOOGLE_API_KEY", os.getenv("GPT_REWARD_API_KEY", ""))),
        ),
    )
    parser.add_argument("--base_url", default=os.getenv("OPENAI_BASE_URL", os.getenv("GPT_REWARD_BASE_URL", os.getenv("GEMINI_OPENAI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/"))))
    parser.add_argument("--request_timeout", type=float, default=float(os.getenv("GPT_RUBRIC_TIMEOUT", "90")), help="OpenAI-compatible request timeout in seconds.")
    parser.add_argument("--max_retries", type=int, default=3)
    parser.add_argument("--max_image_side", type=int, default=1024)
    parser.add_argument("--jpeg_quality", type=int, default=90)
    parser.add_argument("--workers", type=int, default=int(os.getenv("GEMINI_SCORE_WORKERS", "4")), help="Parallel Gemini/API scoring workers within one group.")

    args = parser.parse_args()

    if str(args.group_path or "").strip() or args.batch_root:
        if str(args.group_path or "").strip():
            group_dir = validate_group_dir(args.group_path)
            groups = [group_dir]
            output_root = args.output_root or str(group_dir.parent / "outputs")
            discovery_mode = "single_group_path"
        else:
            groups = discover_group_dirs(args.batch_root, args.group_filter, args.recursive_groups)
            if args.limit_groups > 0:
                groups = groups[:args.limit_groups]
            if not groups:
                raise RuntimeError(f"No group folders matched batch_root={args.batch_root!r}, group_filter={args.group_filter!r}")
            output_root = args.output_root or str(Path(args.batch_root) / "outputs")
            discovery_mode = "batch_root_sorted"
        Path(output_root).mkdir(parents=True, exist_ok=True)
        print("========== Batch Rubric Test ==========")
        print(f"mode          : {discovery_mode}")
        print(f"group_path    : {args.group_path or '<empty>'}")
        print(f"batch_root    : {args.batch_root or '<empty>'}")
        print(f"group_filter  : {args.group_filter or '<all>'}")
        print(f"matched_groups: {len(groups)}")
        print(f"output_root   : {output_root}")
        print("=======================================")
        summary: List[Dict[str, Any]] = []
        for index, group_dir in enumerate(groups, start=1):
            group_args = argparse.Namespace(**vars(args))
            group_args.batch_root = ""
            group_args.data_dir = str(group_dir)
            group_args.source = ""
            group_args.prompt = ""
            group_args.prompt_file = ""
            group_args.requirement = ""
            group_args.samples = None
            group_args.sample_dir = None
            group_args.output_dir = output_dir_for_group(output_root, group_dir)
            expected_valid = int(group_args.expected_samples or 0) or len(list(group_dir.glob("candidate_*.jpg")))
            existing_jsonl = Path(group_args.output_dir) / "scored_samples.jsonl"
            if args.skip_existing and existing_jsonl.exists():
                valid = 0
                compatible = 0
                try:
                    expected_rubric_path = resolve_rubric_yaml_path(group_args.rubric_yaml, str(group_dir))
                    expected_rubric = load_rubric(expected_rubric_path)
                    expected_rubric_text = build_rubric_text(expected_rubric)
                    expected_rubric_sha = hashlib.sha256(expected_rubric_text.encode("utf-8", errors="replace")).hexdigest()
                    for line in existing_jsonl.read_text(encoding="utf-8", errors="ignore").splitlines():
                        if not line.strip():
                            continue
                        existing_row = json.loads(line)
                        if existing_row.get("error"):
                            continue
                        valid += 1
                        metadata = existing_row.get("metadata") or {}
                        if (
                            metadata.get("rubric_text_sha256") == expected_rubric_sha
                            and metadata.get("model") == args.model
                            and metadata.get("scoring_logic") == "generic_ratio_soft_gate_v2"
                        ):
                            compatible += 1
                except Exception:
                    valid = 0
                    compatible = 0
                if valid >= expected_valid and compatible >= expected_valid:
                    print(f"[{index}/{len(groups)}] SKIP {group_dir.name} valid={valid} compatible={compatible}")
                    summary.append({"group": group_dir.name, "status": "skipped", "valid_scores": valid, "compatible_scores": compatible, "output_dir": group_args.output_dir})
                    continue
            print(f"\n[{index}/{len(groups)}] RUN {group_dir.name}")
            try:
                result = run_one_scoring(group_args)
                result["group"] = group_dir.name
                summary.append(result)
            except Exception as exc:
                print(f"[{index}/{len(groups)}] GROUP_ERROR {group_dir.name}: {exc!r}")
                summary.append({"group": group_dir.name, "status": "error", "error": repr(exc), "output_dir": group_args.output_dir})
        summary_path = Path(output_root) / "batch_summary.json"
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print("\n========== Batch Done ==========")
        print(f"summary: {summary_path}")
        print(f"ok_or_partial: {sum(1 for item in summary if item.get('status') in {'ok', 'partial', 'skipped'})}/{len(summary)}")
        return

    run_one_scoring(args)


if __name__ == "__main__":
    main()
