#!/usr/bin/env python3
from __future__ import annotations

import base64
import io
import json
import os
import re
import time
import types
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml
from PIL import Image


L1_SCORE_ORDER = [
    "Prompt Compliance",
    "Localization and Anatomical Integration",
    "Subject Preservation",
    "Task Realism and Image Quality",
]

FAILURE_ORDER = [
    "target_absent_or_wrong",
    "wrong_region_or_over_editing",
    "identity_or_non_target_drift",
    "artifact_or_quality_failure",
]


class L1Dimension:
    def __init__(
        self,
        name: str,
        weight: float,
        definition: str,
        score_guidance: Dict[str, str],
    ) -> None:
        self.name = name
        self.weight = weight
        self.definition = definition
        self.score_guidance = score_guidance


class L1Rubric:
    def __init__(
        self,
        task_key: str,
        version: str,
        judge_role: str,
        judge_instructions: str,
        dimensions: Dict[str, L1Dimension],
        score_order: List[str],
        failure_policy: Dict[str, Any],
        source_path: str,
        raw: Dict[str, Any],
    ) -> None:
        self.task_key = task_key
        self.version = version
        self.judge_role = judge_role
        self.judge_instructions = judge_instructions
        self.dimensions = dimensions
        self.score_order = score_order
        self.failure_policy = failure_policy
        self.source_path = source_path
        self.raw = raw


def load_default_judge_env(path: str = "/home/student/.config/monet/judge.env") -> None:
    env_path = Path(path)
    if not env_path.is_file():
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


def load_rubric(path: str) -> L1Rubric:
    rubric_path = Path(path).expanduser().resolve()
    raw = yaml.safe_load(rubric_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"L1 rubric must be a YAML mapping: {rubric_path}")
    if raw.get("taxonomy") or raw.get("active_l3"):
        raise ValueError("L1-only training rubric must not contain taxonomy or active_l3")

    dimensions_raw = raw.get("l1_dimensions")
    if not isinstance(dimensions_raw, dict) or list(dimensions_raw) != L1_SCORE_ORDER:
        raise ValueError(f"l1_dimensions must contain exactly this order: {L1_SCORE_ORDER}")

    dimensions: Dict[str, L1Dimension] = {}
    for name in L1_SCORE_ORDER:
        spec = dimensions_raw.get(name)
        if not isinstance(spec, dict):
            raise ValueError(f"Missing L1 dimension definition: {name}")
        dimensions[name] = L1Dimension(
            name=name,
            weight=float(spec.get("weight", 0.0)),
            definition=str(spec.get("definition", "")).strip(),
            score_guidance={str(key): str(value) for key, value in (spec.get("score_guidance") or {}).items()},
        )
    weight_sum = sum(dimension.weight for dimension in dimensions.values())
    if abs(weight_sum - 1.0) > 1e-9:
        raise ValueError(f"L1 weights must sum to 1.0, got {weight_sum}")

    policy = raw.get("failure_aware_policy")
    if not isinstance(policy, dict):
        raise ValueError("L1-only rubric requires failure_aware_policy")
    failures = policy.get("failures") or {}
    if list(failures) != FAILURE_ORDER:
        raise ValueError(f"failure_aware_policy.failures must contain exactly this order: {FAILURE_ORDER}")

    return L1Rubric(
        task_key=str(raw.get("task_key", "unified_human_edit_l1")),
        version=str(raw.get("version", "")),
        judge_role=str(raw.get("judge_role", "")).strip(),
        judge_instructions=str(raw.get("judge_instructions", "")).strip(),
        dimensions=dimensions,
        score_order=list(L1_SCORE_ORDER),
        failure_policy=policy,
        source_path=str(rubric_path),
        raw=raw,
    )


def resolve_rubric_yaml_path(rubric_yaml: str, data_dir: str = "") -> str:
    path = Path(rubric_yaml).expanduser()
    if path.is_file():
        return str(path.resolve())
    raise FileNotFoundError(
        "L1-only training uses one unified YAML file; "
        f"expected a file, got {rubric_yaml!r} for data_dir={data_dir!r}"
    )


def _image_data_url(path: str, max_side: int, quality: int) -> str:
    with Image.open(path) as image:
        image = image.convert("RGB")
        if max(image.size) > max_side:
            scale = max_side / float(max(image.size))
            size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
            image = image.resize(size, Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=quality)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def _data_url_inline_part(data_url: str) -> Dict[str, Any]:
    header, encoded = data_url.split(",", 1)
    mime_type = header[len("data:") : header.index(";base64")]
    return {"inlineData": {"mimeType": mime_type, "data": encoded}}


def _response_schema() -> Dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "score_vector": {
                "type": "array",
                "minItems": 4,
                "maxItems": 4,
                "items": {"type": "integer", "minimum": 0, "maximum": 9},
            },
            "failure_severity": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    name: {"type": "integer", "minimum": 0, "maximum": 5}
                    for name in FAILURE_ORDER
                },
                "required": FAILURE_ORDER,
            },
        },
        "required": ["score_vector", "failure_severity"],
    }


def _prompt_text(prompt: str, requirement: str, rubric: L1Rubric) -> str:
    dimensions = []
    for index, name in enumerate(rubric.score_order):
        spec = rubric.dimensions[name]
        dimensions.append({
            "index": index,
            "name": name,
            "weight": spec.weight,
            "definition": spec.definition,
            "score_guidance": spec.score_guidance,
        })
    return f"""
# Edit instruction
{prompt}

# Additional preservation requirement
{requirement or "None"}

# Direct L1 scoring dimensions in required output order
{json.dumps(dimensions, ensure_ascii=False, indent=2)}

# Failure-aware policy
{json.dumps(rubric.failure_policy, ensure_ascii=False, indent=2)}

# Required output schema
{json.dumps(_response_schema(), ensure_ascii=False)}

Compare the source and edited images directly. Return exactly four integer L1
scores in score_vector order and one 0-5 severity for every failure type.
Do not produce L2 groups, L3 facets, facet ids, an overall score, explanations,
markdown, or any additional field. Return JSON only.
""".strip()


def _extract_text_from_gemini(response: Dict[str, Any]) -> str:
    texts = []
    for candidate in response.get("candidates", []) or []:
        for part in (candidate.get("content") or {}).get("parts", []) or []:
            if isinstance(part, dict) and part.get("text") is not None:
                texts.append(str(part["text"]))
    if not texts:
        raise RuntimeError("Gemini response contains no text")
    return "\n".join(texts).strip()


def _parse_json_text(text: str) -> Dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.I)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    vector_match = re.search(r'"?score_vector"?\s*:\s*\[([^\]]]+)\]', cleaned, re.S)
    if not vector_match:
        raise ValueError(f"Could not parse L1 score_vector from response: {text[:500]!r}")
    vector = [int(value) for value in re.findall(r"(?<!\d)([0-9])(?!\d)", vector_match.group(1))]
    severity = {}
    for name in FAILURE_ORDER:
        match = re.search(rf'"?{re.escape(name)}"?\s*:\s*([0-5])', cleaned)
        if not match:
            raise ValueError(f"Missing failure severity {name!r} in response")
        severity[name] = int(match.group(1))
    return {"score_vector": vector, "failure_severity": severity}


def _validate_output(output: Dict[str, Any]) -> Dict[str, Any]:
    vector = output.get("score_vector")
    if not isinstance(vector, list) or len(vector) != 4:
        raise ValueError(f"score_vector must contain exactly four integers, got {vector!r}")
    normalized_vector = []
    for index, value in enumerate(vector):
        if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 9:
            raise ValueError(f"score_vector[{index}] must be an integer from 0 to 9, got {value!r}")
        normalized_vector.append(value)

    severity = output.get("failure_severity")
    if not isinstance(severity, dict) or set(severity) != set(FAILURE_ORDER):
        raise ValueError(f"failure_severity must contain exactly {FAILURE_ORDER}")
    normalized_severity = {}
    for name in FAILURE_ORDER:
        value = severity[name]
        if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 5:
            raise ValueError(f"failure_severity[{name!r}] must be an integer from 0 to 5")
        normalized_severity[name] = value
    return {"score_vector": normalized_vector, "failure_severity": normalized_severity}


def _call_openai_compatible(
    client: Any,
    model: str,
    system_prompt: str,
    user_prompt: str,
    source_url: str,
    edited_url: str,
) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_prompt},
                {"type": "text", "text": "Source image before editing:"},
                {"type": "image_url", "image_url": {"url": source_url}},
                {"type": "text", "text": "Edited candidate image:"},
                {"type": "image_url", "image_url": {"url": edited_url}},
            ],
        },
    ]
    kwargs = {
        "model": model,
        "messages": messages,
        "temperature": 0,
        "response_format": {"type": "json_object"},
    }
    try:
        response = client.chat.completions.create(**kwargs)
    except Exception:
        kwargs.pop("response_format", None)
        response = client.chat.completions.create(**kwargs)
    return str(response.choices[0].message.content)


def _gemini_url(model: str, base_url: str) -> str:
    base = (base_url or "https://generativelanguage.googleapis.com/v1beta").rstrip("/")
    model_name = urllib.parse.quote(model, safe="")
    return f"{base}/models/{model_name}:generateContent"


def _call_native_gemini(
    api_key: str,
    base_url: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    source_url: str,
    edited_url: str,
    timeout: float,
) -> str:
    generation_config: Dict[str, Any] = {
        "temperature": 0,
        "candidateCount": 1,
        "maxOutputTokens": int(os.getenv("GEMINI_NATIVE_MAX_OUTPUT_TOKENS", "2048")),
        "responseMimeType": "application/json",
    }
    if os.getenv("L1_REWARD_USE_RESPONSE_SCHEMA", "0").lower() in {"1", "true", "yes"}:
        schema = _response_schema()
        schema["properties"]["score_vector"].pop("minItems", None)
        schema["properties"]["score_vector"].pop("maxItems", None)
        generation_config["responseJsonSchema"] = schema
    payload = {
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "contents": [{
            "role": "user",
            "parts": [
                {"text": user_prompt},
                {"text": "Source image before editing:"},
                _data_url_inline_part(source_url),
                {"text": "Edited candidate image:"},
                _data_url_inline_part(edited_url),
            ],
        }],
        "generationConfig": generation_config,
    }
    request = urllib.request.Request(
        _gemini_url(model, base_url),
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json", "x-goog-api-key": api_key},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = json.loads(response.read().decode("utf-8"))
    return _extract_text_from_gemini(body)


def call_gemini_l1_judge(
    *,
    client: Any = None,
    model: str,
    source_image_path: str,
    edited_image_path: str,
    prompt: str,
    rubric: L1Rubric,
    requirement: str = "",
    max_retries: int = 3,
    max_image_side: int = 1024,
    jpeg_quality: int = 90,
    api_key: str = "",
    base_url: str = "",
    request_timeout: Optional[float] = None,
) -> Dict[str, Any]:
    source_url = _image_data_url(source_image_path, max_image_side, jpeg_quality)
    edited_url = _image_data_url(edited_image_path, max_image_side, jpeg_quality)
    system_prompt = rubric.judge_role + " " + rubric.judge_instructions
    user_prompt = _prompt_text(prompt, requirement, rubric)
    timeout = float(request_timeout or os.getenv("SINGLE_REWARD_TIMEOUT", "120"))
    key = api_key or os.getenv("SINGLE_REWARD_API_KEY", os.getenv("GEMINI_API_KEY", ""))

    last_error: Optional[Exception] = None
    for attempt in range(max_retries + 1):
        try:
            if client is not None:
                response_text = _call_openai_compatible(
                    client, model, system_prompt, user_prompt, source_url, edited_url
                )
            else:
                if not key:
                    raise RuntimeError("Missing Gemini API key")
                response_text = _call_native_gemini(
                    key, base_url, model, system_prompt, user_prompt,
                    source_url, edited_url, timeout,
                )
            parsed = _validate_output(_parse_json_text(response_text))
            parsed["raw_model_response_text"] = response_text
            return parsed
        except Exception as error:
            last_error = error
            if attempt < max_retries:
                time.sleep(min(2 * (attempt + 1), 10))
    raise RuntimeError(f"L1-only judge failed after {max_retries + 1} attempts: {last_error!r}")


def aggregate_l1(rubric: L1Rubric, judge_output: Dict[str, Any]) -> Dict[str, Any]:
    raw_scores = {
        name: int(score)
        for name, score in zip(rubric.score_order, judge_output["score_vector"])
    }
    capped_scores = dict(raw_scores)
    applied_caps = []
    severity_map = judge_output["failure_severity"]
    cap_by_severity = {
        int(key): int(value)
        for key, value in (rubric.failure_policy.get("cap_by_severity") or {}).items()
    }
    failures = rubric.failure_policy.get("failures") or {}
    for failure_name in FAILURE_ORDER:
        severity = int(severity_map[failure_name])
        cap = cap_by_severity[severity]
        if severity <= 0:
            continue
        for dimension_name in failures[failure_name].get("affected_l1") or []:
            before = capped_scores[dimension_name]
            after = min(before, cap)
            capped_scores[dimension_name] = after
            if after != before:
                applied_caps.append({
                    "failure": failure_name,
                    "severity": severity,
                    "dimension": dimension_name,
                    "raw_score": before,
                    "cap": cap,
                    "final_score": after,
                })

    normalized = {name: score / 9.0 for name, score in capped_scores.items()}
    reward = sum(rubric.dimensions[name].weight * normalized[name] for name in rubric.score_order)
    audit_dimensions = {
        "target_edit_accuracy": normalized["Prompt Compliance"],
        "identity_preservation": normalized["Subject Preservation"],
        "non_target_preservation": normalized["Subject Preservation"],
        "color_lighting_texture_preservation": normalized["Subject Preservation"],
        "photorealism_artifact_control": normalized["Task Realism and Image Quality"],
    }
    return {
        "reward": float(reward),
        "reward_raw_bottom_up": float(reward),
        "reward_for_training": float(reward),
        "display_score_0_to_9": float(9.0 * reward),
        "l1_scores": normalized,
        "l1_raw_scores_0_to_9": raw_scores,
        "l1_capped_scores_0_to_9": capped_scores,
        "l1_weights": {name: rubric.dimensions[name].weight for name in rubric.score_order},
        "failure_severity": severity_map,
        "failure_caps_applied": applied_caps,
        "dimension_scores_0_1": audit_dimensions,
        "dimension_scores_raw_0_9": capped_scores,
    }


def image_consistency_metrics(source_path: str, edited_path: str) -> Dict[str, float]:
    with Image.open(source_path) as source_image, Image.open(edited_path) as edited_image:
        source = np.asarray(source_image.convert("RGB"), dtype=np.float32)
        edited = np.asarray(edited_image.convert("RGB").resize(source_image.size), dtype=np.float32)
    source_gray = source.mean(axis=2)
    edited_gray = edited.mean(axis=2)
    source_edges = np.abs(np.diff(source_gray, axis=0)).mean() + np.abs(np.diff(source_gray, axis=1)).mean()
    edited_edges = np.abs(np.diff(edited_gray, axis=0)).mean() + np.abs(np.diff(edited_gray, axis=1)).mean()
    return {
        "rgb_mean_l2": float(np.linalg.norm(source.mean(axis=(0, 1)) - edited.mean(axis=(0, 1)))),
        "brightness_delta": float(edited_gray.mean() - source_gray.mean()),
        "contrast_delta": float(edited_gray.std() - source_gray.std()),
        "saturation_delta": float((edited.max(axis=2) - edited.min(axis=2)).mean() - (source.max(axis=2) - source.min(axis=2)).mean()),
        "edge_sharpness_delta": float(edited_edges - source_edges),
    }


def apply_soft_preservation_penalties(
    result: Dict[str, Any], metrics: Dict[str, float], rubric: L1Rubric
) -> Dict[str, Any]:
    result = dict(result)
    result["reward_before_soft_penalty"] = result["reward"]
    result["reward_after_soft_penalty"] = result["reward"]
    result["soft_preservation_penalties"] = []
    return result


def apply_training_reward_calibration(
    result: Dict[str, Any], sample_path: str = "", sample_index: int = 0
) -> Dict[str, Any]:
    result = dict(result)
    result["reward_for_training"] = result["reward"]
    result["training_reward_calibration"] = "disabled_for_direct_l1_reward"
    return result


def build_candidate_reason(result: Dict[str, Any], rubric: L1Rubric) -> Tuple[str, Dict[str, Any]]:
    scores = result.get("l1_capped_scores_0_to_9") or {}
    weakest = sorted((int(score), name) for name, score in scores.items())[:2]
    active_failures = {
        name: severity
        for name, severity in (result.get("failure_severity") or {}).items()
        if int(severity) > 0
    }
    reason = "weakest L1: " + ", ".join(f"{name}={score}" for score, name in weakest)
    return reason, {
        "weakest_l1": [{"name": name, "score": score} for score, name in weakest],
        "active_failures": active_failures,
        "failure_caps_applied": result.get("failure_caps_applied", []),
    }


def load_base_module() -> Any:
    return types.SimpleNamespace(
        load_default_judge_env=load_default_judge_env,
        load_rubric=load_rubric,
        resolve_rubric_yaml_path=resolve_rubric_yaml_path,
        call_gemini_l1_judge=call_gemini_l1_judge,
        aggregate_l1=aggregate_l1,
        image_consistency_metrics=image_consistency_metrics,
        apply_soft_preservation_penalties=apply_soft_preservation_penalties,
        apply_training_reward_calibration=apply_training_reward_calibration,
        build_candidate_reason=build_candidate_reason,
    )
