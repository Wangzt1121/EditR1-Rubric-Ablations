#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Edit-R1 reward helper: original test.py flow + official Gemini native API.

This helper is intentionally thin:

1. Load the repository-local Original scorer module:
      reward_server/test_gemini.py
2. Keep its rubric loading, L3/L2/L1 aggregation, soft penalties,
   calibration, grid/log formatting, and CLI behavior unchanged.
3. Add only one native Gemini judge function named ``call_gemini_l3_judge``.

``flow_grpo.rewards.mllm_single_api_score`` calls ``call_gemini_l3_judge`` when
``SINGLE_REWARD_NATIVE_GEMINI=1``. Everything after the L3 JSON response remains
the original ``test.py`` code path.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional


BASE_TEST_PATH = Path(
    os.getenv(
        "EDIT_R1_ORIGINAL_REWARD_SCORER",
        str(Path(__file__).resolve().with_name("test_gemini.py")),
    )
).expanduser()


def _data_url_to_gemini_inline_part(data_url: str) -> Dict[str, Any]:
    header, encoded = str(data_url).split(",", 1)
    mime = "image/jpeg"
    if header.startswith("data:") and ";base64" in header:
        mime = header[len("data:") : header.index(";base64")]
    return {"inlineData": {"mimeType": mime, "data": encoded}}


def _gemini_native_generate_url(model: str, base_url: str = "") -> str:
    base = str(
        os.getenv("GEMINI_API_BASE_URL", "")
        or base_url
        or "https://generativelanguage.googleapis.com/v1beta"
    ).strip()
    if not base:
        base = "https://generativelanguage.googleapis.com/v1beta"
    base = base.rstrip("/")
    if base.endswith(":generateContent"):
        return base
    model_part = urllib.parse.quote(str(model).strip(), safe="")
    if base.endswith("/models"):
        return f"{base}/{model_part}:generateContent"
    return f"{base}/models/{model_part}:generateContent"


def _extract_gemini_text(response_json: Dict[str, Any]) -> str:
    texts: List[str] = []
    for cand in response_json.get("candidates", []) or []:
        content = cand.get("content", {}) or {}
        for part in content.get("parts", []) or []:
            if isinstance(part, dict) and part.get("text") is not None:
                texts.append(str(part.get("text")))
    if not texts:
        raise RuntimeError(
            "Gemini native response has no text parts: "
            + json.dumps(response_json, ensure_ascii=False)[:1000]
        )
    return "\n".join(texts).strip()


def _build_gemini_response_json_schema(base: Any, active_l3: List[str]) -> Dict[str, Any]:
    """Convert the original test.py JSON schema to Gemini responseJsonSchema.

    Keep the schema in the official ``responseJsonSchema`` style used by
    generateContent. We only remove minItems/maxItems because the official
    Gemini endpoint rejected those large facet arrays in prior tests; local
    validation still checks completeness via ``normalize_judge_output`` and
    ``validate_judge_output`` from the original scorer.
    """

    def convert(value: Any) -> Any:
        if isinstance(value, dict):
            out: Dict[str, Any] = {}
            for key, item in value.items():
                if key in {"minItems", "maxItems"}:
                    continue
                if key == "type":
                    if isinstance(item, str):
                        out[key] = item.lower()
                    elif isinstance(item, list):
                        out[key] = [str(x).lower() for x in item]
                    else:
                        out[key] = item
                elif key == "properties" and isinstance(item, dict):
                    out[key] = {str(k): convert(v) for k, v in item.items()}
                else:
                    out[key] = convert(item)
            return out
        if isinstance(value, list):
            return [convert(v) for v in value]
        return value

    return convert(base.build_response_schema(active_l3))


def _install_native_gemini_judge(base: Any) -> Any:
    def call_gemini_l3_judge(
        *,
        client: Any = None,
        model: str,
        source_image_path: str,
        edited_image_path: str,
        prompt: str,
        rubric: Any,
        requirement: str = "",
        max_retries: int = 3,
        max_image_side: int = 1024,
        jpeg_quality: int = 90,
        api_key: str = "",
        base_url: str = "",
        request_timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        # Keep this prompt construction matched to /nvmedata/.../test.py:
        # call_gpt_l3_judge(), including wording and section order.
        active_l3 = rubric.active_l3
        rubric_text = base.build_rubric_text(rubric)
        original_schema = base.build_response_schema(active_l3)
        official_schema = _build_gemini_response_json_schema(base, active_l3)
        consistency_metrics = base.image_consistency_metrics(source_image_path, edited_image_path)

        source_url = base.image_to_data_url(
            source_image_path, max_side=max_image_side, quality=jpeg_quality
        )
        edited_url = base.image_to_data_url(
            edited_image_path, max_side=max_image_side, quality=jpeg_quality
        )

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
{base.format_consistency_metrics(consistency_metrics)}

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
{json.dumps(original_schema, ensure_ascii=False)}

Return only JSON matching the schema. Do not output facet names, evidence,
explanations, markdown, L1/L2 values, an overall score, or failure reasoning.
""".strip()

        key = str(
            api_key
            or os.getenv("GEMINI_API_KEY", os.getenv("GOOGLE_API_KEY", os.getenv("SINGLE_REWARD_API_KEY", "")))
        ).strip()
        if not key:
            raise RuntimeError("Missing Gemini API key. Export GEMINI_API_KEY or GOOGLE_API_KEY.")

        timeout = request_timeout
        if timeout is None or timeout <= 0:
            timeout = float(os.getenv("GEMINI_NATIVE_TIMEOUT", os.getenv("SINGLE_REWARD_TIMEOUT", "90")))

        url = _gemini_native_generate_url(model, base_url)
        payload = {
            "systemInstruction": {"parts": [{"text": system_prompt}]},
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"text": user_text},
                        {"text": "Source image before editing:"},
                        _data_url_to_gemini_inline_part(source_url),
                        {"text": "Edited image generated by the model:"},
                        _data_url_to_gemini_inline_part(edited_url),
                    ],
                }
            ],
            "generationConfig": {
                "temperature": 0,
                "maxOutputTokens": int(os.getenv("GEMINI_NATIVE_MAX_OUTPUT_TOKENS", "8192")),
                "responseMimeType": "application/json",
                "responseJsonSchema": official_schema,
            },
        }

        if str(os.getenv("GEMINI_NATIVE_DEBUG_PAYLOAD", "0")).strip().lower() in {"1", "true", "yes", "y"}:
            safe_payload = json.loads(json.dumps(payload, ensure_ascii=False))
            for content in safe_payload.get("contents", []) or []:
                for part in content.get("parts", []) or []:
                    if isinstance(part, dict) and "inlineData" in part:
                        blob = part.get("inlineData") or {}
                        if blob.get("data"):
                            blob["data"] = f"<base64 {len(blob['data'])} chars>"
            print(
                "[gemini_native_payload_original_flow] "
                + json.dumps(safe_payload, ensure_ascii=False)[:8000],
                flush=True,
            )

        last_err: Optional[Exception] = None
        for attempt in range(1, max_retries + 1):
            try:
                req = urllib.request.Request(
                    url,
                    data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                    headers={
                        "Content-Type": "application/json; charset=utf-8",
                        "x-goog-api-key": key,
                    },
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    response_json = json.loads(resp.read().decode("utf-8"))
                output_text = _extract_gemini_text(response_json)
                output = base.normalize_judge_output(base.parse_json_object(output_text), active_l3)
                base.validate_judge_output(output, active_l3)
                output["gemini_native_api"] = True
                output["gemini_payload_style"] = "official_generate_content_response_json_schema"
                output["original_flow_base"] = str(BASE_TEST_PATH)
                return output
            except urllib.error.HTTPError as err:
                body = err.read().decode("utf-8", errors="replace")[:4000]
                last_err = RuntimeError(f"Gemini native HTTP {err.code}: {body}")
                if err.code in {400, 401, 403, 404}:
                    raise last_err
                if attempt < max_retries:
                    time.sleep(2 * attempt)
                else:
                    raise last_err
            except Exception as err:
                last_err = err
                if attempt < max_retries:
                    time.sleep(2 * attempt)
                else:
                    raise RuntimeError(
                        f"Gemini native judge failed after {max_retries} attempts: {last_err}"
                    ) from err

        raise RuntimeError(f"Unexpected Gemini native judge failure: {last_err}")

    base.call_gemini_l3_judge = call_gemini_l3_judge
    return base


def load_base_module():
    if not BASE_TEST_PATH.exists():
        raise FileNotFoundError(f"Base scorer not found: {BASE_TEST_PATH}")
    spec = importlib.util.spec_from_file_location("wzt_original_test_scorer_native_gemini", BASE_TEST_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import base scorer from {BASE_TEST_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return _install_native_gemini_judge(module)
