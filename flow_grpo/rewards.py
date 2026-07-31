from PIL import Image
import io
import os
import numpy as np
import torch
from collections import defaultdict
import random
import time

def jpeg_incompressibility():
    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
        images = [Image.fromarray(image) for image in images]
        buffers = [io.BytesIO() for _ in images]
        for image, buffer in zip(images, buffers):
            image.save(buffer, format="JPEG", quality=95)
        sizes = [buffer.tell() / 1000 for buffer in buffers]
        return np.array(sizes), {}

    return _fn


def jpeg_compressibility():
    jpeg_fn = jpeg_incompressibility()

    def _fn(images, prompts, metadata):
        rew, meta = jpeg_fn(images, prompts, metadata)
        return -rew / 500, meta

    return _fn


def mllm_score_continue(device):
    """Submits images to GenEval and computes a reward.
    """
    import requests
    from requests.adapters import HTTPAdapter, Retry
    from io import BytesIO
    import pickle

    batch_size = 64
    url = f"http://{os.getenv('REWARD_SERVER', 'localhost:12341')}/mode/logits_non_cot"
    sess = requests.Session()
    retries = Retry(
        total=1000, backoff_factor=1, status_forcelist=[500], allowed_methods=False
    )
    sess.mount("http://", HTTPAdapter(max_retries=retries))

    def _to_uint8_nhwc(batch):
        if isinstance(batch, torch.Tensor):
            array = batch.detach().cpu()
            if array.ndim == 3:
                array = array.unsqueeze(0)
            if array.ndim == 4 and array.shape[1] in (1, 3):
                array = array.permute(0, 2, 3, 1)
            array = array.float().numpy()
        elif isinstance(batch, np.ndarray):
            array = batch
            if array.ndim == 3:
                array = array[None, ...]
        else:
            array = np.array([
                np.array(img.convert("RGB") if hasattr(img, "convert") else img)
                for img in batch
            ])
        if array.dtype != np.uint8:
            if array.size and float(np.nanmax(array)) <= 1.5:
                array = array * 255.0
            array = np.rint(array).clip(0, 255).astype(np.uint8)
        return array

    def _fn(ref_images, images, prompts, metadatas):
        images = _to_uint8_nhwc(images)
        ref_images = _to_uint8_nhwc(ref_images)
        if len(ref_images) == 1 and len(images) > 1:
            ref_images = np.repeat(ref_images, len(images), axis=0)
        if len(ref_images) != len(images):
            raise ValueError(
                f"mllm_score_continue expected matching image/ref counts, "
                f"got images={len(images)} ref_images={len(ref_images)}"
            )

        prompts = list(prompts)
        metadatas = list(metadatas)
        all_scores = []
        for start_idx in range(0, len(images), batch_size):
            end_idx = min(start_idx + batch_size, len(images))
            image_batch = images[start_idx:end_idx]
            ref_image_batch = ref_images[start_idx:end_idx]
            prompt_batch = prompts[start_idx:end_idx]
            metadata_batch = metadatas[start_idx:end_idx]

            jpeg_images = []
            for image in image_batch:
                img = Image.fromarray(image).convert("RGB")
                buffer = BytesIO()
                img.save(buffer, format="JPEG")
                jpeg_images.append(buffer.getvalue())

            ref_jpeg_images = []
            for ref_image in ref_image_batch:
                img = Image.fromarray(ref_image).convert("RGB")
                buffer = BytesIO()
                img.save(buffer, format="JPEG")
                ref_jpeg_images.append(buffer.getvalue())

            data = {
                "ref_images": ref_jpeg_images,
                "images": jpeg_images,
                "prompts": prompt_batch,
                "metadatas": metadata_batch,
            }
            data_bytes = pickle.dumps(data)

            response = sess.post(url, data=data_bytes, timeout=360)
            response.raise_for_status()
            response_data = pickle.loads(response.content)
            all_scores += response_data["scores"]

        return all_scores, {}
    return _fn


def mllm_relative_api_score(device):
    """OpenAI/Gemini-compatible relative reward for Kontext editing.

    This keeps the original training contract used by mllm_score_continue:
    _fn(ref_images, images, prompts, metadatas) -> scores, details.
    """
    import base64
    import json
    import requests
    from io import BytesIO

    endpoint = os.getenv("RELATIVE_REWARD_API_URL", "").strip()
    base_url = os.getenv(
        "RELATIVE_REWARD_BASE_URL",
        os.getenv("GPT_REWARD_BASE_URL", os.getenv("OPENAI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")),
    ).strip()
    api_key = os.getenv(
        "RELATIVE_REWARD_API_KEY",
        os.getenv("OPENAI_API_KEY", os.getenv("GEMINI_API_KEY", os.getenv("GOOGLE_API_KEY", os.getenv("GPT_REWARD_API_KEY", "")))),
    ).strip()
    model = os.getenv(
        "RELATIVE_REWARD_MODEL",
        os.getenv("GPT_RUBRIC_MODEL", os.getenv("GEMINI_REWARD_MODEL", os.getenv("GPT_REWARD_MODEL", "gemini-2.5-flash"))),
    ).strip()
    rubric_yaml = os.getenv("RELATIVE_RUBRIC_YAML", "/nvmedata/workspace2/users/wzt/hair_edit.yaml")
    timeout = float(os.getenv("RELATIVE_REWARD_TIMEOUT", "180"))
    max_retries = int(os.getenv("RELATIVE_REWARD_MAX_RETRIES", "2"))
    jpeg_quality = int(os.getenv("RELATIVE_REWARD_JPEG_QUALITY", "95"))
    group_by = os.getenv("RELATIVE_REWARD_GROUP_BY", "prompt").strip().lower()
    chunk_size = int(os.getenv("RELATIVE_NUM_CANDIDATES", "8"))

    dimensions = [
        "target_edit_accuracy",
        "identity_preservation",
        "non_target_preservation",
        "color_lighting_texture_preservation",
        "photorealism_artifact_control",
    ]
    weights = {
        "target_edit_accuracy": 0.34,
        "identity_preservation": 0.18,
        "non_target_preservation": 0.20,
        "color_lighting_texture_preservation": 0.18,
        "photorealism_artifact_control": 0.10,
    }

    if not endpoint:
        endpoint = base_url.rstrip("/")
        if endpoint.endswith("/chat/completions"):
            pass
        elif endpoint.endswith("/v1") or endpoint.endswith("/openai"):
            endpoint = endpoint + "/chat/completions"
        else:
            endpoint = endpoint + "/v1/chat/completions"

    rubric_text = ""
    try:
        with open(rubric_yaml, "r", encoding="utf-8") as f:
            rubric_text = f.read()
    except Exception as exc:
        rubric_text = f"Rubric file could not be read: {rubric_yaml}. Error: {exc!r}"

    session = requests.Session()

    def _to_pil_image(image):
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        if isinstance(image, torch.Tensor):
            tensor = image.detach().cpu()
            if tensor.ndim == 3 and tensor.shape[0] in (1, 3):
                tensor = tensor.permute(1, 2, 0)
            array = tensor.float().numpy()
            if array.max() <= 1.5:
                array = array * 255.0
            return Image.fromarray(array.round().clip(0, 255).astype(np.uint8)).convert("RGB")
        array = np.array(image)
        if array.dtype != np.uint8:
            if array.max() <= 1.5:
                array = array * 255.0
            array = array.round().clip(0, 255).astype(np.uint8)
        return Image.fromarray(array).convert("RGB")

    def _image_data_url(image):
        buffer = BytesIO()
        _to_pil_image(image).save(buffer, format="JPEG", quality=jpeg_quality)
        encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{encoded}"

    def _extract_json(text):
        raw = str(text or "").strip()
        if raw.startswith("```"):
            raw = raw.strip("`").strip()
            if raw.lower().startswith("json"):
                raw = raw[4:].strip()
        start = raw.find("{")
        end = raw.rfind("}")
        if start >= 0 and end > start:
            raw = raw[start : end + 1]
        return json.loads(raw)

    def _candidate_groups(prompts, metadatas):
        if group_by == "chunk":
            return [list(range(i, min(i + chunk_size, len(prompts)))) for i in range(0, len(prompts), chunk_size)]
        groups = []
        seen = {}
        for idx, prompt in enumerate(prompts):
            meta = metadatas[idx] if isinstance(metadatas, (list, tuple)) and idx < len(metadatas) else {}
            key = (str(prompt), str(meta.get("image", "")) if isinstance(meta, dict) else "")
            if key not in seen:
                seen[key] = len(groups)
                groups.append([])
            groups[seen[key]].append(idx)
        return groups

    def _build_schema(candidate_ids):
        n = len(candidate_ids)
        return {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "candidate_results": {
                    "type": "array",
                    "minItems": n,
                    "maxItems": n,
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "candidate_id": {"type": "string", "enum": candidate_ids},
                            "relative_rank": {"type": "integer", "minimum": 1, "maximum": n},
                            "relative_reason": {"type": "string"},
                            "dimension_scores": {
                                "type": "object",
                                "additionalProperties": False,
                                "properties": {
                                    key: {"type": "integer", "minimum": 0, "maximum": 9}
                                    for key in dimensions
                                },
                                "required": dimensions,
                            },
                            "dimension_reasons": {
                                "type": "object",
                                "additionalProperties": False,
                                "properties": {key: {"type": "string"} for key in dimensions},
                                "required": dimensions,
                            },
                            "failure_tags": {"type": "array", "items": {"type": "string"}},
                        },
                        "required": [
                            "candidate_id",
                            "relative_rank",
                            "relative_reason",
                            "dimension_scores",
                            "dimension_reasons",
                            "failure_tags",
                        ],
                    },
                }
            },
            "required": ["candidate_results"],
        }

    def _score_from_dimensions(dimension_scores):
        score = 0.0
        for key in dimensions:
            value = float(dimension_scores.get(key, 0)) / 9.0
            score += weights[key] * max(0.0, min(1.0, value))
        return float(max(0.0, min(1.0, score)))

    def _call_api(source_image, candidate_images, prompt, metadata):
        candidate_ids = [f"candidate_{i + 1:02d}" for i in range(len(candidate_images))]
        schema = _build_schema(candidate_ids)
        user_text = f"""
You are a strict relative reward judge for human image editing.

Judge one source image, one edit instruction, and multiple anonymous edited candidates together.
Use the candidates as mutual references. Do not infer model identity.
Return JSON only. Do not output an overall score; code will aggregate the five dimensions.

Edit instruction:
{prompt}

Metadata:
{json.dumps(metadata, ensure_ascii=False, default=str)[:4000]}

Score every candidate on these 0-9 dimensions:
- target_edit_accuracy: requested edit correctness, visibility, localization, strength.
- identity_preservation: same person, face shape, facial features, pose, expression.
- non_target_preservation: background, clothing, accessories, body, and unrelated regions unchanged.
- color_lighting_texture_preservation: source color, saturation, brightness, contrast, skin texture, global tone.
- photorealism_artifact_control: realism, sharpness, blending, no blur/smoothing/artifacts.

Ranking priority:
1. The requested edit must be correct.
2. If edit accuracy is similar, identity and non-target preservation dominate.
3. Color/saturation/lighting/texture drift is a major failure.
4. Good target editing cannot compensate for obvious global or non-target damage.

Rubric guidance:
{rubric_text[:12000]}

Candidate ids:
{json.dumps(candidate_ids)}

Required JSON schema:
{json.dumps(schema, ensure_ascii=False)}
""".strip()

        content = [
            {"type": "text", "text": user_text},
            {"type": "text", "text": "Source image before editing:"},
            {"type": "image_url", "image_url": {"url": _image_data_url(source_image)}},
        ]
        for candidate_id, image in zip(candidate_ids, candidate_images):
            content.append({"type": "text", "text": f"Edited candidate {candidate_id}:"})
            content.append({"type": "image_url", "image_url": {"url": _image_data_url(image)}})

        payload = {
            "model": model,
            "temperature": 0,
            "max_tokens": 4096,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a strict image-editing reward judge. Return only valid JSON.",
                },
                {"role": "user", "content": content},
            ],
            "response_format": {"type": "json_object"},
        }
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        last_error = None
        for attempt in range(max_retries + 1):
            try:
                response = session.post(endpoint, headers=headers, json=payload, timeout=timeout)
                if response.status_code >= 400 and "response_format" in payload:
                    fallback_payload = dict(payload)
                    fallback_payload.pop("response_format", None)
                    response = session.post(endpoint, headers=headers, json=fallback_payload, timeout=timeout)
                response.raise_for_status()
                data = response.json()
                text = data["choices"][0]["message"]["content"]
                output = _extract_json(text)
                results = output.get("candidate_results", [])
                if len(results) != len(candidate_images):
                    raise RuntimeError(f"candidate_results length mismatch: {len(results)} vs {len(candidate_images)}")
                by_id = {item.get("candidate_id"): item for item in results}
                return [by_id[candidate_id] for candidate_id in candidate_ids]
            except Exception as exc:
                last_error = exc
                if attempt >= max_retries:
                    raise RuntimeError(f"relative reward API failed: {last_error!r}") from exc

    def _fn(ref_images, images, prompts, metadatas):
        if isinstance(images, torch.Tensor):
            images = images.detach().cpu()
        prompts = list(prompts)
        metadatas_list = list(metadatas) if isinstance(metadatas, (list, tuple)) else [{} for _ in prompts]
        scores = [0.0 for _ in prompts]
        details = []

        for group in _candidate_groups(prompts, metadatas_list):
            source_idx = group[0]
            source_image = ref_images[source_idx] if ref_images is not None else images[source_idx]
            candidate_images = [images[idx] for idx in group]
            prompt = prompts[source_idx]
            metadata = [metadatas_list[idx] for idx in group]
            candidate_results = _call_api(source_image, candidate_images, prompt, metadata)
            for idx, result in zip(group, candidate_results):
                score = _score_from_dimensions(result.get("dimension_scores", {}))
                scores[idx] = score
                details.append(
                    {
                        "index": idx,
                        "score": score,
                        "prompt": prompt,
                        **result,
                    }
                )

        return scores, {"details": details}

    return _fn

def mllm_single_api_score(device):
    """OpenAI/Gemini-compatible single-candidate reward.

    The default mode reuses the repository-local bottom-up L3 scorer. Setting
    SINGLE_REWARD_MODE=l1 selects the direct four-dimension L1 helper instead.
    """
    import fcntl
    import importlib.util
    import json
    import re
    import tempfile
    import traceback
    from concurrent.futures import ThreadPoolExecutor, FIRST_COMPLETED, wait
    from pathlib import Path

    reward_mode = os.getenv("SINGLE_REWARD_MODE", "l3").strip().lower()
    if reward_mode not in {"l1", "l3"}:
        raise ValueError(f"Unsupported SINGLE_REWARD_MODE={reward_mode!r}; use 'l1' or 'l3'")
    native_gemini = os.getenv(
        "SINGLE_REWARD_NATIVE_GEMINI",
        os.getenv("GEMINI_NATIVE_API", "0"),
    ).strip().lower() in {"1", "true", "yes", "y"}
    repo_root = Path(__file__).resolve().parents[1]
    if reward_mode == "l1":
        default_helper_path = repo_root / "reward_server" / "l1_only_training_reward.py"
    else:
        default_helper_path = repo_root / "reward_server" / "test_gemini_reward.py"
    helper_path = Path(
        os.getenv(
            "SINGLE_REWARD_HELPER_PATH",
            str(default_helper_path),
        )
    )
    if not helper_path.exists():
        raise FileNotFoundError(f"Single reward helper not found: {helper_path}")

    spec = importlib.util.spec_from_file_location("edit_r1_single_reward_helper", str(helper_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import single reward helper from {helper_path}")
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    base = helper.load_base_module()

    if hasattr(base, "load_default_judge_env"):
        base.load_default_judge_env()

    api_key = os.getenv(
        "SINGLE_REWARD_API_KEY",
        os.getenv("OPENAI_API_KEY", os.getenv("GEMINI_API_KEY", os.getenv("GOOGLE_API_KEY", os.getenv("GPT_REWARD_API_KEY", "")))),
    ).strip()
    base_url = os.getenv(
        "SINGLE_REWARD_BASE_URL",
        os.getenv("GPT_REWARD_BASE_URL", os.getenv("OPENAI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")),
    ).strip()
    model = os.getenv(
        "SINGLE_REWARD_MODEL",
        os.getenv("GPT_RUBRIC_MODEL", os.getenv("GEMINI_REWARD_MODEL", os.getenv("GPT_REWARD_MODEL", "gemini-2.5-flash"))),
    ).strip()
    rubric_yaml = os.getenv(
        "SINGLE_REWARD_RUBRIC_YAML",
        os.getenv("RUBRIC_YAML", "/nvmedata/workspace2/users/wzt/Hang/face_rubrics"),
    ).strip()
    request_timeout = float(os.getenv("SINGLE_REWARD_TIMEOUT", os.getenv("GPT_RUBRIC_TIMEOUT", "90")))
    max_retries = int(os.getenv("SINGLE_REWARD_MAX_RETRIES", "3"))
    deadline_seconds = float(os.getenv("SINGLE_REWARD_DEADLINE_SECONDS", "120"))
    max_image_side = int(os.getenv("SINGLE_REWARD_MAX_IMAGE_SIDE", "1024"))
    jpeg_quality = int(os.getenv("SINGLE_REWARD_JPEG_QUALITY", "90"))
    workers = int(os.getenv("SINGLE_REWARD_WORKERS", "2"))
    fail_open = os.getenv("SINGLE_REWARD_FAIL_OPEN", "1").strip().lower() in {"1", "true", "yes", "y"}
    default_score = float(os.getenv("SINGLE_REWARD_DEFAULT_SCORE", "0.001"))
    enforce_unique = os.getenv("SINGLE_REWARD_ENFORCE_UNIQUE", "1").strip().lower() not in {"0", "false", "no", "n"}
    use_api_lock = os.getenv("SINGLE_REWARD_API_LOCK", "1").strip().lower() not in {"0", "false", "no", "n"}
    api_lock_path = os.getenv("SINGLE_REWARD_API_LOCK_PATH", "/tmp/edit_r1_single_reward_api.lock")
    api_channels = int(os.getenv("SINGLE_REWARD_API_CHANNELS", "1") or "1")
    api_channels = max(1, api_channels) if use_api_lock else 0
    debug_enabled = os.getenv("SINGLE_REWARD_DEBUG", "0").strip().lower() in {"1", "true", "yes", "y"}
    debug_limit = int(os.getenv("SINGLE_REWARD_DEBUG_LIMIT", "3"))
    debug_state = {"printed": 0}
    native_gemini = os.getenv("SINGLE_REWARD_NATIVE_GEMINI", os.getenv("GEMINI_NATIVE_API", "0")).strip().lower() in {"1", "true", "yes", "y"}

    print(
        "[mllm_single_api_score] "
        f"model={model} base_url={base_url} rubric_root={rubric_yaml} "
        f"workers={workers} max_image_side={max_image_side} jpeg_quality={jpeg_quality} "
        f"api_lock={use_api_lock} api_channels={api_channels} native_gemini={native_gemini} "
        f"reward_mode={reward_mode}"
    )

    def _to_pil_image(image):
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        if isinstance(image, torch.Tensor):
            tensor = image.detach().cpu()
            if tensor.ndim == 3 and tensor.shape[0] in (1, 3):
                tensor = tensor.permute(1, 2, 0)
            array = tensor.float().numpy()
            if array.size and array.max() <= 1.5:
                array = array * 255.0
            return Image.fromarray(array.round().clip(0, 255).astype(np.uint8)).convert("RGB")
        array = np.array(image)
        if array.dtype != np.uint8:
            if array.size and array.max() <= 1.5:
                array = array * 255.0
            array = array.round().clip(0, 255).astype(np.uint8)
        return Image.fromarray(array).convert("RGB")

    def _metadata_at(metadatas, index):
        if isinstance(metadatas, (list, tuple)) and index < len(metadatas):
            item = metadatas[index]
        else:
            item = {}
        return dict(item) if isinstance(item, dict) else {"metadata": item}

    def _source_at(ref_images, images, index):
        if ref_images is None:
            return images[index]
        if isinstance(ref_images, torch.Tensor):
            if len(ref_images) == len(images):
                return ref_images[index]
            return ref_images[index % len(ref_images)]
        if isinstance(ref_images, (list, tuple)):
            if len(ref_images) == len(images):
                return ref_images[index]
            return ref_images[index % len(ref_images)]
        return ref_images

    def _data_dir_for_metadata(metadata):
        for key in ("data_dir", "source_dir", "image_dir"):
            value = metadata.get(key)
            if value:
                return str(value)
        for key in ("source", "source_image", "image", "image_path", "path"):
            value = metadata.get(key)
            if not value:
                continue
            path = Path(str(value))
            if not path.is_absolute():
                root = os.getenv("DATASET_ROOT", "").strip()
                if root:
                    path = Path(root) / path
            return str(path.parent if path.suffix else path)
        return os.getenv("DATASET_ROOT", ".")

    def _rubric_tokens_from_metadata(metadata):
        tokens = []
        for key in (
            "rubric_yaml",
            "rubric_path",
            "rubric_file",
            "rubric_key",
            "category",
            "category_key",
            "task_key",
            "zh_category",
            "category_name",
        ):
            value = metadata.get(key)
            if value:
                tokens.append(str(value))

        for key in ("image", "source", "source_image", "image_path", "path"):
            value = metadata.get(key)
            if not value:
                continue
            path = Path(str(value))
            tokens.extend(str(part) for part in path.parts)

        source_record = metadata.get("source_record")
        if isinstance(source_record, dict):
            for value in source_record.values():
                if isinstance(value, str):
                    path = Path(value)
                    tokens.extend(str(part) for part in path.parts)

        cleaned = []
        seen = set()
        for token in tokens:
            for item in (token, Path(token).stem):
                item = str(item).strip().strip("/\\")
                if not item or item in seen:
                    continue
                cleaned.append(item)
                seen.add(item)
        return cleaned

    def _resolve_rubric_for_metadata(metadata):
        root = Path(str(rubric_yaml)).expanduser()
        tokens = _rubric_tokens_from_metadata(metadata)

        for key in ("rubric_yaml", "rubric_path", "rubric_file"):
            value = metadata.get(key)
            if not value:
                continue
            path = Path(str(value)).expanduser()
            candidates = [path]
            if not path.is_absolute():
                candidates.append(root / path)
            if path.suffix.lower() not in (".yaml", ".yml"):
                candidates.extend([root / f"{path}.yaml", root / f"{path}.yml"])
            for candidate in candidates:
                if candidate.is_file():
                    return str(candidate)

        if root.is_file():
            return str(root)
        if not root.exists():
            raise FileNotFoundError(f"Rubric path not found: {rubric_yaml}")
        if not root.is_dir():
            raise ValueError(f"Rubric path must be a YAML file or directory: {rubric_yaml}")

        rubric_files = [
            path for path in root.iterdir()
            if path.is_file() and path.suffix.lower() in (".yaml", ".yml")
        ]
        by_stem = {path.stem: path for path in rubric_files}
        by_name = {path.name: path for path in rubric_files}

        for token in tokens:
            for candidate in (
                token,
                f"{token}.yaml",
                f"{token}.yml",
                token.replace(" ", "_"),
                token.replace(" ", "_") + ".yaml",
            ):
                if candidate in by_name:
                    return str(by_name[candidate])
                if candidate in by_stem:
                    return str(by_stem[candidate])

        for token in tokens:
            match = re.match(r"^(\d{1,2})(?:[_\\-].*)?$", token)
            if not match:
                continue
            prefix = f"{int(match.group(1)):02d}_"
            matches = [path for path in rubric_files if path.name.startswith(prefix)]
            if len(matches) == 1:
                return str(matches[0])

        data_dir = _data_dir_for_metadata(metadata)
        if hasattr(base, "resolve_rubric_yaml_path"):
            return base.resolve_rubric_yaml_path(str(root), data_dir)

        raise FileNotFoundError(
            f"Cannot resolve rubric under {root} for metadata keys "
            f"category={metadata.get('category')!r}, rubric_key={metadata.get('rubric_key')!r}, "
            f"zh_category={metadata.get('zh_category')!r}, image={metadata.get('image')!r}"
        )

    def _group_key(prompt, metadata):
        image_key = metadata.get("image") or metadata.get("source_image") or metadata.get("sample_id") or metadata.get("path") or ""
        return (str(prompt), str(image_key))

    def _score_one(index, source_image, candidate_image, prompt, metadata, temp_dir):
        source_path = temp_dir / f"source_{index:06d}.jpg"
        candidate_path = temp_dir / f"candidate_{index:06d}.jpg"
        _to_pil_image(source_image).save(source_path, format="JPEG", quality=jpeg_quality)
        _to_pil_image(candidate_image).save(candidate_path, format="JPEG", quality=jpeg_quality)

        resolved_rubric = _resolve_rubric_for_metadata(metadata)
        rubric = base.load_rubric(resolved_rubric)

        if native_gemini:
            client = None
        else:
            client_kwargs = {}
            if api_key:
                client_kwargs["api_key"] = api_key
            if base_url:
                client_kwargs["base_url"] = base_url
            if request_timeout and request_timeout > 0:
                client_kwargs["timeout"] = request_timeout
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise ImportError("Missing dependency: openai. Install it in the training env, or set SINGLE_REWARD_NATIVE_GEMINI=1 for the native Gemini path.") from exc
            client = OpenAI(**client_kwargs)

        def _call_judge():
            judge_kwargs = dict(
                client=client,
                model=model,
                source_image_path=str(source_path),
                edited_image_path=str(candidate_path),
                prompt=str(prompt),
                rubric=rubric,
                requirement=str(metadata.get("requirement", "")),
                max_retries=max_retries,
                max_image_side=max_image_side,
                jpeg_quality=jpeg_quality,
            )
            if native_gemini:
                judge_kwargs.update(
                    api_key=api_key,
                    base_url=base_url,
                    request_timeout=request_timeout,
                )
            if reward_mode == "l1":
                judge_func = getattr(base, "call_gemini_l1_judge")
            else:
                judge_func = getattr(base, "call_gemini_l3_judge", None)
                if judge_func is None:
                    judge_func = getattr(base, "call_gpt_l3_judge")
            return judge_func(**judge_kwargs)

        if use_api_lock:
            Path(api_lock_path).parent.mkdir(parents=True, exist_ok=True)
            while True:
                acquired_file = None
                for channel_idx in range(api_channels):
                    slot_path = (
                        api_lock_path
                        if api_channels == 1
                        else f"{api_lock_path}.{channel_idx:02d}"
                    )
                    lock_file = open(slot_path, "w")
                    try:
                        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        lock_file.close()
                        continue
                    acquired_file = lock_file
                    break
                if acquired_file is None:
                    time.sleep(0.05)
                    continue
                try:
                    judge_output = _call_judge()
                finally:
                    fcntl.flock(acquired_file, fcntl.LOCK_UN)
                    acquired_file.close()
                break
        else:
            judge_output = _call_judge()
        if reward_mode == "l1":
            result = base.aggregate_l1(rubric=rubric, judge_output=judge_output)
        else:
            result = base.aggregate_bottom_up(rubric=rubric, judge_output=judge_output)
        consistency_metrics = base.image_consistency_metrics(str(source_path), str(candidate_path))
        result["image_consistency_metrics"] = consistency_metrics
        result["image_consistency_caps"] = []
        result = base.apply_soft_preservation_penalties(result, consistency_metrics, rubric)
        result = base.apply_training_reward_calibration(
            result,
            sample_path=str(candidate_path),
            sample_index=index + 1,
        )
        reason, reason_details = base.build_candidate_reason(result, rubric)
        result["reason"] = reason
        result["reason_details"] = reason_details
        row = {
            "index": index,
            "score": float(max(0.0, min(1.0, result.get("reward", 0.0)))),
            "prompt": str(prompt),
            "metadata": metadata,
            "rubric_yaml": resolved_rubric,
            "rubric_task_key": getattr(rubric, "task_key", ""),
            "rubric_version": getattr(rubric, "version", ""),
            "model": model,
            "error": None,
            **result,
        }
        if debug_enabled and debug_state["printed"] < debug_limit:
            row["raw_judge_output"] = judge_output
        return row

    dimension_keys = [
        "target_edit_accuracy",
        "identity_preservation",
        "non_target_preservation",
        "color_lighting_texture_preservation",
        "photorealism_artifact_control",
    ]
    dimension_weights = {
        "target_edit_accuracy": 0.35,
        "identity_preservation": 0.20,
        "non_target_preservation": 0.20,
        "color_lighting_texture_preservation": 0.10,
        "photorealism_artifact_control": 0.15,
    }
    dimension_aliases = {
        "target_edit_accuracy": (
            "target_edit_accuracy",
            "target_edit_fulfillment",
            "Target Edit Correctness",
            "Prompt Compliance",
            "prompt_following",
            "edit_group_score",
        ),
        "identity_preservation": (
            "identity_preservation",
            "identity_and_human_consistency",
            "Identity and Face Preservation",
            "Subject Preservation",
        ),
        "non_target_preservation": (
            "non_target_preservation",
            "Non-target Consistency",
            "Subject Preservation",
        ),
        "color_lighting_texture_preservation": (
            "color_lighting_texture_preservation",
            "color_preservation",
            "Color Lighting Texture Preservation",
            "Color/Lighting/Texture Preservation",
            "color_detail_group_score",
            "metric_color_quality_score",
            "Subject Preservation",
        ),
        "photorealism_artifact_control": (
            "photorealism_artifact_control",
            "photorealism_and_visual_quality",
            "Visual Fidelity",
            "visual_quality",
            "quality_group_score",
            "Task Realism and Image Quality",
        ),
    }
    last_success_mean = {"value": default_score}

    def _clamp_score(value, fallback=default_score):
        try:
            value = float(value)
        except Exception:
            value = fallback
        return float(max(0.0, min(1.0, value)))

    def _normalize_dimension_value(value):
        value = float(value)
        if value > 1.0 and value <= 9.0:
            value = value / 9.0
        return float(max(0.0, min(1.0, value)))

    def _row_score(row):
        return _clamp_score(row.get("reward", row.get("score", default_score)))

    def _dimension_reward(dims):
        return _clamp_score(sum(dimension_weights[key] * float(dims[key]) for key in dimension_keys))

    def _is_fallback_row(row):
        return bool(
            row.get("score_is_timeout_fallback")
            or row.get("score_is_placeholder")
            or row.get("score_is_error_fallback")
            or row.get("score_is_parse_fallback")
        )

    def _extract_dimensions(row, allow_fallback=None, fallback=None):
        fallback_score = _row_score(row) if fallback is None else _clamp_score(fallback)
        allow_fallback = _is_fallback_row(row) if allow_fallback is None else bool(allow_fallback)
        l1_scores = row.get("l1_scores") if isinstance(row.get("l1_scores"), dict) else {}
        dim_scores = row.get("dimension_scores_0_1") if isinstance(row.get("dimension_scores_0_1"), dict) else {}
        group_scores = row.get("penalty_group_scores") if isinstance(row.get("penalty_group_scores"), dict) else {}
        stores = (
            ("dimension_scores_0_1", dim_scores),
            ("l1_scores", l1_scores),
            ("penalty_group_scores", group_scores),
            ("row", row),
        )
        result = {}
        sources = {}
        missing = []
        for key in dimension_keys:
            found = False
            for alias in dimension_aliases.get(key, (key,)):
                for source_name, store in stores:
                    if alias not in store or store.get(alias) is None:
                        continue
                    try:
                        result[key] = _normalize_dimension_value(store.get(alias))
                    except Exception:
                        continue
                    sources[key] = f"{source_name}.{alias}"
                    found = True
                    break
                if found:
                    break
            if not found:
                missing.append(key)
                if allow_fallback:
                    result[key] = fallback_score
                    sources[key] = "explicit_fallback"

        row["missing_reward_dimensions"] = missing
        row["dimension_score_sources"] = sources
        if missing and not allow_fallback:
            raise ValueError(f"Missing reward dimensions: {missing}")
        row["normalized_dimension_scores"] = dict(result)
        return result

    def _debug_reward_row(row):
        if not debug_enabled or debug_state["printed"] >= debug_limit:
            return
        debug_state["printed"] += 1
        payload = {
            "index": row.get("index"),
            "sample_id": row.get("metadata", {}).get("sample_id") if isinstance(row.get("metadata"), dict) else None,
            "category": row.get("metadata", {}).get("category") if isinstance(row.get("metadata"), dict) else None,
            "rubric_yaml": row.get("rubric_yaml"),
            "rubric_task_key": row.get("rubric_task_key"),
            "raw_api_response": row.get("raw_judge_output", "<not stored; set SINGLE_REWARD_DEBUG=1>"),
            "parsed_scores": {
                "l1_scores": row.get("l1_scores"),
                "penalty_group_scores": row.get("penalty_group_scores"),
                "dimension_scores_0_1": row.get("dimension_scores_0_1"),
                "dimension_scores_raw_0_9": row.get("dimension_scores_raw_0_9"),
            },
            "missing_keys": row.get("missing_reward_dimensions", []),
            "dimension_score_sources": row.get("dimension_score_sources", {}),
            "normalized_scores": row.get("normalized_dimension_scores", {}),
            "final_reward": row.get("reward", row.get("score")),
            "reward_for_training": _row_score(row),
            "reward_used_for_training": row.get("reward_used_for_training", _row_score(row)),
            "reward_raw_bottom_up_audit": row.get("reward_raw_bottom_up_audit"),
            "reward_before_soft_penalty_audit": row.get("reward_before_soft_penalty_audit"),
            "reward_after_soft_penalty_audit": row.get("reward_after_soft_penalty_audit"),
            "reward_dimension_weighted_audit": row.get("reward_dimension_weighted_audit"),
            "error": row.get("error"),
        }
        print("[mllm_single_api_score][debug_reward]\n" + json.dumps(payload, ensure_ascii=False, indent=2, default=str), flush=True)

    def _mean_completed_score(rows):
        values = [
            _row_score(row) for row in rows
            if not row.get("score_is_timeout_fallback") and not row.get("score_is_placeholder")
        ]
        if values:
            return float(np.mean(values))
        return float(last_success_mean["value"])

    def _error_row(index, prompt, metadata, error, fallback_score=None):
        fallback_score = _clamp_score(
            default_score if fallback_score is None else fallback_score
        )
        row = {
            "index": index,
            "score": fallback_score,
            "reward": fallback_score,
            "reward_for_training": fallback_score,
            "prompt": str(prompt),
            "metadata": metadata,
            "model": model,
            "rubric_yaml": rubric_yaml,
            "error": repr(error),
            "traceback": traceback.format_exc(limit=3),
            "score_is_error_fallback": True,
            "reason": (
                f"Single reward scoring failed; using fallback mean score "
                f"{fallback_score:.6f}: {error!r}"
            ),
        }
        if not fail_open:
            raise RuntimeError(row["reason"]) from error
        print(
            f"[mllm_single_api_score][error_fallback] "
            f"index={index} fallback_mean={fallback_score:.6f} error={error!r}",
            flush=True,
        )
        row["l1_scores"] = {key: fallback_score for key in dimension_keys}
        return row

    def _timeout_row(index, prompt, metadata, fallback_score):
        fallback_score = _clamp_score(fallback_score)
        message = (
            f"Single reward batch exceeded {deadline_seconds:.1f}s; "
            f"using fallback mean score {fallback_score:.6f}."
        )
        return {
            "index": index,
            "score": fallback_score,
            "reward": fallback_score,
            "reward_for_training": fallback_score,
            "prompt": str(prompt),
            "metadata": metadata,
            "model": model,
            "rubric_yaml": rubric_yaml,
            "error": message,
            "score_is_timeout_fallback": True,
            "reason": message,
            "l1_scores": {key: fallback_score for key in dimension_keys},
        }

    def _fn(ref_images, images, prompts, metadatas):
        if isinstance(images, torch.Tensor):
            images = images.detach().cpu()
        prompts = list(prompts)
        scores = [default_score for _ in prompts]
        rows = []

        with tempfile.TemporaryDirectory(prefix="edit_r1_single_reward_") as tmp:
            temp_dir = Path(tmp)
            max_workers = max(1, min(workers, len(prompts)))
            executor = ThreadPoolExecutor(max_workers=max_workers)
            futures = {}
            start_time = time.time()
            try:
                for idx, prompt in enumerate(prompts):
                    metadata = _metadata_at(metadatas, idx)
                    future = executor.submit(
                        _score_one,
                        idx,
                        _source_at(ref_images, images, idx),
                        images[idx],
                        prompt,
                        metadata,
                        temp_dir,
                    )
                    futures[future] = (idx, prompt, metadata)

                pending = set(futures)
                while pending:
                    if deadline_seconds > 0:
                        remaining = deadline_seconds - (time.time() - start_time)
                        if remaining <= 0:
                            break
                    else:
                        remaining = None
                    done, pending = wait(
                        pending,
                        timeout=remaining,
                        return_when=FIRST_COMPLETED,
                    )
                    if not done:
                        break
                    for future in done:
                        idx, prompt, metadata = futures[future]
                        try:
                            row = future.result()
                        except Exception as exc:
                            row = _error_row(idx, prompt, metadata, exc, _mean_completed_score(rows))
                        rows.append(row)

                if pending:
                    fallback_score = _mean_completed_score(rows)
                    print(
                        f"[mllm_single_api_score][timeout] deadline={deadline_seconds:.1f}s "
                        f"pending={len(pending)} fallback_mean={fallback_score:.6f}",
                        flush=True,
                    )
                    for future in pending:
                        future.cancel()
                        idx, prompt, metadata = futures[future]
                        rows.append(_timeout_row(idx, prompt, metadata, fallback_score))
            finally:
                executor.shutdown(wait=False, cancel_futures=True)

        if enforce_unique and hasattr(base, "enforce_unique_training_rewards"):
            groups = defaultdict(list)
            for row in rows:
                groups[_group_key(row.get("prompt", ""), row.get("metadata", {}))].append(row)
            for group_rows in groups.values():
                if len(group_rows) > 1:
                    base.enforce_unique_training_rewards(group_rows)

        details = sorted(rows, key=lambda item: int(item.get("index", 0)))
        dimension_scores = {key: [] for key in dimension_keys}
        parse_failure_count = 0
        missing_key_count = 0
        for row in details:
            try:
                dims = _extract_dimensions(row)
            except Exception as exc:
                if not fail_open:
                    raise
                parse_failure_count += 1
                missing = list(row.get("missing_reward_dimensions", []))
                missing_key_count += len(missing)
                fallback_score = _row_score(row)
                row["score_is_parse_fallback"] = True
                row["error"] = f"Reward dimension parse failed; using explicit fallback score: {exc!r}"
                row["normalized_dimension_scores"] = {key: fallback_score for key in dimension_keys}
                row["dimension_score_sources"] = {key: "parse_failure_fallback" for key in dimension_keys}
                row["score"] = fallback_score
                row["reward"] = fallback_score
                row["reward_for_training"] = fallback_score
                dims = dict(row["normalized_dimension_scores"])
            else:
                missing_key_count += len(row.get("missing_reward_dimensions", []) or [])
                if not _is_fallback_row(row):
                    training_reward = _row_score(row)
                    dimension_reward = _dimension_reward(dims)
                    row["reward_used_for_training"] = training_reward
                    row["reward_raw_bottom_up_audit"] = row.get("reward_raw_bottom_up")
                    row["reward_before_soft_penalty_audit"] = row.get("reward_before_soft_penalty")
                    row["reward_after_soft_penalty_audit"] = row.get("reward_after_soft_penalty")
                    row["reward_dimension_weighted_audit"] = dimension_reward
            idx = int(row["index"])
            scores[idx] = _row_score(row)
            _debug_reward_row(row)
            for key in dimension_keys:
                dimension_scores[key].append(dims[key])

        successful = [
            _row_score(row) for row in details
            if not row.get("score_is_timeout_fallback") and not row.get("score_is_placeholder") and not row.get("score_is_error_fallback") and not row.get("score_is_parse_fallback")
        ]
        if successful:
            last_success_mean["value"] = float(np.mean(successful))

        timeout_count = sum(1 for row in details if row.get("score_is_timeout_fallback"))
        placeholder_count = sum(1 for row in details if row.get("score_is_placeholder") or row.get("score_is_error_fallback"))
        fallback_score_count = timeout_count + placeholder_count + parse_failure_count
        return scores, {
            "details": details,
            "dimension_scores": dimension_scores,
            "timeout_count": timeout_count,
            "placeholder_count": placeholder_count,
            "parse_failure_count": parse_failure_count,
            "missing_key_count": missing_key_count,
            "fallback_score_count": fallback_score_count,
        }

    return _fn

def aesthetic_score(device):
    from flow_grpo.aesthetic_scorer import AestheticScorer

    scorer = AestheticScorer(dtype=torch.float32, device=device)

    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8)
        else:
            images = images.transpose(0, 3, 1, 2)  # NHWC -> NCHW
            images = torch.tensor(images, dtype=torch.uint8)
        scores = scorer(images)
        return scores, {}

    return _fn


def clip_score(device):
    from flow_grpo.clip_scorer import ClipScorer

    scorer = ClipScorer(device=device)

    def _fn(images, prompts, metadata):
        if not isinstance(images, torch.Tensor):
            images = images.transpose(0, 3, 1, 2)  # NHWC -> NCHW
            images = torch.tensor(images, dtype=torch.uint8) / 255.0
        scores = scorer(images, prompts)
        return scores, {}

    return _fn


def hpsv2_score(device):
    from flow_grpo.hpsv2_scorer import HPSv2Scorer

    scorer = HPSv2Scorer(dtype=torch.float32, device=device)

    def _fn(images, prompts, metadata):
        if not isinstance(images, torch.Tensor):
            images = images.transpose(0, 3, 1, 2)  # NHWC -> NCHW
            images = torch.tensor(images, dtype=torch.uint8) / 255.0
        scores = scorer(images, prompts)
        return scores, {}

    return _fn


def pickscore_score(device):
    from flow_grpo.pickscore_scorer import PickScoreScorer

    scorer = PickScoreScorer(dtype=torch.float32, device=device)

    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
            images = [Image.fromarray(image) for image in images]
        scores = scorer(prompts, images)
        return scores, {}

    return _fn


def imagereward_score(device):
    from flow_grpo.imagereward_scorer import ImageRewardScorer

    scorer = ImageRewardScorer(dtype=torch.float32, device=device)

    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
            images = [Image.fromarray(image) for image in images]
        prompts = [prompt for prompt in prompts]
        scores = scorer(prompts, images)
        return scores, {}

    return _fn


def geneval_score(device):
    from flow_grpo.gen_eval import load_geneval

    batch_size = 64
    compute_geneval = load_geneval(device)

    def _fn(images, prompts, metadatas, only_strict):
        del prompts
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
        images_batched = np.array_split(images, np.ceil(len(images) / batch_size))
        metadatas_batched = np.array_split(metadatas, np.ceil(len(metadatas) / batch_size))
        all_scores = []
        all_rewards = []
        all_strict_rewards = []
        all_group_strict_rewards = []
        all_group_rewards = []
        for image_batch, metadata_batched in zip(images_batched, metadatas_batched):
            pil_images = [Image.fromarray(image) for image in image_batch]

            data = {
                "images": pil_images,
                "metadatas": list(metadata_batched),
                "only_strict": only_strict,
            }
            scores, rewards, strict_rewards, group_rewards, group_strict_rewards = compute_geneval(**data)

            all_scores += scores
            all_rewards += rewards
            all_strict_rewards += strict_rewards
            all_group_strict_rewards.append(group_strict_rewards)
            all_group_rewards.append(group_rewards)
        all_group_strict_rewards_dict = defaultdict(list)
        all_group_rewards_dict = defaultdict(list)
        for current_dict in all_group_strict_rewards:
            for key, value in current_dict.items():
                all_group_strict_rewards_dict[key].extend(value)
        all_group_strict_rewards_dict = dict(all_group_strict_rewards_dict)

        for current_dict in all_group_rewards:
            for key, value in current_dict.items():
                all_group_rewards_dict[key].extend(value)
        all_group_rewards_dict = dict(all_group_rewards_dict)

        return all_scores, all_rewards, all_strict_rewards, all_group_rewards_dict, all_group_strict_rewards_dict

    return _fn


def ocr_score(device):
    from flow_grpo.ocr import OcrScorer

    scorer = OcrScorer()

    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
        scores = scorer(images, prompts)
        # change tensor to list
        return scores, {}

    return _fn


def unifiedreward_score_sglang(device):
    import asyncio
    from openai import AsyncOpenAI
    import base64
    from io import BytesIO
    import re

    def pil_image_to_base64(image):
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        encoded_image_text = base64.b64encode(buffered.getvalue()).decode("utf-8")
        base64_qwen = f"data:image;base64,{encoded_image_text}"
        return base64_qwen

    def _extract_scores(text_outputs):
        scores = []
        pattern = r"Final Score:\s*([1-5](?:\.\d+)?)"
        for text in text_outputs:
            match = re.search(pattern, text)
            if match:
                try:
                    scores.append(float(match.group(1)))
                except ValueError:
                    scores.append(0.0)
            else:
                scores.append(0.0)
        return scores

    client = AsyncOpenAI(base_url="http://127.0.0.1:17140/v1", api_key="flowgrpo")

    async def evaluate_image(prompt, image):
        question = f"<image>\nYou are given a text caption and a generated image based on that caption. Your task is to evaluate this image based on two key criteria:\n1. Alignment with the Caption: Assess how well this image aligns with the provided caption. Consider the accuracy of depicted objects, their relationships, and attributes as described in the caption.\n2. Overall Image Quality: Examine the visual quality of this image, including clarity, detail preservation, color accuracy, and overall aesthetic appeal.\nBased on the above criteria, assign a score from 1 to 5 after 'Final Score:'.\nYour task is provided as follows:\nText Caption: [{prompt}]"
        images_base64 = pil_image_to_base64(image)
        response = await client.chat.completions.create(
            model="UnifiedReward-7b-v1.5",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": images_base64},
                        },
                        {
                            "type": "text",
                            "text": question,
                        },
                    ],
                },
            ],
            temperature=0,
        )
        return response.choices[0].message.content

    async def evaluate_batch_image(images, prompts):
        tasks = [evaluate_image(prompt, img) for prompt, img in zip(prompts, images)]
        results = await asyncio.gather(*tasks)
        return results

    def _fn(images, prompts, metadata):
        # 处理Tensor类型转换
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC

        # 转换为PIL Image并调整尺寸
        images = [Image.fromarray(image).resize((512, 512)) for image in images]

        # 执行异步批量评估
        text_outputs = asyncio.run(evaluate_batch_image(images, prompts))
        score = _extract_scores(text_outputs)
        score = [sc / 5.0 for sc in score]
        return score, {}

    return _fn

def dummy():
    def _fn(images, prompts, metadata):
        return [random.random() for _ in range(len(images))], {}
    return _fn


def multi_score(device, score_dict):
    score_functions = {
        "ocr": ocr_score,
        "imagereward": imagereward_score,
        "pickscore": pickscore_score,
        "aesthetic": aesthetic_score,
        "jpeg_compressibility": jpeg_compressibility,
        "unifiedreward": unifiedreward_score_sglang,
        "geneval": geneval_score,
        "clipscore": clip_score,
        "hpsv2": hpsv2_score,
        "mllm_score_continue": mllm_score_continue,
        "mllm_relative_api_score": mllm_relative_api_score,
        "mllm_single_api_score": mllm_single_api_score,
        "dummy": dummy
    }
    score_fns = {}
    for score_name, weight in score_dict.items():
        score_fns[score_name] = (
            score_functions[score_name](device)
            if "device" in score_functions[score_name].__code__.co_varnames
            else score_functions[score_name]()
        )

    # only_strict is only for geneval. During training, only the strict reward is needed, and non-strict rewards don't need to be computed, reducing reward calculation time.
    def _fn(images, prompts, metadata, ref_images=None, only_strict=True):
        total_scores = []
        score_details = {}
        reward_metadata = {"details": []}

        for score_name, weight in score_dict.items():
            if score_name == "geneval":
                scores, rewards, strict_rewards, group_rewards, group_strict_rewards = score_fns[score_name](
                    images, prompts, metadata, only_strict
                )
                score_details["accuracy"] = rewards
                score_details["strict_accuracy"] = strict_rewards
                for key, value in group_strict_rewards.items():
                    score_details[f"{key}_strict_accuracy"] = value
                for key, value in group_rewards.items():
                    score_details[f"{key}_accuracy"] = value
            elif score_name.startswith("mllm_"):
                scores, rewards = score_fns[score_name](ref_images, images, prompts, metadata)
            else:
                scores, rewards = score_fns[score_name](images, prompts, metadata)
            score_details[score_name] = scores
            if isinstance(rewards, dict):
                details = rewards.get("details")
                if isinstance(details, list):
                    reward_metadata.setdefault("details", []).extend(details)
                dimension_scores = rewards.get("dimension_scores")
                if isinstance(dimension_scores, dict):
                    for dim_key, dim_values in dimension_scores.items():
                        if isinstance(dim_values, (list, tuple, np.ndarray)) and len(dim_values) == len(scores):
                            score_details[dim_key] = list(dim_values)
                for meta_key in ("timeout_count", "placeholder_count"):
                    if meta_key in rewards:
                        reward_metadata[meta_key] = reward_metadata.get(meta_key, 0) + int(rewards.get(meta_key) or 0)
            weighted_scores = [weight * score for score in scores]

            if not total_scores:
                total_scores = weighted_scores
            else:
                total_scores = [total + weighted for total, weighted in zip(total_scores, weighted_scores)]

        score_details["avg"] = total_scores
        return score_details, reward_metadata

    return _fn


def main():
    import torchvision.transforms as transforms

    image_paths = [
        "test_cases/nasa.jpg",
    ]

    transform = transforms.Compose(
        [
            transforms.ToTensor(),  # Convert to tensor
        ]
    )

    images = torch.stack([transform(Image.open(image_path).convert("RGB")) for image_path in image_paths])
    prompts = [
        'A astronaut’s glove floating in zero-g with "NASA 2049" on the wrist',
    ]
    metadata = {}  # Example metadata
    score_dict = {"unifiedreward": 1.0}
    # Initialize the multi_score function with a device and score_dict
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scoring_fn = multi_score(device, score_dict)
    # Get the scores
    scores, _ = scoring_fn(images, prompts, metadata)
    # Print the scores
    print("Scores:", scores)


if __name__ == "__main__":
    main()
