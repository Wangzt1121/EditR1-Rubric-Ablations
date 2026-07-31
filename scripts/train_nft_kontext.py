# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from collections import defaultdict
import os
import datetime
from concurrent import futures
import time
import json
import io
from absl import app, flags
import logging
from diffusers import FluxKontextPipeline
from transformers.models.clip.modeling_clip import CLIPEncoderLayer
from transformers.models.t5.modeling_t5 import T5Block
import numpy as np
import flow_grpo.rewards
from flow_grpo.stat_tracking import PerPromptStatTracker
from flow_grpo.diffusers_patch.kontext_pipeline_with_logprob import (
    pipeline_with_logprob,
)
from flow_grpo.diffusers_patch.train_dreambooth_lora_flux import encode_prompt
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import wandb
from functools import partial
import tqdm
import tempfile
from PIL import Image, ImageDraw, ImageFont
from peft import LoraConfig, get_peft_model, PeftModel
import random
from torch.utils.data import Dataset, DataLoader, Sampler
from flow_grpo.ema import EMAModuleWrapper
from flow_grpo.fsdp2_utils import prepare_fsdp_model
from ml_collections import config_flags
from torch.cuda.amp import GradScaler, autocast as torch_autocast
import time

tqdm = partial(tqdm.tqdm, dynamic_ncols=True)


FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "config/base.py", "Training configuration.")

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


def setup_distributed(rank, lock_rank, world_size):
    os.environ["MASTER_ADDR"] = os.getenv("MASTER_ADDR", "localhost")
    os.environ["MASTER_PORT"] = os.getenv("MASTER_PORT", "12355")
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(lock_rank)


def cleanup_distributed():
    dist.destroy_process_group()


def is_main_process(rank):
    return rank == 0


def set_seed(seed: int, rank: int = 0):
    random.seed(seed + rank)
    np.random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    torch.cuda.manual_seed_all(seed + rank)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_candidate_generators(config, device, global_step, epoch, batch_idx, rank, batch_size, world_size):
    if not _env_flag("EDIT_R1_EXPLICIT_CANDIDATE_SEEDS", "1"):
        return None, []
    global_batch_size = max(1, int(batch_size) * max(1, int(world_size)))
    base_seed = int(getattr(config, "seed", 0))
    step_offset = int(global_step) * global_batch_size
    epoch_offset = int(epoch) * 1000003
    batch_offset = int(batch_idx) * global_batch_size
    seeds = []
    generators = []
    for local_idx in range(int(batch_size)):
        sample_index = int(rank) * int(batch_size) + local_idx
        seed = base_seed + epoch_offset + step_offset + batch_offset + sample_index
        seed = int(seed % (2**63 - 1))
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
        seeds.append(seed)
        generators.append(generator)
    return generators, seeds


def _add_lora_delta_to_linear(linear, down, up, alpha, label):
    rank = down.shape[0]
    scale = float(alpha) / float(rank) if alpha is not None else 1.0
    delta = torch.matmul(up.float(), down.float()) * scale
    if tuple(delta.shape) != tuple(linear.weight.shape):
        raise ValueError(
            f"LoRA delta shape mismatch for {label}: "
            f"delta={tuple(delta.shape)} target={tuple(linear.weight.shape)}"
        )
    linear.weight.data.add_(delta.to(device=linear.weight.device, dtype=linear.weight.dtype))


def _merge_lora_pair(transformer, state_dict, prefix, target_names):
    down_key = f"{prefix}.lora_down.weight"
    up_key = f"{prefix}.lora_up.weight"
    if down_key not in state_dict or up_key not in state_dict:
        return 0

    down = state_dict[down_key]
    up = state_dict[up_key]
    alpha = state_dict.get(f"{prefix}.alpha")
    if alpha is not None:
        alpha = alpha.item()

    if len(target_names) == 1:
        chunks = [up]
    elif len(target_names) == 4:
        chunks = list(torch.split(up, [3072, 3072, 3072, up.shape[0] - 9216], dim=0))
    else:
        chunks = list(torch.chunk(up, len(target_names), dim=0))

    if len(chunks) != len(target_names):
        raise ValueError(f"Cannot split {prefix} into {len(target_names)} target modules.")

    for target_name, up_chunk in zip(target_names, chunks):
        linear = transformer.get_submodule(target_name)
        _add_lora_delta_to_linear(linear, down, up_chunk, alpha, f"{prefix} -> {target_name}")
    return len(target_names)


def merge_diffsynth_flux_lora_into_transformer(transformer, lora_path):
    """Merge a DiffSynth FLUX/Kontext LoRA safetensors file into a diffusers transformer."""
    from safetensors.torch import load_file
    import re

    state_dict = load_file(lora_path)
    if not any(key.startswith("lora_unet_") for key in state_dict):
        missing, unexpected = transformer.load_state_dict(state_dict, strict=False)
        logger.info(
            "Loaded transformer checkpoint %s with strict=False: missing=%d unexpected=%d",
            lora_path,
            len(missing),
            len(unexpected),
        )
        return

    double_map = {
        "img_attn_proj": ["attn.to_out.0"],
        "img_attn_qkv": ["attn.to_q", "attn.to_k", "attn.to_v"],
        "img_mlp_0": ["ff.net.0.proj"],
        "img_mlp_2": ["ff.net.2"],
        "img_mod_lin": ["norm1.linear"],
        "txt_attn_proj": ["attn.to_add_out"],
        "txt_attn_qkv": ["attn.add_q_proj", "attn.add_k_proj", "attn.add_v_proj"],
        "txt_mlp_0": ["ff_context.net.0.proj"],
        "txt_mlp_2": ["ff_context.net.2"],
        "txt_mod_lin": ["norm1_context.linear"],
    }
    single_map = {
        "linear1": ["attn.to_q", "attn.to_k", "attn.to_v", "proj_mlp"],
        "linear2": ["proj_out"],
        "modulation_lin": ["norm.linear"],
    }

    applied = 0
    prefixes = sorted(
        key[: -len(".lora_down.weight")]
        for key in state_dict
        if key.endswith(".lora_down.weight")
    )
    with torch.no_grad():
        for prefix in prefixes:
            double_match = re.match(r"lora_unet_double_blocks_(\d+)_(.+)$", prefix)
            if double_match:
                block_idx, module_key = int(double_match.group(1)), double_match.group(2)
                if module_key not in double_map:
                    continue
                targets = [f"transformer_blocks.{block_idx}.{name}" for name in double_map[module_key]]
                applied += _merge_lora_pair(transformer, state_dict, prefix, targets)
                continue

            single_match = re.match(r"lora_unet_single_blocks_(\d+)_(.+)$", prefix)
            if single_match:
                block_idx, module_key = int(single_match.group(1)), single_match.group(2)
                if module_key not in single_map:
                    continue
                targets = [
                    f"single_transformer_blocks.{block_idx}.{name}"
                    for name in single_map[module_key]
                ]
                applied += _merge_lora_pair(transformer, state_dict, prefix, targets)

    if applied == 0:
        raise ValueError(f"No DiffSynth FLUX LoRA tensors were applied from {lora_path}")
    logger.info("Merged %d DiffSynth FLUX LoRA target tensors from %s", applied, lora_path)


def _env_flag(name, default="0"):
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "y"}


def _pil_to_rgb(image):
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


def pad_image_to_square_canvas(image, canvas_size):
    image = image.convert("RGB")
    original_w, original_h = image.size
    scale = min(canvas_size / float(original_w), canvas_size / float(original_h))
    resized_w = max(1, int(round(original_w * scale)))
    resized_h = max(1, int(round(original_h * scale)))
    resample = Image.Resampling.LANCZOS
    resized = image.resize((resized_w, resized_h), resample)
    fill = tuple(int(v) for v in image.resize((1, 1), resample).getpixel((0, 0)))
    canvas = Image.new("RGB", (canvas_size, canvas_size), fill)
    left = (canvas_size - resized_w) // 2
    top = (canvas_size - resized_h) // 2
    crop_box = (left, top, left + resized_w, top + resized_h)
    canvas.paste(resized, (left, top))
    return canvas, {
        "original_size": [original_w, original_h],
        "canvas_size": [canvas_size, canvas_size],
        "resized_size": [resized_w, resized_h],
        "crop_box": list(crop_box),
        "scale": scale,
    }


def crop_canvas_image_to_original(image, metadata):
    pad_info = metadata.get("_edit_r1_pad_info") if isinstance(metadata, dict) else None
    if not pad_info:
        return _pil_to_rgb(image)
    pil = _pil_to_rgb(image)
    crop_box = pad_info.get("crop_box")
    original_size = pad_info.get("original_size")
    if not crop_box or not original_size:
        return pil
    cropped = pil.crop(tuple(int(v) for v in crop_box))
    original_w, original_h = (int(original_size[0]), int(original_size[1]))
    if cropped.size != (original_w, original_h):
        cropped = cropped.resize((original_w, original_h), Image.Resampling.LANCZOS)
    return cropped


def prepare_reward_images(images, ref_images, metadatas):
    reward_images = []
    reward_ref_images = []
    for idx, metadata in enumerate(metadatas):
        metadata = metadata if isinstance(metadata, dict) else {}
        reward_images.append(crop_canvas_image_to_original(images[idx], metadata))
        original_path = metadata.get("_edit_r1_abs_image_path")
        if original_path and os.path.exists(original_path):
            reward_ref_images.append(Image.open(original_path).convert("RGB"))
        elif ref_images is not None:
            ref_idx = idx if len(ref_images) == len(images) else idx % len(ref_images)
            reward_ref_images.append(crop_canvas_image_to_original(ref_images[ref_idx], metadata))
        else:
            reward_ref_images.append(reward_images[-1])
    return reward_images, reward_ref_images


def _image_to_grid_jpeg_bytes(image, max_side=512, quality=90):
    buffer = io.BytesIO()
    pil = _pil_to_rgb(image).copy()
    pil.thumbnail((int(max_side), int(max_side)), Image.Resampling.LANCZOS)
    pil.save(buffer, format="JPEG", quality=int(quality))
    return buffer.getvalue()


def _grid_jpeg_bytes_to_image(data):
    return Image.open(io.BytesIO(data)).convert("RGB")


def _safe_filename(text, max_len=80):
    safe = []
    for char in str(text):
        if char.isalnum() or char in {"-", "_"}:
            safe.append(char)
        else:
            safe.append("_")
    value = "".join(safe).strip("_")
    return (value or "sample")[:max_len]


def _score_from_rewards(rewards, key, idx, default=None):
    if not isinstance(rewards, dict) or key not in rewards:
        return default
    value = rewards.get(key)
    try:
        if isinstance(value, torch.Tensor):
            flat = value.detach().cpu().reshape(-1)
            return float(flat[int(idx)].item()) if int(idx) < flat.numel() else default
        if isinstance(value, np.ndarray):
            flat = value.reshape(-1)
            return float(flat[int(idx)]) if int(idx) < flat.size else default
        if isinstance(value, (list, tuple)):
            return float(value[int(idx)]) if int(idx) < len(value) else default
        return float(value)
    except Exception:
        return default


def _detail_for_index(reward_metadata, idx):
    if not isinstance(reward_metadata, dict):
        return {}
    for detail in reward_metadata.get("details", []) or []:
        if isinstance(detail, dict) and int(detail.get("index", -1)) == int(idx):
            return detail
    return {}


def _grid_font(size=14, bold=False):
    font_env = os.getenv("SCORED_GRID_FONT", "").strip()
    candidates = [font_env] if font_env else []
    candidates.extend(
        [
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc" if bold else "",
            "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ]
    )
    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            try:
                return ImageFont.truetype(candidate, size=size)
            except Exception:
                continue
    return ImageFont.load_default()


def _grid_text_width(draw, text, font):
    try:
        bbox = draw.textbbox((0, 0), str(text), font=font)
        return bbox[2] - bbox[0]
    except Exception:
        return len(str(text)) * max(6, int(getattr(font, "size", 12) * 0.55))


def _wrap_grid_text(draw, text, width, max_lines=2, font=None):
    font = font or _grid_font(13)
    raw = str(text).replace("\n", " ").strip()
    if not raw:
        return []
    words = raw.split()
    if len(words) <= 1:
        chars = list(raw)
        lines = []
        current = ""
        for ch in chars:
            trial = current + ch
            if _grid_text_width(draw, trial, font) <= width or not current:
                current = trial
            else:
                lines.append(current)
                current = ch
            if len(lines) >= max_lines:
                break
        if current and len(lines) < max_lines:
            lines.append(current)
        return lines[:max_lines]

    lines = []
    current = ""
    for word in words:
        trial = word if not current else current + " " + word
        if _grid_text_width(draw, trial, font) <= width or not current:
            current = trial
        else:
            lines.append(current)
            current = word
        if len(lines) >= max_lines:
            break
    if current and len(lines) < max_lines:
        lines.append(current)
    return lines[:max_lines]


def _format_grid_score(value, digits=2):
    try:
        if value is None:
            return "n/a"
        value = float(value)
        if not np.isfinite(value):
            return "n/a"
        return f"{value:.{digits}f}"
    except Exception:
        return "n/a"


def _source_group_key(metadata, prompt):
    if not isinstance(metadata, dict):
        metadata = {}
    source = (
        metadata.get("_edit_r1_abs_image_path")
        or metadata.get("image")
        or metadata.get("source_image")
        or metadata.get("sample_id")
        or "unknown_source"
    )
    sample_id = metadata.get("sample_id") or os.path.splitext(os.path.basename(str(source)))[0]
    return str(source), str(prompt), str(sample_id)


def _short_source_label(metadata):
    if not isinstance(metadata, dict):
        return "unknown_source"
    sample_id = metadata.get("sample_id")
    image_path = metadata.get("_edit_r1_abs_image_path") or metadata.get("image") or ""
    category = metadata.get("category") or metadata.get("zh_category") or ""
    stem = os.path.splitext(os.path.basename(str(image_path)))[0] if image_path else "source"
    return str(sample_id or f"{category}_{stem}" or stem)


def save_scored_candidate_grid(config, global_step, epoch, batch_idx, rank, world_size, images, ref_images, prompts, metadatas, rewards, reward_metadata):
    if not _env_flag("SAVE_SCORED_CANDIDATE_GRIDS", "1"):
        return None
    try:
        max_side = int(os.getenv("SCORED_GRID_IMAGE_SIDE", "512"))
        jpeg_quality = int(os.getenv("SCORED_GRID_JPEG_QUALITY", "90"))
        expected_candidates = int(os.getenv("SCORED_GRID_EXPECTED_CANDIDATES", "8"))
        local_records = []
        for idx, image in enumerate(images):
            metadata = metadatas[idx] if idx < len(metadatas) and isinstance(metadatas[idx], dict) else {}
            detail = _detail_for_index(reward_metadata, idx)
            prompt = str(prompts[idx] if idx < len(prompts) else metadata.get("prompt", ""))
            source_idx = idx if idx < len(ref_images) else 0
            group_source, group_prompt, group_sample_id = _source_group_key(metadata, prompt)
            scores = {
                "avg": _score_from_rewards(rewards, "avg", idx, detail.get("score")),
                "target": _score_from_rewards(rewards, "target_edit_accuracy", idx),
                "identity": _score_from_rewards(rewards, "identity_preservation", idx),
                "non_target": _score_from_rewards(rewards, "non_target_preservation", idx),
                "color": _score_from_rewards(rewards, "color_lighting_texture_preservation", idx),
                "quality": _score_from_rewards(rewards, "photorealism_artifact_control", idx),
            }
            local_records.append(
                {
                    "rank": int(rank),
                    "local_index": int(idx),
                    "candidate_index": int(rank) * max(1, len(images)) + int(idx),
                    "prompt": prompt,
                    "sample_id": group_sample_id,
                    "source_key": group_source,
                    "group_key": (group_source, group_prompt),
                    "source_label": _short_source_label(metadata),
                    "category": str(metadata.get("category") or metadata.get("zh_category") or ""),
                    "source_jpeg": _image_to_grid_jpeg_bytes(ref_images[source_idx], max_side=max_side, quality=jpeg_quality),
                    "candidate_jpeg": _image_to_grid_jpeg_bytes(image, max_side=max_side, quality=jpeg_quality),
                    "scores": scores,
                    "reason": str(detail.get("reason", ""))[:180],
                    "warning": bool(detail.get("judge_uniform_score_warning")),
                    "cap_reason": str(detail.get("reward_saturation_cap_reason") or ""),
                }
            )

        gathered = [None for _ in range(world_size)] if dist.is_initialized() and world_size > 1 else [local_records]
        if dist.is_initialized() and world_size > 1:
            dist.all_gather_object(gathered, local_records)
        if not is_main_process(rank):
            return None

        records = []
        for part in gathered:
            if isinstance(part, list):
                records.extend(part)
        if not records:
            return None

        groups = defaultdict(list)
        for record in records:
            groups[record.get("group_key")].append(record)

        out_dir = os.path.join(
            config.save_dir,
            "scored_candidate_grids",
            f"step_{int(global_step):08d}_epoch_{int(epoch):06d}",
        )
        os.makedirs(out_dir, exist_ok=True)
        saved_paths = []

        title_font = _grid_font(16, bold=True)
        text_font = _grid_font(13)
        small_font = _grid_font(12)
        tile_w, image_h, text_h = 420, 330, 154
        margin = 12

        for group_idx, ((source_key, prompt), group_records) in enumerate(groups.items()):
            group_records = sorted(
                group_records,
                key=lambda item: (int(item.get("candidate_index", 0)), int(item.get("rank", 0)), int(item.get("local_index", 0))),
            )[:expected_candidates]
            if not group_records:
                continue

            source_record = group_records[0]
            sample_id = source_record.get("sample_id", "source")
            category = source_record.get("category", "")
            canvas = Image.new("RGB", (tile_w * 3, (image_h + text_h) * 3), "white")
            draw = ImageDraw.Draw(canvas)
            cells = [{"kind": "source", **source_record}] + group_records

            for cell_idx, record in enumerate(cells[:9]):
                row, col = divmod(cell_idx, 3)
                x = col * tile_w
                y = row * (image_h + text_h)
                draw.rectangle((x, y, x + tile_w - 1, y + image_h + text_h - 1), outline=(205, 205, 205), width=1)
                pil = _grid_jpeg_bytes_to_image(record["source_jpeg"] if cell_idx == 0 else record["candidate_jpeg"])
                pil.thumbnail((tile_w - 2 * margin, image_h - 2 * margin), Image.Resampling.LANCZOS)
                bg = Image.new("RGB", (tile_w, image_h), (246, 246, 246))
                bg.paste(pil, ((tile_w - pil.width) // 2, (image_h - pil.height) // 2))
                canvas.paste(bg, (x, y))
                tx = x + 12
                ty = y + image_h + 8
                if cell_idx == 0:
                    draw.text((tx, ty), "SOURCE", fill=(0, 0, 0), font=title_font)
                    header = f"{category} | {record.get('source_label', sample_id)} | n={len(group_records)}/{expected_candidates}"
                    draw.text((tx, ty + 22), header[:72], fill=(35, 35, 35), font=text_font)
                    for line_no, line in enumerate(_wrap_grid_text(draw, prompt, tile_w - 24, max_lines=5, font=small_font)):
                        draw.text((tx, ty + 44 + line_no * 18), line, fill=(65, 65, 65), font=small_font)
                    continue

                scores = record.get("scores", {}) or {}
                avg = scores.get("avg")
                title = f"C{cell_idx:02d}  avg={_format_grid_score(avg, 3)}"
                draw.text((tx, ty), title, fill=(0, 0, 0), font=title_font)
                metric_line = (
                    f"T={_format_grid_score(scores.get('target'))}  I={_format_grid_score(scores.get('identity'))}  "
                    f"N={_format_grid_score(scores.get('non_target'))}  C={_format_grid_score(scores.get('color'))}  "
                    f"Q={_format_grid_score(scores.get('quality'))}"
                )
                draw.text((tx, ty + 24), metric_line, fill=(30, 30, 30), font=text_font)
                detail_line = f"rank={record.get('rank')} local={record.get('local_index')} cand={record.get('candidate_index')}"
                draw.text((tx, ty + 44), detail_line, fill=(90, 90, 90), font=small_font)
                reason_y = ty + 64
                if record.get("warning"):
                    draw.text((tx, reason_y), f"WARN: {record.get('cap_reason', 'uniform_score')}"[:70], fill=(170, 50, 40), font=small_font)
                    reason_y += 18
                for line_no, line in enumerate(_wrap_grid_text(draw, record.get("reason", ""), tile_w - 24, max_lines=4, font=small_font)):
                    draw.text((tx, reason_y + line_no * 18), line, fill=(75, 75, 75), font=small_font)

            filename = (
                f"step_{int(global_step):08d}_batch_{int(batch_idx):03d}_group_{group_idx:02d}_"
                f"{_safe_filename(sample_id)}.jpg"
            )
            path = os.path.join(out_dir, filename)
            canvas.save(path, quality=92)
            saved_paths.append(path)
        return saved_paths
    except Exception as exc:
        logger.warning(f"Failed to save scored candidate grid: {exc!r}")
        return None


class PromptImageDataset(Dataset):
    def __init__(self, dataset, resolution=512, split="train"):
        self.dataset = dataset
        self.resolution = resolution
        self.pad_to_square = _env_flag("EDIT_R1_PAD_TO_SQUARE", "1")
        self.canvas_size = int(os.getenv("EDIT_R1_CANVAS_SIZE", str(resolution)))
        if self.canvas_size % 16 != 0:
            raise ValueError(f"EDIT_R1_CANVAS_SIZE must be divisible by 16, got {self.canvas_size}")
        self.file_path = os.path.join(dataset, f"{split}_metadata.jsonl")
        split_name = str(split).strip().lower()
        is_eval_split = split_name in {"test", "eval", "validation", "val"}
        prompt_metadata_file = ""
        metadata_root_env = "PROMPT_METADATA_ROOT"
        if is_eval_split:
            eval_metadata_file = os.getenv("EVAL_PROMPT_METADATA_FILE", "").strip()
            if eval_metadata_file:
                prompt_metadata_file = os.path.abspath(os.path.expanduser(eval_metadata_file))
                metadata_root_env = "EVAL_PROMPT_METADATA_ROOT"
                if not os.path.exists(prompt_metadata_file):
                    raise FileNotFoundError(
                        f"EVAL_PROMPT_METADATA_FILE does not exist: {prompt_metadata_file}"
                    )
        if not prompt_metadata_file and not (is_eval_split and os.path.exists(self.file_path)):
            prompt_metadata_file = os.getenv("PROMPT_METADATA_FILE", "").strip()
            if prompt_metadata_file:
                prompt_metadata_file = os.path.abspath(os.path.expanduser(prompt_metadata_file))
        if prompt_metadata_file and os.path.exists(prompt_metadata_file):
            with open(prompt_metadata_file, "r", encoding="utf-8") as f:
                raw_metadatas = [json.loads(line) for line in f if line.strip()]
            metadata_source = prompt_metadata_file
            self.metadata_root = os.getenv(
                metadata_root_env,
                os.path.dirname(prompt_metadata_file),
            )
        elif os.path.exists(self.file_path):
            with open(self.file_path, "r", encoding="utf-8") as f:
                raw_metadatas = [json.loads(line) for line in f if line.strip()]
            metadata_source = self.file_path
            self.metadata_root = self.dataset
        else:
            raw_metadatas = self._build_grpo_metadatas(split)
            metadata_source = os.getenv(
                "PROMPT_METADATA_FILE",
                "/nvmedata/workspace2/users/wzt/dataset/Prehuman/train_metadata_13_15_19_21_25.jsonl",
            )
            self.metadata_root = self.dataset
        metadata_limit = int(os.getenv("EDIT_R1_METADATA_LIMIT", "0") or "0")
        if metadata_limit > 0:
            raw_metadatas = raw_metadatas[:metadata_limit]
        self.metadatas = []
        self.prompts = []
        self.image_paths = []
        self.target_sizes = []
        multiple_of = 16
        max_area = resolution * resolution
        skipped = defaultdict(int)
        for item in raw_metadatas:
            image_path = item["image"]
            if os.path.isabs(str(image_path)):
                abs_image_path = str(image_path)
            else:
                abs_image_path = os.path.join(self.metadata_root, str(image_path))
                if not os.path.exists(abs_image_path):
                    dataset_candidate = os.path.join(self.dataset, str(image_path))
                    if os.path.exists(dataset_candidate):
                        abs_image_path = dataset_candidate
            try:
                with Image.open(abs_image_path) as image:
                    w, h = image.size
            except Exception:
                skipped["unreadable"] += 1
                continue
            if not self.pad_to_square:
                if h % multiple_of != 0 or w % multiple_of != 0:
                    skipped["not_divisible_by_16"] += 1
                    continue
                if h * w > max_area:
                    skipped["too_large"] += 1
                    continue
            metadata = dict(item)
            metadata["_edit_r1_abs_image_path"] = abs_image_path
            metadata["_edit_r1_original_size"] = [w, h]
            self.metadatas.append(metadata)
            self.prompts.append(item["prompt"])
            self.image_paths.append(image_path)
            if self.pad_to_square:
                self.target_sizes.append((self.canvas_size, self.canvas_size))
            else:
                self.target_sizes.append((h, w))
        if not self.metadatas:
            raise ValueError(
                f"No compatible images found in {self.dataset}."
            )
        if skipped:
            logger.warning(
                f"PromptImageDataset({split}) kept {len(self.metadatas)}/{len(raw_metadatas)} "
                f"images from {metadata_source}; skipped {dict(skipped)}."
            )
        logger.info(
            f"PromptImageDataset({split}) pad_to_square={self.pad_to_square} "
            f"canvas_size={self.canvas_size} kept={len(self.metadatas)} "
            f"metadata_source={metadata_source}"
        )

    @staticmethod
    def _task_prefix_from_text(value):
        import re

        match = re.search(r"(?<!\d)(\d{1,2})(?:[_\-]|$)", str(value or ""))
        if match:
            return f"{int(match.group(1)):02d}"
        return None

    @classmethod
    def _task_prefix_from_metadata(cls, item):
        for key in ("rubric_key", "category", "category_key", "task_key", "image"):
            prefix = cls._task_prefix_from_text(item.get(key, ""))
            if prefix:
                return prefix
        return None

    @staticmethod
    def _selected_task_prefixes():
        import re

        raw = os.getenv("EDIT_R1_TASK_PREFIXES", "").strip()
        if not raw:
            category_filter = os.getenv("CATEGORY_FILTER", "").strip()
            if category_filter and category_filter.lower() not in {"all", "*"}:
                raw = category_filter
        if raw:
            selected = []
            for token in re.split(r"[,;\s]+", raw):
                token = token.strip()
                if not token:
                    continue
                range_match = re.match(r"^(\d{1,2})-(\d{1,2})$", token)
                if range_match:
                    start, end = int(range_match.group(1)), int(range_match.group(2))
                    selected.extend(f"{idx:02d}" for idx in range(start, end + 1))
                    continue
                match = re.search(r"(?<!\d)(\d{1,2})(?:[_\-]|$)", token)
                if match:
                    selected.append(f"{int(match.group(1)):02d}")
            if selected:
                return sorted(set(selected))

        return ['13', '15', '19', '21', '25']

    def _build_grpo_metadatas(self, split):
        from pathlib import Path
        import copy

        prompt_metadata_file = os.getenv(
            "PROMPT_METADATA_FILE",
            "/nvmedata/workspace2/users/wzt/dataset/Prehuman/train_metadata_13_15_19_21_25.jsonl",
        )
        if not os.path.exists(prompt_metadata_file):
            raise FileNotFoundError(
                f"{self.file_path} does not exist and PROMPT_METADATA_FILE was not found: "
                f"{prompt_metadata_file}"
            )

        prompts_by_prefix = defaultdict(list)
        with open(prompt_metadata_file, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                prefix = self._task_prefix_from_metadata(item)
                if prefix:
                    prompts_by_prefix[prefix].append(item)

        selected_prefixes = set(self._selected_task_prefixes())
        dataset_root = Path(self.dataset)
        task_dirs = []
        for path in dataset_root.iterdir():
            if not path.is_dir():
                continue
            prefix = self._task_prefix_from_text(path.name)
            if prefix and prefix in selected_prefixes:
                task_dirs.append((prefix, path))
        task_dirs.sort(key=lambda item: int(item[0]))
        if not task_dirs:
            raise FileNotFoundError(
                f"No GRPO task folders matching prefixes {sorted(selected_prefixes)} under {self.dataset}"
            )

        image_suffixes = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
        raw_metadatas = []
        for prefix, task_dir in task_dirs:
            prompt_pool = prompts_by_prefix.get(prefix, [])
            if not prompt_pool:
                raise ValueError(
                    f"No prompt metadata with task prefix {prefix} found in {prompt_metadata_file}"
                )

            image_dir = task_dir / split
            if not image_dir.exists() and split == "train" and (task_dir / "test").exists():
                image_dir = task_dir / "test"
            if not image_dir.exists():
                image_dir = task_dir

            image_paths = sorted(
                path for path in image_dir.rglob("*")
                if path.is_file() and path.suffix.lower() in image_suffixes
            )
            for idx, image_path in enumerate(image_paths):
                prompt_meta = copy.deepcopy(prompt_pool[idx % len(prompt_pool)])
                rel_image = str(image_path.relative_to(dataset_root))
                prompt_meta["image"] = rel_image
                prompt_meta["source_image"] = rel_image
                prompt_meta["grpo_task_dir"] = task_dir.name
                prompt_meta["grpo_image_index"] = idx
                prompt_meta["prompt_metadata_file"] = prompt_metadata_file
                prompt_meta["sample_id"] = (
                    prompt_meta.get("sample_id")
                    or f"{prefix}_{task_dir.name}_grpo_{idx:06d}"
                )
                raw_metadatas.append(prompt_meta)

        logger.info(
            f"Built {len(raw_metadatas)} GRPO metadata rows from {self.dataset}; "
            f"task_prefixes={sorted(selected_prefixes)}; prompt_metadata={prompt_metadata_file}"
        )
        return raw_metadatas

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        metadata = dict(self.metadatas[idx])
        item = {"prompt": self.prompts[idx], "metadata": metadata}
        # Assuming 'image' in metadata contains a path to the image file
        image_path = metadata["image"]
        abs_image_path = metadata.get("_edit_r1_abs_image_path")
        if not abs_image_path:
            abs_image_path = image_path if os.path.isabs(str(image_path)) else os.path.join(self.metadata_root, str(image_path))
        item["prompt_with_image_path"] = f"{self.prompts[idx]}_{image_path}"
        image = Image.open(abs_image_path).convert("RGB")
        w, h = image.size
        if self.pad_to_square:
            image, pad_info = pad_image_to_square_canvas(image, self.canvas_size)
            metadata["_edit_r1_pad_info"] = pad_info
            item["target_size"] = (self.canvas_size, self.canvas_size)
        else:
            metadata["_edit_r1_pad_info"] = {
                "original_size": [w, h],
                "canvas_size": [w, h],
                "resized_size": [w, h],
                "crop_box": [0, 0, w, h],
                "scale": 1.0,
            }
            item["target_size"] = (h, w)
        item["metadata"] = metadata
        item["image"] = image
        return item

    @staticmethod
    def collate_fn(examples):
        prompts = [example["prompt"] for example in examples]
        metadatas = [example["metadata"] for example in examples]
        images = [example["image"] for example in examples]
        target_sizes = [example["target_size"] for example in examples]
        if len(set(target_sizes)) != 1:
            raise ValueError(
                f"Mixed image sizes in one batch are not supported: {sorted(set(target_sizes))}. "
                "Use size-bucketed sampling or batch_size=1 for evaluation."
            )
        prompt_with_image_paths = [
            example["prompt_with_image_path"] for example in examples
        ]
        return prompts, metadatas, images, prompt_with_image_paths, target_sizes[0]


class DistributedKRepeatSampler(Sampler):
    def __init__(
        self, dataset, batch_size, k, num_replicas, rank, seed=0, banned_prompts=None
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.k = k  # k means the number of images per prompt
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed
        self.total_samples = self.num_replicas * self.batch_size
        assert (
            self.total_samples % self.k == 0
        ), f"k can not div n*b, k{k}-num_replicas{num_replicas}-batch_size{batch_size}"
        self.m = self.total_samples // self.k
        self.epoch = 0

        self.banned_prompts = banned_prompts if banned_prompts is not None else set()
        self.last_banned_prompts_len = len(self.banned_prompts)
        self.valid_indices_cache = None

    def get_valid_indices(self):
        start_time = time.time()
        if self.valid_indices_cache is None or self.last_banned_prompts_len != len(self.banned_prompts):
            self.valid_indices_cache = [
                i for i, prompt in enumerate(self.dataset.prompts)
                if prompt not in self.banned_prompts
            ]
            self.last_banned_prompts_len = len(self.banned_prompts)
        print("get valid indices time: ", time.time() - start_time)
        return self.valid_indices_cache

    def __iter__(self):
        while True:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)

            valid_indices = self.get_valid_indices()
            if _env_flag("EDIT_R1_SEQUENTIAL_ONCE", "0"):
                if not valid_indices:
                    raise ValueError("No valid dataset indices available for sequential sampling.")
                start = (int(self.epoch) * int(self.m)) % len(valid_indices)
                positions = [(start + offset) % len(valid_indices) for offset in range(self.m)]
                indices = [valid_indices[pos] for pos in positions]
            else:
                valid_by_size = defaultdict(list)
                for idx in valid_indices:
                    valid_by_size[self.dataset.target_sizes[idx]].append(idx)

                eligible_sizes = [
                    size for size, indices_for_size in valid_by_size.items()
                    if len(indices_for_size) >= self.m
                ]
                if eligible_sizes:
                    size_idx = torch.randint(
                        len(eligible_sizes), (1,), generator=g
                    ).item()
                    selected_pool = valid_by_size[eligible_sizes[size_idx]]
                    indices = torch.tensor(selected_pool)[
                        torch.randperm(len(selected_pool), generator=g)[: self.m]
                    ].tolist()
                else:
                    # If a rare size bucket has fewer prompts than a full distributed
                    # batch needs, keep the batch same-sized and sample with replacement.
                    largest_size, selected_pool = max(
                        valid_by_size.items(), key=lambda item: len(item[1])
                    )
                    replacement_positions = torch.randint(
                        len(selected_pool), (self.m,), generator=g
                    ).tolist()
                    indices = [selected_pool[pos] for pos in replacement_positions]

            repeated_indices = [idx for idx in indices for _ in range(self.k)]
            shuffled_indices = torch.randperm(
                len(repeated_indices), generator=g
            ).tolist()
            shuffled_samples = [repeated_indices[i] for i in shuffled_indices]

            per_card_samples = []
            for i in range(self.num_replicas):
                start = i * self.batch_size
                end = start + self.batch_size
                per_card_samples.append(shuffled_samples[start:end])
            yield per_card_samples[self.rank]

    def set_epoch(self, epoch):
        self.epoch = epoch


def gather_tensor_to_all(tensor, world_size):
    gathered_tensors = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered_tensors, tensor)
    return torch.cat(gathered_tensors, dim=0).cpu()


def compute_text_embeddings(
    prompt, text_encoders, tokenizers, max_sequence_length, device
):
    with torch.no_grad():
        prompt_embeds, pooled_prompt_embeds, text_ids = encode_prompt(
            text_encoders, tokenizers, prompt, max_sequence_length
        )
        prompt_embeds = prompt_embeds.to(device)
        pooled_prompt_embeds = pooled_prompt_embeds.to(device)
    return prompt_embeds, pooled_prompt_embeds, text_ids


def return_decay(step, decay_type):
    if decay_type == 0:
        flat = 0
        uprate = 0.0
        uphold = 0.0
    elif decay_type == 1:
        flat = 0
        uprate = 0.001
        uphold = 0.5
    elif decay_type == 2:
        flat = 75
        uprate = 0.0075
        uphold = 0.999
    else:
        assert False

    if step < flat:
        return 0.0
    else:
        decay = (step - flat) * uprate
        return min(decay, uphold)


def calculate_zero_std_ratio(prompts, gathered_rewards):
    prompt_array = np.array(prompts)
    unique_prompts, inverse_indices, counts = np.unique(
        prompt_array, return_inverse=True, return_counts=True
    )
    grouped_rewards = gathered_rewards["avg"][np.argsort(inverse_indices), 0]
    split_indices = np.cumsum(counts)[:-1]
    reward_groups = np.split(grouped_rewards, split_indices)
    prompt_std_devs = np.array([np.std(group) for group in reward_groups])
    zero_std_count = np.count_nonzero(prompt_std_devs == 0)
    zero_std_ratio = zero_std_count / len(prompt_std_devs)
    return zero_std_ratio, prompt_std_devs.mean()


def _flatten_reward_values(values):
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim > 1:
        arr = arr[:, 0]
    arr = arr.reshape(-1)
    return arr[np.isfinite(arr)]


def _aggregate_reward_rows(rows):
    grouped = defaultdict(list)
    for row in rows:
        if "global_step" in row:
            grouped[int(row["global_step"])].append(row)

    aggregated = []
    for step in sorted(grouped):
        group = grouped[step]
        out = {"global_step": step}
        numeric_keys = sorted(
            {
                key
                for item in group
                for key, value in item.items()
                if key != "global_step" and isinstance(value, (int, float))
            }
        )
        for key in numeric_keys:
            values = [float(item[key]) for item in group if key in item]
            if values:
                out[key] = float(np.mean(values))
        aggregated.append(out)
    return aggregated


def _reward_curve_font(size, bold=False):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
        "/usr/share/fonts/dejavu/DejaVuSerif-Bold.ttf" if bold else "/usr/share/fonts/dejavu/DejaVuSerif.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size=size)
            except Exception:
                pass
    return ImageFont.load_default()


def _text_size(draw, text, font):
    try:
        box = draw.textbbox((0, 0), str(text), font=font)
        return box[2] - box[0], box[3] - box[1]
    except Exception:
        return len(str(text)) * 8, 14


def _center_text(draw, xy, text, font, fill=(0, 0, 0)):
    x, y = xy
    w, h = _text_size(draw, text, font)
    draw.text((int(x - w / 2), int(y - h / 2)), text, fill=fill, font=font)


def _draw_rotated_text(img, xy, text, font, fill=(0, 0, 0), angle=90):
    text = str(text)
    probe = ImageDraw.Draw(Image.new("RGBA", (1, 1), (255, 255, 255, 0)))
    w, h = _text_size(probe, text, font)
    layer = Image.new("RGBA", (w + 12, h + 12), (255, 255, 255, 0))
    layer_draw = ImageDraw.Draw(layer)
    layer_draw.text((6, 6), text, fill=fill + (255,), font=font)
    layer = layer.rotate(angle, expand=True)
    x, y = xy
    img.paste(layer, (int(x - layer.width / 2), int(y - layer.height / 2)), layer)


def _format_axis_value(value):
    value = float(value)
    if abs(value) >= 100:
        return f"{value:.0f}"
    if abs(value) >= 10:
        return f"{value:.1f}"
    return f"{value:.2f}"


def _format_curve_point_value(value):
    value = float(value)
    if not np.isfinite(value):
        return "nan"
    if value == 0.0:
        return "0"
    if abs(value) < 0.01:
        return f"{value:.2e}"
    if abs(value) < 1.0:
        return f"{value:.3f}"
    if abs(value) < 10.0:
        return f"{value:.2f}"
    return f"{value:.1f}"


def _draw_curve_value_label(draw, box, x, y, value, font, color, offset):
    label = _format_curve_point_value(value)
    tw, th = _text_size(draw, label, font)
    dx, dy = offset
    tx = int(x + dx)
    ty = int(y + dy)
    left, top, right, bottom = box
    tx = max(left + 4, min(right - tw - 8, tx))
    ty = max(top + 4, min(bottom - th - 6, ty))
    pad_x, pad_y = 3, 2
    draw.rounded_rectangle(
        (tx - pad_x, ty - pad_y, tx + tw + pad_x, ty + th + pad_y),
        radius=3,
        fill=(255, 255, 255),
        outline=(220, 220, 220),
        width=1,
    )
    draw.text((tx, ty), label, fill=color, font=font)


def _plot_points(
    draw,
    rows,
    key,
    box,
    x_min,
    x_max,
    y_min,
    y_max,
    color,
    width=3,
    marker=4,
    smooth=False,
    annotate_values=False,
    value_font=None,
    label_offset=(6, -18),
):
    left, top, right, bottom = box
    plot_w = right - left
    plot_h = bottom - top
    points = []
    label_points = []
    for row in rows:
        value = row.get(key)
        if not isinstance(value, (int, float)) or not np.isfinite(float(value)):
            continue
        x = int(left + (int(row["global_step"]) - x_min) / max(1, x_max - x_min) * plot_w)
        y = int(bottom - (float(value) - y_min) / max(1e-12, y_max - y_min) * plot_h)
        y = max(top, min(bottom, y))
        points.append((x, y))
        label_points.append((x, y, float(value)))
    if len(points) >= 2:
        draw.line(points, fill=color, width=width)
    if marker:
        for point in points:
            r = marker
            if smooth:
                draw.ellipse((point[0] - r, point[1] - r, point[0] + r, point[1] + r), outline=color, width=2)
            else:
                draw.ellipse((point[0] - r, point[1] - r, point[0] + r, point[1] + r), fill=color)
    if annotate_values and value_font is not None:
        for x, y, value in label_points:
            _draw_curve_value_label(draw, box, x, y, value, value_font, color, label_offset)
    return points


def _draw_academic_axes(img, draw, box, x_min, x_max, y_min, y_max, xlabel, ylabel, font_sm, font_md):
    left, top, right, bottom = box
    axis_color = (35, 35, 35)
    grid_color = (226, 226, 226)
    draw.rectangle((left, top, right, bottom), outline=axis_color, width=2)

    for frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
        y = int(bottom - frac * (bottom - top))
        draw.line((left, y, right, y), fill=grid_color, width=1)
        value = y_min + frac * (y_max - y_min)
        draw.text((left - 74, y - 9), _format_axis_value(value), fill=(60, 60, 60), font=font_sm)

    span = int(x_max) - int(x_min)
    if span > 6:
        tick_steps = np.linspace(int(x_min), int(x_max), 6).round().astype(int).tolist()
    else:
        tick_steps = list(range(int(x_min), int(x_max) + 1))
    last_x = None
    for step in sorted(set(tick_steps)):
        x = int(left + (step - x_min) / max(1, x_max - x_min) * (right - left))
        if last_x is not None and abs(x - last_x) < 50:
            continue
        last_x = x
        draw.line((x, bottom, x, bottom + 7), fill=axis_color, width=1)
        label = str(step)
        tw, _ = _text_size(draw, label, font_sm)
        draw.text((x - tw // 2, bottom + 14), label, fill=(60, 60, 60), font=font_sm)

    _center_text(draw, ((left + right) / 2, bottom + 58), xlabel, font_md, fill=(25, 25, 25))
    _draw_rotated_text(img, (left - 104, (top + bottom) / 2), ylabel, font_md, fill=(25, 25, 25), angle=90)


def _plot_local_reward_curve(metrics_path, image_path):
    os.makedirs(os.path.dirname(image_path), exist_ok=True)
    rows, latest_run_name = _prepare_curve_rows(metrics_path)
    width, height = 1700, 760
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    font_sm = _reward_curve_font(17)
    font_md = _reward_curve_font(21)
    font_lg = _reward_curve_font(28, bold=True)
    font_title = _reward_curve_font(34, bold=True)

    draw.line((45, 18, width - 45, 18), fill=(0, 0, 0), width=3)
    draw.line((45, height - 22, width - 45, height - 22), fill=(0, 0, 0), width=2)
    _center_text(draw, (width / 2, 50), "Reward Optimization Diagnostics", font_title)
    subtitle = "Global-step trends; raw train reward is noisy, EMA/rolling curves show optimization direction"
    if latest_run_name:
        subtitle += f" | run: {latest_run_name}"
    _center_text(draw, (width / 2, 86), subtitle[:150], font_sm, fill=(70, 70, 70))

    if not rows:
        _center_text(draw, (width / 2, height / 2), "No reward metrics have been written yet.", font_md)
        img.save(image_path)
        return

    ema_alpha = float(os.getenv("REWARD_CURVE_EMA_ALPHA", "0.1"))
    raw_window = int(os.getenv("REWARD_CURVE_ROLLING_WINDOW", "10"))
    _add_curve_smoothing(rows, "reward_avg_mean", "reward_avg_ema", "reward_avg_rolling", ema_alpha, raw_window)

    steps = [int(row["global_step"]) for row in rows if "global_step" in row]
    unique_steps = sorted(set(steps))
    latest = rows[-1]
    x_min, x_max = min(steps), max(steps)
    if x_min == x_max:
        x_max = x_min + 1

    panel_a = (115, 145, 795, 540)
    panel_b = (955, 145, 1635, 540)
    _center_text(draw, ((panel_a[0] + panel_a[2]) / 2, 116), "(a) Train Reward", font_lg)
    _center_text(draw, ((panel_b[0] + panel_b[2]) / 2, 116), "(b) Reward Dimensions", font_lg)
    _draw_academic_axes(img, draw, panel_a, x_min, x_max, 0.0, 1.0, "Global step", "Training reward score (0-1)", font_sm, font_md)
    _draw_academic_axes(img, draw, panel_b, x_min, x_max, 0.0, 1.0, "Global step", "Reward dimension score (0-1)", font_sm, font_md)

    if len(unique_steps) < 2:
        msg = "Not enough training steps to plot reward trend. Need at least 2 global steps."
        draw.text((panel_a[0] + 18, panel_a[1] + 18), msg, fill=(170, 70, 0), font=font_md)
    elif unique_steps == [0]:
        msg = "Only step 0 is available; this is a smoke-test snapshot, not a training trend."
        draw.text((panel_a[0] + 18, panel_a[1] + 18), msg, fill=(170, 70, 0), font=font_md)

    main_series = [
        ("reward_avg_mean", "raw reward", (90, 143, 190), 2, 3, False),
        ("reward_avg_ema", "EMA reward", (213, 94, 0), 5, 0, True),
        ("reward_avg_rolling", f"rolling mean ({raw_window})", (70, 70, 70), 3, 0, True),
    ]
    for key, _, color, line_w, marker, smooth in main_series:
        _plot_points(draw, rows, key, panel_a, x_min, x_max, 0.0, 1.0, color, width=line_w, marker=marker, smooth=smooth)

    dim_series = [
        ("reward_target_edit_accuracy_mean", "target", (213, 94, 0)),
        ("reward_identity_preservation_mean", "identity", (0, 114, 178)),
        ("reward_non_target_preservation_mean", "non-target", (0, 158, 115)),
        ("reward_color_lighting_texture_preservation_mean", "color/texture", (204, 121, 167)),
        ("reward_photorealism_artifact_control_mean", "quality", (80, 80, 80)),
    ]
    for key, _, color in dim_series:
        _plot_points(draw, rows, key, panel_b, x_min, x_max, 0.0, 1.0, color, width=3, marker=2, smooth=True)

    legend_y = 612
    legend_x = panel_a[0]
    for idx, (_, label, color, line_w, _, _) in enumerate(main_series):
        x = legend_x + idx * 205
        draw.line((x, legend_y + 10, x + 34, legend_y + 10), fill=color, width=line_w)
        draw.text((x + 44, legend_y), label, fill=(25, 25, 25), font=font_sm)

    dim_legend_x = panel_b[0]
    for idx, (_, label, color) in enumerate(dim_series):
        x = dim_legend_x + (idx % 3) * 215
        y = legend_y + (idx // 3) * 28
        draw.line((x, y + 10, x + 34, y + 10), fill=color, width=3)
        draw.text((x + 44, y), label, fill=(25, 25, 25), font=font_sm)

    latest_items = []
    for key, label in [
        ("reward_avg_mean", "raw"),
        ("reward_avg_ema", "ema"),
        ("reward_avg_rolling", "rolling"),
        ("reward_avg_std", "reward_std"),
        ("zero_std_ratio", "zero_std_ratio"),
        ("reward_std_mean", "group_reward_std"),
        ("reward_timeout_count", "timeouts"),
        ("reward_parse_failure_count", "parse_fail"),
        ("reward_missing_key_count", "missing_keys"),
        ("reward_fallback_score_count", "fallbacks"),
    ]:
        if key in latest:
            latest_items.append(f"{label}={float(latest[key]):.4g}")
    latest_text = f"Latest step={latest.get('global_step', 0)}, epoch={latest.get('epoch', 0):.0f}; " + ", ".join(latest_items[:8])
    draw.text((115, 680), latest_text[:190], fill=(45, 45, 45), font=font_sm)
    note = "Interpretation: optimize from EMA/rolling and fixed validation reward, not from one noisy raw training step."
    draw.text((115, 710), note, fill=(75, 75, 75), font=font_sm)
    img.save(image_path)


def _read_jsonl_rows(path):
    rows = []
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except Exception:
                    continue
    return rows


def _rows_by_latest_contiguous_run_chain(rows):
    """Select the latest resume chain instead of only the latest run.

    A resumed training job writes a new run_name but keeps increasing global_step.
    Plotting only the latest run makes curves start at the resume point, while
    plotting every row can mix old smoke tests with duplicate step 0/1 records.
    This function walks backward over run_name blocks and keeps only the newest
    non-overlapping, step-contiguous chain, e.g. 0-3 + 4-7 + 8-9.
    """
    if not rows:
        return rows, None

    mode = os.getenv("REWARD_CURVE_RUN_MODE", "chain").strip().lower()
    if mode in {"all", "all_runs"}:
        return rows, "all_runs"

    latest_run_name = rows[-1].get("run_name")
    if not latest_run_name or mode in {"latest", "latest_run"}:
        if latest_run_name:
            rows = [row for row in rows if row.get("run_name") == latest_run_name]
        return rows, latest_run_name

    run_order = []
    grouped = defaultdict(list)
    for row in rows:
        run_name = row.get("run_name") or "<none>"
        if run_name not in grouped:
            run_order.append(run_name)
        grouped[run_name].append(row)

    selected_names = []
    selected_rows = []
    current_min_step = None
    for run_name in reversed(run_order):
        run_rows = grouped.get(run_name, [])
        steps = [int(row["global_step"]) for row in run_rows if "global_step" in row]
        if not steps:
            continue
        min_step = min(steps)
        if current_min_step is None:
            selected_names.append(run_name)
            selected_rows.extend(run_rows)
            current_min_step = min_step
            continue

        # Keep only the non-overlapping prefix of the previous run. This handles
        # resume jobs that both record the boundary eval step, e.g. previous run
        # has 4/6/8 and latest run has 8/10: keep 4/6 from previous and 8/10 from latest.
        prefix_rows = [
            row for row in run_rows
            if "global_step" in row and int(row["global_step"]) < current_min_step
        ]
        if prefix_rows:
            selected_names.append(run_name)
            selected_rows.extend(prefix_rows)
            current_min_step = min(int(row["global_step"]) for row in prefix_rows)

    selected_rows = sorted(
        selected_rows,
        key=lambda row: (int(row.get("global_step", 0)), float(row.get("timestamp", 0.0) or 0.0)),
    )
    if len(selected_names) <= 1:
        label = latest_run_name
    else:
        label = f"resume_chain:{selected_names[-1]}..{selected_names[0]}"
    return selected_rows, label


def _latest_run_rows(rows):
    return _rows_by_latest_contiguous_run_chain(rows)


def _prepare_curve_rows(path):
    rows, run_name = _latest_run_rows(_read_jsonl_rows(path))
    return _aggregate_reward_rows(rows), run_name


def _add_curve_smoothing(rows, key, ema_key, rolling_key, alpha=0.1, window=10):
    ema_value = None
    rolling_values = []
    for row in rows:
        value = row.get(key)
        if not isinstance(value, (int, float)) or not np.isfinite(float(value)):
            continue
        value = float(value)
        ema_value = value if ema_value is None else (1.0 - alpha) * ema_value + alpha * value
        rolling_values.append(value)
        row[ema_key] = float(ema_value)
        row[rolling_key] = float(np.mean(rolling_values[-window:]))


def _draw_monitor_curve(
    rows,
    image_path,
    title,
    subtitle,
    series,
    latest_keys=None,
    y_min=None,
    y_max=None,
    footnote=None,
    x_label="Global step",
    y_label="Metric value",
):
    os.makedirs(os.path.dirname(image_path), exist_ok=True)
    width, height = 1500, 780
    box = (120, 145, 1090, 610)
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    font_sm = _reward_curve_font(17)
    font_md = _reward_curve_font(21)
    font_lg = _reward_curve_font(30, bold=True)
    font_title = _reward_curve_font(34, bold=True)

    draw.line((45, 18, width - 45, 18), fill=(0, 0, 0), width=3)
    draw.line((45, height - 22, width - 45, height - 22), fill=(0, 0, 0), width=2)
    _center_text(draw, (width / 2, 48), title, font_title)
    if subtitle:
        _center_text(draw, (width / 2, 86), subtitle[:145], font_sm, fill=(70, 70, 70))

    if not rows:
        _center_text(draw, (width / 2, height / 2), "No metrics have been written yet.", font_md)
        img.save(image_path)
        return

    steps = [int(row["global_step"]) for row in rows if "global_step" in row]
    if not steps:
        _center_text(draw, (width / 2, height / 2), "No global_step values found in metrics.", font_md)
        img.save(image_path)
        return
    x_min, x_max = min(steps), max(steps)
    if x_min == x_max:
        x_max = x_min + 1

    active_series = [
        item for item in series
        if any(item[0] in row and isinstance(row.get(item[0]), (int, float)) for row in rows)
    ]
    values = []
    for key, _, _, _ in active_series:
        for row in rows:
            value = row.get(key)
            if isinstance(value, (int, float)) and np.isfinite(float(value)):
                values.append(float(value))
    if values:
        auto_min = min(values)
        auto_max = max(values)
        span = max(auto_max - auto_min, 1e-6)
        low = auto_min - 0.12 * span if y_min is None else y_min
        high = auto_max + 0.12 * span if y_max is None else y_max
    else:
        low = 0.0 if y_min is None else y_min
        high = 1.0 if y_max is None else y_max
    if abs(high - low) < 1e-8:
        high = low + 1.0

    _draw_academic_axes(img, draw, box, x_min, x_max, low, high, x_label, y_label, font_sm, font_md)
    unique_steps = sorted(set(steps))
    if len(unique_steps) < 2:
        msg = "Not enough training steps to plot a trend. Need at least 2 global steps."
        draw.text((box[0] + 18, box[1] + 18), msg, fill=(170, 70, 0), font=font_md)

    legend_x = 1145
    legend_y = box[1] + 8
    draw.text((legend_x, legend_y - 38), "Legend", fill=(0, 0, 0), font=font_lg)
    annotate_values = _env_flag("REWARD_CURVE_ANNOTATE_VALUES", "1")
    value_font = _reward_curve_font(int(os.getenv("REWARD_CURVE_VALUE_FONT_SIZE", "11")))
    value_offsets = [
        (6, -20),
        (6, 8),
        (-54, -20),
        (-54, 8),
        (14, -34),
        (14, 22),
        (-66, -34),
        (-66, 22),
    ]
    for idx, (key, label, color, line_w) in enumerate(active_series):
        marker = 3 if idx == 0 else (2 if annotate_values else 0)
        _plot_points(
            draw,
            rows,
            key,
            box,
            x_min,
            x_max,
            low,
            high,
            color,
            width=line_w,
            marker=marker,
            smooth=idx != 0,
            annotate_values=annotate_values,
            value_font=value_font,
            label_offset=value_offsets[idx % len(value_offsets)],
        )
        y = legend_y + idx * 31
        draw.line((legend_x, y + 11, legend_x + 36, y + 11), fill=color, width=line_w)
        draw.text((legend_x + 48, y), label, fill=(25, 25, 25), font=font_sm)

    latest = rows[-1]
    latest_y = legend_y + max(155, len(active_series) * 31 + 35)
    draw.text((legend_x, latest_y), "Latest", fill=(0, 0, 0), font=font_lg)
    latest_lines = [
        f"step: {latest.get('global_step', 0)}",
        f"epoch: {latest.get('epoch', 0):.0f}" if "epoch" in latest else None,
    ]
    if latest_keys:
        for key, label in latest_keys:
            value = latest.get(key)
            if isinstance(value, (int, float)) and np.isfinite(float(value)):
                latest_lines.append(f"{label}: {float(value):.4g}")
    for idx, line in enumerate([line for line in latest_lines if line]):
        draw.text((legend_x, latest_y + 40 + idx * 24), line, fill=(35, 35, 35), font=font_sm)

    if footnote:
        draw.text((120, 702), footnote[:150], fill=(70, 70, 70), font=font_sm)
    img.save(image_path)


def _merge_metric_rows(*row_groups):
    by_step = {}
    for rows in row_groups:
        for row in rows:
            if "global_step" not in row:
                continue
            step = int(row["global_step"])
            by_step.setdefault(step, {"global_step": step}).update(row)
    return [by_step[step] for step in sorted(by_step)]


def _plot_reward_train_eval_curve(metrics_path, eval_metrics_path, image_path):
    del metrics_path
    eval_rows, eval_run_name = _prepare_curve_rows(eval_metrics_path)
    alpha = float(os.getenv("REWARD_CURVE_EMA_ALPHA", "0.1"))
    window = int(os.getenv("REWARD_CURVE_ROLLING_WINDOW", "10"))
    _add_curve_smoothing(eval_rows, "eval_reward_mean", "eval_reward_ema", "eval_reward_rolling", alpha, window)
    subtitle = "fixed validation reward over global step"
    if eval_run_name:
        subtitle += f" | run: {eval_run_name}"
    if not eval_rows:
        subtitle += " | no eval_metrics.jsonl yet: set SKIP_EVAL=0 and EVAL_FREQ=1"
    series = [
        ("eval_reward_mean", "fixed eval raw", (44, 160, 44), 4),
        ("eval_reward_ema", "fixed eval EMA", (232, 126, 4), 5),
        ("eval_reward_rolling", f"fixed eval rolling {window}", (85, 85, 85), 3),
    ]
    latest_keys = [
        ("eval_reward_mean", "eval_raw"),
        ("eval_reward_ema", "eval_ema"),
        ("eval_reward_rolling", "eval_roll"),
        ("eval_reward_std", "eval_std"),
    ]
    _draw_monitor_curve(
        eval_rows,
        image_path,
        "01 Fixed Eval Reward Curve",
        subtitle,
        series,
        latest_keys=latest_keys,
        y_min=0.0,
        y_max=1.0,
        footnote="This is the formal reward curve: same fixed eval set, same metric, plotted by global step.",
        y_label="Fixed eval reward score (0-1)",
    )


def _plot_loss_policy_total_curve(metrics_path, image_path):
    rows, run_name = _prepare_curve_rows(metrics_path)
    subtitle = "policy / total / KL loss"
    if run_name:
        subtitle += f" | run: {run_name}"
    series = [
        ("policy_loss", "policy_loss", (48, 113, 181), 3),
        ("total_loss", "total_loss", (214, 39, 40), 4),
        ("kl_div_loss", "kl_div_loss", (232, 126, 4), 3),
        ("unweighted_policy_loss", "unweighted_policy", (85, 85, 85), 2),
    ]
    latest_keys = [
        ("policy_loss", "policy"),
        ("total_loss", "total"),
        ("kl_div_loss", "kl_loss"),
        ("unweighted_policy_loss", "unweighted"),
        ("learning_rate", "lr"),
    ]
    _draw_monitor_curve(
        rows,
        image_path,
        "(b) Loss / Policy / Total",
        subtitle,
        series,
        latest_keys=latest_keys,
        y_label="Loss value",
    )


def _plot_kl_clip_fraction_curve(metrics_path, image_path):
    rows, run_name = _prepare_curve_rows(metrics_path)
    subtitle = "KL and clipped-advantage fraction"
    if run_name:
        subtitle += f" | run: {run_name}"
    series = [
        ("kl_div", "kl_div", (48, 113, 181), 3),
        ("old_kl_div", "old_kl_div", (148, 103, 189), 3),
        ("kl_div_loss", "kl_div_loss", (232, 126, 4), 3),
        ("advantage_clip_fraction", "clip_fraction", (214, 39, 40), 4),
        ("old_deviate", "policy_change", (85, 85, 85), 3),
    ]
    latest_keys = [
        ("kl_div", "kl"),
        ("old_kl_div", "old_kl"),
        ("kl_div_loss", "kl_loss"),
        ("advantage_clip_fraction", "clip_frac"),
        ("old_deviate", "policy_change"),
        ("valid_fraction", "valid_frac"),
    ]
    _draw_monitor_curve(
        rows,
        image_path,
        "(c) KL / Clip Fraction",
        subtitle,
        series,
        latest_keys=latest_keys,
        y_min=0.0,
        y_label="KL divergence / fraction",
    )


def _plot_grad_reward_signal_curve(training_metrics_path, reward_metrics_path, image_path):
    train_rows, train_run = _prepare_curve_rows(training_metrics_path)
    reward_rows, reward_run = _prepare_curve_rows(reward_metrics_path)
    rows = _merge_metric_rows(reward_rows, train_rows)
    subtitle = "reward spread + advantage signal + grad norm ratio"
    run_name = train_run or reward_run
    if run_name:
        subtitle += f" | run: {run_name}"
    series = [
        ("reward_avg_std", "reward_avg_std", (48, 113, 181), 3),
        ("reward_std_mean", "group_reward_std", (44, 160, 44), 3),
        ("zero_std_ratio", "zero_std_ratio", (214, 39, 40), 4),
        ("advantage_abs_mean", "adv_abs_mean", (148, 103, 189), 3),
        ("advantage_std", "adv_std", (140, 86, 75), 3),
        ("grad_norm_clip_ratio", "grad_norm/max", (85, 85, 85), 3),
    ]
    latest_keys = [
        ("reward_avg_std", "reward_std"),
        ("reward_std_mean", "group_std"),
        ("zero_std_ratio", "zero_std"),
        ("advantage_abs_mean", "adv_abs"),
        ("advantage_std", "adv_std"),
        ("grad_norm", "grad_norm"),
        ("grad_norm_clip_ratio", "grad/max"),
    ]
    _draw_monitor_curve(
        rows,
        image_path,
        "(d) Grad / Reward Signal",
        subtitle,
        series,
        latest_keys=latest_keys,
        y_min=0.0,
        y_label="Signal magnitude",
    )


def _plot_eval_reward_dimensions_curve(metrics_path, eval_metrics_path, image_path):
    del metrics_path
    eval_rows, eval_run_name = _prepare_curve_rows(eval_metrics_path)
    subtitle = "fixed eval reward dimensions"
    if eval_run_name:
        subtitle += f" | run: {eval_run_name}"
    has_eval_dims = any(
        key.startswith("eval_reward_") and key.endswith("_mean") and key != "eval_reward_mean"
        for row in eval_rows
        for key in row
    )
    if not eval_rows or not has_eval_dims:
        subtitle += " | unavailable for scalar reward or eval not recorded"
    prefix = "eval_reward"
    series = [
        (f"{prefix}_target_edit_accuracy_mean", "target edit", (44, 160, 44), 3),
        (f"{prefix}_identity_preservation_mean", "identity", (148, 103, 189), 3),
        (f"{prefix}_non_target_preservation_mean", "non-target", (140, 86, 75), 3),
        (f"{prefix}_color_lighting_texture_preservation_mean", "color/texture", (214, 39, 40), 3),
        (f"{prefix}_photorealism_artifact_control_mean", "quality", (85, 85, 85), 3),
    ]
    latest_keys = [
        (f"{prefix}_target_edit_accuracy_mean", "target"),
        (f"{prefix}_identity_preservation_mean", "identity"),
        (f"{prefix}_non_target_preservation_mean", "non_target"),
        (f"{prefix}_color_lighting_texture_preservation_mean", "color"),
        (f"{prefix}_photorealism_artifact_control_mean", "quality"),
    ]
    _draw_monitor_curve(
        eval_rows,
        image_path,
        "02 Reward Dimension Curve",
        subtitle,
        series,
        latest_keys=latest_keys,
        y_min=0.0,
        y_max=1.0,
        footnote="Shown only when the reward backend returns dimension scores, e.g. Gemini rubric reward.",
        y_label="Fixed eval dimension score (0-1)",
    )


def _plot_policy_change_curve(metrics_path, image_path):
    rows, run_name = _prepare_curve_rows(metrics_path)
    subtitle = "current policy vs old/reference policy"
    if run_name:
        subtitle += f" | run: {run_name}"
    series = [
        ("old_deviate", "current vs old", (48, 113, 181), 4),
        ("old_deviate_max", "max current vs old", (31, 119, 180), 2),
        ("kl_div", "current vs ref", (232, 126, 4), 3),
        ("old_kl_div", "old vs ref", (148, 103, 189), 3),
        ("grad_norm_clip_ratio", "grad_norm/max", (214, 39, 40), 3),
    ]
    latest_keys = [
        ("old_deviate", "policy_change"),
        ("old_deviate_max", "policy_max"),
        ("kl_div", "kl_ref"),
        ("old_kl_div", "old_kl"),
        ("grad_norm_clip_ratio", "grad/max"),
    ]
    _draw_monitor_curve(
        rows,
        image_path,
        "(f) Policy Change",
        subtitle,
        series,
        latest_keys=latest_keys,
        y_min=0.0,
        footnote="For this diffusion GRPO code, policy_change is prediction MSE between current and old adapter.",
        y_label="Policy divergence / gradient ratio",
    )


def _plot_advantage_distribution(metrics_path, image_path):
    rows, run_name = _prepare_curve_rows(metrics_path)
    os.makedirs(os.path.dirname(image_path), exist_ok=True)
    width, height = 1200, 760
    left, right, top, bottom = 100, 340, 110, 110
    plot_w = width - left - right
    plot_h = height - top - bottom
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    font_sm = _reward_curve_font(16)
    font_md = _reward_curve_font(20)
    font_lg = _reward_curve_font(32, bold=True)

    draw.text((left, 30), "07 Advantage Distribution", fill=(0, 0, 0), font=font_lg)
    subtitle = "latest grouped advantages"
    if run_name:
        subtitle += f" | run: {run_name}"
    draw.text((left, 70), subtitle[:100], fill=(70, 70, 70), font=font_sm)

    hist_keys = [f"advantage_hist_{idx:02d}" for idx in range(21)]
    latest = None
    for row in reversed(rows):
        if any(key in row for key in hist_keys):
            latest = row
            break
    if latest is None:
        draw.text((left, top), "No advantage histogram has been written yet.", fill=(0, 0, 0), font=font_md)
        img.save(image_path)
        return

    counts = np.asarray([float(latest.get(key, 0.0)) for key in hist_keys], dtype=np.float64)
    if counts.sum() <= 0:
        draw.text((left, top), "Advantage histogram is empty.", fill=(0, 0, 0), font=font_md)
        img.save(image_path)
        return
    counts = counts / counts.sum()
    bins = np.linspace(-5.0, 5.0, len(hist_keys))
    max_count = max(float(counts.max()), 1e-9)

    draw.rectangle((left, top, left + plot_w, top + plot_h), outline=(0, 0, 0), width=2)
    for frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
        y = int(top + plot_h - frac * plot_h)
        draw.line((left, y, left + plot_w, y), fill=(232, 232, 232), width=1)
        draw.text((left - 80, y - 10), f"{frac * max_count:.3f}", fill=(80, 80, 80), font=font_sm)
    bar_gap = 4
    bar_w = max(4, int(plot_w / len(hist_keys)) - bar_gap)
    for idx, value in enumerate(counts):
        x0 = int(left + idx * plot_w / len(hist_keys) + bar_gap // 2)
        x1 = x0 + bar_w
        y1 = top + plot_h
        y0 = int(y1 - (float(value) / max_count) * plot_h)
        color = (48, 113, 181) if bins[idx] < 0 else (214, 39, 40)
        if abs(bins[idx]) < 1e-6:
            color = (85, 85, 85)
        draw.rectangle((x0, y0, x1, y1), fill=color)
    for value in [-5, -2.5, 0, 2.5, 5]:
        x = int(left + (value + 5.0) / 10.0 * plot_w)
        draw.line((x, top + plot_h, x, top + plot_h + 7), fill=(0, 0, 0), width=1)
        draw.text((x - 18, top + plot_h + 14), f"{value:g}", fill=(65, 65, 65), font=font_sm)
    _center_text(draw, (left + plot_w / 2, height - 55), "Standardized advantage value", font=font_md, fill=(45, 45, 45))
    _draw_rotated_text(img, (left - 82, top + plot_h / 2), "Probability mass", font=font_md, fill=(45, 45, 45), angle=90)

    latest_x = width - right + 42
    draw.text((latest_x, top), "Latest", fill=(0, 0, 0), font=font_lg)
    latest_lines = [
        f"step: {latest.get('global_step', 0)}",
        f"adv_abs: {float(latest.get('advantage_abs_mean', 0.0)):.4g}",
        f"adv_std: {float(latest.get('advantage_std', 0.0)):.4g}",
        f"pos_ratio: {float(latest.get('positive_advantage_ratio', 0.0)):.4g}",
        f"clip_frac: {float(latest.get('advantage_clip_fraction', 0.0)):.4g}",
    ]
    for idx, line in enumerate(latest_lines):
        draw.text((latest_x, top + 44 + idx * 26), line, fill=(35, 35, 35), font=font_sm)
    draw.text(
        (left, height - 30),
        "Convergence usually shows centered advantages with shrinking variance, but not zero reward spread.",
        fill=(70, 70, 70),
        font=font_sm,
    )
    img.save(image_path)


def _metric_slope(rows, key, window=5):
    values = [
        float(row[key])
        for row in rows
        if key in row and isinstance(row.get(key), (int, float)) and np.isfinite(float(row[key]))
    ]
    if len(values) < 2:
        return None
    tail = values[-window:]
    if len(tail) < 2:
        return None
    return float(tail[-1] - tail[0]) / max(1, len(tail) - 1)


def _plot_grpo_convergence_checklist(training_metrics_path, reward_metrics_path, eval_metrics_path, image_path):
    train_rows, train_run = _prepare_curve_rows(training_metrics_path)
    reward_rows, reward_run = _prepare_curve_rows(reward_metrics_path)
    eval_rows, eval_run = _prepare_curve_rows(eval_metrics_path)
    rows = _merge_metric_rows(reward_rows, train_rows, eval_rows)
    os.makedirs(os.path.dirname(image_path), exist_ok=True)
    width, height = 1300, 820
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    font_sm = _reward_curve_font(17)
    font_md = _reward_curve_font(22)
    font_lg = _reward_curve_font(34, bold=True)
    font_xl = _reward_curve_font(42, bold=True)
    draw.text((70, 34), "08 GRPO Convergence Checklist", fill=(0, 0, 0), font=font_xl)
    run_name = train_run or reward_run or eval_run
    if run_name:
        draw.text((70, 86), f"run: {run_name}", fill=(70, 70, 70), font=font_sm)

    if not rows:
        draw.text((70, 150), "No metrics have been written yet.", fill=(0, 0, 0), font=font_md)
        img.save(image_path)
        return
    latest = rows[-1]
    reward_slope = _metric_slope(reward_rows, "reward_avg_rolling", window=5)
    if reward_slope is None:
        tmp_rows = [dict(row) for row in reward_rows]
        _add_curve_smoothing(tmp_rows, "reward_avg_mean", "reward_avg_ema", "reward_avg_rolling", 0.1, 10)
        reward_slope = _metric_slope(tmp_rows, "reward_avg_rolling", window=5)
    eval_slope = _metric_slope(eval_rows, "eval_reward_mean", window=5)
    kl_slope = _metric_slope(train_rows, "kl_div", window=5)
    policy_slope = _metric_slope(train_rows, "old_deviate", window=5)

    reward_avg = float(latest.get("reward_avg_mean", 0.0))
    reward_std = float(latest.get("reward_avg_std", 0.0))
    zero_std = float(latest.get("zero_std_ratio", 0.0))
    kl_latest = float(latest.get("kl_div", 0.0))
    policy_latest = float(latest.get("old_deviate", 0.0))
    adv_std = float(latest.get("advantage_std", 0.0))
    clip_frac = float(latest.get("advantage_clip_fraction", 0.0))
    timeout_count = float(latest.get("reward_timeout_count", 0.0))

    checks = [
        (
            "Reward plateau",
            reward_slope is not None and abs(reward_slope) < 0.01,
            "rolling reward slope in recent steps is small" if reward_slope is not None else "need more reward points",
            reward_slope,
        ),
        (
            "Eval stable",
            eval_slope is not None and abs(eval_slope) < 0.01,
            "fixed eval reward is stable" if eval_slope is not None else "SKIP_EVAL=1 or eval not recorded",
            eval_slope,
        ),
        (
            "KL stable",
            (kl_slope is not None and abs(kl_slope) < 0.002) or kl_latest < 1e-6,
            "KL not increasing quickly",
            kl_slope,
        ),
        (
            "Policy change small",
            (policy_slope is not None and policy_slope <= 0.0) or policy_latest < 1e-5,
            "current-vs-old prediction drift is small",
            policy_slope,
        ),
        (
            "Advantage signal alive",
            adv_std > 0.02 and zero_std < 0.5,
            "group reward still provides ranking signal",
            adv_std,
        ),
        (
            "No update clipping pressure",
            clip_frac < 0.2,
            "few advantages are clipped",
            clip_frac,
        ),
        (
            "Reward API healthy",
            timeout_count == 0.0,
            "no reward timeout in latest reward step",
            timeout_count,
        ),
        (
            "Reward hacking risk low",
            not (reward_avg > 0.8 and reward_std < 0.05),
            "high reward with tiny variance is suspicious",
            reward_avg,
        ),
    ]

    start_y = 145
    row_h = 62
    for idx, (name, ok, desc, value) in enumerate(checks):
        y = start_y + idx * row_h
        color = (44, 160, 44) if ok else (214, 39, 40)
        symbol = "OK" if ok else "CHECK"
        draw.rounded_rectangle((70, y, 1210, y + 46), radius=8, outline=(225, 225, 225), width=1, fill=(250, 250, 250))
        draw.text((92, y + 11), symbol, fill=color, font=font_md)
        draw.text((190, y + 10), name, fill=(0, 0, 0), font=font_md)
        draw.text((520, y + 12), desc, fill=(70, 70, 70), font=font_sm)
        value_text = "n/a" if value is None else f"{float(value):.4g}"
        draw.text((1080, y + 12), value_text, fill=(45, 45, 45), font=font_sm)

    summary_y = start_y + len(checks) * row_h + 20
    ok_count = sum(1 for _, ok, _, _ in checks if ok)
    draw.text((70, summary_y), f"Summary: {ok_count}/{len(checks)} checks pass", fill=(0, 0, 0), font=font_lg)
    draw.text(
        (70, summary_y + 48),
        "Do not stop by this image alone: inspect scored_candidate_grids for human visual quality and reward hacking.",
        fill=(70, 70, 70),
        font=font_sm,
    )
    img.save(image_path)


def _plot_local_monitor_curves(config):
    curve_dir = os.path.join(config.save_dir, "reward_curves")
    reward_metrics_path = os.path.join(config.save_dir, "reward_metrics.jsonl")
    training_metrics_path = os.path.join(config.save_dir, "training_metrics.jsonl")
    eval_metrics_path = os.path.join(config.save_dir, "eval_metrics.jsonl")

    # Keep only a small, paper-facing set of figures.
    _plot_reward_train_eval_curve(
        reward_metrics_path,
        eval_metrics_path,
        os.path.join(curve_dir, "01_fixed_eval_reward_curve.png"),
    )
    _plot_eval_reward_dimensions_curve(
        reward_metrics_path,
        eval_metrics_path,
        os.path.join(curve_dir, "02_reward_dimension_curve.png"),
    )
    if os.path.exists(training_metrics_path):
        _plot_loss_policy_total_curve(training_metrics_path, os.path.join(curve_dir, "03_policy_loss_curve.png"))
        _plot_kl_clip_fraction_curve(training_metrics_path, os.path.join(curve_dir, "04_kl_clip_curve.png"))
    if os.path.exists(reward_metrics_path) or os.path.exists(training_metrics_path):
        _plot_grad_reward_signal_curve(
            training_metrics_path,
            reward_metrics_path,
            os.path.join(curve_dir, "05_reward_signal_curve.png"),
        )


def write_local_eval_metrics(config, global_step, epoch, final_rewards):
    row = {
        "timestamp": time.time(),
        "global_step": int(global_step),
        "epoch": int(epoch),
    }
    run_name = getattr(config, "run_name", None)
    if run_name:
        row["run_name"] = str(run_name)
    for key, values in final_rewards.items():
        vals = _flatten_reward_values(values)
        vals = vals[vals != -10]
        if vals.size == 0:
            continue
        if key == "avg":
            row["eval_reward_mean"] = float(vals.mean())
            row["eval_reward_std"] = float(vals.std())
            row["eval_reward_min"] = float(vals.min())
            row["eval_reward_max"] = float(vals.max())
        else:
            row[f"eval_reward_{key}_mean"] = float(vals.mean())
            row[f"eval_reward_{key}_std"] = float(vals.std())
            row[f"eval_reward_{key}_min"] = float(vals.min())
            row[f"eval_reward_{key}_max"] = float(vals.max())

    os.makedirs(config.save_dir, exist_ok=True)
    metrics_path = os.path.join(config.save_dir, "eval_metrics.jsonl")
    with open(metrics_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    _plot_local_monitor_curves(config)
    return metrics_path


def write_local_training_metrics(config, global_step, epoch, inner_epoch, gradient_update_times, metrics):
    row = {
        "timestamp": time.time(),
        "global_step": int(global_step),
        "epoch": int(epoch),
        "inner_epoch": int(inner_epoch),
        "gradient_update_times": int(gradient_update_times),
    }
    run_name = getattr(config, "run_name", None)
    if run_name:
        row["run_name"] = str(run_name)
    for key, value in metrics.items():
        try:
            row[key] = float(value)
        except Exception:
            pass

    max_grad_norm = getattr(config.train, "max_grad_norm", None)
    if max_grad_norm and "grad_norm" in row:
        row["grad_norm_clip_ratio"] = float(row["grad_norm"] / max(float(max_grad_norm), 1e-12))

    os.makedirs(config.save_dir, exist_ok=True)
    metrics_path = os.path.join(config.save_dir, "training_metrics.jsonl")
    with open(metrics_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

    _plot_local_monitor_curves(config)
    return metrics_path


def write_local_reward_details(config, global_step, epoch, rank, reward_metadata):
    if not isinstance(reward_metadata, dict):
        return None
    details = reward_metadata.get("details")
    if not isinstance(details, list) or not details:
        return None

    os.makedirs(config.save_dir, exist_ok=True)
    details_path = os.path.join(config.save_dir, "reward_details.jsonl")
    timeout_count = int(reward_metadata.get("timeout_count", 0) or 0)
    placeholder_count = int(reward_metadata.get("placeholder_count", 0) or 0)
    parse_failure_count = int(reward_metadata.get("parse_failure_count", 0) or 0)
    missing_key_count = int(reward_metadata.get("missing_key_count", 0) or 0)
    fallback_score_count = int(reward_metadata.get("fallback_score_count", 0) or 0)

    with open(details_path, "a", encoding="utf-8") as f:
        try:
            import fcntl
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        except Exception:
            fcntl = None
        try:
            for detail in details:
                row = {
                    "timestamp": time.time(),
                    "global_step": int(global_step),
                    "epoch": int(epoch),
                    "rank": int(rank),
                    "reward_timeout_count": timeout_count,
                    "reward_placeholder_count": placeholder_count,
                    "reward_parse_failure_count": parse_failure_count,
                    "reward_missing_key_count": missing_key_count,
                    "reward_fallback_score_count": fallback_score_count,
                }
                if isinstance(detail, dict):
                    row.update(detail)
                else:
                    row["detail"] = detail
                f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        finally:
            try:
                if fcntl is not None:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            except Exception:
                pass
    return details_path


def write_local_reward_metrics(config, global_step, epoch, gathered_rewards_dict, extra_metrics=None):
    row = {
        "timestamp": time.time(),
        "global_step": int(global_step),
        "epoch": int(epoch),
    }
    run_name = getattr(config, "run_name", None)
    if run_name:
        row["run_name"] = str(run_name)
    for key, values in gathered_rewards_dict.items():
        vals = _flatten_reward_values(values)
        if vals.size == 0:
            continue
        row[f"reward_{key}_mean"] = float(vals.mean())
        row[f"reward_{key}_std"] = float(vals.std())
        row[f"reward_{key}_min"] = float(vals.min())
        row[f"reward_{key}_max"] = float(vals.max())
    if extra_metrics:
        for key, value in extra_metrics.items():
            try:
                row[key] = float(value)
            except Exception:
                pass

    os.makedirs(config.save_dir, exist_ok=True)
    metrics_path = os.path.join(config.save_dir, "reward_metrics.jsonl")
    with open(metrics_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

    curve_path = os.path.join(config.save_dir, "reward_curves", "01_fixed_eval_reward_curve.png")
    _plot_local_monitor_curves(config)
    return metrics_path, curve_path


def _reward_value_list(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().reshape(-1).tolist()
    if isinstance(value, np.ndarray):
        return value.reshape(-1).tolist()
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def submit_reward_futures(
    executor,
    reward_fn,
    reward_images,
    prompts,
    prompt_metadata,
    reward_ref_images,
    only_strict=True,
):
    """Submit reward jobs.

    By default, each generated image is submitted as its own reward job. This
    lets Gemini scoring start as soon as a generated image is available from the
    sampling batch, and prevents one failed/slow candidate from holding the rest
    of the batch inside the same reward call.
    """
    per_image = _env_flag("EDIT_R1_REWARD_PER_IMAGE", "1")
    if not per_image:
        return {
            "mode": "batch",
            "futures": [
                executor.submit(
                    reward_fn,
                    reward_images,
                    prompts,
                    prompt_metadata,
                    reward_ref_images,
                    only_strict=only_strict,
                )
            ],
        }

    future_items = []
    for idx in range(len(prompts)):
        future_items.append(
            (
                idx,
                executor.submit(
                    reward_fn,
                    [reward_images[idx]],
                    [prompts[idx]],
                    [prompt_metadata[idx]],
                    [reward_ref_images[idx]],
                    only_strict=only_strict,
                ),
            )
        )
    return {"mode": "per_image", "futures": future_items}


def collect_reward_futures(reward_jobs, expected_count):
    if not isinstance(reward_jobs, dict) or reward_jobs.get("mode") == "batch":
        future = reward_jobs["futures"][0] if isinstance(reward_jobs, dict) else reward_jobs
        return future.result()

    ordered = [None for _ in range(expected_count)]
    reward_metadata = {"details": []}
    for idx, future in reward_jobs.get("futures", []):
        rewards, metadata = future.result()
        ordered[int(idx)] = (rewards, metadata)

        if isinstance(metadata, dict):
            for meta_key in ("timeout_count", "placeholder_count", "parse_failure_count", "missing_key_count", "fallback_score_count"):
                if meta_key in metadata:
                    reward_metadata[meta_key] = reward_metadata.get(meta_key, 0) + int(metadata.get(meta_key) or 0)
            details = metadata.get("details")
            if isinstance(details, list):
                for detail in details:
                    if isinstance(detail, dict):
                        detail_row = dict(detail)
                        detail_row["single_reward_index"] = detail_row.get("index")
                        detail_row["index"] = int(idx)
                        detail_row["batch_local_index"] = int(idx)
                    else:
                        detail_row = {"index": int(idx), "detail": detail}
                    reward_metadata.setdefault("details", []).append(detail_row)

    reward_values = {}
    for idx, result in enumerate(ordered):
        if result is None:
            continue
        rewards, _metadata = result
        if not isinstance(rewards, dict):
            continue
        for key, value in rewards.items():
            values = _reward_value_list(value)
            scalar = values[0] if values else 0.0
            reward_values.setdefault(key, [None for _ in range(expected_count)])[idx] = scalar

    avg_values = reward_values.get("avg", [0.0 for _ in range(expected_count)])
    merged_rewards = {}
    for key, values in reward_values.items():
        fixed = []
        for idx, value in enumerate(values):
            if value is None:
                value = avg_values[idx] if key != "avg" and idx < len(avg_values) else 0.0
            fixed.append(float(value))
        merged_rewards[key] = np.asarray(fixed, dtype=np.float32)

    if "avg" not in merged_rewards:
        merged_rewards["avg"] = np.zeros(expected_count, dtype=np.float32)
    return merged_rewards, reward_metadata


def eval_fn(
    pipeline,
    test_dataloader,
    text_encoders,
    tokenizers,
    config,
    device,
    rank,
    world_size,
    global_step,
    epoch,
    reward_fn,
    executor,
    mixed_precision_dtype,
    ema,
    transformer_trainable_parameters,
):
    if config.train.ema and ema is not None:
        ema.copy_ema_to(transformer_trainable_parameters, store_temp=True)

    pipeline.transformer.eval()
    all_rewards = defaultdict(list)

    test_sampler = (
        DistributedSampler(
            test_dataloader.dataset, num_replicas=world_size, rank=rank, shuffle=False
        )
        if world_size > 1
        else None
    )
    eval_loader = DataLoader(
        test_dataloader.dataset,
        batch_size=1,
        sampler=test_sampler,
        collate_fn=test_dataloader.collate_fn,
        num_workers=test_dataloader.num_workers,
    )

    max_eval_batches = int(os.getenv("MAX_EVAL_BATCHES", "0") or "0")
    for eval_batch_idx, test_batch in enumerate(tqdm(
        eval_loader,
        desc="Eval: ",
        disable=not is_main_process(rank),
        position=0,
    )):
        if max_eval_batches > 0 and eval_batch_idx >= max_eval_batches:
            break
        prompts, prompt_metadata, ref_images, prompt_with_image_paths, target_size = test_batch
        sample_height, sample_width = target_size
        prompt_embeds, pooled_prompt_embeds, _ = compute_text_embeddings(
            prompts, text_encoders, tokenizers, max_sequence_length=128, device=device
        )
        current_batch_size = len(prompt_embeds)
        with torch_autocast(
            enabled=(config.mixed_precision in ["fp16", "bf16"]),
            dtype=mixed_precision_dtype,
        ):
            with torch.no_grad():
                images, _, _, _, _, _ = pipeline_with_logprob(
                    pipeline,
                    prompt_embeds=prompt_embeds,
                    pooled_prompt_embeds=pooled_prompt_embeds,
                    image=ref_images,
                    num_inference_steps=config.sample.eval_num_steps,
                    guidance_scale=config.sample.guidance_scale,
                    output_type="pt",
                    height=sample_height,
                    width=sample_width,
                    noise_level=config.sample.noise_level,
                    deterministic=True,
                    solver="flow",
                    max_area=sample_height * sample_width,
                    _auto_resize=False,
                )

        reward_images, reward_ref_images = prepare_reward_images(
            images, ref_images, prompt_metadata
        )
        rewards_future = executor.submit(
            reward_fn,
            reward_images,
            prompts,
            prompt_metadata,
            reward_ref_images,
            only_strict=False,
        )
        time.sleep(0)
        rewards, reward_metadata = rewards_future.result()

        for key, value in rewards.items():
            rewards_tensor = torch.as_tensor(value, device=device).float()
            gathered_value = gather_tensor_to_all(rewards_tensor, world_size)
            all_rewards[key].append(gathered_value.numpy())

    if is_main_process(rank):
        final_rewards = {
            key: np.concatenate(value_list) for key, value_list in all_rewards.items()
        }

        images_to_log = reward_images
        prompts_to_log = prompts

        with tempfile.TemporaryDirectory() as tmpdir:
            num_samples_to_log = min(15, len(images_to_log))
            for idx in range(num_samples_to_log):
                pil = _pil_to_rgb(images_to_log[idx])
                pil.save(os.path.join(tmpdir, f"{idx}.jpg"))

            sampled_prompts_log = [prompts_to_log[i] for i in range(num_samples_to_log)]
            sampled_rewards_log = [
                {k: final_rewards[k][i] for k in final_rewards}
                for i in range(num_samples_to_log)
            ]

            wandb.log(
                {
                    "eval_images": [
                        wandb.Image(
                            os.path.join(tmpdir, f"{idx}.jpg"),
                            caption=f"{prompt:.1000} | "
                            + " | ".join(
                                f"{k}: {v:.2f}" for k, v in reward.items() if v != -10
                            ),
                        )
                        for idx, (prompt, reward) in enumerate(
                            zip(sampled_prompts_log, sampled_rewards_log)
                        )
                    ],
                    **{
                        f"eval_reward_{key}": np.mean(value[value != -10])
                        for key, value in final_rewards.items()
                    },
                },
                step=global_step,
            )
            eval_metrics_path = write_local_eval_metrics(
                config,
                global_step,
                epoch,
                final_rewards,
            )
            logger.info(f"Local eval metrics saved to {eval_metrics_path}")

    if config.train.ema and ema is not None:
        ema.copy_temp_to(transformer_trainable_parameters)

    if world_size > 1:
        dist.barrier()


def save_ckpt(
    save_dir,
    transformer_ddp,
    global_step,
    rank,
    ema,
    transformer_trainable_parameters,
    config,
    optimizer,
    scaler,
):
    if is_main_process(rank):
        save_root = os.path.join(save_dir, "checkpoints", f"checkpoint-{global_step}")
        save_root_lora = os.path.join(save_root, "lora")
        os.makedirs(save_root_lora, exist_ok=True)

        model_to_save = transformer_ddp.module

        if config.train.ema and ema is not None:
            ema.copy_ema_to(transformer_trainable_parameters, store_temp=True)

        model_to_save.save_pretrained(save_root_lora)  # For LoRA/PEFT models

        torch.save(optimizer.state_dict(), os.path.join(save_root, "optimizer.pt"))
        if scaler is not None:
            torch.save(scaler.state_dict(), os.path.join(save_root, "scaler.pt"))

        if config.train.ema and ema is not None:
            ema.copy_temp_to(transformer_trainable_parameters)
        logger.info(f"Saved checkpoint to {save_root}")


def main(_):
    config = FLAGS.config

    # --- Distributed Setup ---
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    setup_distributed(rank, local_rank, world_size)
    device = torch.device(f"cuda:{local_rank}")

    unique_id = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
    if not config.run_name:
        config.run_name = unique_id
    else:
        config.run_name += "_" + unique_id

    # --- WandB Init (only on main process) ---
    if is_main_process(rank):
        log_dir = os.path.join(config.logdir, config.run_name)
        os.makedirs(log_dir, exist_ok=True)
        wandb.init(
            project="flow-grpo",
            name=config.run_name,
            config=config.to_dict(),
            dir=log_dir,
        )
    logger.info(f"\n{config}")

    set_seed(config.seed, rank)  # Pass rank for different seeds per process

    # --- Mixed Precision Setup ---
    mixed_precision_dtype = None
    if config.mixed_precision == "fp16":
        mixed_precision_dtype = torch.float16
    elif config.mixed_precision == "bf16":
        mixed_precision_dtype = torch.bfloat16

    enable_amp = mixed_precision_dtype is not None
    scaler = GradScaler(enabled=enable_amp)

    # --- Load pipeline and models ---
    pipeline = FluxKontextPipeline.from_pretrained(config.pretrained.model)
    pipeline.vae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    pipeline.text_encoder_2.requires_grad_(False)
    pipeline.transformer.requires_grad_(not config.use_lora)
    text_encoders = [pipeline.text_encoder, pipeline.text_encoder_2]
    tokenizers = [pipeline.tokenizer, pipeline.tokenizer_2]
    pipeline.safety_checker = None
    pipeline.set_progress_bar_config(
        position=1,
        disable=not is_main_process(rank),
        leave=False,
        desc="Timestep",
        dynamic_ncols=True,
    )

    text_encoder_dtype = mixed_precision_dtype if enable_amp else torch.float32

    pipeline.vae.to(device, dtype=torch.float32)  # VAE usually fp32

    prepare_fsdp_model(
        pipeline.text_encoder,
        shard_conditions=[lambda n, m: isinstance(m, (CLIPEncoderLayer,))],
        cpu_offload=False,
        weight_dtype=text_encoder_dtype,
    )

    prepare_fsdp_model(
        pipeline.text_encoder_2,
        shard_conditions=[lambda n, m: isinstance(m, (T5Block,))],
        cpu_offload=False,
        weight_dtype=text_encoder_dtype,
    )

    transformer = pipeline.transformer.to(device)
    # pipeline.transformer.compile(fullgraph=True)
    pipeline.transformer.enable_gradient_checkpointing()

    if config.use_lora:
        target_modules = [
            "attn.add_k_proj",
            "attn.add_q_proj",
            "attn.add_v_proj",
            "attn.to_add_out",
            "attn.to_k",
            "attn.to_out.0",
            "attn.to_q",
            "attn.to_v",
        ]
        transformer_lora_config = LoraConfig(
            r=32,
            lora_alpha=64,
            init_lora_weights="gaussian",
            target_modules=target_modules,
        )
        if config.train.lora_path:
            if os.path.isdir(config.train.lora_path):
                transformer = PeftModel.from_pretrained(transformer, config.train.lora_path)
                transformer.set_adapter("default")
            elif os.path.isfile(config.train.lora_path) and config.train.lora_path.endswith(".safetensors"):
                merge_diffsynth_flux_lora_into_transformer(transformer, config.train.lora_path)
                transformer = get_peft_model(transformer, transformer_lora_config)
            else:
                raise FileNotFoundError(
                    "config.train.lora_path must be a PEFT adapter directory or a "
                    f"DiffSynth/transformer .safetensors file, got: {config.train.lora_path}"
                )
        else:
            transformer = get_peft_model(transformer, transformer_lora_config)
        transformer.add_adapter("old", transformer_lora_config)
        transformer.set_adapter("default")
    transformer_ddp = DDP(
        transformer,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=False,
    )
    transformer_ddp.module.set_adapter("default")
    transformer_trainable_parameters = list(
        filter(lambda p: p.requires_grad, transformer_ddp.module.parameters())
    )
    transformer_ddp.module.set_adapter("old")
    old_transformer_trainable_parameters = list(
        filter(lambda p: p.requires_grad, transformer_ddp.module.parameters())
    )
    transformer_ddp.module.set_adapter("default")

    if config.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # --- Optimizer ---
    optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(
        transformer_trainable_parameters,  # Use params from original model for optimizer
        lr=config.train.learning_rate,
        betas=(config.train.adam_beta1, config.train.adam_beta2),
        weight_decay=config.train.adam_weight_decay,
        eps=config.train.adam_epsilon,
    )

    # --- Datasets and Dataloaders ---
    train_dataset = PromptImageDataset(
        config.dataset, config.resolution, "train")
    test_dataset = PromptImageDataset(config.dataset, config.resolution, "test")

    train_sampler = DistributedKRepeatSampler(
        dataset=train_dataset,
        batch_size=config.sample.train_batch_size,  # This is per-GPU batch size
        k=config.sample.num_image_per_prompt,
        num_replicas=world_size,
        rank=rank,
        seed=config.seed,
        banned_prompts=None,
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        num_workers=0,
        collate_fn=train_dataset.collate_fn,
        pin_memory=True,
    )

    test_sampler = (
        DistributedSampler(
            test_dataset, num_replicas=world_size, rank=rank, shuffle=False
        )
        if world_size > 1
        else None
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=1,
        sampler=test_sampler,  # Use distributed sampler for eval
        collate_fn=test_dataset.collate_fn,
        num_workers=0,
        pin_memory=True,
    )

    # --- Prompt Trackering ---
    if config.sample.num_image_per_prompt == 1:
        config.per_prompt_stat_tracking = False
    if config.per_prompt_stat_tracking:
        stat_tracker = PerPromptStatTracker(
            config.sample.global_std,
            config.sample.ban_std_thres,
            config.sample.ban_mean_thres,
        )
    else:
        assert False

    executor = futures.ThreadPoolExecutor(max_workers=8)  # Async reward computation

    # Train!
    samples_per_epoch = (
        config.sample.train_batch_size
        * world_size
        * config.sample.num_batches_per_epoch
    )
    total_train_batch_size = (
        config.train.batch_size * world_size * config.train.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num Epochs = {config.num_epochs}")
    logger.info(f"  Sample batch size per device = {config.sample.train_batch_size}")
    logger.info(f"  Train batch size per device = {config.train.batch_size}")
    logger.info(
        f"  Gradient Accumulation steps = {config.train.gradient_accumulation_steps}"
    )
    logger.info("")
    logger.info(f"  Total number of samples per epoch = {samples_per_epoch}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}"
    )
    logger.info(
        f"  Number of gradient updates per inner epoch = {samples_per_epoch // total_train_batch_size}"
    )
    logger.info(f"  Number of inner epochs = {config.train.num_inner_epochs}")

    reward_fn = getattr(flow_grpo.rewards, "multi_score")(
        device, config.reward_fn
    )  # Pass device
    eval_reward_fn = getattr(flow_grpo.rewards, "multi_score")(
        device, config.reward_fn
    )  # Pass device

    # --- Resume from checkpoint ---
    first_epoch = 0
    global_step = 0
    if config.resume_from:
        logger.info(f"Resuming from {config.resume_from}")
        # Assuming checkpoint dir contains lora, optimizer.pt, scaler.pt
        lora_path = os.path.join(config.resume_from, "lora")
        if os.path.exists(lora_path):  # Check if it's a PEFT model save
            transformer_ddp.module.load_adapter(
                lora_path, adapter_name="default", is_trainable=True
            )
            transformer_ddp.module.load_adapter(
                lora_path, adapter_name="old", is_trainable=False
            )
        else:  # Try loading full state dict if it's not a PEFT save structure
            model_ckpt_path = os.path.join(
                config.resume_from, "transformer_model.pt"
            )  # Or specific name
            if os.path.exists(model_ckpt_path):
                transformer_ddp.module.load_state_dict(
                    torch.load(model_ckpt_path, map_location=device)
                )

        opt_path = os.path.join(config.resume_from, "optimizer.pt")
        if os.path.exists(opt_path):
            optimizer.load_state_dict(torch.load(opt_path, map_location=device))

        scaler_path = os.path.join(config.resume_from, "scaler.pt")
        if os.path.exists(scaler_path) and enable_amp:
            scaler.load_state_dict(torch.load(scaler_path, map_location=device))

        # Extract epoch and step from checkpoint name, e.g., "checkpoint-1000" -> global_step = 1000
        try:
            global_step = int(os.path.basename(config.resume_from).split("-")[-1])
            logger.info(
                f"Resumed global_step to {global_step}. Epoch estimation might be needed."
            )
        except ValueError:
            logger.warning(
                f"Could not parse global_step from checkpoint name: {config.resume_from}. Starting global_step from 0."
            )
            global_step = 0

    ema = None
    if config.train.ema:
        ema = EMAModuleWrapper(
            transformer_trainable_parameters,
            decay=0.9,
            update_step_interval=1,
            device=device,
        )

    num_train_timesteps = int(config.sample.num_steps * config.train.timestep_fraction)

    logger.info("***** Running training *****")

    train_iter = iter(train_dataloader)
    optimizer.zero_grad()

    for src_param, tgt_param in zip(
        transformer_trainable_parameters,
        old_transformer_trainable_parameters,
        strict=True,
    ):
        tgt_param.data.copy_(src_param.detach().data)
        assert src_param is not tgt_param

    for epoch in range(first_epoch, config.num_epochs):
        if hasattr(train_sampler, "set_epoch"):
            train_sampler.set_epoch(epoch)

        # SAMPLING
        pipeline.transformer.eval()
        samples_data_list = []
        collect_reward_per_source = _env_flag("EDIT_R1_COLLECT_REWARD_PER_SOURCE", "0")
        epoch_reward_timeout_count = 0
        epoch_reward_placeholder_count = 0
        epoch_reward_parse_failure_count = 0
        epoch_reward_missing_key_count = 0
        epoch_reward_fallback_score_count = 0

        def _collect_and_store_sample_reward(sample_item):
            nonlocal epoch_reward_timeout_count
            nonlocal epoch_reward_placeholder_count
            nonlocal epoch_reward_parse_failure_count
            nonlocal epoch_reward_missing_key_count
            nonlocal epoch_reward_fallback_score_count

            rewards, reward_metadata = collect_reward_futures(
                sample_item["reward_jobs"],
                int(sample_item["prompt_ids"].shape[0]),
            )
            if isinstance(reward_metadata, dict):
                epoch_reward_timeout_count += int(reward_metadata.get("timeout_count", 0) or 0)
                epoch_reward_placeholder_count += int(reward_metadata.get("placeholder_count", 0) or 0)
                epoch_reward_parse_failure_count += int(reward_metadata.get("parse_failure_count", 0) or 0)
                epoch_reward_missing_key_count += int(reward_metadata.get("missing_key_count", 0) or 0)
                epoch_reward_fallback_score_count += int(reward_metadata.get("fallback_score_count", 0) or 0)
            details_path = write_local_reward_details(
                config, global_step, epoch, rank, reward_metadata
            )
            if details_path and is_main_process(rank):
                logger.info(f"Local reward details appended to {details_path}")
            scored_grid_paths = save_scored_candidate_grid(
                config,
                global_step,
                epoch,
                int(sample_item.get("sample_batch_idx", 0)),
                rank,
                world_size,
                sample_item.get("reward_images_local", []),
                sample_item.get("reward_ref_images_local", []),
                sample_item.get("prompts_local", []),
                sample_item.get("prompt_metadata_local", []),
                rewards,
                reward_metadata,
            )
            if scored_grid_paths and is_main_process(rank):
                logger.info(f"Scored candidate grid saved to {scored_grid_paths[-1]}")
            sample_item["rewards"] = {
                k: torch.as_tensor(v, device=device).float() for k, v in rewards.items()
            }
            for transient_key in (
                "reward_jobs",
                "reward_images_local",
                "reward_ref_images_local",
                "prompts_local",
                "prompt_metadata_local",
                "sample_batch_idx",
            ):
                sample_item.pop(transient_key, None)

        for i in tqdm(
            range(config.sample.num_batches_per_epoch),
            desc=f"Epoch {epoch}: sampling",
            disable=not is_main_process(rank),
            position=0,
        ):
            transformer_ddp.module.set_adapter("default")
            if hasattr(train_sampler, "set_epoch") and isinstance(
                train_sampler, DistributedKRepeatSampler
            ):
                train_sampler.banned_prompts = (
                    stat_tracker.banned_prompts if config.sample.ban_prompt else set()
                )
                train_sampler.set_epoch(epoch * config.sample.num_batches_per_epoch + i)

            prompts, prompt_metadata, ref_images, prompt_with_image_paths, target_size = next(
                train_iter
            )
            sample_height, sample_width = target_size

            prompt_embeds, pooled_prompt_embeds, _ = compute_text_embeddings(
                prompts,
                text_encoders,
                tokenizers,
                max_sequence_length=128,
                device=device,
            )
            prompt_ids = tokenizers[0](
                prompts,
                padding="max_length",
                max_length=256,
                truncation=True,
                return_tensors="pt",
            ).input_ids.to(device)

            sample_generators, sample_seeds = make_candidate_generators(
                config,
                device,
                global_step,
                epoch,
                i,
                rank,
                len(prompts),
                world_size,
            )
            if sample_seeds and _env_flag("EDIT_R1_PRINT_SAMPLE_SEEDS", "1") and epoch == 0 and i == 0:
                logger.info(
                    "Candidate sampling seeds "
                    f"rank={rank} batch={i} global_step={global_step}: {sample_seeds}"
                )

            transformer_ddp.module.set_adapter("old")
            with torch_autocast(enabled=enable_amp, dtype=mixed_precision_dtype):
                with torch.no_grad():
                    images, latents, latent_ids, text_ids, image_latents, _ = (
                        pipeline_with_logprob(
                            pipeline,
                            image=ref_images,
                            prompt_embeds=prompt_embeds,
                            pooled_prompt_embeds=pooled_prompt_embeds,
                            num_inference_steps=config.sample.num_steps,
                            guidance_scale=config.sample.guidance_scale,
                            output_type="pt",
                            height=sample_height,
                            width=sample_width,
                            noise_level=config.sample.noise_level,
                            deterministic=config.sample.deterministic,
                            solver=config.sample.solver,
                            max_area=sample_height * sample_width,
                            generator=sample_generators,
                            _auto_resize=False
                        )
                    )
            transformer_ddp.module.set_adapter("default")

            latents = torch.stack(latents, dim=1)
            timesteps = pipeline.scheduler.timesteps.repeat(len(prompts), 1).to(device)

            reward_images, reward_ref_images = prepare_reward_images(
                images, ref_images, prompt_metadata
            )

            if _env_flag("EDIT_R1_SAMPLE_ONLY", "0"):
                sample_grid_paths = save_scored_candidate_grid(
                    config,
                    global_step,
                    epoch,
                    i,
                    rank,
                    world_size,
                    reward_images,
                    reward_ref_images,
                    prompts,
                    prompt_metadata,
                    {},
                    {"details": []},
                )
                if sample_grid_paths and is_main_process(rank):
                    logger.info(f"Sample-only candidate grid saved to {sample_grid_paths[-1]}")
                del images, latents, reward_images, reward_ref_images
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue

            reward_jobs = submit_reward_futures(
                executor,
                reward_fn,
                reward_images,
                prompts,
                prompt_metadata,
                reward_ref_images,
                only_strict=True,
            )
            time.sleep(0)

            if _env_flag("EDIT_R1_RUBRIC_CHECK_ONLY", "0"):
                rewards, reward_metadata = collect_reward_futures(
                    reward_jobs,
                    int(prompt_ids.shape[0]),
                )
                details_path = write_local_reward_details(
                    config, global_step, epoch, rank, reward_metadata
                )
                scored_grid_paths = save_scored_candidate_grid(
                    config,
                    global_step,
                    epoch,
                    i,
                    rank,
                    world_size,
                    reward_images,
                    reward_ref_images,
                    prompts,
                    prompt_metadata,
                    rewards,
                    reward_metadata,
                )
                if details_path and is_main_process(rank):
                    logger.info(f"Rubric-check reward details appended to {details_path}")
                if scored_grid_paths and is_main_process(rank):
                    logger.info(f"Rubric-check scored grid saved to {scored_grid_paths[-1]}")
                del images, latents, reward_images, reward_ref_images, reward_jobs
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue

            sample_item = {
                "prompt_ids": prompt_ids,
                "prompt_embeds": prompt_embeds,
                "pooled_prompt_embeds": pooled_prompt_embeds,
                "timesteps": timesteps,
                "text_ids": text_ids.unsqueeze(0).repeat(len(prompt_ids), 1, 1),
                "latent_ids": latent_ids.unsqueeze(0).repeat(len(prompt_ids), 1, 1),
                "next_timesteps": torch.concatenate(
                    [timesteps[:, 1:], torch.zeros_like(timesteps[:, :1])], dim=1
                ),
                "latents_clean": latents[:, -1],
                "image_latents": image_latents,
                "reward_jobs": reward_jobs,
                "reward_images_local": reward_images,
                "reward_ref_images_local": reward_ref_images,
                "prompts_local": prompts,
                "prompt_metadata_local": prompt_metadata,
                "sample_batch_idx": i,
            }
            samples_data_list.append(sample_item)
            if collect_reward_per_source:
                _collect_and_store_sample_reward(sample_item)


        if _env_flag("EDIT_R1_RUBRIC_CHECK_ONLY", "0"):
            if is_main_process(rank):
                logger.info("Rubric-check-only epoch finished after sampling, scoring, and grid saving; skipping training.")
            continue

        for sample_item in tqdm(
            samples_data_list,
            desc="Waiting for rewards",
            disable=not is_main_process(rank),
            position=0,
        ):
            if "rewards" in sample_item:
                continue
            _collect_and_store_sample_reward(sample_item)

        # Collate samples
        collated_samples = {
            k: (
                torch.cat([s[k] for s in samples_data_list], dim=0)
                if not isinstance(samples_data_list[0][k], dict)
                else {
                    sk: torch.cat([s[k][sk] for s in samples_data_list], dim=0)
                    for sk in samples_data_list[0][k]
                }
            )
            for k in samples_data_list[0].keys()
        }

        # Logging images (main process)
        if epoch % 10 == 0 and is_main_process(rank):
            images_to_log = reward_images  # from last sampling batch on this rank
            prompts_to_log = prompts  # from last sampling batch on this rank
            rewards_to_log = collated_samples["rewards"]["avg"][
                -len(images_to_log) :
            ].cpu()

            with tempfile.TemporaryDirectory() as tmpdir:
                num_to_log = min(15, len(images_to_log))
                for idx in range(num_to_log):  # log first N
                    pil = _pil_to_rgb(images_to_log[idx])
                    pil.save(os.path.join(tmpdir, f"{idx}.jpg"))

                wandb.log(
                    {
                        "images": [
                            wandb.Image(
                                os.path.join(tmpdir, f"{idx}.jpg"),
                                caption=f"{prompts_to_log[idx]:.100} | avg: {rewards_to_log[idx]:.2f}",
                            )
                            for idx in range(num_to_log)
                        ],
                    },
                    step=global_step,
                )
        collated_samples["rewards"]["avg"] = (
            collated_samples["rewards"]["avg"]
            .unsqueeze(1)
            .repeat(1, num_train_timesteps)
        )

        # Gather rewards across processes
        gathered_rewards_dict = {}
        for key, value_tensor in collated_samples["rewards"].items():
            gathered_rewards_dict[key] = gather_tensor_to_all(
                value_tensor, world_size
            ).numpy()

        if is_main_process(rank):  # logging
            wandb.log(
                {
                    "epoch": epoch,
                    **{
                        f"reward_{k}": v.mean()
                        for k, v in gathered_rewards_dict.items()
                        if "_strict_accuracy" not in k and "_accuracy" not in k
                    },
                },
                step=global_step,
            )

        if config.per_prompt_stat_tracking:
            prompt_ids_all = gather_tensor_to_all(
                collated_samples["prompt_ids"], world_size
            )
            prompts_all_decoded = pipeline.tokenizer.batch_decode(
                prompt_ids_all.cpu().numpy(), skip_special_tokens=True
            )
            # Stat tracker update expects numpy arrays for rewards
            advantages, stds, means = stat_tracker.update(
                prompts_all_decoded, gathered_rewards_dict["avg"]
            )

            if is_main_process(rank):
                group_size, trained_prompt_num = stat_tracker.get_stats()
                zero_std_ratio, reward_std_mean = calculate_zero_std_ratio(
                    prompts_all_decoded, gathered_rewards_dict
                )
                wandb.log(
                    {
                        "banned_prompt_num": len(stat_tracker.banned_prompts) if config.sample.ban_prompt else 0,
                        "group_size": group_size,
                        "trained_prompt_num": trained_prompt_num,
                        "reward_timeout_count": epoch_reward_timeout_count,
                        "reward_placeholder_count": epoch_reward_placeholder_count,
                        "reward_parse_failure_count": epoch_reward_parse_failure_count,
                        "reward_missing_key_count": epoch_reward_missing_key_count,
                        "reward_fallback_score_count": epoch_reward_fallback_score_count,
                        "zero_std_ratio": zero_std_ratio,
                        "reward_std_mean": reward_std_mean,
                        "mean_reward_100": stat_tracker.get_mean_of_top_rewards(100),
                        "mean_reward_75": stat_tracker.get_mean_of_top_rewards(75),
                        "mean_reward_50": stat_tracker.get_mean_of_top_rewards(50),
                        "mean_reward_25": stat_tracker.get_mean_of_top_rewards(25),
                        "mean_reward_10": stat_tracker.get_mean_of_top_rewards(10),
                    },
                    step=global_step,
                )
                metrics_path, curve_path = write_local_reward_metrics(
                    config,
                    global_step,
                    epoch,
                    gathered_rewards_dict,
                    {
                        "zero_std_ratio": zero_std_ratio,
                        "reward_std_mean": reward_std_mean,
                        "group_size": group_size,
                        "trained_prompt_num": trained_prompt_num,
                        "reward_timeout_count": epoch_reward_timeout_count,
                        "reward_placeholder_count": epoch_reward_placeholder_count,
                        "reward_parse_failure_count": epoch_reward_parse_failure_count,
                        "reward_missing_key_count": epoch_reward_missing_key_count,
                        "reward_fallback_score_count": epoch_reward_fallback_score_count,
                    },
                )
                logger.info(
                    f"Local reward metrics saved to {metrics_path}; curve saved to {curve_path}"
                )
            stat_tracker.clear()
        else:
            avg_rewards_all = gathered_rewards_dict["avg"]
            advantages = (avg_rewards_all - avg_rewards_all.mean()) / (
                avg_rewards_all.std() + 1e-4
            )
        # Distribute advantages back to processes
        samples_per_gpu = collated_samples["timesteps"].shape[0]

        if stds.ndim == 1:
            stds = stds[:, None]
        if means.ndim == 1:
            means = means[:, None]
        if advantages.ndim == 1:
            advantages = advantages[:, None]

        if stds.shape[0] == world_size * samples_per_gpu and means.shape[0] == world_size * samples_per_gpu:
            collated_samples["stds"] = torch.from_numpy(
                stds.reshape(world_size, samples_per_gpu, -1)[rank]
            ).to(device)
            collated_samples["means"] = torch.from_numpy(
                means.reshape(world_size, samples_per_gpu, -1)[rank]
            ).to(device)
        else:
            assert False

        if advantages.shape[0] == world_size * samples_per_gpu:
            collated_samples["advantages"] = torch.from_numpy(
                advantages.reshape(world_size, samples_per_gpu, -1)[rank]
            ).to(device)
        else:
            assert False

        if is_main_process(rank):
            logger.info(
                f"Advantages mean: {collated_samples['advantages'].abs().mean().item()}"
            )

        del collated_samples["rewards"]
        del collated_samples["prompt_ids"]

        num_batches = (
            config.sample.num_batches_per_epoch
            * config.sample.train_batch_size
            // config.train.batch_size
        )

        filtered_samples = collated_samples
        if config.sample.ban_prompt:
            valid_mask = (collated_samples["stds"] > config.sample.ban_std_thres) & \
             (collated_samples["means"] < config.sample.ban_mean_thres)
        else:
            valid_mask = torch.ones_like(collated_samples["stds"])
        filtered_samples["valid_mask"] = valid_mask

        total_batch_size_filtered, num_timesteps_filtered = filtered_samples[
            "timesteps"
        ].shape

        # TRAINING
        transformer_ddp.train()  # Sets DDP model and its submodules to train mode.

        # Total number of backward passes before an optimizer step
        effective_grad_accum_steps = (
            config.train.gradient_accumulation_steps * num_train_timesteps
        )

        current_accumulated_steps = 0  # Counter for backward passes
        gradient_update_times = 0

        for inner_epoch in range(config.train.num_inner_epochs):

            if total_batch_size_filtered == 0: # If all samples are banned, break
                print("All samples are banned, break")
                break

            perm = torch.randperm(total_batch_size_filtered, device=device)
            shuffled_filtered_samples = {
                k: v[perm] for k, v in filtered_samples.items()
            }

            perms_time = torch.stack(
                [
                    torch.randperm(num_timesteps_filtered, device=device)
                    for _ in range(total_batch_size_filtered)
                ]
            )
            for key in ["timesteps", "next_timesteps"]:
                shuffled_filtered_samples[key] = shuffled_filtered_samples[key][
                    torch.arange(total_batch_size_filtered, device=device)[:, None],
                    perms_time,
                ]

            training_batch_size = total_batch_size_filtered // num_batches

            samples_batched_list = []
            for k_batch in range(num_batches):
                batch_dict = {}
                start = k_batch * training_batch_size
                end = (k_batch + 1) * training_batch_size
                for key, val_tensor in shuffled_filtered_samples.items():
                    batch_dict[key] = val_tensor[start:end]
                samples_batched_list.append(batch_dict)

            info_accumulated = defaultdict(
                list
            )  # For accumulating stats over one grad acc cycle

            for i, train_sample_batch in tqdm(
                list(enumerate(samples_batched_list)),
                desc=f"Epoch {epoch}.{inner_epoch}: training",
                position=0,
                disable=not is_main_process(rank),
            ):
                current_micro_batch_size = len(train_sample_batch["prompt_embeds"])
                embeds = train_sample_batch["prompt_embeds"]
                pooled_embeds = train_sample_batch["pooled_prompt_embeds"]

                # Loop over timesteps for this micro-batch
                for j_idx, j_timestep_orig_idx in tqdm(
                    enumerate(range(num_train_timesteps)),
                    desc="Timestep",
                    position=1,
                    leave=False,
                    disable=not is_main_process(rank),
                ):
                    assert j_idx == j_timestep_orig_idx
                    x0 = train_sample_batch["latents_clean"]

                    t = train_sample_batch["timesteps"][:, j_idx] / 1000.0

                    t_expanded = t.view(-1, *([1] * (len(x0.shape) - 1)))

                    noise = torch.randn_like(x0.float())

                    xt = (1 - t_expanded) * x0 + t_expanded * noise

                    xt_input = torch.cat(
                        [xt, train_sample_batch["image_latents"]], dim=1
                    )
                    guidance = torch.tensor(
                        [config.sample.guidance_scale], device=device
                    )
                    guidance = guidance.expand(xt_input.shape[0])

                    with torch_autocast(
                        enabled=enable_amp, dtype=mixed_precision_dtype
                    ):
                        transformer_ddp.module.set_adapter("old")
                        with torch.no_grad():
                            # prediction v
                            old_prediction = transformer_ddp(
                                hidden_states=xt_input,
                                guidance=guidance,
                                timestep=t,
                                encoder_hidden_states=embeds,
                                txt_ids=train_sample_batch["text_ids"][0],
                                img_ids=train_sample_batch["latent_ids"][0],
                                pooled_projections=pooled_embeds,
                                return_dict=False,
                            )[0].detach()[:, : xt.shape[1]]
                        transformer_ddp.module.set_adapter("default")

                        # prediction v
                        forward_prediction = transformer_ddp(
                            hidden_states=xt_input,
                            guidance=guidance,
                            timestep=t,
                            encoder_hidden_states=embeds,
                            txt_ids=train_sample_batch["text_ids"][0],
                            img_ids=train_sample_batch["latent_ids"][0],
                            pooled_projections=pooled_embeds,
                            return_dict=False,
                        )[0][:, : xt.shape[1]]

                        with torch.no_grad():  # Reference model part
                            # For LoRA, disable adapter.
                            if config.use_lora:
                                with transformer_ddp.module.disable_adapter():
                                    ref_forward_prediction = transformer_ddp(
                                        hidden_states=xt_input,
                                        guidance=guidance,
                                        timestep=t,
                                        encoder_hidden_states=embeds,
                                        pooled_projections=pooled_embeds,
                                        txt_ids=train_sample_batch["text_ids"][0],
                                        img_ids=train_sample_batch["latent_ids"][0],
                                        return_dict=False,
                                    )[0][:, : xt.shape[1]]
                                transformer_ddp.module.set_adapter("default")
                            else:  # Full model - this requires a frozen copy of the model
                                assert False
                    loss_terms = {}

                    valid_mask = train_sample_batch["valid_mask"][:, j_idx].float()
                    # Policy Gradient Loss
                    advantages_clip = torch.clamp(
                        train_sample_batch["advantages"][:, j_idx],
                        -config.train.adv_clip_max,
                        config.train.adv_clip_max,
                    )
                    if hasattr(config.train, "adv_mode"):
                        if config.train.adv_mode == "positive_only":
                            advantages_clip = torch.clamp(
                                advantages_clip, 0, config.train.adv_clip_max
                            )
                        elif config.train.adv_mode == "negative_only":
                            advantages_clip = torch.clamp(
                                advantages_clip, -config.train.adv_clip_max, 0
                            )
                        elif config.train.adv_mode == "one_only":
                            advantages_clip = torch.where(
                                advantages_clip > 0,
                                torch.ones_like(advantages_clip),
                                torch.zeros_like(advantages_clip),
                            )
                        elif config.train.adv_mode == "binary":
                            advantages_clip = torch.sign(advantages_clip)

                    # normalize advantage
                    normalized_advantages_clip = (
                        advantages_clip / config.train.adv_clip_max
                    ) / 2.0 + 0.5
                    r = torch.clamp(normalized_advantages_clip, 0, 1)
                    loss_terms["x0_norm"] = torch.mean(x0**2).detach()
                    loss_terms["x0_norm_max"] = torch.max(x0**2).detach()
                    loss_terms["old_deviate"] = torch.mean(
                        (forward_prediction - old_prediction) ** 2
                    ).detach()
                    loss_terms["old_deviate_max"] = torch.max(
                        (forward_prediction - old_prediction) ** 2
                    ).detach()

                    positive_prediction = (
                        config.beta * forward_prediction
                        + (1 - config.beta) * old_prediction.detach()
                    )
                    implicit_negative_prediction = (
                        1.0 + config.beta
                    ) * old_prediction.detach() - config.beta * forward_prediction

                    # adaptive weighting
                    x0_prediction = xt - t_expanded * positive_prediction
                    with torch.no_grad():
                        weight_factor = (
                            torch.abs(x0_prediction.double() - x0.double())
                            .mean(dim=tuple(range(1, x0.ndim)), keepdim=True)
                            .clip(min=0.00001)
                        )
                    # if is_main_process(rank):
                    #     print("x0_prediction", x0_prediction, x0_prediction.shape)
                    positive_loss = ((x0_prediction - x0) ** 2 / weight_factor).mean(
                        dim=tuple(range(1, x0.ndim))
                    )
                    negative_x0_prediction = (
                        xt - t_expanded * implicit_negative_prediction
                    )
                    with torch.no_grad():
                        negative_weight_factor = (
                            torch.abs(negative_x0_prediction.double() - x0.double())
                            .mean(dim=tuple(range(1, x0.ndim)), keepdim=True)
                            .clip(min=0.00001)
                        )
                    negative_loss = (
                        (negative_x0_prediction - x0) ** 2 / negative_weight_factor
                    ).mean(dim=tuple(range(1, x0.ndim)))
                    ori_policy_loss = (
                        r * positive_loss * valid_mask / config.beta
                        + (1.0 - r) * negative_loss * valid_mask / config.beta
                    )

                    def mean_by_mask(x, mask):
                        if mask.sum() == 0:
                            return x.sum() * 0
                        return x.sum() / mask.sum()

                    def std_by_mask(x, mask):
                        valid = mask > 0
                        if valid.sum() <= 1:
                            return x.sum() * 0
                        return x[valid].float().std(unbiased=False)

                    raw_advantages = train_sample_batch["advantages"][:, j_idx].float()
                    loss_terms["advantage_abs_mean"] = mean_by_mask(
                        raw_advantages.abs(), valid_mask
                    ).detach()
                    loss_terms["advantage_std"] = std_by_mask(
                        raw_advantages, valid_mask
                    ).detach()
                    loss_terms["advantage_clip_fraction"] = mean_by_mask(
                        (raw_advantages.abs() > config.train.adv_clip_max).float(),
                        valid_mask,
                    ).detach()
                    loss_terms["positive_advantage_ratio"] = mean_by_mask(
                        (raw_advantages > 0).float(),
                        valid_mask,
                    ).detach()
                    loss_terms["valid_fraction"] = valid_mask.mean().detach()
                    valid_advantages = raw_advantages[valid_mask > 0].detach().float()
                    if valid_advantages.numel() > 0:
                        advantage_hist = torch.histc(
                            valid_advantages.clamp(-5.0, 5.0),
                            bins=21,
                            min=-5.0,
                            max=5.0,
                        )
                    else:
                        advantage_hist = torch.zeros(21, device=device)
                    for hist_idx in range(21):
                        loss_terms[f"advantage_hist_{hist_idx:02d}"] = advantage_hist[hist_idx].detach()

                    policy_loss = mean_by_mask(ori_policy_loss * config.train.adv_clip_max, valid_mask)
                    loss = policy_loss
                    loss_terms["policy_loss"] = policy_loss.detach()
                    loss_terms["unweighted_policy_loss"] = (
                        mean_by_mask(ori_policy_loss, valid_mask).detach()
                    )

                    kl_div_loss = (
                        (forward_prediction - ref_forward_prediction) ** 2
                    ).mean(dim=tuple(range(1, x0.ndim))) * valid_mask

                    loss += config.train.beta * mean_by_mask(kl_div_loss, valid_mask)
                    kl_div_loss = mean_by_mask(kl_div_loss, valid_mask)
                    loss_terms["kl_div_loss"] = kl_div_loss.detach()
                    loss_terms["kl_div"] = mean_by_mask(
                        ((forward_prediction - ref_forward_prediction) ** 2).mean(
                            dim=tuple(range(1, x0.ndim))
                        ) * valid_mask, valid_mask
                    ).detach()
                    loss_terms["old_kl_div"] = mean_by_mask(
                        ((old_prediction - ref_forward_prediction) ** 2).mean(
                            dim=tuple(range(1, x0.ndim))
                        ) * valid_mask, valid_mask
                    ).detach()
                    # if is_main_process(rank):
                    #     print("valid_mask", valid_mask.shape, valid_mask, positive_loss, positive_loss.shape)
                    # ori_policy_loss = (
                    #     r * positive_loss / config.beta
                    #     + (1.0 - r) * negative_loss / config.beta
                    # )
                    # policy_loss = (ori_policy_loss * config.train.adv_clip_max).mean()

                    # loss = policy_loss
                    # loss_terms["policy_loss"] = policy_loss.detach()
                    # loss_terms["unweighted_policy_loss"] = (
                    #     ori_policy_loss.mean().detach()
                    # )

                    # kl_div_loss = (
                    #     (forward_prediction - ref_forward_prediction) ** 2
                    # ).mean(dim=tuple(range(1, x0.ndim)))

                    # loss += config.train.beta * torch.mean(kl_div_loss)
                    # kl_div_loss = torch.mean(kl_div_loss)
                    # loss_terms["kl_div_loss"] = torch.mean(kl_div_loss).detach()
                    # loss_terms["kl_div"] = torch.mean(
                    #     ((forward_prediction - ref_forward_prediction) ** 2).mean(
                    #         dim=tuple(range(1, x0.ndim))
                    #     )
                    # ).detach()
                    # loss_terms["old_kl_div"] = torch.mean(
                    #     ((old_prediction - ref_forward_prediction) ** 2).mean(
                    #         dim=tuple(range(1, x0.ndim))
                    #     )
                    # ).detach()

                    loss_terms["total_loss"] = loss.detach()

                    # Scale loss for gradient accumulation and DDP (DDP averages grads, so no need to divide by world_size here)
                    scaled_loss = loss / effective_grad_accum_steps
                    if mixed_precision_dtype == torch.float16:
                        scaler.scale(scaled_loss).backward()  # one accumulation
                    else:
                        scaled_loss.backward()
                    current_accumulated_steps += 1

                    for k_info, v_info in loss_terms.items():
                        info_accumulated[k_info].append(v_info)

                    if current_accumulated_steps % effective_grad_accum_steps == 0:
                        if mixed_precision_dtype == torch.float16:
                            scaler.unscale_(optimizer)
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            transformer_ddp.module.parameters(),
                            config.train.max_grad_norm,
                        )
                        if mixed_precision_dtype == torch.float16:
                            scaler.step(optimizer)
                        else:
                            optimizer.step()
                        gradient_update_times += 1
                        if mixed_precision_dtype == torch.float16:
                            scaler.update()
                        optimizer.zero_grad()

                        log_info = {
                            k: torch.mean(torch.stack(v_list)).item()
                            for k, v_list in info_accumulated.items()
                        }
                        info_tensor = torch.tensor(
                            [log_info[k] for k in sorted(log_info.keys())],
                            device=device,
                        )
                        dist.all_reduce(info_tensor, op=dist.ReduceOp.AVG)
                        reduced_log_info = {
                            k: info_tensor[ki].item()
                            for ki, k in enumerate(sorted(log_info.keys()))
                        }
                        grad_norm_tensor = torch.tensor(
                            float(grad_norm.detach().item() if torch.is_tensor(grad_norm) else grad_norm),
                            device=device,
                        )
                        dist.all_reduce(grad_norm_tensor, op=dist.ReduceOp.AVG)
                        reduced_log_info["grad_norm"] = grad_norm_tensor.item()
                        reduced_log_info["learning_rate"] = float(optimizer.param_groups[0].get("lr", 0.0))
                        if is_main_process(rank):
                            wandb.log(
                                {
                                    "step": global_step,
                                    "gradient_update_times": gradient_update_times,
                                    "epoch": epoch,
                                    "inner_epoch": inner_epoch,
                                    **reduced_log_info,
                                }
                            )
                            training_metrics_path = write_local_training_metrics(
                                config,
                                global_step,
                                epoch,
                                inner_epoch,
                                gradient_update_times,
                                reduced_log_info,
                            )
                            logger.info(
                                f"Local training metrics saved to {training_metrics_path}"
                            )

                        global_step += 1  # gradient step
                        info_accumulated = defaultdict(
                            list
                        )  # Reset for next accumulation cycle

                if (
                    config.train.ema
                    and ema is not None
                    and (current_accumulated_steps % effective_grad_accum_steps == 0)
                ):
                    ema.step(transformer_trainable_parameters, global_step)

        if world_size > 1:
            dist.barrier()

        with torch.no_grad():
            decay = return_decay(global_step, config.decay_type)
            for src_param, tgt_param in zip(
                transformer_trainable_parameters,
                old_transformer_trainable_parameters,
                strict=True,
            ):
                tgt_param.data.copy_(
                    tgt_param.detach().data * decay
                    + src_param.detach().clone().data * (1.0 - decay)
                )

        skip_eval = os.getenv("SKIP_EVAL", "0").strip().lower() in {
            "1",
            "true",
            "yes",
            "y",
        }
        eval_due = global_step > 0 and global_step % config.eval_freq == 0
        if eval_due and not config.debug and not skip_eval:
            eval_fn(
                pipeline,
                test_dataloader,
                text_encoders,
                tokenizers,
                config,
                device,
                rank,
                world_size,
                global_step,
                epoch + 1,
                eval_reward_fn,
                executor,
                mixed_precision_dtype,
                ema,
                transformer_trainable_parameters,
            )
        elif eval_due and skip_eval and is_main_process(rank):
            logger.info("Skipping eval because SKIP_EVAL=1.")

        if (
            (epoch + 1) % config.save_freq == 0
            and is_main_process(rank)
            and not config.debug
        ):
            save_ckpt(
                config.save_dir,
                transformer_ddp,
                global_step,
                rank,
                ema,
                transformer_trainable_parameters,
                config,
                optimizer,
                scaler,
            )

    if is_main_process(rank):
        wandb.finish()
    cleanup_distributed()


if __name__ == "__main__":
    app.run(main)
