"""Inference contracts for E2-H, without changing the frozen E1 interface."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from examples.umift.infer import (
    _copy_value,
    _numpy,
    _structure_runtime_configs,
    strict_dcp_key_evidence,
    validate_checkpoint_path,
)


def validate_history_sample(sample: dict[str, Any]) -> int:
    history_frames = int(sample["history_frames"])
    if history_frames not in (1, 5, 9, 17):
        raise ValueError("history_frames must be 1, 5, 9, or 17")
    if sample.get("mode") != "forward_dynamics" or sample.get("ai_caption") != "":
        raise ValueError("E2-H requires FD with the frozen empty prompt")
    if tuple(sample["video"].shape) != (3, history_frames + 16, 256, 256):
        raise ValueError("video must contain exactly H history and 16 future RGB frames")
    if tuple(sample["action"].shape) != (16, 64):
        raise ValueError("E2-H requires the same 16 future actions, padded to 64")
    if float(_numpy(sample["conditioning_fps"]).reshape(-1)[0]) != 15.0:
        raise ValueError("conditioning_fps must remain 15")
    plan = sample["sequence_plan"]
    expected = list(range((history_frames - 1) // 4 + 1))
    if list(plan.condition_frame_indexes_vision) != expected:
        raise ValueError("vision condition must be exactly the causal latent prefix")
    if list(plan.condition_frame_indexes_action) != list(range(16)):
        raise ValueError("exactly 16 future actions must be clean conditions")
    if plan.action_start_frame_offset != history_frames:
        raise ValueError("the first action destination must align with RGB index H")
    if not (plan.has_text and plan.has_vision and plan.has_action):
        raise ValueError("FD requires text, vision, and action streams")
    return history_frames


def build_history_batch(sample: dict[str, Any]) -> dict[str, Any]:
    """Remove future truth before handing the sample to the model."""
    import torch

    history_frames = validate_history_sample(sample)
    video = _copy_value(sample["video"])
    video[:, history_frames:] = 0
    image_size = sample["image_size"]
    if image_size.ndim == 1:
        image_size = image_size.unsqueeze(0)
    batch = {
        "video": [video],
        "action": [sample["action"]],
        "raw_action_dim": [sample["raw_action_dim"]],
        "mode": ["forward_dynamics"],
        "ai_caption": [""],
        "image_size": [image_size],
        "fps": torch.tensor([15.0], device=video.device),
        "conditioning_fps": torch.tensor([15.0], device=video.device),
        "num_frames": torch.tensor([history_frames + 16], device=video.device),
        "domain_id": [sample["domain_id"]],
        "sequence_plan": [sample["sequence_plan"]],
    }
    if "text_token_ids" in sample:
        batch["text_token_ids"] = [sample["text_token_ids"]]
    return batch


def run_history_prediction(
    model: Any, batch: dict[str, Any], *, history_frames: int, noise_seed: int, num_steps: int
) -> np.ndarray:
    outputs = model.generate_samples_from_batch(
        batch, seed=[int(noise_seed)], num_steps=int(num_steps), guidance=1.0,
        has_negative_prompt=False, upsample_task=None,
    )
    vision = outputs.get("vision")
    if not isinstance(vision, list) or len(vision) != 1:
        raise ValueError("model must return exactly one vision latent")
    value = _numpy(model.decode(vision[0])).astype(np.float32, copy=False)
    if value.shape != (1, 3, history_frames + 16, 256, 256):
        raise ValueError(f"unexpected decoded E2-H shape: {value.shape}")
    if not np.isfinite(value).all():
        raise ValueError("decoded E2-H video contains non-finite values")
    return np.moveaxis((np.clip(value[0], -1, 1) + 1) / 2, 0, -1).astype(np.float32)


def load_history_model(
    sft_toml: Path, checkpoint: Path, history_frames: int, *, independent_windows: bool = True
) -> tuple[Any, Any, dict[str, Any]]:
    from cosmos_framework.configs.toml_config.sft_config import load_experiment_from_toml
    from cosmos_framework.inference.common.config import unstructure_config
    from cosmos_framework.inference.model import Cosmos3OmniConfig, Cosmos3OmniModel

    validate_checkpoint_path(checkpoint)
    if history_frames not in (1, 5, 9, 17):
        raise ValueError("unsupported history length")
    resolved = load_experiment_from_toml(sft_toml)
    model_dict = unstructure_config(resolved.model, invalid="ignore")
    cfg = model_dict["config"]
    if not cfg.get("action_gen") or not cfg.get("vision_gen") or cfg.get("sound_gen"):
        raise ValueError("resolved model is not the Edge vision+action FD graph")
    if str(cfg.get("resolution")) != "256" or cfg["tokenizer"]["encode_exact_durations"] != [history_frames + 16]:
        raise ValueError("resolved model does not match this H and resolution")
    if independent_windows and resolved.trainer.callbacks.compile_tokenizer.enabled:
        raise ValueError("independent inference requires tokenizer compilation disabled")
    parallelism, compile_config, quantization = _structure_runtime_configs(
        cfg, independent_windows=independent_windows
    )
    wrapper = Cosmos3OmniModel.from_pretrained_dcp(
        checkpoint, config=Cosmos3OmniConfig(model=model_dict),
        parallelism_config=parallelism, compile_config=compile_config,
        quantization_config=quantization,
    )
    wrapper.eval()
    evidence = strict_dcp_key_evidence(wrapper.model, checkpoint)
    evidence.update({"checkpoint": str(checkpoint.resolve()), "history_frames": history_frames,
                     "loader": "Cosmos3OmniModel.from_pretrained_dcp"})
    return wrapper.model, resolved, evidence
