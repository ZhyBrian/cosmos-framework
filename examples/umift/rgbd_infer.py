"""E3-Dout float RGBD inference, isolated from the frozen RGB-only interface."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from examples.umift.infer import _copy_value, _numpy


def validate_rgbd_sample(sample: dict[str, Any]) -> None:
    import torch

    if (sample.get("history_frames") != 5 or sample.get("mode") != "forward_dynamics"
            or sample.get("ai_caption") != ""):
        raise ValueError("E3-Dout requires H5, forward dynamics and the empty prompt")
    video = sample["video"]
    if (not isinstance(video, torch.Tensor) or video.dtype != torch.float32
            or tuple(video.shape) != (3, 21, 256, 512)):
        raise ValueError("RGBD video must be float32 [3,21,256,512]")
    if not torch.isfinite(video).all() or video.min() < -1 or video.max() > 1:
        raise ValueError("RGBD input must be finite and normalized to [-1,1]")
    if sample.get("is_preprocessed") is not True:
        raise ValueError("RGBD sample requires literal is_preprocessed=True")
    if tuple(sample["action"].shape) != (16, 64):
        raise ValueError("E3-Dout requires 16 future actions padded to 64")
    if float(_numpy(sample["conditioning_fps"]).reshape(-1)[0]) != 15:
        raise ValueError("E3-Dout conditioning_fps must remain 15")
    if _numpy(sample["image_size"]).reshape(-1).tolist() != [256, 512, 256, 512]:
        raise ValueError("RGBD canvas image_size must preserve both 256-pixel tiles")
    plan = sample["sequence_plan"]
    if (list(plan.condition_frame_indexes_vision) != [0, 1]
            or list(plan.condition_frame_indexes_action) != list(range(16))
            or plan.action_start_frame_offset != 5):
        raise ValueError("RGBD requires two clean vision latents and action offset 5")
    if not (plan.has_text and plan.has_vision and plan.has_action):
        raise ValueError("RGBD FD requires text, vision and action streams")


def build_rgbd_batch(sample: dict[str, Any]) -> dict[str, Any]:
    """Whitelist conditions and remove the whole future RGB/depth canvas."""
    import torch

    validate_rgbd_sample(sample)
    video = _copy_value(sample["video"])
    video[:, 5:] = 0
    image_size = sample["image_size"].reshape(1, 4)
    batch = {
        "video": [video.unsqueeze(0)],
        "is_preprocessed": True,
        "action": [_copy_value(sample["action"])],
        "raw_action_dim": [sample["raw_action_dim"]],
        "mode": ["forward_dynamics"],
        "ai_caption": [""],
        "image_size": [image_size],
        "fps": torch.tensor([15.0], device=video.device),
        "conditioning_fps": torch.tensor([15.0], device=video.device),
        "num_frames": torch.tensor([21], device=video.device),
        "domain_id": [sample["domain_id"]],
        "sequence_plan": [sample["sequence_plan"]],
    }
    if "text_token_ids" in sample:
        batch["text_token_ids"] = [_copy_value(sample["text_token_ids"])]
    return batch


def run_rgbd_prediction(model: Any, batch: dict[str, Any], *, noise_seed: int,
                        num_steps: int) -> np.ndarray:
    """Return raw float32 THWC normalized canvas; no display clamp in physical metrics."""
    outputs = model.generate_samples_from_batch(
        batch, seed=[int(noise_seed)], num_steps=int(num_steps), guidance=1.0,
        has_negative_prompt=False, upsample_task=None,
    )
    vision = outputs.get("vision")
    if not isinstance(vision, list) or len(vision) != 1:
        raise ValueError("model must return exactly one joint RGBD latent")
    value = _numpy(model.decode(vision[0])).astype(np.float32, copy=False)
    if value.shape != (1, 3, 21, 256, 512) or not np.isfinite(value).all():
        raise ValueError(f"invalid decoded RGBD canvas: {value.shape}")
    return np.moveaxis(value[0], 0, -1).copy()


def split_prediction(canvas: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Separate display-bounded RGB and unbounded metre-valued depth for evaluation."""
    value = np.asarray(canvas)
    if value.ndim != 4 or value.shape[1:] != (256, 512, 3) or not np.isfinite(value).all():
        raise ValueError("prediction must be finite THWC 256x512 RGBD canvas")
    rgb = np.clip((value[:, :, :256] + 1) / 2, 0, 1).astype(np.float32)
    depth = ((value[:, :, 256:].mean(axis=-1) + 1) / 4).astype(np.float32)
    return rgb, depth


def load_rgbd_model(sft_toml: Path, checkpoint: Path):
    from cosmos_framework.configs.toml_config.sft_config import load_experiment_from_toml
    from examples.umift.history_infer import load_history_model

    config = load_experiment_from_toml(sft_toml)
    if config.job.name != "action_fd_umift_edge_rgbd_h5":
        raise ValueError("E3-Dout requires its isolated RGBD experiment configuration")
    expected_targets = (
        (config.dataloader_train._target_, "get_umift_rgbd_packing_dataloader"),
        (config.dataloader_train.dataloader.datasets.umift.dataset._target_, "get_umift_rgbd_sft_dataset"),
    )
    for target, expected in expected_targets:
        name = getattr(target, "__name__", str(target).rsplit(".", 1)[-1])
        if name != expected:
            raise ValueError(f"E3 resolved training target must be {expected}, got {name}")
    model, resolved, evidence = load_history_model(sft_toml, checkpoint, 5, independent_windows=True)
    if evidence["model_key_count"] != 549 or evidence["checkpoint_key_count"] != 549:
        raise ValueError("E3 must preserve exactly the 549 Edge model/checkpoint keys")
    evidence.update(experiment="E3-Dout", canvas_shape=[3, 21, 256, 512],
                    depth_units="metres", depth_representation="linear_gray3_range_0_to_0.5")
    return model, resolved, evidence
