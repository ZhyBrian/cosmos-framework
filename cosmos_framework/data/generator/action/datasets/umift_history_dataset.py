# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""UMI-FT forward-dynamics samples with a clean RGB history prefix."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import IterableDataset

from cosmos_framework.data.generator.action.datasets.umift_zarr_dataset import (
    UMIFTZarrIterableDataset,
    _make_umift_action_normalizer,
)


SUPPORTED_HISTORY_FRAMES = (1, 5, 9, 17)
_SOURCE_STRIDE = 2
_FUTURE_FRAMES = 16


def _resize_rgb(rgb: np.ndarray) -> torch.Tensor:
    """Apply the legacy UMI-FT RGB224-to-RGB256 conversion exactly."""
    video = torch.from_numpy(np.ascontiguousarray(rgb)).permute(0, 3, 1, 2).float()
    video = F.interpolate(video, size=(256, 256), mode="bilinear", align_corners=False, antialias=True)
    return video.round().clamp_(0, 255).to(torch.uint8).permute(1, 0, 2, 3).contiguous()


class UMIFTHistoryDataset(UMIFTZarrIterableDataset):
    """Extend each legacy 17-frame UMI-FT window with a clean RGB history prefix."""

    def __init__(
        self, *args: Any, history_frames: int = 1, split: str = "refit_train", **kwargs: Any
    ) -> None:
        history_frames = int(history_frames)
        if history_frames not in SUPPORTED_HISTORY_FRAMES:
            raise ValueError(
                f"history_frames must be one of {SUPPORTED_HISTORY_FRAMES}, got {history_frames}"
            )
        if split.lower() not in ("refit_train", "history"):
            raise ValueError("UMI-FT history dataset split must be 'refit_train' or 'history'")
        self.history_frames = history_frames
        super().__init__(*args, split=split, **kwargs)

    def _load_window(self, episode, start: int, *, physical_action=None) -> dict[str, Any]:
        sample = super()._load_window(episode, start, physical_action=physical_action)

        import zarr

        group = zarr.open_group(self.zarr_path, mode="r")["data"][f"episode_{episode.episode_id}"]
        requested = start - _SOURCE_STRIDE * np.arange(self.history_frames - 1, -1, -1, dtype=np.int64)
        real_mask = requested >= 0
        history_indices = np.maximum(requested, 0)
        history_rgb = np.asarray(group["rgb_0"].oindex[history_indices], dtype=np.uint8)
        history_timestamps = np.asarray(
            group["rgb_time_stamps_0"].oindex[history_indices], dtype=np.float64
        ).reshape(-1)
        history_robot_timestamps = np.asarray(
            group["robot_time_stamps_0"].oindex[history_indices], dtype=np.float64
        ).reshape(-1)
        if not np.isfinite(history_timestamps).all() or not np.isfinite(history_robot_timestamps).all():
            raise ValueError(f"episode_{episode.episode_id} history timestamps must be finite")
        max_alignment_error = float(np.max(np.abs(history_timestamps - history_robot_timestamps)))
        if max_alignment_error > 0.020 + 1e-12:
            raise ValueError(f"episode_{episode.episode_id} history RGB/pose timestamp error exceeds 20 ms")

        history_video = _resize_rgb(history_rgb)
        future_video = sample["video"]
        sample["video"] = torch.cat((history_video, future_video[:, 1:]), dim=1)

        future_source_indices = sample["source_indices"]
        future_timestamps = sample["timestamps"]
        sample["history_frames"] = self.history_frames
        sample["history_source_indices"] = torch.from_numpy(history_indices.copy())
        sample["history_timestamps"] = torch.from_numpy(history_timestamps.copy())
        sample["history_real_mask"] = torch.from_numpy(real_mask.copy())
        sample["history_padding_count"] = int((~real_mask).sum())
        sample["future_source_indices"] = future_source_indices.clone()
        sample["future_timestamps"] = future_timestamps.clone()
        sample["future_pose7_wxyz_m"] = sample["pose7_wxyz_m"].clone()
        sample["video_source_indices"] = torch.cat(
            (sample["history_source_indices"], future_source_indices[1:])
        )
        sample["video_timestamps"] = torch.cat((sample["history_timestamps"], future_timestamps[1:]))

        latent_condition_frames = (self.history_frames - 1) // 4 + 1
        sequence_plan = sample["sequence_plan"]
        sequence_plan.condition_frame_indexes_vision = list(range(latent_condition_frames))
        sequence_plan.condition_frame_indexes_action = list(range(_FUTURE_FRAMES))
        sequence_plan.action_start_frame_offset = self.history_frames
        sample["condition_target_summary"] = (
            f"vision_clean=[0..{latent_condition_frames - 1}];"
            "action_clean=[0..15];"
            f"vision_target=[{latent_condition_frames}..{latent_condition_frames + 3}]"
        )
        return sample


def get_umift_history_sft_dataset(
    zarr_path: str,
    split: str = "refit_train",
    stage: str = "e1",
    seed: int = 42,
    resolution: str | int = "256",
    fps: float = 15.0,
    *,
    history_frames: int = 1,
    mode: str = "forward_dynamics",
    tokenizer_config: dict | None = None,
    max_action_dim: int = 64,
) -> IterableDataset:
    """Build the transformed UMI-FT history stream consumed by the packing loader."""
    if mode != "forward_dynamics":
        raise ValueError("UMI-FT history dataset only supports mode='forward_dynamics'")
    from cosmos_framework.data.generator.action.utils.transforms import ActionTransformPipeline

    transform = ActionTransformPipeline(
        tokenizer_config=tokenizer_config,
        cfg_dropout_rate=0.0,
        max_action_dim=max_action_dim,
        append_viewpoint_info=False,
        append_duration_fps_timestamps=False,
        append_resolution_info=False,
        append_idle_frames=False,
        format_prompt_as_json=False,
        enable_mode_specific_prompt=False,
    )
    return UMIFTHistoryDataset(
        zarr_path,
        split=split,
        stage=stage,
        seed=seed,
        fps=fps,
        resolution=resolution,
        history_frames=history_frames,
        transform=transform,
        action_normalizer=_make_umift_action_normalizer(),
    )


__all__ = [
    "SUPPORTED_HISTORY_FRAMES",
    "UMIFTHistoryDataset",
    "get_umift_history_sft_dataset",
]
