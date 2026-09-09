# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""UMI-FT H5 forward dynamics with a float RGB/depth spatial canvas."""

from __future__ import annotations

from typing import Any, Iterator

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import IterableDataset

from cosmos_framework.data.generator.action.datasets.umift_history_dataset import (
    UMIFTHistoryDataset,
)
from cosmos_framework.data.generator.action.datasets.umift_zarr_dataset import (
    _make_umift_action_normalizer,
    get_umift_packing_dataloader,
)


_HISTORY_FRAMES = 5
_RGB_HEIGHT = 256
_TILE_WIDTH = 256
_CANVAS_WIDTH = 512
_VIDEO_FRAMES = _HISTORY_FRAMES + 16
_DEPTH_MIN_M = 0.0
_DEPTH_MAX_M = 0.5


def _require_float32_tensor(value: Any, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.dtype != torch.float32:
        raise TypeError(f"{name} must use float32, got {value.dtype}")
    return value


def _require_finite(value: torch.Tensor, name: str) -> None:
    if not bool(torch.isfinite(value).all().item()):
        raise ValueError(f"{name} must contain only finite values")


def pack_rgbd_canvas(rgb: torch.Tensor, depth_m: torch.Tensor) -> torch.Tensor:
    """Pack normalized RGB and metre-valued depth into a float32 two-tile canvas."""
    rgb = _require_float32_tensor(rgb, "rgb")
    depth_m = _require_float32_tensor(depth_m, "depth_m")
    if (
        rgb.ndim != 4
        or rgb.shape[0] != 3
        or tuple(rgb.shape[-2:]) != (_RGB_HEIGHT, _TILE_WIDTH)
    ):
        raise ValueError(f"rgb must have shape [3,T,256,256], got {tuple(rgb.shape)}")
    if depth_m.ndim != 3 or tuple(depth_m.shape[-2:]) != (_RGB_HEIGHT, _TILE_WIDTH):
        raise ValueError(
            f"depth_m must have shape [T,256,256], got {tuple(depth_m.shape)}"
        )
    if depth_m.shape[0] != rgb.shape[1]:
        raise ValueError(
            f"rgb and depth_m must have the same frame count, got {rgb.shape[1]} and {depth_m.shape[0]}"
        )
    _require_finite(rgb, "rgb")
    _require_finite(depth_m, "depth_m")
    if bool(((rgb < -1.0) | (rgb > 1.0)).any().item()):
        raise ValueError("rgb values must be in range [-1, 1]")
    if bool(((depth_m < _DEPTH_MIN_M) | (depth_m > _DEPTH_MAX_M)).any().item()):
        raise ValueError("depth_m values must be in range [0, 0.5] metres")

    depth_normalized = (4.0 * depth_m - 1.0).unsqueeze(0).expand(3, -1, -1, -1)
    return torch.cat((rgb, depth_normalized), dim=-1).contiguous()


def unpack_rgbd_canvas(canvas: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Split a decoded float canvas and invert depth without clamping predictions."""
    canvas = _require_float32_tensor(canvas, "canvas")
    if (
        canvas.ndim != 4
        or canvas.shape[0] != 3
        or tuple(canvas.shape[-2:])
        != (
            _RGB_HEIGHT,
            _CANVAS_WIDTH,
        )
    ):
        raise ValueError(
            f"canvas must have shape [3,T,256,512], got {tuple(canvas.shape)}"
        )
    _require_finite(canvas, "canvas")
    rgb = canvas[..., :_TILE_WIDTH]
    depth_normalized = canvas[..., _TILE_WIDTH:].mean(dim=0)
    return rgb, (depth_normalized + 1.0) / 4.0


def _resize_depth_nearest(depth_m: np.ndarray) -> torch.Tensor:
    depth = (
        torch.from_numpy(np.ascontiguousarray(depth_m))
        .to(dtype=torch.float32)
        .unsqueeze(1)
    )
    return F.interpolate(depth, size=(_RGB_HEIGHT, _TILE_WIDTH), mode="nearest")[
        :, 0
    ].contiguous()


class UMIFTRGBDHistoryDataset(UMIFTHistoryDataset):
    """H5 UMI-FT samples with RGB and depth packed after the legacy RGB transform."""

    def __init__(
        self, *args: Any, history_frames: int = _HISTORY_FRAMES, **kwargs: Any
    ) -> None:
        if int(history_frames) != _HISTORY_FRAMES:
            raise ValueError(
                f"UMI-FT RGBD requires history_frames=5, got {history_frames}"
            )
        super().__init__(*args, history_frames=_HISTORY_FRAMES, **kwargs)

    def _read_episode_index(self):
        episodes = super()._read_episode_index()

        import zarr

        data = zarr.open_group(self.zarr_path, mode="r")["data"]
        for episode in episodes:
            group = data[f"episode_{episode.episode_id}"]
            if "depth_0" not in group:
                raise ValueError(f"episode_{episode.episode_id} is missing depth_0")
            shape = tuple(group["depth_0"].shape)
            expected = (episode.length, 224, 224, 3)
            if shape != expected:
                raise ValueError(
                    f"episode_{episode.episode_id} depth_0 must have shape {expected}, got {shape}"
                )
        return episodes

    def _load_window(
        self, episode, start: int, *, physical_action=None
    ) -> dict[str, Any]:
        sample = super()._load_window(episode, start, physical_action=physical_action)

        import zarr

        indices = (
            sample["video_source_indices"]
            .detach()
            .cpu()
            .numpy()
            .astype(np.int64, copy=False)
        )
        if indices.shape != (_VIDEO_FRAMES,):
            raise ValueError(
                f"video_source_indices must have shape ({_VIDEO_FRAMES},), got {indices.shape}"
            )
        group = zarr.open_group(self.zarr_path, mode="r")["data"][
            f"episode_{episode.episode_id}"
        ]
        depth_rgb = np.asarray(group["depth_0"].oindex[indices])
        if depth_rgb.shape != (_VIDEO_FRAMES, 224, 224, 3):
            raise ValueError(
                f"sampled depth must have shape ({_VIDEO_FRAMES},224,224,3), got {depth_rgb.shape}"
            )
        if not np.issubdtype(depth_rgb.dtype, np.floating):
            raise TypeError(
                f"depth_0 must be floating point metres, got {depth_rgb.dtype}"
            )
        if not np.isfinite(depth_rgb).all():
            raise ValueError(
                f"episode_{episode.episode_id} sampled depth must be finite"
            )
        if np.any(depth_rgb < _DEPTH_MIN_M) or np.any(depth_rgb > _DEPTH_MAX_M):
            raise ValueError(
                f"episode_{episode.episode_id} sampled depth must be in range [0, 0.5] metres"
            )
        if not np.array_equal(
            depth_rgb[..., 0], depth_rgb[..., 1]
        ) or not np.array_equal(depth_rgb[..., 0], depth_rgb[..., 2]):
            raise ValueError(
                f"episode_{episode.episode_id} depth_0 channels must be exactly equal"
            )

        video = sample["video"]
        if not isinstance(video, torch.Tensor) or video.dtype != torch.uint8:
            raise TypeError(
                "legacy transformed RGB video must remain uint8 before RGBD packing"
            )
        if tuple(video.shape) != (3, _VIDEO_FRAMES, _RGB_HEIGHT, _TILE_WIDTH):
            raise ValueError(
                "legacy RGB transform must preserve the H5 [3,21,256,256] shape before RGBD packing, "
                f"got {tuple(video.shape)}"
            )
        rgb_normalized = video.to(dtype=torch.float32) / 127.5 - 1.0
        depth_m = _resize_depth_nearest(depth_rgb[..., 0])
        sample["video"] = pack_rgbd_canvas(rgb_normalized, depth_m)
        sample["depth_m"] = depth_m
        sample["depth_observed_mask"] = depth_m > 0.0
        sample["depth_metric_mask"] = (depth_m > 0.0) & (depth_m < _DEPTH_MAX_M)
        sample["depth_cap_mask"] = depth_m >= _DEPTH_MAX_M
        sample["image_size"] = torch.tensor(
            [_RGB_HEIGHT, _CANVAS_WIDTH, _RGB_HEIGHT, _CANVAS_WIDTH],
            dtype=torch.float32,
        )
        if "video_num_frames" in sample:
            sample["video_num_frames"] = _VIDEO_FRAMES
        if "num_frames" in sample:
            sample["num_frames"] = _VIDEO_FRAMES
        sample["is_preprocessed"] = True
        return sample


def get_umift_rgbd_sft_dataset(
    zarr_path: str,
    split: str = "refit_train",
    stage: str = "e1",
    seed: int = 42,
    resolution: str | int = "256",
    fps: float = 15.0,
    *,
    history_frames: int = _HISTORY_FRAMES,
    mode: str = "forward_dynamics",
    tokenizer_config: dict | None = None,
    max_action_dim: int = 64,
) -> IterableDataset:
    """Build the transformed fixed-H5 UMI-FT RGBD stream."""
    if int(history_frames) != _HISTORY_FRAMES:
        raise ValueError(f"UMI-FT RGBD requires history_frames=5, got {history_frames}")
    if mode != "forward_dynamics":
        raise ValueError("UMI-FT RGBD dataset only supports mode='forward_dynamics'")
    from cosmos_framework.data.generator.action.utils.transforms import (
        ActionTransformPipeline,
    )

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
    return UMIFTRGBDHistoryDataset(
        zarr_path,
        split=split,
        stage=stage,
        seed=seed,
        fps=fps,
        resolution=resolution,
        history_frames=_HISTORY_FRAMES,
        transform=transform,
        action_normalizer=_make_umift_action_normalizer(),
    )


def _marker_is_verified_true(value: Any) -> bool:
    if value is True:
        return True
    if isinstance(value, torch.Tensor):
        return (
            value.dtype == torch.bool and value.numel() > 0 and bool(value.all().item())
        )
    if isinstance(value, (list, tuple)):
        return bool(value) and all(_marker_is_verified_true(item) for item in value)
    return False


def _normalize_packed_video_item(item: Any) -> torch.Tensor:
    while isinstance(item, (list, tuple)):
        if len(item) != 1:
            raise ValueError("UMI-FT RGBD expects exactly one vision item per sample")
        item = item[0]
    if not isinstance(item, torch.Tensor):
        raise TypeError("UMI-FT RGBD packed video must be a tensor")
    if not torch.is_floating_point(item):
        raise TypeError("UMI-FT RGBD packed video must be floating point")
    if item.dtype != torch.float32:
        raise TypeError(f"UMI-FT RGBD packed video must use float32, got {item.dtype}")
    if item.ndim == 4:
        item = item.unsqueeze(0)
    if tuple(item.shape) != (1, 3, _VIDEO_FRAMES, _RGB_HEIGHT, _CANVAS_WIDTH):
        raise ValueError(
            "UMI-FT RGBD packed video must have shape [1,3,21,256,512], "
            f"got {tuple(item.shape)}"
        )
    _require_finite(item, "UMI-FT RGBD packed video")
    if bool(((item < -1.0) | (item > 1.0)).any().item()):
        raise ValueError("UMI-FT RGBD packed video values must be in range [-1, 1]")
    return item.contiguous()


class _UMIFTRGBDPackingDataLoader:
    """Validate the packed float boundary while delegating resume state to E1's loader."""

    def __init__(self, delegate: Any) -> None:
        self._delegate = delegate

    def __getattr__(self, name: str) -> Any:
        return getattr(self._delegate, name)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        for batch in self._delegate:
            if not _marker_is_verified_true(batch.get("is_preprocessed")):
                raise ValueError(
                    "UMI-FT RGBD is_preprocessed marker must contain only boolean True values"
                )
            videos = batch.get("video")
            if not isinstance(videos, list) or not videos:
                raise TypeError(
                    "UMI-FT RGBD packed batch must contain a non-empty video list"
                )
            batch["video"] = [_normalize_packed_video_item(item) for item in videos]
            batch["is_preprocessed"] = True
            yield batch

    def __len__(self) -> int:
        return len(self._delegate)

    def set_start_iteration(self, iteration: int) -> None:
        self._delegate.set_start_iteration(iteration)


def get_umift_rgbd_packing_dataloader(**kwargs: Any) -> Any:
    """Wrap the legacy resume-aware packer with the E3 float-video boundary."""
    return _UMIFTRGBDPackingDataLoader(get_umift_packing_dataloader(**kwargs))


__all__ = [
    "UMIFTRGBDHistoryDataset",
    "get_umift_rgbd_packing_dataloader",
    "get_umift_rgbd_sft_dataset",
    "pack_rgbd_canvas",
    "unpack_rgbd_canvas",
]
