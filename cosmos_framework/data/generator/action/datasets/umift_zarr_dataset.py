# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Streaming adapter from the UMI-FT Zarr store to Cosmos action SFT samples."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import IterableDataset, get_worker_info


HISTORY_EPISODES = (13, 43, 49)
TRAIN_EPISODES = tuple(i for i in range(50) if i not in HISTORY_EPISODES)
DEV_EPISODES = tuple(range(50, 59))

_SOURCE_STRIDE = 2
_VIDEO_FRAMES = 17
_ACTION_STEPS = 16
_SOURCE_SPAN = (_VIDEO_FRAMES - 1) * _SOURCE_STRIDE
_OVERFIT_STARTS = (0, 64, 128, 192)
_EVAL_START_STEP = 32

_NORMALIZER_PATH = Path(__file__).resolve().parent.parent / "normalizer_stats/umi_lerobot_stats.json"


def _load_official_umi_quantiles() -> tuple[np.ndarray, np.ndarray]:
    with _NORMALIZER_PATH.open() as handle:
        stats = json.load(handle)
    q01 = np.asarray(stats["q01"][:10], dtype=np.float32)
    q99 = np.asarray(stats["q99"][:10], dtype=np.float32)
    if q01.shape != (10,) or q99.shape != (10,):
        raise ValueError(f"official UMI quantiles must each contain at least 10 columns: {_NORMALIZER_PATH}")
    return q01, q99


_UMI_Q01, _UMI_Q99 = _load_official_umi_quantiles()
_NORMALIZER_SUMMARY = json.dumps(
    {"name": "umi_lerobot_quantile", "method": "quantile", "columns": "right_arm_first_10"},
    sort_keys=True,
)
_RGB_TRANSFORM_SUMMARY = json.dumps(
    {"source": "RGB224", "target": "RGB256", "interpolation": "bilinear", "antialias": True, "random": False},
    sort_keys=True,
)


def _quat_wxyz_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(quaternion, dtype=np.float64)
    norms = np.linalg.norm(quaternion, axis=-1, keepdims=True)
    if np.any(norms < 1e-12):
        raise ValueError("pose contains a zero-norm quaternion")
    q = quaternion / norms
    w, x, y, z = np.moveaxis(q, -1, 0)
    return np.stack(
        [
            1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
            2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
            2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
        ],
        axis=-1,
    ).reshape(*q.shape[:-1], 3, 3)


def _framewise_actions(poses_wxyz: np.ndarray) -> np.ndarray:
    """Compute ``Q_i^-1 Q_(i+1)`` as body translation + column rot6d + g=0."""
    poses = np.asarray(poses_wxyz, dtype=np.float64)
    if poses.ndim != 2 or poses.shape[1] != 7 or poses.shape[0] < 2:
        raise ValueError(f"poses must have shape (T>=2, 7), got {poses.shape}")
    rotations = _quat_wxyz_to_matrix(poses[:, 3:])
    relative_rotation = np.swapaxes(rotations[:-1], -1, -2) @ rotations[1:]
    relative_translation = np.einsum(
        "tij,tj->ti", np.swapaxes(rotations[:-1], -1, -2), poses[1:, :3] - poses[:-1, :3]
    )
    # Framework rot6d is the first two matrix columns, column by column.
    rot6d = relative_rotation[:, :, :2].transpose(0, 2, 1).reshape(-1, 6)
    gripper = np.zeros((poses.shape[0] - 1, 1), dtype=np.float64)
    return np.concatenate((relative_translation, rot6d, gripper), axis=-1).astype(np.float32)


def normalize_umift_action(action: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
    """Apply the official UMI right-arm 10D quantile normalizer once."""
    is_tensor = isinstance(action, torch.Tensor)
    action_np = action.detach().cpu().numpy() if is_tensor else np.asarray(action)
    denominator = np.maximum(_UMI_Q99 - _UMI_Q01, 1e-8)
    result = (2.0 * (np.asarray(action_np, dtype=np.float32) - _UMI_Q01) / denominator - 1.0).astype(np.float32)
    if is_tensor:
        return torch.from_numpy(result).to(device=action.device, dtype=action.dtype)
    return result


def denormalize_umift_action(action: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
    """Invert :func:`normalize_umift_action` for evaluation and action substitutions."""
    is_tensor = isinstance(action, torch.Tensor)
    action_np = action.detach().cpu().numpy() if is_tensor else np.asarray(action)
    result = (0.5 * (np.asarray(action_np, dtype=np.float32) + 1.0) * (_UMI_Q99 - _UMI_Q01) + _UMI_Q01)
    result = result.astype(np.float32)
    if is_tensor:
        return torch.from_numpy(result).to(device=action.device, dtype=action.dtype)
    return result


def _make_umift_action_normalizer() -> Any:
    """Build the framework-native invertible affine form of UMI quantile normalization."""
    from cosmos_framework.data.generator.action.utils.action_processing import ActionAffineNormalization

    q01 = torch.from_numpy(_UMI_Q01.copy())
    q99 = torch.from_numpy(_UMI_Q99.copy())
    return ActionAffineNormalization(
        offset=(q99 + q01) / 2.0,
        scale=(q99 - q01).clamp(min=1e-8) / 2.0,
    )


def _split_episode_ids(split: str, episode_count: int = 59) -> tuple[int, ...]:
    split_key = split.lower()
    if split_key == "train":
        selected = TRAIN_EPISODES
    elif split_key in ("dev", "val", "validation"):
        selected = DEV_EPISODES
    elif split_key == "history":
        selected = HISTORY_EPISODES
    else:
        raise ValueError("split must be one of: train, dev, history")
    return tuple(i for i in selected if i < episode_count)


@dataclass(frozen=True)
class _Episode:
    episode_id: int
    source_id: str
    session_id: str
    length: int

    @property
    def window_count(self) -> int:
        return max(0, self.length - _SOURCE_SPAN)


class UMIFTZarrIterableDataset(IterableDataset):
    """Read UMI-FT windows lazily and optionally pass each through the action transform.

    ``state_dict`` records the next global draw index. Exact resume is supported for
    ``num_workers=0``. With DataLoader workers, checkpoint each worker cursor at the
    loader/distributor layer because worker-process mutations are not visible to the parent.
    """

    def __init__(
        self,
        zarr_path: str,
        *,
        split: str = "train",
        stage: str = "e1",
        seed: int = 42,
        fps: float = 15.0,
        resolution: str | int = "256",
        transform: Callable[[dict[str, Any], str | int], dict[str, Any]] | None = None,
        action_normalizer: Any = None,
    ) -> None:
        super().__init__()
        self.zarr_path = str(Path(zarr_path))
        self.split = split.lower()
        self.stage = stage.lower()
        self.seed = int(seed)
        self.fps = float(fps)
        self.resolution = resolution
        self.transform = transform
        self.action_normalizer = action_normalizer
        self.shard_world_size = 1
        self.shard_rank = 0
        self._next_draw_index = 0
        self._local_sample_offset = 0
        if self.stage not in ("smoke", "overfit", "e1"):
            raise ValueError("stage must be one of: smoke, overfit, e1")
        if self.fps <= 0:
            raise ValueError("fps must be positive")
        if str(self.resolution) != "256":
            raise ValueError("UMI-FT E1 requires resolution='256'")
        self._episodes = self._read_episode_index()
        if not self._episodes:
            raise ValueError(f"no usable episodes for split={self.split!r}")

    def _read_episode_index(self) -> tuple[_Episode, ...]:
        import zarr

        root = zarr.open_group(self.zarr_path, mode="r")
        data = root["data"]
        discovered = sorted(
            int(name.removeprefix("episode_"))
            for name in data.group_keys()
            if name.startswith("episode_") and name.removeprefix("episode_").isdigit()
        )
        allowed = set(_split_episode_ids(self.split, max(discovered, default=-1) + 1))
        episodes: list[_Episode] = []
        for episode_id in discovered:
            if episode_id not in allowed:
                continue
            group = data[f"episode_{episode_id}"]
            lengths = {
                int(group["rgb_0"].shape[0]),
                int(group["ts_pose_fb_0"].shape[0]),
                int(group["rgb_time_stamps_0"].shape[0]),
                int(group["robot_time_stamps_0"].shape[0]),
            }
            if len(lengths) != 1:
                raise ValueError(f"episode_{episode_id} RGB/pose/timestamp lengths differ: {sorted(lengths)}")
            source_id = str(group.attrs.get("src", f"episode_{episode_id}"))
            session_id = source_id.split("#", 1)[0]
            episode = _Episode(episode_id, source_id, session_id, lengths.pop())
            if episode.window_count:
                episodes.append(episode)
        return tuple(episodes)

    @property
    def total_images(self) -> int:
        return sum(episode.window_count for episode in self._episodes)

    def __len__(self) -> int:
        if self.stage == "smoke":
            return 1
        if self.stage == "overfit":
            return sum(start + _SOURCE_SPAN < self._episodes[0].length for start in _OVERFIT_STARTS)
        if self.split != "train":
            return sum(len(range(0, episode.window_count, _EVAL_START_STEP)) for episode in self._episodes)
        return self.total_images

    def state_dict(self) -> dict[str, int]:
        return {
            "next_draw_index": self._next_draw_index,
            "local_sample_offset": self._local_sample_offset,
        }

    def load_state_dict(self, state: dict[str, int]) -> None:
        next_draw_index = int(state["next_draw_index"])
        if next_draw_index < 0:
            raise ValueError("next_draw_index must be non-negative")
        self._next_draw_index = next_draw_index
        self._local_sample_offset = int(state.get("local_sample_offset", 0))
        if self._local_sample_offset < 0:
            raise ValueError("local_sample_offset must be non-negative")

    def set_start_iteration(self, iteration: int) -> None:
        """Restore the next rank-local microbatch position before iteration starts."""
        iteration = int(iteration)
        if iteration < 0:
            raise ValueError("iteration must be non-negative")
        worker = get_worker_info()
        num_workers = worker.num_workers if worker is not None else 1
        if num_workers != 1:
            raise RuntimeError("exact UMI-FT resume requires num_workers=0")
        self._local_sample_offset = iteration
        self._next_draw_index = iteration * max(1, int(self.shard_world_size))

    def get_window(
        self,
        episode_id: int,
        start: int,
        *,
        physical_action: torch.Tensor | np.ndarray | None = None,
    ) -> dict[str, Any]:
        """Load one explicit legal window, optionally substituting a physical 16x10 action."""
        episode = next((item for item in self._episodes if item.episode_id == int(episode_id)), None)
        if episode is None:
            raise ValueError(f"episode_{episode_id} is not in split={self.split!r}")
        if int(start) < 0 or int(start) + _SOURCE_SPAN >= episode.length:
            raise ValueError(f"illegal window episode_{episode_id}/start={start}")
        return self._load_window(episode, int(start), physical_action=physical_action)

    def _finite_windows(self) -> list[tuple[_Episode, int]]:
        if self.stage == "smoke":
            return [(self._episodes[0], 0)]
        if self.stage == "overfit":
            episode = self._episodes[0]
            return [(episode, start) for start in _OVERFIT_STARTS if start + _SOURCE_SPAN < episode.length]
        return [
            (episode, start)
            for episode in self._episodes
            for start in range(0, episode.window_count, _EVAL_START_STEP)
        ]

    def _training_draws(self, start_index: int) -> Iterator[tuple[int, _Episode, int]]:
        by_session: dict[str, list[_Episode]] = defaultdict(list)
        for episode in self._episodes:
            by_session[episode.session_id].append(episode)
        sessions = sorted(by_session)
        rng = np.random.Generator(np.random.PCG64(self.seed))
        for draw_index in range(start_index):
            session = sessions[int(rng.integers(len(sessions)))]
            total = sum(ep.window_count for ep in by_session[session])
            rng.integers(total)
        draw_index = start_index
        while True:
            session = sessions[int(rng.integers(len(sessions)))]
            episodes = by_session[session]
            offset = int(rng.integers(sum(ep.window_count for ep in episodes)))
            for episode in episodes:
                if offset < episode.window_count:
                    yield draw_index, episode, offset
                    break
                offset -= episode.window_count
            draw_index += 1

    def __iter__(self) -> Iterator[dict[str, Any]]:
        worker = get_worker_info()
        worker_id = worker.id if worker is not None else 0
        num_workers = worker.num_workers if worker is not None else 1
        total_shards = max(1, int(self.shard_world_size) * num_workers)
        global_shard = int(self.shard_rank) * num_workers + worker_id
        if not 0 <= global_shard < total_shards:
            raise ValueError("invalid shard_rank/shard_world_size")

        if self.stage == "e1" and self.split == "train":
            draws = self._training_draws(self._next_draw_index)
        elif self.stage in ("smoke", "overfit"):
            windows = self._finite_windows()
            local_index = self._local_sample_offset

            def repeated_draws() -> Iterator[tuple[int, _Episode, int]]:
                nonlocal local_index
                while True:
                    window_index = global_shard if self.stage == "overfit" else local_index + global_shard
                    episode, start = windows[window_index % len(windows)]
                    yield local_index, episode, start
                    local_index += 1

            for local_index, episode, start in repeated_draws():
                sample = self._load_window(episode, start)
                self._local_sample_offset = local_index + 1
                yield sample
            return
        else:
            windows = self._finite_windows()
            draws = ((i, episode, start) for i, (episode, start) in enumerate(windows))

        for draw_index, episode, start in draws:
            if draw_index % total_shards != global_shard:
                continue
            sample = self._load_window(episode, start)
            self._next_draw_index = draw_index + 1
            yield sample

    def _load_window(
        self,
        episode: _Episode,
        start: int,
        *,
        physical_action: torch.Tensor | np.ndarray | None = None,
    ) -> dict[str, Any]:
        import zarr

        root = zarr.open_group(self.zarr_path, mode="r")
        group = root["data"][f"episode_{episode.episode_id}"]
        indices = start + _SOURCE_STRIDE * np.arange(_VIDEO_FRAMES, dtype=np.int64)
        if int(indices[-1]) >= episode.length:
            raise IndexError(f"window crosses episode_{episode.episode_id}")
        rgb = np.asarray(group["rgb_0"].oindex[indices], dtype=np.uint8)
        poses = np.asarray(group["ts_pose_fb_0"].oindex[indices], dtype=np.float64)
        rgb_timestamps = np.asarray(group["rgb_time_stamps_0"].oindex[indices], dtype=np.float64).reshape(-1)
        robot_timestamps = np.asarray(group["robot_time_stamps_0"].oindex[indices], dtype=np.float64).reshape(-1)
        if not np.isfinite(poses).all():
            raise ValueError(f"episode_{episode.episode_id} pose values must be finite")
        if not np.isfinite(rgb_timestamps).all() or not np.isfinite(robot_timestamps).all():
            raise ValueError(f"episode_{episode.episode_id} timestamps must be finite")
        if np.any(np.diff(rgb_timestamps) <= 0) or np.any(np.diff(robot_timestamps) <= 0):
            raise ValueError(f"episode_{episode.episode_id} timestamps must be strictly increasing")
        max_alignment_error = float(np.max(np.abs(rgb_timestamps - robot_timestamps)))
        if max_alignment_error > 0.020 + 1e-12:
            raise ValueError(
                f"episode_{episode.episode_id} RGB/pose timestamp error {max_alignment_error:.6f}s exceeds 20 ms"
            )
        relative_timestamps = rgb_timestamps - rgb_timestamps[0]
        ideal_timestamps = np.arange(_VIDEO_FRAMES, dtype=np.float64) / self.fps
        max_grid_error = float(np.max(np.abs(relative_timestamps - ideal_timestamps)))
        if max_grid_error > 0.020 + 1e-12:
            raise ValueError(
                f"episode_{episode.episode_id} timestamp error {max_grid_error:.6f}s exceeds "
                f"the 20 ms tolerance from the ideal {self.fps:g} Hz grid"
            )
        physical_action_np = _framewise_actions(poses)
        if physical_action is not None:
            action_array = (
                physical_action.detach().cpu().numpy()
                if isinstance(physical_action, torch.Tensor)
                else physical_action
            )
            physical_action_np = np.asarray(
                action_array,
                dtype=np.float32,
            )
            if physical_action_np.shape != (_ACTION_STEPS, 10):
                raise ValueError(f"physical_action must have shape (16, 10), got {physical_action_np.shape}")
        physical_action = torch.from_numpy(physical_action_np)
        model_action = (
            self.action_normalizer.normalize_action(physical_action)
            if self.action_normalizer is not None
            else normalize_umift_action(physical_action)
        )
        video = torch.from_numpy(np.ascontiguousarray(rgb)).permute(0, 3, 1, 2).float()
        video = F.interpolate(video, size=(256, 256), mode="bilinear", align_corners=False, antialias=True)
        video = video.round().clamp_(0, 255).to(torch.uint8).permute(1, 0, 2, 3).contiguous()
        sample: dict[str, Any] = {
            "ai_caption": "",
            "video": video,
            "action": physical_action,
            "conditioning_fps": torch.tensor(self.fps, dtype=torch.float32),
            "mode": "forward_dynamics",
            "domain_id": torch.tensor(6, dtype=torch.long),
            "viewpoint": "wrist_view",
            "idle_frames": torch.tensor(0, dtype=torch.long),
            "episode_id": episode.episode_id,
            "source_id": episode.source_id,
            "session_id": episode.session_id,
            "window_start": start,
            "source_indices": torch.from_numpy(indices.copy()),
            "timestamps": torch.from_numpy(rgb_timestamps.copy()),
            "pose7_wxyz_m": torch.from_numpy(poses.copy()),
            "physical_action": physical_action.clone(),
            "model_action": model_action.clone(),
            "normalizer_summary": _NORMALIZER_SUMMARY,
            "rgb_transform_summary": _RGB_TRANSFORM_SUMMARY,
            "condition_target_summary": "vision_clean=[0];action_clean=[0..15];vision_target=[1..16]",
        }
        if self.transform is not None:
            return self.transform(
                sample,
                self.resolution,
                action_normalizer=self.action_normalizer,
                action_valid_mask=torch.ones(10, dtype=torch.bool),
            )
        return sample


def get_umift_zarr_sft_dataset(
    zarr_path: str,
    split: str = "train",
    stage: str = "e1",
    seed: int = 42,
    resolution: str | int = "256",
    fps: float = 15.0,
    *,
    mode: str = "forward_dynamics",
    tokenizer_config: dict | None = None,
    max_action_dim: int = 64,
) -> IterableDataset:
    """Build the transformed UMI-FT FD stream consumed by the packing loader."""
    if mode != "forward_dynamics":
        raise ValueError("UMI-FT E1 only supports mode='forward_dynamics'")
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
    action_normalizer = _make_umift_action_normalizer()
    return UMIFTZarrIterableDataset(
        zarr_path,
        split=split,
        stage=stage,
        seed=seed,
        fps=fps,
        resolution=resolution,
        transform=transform,
        action_normalizer=action_normalizer,
    )


def get_umift_dataloader_generator(seed: int = 42) -> torch.Generator:
    """Return an isolated DataLoader base-seed generator for exact RNG resume.

    ``DataLoader.__iter__`` draws ``_base_seed`` even with ``num_workers=0``.
    Supplying this generator prevents that bookkeeping draw from advancing the
    model's global Torch CPU RNG restored by DCP.
    """
    return torch.Generator().manual_seed(int(seed))


def get_umift_packing_dataloader(**kwargs: Any) -> Any:
    """Build a one-sample packer that propagates trainer resume offsets to UMI-FT.

    With ``max_samples_per_batch=1`` the packing loop consumes exactly one source
    window per yielded microbatch.  The parent loader prewarms one sample during
    construction; resume discards that buffer and its already-started iterator,
    then creates a fresh iterator after restoring the dataset offset.
    """
    from cosmos_framework.data.generator.joint_dataloader import PackingDataLoader

    if kwargs.get("max_samples_per_batch") != 1:
        raise ValueError("exact UMI-FT resume requires max_samples_per_batch=1")

    class _ResumeAwareUMIFTPackingDataLoader(PackingDataLoader):
        def set_start_iteration(self, iteration: int) -> None:
            super().set_start_iteration(iteration)
            rank_partitioned_loader = self.dataloader_list[0]
            dataset = getattr(rank_partitioned_loader, "dataset", None)
            if not isinstance(dataset, UMIFTZarrIterableDataset):
                raise TypeError("UMI-FT packing loader expected UMIFTZarrIterableDataset")
            dataset.set_start_iteration(iteration)
            for buffer in self.buffers:
                buffer.clear()
            self.dataloaders = [iter(loader) for loader in self.dataloader_list]

    return _ResumeAwareUMIFTPackingDataLoader(**kwargs)


__all__ = [
    "UMIFTZarrIterableDataset",
    "denormalize_umift_action",
    "get_umift_packing_dataloader",
    "get_umift_dataloader_generator",
    "get_umift_zarr_sft_dataset",
    "normalize_umift_action",
]
