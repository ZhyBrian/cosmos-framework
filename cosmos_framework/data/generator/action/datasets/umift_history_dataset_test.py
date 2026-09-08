from __future__ import annotations

import importlib.util
import math
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch
import zarr


_DATASET_DIR = Path(__file__).resolve().parent


def _load_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, _DATASET_DIR / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_BASE_NAME = "cosmos_framework.data.generator.action.datasets.umift_zarr_dataset"
_BASE = _load_module(_BASE_NAME, "umift_zarr_dataset.py")
_HISTORY = _load_module("umift_history_dataset_under_test", "umift_history_dataset.py")


def _quat_z(degrees: float) -> list[float]:
    half = math.radians(degrees) / 2.0
    return [math.cos(half), 0.0, 0.0, math.sin(half)]


def _write_episode(root: zarr.Group, episode_id: int, length: int = 50) -> None:
    group = root.require_group("data").create_group(f"episode_{episode_id}")
    group.attrs["src"] = f"session-{episode_id}#segment"
    rgb = np.zeros((length, 4, 4, 3), dtype=np.uint8)
    for frame in range(length):
        rgb[frame] = frame
    poses = np.zeros((length, 7), dtype=np.float64)
    poses[:, 0] = np.arange(length) * 0.01
    for frame in range(length):
        poses[frame, 3:] = _quat_z(frame * 2.0)
    timestamps = np.arange(length, dtype=np.float64)[:, None] / 30.0
    group.create_dataset("rgb_0", data=rgb, chunks=(16, 4, 4, 3))
    group.create_dataset("ts_pose_fb_0", data=poses)
    group.create_dataset("rgb_time_stamps_0", data=timestamps)
    group.create_dataset("robot_time_stamps_0", data=timestamps)


def _make_store(path: Path, episode_ids=(0,)) -> str:
    root = zarr.open_group(str(path), mode="w")
    for episode_id in episode_ids:
        _write_episode(root, episode_id)
    return str(path)


class _PlanTransform:
    def __init__(self):
        self.input_videos: list[torch.Tensor] = []

    def __call__(self, sample, resolution, **kwargs):
        self.input_videos.append(sample["video"].clone())
        sample["sequence_plan"] = types.SimpleNamespace(
            condition_frame_indexes_vision=[0],
            condition_frame_indexes_action=list(range(16)),
            action_start_frame_offset=1,
        )
        return sample


@pytest.mark.parametrize("history_frames", [1, 5, 9, 17])
def test_history_window_prepends_stride2_pixels_and_preserves_future(history_frames, tmp_path) -> None:
    store = _make_store(tmp_path / f"h{history_frames}.zarr")
    transform = _PlanTransform()
    dataset = _HISTORY.UMIFTHistoryDataset(
        store, split="refit_train", stage="e1", history_frames=history_frames, transform=transform
    )

    sample = dataset.get_window(0, 12)
    expected_history = list(range(12 - 2 * (history_frames - 1), 13, 2))
    expected_history = [max(0, index) for index in expected_history]

    assert sample["video"].shape == (3, history_frames + 16, 256, 256)
    assert sample["history_frames"] == history_frames
    assert sample["history_source_indices"].tolist() == expected_history
    assert sample["source_indices"].tolist() == list(range(12, 45, 2))
    assert sample["future_source_indices"].tolist() == sample["source_indices"].tolist()
    assert sample["future_pose7_wxyz_m"].shape == (17, 7)
    assert sample["video_source_indices"].tolist() == expected_history + list(range(14, 45, 2))
    assert sample["video_timestamps"].shape == (history_frames + 16,)
    assert torch.equal(sample["video"][:, history_frames:], transform.input_videos[0][:, 1:])
    assert sample["physical_action"].shape == (16, 10)
    assert torch.count_nonzero(sample["physical_action"][:, 3:9]).item() > 0
    identity_rot6d = torch.tensor([1, 0, 0, 0, 1, 0], dtype=sample["physical_action"].dtype)
    assert not torch.allclose(sample["physical_action"][0, 3:9], identity_rot6d)
    plan = sample["sequence_plan"]
    assert plan.condition_frame_indexes_vision == list(range((history_frames - 1) // 4 + 1))
    assert plan.condition_frame_indexes_action == list(range(16))
    assert plan.action_start_frame_offset == history_frames


def test_episode_start_repeats_first_frame_and_marks_padding(tmp_path) -> None:
    store = _make_store(tmp_path / "boundary.zarr")
    sample = _HISTORY.UMIFTHistoryDataset(
        store, split="refit_train", stage="e1", history_frames=9, transform=_PlanTransform()
    ).get_window(0, 4)

    assert sample["history_source_indices"].tolist() == [0, 0, 0, 0, 0, 0, 0, 2, 4]
    assert sample["history_real_mask"].tolist() == [False] * 6 + [True] * 3
    assert sample["history_padding_count"] == 6
    assert sample["history_timestamps"].tolist()[:7] == [0.0] * 7
    for index in range(6):
        assert torch.equal(sample["video"][:, index], sample["video"][:, 6])


def test_h1_is_pixel_equivalent_to_the_legacy_transformed_input(tmp_path) -> None:
    store = _make_store(tmp_path / "h1.zarr")
    legacy_transform = _PlanTransform()
    history_transform = _PlanTransform()
    legacy = _BASE.UMIFTZarrIterableDataset(
        store, split="refit_train", stage="e1", transform=legacy_transform
    ).get_window(0, 6)
    history = _HISTORY.UMIFTHistoryDataset(
        store, split="refit_train", stage="e1", history_frames=1, transform=history_transform
    ).get_window(0, 6)

    assert torch.equal(history["video"], legacy["video"])
    assert torch.equal(history_transform.input_videos[0], legacy_transform.input_videos[0])
    assert torch.equal(history["physical_action"], legacy["physical_action"])
    assert torch.equal(history["model_action"], legacy["model_action"])


def test_all_histories_reuse_identical_future_actions(tmp_path) -> None:
    store = _make_store(tmp_path / "actions.zarr")
    samples = [
        _HISTORY.UMIFTHistoryDataset(
            store, split="refit_train", stage="e1", history_frames=history_frames, transform=_PlanTransform()
        ).get_window(0, 12)
        for history_frames in _HISTORY.SUPPORTED_HISTORY_FRAMES
    ]

    for sample in samples[1:]:
        torch.testing.assert_close(sample["physical_action"], samples[0]["physical_action"])
        torch.testing.assert_close(sample["model_action"], samples[0]["model_action"])
        assert sample["action"].shape[0] == 16


def test_future_pixel_changes_do_not_change_history_pixels(tmp_path) -> None:
    store = _make_store(tmp_path / "no_leak.zarr")
    dataset = _HISTORY.UMIFTHistoryDataset(
        store, split="refit_train", stage="e1", history_frames=9, transform=_PlanTransform()
    )
    before = dataset.get_window(0, 12)["video"][:, :9].clone()
    root = zarr.open_group(store, mode="a")
    future = root["data/episode_0/rgb_0"][:]
    future[14:45] = 255
    root["data/episode_0/rgb_0"][:] = future

    after = dataset.get_window(0, 12)["video"][:, :9]
    assert torch.equal(after, before)


def test_refit_split_keeps_all_56_episodes_and_legacy_defaults(tmp_path) -> None:
    store = _make_store(tmp_path / "splits.zarr", tuple(range(59)))
    dataset = _HISTORY.UMIFTHistoryDataset(
        store, split="refit_train", stage="e1", history_frames=5, transform=_PlanTransform()
    )

    assert [episode.episode_id for episode in dataset._episodes] == [
        episode_id for episode_id in range(59) if episode_id not in (13, 43, 49)
    ]
    assert _BASE.TRAIN_EPISODES == tuple(i for i in range(50) if i not in (13, 43, 49))
    assert _BASE.HISTORY_EPISODES == (13, 43, 49)


def test_history_split_supports_explicit_held_out_window(tmp_path) -> None:
    store = _make_store(tmp_path / "held-out.zarr", (13, 43, 49))
    dataset = _HISTORY.UMIFTHistoryDataset(
        store, split="history", stage="e1", history_frames=17, transform=_PlanTransform()
    )

    sample = dataset.get_window(13, 8)
    assert sample["episode_id"] == 13
    assert sample["video"].shape[1] == 33
    assert sample["history_padding_count"] == 12


@pytest.mark.parametrize("bad_history", [0, 2, 13, 18])
def test_rejects_unsupported_history_lengths(bad_history, tmp_path) -> None:
    store = _make_store(tmp_path / f"bad-{bad_history}.zarr")
    with pytest.raises(ValueError, match="history_frames"):
        _HISTORY.UMIFTHistoryDataset(
            store, split="refit_train", stage="e1", history_frames=bad_history, transform=_PlanTransform()
        )


def test_rejects_non_e2_split(tmp_path) -> None:
    store = _make_store(tmp_path / "split.zarr")
    with pytest.raises(ValueError, match="refit_train.*history"):
        _HISTORY.UMIFTHistoryDataset(
            store, split="train", stage="e1", history_frames=5, transform=_PlanTransform()
        )
