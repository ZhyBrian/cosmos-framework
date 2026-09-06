from __future__ import annotations

import math
import importlib.util
import sys
import types
from itertools import islice
from pathlib import Path

import numpy as np
import pytest
import torch
import zarr

_MODULE_PATH = Path(__file__).with_name("umift_zarr_dataset.py")
_SPEC = importlib.util.spec_from_file_location("umift_zarr_dataset_under_test", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)
HISTORY_EPISODES = _MODULE.HISTORY_EPISODES
TRAIN_EPISODES = _MODULE.TRAIN_EPISODES
UMIFTZarrIterableDataset = _MODULE.UMIFTZarrIterableDataset
_framewise_actions = _MODULE._framewise_actions
_split_episode_ids = _MODULE._split_episode_ids
get_umift_zarr_sft_dataset = _MODULE.get_umift_zarr_sft_dataset
get_umift_dataloader_generator = _MODULE.get_umift_dataloader_generator
normalize_umift_action = _MODULE.normalize_umift_action
denormalize_umift_action = _MODULE.denormalize_umift_action


def _load_real_action_processing_module():
    module_name = "cosmos_framework.data.generator.action.utils.action_processing"
    module_path = _MODULE_PATH.resolve().parent.parent / "utils/action_processing.py"
    fake_utils = types.ModuleType("cosmos_framework.utils")
    fake_utils.log = types.SimpleNamespace(info=lambda *args, **kwargs: None)
    sys.modules.setdefault("cosmos_framework.utils", fake_utils)
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _quat_z(degrees: float) -> list[float]:
    half = math.radians(degrees) / 2.0
    return [math.cos(half), 0.0, 0.0, math.sin(half)]


def _write_episode(root: zarr.Group, episode_id: int, length: int, *, bad_timestamps: bool = False) -> None:
    group = root.require_group("data").create_group(f"episode_{episode_id}")
    group.attrs["src"] = f"session-{episode_id // 2}#seg{episode_id % 2}"
    rgb = np.arange(length * 4 * 4 * 3, dtype=np.uint8).reshape(length, 4, 4, 3)
    poses = np.zeros((length, 7), dtype=np.float64)
    poses[:, 0] = np.arange(length, dtype=np.float64) * 0.01
    poses[:, 3] = 1.0
    timestamps = np.arange(length, dtype=np.float64)[:, None] / 30.0
    if bad_timestamps:
        timestamps[4] = timestamps[2]
    group.create_dataset("rgb_0", data=rgb, chunks=(16, 4, 4, 3))
    group.create_dataset("ts_pose_fb_0", data=poses)
    group.create_dataset("rgb_time_stamps_0", data=timestamps)
    group.create_dataset("robot_time_stamps_0", data=timestamps)


def _make_store(path, lengths=(40, 40), *, bad_timestamps: bool = False) -> str:
    root = zarr.open_group(str(path), mode="w")
    for episode_id, length in enumerate(lengths):
        _write_episode(root, episode_id, length, bad_timestamps=bad_timestamps and episode_id == 0)
    return str(path)


def test_framewise_actions_use_body_delta_column_rot6d_and_zero_gripper() -> None:
    poses = np.array(
        [
            [0, 0, 0, *_quat_z(90)],
            [1, 0, 0, *_quat_z(180)],
            [1, -2, 0, *_quat_z(180)],
        ],
        dtype=np.float64,
    )

    actions = _framewise_actions(poses)

    np.testing.assert_allclose(actions[0, :3], [0, -1, 0], atol=1e-6)
    np.testing.assert_allclose(actions[0, 3:9], [0, 1, 0, -1, 0, 0], atol=1e-6)
    np.testing.assert_allclose(actions[1, :3], [0, 2, 0], atol=1e-6)
    np.testing.assert_array_equal(actions[:, 9], [0, 0])


def test_framewise_actions_chain_back_to_absolute_trajectory() -> None:
    poses = np.array(
        [
            [0.2, -0.1, 0.4, *_quat_z(10)],
            [0.3, 0.1, 0.4, *_quat_z(25)],
            [0.6, 0.2, 0.5, *_quat_z(-20)],
        ],
        dtype=np.float64,
    )
    actions = _framewise_actions(poses)
    position = poses[0, :3].copy()
    rotation = np.array(
        [[math.cos(math.radians(10)), -math.sin(math.radians(10)), 0],
         [math.sin(math.radians(10)), math.cos(math.radians(10)), 0], [0, 0, 1]],
        dtype=np.float64,
    )
    for action, expected_pose in zip(actions, poses[1:], strict=True):
        position = position + rotation @ action[:3]
        c0, c1 = action[3:6], action[6:9]
        relative_rotation = np.stack([c0, c1, np.cross(c0, c1)], axis=1)
        rotation = rotation @ relative_rotation
        np.testing.assert_allclose(position, expected_pose[:3], atol=1e-6)
    expected_final = np.array(
        [[math.cos(math.radians(-20)), -math.sin(math.radians(-20)), 0],
         [math.sin(math.radians(-20)), math.cos(math.radians(-20)), 0], [0, 0, 1]],
    )
    np.testing.assert_allclose(rotation, expected_final, atol=1e-6)


def test_split_contract_excludes_history_from_training() -> None:
    assert _split_episode_ids("train", 59) == TRAIN_EPISODES
    assert _split_episode_ids("dev", 59) == tuple(range(50, 59))
    assert _split_episode_ids("history", 59) == HISTORY_EPISODES
    assert set(TRAIN_EPISODES).isdisjoint(HISTORY_EPISODES)


def test_window_never_crosses_episode_and_preserves_source_metadata(tmp_path) -> None:
    store = _make_store(tmp_path / "tiny.zarr")
    dataset = UMIFTZarrIterableDataset(store, split="train", stage="smoke", transform=None)

    sample = next(iter(dataset))

    assert sample["episode_id"] == 0
    assert sample["source_id"] == "session-0#seg0"
    assert sample["source_indices"].tolist() == list(range(0, 33, 2))
    assert sample["video"].shape == (3, 17, 256, 256)
    assert sample["physical_action"].shape == (16, 10)
    assert sample["timestamps"].shape == (17,)
    assert sample["source_indices"][-1].item() < 40


def test_zero_physical_gripper_normalizes_to_minus_one(tmp_path) -> None:
    store = _make_store(tmp_path / "tiny.zarr")
    sample = next(iter(UMIFTZarrIterableDataset(store, split="train", stage="smoke", transform=None)))

    assert torch.equal(sample["physical_action"][:, 9], torch.zeros(16))
    assert torch.equal(sample["model_action"][:, 9], -torch.ones(16))


def test_bad_or_misaligned_timestamps_are_rejected(tmp_path) -> None:
    store = _make_store(tmp_path / "bad.zarr", bad_timestamps=True)
    dataset = UMIFTZarrIterableDataset(store, split="train", stage="smoke", transform=None)

    with pytest.raises(ValueError, match="timestamps"):
        next(iter(dataset))


def test_timestamp_alignment_allows_20ms_and_rejects_more_or_nonfinite_pose(tmp_path) -> None:
    store = _make_store(tmp_path / "tolerance.zarr")
    root = zarr.open_group(store, mode="a")
    robot_ts = root["data/episode_0/robot_time_stamps_0"][:]
    robot_ts += 0.020
    root["data/episode_0/robot_time_stamps_0"][:] = robot_ts
    next(iter(UMIFTZarrIterableDataset(store, split="train", stage="smoke", transform=None)))

    robot_ts += 0.001
    root["data/episode_0/robot_time_stamps_0"][:] = robot_ts
    with pytest.raises(ValueError, match="20 ms"):
        next(iter(UMIFTZarrIterableDataset(store, split="train", stage="smoke", transform=None)))

    root["data/episode_0/robot_time_stamps_0"][:] = root["data/episode_0/rgb_time_stamps_0"][:]
    poses = root["data/episode_0/ts_pose_fb_0"][:]
    poses[2, 0] = np.nan
    root["data/episode_0/ts_pose_fb_0"][:] = poses
    with pytest.raises(ValueError, match="finite"):
        next(iter(UMIFTZarrIterableDataset(store, split="train", stage="smoke", transform=None)))


def test_window_rejects_timestamp_jump_from_ideal_15hz_grid(tmp_path) -> None:
    store = _make_store(tmp_path / "grid.zarr")
    root = zarr.open_group(store, mode="a")
    for key in ("rgb_time_stamps_0", "robot_time_stamps_0"):
        timestamps = root[f"data/episode_0/{key}"][:]
        timestamps[16:] += 0.021
        root[f"data/episode_0/{key}"][:] = timestamps

    with pytest.raises(ValueError, match="15 Hz grid"):
        next(iter(UMIFTZarrIterableDataset(store, split="train", stage="smoke", transform=None)))


def test_training_stream_resume_reproduces_next_samples(tmp_path) -> None:
    store = _make_store(tmp_path / "resume.zarr", lengths=(80, 80))
    first = UMIFTZarrIterableDataset(store, split="train", stage="e1", seed=123, transform=None)
    iterator = iter(first)
    consumed = list(islice(iterator, 5))
    state = first.state_dict()
    expected = [(x["episode_id"], x["window_start"]) for x in islice(iterator, 4)]

    resumed = UMIFTZarrIterableDataset(store, split="train", stage="e1", seed=123, transform=None)
    resumed.load_state_dict(state)
    actual = [(x["episode_id"], x["window_start"]) for x in islice(iter(resumed), 4)]

    assert len(consumed) == 5
    assert actual == expected


@pytest.mark.parametrize("stage, expected_starts", [("smoke", {0}), ("overfit", {0, 64, 128, 192})])
def test_e0_stream_repeats_forever_without_empty_rank(tmp_path, stage, expected_starts) -> None:
    store = _make_store(tmp_path / f"{stage}.zarr", lengths=(240, 40))
    for rank in range(4):
        dataset = UMIFTZarrIterableDataset(store, split="train", stage=stage, transform=None)
        dataset.shard_world_size = 4
        dataset.shard_rank = rank
        starts = [sample["window_start"] for sample in islice(iter(dataset), 6)]
        assert len(starts) == 6
        assert set(starts) <= expected_starts


def test_set_start_iteration_restores_rank_local_microbatch_position(tmp_path) -> None:
    store = _make_store(tmp_path / "microstep.zarr", lengths=(80, 80))
    for rank in range(4):
        uninterrupted = UMIFTZarrIterableDataset(store, split="train", stage="e1", seed=123, transform=None)
        uninterrupted.shard_world_size = 4
        uninterrupted.shard_rank = rank
        full = [(x["episode_id"], x["window_start"]) for x in islice(iter(uninterrupted), 10)]

        resumed = UMIFTZarrIterableDataset(store, split="train", stage="e1", seed=123, transform=None)
        resumed.shard_world_size = 4
        resumed.shard_rank = rank
        resumed.set_start_iteration(5)
        suffix = [(x["episode_id"], x["window_start"]) for x in islice(iter(resumed), 5)]

        assert suffix == full[5:]


def test_dataloader_iterator_does_not_advance_global_torch_cpu_rng(tmp_path) -> None:
    store = _make_store(tmp_path / "loader_rng.zarr")
    dataset = UMIFTZarrIterableDataset(store, split="train", stage="smoke", transform=None)
    torch.manual_seed(123)
    expected_state = torch.get_rng_state().clone()

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=None,
        num_workers=0,
        generator=get_umift_dataloader_generator(seed=42),
    )
    iterator = iter(loader)
    next(iterator)

    assert torch.equal(torch.get_rng_state(), expected_state)


def test_factory_rejects_non_forward_dynamics_mode(tmp_path) -> None:
    store = _make_store(tmp_path / "mode.zarr")

    with pytest.raises(ValueError, match="forward_dynamics"):
        get_umift_zarr_sft_dataset(store, mode="wam")


def test_explicit_window_and_normalizer_round_trip_support_evaluation(tmp_path) -> None:
    store = _make_store(tmp_path / "eval.zarr", lengths=(80, 80))
    dataset = UMIFTZarrIterableDataset(store, split="train", stage="e1", transform=None)

    sample = dataset.get_window(episode_id=1, start=32)
    reconstructed = denormalize_umift_action(normalize_umift_action(sample["physical_action"]))

    assert sample["episode_id"] == 1
    assert sample["window_start"] == 32
    torch.testing.assert_close(reconstructed, sample["physical_action"], atol=1e-6, rtol=1e-6)


def test_real_action_processor_normalizes_once_and_preserves_physical_action_raw() -> None:
    processing = _load_real_action_processing_module()
    normalizer = _MODULE._make_umift_action_normalizer()
    processor = processing.ActionProcessor(max_action_dim=64)
    physical = torch.zeros(16, 10)
    physical[:, 3] = 1.0
    physical[:, 7] = 1.0

    result = processor.preprocess_action(
        {"action": physical.clone()},
        physical,
        action_normalizer=normalizer,
    )

    torch.testing.assert_close(result["action_raw"], physical)
    torch.testing.assert_close(result["action"][:, :10], normalize_umift_action(physical))
    assert result["action"].shape == (16, 64)
    assert result["raw_action_dim"].item() == 10
    restored = processor.postprocess_action(result["action"], result["action_processing_record"])
    torch.testing.assert_close(restored, physical, atol=1e-6, rtol=1e-6)


def test_factory_transformed_sample_reaches_real_action_processor(tmp_path) -> None:
    processing = _load_real_action_processing_module()

    class MinimalPipeline:
        def __init__(self, *, max_action_dim, **kwargs):
            self.processor = processing.ActionProcessor(max_action_dim=max_action_dim)

        def __call__(self, sample, resolution, *, action_normalizer=None, **kwargs):
            assert resolution == "256"
            return self.processor.preprocess_action(
                sample,
                sample["action"],
                action_normalizer=action_normalizer,
                action_valid_mask=kwargs.get("action_valid_mask"),
            )

    transforms = types.ModuleType("cosmos_framework.data.generator.action.utils.transforms")
    transforms.ActionTransformPipeline = MinimalPipeline
    sys.modules[transforms.__name__] = transforms
    store = _make_store(tmp_path / "factory.zarr")

    sample = get_umift_zarr_sft_dataset(store, stage="smoke", max_action_dim=64).__iter__().__next__()

    assert sample["action_raw"].shape == (16, 10)
    assert sample["action"].shape == (16, 64)
    assert sample["action_valid_mask"].tolist() == [True] * 10 + [False] * 54
    torch.testing.assert_close(sample["action"][:, :10], sample["model_action"])
    restored = processing.ActionProcessor.postprocess_action(
        sample["action"], sample["action_processing_record"]
    )
    torch.testing.assert_close(restored, sample["physical_action"], atol=1e-6, rtol=1e-6)
