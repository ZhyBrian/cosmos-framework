from __future__ import annotations

import importlib.util
import math
import sys
import tempfile
import types
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest import mock

import numpy as np
import torch
import zarr


_DATASET_PACKAGE = "cosmos_framework.data.generator.action.datasets"
_DATASET_DIR = Path(__file__).resolve().parents[2] / _DATASET_PACKAGE.replace(".", "/")
_MODULE_SNAPSHOT = {
    name: module
    for name, module in sys.modules.items()
    if name.startswith(_DATASET_PACKAGE)
}


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


try:
    from cosmos_framework.data.generator.action.datasets import (
        umift_history_dataset as _HISTORY,
    )
    from cosmos_framework.data.generator.action.datasets import (
        umift_rgbd_dataset as _RGBD,
    )
    from cosmos_framework.data.generator.action.datasets import (
        umift_zarr_dataset as _BASE,
    )
except ModuleNotFoundError as error:
    if error.name != "pyarrow":
        raise
    _BASE_NAME = f"{_DATASET_PACKAGE}.umift_zarr_dataset"
    _HISTORY_NAME = f"{_DATASET_PACKAGE}.umift_history_dataset"
    try:
        _BASE = _load_module(_BASE_NAME, _DATASET_DIR / "umift_zarr_dataset.py")
        _HISTORY = _load_module(
            _HISTORY_NAME, _DATASET_DIR / "umift_history_dataset.py"
        )
        _RGBD = _load_module(
            "umift_rgbd_dataset_under_test", _DATASET_DIR / "umift_rgbd_dataset.py"
        )
    finally:
        for _name in tuple(sys.modules):
            if _name.startswith(_DATASET_PACKAGE) and _name not in _MODULE_SNAPSHOT:
                sys.modules.pop(_name, None)
        sys.modules.update(_MODULE_SNAPSHOT)


def _quat_z(degrees: float) -> list[float]:
    half = math.radians(degrees) / 2.0
    return [math.cos(half), 0.0, 0.0, math.sin(half)]


def _write_episode(
    root: zarr.Group, episode_id: int, length: int = 50, *, materialize: bool = True
) -> None:
    group = root.require_group("data").create_group(f"episode_{episode_id}")
    group.attrs["src"] = f"session-{episode_id}#segment"
    if materialize:
        rgb = np.zeros((length, 4, 4, 3), dtype=np.uint8)
        depth = np.empty((length, 224, 224, 3), dtype=np.float16)
        for frame in range(length):
            rgb[frame] = frame
            depth[frame] = np.float16(frame / 1000.0)
        depth[0] = np.float16(0.0)
        depth[2] = np.float16(0.5)
        depth[4] = np.float16(0.12345)
        group.create_dataset("rgb_0", data=rgb, chunks=(16, 4, 4, 3))
        group.create_dataset("depth_0", data=depth, chunks=(1, 224, 224, 3))
    else:
        group.create_dataset(
            "rgb_0", shape=(length, 4, 4, 3), dtype=np.uint8, chunks=(16, 4, 4, 3)
        )
        group.create_dataset(
            "depth_0",
            shape=(length, 224, 224, 3),
            dtype=np.float16,
            chunks=(1, 224, 224, 3),
        )
    poses = np.zeros((length, 7), dtype=np.float64)
    poses[:, 0] = np.arange(length) * 0.01
    for frame in range(length):
        poses[frame, 3:] = _quat_z(frame * 2.0)
    timestamps = np.arange(length, dtype=np.float64)[:, None] / 30.0
    group.create_dataset("ts_pose_fb_0", data=poses)
    group.create_dataset("rgb_time_stamps_0", data=timestamps)
    group.create_dataset("robot_time_stamps_0", data=timestamps)


def _make_store(path: Path, episode_ids=(0,), *, materialize: bool = True) -> str:
    root = zarr.open_group(str(path), mode="w")
    for episode_id in episode_ids:
        _write_episode(root, episode_id, materialize=materialize)
    return str(path)


class _PlanTransform:
    def __init__(self) -> None:
        self.input_shapes: list[tuple[int, ...]] = []

    def __call__(self, sample, resolution, **kwargs):
        self.input_shapes.append(tuple(sample["video"].shape))
        sample["image_size"] = torch.tensor([999, 999, 999, 999], dtype=torch.float32)
        sample["video_num_frames"] = 17
        sample["num_frames"] = 17
        sample["sequence_plan"] = types.SimpleNamespace(
            condition_frame_indexes_vision=[0],
            condition_frame_indexes_action=list(range(16)),
            action_start_frame_offset=1,
        )
        return sample


class _FakeResumeAwareLoader:
    def __init__(self, batch: dict) -> None:
        self.batch = batch
        self.start_iteration = None

    def __iter__(self):
        yield self.batch

    def __len__(self):
        return 1

    def set_start_iteration(self, iteration: int) -> None:
        self.start_iteration = iteration


class RGBDDatasetTest(unittest.TestCase):
    @contextmanager
    def _temp_path(self):
        with tempfile.TemporaryDirectory(prefix="umift-rgbd-test-") as directory:
            yield Path(directory)

    def test_float_codec_preserves_non_uint8_depth_and_tile_layout(self) -> None:
        rgb = torch.linspace(-1.0, 1.0, 3 * 2 * 256 * 256, dtype=torch.float32).reshape(
            3, 2, 256, 256
        )
        depth = torch.full((2, 256, 256), 0.12345, dtype=torch.float32)

        canvas = _RGBD.pack_rgbd_canvas(rgb, depth)
        restored_rgb, restored_depth = _RGBD.unpack_rgbd_canvas(canvas)

        self.assertEqual(canvas.dtype, torch.float32)
        self.assertEqual(canvas.shape, (3, 2, 256, 512))
        self.assertTrue(torch.equal(canvas[..., :256], rgb))
        torch.testing.assert_close(
            canvas[:, :, :, 256:],
            (4.0 * depth - 1.0).unsqueeze(0).expand(3, -1, -1, -1),
        )
        self.assertTrue(torch.equal(restored_rgb, rgb))
        torch.testing.assert_close(restored_depth, depth, atol=2e-8, rtol=0)
        uint8_round_trip = round(0.12345 * 255.0) / 255.0
        self.assertNotAlmostEqual(
            restored_depth[0, 0, 0].item(), uint8_round_trip, places=5
        )

    def test_pack_rejects_malformed_or_lossy_inputs(self) -> None:
        cases = [
            (torch.zeros(3, 1, 255, 256), torch.zeros(1, 256, 256), "shape"),
            (
                torch.zeros(3, 1, 256, 256, dtype=torch.float64),
                torch.zeros(1, 256, 256),
                "float32",
            ),
            (torch.zeros(3, 1, 256, 256), torch.zeros(1, 256, 255), "shape"),
            (torch.zeros(3, 1, 256, 256), torch.full((1, 256, 256), 0.5001), "range"),
            (
                torch.full((3, 1, 256, 256), float("nan")),
                torch.zeros(1, 256, 256),
                "finite",
            ),
        ]
        for rgb, depth, message in cases:
            with (
                self.subTest(message=message),
                self.assertRaisesRegex((TypeError, ValueError), message),
            ):
                _RGBD.pack_rgbd_canvas(rgb, depth)

    def test_unpack_rejects_nonfinite_but_does_not_clamp_decoded_depth(self) -> None:
        canvas = torch.zeros((3, 1, 256, 512), dtype=torch.float32)
        canvas[..., 256:] = 2.0
        _, depth = _RGBD.unpack_rgbd_canvas(canvas)
        self.assertTrue(torch.equal(depth, torch.full_like(depth, 0.75)))

        canvas[0, 0, 0, 0] = torch.inf
        with self.assertRaisesRegex(ValueError, "finite"):
            _RGBD.unpack_rgbd_canvas(canvas)

    def test_rgbd_h5_reuses_parent_indices_actions_and_transform_before_packing(
        self,
    ) -> None:
        with self._temp_path() as tmp_path:
            store = _make_store(tmp_path / "aligned.zarr")
            rgbd_transform = _PlanTransform()
            legacy_transform = _PlanTransform()
            rgbd = _RGBD.UMIFTRGBDHistoryDataset(
                store, split="refit_train", stage="e1", transform=rgbd_transform
            ).get_window(0, 0)
            legacy = _HISTORY.UMIFTHistoryDataset(
                store,
                split="refit_train",
                stage="e1",
                history_frames=5,
                transform=legacy_transform,
            ).get_window(0, 0)

            self.assertEqual(rgbd_transform.input_shapes, [(3, 17, 256, 256)])
            self.assertEqual(rgbd["video"].shape, (3, 21, 256, 512))
            self.assertEqual(rgbd["video"].dtype, torch.float32)
            self.assertEqual(rgbd["video_num_frames"], 21)
            self.assertEqual(rgbd["num_frames"], 21)
            self.assertIs(rgbd["is_preprocessed"], True)
            self.assertEqual(rgbd["image_size"].tolist(), [256.0, 512.0, 256.0, 512.0])
            self.assertEqual(
                rgbd["video_source_indices"].tolist(),
                [0, 0, 0, 0, 0] + list(range(2, 33, 2)),
            )
            self.assertEqual(
                rgbd["history_real_mask"].tolist(), [False, False, False, False, True]
            )
            self.assertTrue(
                torch.equal(rgbd["physical_action"], legacy["physical_action"])
            )
            self.assertTrue(torch.equal(rgbd["model_action"], legacy["model_action"]))
            self.assertTrue(torch.equal(rgbd["action"], legacy["action"]))
            self.assertEqual(
                rgbd["sequence_plan"].condition_frame_indexes_vision, [0, 1]
            )
            self.assertEqual(
                rgbd["sequence_plan"].condition_frame_indexes_action, list(range(16))
            )
            self.assertEqual(rgbd["sequence_plan"].action_start_frame_offset, 5)
            self.assertEqual(rgbd["depth_m"].dtype, torch.float32)
            self.assertEqual(rgbd["depth_m"].shape, (21, 256, 256))
            torch.testing.assert_close(
                rgbd["depth_m"][:, 0, 0],
                torch.tensor(
                    [0.0] * 5 + [0.5, 0.12345] + [i / 1000 for i in range(6, 33, 2)]
                ),
                atol=2e-4,
                rtol=0,
            )
            self.assertFalse(rgbd["depth_observed_mask"][0].any())
            self.assertTrue(rgbd["depth_cap_mask"][5].all())
            self.assertFalse(rgbd["depth_metric_mask"][5].any())
            self.assertTrue(rgbd["depth_metric_mask"][6].all())

    def test_refit_split_keeps_56_episodes_and_excludes_held_out_ids(self) -> None:
        with self._temp_path() as tmp_path:
            store = _make_store(
                tmp_path / "split.zarr", tuple(range(59)), materialize=False
            )
            dataset = _RGBD.UMIFTRGBDHistoryDataset(
                store, split="refit_train", stage="e1", transform=_PlanTransform()
            )
            ids = [episode.episode_id for episode in dataset._episodes]
            self.assertEqual(len(ids), 56)
            self.assertEqual(
                ids,
                [
                    episode_id
                    for episode_id in range(59)
                    if episode_id not in (13, 43, 49)
                ],
            )

    def test_history_split_and_h5_are_fixed(self) -> None:
        with self._temp_path() as tmp_path:
            store = _make_store(
                tmp_path / "history.zarr", (13, 43, 49), materialize=False
            )
            dataset = _RGBD.UMIFTRGBDHistoryDataset(
                store, split="history", stage="e1", transform=_PlanTransform()
            )
            self.assertEqual(
                [episode.episode_id for episode in dataset._episodes], [13, 43, 49]
            )
            self.assertEqual(dataset.history_frames, 5)
            with self.assertRaisesRegex(ValueError, "history_frames=5"):
                _RGBD.UMIFTRGBDHistoryDataset(
                    store,
                    split="history",
                    stage="e1",
                    history_frames=9,
                    transform=_PlanTransform(),
                )

    def test_dataset_rejects_depth_channel_or_length_mismatch(self) -> None:
        with self._temp_path() as tmp_path:
            store = _make_store(tmp_path / "bad-depth.zarr")
            root = zarr.open_group(store, mode="a")
            del root["data/episode_0/depth_0"]
            root["data/episode_0"].create_dataset(
                "depth_0", shape=(49, 224, 224, 1), dtype=np.float16
            )
            with self.assertRaisesRegex(ValueError, "depth"):
                _RGBD.UMIFTRGBDHistoryDataset(
                    store, split="refit_train", stage="e1", transform=_PlanTransform()
                )

    def test_rgbd_packing_wrapper_validates_and_normalizes_float_batch(self) -> None:
        video = torch.full((3, 21, 256, 512), 0.12345, dtype=torch.float32)
        parent = _FakeResumeAwareLoader(
            {
                "video": [[video]],
                "is_preprocessed": [torch.tensor([True])],
                "window_start": [0],
            }
        )
        forwarded = {}

        def make_parent(**kwargs):
            forwarded.update(kwargs)
            return parent

        with mock.patch.object(_RGBD, "get_umift_packing_dataloader", make_parent):
            loader = _RGBD.get_umift_rgbd_packing_dataloader(
                max_samples_per_batch=1, sentinel=7
            )
            loader.set_start_iteration(11)
            batch = next(iter(loader))

        self.assertEqual(forwarded, {"max_samples_per_batch": 1, "sentinel": 7})
        self.assertEqual(parent.start_iteration, 11)
        self.assertIs(batch["is_preprocessed"], True)
        self.assertEqual(len(batch["video"]), 1)
        self.assertEqual(batch["video"][0].shape, (1, 3, 21, 256, 512))
        self.assertEqual(batch["video"][0].dtype, torch.float32)
        self.assertAlmostEqual(
            batch["video"][0][0, 0, 0, 0, 0].item(), 0.12345, places=6
        )

    def test_rgbd_packing_wrapper_rejects_invalid_marker_dtype_or_range(self) -> None:
        cases = [
            (
                torch.zeros((3, 21, 256, 512), dtype=torch.uint8),
                [torch.tensor([True])],
                "floating",
            ),
            (
                torch.zeros((3, 21, 256, 512), dtype=torch.float32),
                [torch.tensor([False])],
                "is_preprocessed",
            ),
            (
                torch.zeros((3, 21, 256, 512), dtype=torch.float32),
                [torch.tensor([True])],
                "range",
            ),
        ]
        cases[-1][0][0, 0, 0, 0] = 1.1
        for video, marker, message in cases:
            parent = _FakeResumeAwareLoader(
                {"video": [[video]], "is_preprocessed": marker}
            )
            with (
                self.subTest(message=message),
                mock.patch.object(
                    _RGBD, "get_umift_packing_dataloader", lambda **kwargs: parent
                ),
            ):
                loader = _RGBD.get_umift_rgbd_packing_dataloader(
                    max_samples_per_batch=1
                )
                with self.assertRaisesRegex((TypeError, ValueError), message):
                    next(iter(loader))


if __name__ == "__main__":
    unittest.main()
