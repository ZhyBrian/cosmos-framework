from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from examples.umift import score_rgbd_rollout as scorer


def _fake_evaluate_video_pair(truth, prediction, *, include_lpips, lpips_metric):
    assert truth.dtype == prediction.dtype == np.float32
    assert truth.shape == prediction.shape == (17, 3, 3, 3)
    assert include_lpips and lpips_metric == "fake-lpips"
    rows = []
    for index in range(1, 17):
        error = float(np.mean(np.abs(prediction[index] - truth[index])))
        rows.append(
            {
                "mse": error * error,
                "psnr": 10.0 - error,
                "ssim": 1.0 - error,
                "lpips": error,
            }
        )
    return {"per_frame": rows}


class ScoreRGBDRolloutTest(unittest.TestCase):
    def test_anchor_accepts_expected_float32_normalization_round_trip(self) -> None:
        truth_rgb = np.full((4, 3, 3, 3), np.float32(32.0 / 255.0), np.float32)
        truth_depth = np.full((4, 3, 3), 0.1, np.float32)
        rgb_round_trip = (
            (truth_rgb * np.float32(2.0) - np.float32(1.0)) + np.float32(1.0)
        ) / np.float32(2.0)
        scalar = truth_depth * np.float32(4.0) - np.float32(1.0)
        depth_round_trip = (
            np.repeat(scalar[..., None], 3, axis=-1).mean(axis=-1) + np.float32(1.0)
        ) / np.float32(4.0)
        self.assertFalse(np.array_equal(rgb_round_trip, truth_rgb))
        self.assertFalse(np.array_equal(depth_round_trip, truth_depth))
        predictions = {
            method: (rgb_round_trip.copy(), depth_round_trip.copy())
            for method in scorer.MODEL_METHODS
        }

        result = scorer.score_episode_arrays(
            truth_rgb,
            truth_depth,
            predictions,
            first_block_steps=3,
            evaluate_video_pair_fn=_fake_evaluate_video_pair,
            lpips_metric="fake-lpips",
        )

        self.assertEqual(tuple(result), scorer.METHODS)

    def test_episode_scoring_excludes_anchor_and_keeps_raw_depth_diagnostics(
        self,
    ) -> None:
        frame_count = 10
        truth_rgb = np.full((frame_count, 3, 3, 3), 0.12345, np.float32)
        truth_depth = np.full((frame_count, 3, 3), 0.2, np.float32)
        truth_depth[:, 0, 0] = 0.0
        truth_depth[:, 0, 1] = 0.5
        predictions = {}
        for offset, method in enumerate(scorer.MODEL_METHODS, start=1):
            rgb = truth_rgb.copy()
            depth = truth_depth.copy()
            rgb[1:] += np.float32(offset / 100.0)
            depth[1:, 1:, :] += np.float32(offset / 100.0)
            depth[1:, 0, 0] = -0.1 * offset
            depth[1:, 0, 1] = 0.4
            predictions[method] = (rgb, depth)
        truth_rgb_before = truth_rgb.copy()
        truth_depth_before = truth_depth.copy()
        predictions_before = {
            method: (rgb.copy(), depth.copy())
            for method, (rgb, depth) in predictions.items()
        }

        result = scorer.score_episode_arrays(
            truth_rgb,
            truth_depth,
            predictions,
            first_block_steps=4,
            evaluate_video_pair_fn=_fake_evaluate_video_pair,
            lpips_metric="fake-lpips",
        )

        self.assertEqual(tuple(result), scorer.METHODS)
        self.assertEqual(result["P"]["frame_indices"], list(range(1, frame_count)))
        self.assertEqual(
            result["E3-A"]["segments"]["first_block"]["frame_indices"], [1, 2, 3, 4]
        )
        self.assertEqual(
            result["E3-A"]["segments"]["early_third"]["frame_indices"], [1, 2, 3]
        )
        self.assertEqual(
            result["E3-A"]["segments"]["middle_third"]["frame_indices"], [4, 5, 6]
        )
        self.assertEqual(
            result["E3-A"]["segments"]["late_third"]["frame_indices"], [7, 8, 9]
        )
        first = result["E3-A"]["frames"][0]
        self.assertEqual(first["frame_index"], 1)
        self.assertGreater(first["depth_out_of_range_fraction"], 0.0)
        self.assertEqual(first["depth_zero_fraction"], 1.0 / 9.0)
        self.assertEqual(first["depth_cap_fraction"], 1.0 / 9.0)
        self.assertAlmostEqual(first["depth_cap_underprediction_m"], 0.1)
        self.assertAlmostEqual(first["depth_zero_region_mean_m"], -0.2)
        self.assertAlmostEqual(first["rgb_lpips"], 0.02, places=6)
        self.assertAlmostEqual(first["rgb_temporal_l1"], 3 * 3 * 3 * 0.02, places=5)
        self.assertAlmostEqual(result["P"]["frames"][0]["rgb_lpips"], 0.0)
        np.testing.assert_array_equal(truth_rgb, truth_rgb_before)
        np.testing.assert_array_equal(truth_depth, truth_depth_before)
        for method in scorer.MODEL_METHODS:
            np.testing.assert_array_equal(
                predictions[method][0], predictions_before[method][0]
            )
            np.testing.assert_array_equal(
                predictions[method][1], predictions_before[method][1]
            )

    def test_temporal_l1_is_continuous_across_legacy_evaluator_blocks(self) -> None:
        truth_rgb = np.zeros((18, 3, 3, 3), np.float32)
        truth_depth = np.full((18, 3, 3), 0.2, np.float32)
        predictions = {}
        for method in scorer.MODEL_METHODS:
            rgb = truth_rgb.copy()
            rgb[1:17] = 0.1
            rgb[17] = 0.2
            predictions[method] = (rgb, truth_depth.copy())

        result = scorer.score_episode_arrays(
            truth_rgb,
            truth_depth,
            predictions,
            first_block_steps=16,
            evaluate_video_pair_fn=_fake_evaluate_video_pair,
            lpips_metric="fake-lpips",
        )

        frame_17 = result["B0"]["frames"][16]
        self.assertEqual(frame_17["frame_index"], 17)
        self.assertAlmostEqual(frame_17["rgb_temporal_l1"], 3 * 3 * 3 * 0.1, places=5)

    def test_session_equal_aggregation_weights_sessions_after_frame_means(self) -> None:
        rows = [
            {"raw_session": "a", "metric": 0.0, "optional": None},
            {"raw_session": "a", "metric": 2.0, "optional": 4.0},
            {"raw_session": "b", "metric": 9.0, "optional": 8.0},
        ]

        result = scorer.aggregate_session_equal_frame_rows(rows, ("metric", "optional"))

        self.assertEqual(result["sessions"]["a"]["metric"], 1.0)
        self.assertEqual(result["sessions"]["b"]["metric"], 9.0)
        self.assertEqual(result["overall"]["metric"], 5.0)
        self.assertEqual(result["overall"]["optional"], 6.0)
        self.assertEqual(result["frame_counts"], {"a": 2, "b": 1})
        self.assertEqual(result["value_counts"]["optional"], {"a": 1, "b": 1})

    def test_prediction_loader_checks_files_and_complete_chunk_inventory(self) -> None:
        with tempfile.TemporaryDirectory(prefix="score-rgbd-rollout-") as directory:
            root = Path(directory)
            output_dir = root / "rgbd_inference" / "B0"
            output_dir.mkdir(parents=True)
            stem = output_dir / "episode_13_start_50"
            rgb_path = stem.with_name(stem.name + "_rgb.npy")
            depth_path = stem.with_name(stem.name + "_depth_m.npy")
            rgb = np.zeros((4, 3, 3, 3), np.float32)
            depth = np.full((4, 3, 3), 0.2, np.float32)
            np.save(rgb_path, rgb)
            np.save(depth_path, depth)
            chunk = {"index": 0, "output_start": 0, "steps": 3, "noise_seed": 7}
            metadata = {
                "method": "B0",
                "episode_id": 13,
                "raw_session": "session-13",
                "start_percent": 50,
                "frame_count": 4,
                "complete_requested_suffix": True,
                "manifest_sha256": "manifest-sha",
                "checkpoint_id": "/checkpoint/base",
                "future_gt_refresh_count": 0,
                "rgb_path": str(rgb_path),
                "rgb_sha256": scorer.file_sha(rgb_path),
                "raw_depth_m_path": str(depth_path),
                "raw_depth_m_sha256": scorer.file_sha(depth_path),
                "chunks": [
                    {
                        **chunk,
                        "retained_rgb_sha256": hashlib.sha256(
                            np.ascontiguousarray(rgb[1:]).tobytes()
                        ).hexdigest(),
                        "retained_raw_depth_m_sha256": hashlib.sha256(
                            np.ascontiguousarray(depth[1:]).tobytes()
                        ).hexdigest(),
                    }
                ],
            }
            metadata_path = stem.with_suffix(".json")
            metadata_path.write_text(json.dumps(metadata))
            manifest = {
                "base_checkpoint": "/checkpoint/base",
                "selected_checkpoint": "/checkpoint/e3",
            }
            episode = {
                "episode_id": 13,
                "raw_session": "session-13",
                "start_percent": 50,
                "frame_count": 4,
                "chunks": [chunk],
            }

            loaded_rgb, loaded_depth, inventory = scorer._load_prediction(
                root,
                manifest,
                "manifest-sha",
                episode,
                "B0",
                rgb.shape,
                depth.shape,
            )

            self.assertFalse(loaded_rgb.flags.writeable)
            self.assertFalse(loaded_depth.flags.writeable)
            self.assertEqual(inventory["metadata_path"], str(metadata_path.resolve()))
            metadata["chunks"] = []
            metadata_path.write_text(json.dumps(metadata))
            with self.assertRaisesRegex(ValueError, "missing or extra chunks"):
                scorer._load_prediction(
                    root,
                    manifest,
                    "manifest-sha",
                    episode,
                    "B0",
                    rgb.shape,
                    depth.shape,
                )

    def test_verified_npy_load_rejects_hash_change_nonfinite_and_shape(self) -> None:
        with tempfile.TemporaryDirectory(prefix="score-rgbd-") as directory:
            path = Path(directory) / "array.npy"
            value = np.zeros((2, 3, 3), np.float32)
            np.save(path, value)
            expected = hashlib.sha256(path.read_bytes()).hexdigest()
            loaded = scorer.verified_load_npy(
                path,
                expected_sha256=expected,
                expected_shape=value.shape,
                label="depth",
            )
            self.assertEqual(loaded.dtype, np.float32)
            self.assertFalse(loaded.flags.writeable)

            value[0, 0, 0] = np.nan
            np.save(path, value)
            changed = hashlib.sha256(path.read_bytes()).hexdigest()
            with self.assertRaisesRegex(ValueError, "finite"):
                scorer.verified_load_npy(
                    path,
                    expected_sha256=changed,
                    expected_shape=value.shape,
                    label="depth",
                )
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                scorer.verified_load_npy(
                    path,
                    expected_sha256=expected,
                    expected_shape=value.shape,
                    label="depth",
                )
            with self.assertRaisesRegex(ValueError, "shape"):
                scorer.verified_load_npy(
                    path,
                    expected_sha256=changed,
                    expected_shape=(3, 3, 3),
                    label="depth",
                )


if __name__ == "__main__":
    unittest.main()
