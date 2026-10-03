"""Pure numpy contracts for local window comparison and raw depth scoring."""

from __future__ import annotations

import unittest

import numpy as np

from examples.umift.local_backup_compare import compare_reference, depth_pixels, score_runs, validate_runs


class LocalBackupCompareTest(unittest.TestCase):
    def runs(self, *, depth=False):
        truth = np.full((17, 256, 256, 3), .2, np.float32)
        truth[0] = 0
        prediction = np.full_like(truth, .3)
        prediction[0] = truth[0]
        common = {
            "truth": truth, "prediction": prediction,
            "history_source_indices": np.zeros(5, np.int64), "source_indices": 2 * np.arange(17),
            "timestamps": np.arange(17) / 10, "noise_seed": np.array(495100992), "num_steps": np.array(30),
            "physical_action": np.zeros((16, 10), np.float32),
        }
        if depth:
            depth_truth = np.full((17, 256, 256), .2, np.float32)
            depth_truth[0] = .1
            depth_prediction = np.ones_like(depth_truth)
            depth_prediction[0] = depth_truth[0]
            common.update(depth_truth=depth_truth, depth_prediction=depth_prediction)
        runs = {method: dict(common) for method in ("A", "Z", "S", "B0")}
        runs["Z"]["physical_action"] = np.tile([0, 0, 0, 1, 0, 0, 0, 1, 0, 0], (16, 1))
        return runs

    def test_branches_share_inputs_but_may_have_different_actions(self):
        runs = self.runs()
        self.assertFalse(validate_runs(runs))
        runs["S"]["physical_action"] = np.ones((16, 10))
        self.assertFalse(validate_runs(runs))
        runs["S"]["noise_seed"] = np.array(42)
        with self.assertRaisesRegex(ValueError, "noise_seed"):
            validate_runs(runs)

    def test_truth_indices_time_and_history_must_match(self):
        for key in ("truth", "history_source_indices", "source_indices", "timestamps"):
            runs = self.runs()
            runs["B0"][key] = runs["B0"][key].copy()
            runs["B0"][key].reshape(-1)[-1] += 1
            with self.assertRaisesRegex(ValueError, key):
                validate_runs(runs)

    def test_rgbd_presence_and_raw_predictions(self):
        runs = self.runs(depth=True)
        self.assertTrue(validate_runs(runs))
        del runs["Z"]["depth_prediction"]
        with self.assertRaisesRegex(ValueError, "depth modality"):
            validate_runs(runs)

    def test_reference_truth_is_exact_and_reports_anchor_separately(self):
        correct = self.runs()["A"]
        reference = {"truth": correct["truth"], "prediction": correct["prediction"].copy()}
        reference["prediction"][0] = .5
        reference["prediction"][1:] += .01
        result = compare_reference(correct, reference)
        self.assertAlmostEqual(result["rgb"]["all_17_max_abs_difference"], .5)
        self.assertAlmostEqual(result["rgb"]["future_16_mean_abs_difference"], .01, places=6)
        reference["truth"] = reference["truth"].copy()
        reference["truth"][1, 0, 0, 0] += .01
        with self.assertRaisesRegex(ValueError, "truth differs"):
            compare_reference(correct, reference)

    def test_lpips_transform_future_only_and_unclipped_depth_joint(self):
        runs = self.runs(depth=True)
        calls = []

        def metric(truth, prediction):
            calls.append(truth.copy())
            self.assertEqual(truth.shape, (16, 3, 256, 256))
            self.assertAlmostEqual(float(truth.mean()), -.6, places=5)
            return np.abs(truth - prediction).mean(axis=(1, 2, 3))

        result = score_runs(runs, metric)
        self.assertEqual(len(calls), 5)
        self.assertAlmostEqual(result["A"]["depth_mae_m"], .8, places=6)
        self.assertEqual(result["A"]["depth"]["mean"]["depth_out_of_range_fraction"], 1)
        self.assertAlmostEqual(result["P"]["joint_50_50_relative_to_persistence"], 1, places=6)
        self.assertAlmostEqual(result["A"]["joint_50_50_relative_to_persistence"], 4.25, places=5)

    def test_display_uses_gt_zero_only_and_does_not_mutate_raw_values(self):
        depth = np.array([[0, -.1, .25, .8]], np.float32)
        original = depth.copy()
        gt = depth_pixels(depth, ground_truth=True)
        pred = depth_pixels(depth, ground_truth=False)
        np.testing.assert_array_equal(gt[0, 0], [0, 0, 0])
        np.testing.assert_array_equal(pred[0, 0], [59, 76, 192])
        np.testing.assert_array_equal(pred[0, -1], [180, 4, 38])
        np.testing.assert_array_equal(depth, original)


if __name__ == "__main__":
    unittest.main()
