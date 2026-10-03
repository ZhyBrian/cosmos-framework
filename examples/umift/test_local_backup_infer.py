"""Small CPU tests for the separate local backup inference contracts."""

from __future__ import annotations

import argparse
import copy
import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from examples.umift.local_backup_infer import (
    DEFAULT_NOISE_SEED,
    DEFAULT_NUM_STEPS,
    anchor_prediction,
    check_sampling_reference,
    choose_sample,
    find_ffmpeg,
    local_asset_paths,
    single_gpu_runtime_options,
    validate_causal_sample,
    validate_conditioned_video,
    validate_model_contract,
)


class LocalBackupContractTest(unittest.TestCase):
    def test_ffmpeg_prefers_existing_path_binary(self):
        with patch("examples.umift.local_backup_infer.shutil.which", return_value="/usr/bin/ffmpeg"):
            self.assertEqual(find_ffmpeg(), "/usr/bin/ffmpeg")

    def test_ffmpeg_uses_existing_imageio_bundle_when_path_has_none(self):
        bundle = SimpleNamespace(get_ffmpeg_exe=Mock(return_value="/env/imageio_ffmpeg/binaries/ffmpeg"))
        with patch("examples.umift.local_backup_infer.shutil.which", return_value=None):
            with patch.dict("sys.modules", {"imageio_ffmpeg": bundle}):
                self.assertEqual(find_ffmpeg(), "/env/imageio_ffmpeg/binaries/ffmpeg")
        bundle.get_ffmpeg_exe.assert_called_once_with()

    def test_ffmpeg_missing_environment_is_explicit(self):
        with patch("examples.umift.local_backup_infer.shutil.which", return_value=None):
            with patch.dict("sys.modules", {"imageio_ffmpeg": None}):
                with self.assertRaisesRegex(RuntimeError, "ffmpeg is unavailable"):
                    find_ffmpeg()

    def model_dict(self):
        return {"config": {
            "action_gen": True, "vision_gen": True, "sound_gen": False, "resolution": "256",
            "tokenizer": {"encode_exact_durations": [21]},
            "parallelism": {"data_parallel_shard_degree": 4, "fsdp_master_dtype": "float32"},
            "compile": {"enabled": True}, "quantization": {"method": None},
        }}

    def sample(self, modality="rgb"):
        return {
            "video": np.zeros((3, 21, 256, 512 if modality == "rgbd" else 256), np.float32),
            "history_source_indices": np.array([0, 0, 0, 0, 0]),
            "source_indices": 2 * np.arange(17),
            "sequence_plan": SimpleNamespace(condition_frame_indexes_vision=[0, 1], action_start_frame_offset=5),
        }

    def test_modalities_require_the_original_dataset_target(self):
        for modality, target in (("rgb", "get_umift_history_sft_dataset"),
                                 ("rgbd", "get_umift_rgbd_sft_dataset")):
            validate_model_contract(self.model_dict(), target, modality)
            with self.assertRaisesRegex(ValueError, "requires"):
                validate_model_contract(self.model_dict(), target, "rgbd" if modality == "rgb" else "rgb")

    def test_runtime_overrides_do_not_modify_original_model(self):
        cfg = self.model_dict()["config"]
        original = copy.deepcopy(cfg)
        parallelism, compile_options, quantization = single_gpu_runtime_options(cfg)
        self.assertEqual(cfg, original)
        self.assertTrue(parallelism["enable_inference_mode"])
        self.assertTrue(all(parallelism[key] == 1 for key in (
            "data_parallel_shard_degree", "data_parallel_replicate_degree",
            "context_parallel_shard_degree", "cfg_parallel_shard_degree", "vae_load_balance_group_size",
        )))
        self.assertFalse(compile_options["enabled"])
        self.assertEqual(quantization, cfg["quantization"])

    def test_local_path_mapping_is_explicit_and_checked(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            vae = root / "vae.pth"
            vae.touch()
            paths = {"WAN_VAE_PATH": str(vae), "EDGE_HF_SNAPSHOT_PATH": str(root),
                     "BASE_CHECKPOINT_PATH": str(root)}
            self.assertEqual(local_asset_paths(paths), paths)
            paths["WAN_VAE_PATH"] = "remote/vae.pth"
            with self.assertRaisesRegex(ValueError, "absolute local"):
                local_asset_paths(paths)
            paths["WAN_VAE_PATH"] = str(root / "missing.pth")
            with self.assertRaises(FileNotFoundError):
                local_asset_paths(paths)

    def test_seed_and_steps_are_preserved_against_reference(self):
        self.assertEqual((DEFAULT_NOISE_SEED, DEFAULT_NUM_STEPS), (495100992, 30))
        check_sampling_reference({"noise_seed": np.array(495100992), "num_steps": 30},
                                 noise_seed=495100992, num_steps=30)
        with self.assertRaisesRegex(ValueError, "noise_seed"):
            check_sampling_reference({"noise_seed": 1}, noise_seed=495100992, num_steps=30)
        with self.assertRaisesRegex(ValueError, "num_steps"):
            check_sampling_reference({"num_steps": 10}, noise_seed=495100992, num_steps=30)

    def test_history_cannot_contain_future_indices_or_condition_latents(self):
        sample = self.sample()
        validate_causal_sample(sample, "rgb")
        sample["history_source_indices"][-1] = 2
        with self.assertRaisesRegex(ValueError, "future leakage"):
            validate_causal_sample(sample, "rgb")
        sample = self.sample()
        sample["sequence_plan"].condition_frame_indexes_vision = [0, 1, 2]
        with self.assertRaisesRegex(ValueError, "causal"):
            validate_causal_sample(sample, "rgb")
        with self.assertRaisesRegex(ValueError, "rgbd input"):
            validate_causal_sample(self.sample("rgb"), "rgbd")

    def test_future_canvas_must_be_zero_for_both_modalities(self):
        for modality in ("rgb", "rgbd"):
            video = self.sample(modality)["video"]
            if modality == "rgbd":
                video = video[None]
            validate_conditioned_video(video, modality)
            video.reshape(-1)[-1] = 1
            with self.assertRaisesRegex(ValueError, "future RGB/depth"):
                validate_conditioned_video(video, modality)

    def test_output_uses_observed_anchor_and_only_generated_future(self):
        truth = np.ones((17, 2, 2, 3), np.float32)
        decoded = np.zeros((21, 2, 2, 3), np.float32)
        decoded[:5] = 7
        result = anchor_prediction(truth, decoded)
        np.testing.assert_array_equal(result[0], truth[0])
        np.testing.assert_array_equal(result[1:], decoded[5:])

    def test_static_action_is_physical_rotation_identity_before_normalization(self):
        calls = []
        dataset = SimpleNamespace(get_window=lambda episode, start, **kw: calls.append(kw) or kw)
        args = argparse.Namespace(action_mode="Z", episode=13, start=0)
        result = choose_sample(dataset, args)
        expected = np.tile([0, 0, 0, 1, 0, 0, 0, 1, 0, 0], (16, 1))
        np.testing.assert_array_equal(result["physical_action"], expected)
        self.assertEqual(len(calls), 1)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch environment is not available")
    def test_existing_batch_helpers_clear_future_without_mutating_source(self):
        import torch

        from examples.umift.history_infer import build_history_batch
        from examples.umift.rgbd_infer import build_rgbd_batch

        for modality, builder in (("rgb", build_history_batch), ("rgbd", build_rgbd_batch)):
            width = 512 if modality == "rgbd" else 256
            sample = self.sample(modality)
            sample.update(
                history_frames=5, mode="forward_dynamics", ai_caption="",
                video=torch.ones((3, 21, 256, width), dtype=torch.float32 if modality == "rgbd" else torch.uint8),
                action=torch.zeros((16, 64)), raw_action_dim=10,
                conditioning_fps=torch.tensor(15.0), domain_id=torch.tensor(6),
                image_size=torch.tensor([256, width, 256, width]),
                is_preprocessed=modality == "rgbd", depth_truth="future truth must never enter the batch",
            )
            plan = sample["sequence_plan"]
            plan.condition_frame_indexes_action = list(range(16))
            plan.has_text = plan.has_vision = plan.has_action = True
            batch = builder(sample)
            validate_conditioned_video(batch["video"][0], modality)
            self.assertTrue(torch.all(sample["video"][:, 5:] == 1))
            self.assertNotIn("depth_truth", batch)


if __name__ == "__main__":
    unittest.main()
