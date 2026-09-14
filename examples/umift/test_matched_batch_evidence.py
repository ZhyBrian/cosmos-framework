from __future__ import annotations

import importlib.util
import json
import random
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


callback_module = types.ModuleType("cosmos_framework.utils.callback")
callback_module.Callback = object
sys.modules.setdefault("cosmos_framework.utils.callback", callback_module)

from examples.umift.matched_batch_evidence import MatchedBatchEvidence  # noqa: E402


class MatchedBatchEvidenceTest(unittest.TestCase):
    def test_records_first_eight_microbatches_without_advancing_rng(self):
        with tempfile.TemporaryDirectory() as directory:
            callback = MatchedBatchEvidence(num_microbatches=8)
            callback.config = SimpleNamespace(job=SimpleNamespace(path_local=directory))
            with mock.patch("torch.distributed.is_initialized", return_value=True), mock.patch(
                "torch.distributed.get_rank", return_value=2
            ):
                callback.on_train_start(model=None, iteration=0)
                for microbatch in range(9):
                    random.seed(100 + microbatch)
                    np.random.seed(200 + microbatch)
                    torch.manual_seed(300 + microbatch)
                    python_state = random.getstate()
                    numpy_state = np.random.get_state()
                    torch_state = torch.get_rng_state().clone()
                    latent = torch.tensor([float(microbatch)], requires_grad=True)
                    callback.on_training_step_batch_end(
                        model=None,
                        data_batch={
                            "episode_id": torch.tensor([microbatch]),
                            "window_start": torch.tensor([microbatch * 2]),
                            "source_id": [f"episode_{microbatch}"],
                            "source_indices": torch.tensor([[microbatch, microbatch + 2]]),
                            "action": torch.tensor([[microbatch, 0.0]]),
                        },
                        output_batch={
                            "x0": [latent],
                            "xt": [latent + 1],
                            "sigma": torch.tensor([0.25]),
                            "condition_mask_vision": [torch.tensor([1, 0])],
                        },
                        loss=latent.sum(),
                        iteration=microbatch // 4,
                    )
                    self.assertEqual(random.getstate(), python_state)
                    current_numpy = np.random.get_state()
                    self.assertEqual(current_numpy[0], numpy_state[0])
                    np.testing.assert_array_equal(current_numpy[1], numpy_state[1])
                    self.assertEqual(current_numpy[2:], numpy_state[2:])
                    self.assertTrue(torch.equal(torch.get_rng_state(), torch_state))

            path = Path(directory) / "matched_microbatches_rank_2.jsonl"
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual(len(rows), 8)
            self.assertEqual([row["microbatch_index"] for row in rows], list(range(8)))
            self.assertEqual([row["iteration"] for row in rows], [0, 0, 0, 0, 1, 1, 1, 1])
            self.assertTrue(all(row["rank"] == 2 for row in rows))
            self.assertNotIn("device", rows[0])
            self.assertEqual(rows[3]["episode_id"], 3)
            self.assertEqual(rows[3]["window_start"], 6)
            self.assertEqual(len(rows[0]["x0_sha256"]), 64)
            self.assertNotEqual(rows[0]["x0_sha256"], rows[1]["x0_sha256"])

    def test_both_continuation_configs_register_the_same_callback(self):
        registered = {}

        class FakeConfigStore:
            @classmethod
            def instance(cls):
                return cls()

            def store(self, *, name, node, **kwargs):
                registered[name] = node

        hydra_module = types.ModuleType("hydra")
        hydra_core_module = types.ModuleType("hydra.core")
        config_store_module = types.ModuleType("hydra.core.config_store")
        config_store_module.ConfigStore = FakeConfigStore
        base_module_name = (
            "cosmos_framework.configs.base.experiment.action.posttrain_config."
            "action_fd_umift_edge_rgbd"
        )
        base_module = types.ModuleType(base_module_name)
        base_module.action_fd_umift_edge_rgbd_h5 = SimpleNamespace(
            job=SimpleNamespace(name="base"),
            scheduler=SimpleNamespace(cycle_lengths=[], f_min=[], warm_up_steps=[]),
            trainer=SimpleNamespace(max_iter=0, callbacks={}),
            checkpoint=SimpleNamespace(load_training_state=True, save_iter=0),
            model=SimpleNamespace(_target_="base", depth_aux_weight=None),
        )
        replacements = {
            "hydra": hydra_module,
            "hydra.core": hydra_core_module,
            "hydra.core.config_store": config_store_module,
            base_module_name: base_module,
        }
        config_path = (
            Path(__file__).parents[2]
            / "cosmos_framework/configs/base/experiment/action/posttrain_config/action_fd_umift_edge_rgbd_d1.py"
        )
        spec = importlib.util.spec_from_file_location("matched_evidence_config_test", config_path)
        module = importlib.util.module_from_spec(spec)
        with mock.patch.dict(sys.modules, replacements):
            spec.loader.exec_module(module)

        target = "examples.umift.matched_batch_evidence.MatchedBatchEvidence"
        for name in ("action_fd_umift_edge_rgbd_b_continue", "action_fd_umift_edge_rgbd_d1"):
            callback = registered[name].trainer.callbacks["matched_batch_evidence"]
            self.assertEqual(callback["_target_"], target)
            self.assertEqual(callback["num_microbatches"], 8)


if __name__ == "__main__":
    unittest.main()
