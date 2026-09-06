from __future__ import annotations

import pickle
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
import torch.distributed.checkpoint as dcp

from examples.umift.compare_checkpoints import compare_checkpoints, resolve_iteration


def _write(root: Path, model_value: float, *, iteration: int = 10, rng_delta: int = 0) -> Path:
    checkpoint = root / f"iter_{iteration:09d}"
    states = {
        "model": {"weight": torch.tensor([model_value, 2.0]), "frozen": torch.tensor([7.0])},
        "optim": {
            "state": {
                "weight": {
                    "exp_avg": torch.tensor([0.1, 0.2]),
                    "exp_avg_sq": torch.tensor([0.01, 0.04]),
                }
            },
            "param_groups": [{"lr": 1e-4, "step": 10}],
        },
        "scheduler": {"last_epoch": 10, "_last_lr": [1e-4]},
        "trainer": {
            "iteration": iteration,
            "rng_state_0": {
                "torch": torch.tensor([1 + rng_delta, 2], dtype=torch.uint8),
                "torch_cuda": torch.tensor([3, 4], dtype=torch.uint8),
                "numpy_packed_len": torch.tensor(1),
                "numpy_packed_bytes": torch.tensor([5], dtype=torch.uint8),
                "random_packed_len": torch.tensor(1),
                "random_packed_bytes": torch.tensor([6], dtype=torch.uint8),
            },
        },
    }
    for component, state in states.items():
        dcp.save(state, checkpoint_id=checkpoint / component)
    (checkpoint / "dataloader").mkdir()
    with (checkpoint / "dataloader" / "rank_0.pkl").open("wb") as stream:
        pickle.dump({"next_draw_index": 10}, stream)
    root.mkdir(exist_ok=True)
    (root / "latest_checkpoint.txt").write_text(checkpoint.name + "\n")
    return checkpoint


def test_identical_dcp_components_and_rng_pass(tmp_path: Path) -> None:
    a = _write(tmp_path / "a", 1.0)
    b = _write(tmp_path / "b", 1.0)
    report = compare_checkpoints(a, b)
    assert report["pass"]
    assert report["model"]["all_parameters"]["bitwise_equal"]
    assert report["rng"]["per_rank"]["0"]["state_equal"]
    assert report["components"]["optim"]["categories"]["optimizer_first_moment"]["count"] == 1
    assert report["components"]["optim"]["categories"]["optimizer_second_moment"]["count"] == 1


def test_small_float_difference_reported_separately_from_bitwise(tmp_path: Path) -> None:
    a = _write(tmp_path / "a", 1.0)
    b = _write(tmp_path / "b", 1.0 + 5e-7)
    report = compare_checkpoints(a, b, atol=1e-6, rtol=0.0)
    model = report["model"]["all_parameters"]
    assert report["pass"]
    assert not model["bitwise_equal"]
    assert model["tolerance_pass"]
    assert model["max_abs"] > 0
    assert report["model"]["differing_parameter_names"] == ["weight"]


def test_metadata_shape_mismatch_fails_before_values(tmp_path: Path) -> None:
    a = _write(tmp_path / "a", 1.0)
    b = _write(tmp_path / "b", 1.0)
    dcp.save({"weight": torch.ones(3), "frozen": torch.tensor([7.0])}, checkpoint_id=b / "model")
    with pytest.raises(ValueError, match="metadata mismatch"):
        compare_checkpoints(a, b)


def test_rng_or_dataloader_mismatch_fails(tmp_path: Path) -> None:
    a = _write(tmp_path / "a", 1.0)
    b = _write(tmp_path / "b", 1.0, rng_delta=1)
    report = compare_checkpoints(a, b)
    assert not report["pass"]
    assert not report["rng"]["all_sequences_identical"]
    with (b / "dataloader" / "rank_0.pkl").open("wb") as stream:
        pickle.dump({"next_draw_index": 9}, stream)
    assert not compare_checkpoints(a, b)["dataloader"]["pass"]


def test_latest_marker_resolution(tmp_path: Path) -> None:
    checkpoint = _write(tmp_path / "job" / "checkpoints", 1.0)
    assert resolve_iteration(tmp_path / "job") == checkpoint.resolve()
