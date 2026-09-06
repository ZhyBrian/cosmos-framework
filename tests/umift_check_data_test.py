from __future__ import annotations

import hashlib
import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace

import torch
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "examples/umift/check_data.py"
SPEC = importlib.util.spec_from_file_location("umift_check_data", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_evidence_fields_come_from_executed_arguments(tmp_path) -> None:
    dataset = tmp_path / "relative-parent" / "dataset.zarr"
    dataset.mkdir(parents=True)
    root_metadata = b'{"zarr_format":2}'
    (dataset / ".zgroup").write_bytes(root_metadata)
    digest = hashlib.sha256()
    digest.update(b".zgroup\0")
    digest.update(root_metadata)
    digest.update(b"\0")
    args = SimpleNamespace(dataset=dataset, samples=37, resume_microbatch=19, stage="overfit")

    report = MODULE._evidence_fields(args, world_size=4)

    assert report == {
        "schema_version": 1,
        "samples": 37,
        "resume_microbatch": 19,
        "seed": 42,
        "dataset": str(dataset.resolve()),
        "dataset_root_metadata_sha256": digest.hexdigest(),
        "stage": "overfit",
        "world_size": 4,
    }


def test_validate_batch_checks_fd_masks_and_action_spaces() -> None:
    physical = torch.zeros(1, 16, 10)
    model = torch.arange(160, dtype=torch.float32).reshape(1, 16, 10)
    action = torch.cat((model[0], torch.zeros(16, 54)), dim=-1)
    plan = SimpleNamespace(
        has_vision=True,
        has_action=True,
        has_text=True,
        condition_frame_indexes_vision=[0],
        condition_frame_indexes_action=list(range(16)),
    )
    report = MODULE._validate_batch(
        {
            "physical_action": physical,
            "action_raw": [physical[0].clone()],
            "model_action": model,
            "action": [action],
            "action_valid_mask": [torch.tensor([True] * 10 + [False] * 54)],
            "text_token_ids": [torch.tensor([1, 2, 3])],
            "sequence_plan": [plan],
            "episode_id": torch.tensor([3]),
            "window_start": torch.tensor([64]),
            "source_id": ["session#seg0"],
            "video": [torch.zeros(3, 17, 256, 256)],
            "source_indices": torch.arange(0, 34, 2).unsqueeze(0),
            "timestamps": (torch.arange(17, dtype=torch.float64) / 15).unsqueeze(0),
        }
    )

    assert report["sample_id"] == (3, 64, "session#seg0")
    assert report["text_tokens"] == [1, 2, 3]
    assert report["video_shape"] == [3, 17, 256, 256]
    assert report["source_indices"] == list(range(0, 34, 2))
    assert report["max_time_grid_error_ms"] == pytest.approx(0.0)


def test_validate_batch_rejects_multi_sample_sidecar_batch() -> None:
    with pytest.raises(AssertionError, match="physical_action must be"):
        MODULE._single_sidecar(torch.zeros(2, 16, 10), name="physical_action", shape=(16, 10))


def test_configure_environment_uses_cosmos_prefixed_tmp_paths(tmp_path, monkeypatch) -> None:
    cosmos_tmp = tmp_path / "cosmos_runs" / "tmp"
    cosmos_tmp.mkdir(parents=True)
    monkeypatch.setattr(MODULE.tempfile, "gettempdir", lambda: str(cosmos_tmp))
    monkeypatch.delenv("BASE_CHECKPOINT_PATH", raising=False)
    monkeypatch.delenv("WAN_VAE_PATH", raising=False)

    MODULE._configure_environment(
        SimpleNamespace(
            dataset=tmp_path / "dataset.zarr",
            hf_snapshot=tmp_path / "hf",
            stage="smoke",
        )
    )

    assert os.environ["BASE_CHECKPOINT_PATH"] == str(cosmos_tmp / "cosmos-umift-check-data-unused-dcp")
    assert os.environ["WAN_VAE_PATH"] == str(cosmos_tmp / "cosmos-umift-check-data-unused-vae.pth")
