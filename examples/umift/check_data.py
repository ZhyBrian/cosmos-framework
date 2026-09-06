#!/usr/bin/env python3
"""CPU/Gloo end-to-end check for the UMI-FT factory and PackingDataLoader."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any


def _one(value: Any) -> Any:
    while isinstance(value, list) and len(value) == 1:
        value = value[0]
    return value


def _sample_id(batch: dict[str, Any]) -> tuple[int, int, str]:
    episode = _one(batch["episode_id"])
    start = _one(batch["window_start"])
    source = _one(batch["source_id"])
    if hasattr(episode, "item"):
        episode = episode.item()
    if hasattr(start, "item"):
        start = start.item()
    return int(episode), int(start), str(source)


def _single_sidecar(value: Any, *, name: str, shape: tuple[int, ...]) -> Any:
    """Unwrap only the single-sample batch axis added to tensor sidecars."""
    value = _one(value)
    actual = tuple(value.shape)
    if actual == shape:
        return value
    assert actual == (1, *shape), f"{name} must be {shape} or single-batch {(1, *shape)}, got {actual}"
    return value[0]


def _validate_batch(batch: dict[str, Any]) -> dict[str, Any]:
    import torch

    physical = _single_sidecar(batch["physical_action"], name="physical_action", shape=(16, 10))
    action_raw = _single_sidecar(batch["action_raw"], name="action_raw", shape=(16, 10))
    model_action = _single_sidecar(batch["model_action"], name="model_action", shape=(16, 10))
    action = _single_sidecar(batch["action"], name="action", shape=(16, 64))
    valid_mask = _one(batch["action_valid_mask"])
    text_tokens = _one(batch["text_token_ids"])
    plan = _one(batch["sequence_plan"])
    video = _single_sidecar(batch["video"], name="video", shape=(3, 17, 256, 256))
    source_indices = _single_sidecar(batch["source_indices"], name="source_indices", shape=(17,))
    timestamps = _single_sidecar(batch["timestamps"], name="timestamps", shape=(17,))
    torch.testing.assert_close(action_raw, physical)
    torch.testing.assert_close(action[..., :10], model_action, atol=2e-6, rtol=2e-6)
    assert tuple(action.shape) == (16, 64)
    assert valid_mask.dtype == torch.bool and valid_mask.tolist() == [True] * 10 + [False] * 54
    assert plan.has_vision and plan.has_action and plan.has_text
    assert plan.condition_frame_indexes_vision == [0]
    assert plan.condition_frame_indexes_action == list(range(16))
    assert source_indices.tolist() == list(range(int(source_indices[0]), int(source_indices[0]) + 34, 2))
    relative_time = timestamps - timestamps[0]
    ideal_time = torch.arange(17, dtype=relative_time.dtype, device=relative_time.device) / 15.0
    max_time_grid_error_ms = float(torch.max(torch.abs(relative_time - ideal_time)).item() * 1000.0)
    assert max_time_grid_error_ms <= 20.0 + 1e-9
    return {
        "sample_id": _sample_id(batch),
        "text_tokens": text_tokens.detach().cpu().tolist(),
        "physical_shape": list(physical.shape),
        "model_shape": list(action.shape),
        "video_shape": list(video.shape),
        "source_indices": source_indices.detach().cpu().tolist(),
        "max_time_grid_error_ms": max_time_grid_error_ms,
    }


def _configure_environment(args: argparse.Namespace) -> None:
    os.environ["DATASET_PATH"] = str(args.dataset.resolve())
    os.environ["EDGE_HF_SNAPSHOT_PATH"] = str(args.hf_snapshot.resolve())
    temp_dir = Path(tempfile.gettempdir())
    os.environ.setdefault("BASE_CHECKPOINT_PATH", str(temp_dir / "cosmos-umift-check-data-unused-dcp"))
    os.environ.setdefault("WAN_VAE_PATH", str(temp_dir / "cosmos-umift-check-data-unused-vae.pth"))
    os.environ["UMIFT_STAGE"] = args.stage
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"


def _load_config(args: argparse.Namespace) -> Any:
    _configure_environment(args)
    from cosmos_framework.configs.toml_config.sft_config import load_experiment_from_toml

    return load_experiment_from_toml(args.toml)


def _init_gloo() -> tuple[int, int]:
    import torch

    if torch.distributed.is_initialized():
        return torch.distributed.get_rank(), torch.distributed.get_world_size()
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1:
        torch.distributed.init_process_group("gloo", init_method="env://")
    else:
        handle = tempfile.NamedTemporaryFile(prefix="cosmos-umift-check-data-gloo-", delete=True)
        torch.distributed.init_process_group("gloo", init_method=f"file://{handle.name}", rank=0, world_size=1)
    return torch.distributed.get_rank(), torch.distributed.get_world_size()


def _build_loader(config: Any) -> Any:
    from cosmos_framework.utils.lazy_config import instantiate

    return instantiate(config.dataloader_train)


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    assert args.dataset.is_dir(), args.dataset
    assert args.hf_snapshot.is_dir(), args.hf_snapshot
    rank, world_size = _init_gloo()
    config = _load_config(args)
    loader = _build_loader(config)
    full_iter = iter(loader)
    full = [_validate_batch(next(full_iter)) for _ in range(args.samples)]
    token_reference = full[0]["text_tokens"]
    assert all(item["text_tokens"] == token_reference for item in full), "empty-caption tokens changed within rank"

    resumed = _build_loader(config)
    resumed.set_start_iteration(args.resume_microbatch)
    resumed_iter = iter(resumed)
    suffix = [_validate_batch(next(resumed_iter)) for _ in range(args.samples - args.resume_microbatch)]
    assert [item["sample_id"] for item in suffix] == [
        item["sample_id"] for item in full[args.resume_microbatch :]
    ]
    local = {
        "rank": rank,
        "world_size": world_size,
        "stage": args.stage,
        "resume_offset_unit": "rank-local microbatch",
        "sample_ids": [item["sample_id"] for item in full],
        "text_tokens": token_reference,
        "resume_suffix_matches": True,
    }
    reports: list[Any] = [None] * world_size
    torch.distributed.all_gather_object(reports, local)
    if world_size > 1 and args.stage != "smoke":
        sequences = [report["sample_ids"] for report in reports]
        assert len({json.dumps(sequence) for sequence in sequences}) == world_size, "rank streams are not distinct"
    return {"ranks": reports}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--hf-snapshot", type=Path, required=True)
    parser.add_argument(
        "--toml", type=Path, default=Path("examples/toml/sft_config/action_fd_umift_edge.toml")
    )
    parser.add_argument("--stage", choices=("smoke", "overfit", "e1"), default="smoke")
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--resume-microbatch", type=int, default=5)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args(argv)
    if not 0 <= args.resume_microbatch < args.samples:
        parser.error("resume-microbatch must satisfy 0 <= offset < samples")
    report = run(args)
    if int(os.environ.get("RANK", "0")) == 0:
        payload = json.dumps(report, indent=2, sort_keys=True)
        print(payload)
        if args.json_out:
            args.json_out.parent.mkdir(parents=True, exist_ok=True)
            args.json_out.write_text(payload + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
