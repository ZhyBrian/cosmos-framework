"""Audit E2-H resolved configs, selected real windows, and metadata-only draw traces."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


HISTORY_FRAMES = (1, 5, 9, 17)
TRAIN_EPISODES = tuple(index for index in range(59) if index not in (13, 43, 49))
DRAW_COUNT = 48_000


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zarr", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def _assert_equal(actual: Any, expected: Any, label: str) -> None:
    import torch

    if not torch.equal(actual, expected):
        raise AssertionError(f"{label} differs")


def _load_configs(zarr_path: Path) -> tuple[dict[int, Any], dict[str, Any]]:
    os.environ["DATASET_PATH"] = str(zarr_path.resolve())
    os.environ["UMIFT_STAGE"] = "e1"
    os.environ.setdefault("BASE_CHECKPOINT_PATH", "/audit/not-loaded/base-checkpoint")
    os.environ.setdefault("WAN_VAE_PATH", "/audit/not-loaded/vae.pt")
    os.environ.setdefault("EDGE_HF_SNAPSHOT_PATH", "/audit/not-loaded/hf-snapshot")

    from cosmos_framework.configs.toml_config.sft_config import load_experiment_from_toml

    config_dir = Path(__file__).resolve().parents[1] / "toml/sft_config"
    resolved: dict[int, Any] = {}
    reports: dict[str, Any] = {}
    for history_frames in HISTORY_FRAMES:
        toml = config_dir / f"action_fd_umift_edge_h{history_frames}.toml"
        config = load_experiment_from_toml(toml)
        dataset_cfg = config.dataloader_train.dataloader.datasets.umift.dataset
        assert int(config.trainer.max_iter) == 3000
        assert int(config.checkpoint.save_iter) == 500
        assert list(config.scheduler.cycle_lengths) == [3000]
        assert list(config.scheduler.warm_up_steps) == [100]
        assert int(config.dataloader_train.dataloader.batch_size) == 1
        assert int(config.dataloader_train.batcher.max_batch_size) == 1
        assert int(config.trainer.grad_accum_iter) == 4
        assert str(dataset_cfg.split) == "refit_train"
        assert int(dataset_cfg.history_frames) == history_frames
        assert list(config.model.config.tokenizer.encode_exact_durations) == [history_frames + 16]
        resolved[history_frames] = config
        reports[str(history_frames)] = {
            "toml": str(toml),
            "max_iter": int(config.trainer.max_iter),
            "save_iter": int(config.checkpoint.save_iter),
            "cycle_lengths": list(config.scheduler.cycle_lengths),
            "warm_up_steps": list(config.scheduler.warm_up_steps),
            "microbatch": int(config.dataloader_train.dataloader.batch_size),
            "grad_accum_iter": int(config.trainer.grad_accum_iter),
            "split": str(dataset_cfg.split),
            "history_frames": int(dataset_cfg.history_frames),
            "encode_exact_durations": list(config.model.config.tokenizer.encode_exact_durations),
            "max_samples_per_batch": int(config.dataloader_train.batcher.max_batch_size),
            "normalize_loss_by_active": bool(
                config.model.config.rectified_flow_training_config.normalize_loss_by_active
            ),
        }
    return resolved, reports


def _build_datasets(zarr_path: Path, resolved: dict[int, Any]) -> tuple[dict[int, Any], Any]:
    from cosmos_framework.data.generator.action.datasets.umift_history_dataset import (
        get_umift_history_sft_dataset,
    )
    from cosmos_framework.data.generator.action.datasets.umift_zarr_dataset import (
        get_umift_zarr_sft_dataset,
    )

    history_datasets = {
        history_frames: get_umift_history_sft_dataset(
            str(zarr_path),
            split="refit_train",
            stage="e1",
            seed=42,
            resolution="256",
            fps=15.0,
            history_frames=history_frames,
            mode="forward_dynamics",
            tokenizer_config=None,
            max_action_dim=int(resolved[history_frames].model.config.max_action_dim),
        )
        for history_frames in HISTORY_FRAMES
    }
    legacy = get_umift_zarr_sft_dataset(
        str(zarr_path),
        split="refit_train",
        stage="e1",
        seed=42,
        resolution="256",
        fps=15.0,
        mode="forward_dynamics",
        tokenizer_config=None,
        max_action_dim=int(resolved[1].model.config.max_action_dim),
    )
    return history_datasets, legacy


def _audit_selected_windows(zarr_path: Path, datasets: dict[int, Any], legacy: Any) -> dict[str, Any]:
    import torch
    import zarr

    episode_ids = tuple(episode.episode_id for episode in datasets[1]._episodes)
    assert episode_ids == TRAIN_EPISODES
    assert all(
        tuple(episode.episode_id for episode in dataset._episodes) == TRAIN_EPISODES
        for dataset in datasets.values()
    )
    root = zarr.open_group(str(zarr_path), mode="r")
    checked = 0
    max_rgb_robot_alignment_seconds = 0.0
    padding_frames = Counter({str(history_frames): 0 for history_frames in HISTORY_FRAMES})

    for episode in datasets[1]._episodes:
        anchors = (0, (episode.window_count - 1) // 2, episode.window_count - 1)
        for anchor in anchors:
            h1 = datasets[1].get_window(episode.episode_id, anchor)
            old = legacy.get_window(episode.episode_id, anchor)
            _assert_equal(h1["video"], old["video"], "H1/legacy video")
            for key in ("action", "physical_action", "model_action"):
                _assert_equal(h1[key], old[key], f"H1/legacy {key}")

            for history_frames, dataset in datasets.items():
                sample = dataset.get_window(episode.episode_id, anchor)
                assert tuple(sample["video"].shape) == (3, history_frames + 16, 256, 256)
                for key in ("action", "physical_action", "model_action"):
                    _assert_equal(sample[key], h1[key], f"H{history_frames}/H1 {key}")
                _assert_equal(
                    sample["video"][:, history_frames:],
                    h1["video"][:, 1:],
                    f"H{history_frames}/H1 future RGB",
                )

                requested = anchor - 2 * np.arange(history_frames - 1, -1, -1, dtype=np.int64)
                expected_mask = torch.from_numpy(requested >= 0)
                expected_indices = torch.from_numpy(np.maximum(requested, 0).copy())
                _assert_equal(sample["history_real_mask"], expected_mask, "history real mask")
                _assert_equal(sample["history_source_indices"], expected_indices, "history source indices")
                assert bool(torch.all(sample["history_source_indices"] <= anchor))
                assert int(sample["history_padding_count"]) == int((requested < 0).sum())
                padding_frames[str(history_frames)] += int(sample["history_padding_count"])

                group = root["data"][f"episode_{episode.episode_id}"]
                video_indices = sample["video_source_indices"].numpy()
                rgb_ts = np.asarray(group["rgb_time_stamps_0"].oindex[video_indices]).reshape(-1)
                robot_ts = np.asarray(group["robot_time_stamps_0"].oindex[video_indices]).reshape(-1)
                assert np.isfinite(rgb_ts).all() and np.isfinite(robot_ts).all()
                assert np.array_equal(sample["video_timestamps"].numpy(), rgb_ts)
                alignment = float(np.max(np.abs(rgb_ts - robot_ts)))
                assert alignment <= 0.020 + 1e-12
                max_rgb_robot_alignment_seconds = max(max_rgb_robot_alignment_seconds, alignment)
                checked += 1

    expected_checks = len(TRAIN_EPISODES) * 3 * len(HISTORY_FRAMES)
    assert checked == expected_checks
    return {
        "episodes": len(TRAIN_EPISODES),
        "anchors_per_episode": 3,
        "history_variants": list(HISTORY_FRAMES),
        "transformed_samples_checked": checked,
        "h1_legacy_windows_checked": len(TRAIN_EPISODES) * 3,
        "max_rgb_robot_alignment_seconds": max_rgb_robot_alignment_seconds,
        "selected_window_padding_frames": dict(padding_frames),
    }


def _audit_draws(datasets: dict[int, Any]) -> dict[str, Any]:
    traces: dict[int, dict[str, Any]] = {}
    for history_frames, dataset in datasets.items():
        digest = hashlib.sha256()
        session_counts: Counter[str] = Counter()
        episode_counts: Counter[str] = Counter()
        padded_frames = 0
        for draw_index, episode, anchor in dataset._training_draws(0):
            if draw_index == DRAW_COUNT:
                break
            digest.update(f"{draw_index}\t{episode.session_id}\t{episode.episode_id}\t{anchor}\n".encode())
            session_counts[episode.session_id] += 1
            episode_counts[str(episode.episode_id)] += 1
            padded_frames += max(0, history_frames - (anchor // 2 + 1))
        assert sum(episode_counts.values()) == DRAW_COUNT
        traces[history_frames] = {
            "sha256": digest.hexdigest(),
            "draw_count": DRAW_COUNT,
            "session_counts": dict(sorted(session_counts.items())),
            "episode_counts": dict(sorted(episode_counts.items(), key=lambda item: int(item[0]))),
            "history_padding_frames": padded_frames,
            "history_padding_fraction": padded_frames / (DRAW_COUNT * history_frames),
        }
    hashes = {trace["sha256"] for trace in traces.values()}
    assert len(hashes) == 1
    return {"trace_sha_equal_across_history": True, "by_history": {str(key): value for key, value in traces.items()}}


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if not args.zarr.is_dir():
        raise FileNotFoundError(f"--zarr is not a directory: {args.zarr}")
    if args.output.exists():
        raise FileExistsError(f"--output already exists: {args.output}")
    resolved, config_report = _load_configs(args.zarr)
    datasets, legacy = _build_datasets(args.zarr, resolved)
    report = {
        "passed": True,
        "zarr": str(args.zarr.resolve()),
        "configs": config_report,
        "selected_windows": _audit_selected_windows(args.zarr, datasets, legacy),
        "training_draws": _audit_draws(datasets),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
