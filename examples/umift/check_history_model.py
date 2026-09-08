"""Run the four-rank E2 history model correctness preflight on one real window per rank."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np


SUPPORTED_HISTORY_FRAMES = (1, 5, 9, 17)
NUM_STEPS = 30


def _history_frames(value: str) -> int:
    parsed = int(value)
    if parsed not in SUPPORTED_HISTORY_FRAMES:
        raise argparse.ArgumentTypeError(f"must be one of {SUPPORTED_HISTORY_FRAMES}")
    return parsed


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zarr", type=Path, required=True)
    parser.add_argument("--sft-toml", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New directory for rank JSON reports.")
    parser.add_argument("--history-frames", type=_history_frames, required=True)
    return parser.parse_args(argv)


def _shape(value: Any) -> list[int]:
    return [int(item) for item in value.shape]


def _comparison(actual: Any, expected: Any) -> dict[str, Any]:
    import torch

    if actual.shape != expected.shape:
        return {
            "shape_equal": False,
            "actual_shape": _shape(actual),
            "expected_shape": _shape(expected),
        }
    delta = actual.detach().float() - expected.detach().float()
    return {
        "shape_equal": True,
        "actual_shape": _shape(actual),
        "expected_shape": _shape(expected),
        "exact_equal": bool(torch.equal(actual, expected)),
        "max_abs": float(delta.abs().max().item()),
        "mean_abs": float(delta.abs().mean().item()),
    }


def _numpy_comparison(actual: np.ndarray, expected: np.ndarray) -> dict[str, Any]:
    if actual.shape != expected.shape:
        return {
            "shape_equal": False,
            "actual_shape": _shape(actual),
            "expected_shape": _shape(expected),
        }
    delta = actual.astype(np.float32, copy=False) - expected.astype(np.float32, copy=False)
    return {
        "shape_equal": True,
        "actual_shape": _shape(actual),
        "expected_shape": _shape(expected),
        "exact_equal": bool(np.array_equal(actual, expected)),
        "max_abs": float(np.abs(delta).max()),
        "mean_abs": float(np.abs(delta).mean()),
    }


def _normalized_video(video: Any) -> Any:
    import torch

    return video.unsqueeze(0).to(device="cuda", dtype=torch.float32) / 127.5 - 1.0


def _validate_launch(args: argparse.Namespace) -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0,1,2,3":
        raise ValueError("history model check requires CUDA_VISIBLE_DEVICES=0,1,2,3")
    if os.environ.get("WORLD_SIZE") != "4":
        raise ValueError("history model check requires torchrun --nproc_per_node=4")
    for label in ("zarr", "sft_toml", "checkpoint"):
        path = getattr(args, label)
        if not path.exists():
            raise FileNotFoundError(f"--{label.replace('_', '-')} does not exist: {path}")
    if args.output.exists():
        raise FileExistsError(f"--output must be a new directory: {args.output}")


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    from cosmos_framework.data.generator.action.datasets.umift_history_dataset import (
        get_umift_history_sft_dataset,
    )
    from examples.umift.history_infer import (
        build_history_batch,
        load_history_model,
        run_history_prediction,
        validate_history_sample,
    )
    from examples.umift.infer import _move_batch_to_cuda, validate_independent_parallelism

    model, resolved, load_evidence = load_history_model(
        args.sft_toml,
        args.checkpoint,
        history_frames=args.history_frames,
        independent_windows=True,
    )
    model.eval()
    validate_independent_parallelism(model.parallel_dims)
    if not callable(getattr(model, "encode", None)) or not callable(getattr(model, "decode", None)):
        raise TypeError("loaded history model must expose callable encode and decode methods")
    if not torch.distributed.is_initialized() or torch.distributed.get_world_size() != 4:
        raise RuntimeError("load_history_model must initialize a four-rank process group")
    rank = torch.distributed.get_rank()
    tokenizer_config = resolved.model.config.vlm_config.tokenizer
    dataset = get_umift_history_sft_dataset(
        str(args.zarr),
        split="history",
        stage="e1",
        seed=42,
        resolution="256",
        fps=15.0,
        history_frames=args.history_frames,
        mode="forward_dynamics",
        tokenizer_config=tokenizer_config,
        max_action_dim=int(resolved.model.config.max_action_dim),
    )
    window_start = 64 + rank * 32
    sample = dataset.get_window(13, window_start)
    validate_history_sample(sample)
    expected_frames = args.history_frames + 16
    if tuple(sample["video"].shape) != (3, expected_frames, 256, 256):
        raise ValueError(f"unexpected history video shape: {tuple(sample['video'].shape)}")
    plan = sample["sequence_plan"]
    expected_condition_indexes = list(range((args.history_frames - 1) // 4 + 1))
    if list(plan.condition_frame_indexes_vision) != expected_condition_indexes:
        raise ValueError(f"unexpected vision condition indexes: {plan.condition_frame_indexes_vision}")
    if list(plan.condition_frame_indexes_action) != list(range(16)):
        raise ValueError("history model check requires exactly 16 clean action steps")
    if int(plan.action_start_frame_offset) != args.history_frames:
        raise ValueError(f"action temporal offset must equal H={args.history_frames}")

    original_video = sample["video"].clone()
    perturbed_video = original_video.clone()
    perturbed_video[:, args.history_frames :] = 255 - perturbed_video[:, args.history_frames :]
    if torch.equal(original_video[:, args.history_frames :], perturbed_video[:, args.history_frames :]):
        raise AssertionError("future RGB perturbation did not change the future pixels")
    if not torch.equal(original_video[:, : args.history_frames], perturbed_video[:, : args.history_frames]):
        raise AssertionError("future RGB perturbation changed the pixel history prefix")

    with torch.inference_mode():
        full_latent = model.encode(_normalized_video(original_video))
        perturbed_latent = model.encode(_normalized_video(perturbed_video))
        history_latent = model.encode(_normalized_video(original_video[:, : args.history_frames]))
        decoded = model.decode(full_latent)
        expected_decoded_shape = (1, 3, expected_frames, 256, 256)
        if tuple(decoded.shape) != expected_decoded_shape:
            raise ValueError(f"VAE decode must return {expected_decoded_shape}, got {tuple(decoded.shape)}")
        latent_prefix_frames = len(expected_condition_indexes)
        full_prefix = full_latent[:, :, :latent_prefix_frames]
        perturb_prefix_comparison = _comparison(full_prefix, perturbed_latent[:, :, :latent_prefix_frames])
        if not perturb_prefix_comparison.get("shape_equal"):
            raise AssertionError(
                f"future RGB changed the encoded history-prefix shape: {perturb_prefix_comparison}"
            )
        if perturb_prefix_comparison["max_abs"] != 0.0:
            raise AssertionError(
                f"future RGB leaked into the encoded history prefix: {perturb_prefix_comparison}"
            )
        history_encode_comparison = _comparison(full_prefix, history_latent)

        batch = _move_batch_to_cuda(build_history_batch(sample))
        prediction = run_history_prediction(
            model,
            batch,
            history_frames=args.history_frames,
            noise_seed=0,
            num_steps=NUM_STEPS,
        )
        prediction = np.asarray(prediction, dtype=np.float32)
        if prediction.shape != (expected_frames, 256, 256, 3):
            raise ValueError(
                f"history prediction must be THWC ({expected_frames},256,256,3), got {prediction.shape}"
            )
        if not np.isfinite(prediction).all():
            raise ValueError("history prediction contains non-finite values")

        h1_legacy = None
        if args.history_frames == 1:
            from examples.umift.infer import build_singleton_batch, run_forward_dynamics

            legacy_batch = _move_batch_to_cuda(build_singleton_batch(sample))
            legacy_prediction = run_forward_dynamics(
                model, legacy_batch, noise_seed=0, num_steps=NUM_STEPS
            )
            h1_legacy = _numpy_comparison(
                prediction, np.asarray(legacy_prediction, dtype=np.float32)
            )

    report = {
        "rank": rank,
        "world_size": torch.distributed.get_world_size(),
        "episode_id": 13,
        "window_start": window_start,
        "history_frames": args.history_frames,
        "noise_seed": 0,
        "num_steps": NUM_STEPS,
        "model_training": bool(model.training),
        "cuda_device": int(torch.cuda.current_device()),
        "cuda_device_name": torch.cuda.get_device_name(),
        "load_evidence": load_evidence,
        "shapes": {
            "sample_video_cthw": _shape(sample["video"]),
            "history_condition_cthw": _shape(sample["video"][:, : args.history_frames]),
            "future_target_cthw": _shape(sample["video"][:, args.history_frames :]),
            "action_condition_td": _shape(sample["action"]),
            "physical_action_td": _shape(sample["physical_action"]),
            "batch_video_bcthw": [len(batch["video"]), *_shape(batch["video"][0])],
            "batch_action_btd": [len(batch["action"]), *_shape(batch["action"][0])],
            "full_vae_input_bcthw": [1, *_shape(sample["video"])],
            "full_vae_latent": _shape(full_latent),
            "history_vae_latent": _shape(history_latent),
            "decoded_bcthw": _shape(decoded),
            "prediction_thwc": _shape(prediction),
        },
        "dtypes": {
            "sample_video": str(sample["video"].dtype),
            "full_vae_latent": str(full_latent.dtype),
            "history_vae_latent": str(history_latent.dtype),
            "decoded": str(decoded.dtype),
            "prediction": str(prediction.dtype),
        },
        "conditioning": {
            "vision_frame_indexes": list(plan.condition_frame_indexes_vision),
            "action_step_indexes": list(plan.condition_frame_indexes_action),
            "action_start_frame_offset": int(plan.action_start_frame_offset),
            "expected_future_frames": 16,
        },
        "vae": {
            "future_pixel_perturbation_preserved_history_exactly": True,
            "future_perturbation_prefix": perturb_prefix_comparison,
            "history_only_vs_full_prefix": history_encode_comparison,
            "decoded_finite": bool(torch.isfinite(decoded).all().item()),
        },
        "generation": {"finite": True, "h1_legacy_same_noise_comparison": h1_legacy},
        "loss_mask": {
            "checked": False,
            "reason": "This inference preflight does not expose the training packed-sequence loss mask.",
        },
    }
    return report


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    _validate_launch(args)
    import torch

    from cosmos_framework.inference.common.init import init_script

    init_script()
    try:
        report = run(args)
        rank = int(report["rank"])
        if rank == 0:
            args.output.mkdir(parents=True)
        torch.distributed.barrier()
        output_path = args.output / f"rank_{rank}.json"
        output_path.write_text(
            json.dumps(report, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
        torch.distributed.barrier()
        return 0
    finally:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(main())
