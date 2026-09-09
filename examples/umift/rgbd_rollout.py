"""E3-Dout full-suffix RGBD rollout with float feedback and frozen actions."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np


HISTORY_FRAMES = 5
FUTURE_FRAMES = 16
MODEL_METHODS = ("B0", "E3-A", "E3-Z", "E3-S")
HELD_OUT_EPISODES = (13, 43, 49)
_SOURCE_EPISODE_FIELDS = (
    "episode_id",
    "source_id",
    "raw_session",
    "source_length",
    "frame_count",
    "actions_path",
    "frame_indices_path",
    "truth_path",
    "input_files_sha256",
    "chunks",
)
_SOURCE_EPISODE_DEFAULTS = {
    "start_percent": lambda episode: 0,
    "initial_selected_frame": lambda episode: 0,
    "initial_source_frame": lambda episode: 0,
    "parent_frame_count": lambda episode: episode["frame_count"],
    "initial_episode_elapsed_seconds": lambda episode: 0.0,
}


def array_sha(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canvas_from_rgb_depth(rgb: np.ndarray, depth_m: np.ndarray) -> np.ndarray:
    """Rebuild the canonical float canvas used for generated-history feedback.

    RGB is clipped in display space and mapped to ``[-1, 1]``. Raw saved depth
    remains untouched on disk; only this feedback projection clips it to
    ``[0, 0.5]`` and repeats the scalar normalized depth over three channels.
    """
    rgb_value = np.asarray(rgb)
    depth_value = np.asarray(depth_m)
    if rgb_value.dtype != np.float32 or depth_value.dtype != np.float32:
        raise ValueError("feedback RGB and depth must both be float32")
    if (
        rgb_value.ndim != 4
        or rgb_value.shape[-1] != 3
        or depth_value.shape != rgb_value.shape[:3]
        or not np.isfinite(rgb_value).all()
        or not np.isfinite(depth_value).all()
    ):
        raise ValueError("feedback requires finite RGB THWC and matching depth THW")
    left = np.clip(rgb_value, 0.0, 1.0) * np.float32(2.0) - np.float32(1.0)
    scalar = np.clip(depth_value, 0.0, 0.5) * np.float32(4.0) - np.float32(1.0)
    right = np.repeat(scalar[..., None], 3, axis=-1)
    return np.concatenate((left, right), axis=2).astype(np.float32, copy=False)


def _split_raw_canvas(canvas: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    value = np.asarray(canvas)
    if (
        value.dtype != np.float32
        or value.ndim != 4
        or value.shape[-1] != 3
        or value.shape[2] != 2 * value.shape[1]
        or not np.isfinite(value).all()
    ):
        raise ValueError("prediction must be finite float32 T,H,2H,3 RGBD canvas")
    midpoint = value.shape[2] // 2
    rgb = np.clip((value[:, :, :midpoint] + 1.0) / 2.0, 0.0, 1.0).astype(np.float32)
    depth = ((value[:, :, midpoint:].mean(axis=-1) + 1.0) / 4.0).astype(np.float32)
    return rgb, depth


def rollout(
    initial_history: np.ndarray,
    chunks: list[dict[str, Any]],
    predict: Callable[[np.ndarray, dict[str, Any]], np.ndarray],
    rgb_output: np.ndarray,
    depth_output: np.ndarray,
    *,
    initial_history_real_mask: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    """Roll the full requested suffix using only initial H5 RGBD and generated feedback."""
    history = np.asarray(initial_history)
    if (
        history.dtype != np.float32
        or history.ndim != 4
        or history.shape[0] != HISTORY_FRAMES
        or history.shape[-1] != 3
        or history.shape[2] != 2 * history.shape[1]
        or not np.isfinite(history).all()
        or history.min() < -1.0
        or history.max() > 1.0
    ):
        raise ValueError("initial history must be finite float32 H5 normalized RGBD canvas")
    height = history.shape[1]
    width = history.shape[2] // 2
    depth_tile = history[:, :, width:]
    if not (
        np.array_equal(depth_tile[..., 0], depth_tile[..., 1])
        and np.array_equal(depth_tile[..., 0], depth_tile[..., 2])
    ):
        raise ValueError("feedback history depth tile must be the canonical scalar gray3 projection")
    if initial_history_real_mask is None:
        real_mask = np.ones(HISTORY_FRAMES, dtype=bool)
    else:
        real_mask = np.asarray(initial_history_real_mask)
        if real_mask.dtype != np.bool_ or real_mask.shape != (HISTORY_FRAMES,):
            raise ValueError("initial history real mask must be bool[5]")
        real_mask = real_mask.copy()
    count = 1 + sum(int(chunk["steps"]) for chunk in chunks)
    if (
        rgb_output.dtype != np.float32
        or rgb_output.shape != (count, height, width, 3)
        or depth_output.dtype != np.float32
        or depth_output.shape != (count, height, width)
    ):
        raise ValueError("RGB/depth outputs do not match retained suffix dimensions")

    history = history.copy()
    initial_sha = array_sha(history)
    anchor_rgb, anchor_depth = _split_raw_canvas(history[-1:])
    rgb_output[0] = anchor_rgb[0]
    depth_output[0] = anchor_depth[0]
    records: list[dict[str, Any]] = []
    expected_start = 0
    for expected_index, chunk in enumerate(chunks):
        steps = int(chunk["steps"])
        start = int(chunk["output_start"])
        if (
            int(chunk["index"]) != expected_index
            or start != expected_start
            or not 1 <= steps <= FUTURE_FRAMES
            or int(chunk.get("padding_steps", FUTURE_FRAMES - steps)) != FUTURE_FRAMES - steps
            or "noise_seed" not in chunk
        ):
            raise ValueError("chunks must be contiguous 16-step blocks with an audited noise seed")
        before = history.copy()
        before_sha = array_sha(before)
        started = time.perf_counter()
        prediction = np.asarray(predict(before.copy(), chunk))
        expected_shape = (HISTORY_FRAMES + FUTURE_FRAMES, height, 2 * width, 3)
        if prediction.dtype != np.float32 or prediction.shape != expected_shape or not np.isfinite(prediction).all():
            raise ValueError(f"invalid raw RGBD prediction: {prediction.dtype} {prediction.shape}")
        predicted_rgb, predicted_depth = _split_raw_canvas(prediction)
        retained_rgb = predicted_rgb[HISTORY_FRAMES : HISTORY_FRAMES + steps]
        retained_depth = predicted_depth[HISTORY_FRAMES : HISTORY_FRAMES + steps]
        rgb_output[start + 1 : start + steps + 1] = retained_rgb
        depth_output[start + 1 : start + steps + 1] = retained_depth

        generated_canvas = canvas_from_rgb_depth(
            np.asarray(rgb_output[start + 1 : start + steps + 1]),
            np.asarray(depth_output[start + 1 : start + steps + 1]),
        )
        history = np.concatenate((before, generated_canvas), axis=0)[-HISTORY_FRAMES:].copy()
        record = {
            **chunk,
            "history_sha256": before_sha,
            "history_source": "initial_observed_h5_rgbd" if expected_index == 0 else "generated_rolling_rgbd",
            "initial_history_slots": HISTORY_FRAMES if expected_index == 0 else 0,
            "initial_real_slots": int(real_mask.sum()) if expected_index == 0 else 0,
            "initial_padding_slots": int((~real_mask).sum()) if expected_index == 0 else 0,
            "initial_history_sha256": initial_sha,
            "retained_rgb_sha256": array_sha(retained_rgb),
            "retained_raw_depth_m_sha256": array_sha(retained_depth),
            "feedback_terminal_sha256": array_sha(history[-1]),
            "feedback_history_sha256": array_sha(history),
            "feedback_projection": "clip RGB to [0,1]; clip depth to [0,0.5]m; gray3; float32",
            "seconds": time.perf_counter() - started,
        }
        if records and before_sha != records[-1]["feedback_history_sha256"]:
            raise ValueError("broken generated RGBD feedback chain")
        records.append(record)
        expected_start += steps
    return records


def _array_summary(
    array: np.ndarray, *, depth_m: bool = False, block_frames: int = 64
) -> dict[str, float | int]:
    minimum = float("inf")
    maximum = float("-inf")
    total = 0.0
    count = 0
    outside = 0
    for start in range(0, len(array), block_frames):
        block = np.asarray(array[start : start + block_frames], dtype=np.float64)
        if not np.isfinite(block).all():
            raise ValueError("saved rollout contains non-finite values")
        minimum = min(minimum, float(block.min()))
        maximum = max(maximum, float(block.max()))
        total += float(block.sum())
        count += int(block.size)
        if depth_m:
            outside += int(((block < 0.0) | (block > 0.5)).sum())
    result = {
        "count": count,
        "min": minimum,
        "max": maximum,
        "mean": total / count,
    }
    if depth_m:
        result["outside_depth_display_range_count"] = outside
        result["outside_depth_display_range_fraction"] = outside / count
    return result


def _run_evidence_path(output_root: Path, phase: str, rank: int) -> Path:
    path = output_root / f"run_{phase}_rank{rank}.json"
    if path.exists():
        raise FileExistsError(f"refusing to overwrite rollout run evidence: {path}")
    return path


def _validate_manifest(manifest: dict[str, Any], manifest_path: Path) -> None:
    if (
        manifest.get("experiment_id") != "E3-Dout"
        or manifest.get("protocol") != "e3-dout-rgbd-open-loop-v1"
        or manifest.get("history_frames") != HISTORY_FRAMES
        or manifest.get("future_frames") != FUTURE_FRAMES
    ):
        raise ValueError("prepared manifest is not the E3-Dout H5 RGBD rollout protocol")
    source_path = Path(manifest["source_manifest"])
    if manifest.get("source_manifest_sha256") != file_sha(source_path):
        raise ValueError("frozen suffix source manifest identity changed")
    source = json.loads(source_path.read_text())
    for field in ("base_checkpoint", "zarr_path"):
        if not source.get(field) or manifest.get(field) != source.get(field):
            raise ValueError(f"prepared {field} differs from the bound source manifest")
    for field, fixed_value in (("num_steps", 30), ("sampling_seed", 0), ("guidance", 1.0)):
        if source.get(field) != fixed_value or manifest.get(field) != source.get(field):
            raise ValueError(f"prepared {field} differs from the bound source manifest")
    source_episodes = source.get("episodes")
    prepared_episodes = manifest.get("episodes")
    if (
        not isinstance(source_episodes, list)
        or not isinstance(prepared_episodes, list)
        or tuple(episode.get("episode_id") for episode in source_episodes) != HELD_OUT_EPISODES
        or len(prepared_episodes) != len(source_episodes)
    ):
        raise ValueError("prepared episodes differ from the bound source manifest")
    for source_episode, prepared_episode in zip(source_episodes, prepared_episodes, strict=True):
        for field in _SOURCE_EPISODE_FIELDS:
            if prepared_episode.get(field) != source_episode.get(field):
                raise ValueError(
                    f"prepared episode {prepared_episode.get('episode_id')} {field} differs from source manifest"
                )
        for field, default in _SOURCE_EPISODE_DEFAULTS.items():
            expected = source_episode.get(field, default(source_episode))
            if prepared_episode.get(field) != expected:
                raise ValueError(
                    f"prepared episode {prepared_episode.get('episode_id')} {field} differs from source manifest"
                )
    if manifest.get("selection_sha256") != file_sha(Path(manifest["selection_file"])):
        raise ValueError("selected checkpoint identity changed")
    selected_checkpoint = Path(manifest["selected_checkpoint"])
    if manifest.get("checkpoint_metadata_sha256") != file_sha(selected_checkpoint / ".metadata"):
        raise ValueError("selected checkpoint metadata identity changed")
    if manifest.get("zarr_complete_manifest_sha256") != (
        "308f4d46132885375eb00c5578d52529512f0bca2c4e422c13724bfef1c6b9e6"
    ):
        raise ValueError("prepared rollout lacks the audited canonical Zarr tree identity")
    if len(manifest.get("episodes", [])) != len(HELD_OUT_EPISODES):
        raise ValueError("prepared manifest must contain the three frozen held-out episodes")
    if not manifest_path.is_absolute():
        raise ValueError("manifest path must be resolved before validation")


def _initial_history(zarr_path: str, episode: dict[str, Any]):
    import torch
    import zarr

    from cosmos_framework.data.generator.action.datasets.umift_history_dataset import _resize_rgb
    from cosmos_framework.data.generator.action.datasets.umift_rgbd_dataset import (
        _resize_depth_nearest,
        pack_rgbd_canvas,
    )

    anchor = int(episode["initial_source_frame"])
    expected_indices = np.maximum(
        anchor - 2 * np.arange(HISTORY_FRAMES - 1, -1, -1, dtype=np.int64), 0
    )
    group = zarr.open_group(zarr_path, mode="r")["data"][f"episode_{episode['episode_id']}"]
    source_rgb = np.asarray(group["rgb_0"].oindex[expected_indices], dtype=np.uint8)
    source_depth = np.asarray(group["depth_0"].oindex[expected_indices])
    if (
        source_depth.shape != (HISTORY_FRAMES, 224, 224, 3)
        or not np.issubdtype(source_depth.dtype, np.floating)
        or not np.isfinite(source_depth).all()
        or np.any(source_depth < 0.0)
        or np.any(source_depth > 0.5)
        or not np.array_equal(source_depth[..., 0], source_depth[..., 1])
        or not np.array_equal(source_depth[..., 0], source_depth[..., 2])
    ):
        raise ValueError("initial canonical depth must be repeated-channel metres in [0,0.5]")
    rgb = _resize_rgb(source_rgb).to(dtype=torch.float32) / 127.5 - 1.0
    depth = _resize_depth_nearest(source_depth[..., 0])
    canvas = pack_rgbd_canvas(rgb, depth)
    history = canvas.permute(1, 2, 3, 0).cpu().numpy().astype(np.float32, copy=True)
    timestamps = np.asarray(
        group["rgb_time_stamps_0"].oindex[expected_indices], dtype=np.float64
    ).reshape(-1)
    robot_timestamps = np.asarray(
        group["robot_time_stamps_0"].oindex[expected_indices], dtype=np.float64
    ).reshape(-1)
    if (
        not np.isfinite(timestamps).all()
        or not np.isfinite(robot_timestamps).all()
        or np.max(np.abs(timestamps - robot_timestamps)) > 0.020 + 1e-12
    ):
        raise ValueError("initial RGBD history timestamps are invalid or exceed 20 ms alignment")
    real_mask = anchor - 2 * np.arange(HISTORY_FRAMES - 1, -1, -1, dtype=np.int64) >= 0
    return history, expected_indices, timestamps, real_mask


def infer(args: argparse.Namespace) -> None:
    import torch

    from cosmos_framework.data.generator.action.datasets.umift_rgbd_dataset import (
        get_umift_rgbd_sft_dataset,
    )
    from cosmos_framework.inference.common.init import get_rank, init_script
    from examples.umift.infer import (
        _move_batch_to_cuda,
        validate_independent_parallelism,
        validate_launch_environment,
    )
    from examples.umift.rgbd_infer import build_rgbd_batch, load_rgbd_model, run_rgbd_prediction

    validate_launch_environment(dict(os.environ))
    init_script()
    manifest_path = (args.root / "prepared" / "manifest.json").resolve()
    manifest_sha = file_sha(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    _validate_manifest(manifest, manifest_path)
    rank = get_rank()
    output_root = args.root / ("rgbd_smoke" if args.max_chunks else "rgbd_inference")
    run_path = _run_evidence_path(output_root, args.phase, rank)
    checkpoint = Path(manifest["base_checkpoint"] if args.phase == "base" else manifest["selected_checkpoint"])
    model, resolved, load_evidence = load_rgbd_model(args.sft_toml, checkpoint)
    validate_independent_parallelism(model.parallel_dims)
    dataset = get_umift_rgbd_sft_dataset(
        manifest["zarr_path"],
        split="history",
        stage="e1",
        history_frames=HISTORY_FRAMES,
        tokenizer_config=resolved.model.config.vlm_config.tokenizer,
        max_action_dim=int(resolved.model.config.max_action_dim),
    )
    methods = ("B0",) if args.phase == "base" else (("E3-A",) if args.max_chunks else MODEL_METHODS[1:])
    results = []
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        for episode_index, episode in enumerate(manifest["episodes"]):
            if episode_index % 4 != rank:
                continue
            for filename, expected in episode["input_files_sha256"].items():
                if file_sha(Path(filename)) != expected:
                    raise ValueError(f"frozen suffix input changed: {filename}")
            all_chunks = episode["chunks"]
            chunks = all_chunks[: args.max_chunks] if args.max_chunks else all_chunks
            if not chunks:
                raise ValueError("requested suffix contains no rollout block")
            count = 1 + sum(int(chunk["steps"]) for chunk in chunks)
            if not args.max_chunks and count != int(episode["frame_count"]):
                raise ValueError("frozen blocks do not cover the complete requested suffix")
            with np.load(episode["actions_path"], allow_pickle=False) as action_archive:
                action_arrays = {key: np.asarray(action_archive[key], dtype=np.float32) for key in ("A", "Z", "S")}
            if any(value.shape != (len(all_chunks), 16, 10) for value in action_arrays.values()):
                raise ValueError("frozen A/Z/S actions must each have shape [blocks,16,10]")
            initial_history, history_indices, history_timestamps, history_real_mask = _initial_history(
                manifest["zarr_path"], episode
            )
            for method in methods:
                action_key = "A" if method in ("B0", "E3-A") else method[-1]
                physical_actions = action_arrays[action_key]
                output_dir = output_root / method
                output_dir.mkdir(parents=True, exist_ok=True)
                label = int(episode["start_percent"])
                stem = output_dir / f"episode_{episode['episode_id']}_start_{label}"
                rgb_path = stem.with_name(stem.name + "_rgb.npy")
                depth_path = stem.with_name(stem.name + "_depth_m.npy")
                metadata_path = stem.with_suffix(".json")
                if rgb_path.exists() or depth_path.exists() or metadata_path.exists():
                    raise FileExistsError(f"refusing to overwrite RGBD rollout: {stem}")
                rgb_output = np.lib.format.open_memmap(
                    rgb_path, mode="w+", dtype=np.float32, shape=(count, 256, 256, 3)
                )
                depth_output = np.lib.format.open_memmap(
                    depth_path, mode="w+", dtype=np.float32, shape=(count, 256, 256)
                )
                model_action_hashes: dict[int, str] = {}

                def predict(history: np.ndarray, chunk: dict[str, Any]) -> np.ndarray:
                    chunk_index = int(chunk["index"])
                    physical_action = physical_actions[chunk_index]
                    expected_hash = chunk[f"{action_key}_physical_action_sha256"]
                    if array_sha(physical_action) != expected_hash:
                        raise ValueError("physical action differs from the frozen A/Z/S fixture")
                    sample = dataset.get_window(
                        int(episode["episode_id"]), 0, physical_action=physical_action
                    )
                    sample["video"].zero_()
                    sample["video"][:, :HISTORY_FRAMES] = torch.from_numpy(history.copy()).permute(3, 0, 1, 2)
                    batch = build_rgbd_batch(sample)
                    batch_video = batch["video"][0]
                    if torch.count_nonzero(batch_video[:, :, HISTORY_FRAMES:]).item() != 0:
                        raise ValueError("future RGB or depth entered the rollout inference batch")
                    observed = batch_video[0, :, :HISTORY_FRAMES].permute(1, 2, 3, 0).cpu().numpy()
                    if not np.array_equal(observed, history):
                        raise ValueError("float rolling RGBD history changed while building the batch")
                    model_action_hashes[chunk_index] = array_sha(sample["action"].cpu().numpy())
                    return run_rgbd_prediction(
                        model,
                        _move_batch_to_cuda(batch),
                        noise_seed=int(chunk["noise_seed"]),
                        num_steps=int(manifest["num_steps"]),
                    )

                chunk_records = rollout(
                    initial_history,
                    chunks,
                    predict,
                    rgb_output,
                    depth_output,
                    initial_history_real_mask=history_real_mask,
                )
                rgb_output.flush()
                depth_output.flush()
                for record in chunk_records:
                    record["physical_action_sha256"] = record[f"{action_key}_physical_action_sha256"]
                    record["padded_model_action_sha256"] = model_action_hashes[int(record["index"])]
                rgb_summary = _array_summary(rgb_output)
                depth_summary = _array_summary(depth_output, depth_m=True)
                del rgb_output, depth_output
                result = {
                    "experiment_id": "E3-Dout",
                    "protocol": manifest["protocol"],
                    "episode_id": episode["episode_id"],
                    "raw_session": episode["raw_session"],
                    "start_percent": label,
                    "initial_selected_frame": episode["initial_selected_frame"],
                    "initial_source_frame": episode["initial_source_frame"],
                    "parent_frame_count": episode["parent_frame_count"],
                    "history_frames": HISTORY_FRAMES,
                    "history_source_indices": history_indices.tolist(),
                    "history_timestamps": history_timestamps.tolist(),
                    "history_real_mask": history_real_mask.tolist(),
                    "history_padding_count": int((~history_real_mask).sum()),
                    "initial_history_sha256": array_sha(initial_history),
                    "method": method,
                    "action_source": action_key,
                    "frame_count": count,
                    "complete_requested_suffix": count == int(episode["frame_count"]),
                    "checkpoint_id": str(checkpoint),
                    "checkpoint_metadata_sha256": file_sha(checkpoint / ".metadata"),
                    "checkpoint_rgbd_pretrained": False,
                    "selected_iteration": manifest["selected_iteration"],
                    "rgb_path": str(rgb_path.resolve()),
                    "rgb_sha256": file_sha(rgb_path),
                    "raw_depth_m_path": str(depth_path.resolve()),
                    "raw_depth_m_sha256": file_sha(depth_path),
                    "rgb_summary": rgb_summary,
                    "raw_depth_m_summary": depth_summary,
                    "manifest_sha256": manifest_sha,
                    "source_manifest_sha256": manifest["source_manifest_sha256"],
                    "selection_sha256": manifest["selection_sha256"],
                    "frame_indices_sha256": file_sha(Path(episode["frame_indices_path"])),
                    "true_pts_sha256": array_sha(
                        np.asarray(episode["true_pts"], dtype=np.float64)
                    ),
                    "chunks": chunk_records,
                    "rank": rank,
                    "initial_h5_rgbd_only": True,
                    "future_gt_refresh_count": 0,
                    "current_frame_display": "true observed RGBD anchor; never decoder reconstruction",
                    "depth_units": "metres",
                    "prediction_depth_validity": "no independent validity prediction; exact zero is display sentinel only",
                    "num_steps": int(manifest["num_steps"]),
                }
                metadata_path.write_text(json.dumps(result, indent=2) + "\n")
                results.append({"method": method, "episode": episode["episode_id"], "start": label, "frames": count})
                print(json.dumps({"rgbd_rollout_done": results[-1]}), flush=True)
    if file_sha(manifest_path) != manifest_sha:
        raise ValueError("prepared manifest changed during inference")
    report = {
        "experiment_id": "E3-Dout",
        "rank": rank,
        "phase": args.phase,
        "results": results,
        "load_evidence": load_evidence,
        "wall_seconds": time.perf_counter() - started,
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "complete": True,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    with run_path.open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    torch.distributed.barrier()
    torch.distributed.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--sft-toml", type=Path, required=True)
    parser.add_argument("--phase", choices=("base", "finetuned"), required=True)
    parser.add_argument("--max-chunks", type=int, default=0)
    args = parser.parse_args()
    if args.max_chunks < 0:
        parser.error("max-chunks must be nonnegative")
    infer(args)


if __name__ == "__main__":
    main()
