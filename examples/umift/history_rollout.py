"""Run E2-H open-loop rollouts from frozen UMI-FT suffix manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Callable

import numpy as np

SUPPORTED_HISTORY_FRAMES = (1, 5, 9, 17)


def array_sha(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def select_initial_history(
    source_rgb: np.ndarray, *, anchor_source_index: int, history_frames: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Select a chronological stride-2 history, repeating source frame zero at the head."""
    source_rgb = np.asarray(source_rgb)
    if history_frames not in SUPPORTED_HISTORY_FRAMES:
        raise ValueError(f"history_frames must be one of {SUPPORTED_HISTORY_FRAMES}")
    if source_rgb.ndim != 4 or source_rgb.shape[-1] != 3 or source_rgb.dtype != np.uint8:
        raise ValueError("source_rgb must be uint8 THWC RGB")
    if not 0 <= anchor_source_index < source_rgb.shape[0]:
        raise ValueError("anchor_source_index is outside source_rgb")
    requested = anchor_source_index - 2 * np.arange(history_frames - 1, -1, -1, dtype=np.int64)
    real_mask = requested >= 0
    source_indices = np.maximum(requested, 0)
    return source_rgb[source_indices].copy(), source_indices, real_mask


def _quantize_generated(frames: np.ndarray) -> tuple[np.ndarray, float]:
    frames = np.asarray(frames)
    if not np.isfinite(frames).all() or frames.min() < 0 or frames.max() > 1:
        raise ValueError("generated frames must be finite RGB in [0,1]")
    quantized = np.rint(frames * 255).astype(np.uint8)
    error = float(np.abs(quantized.astype(np.float32) / 255 - frames).max())
    if error > 0.5 / 255 + 1e-7:
        raise ValueError("unexpected feedback quantization error")
    return quantized, error


def rollout(
    initial_history: np.ndarray,
    chunks: list[dict],
    predict: Callable[[np.ndarray, dict], np.ndarray],
    output: np.ndarray,
) -> list[dict]:
    """Roll forward using only the initial history and recursively generated RGB."""
    initial_history = np.asarray(initial_history)
    if initial_history.dtype != np.uint8 or initial_history.ndim != 4 or initial_history.shape[-1] != 3:
        raise ValueError("initial_history must be uint8 THWC RGB")
    history_frames, height, width, channels = initial_history.shape
    if history_frames not in SUPPORTED_HISTORY_FRAMES:
        raise ValueError("unsupported history length")
    expected_shape = (1 + sum(int(chunk["steps"]) for chunk in chunks), height, width, channels)
    if output.shape != expected_shape:
        raise ValueError(f"output shape {output.shape} does not match {expected_shape}")

    history = initial_history.copy()
    initial_sha = array_sha(history)
    output[0] = history[-1].astype(np.float32) / 255
    records: list[dict] = []
    expected_start = 0
    for expected_index, chunk in enumerate(chunks):
        steps = int(chunk["steps"])
        start = int(chunk["output_start"])
        if int(chunk["index"]) != expected_index or not 1 <= steps <= 16 or start != expected_start:
            raise ValueError("chunk index/output_start must be contiguous with 1..16 retained steps")
        before = history.copy()
        before_sha = array_sha(before)
        started = time.perf_counter()
        prediction = np.asarray(predict(before.copy(), chunk))
        expected_prediction_shape = (history_frames + 16, height, width, channels)
        if prediction.shape != expected_prediction_shape:
            raise ValueError(
                f"prediction shape {prediction.shape} does not match {expected_prediction_shape}"
            )
        if not np.isfinite(prediction).all() or prediction.min() < 0 or prediction.max() > 1:
            raise ValueError("prediction must be finite RGB in [0,1] across all H+16 frames")
        generated = prediction[history_frames : history_frames + steps]
        quantized, quantization_error = _quantize_generated(generated)
        output[start + 1 : start + steps + 1] = generated
        history = np.concatenate((history, quantized), axis=0)[-history_frames:].copy()
        initial_remaining = max(0, history_frames - start)
        history_source = (
            "initial_observed_history"
            if not records
            else "mixed_initial_and_generated_history"
            if initial_remaining
            else "generated_rolling_history"
        )
        record = {
            **chunk,
            "condition_sha256": array_sha(before[-1]),
            "condition_source": "observed_anchor" if not records else "previous_generated_terminal",
            "feedback_sha256": array_sha(history[-1]),
            "decoded_condition_sha256": array_sha(prediction[history_frames - 1]),
            "history_sha256": before_sha,
            "history_source": history_source,
            "initial_observations_remaining": initial_remaining,
            "initial_history_sha256": initial_sha,
            "feedback_history_sha256": array_sha(history),
            "decoded_history_sha256": array_sha(prediction[:history_frames]),
            "feedback_max_quantization_error": quantization_error,
            "seconds": time.perf_counter() - started,
        }
        if records and before_sha != records[-1]["feedback_history_sha256"]:
            raise ValueError("broken generated-history feedback chain")
        records.append(record)
        expected_start += steps
    return records


def _initial_history_from_manifest(manifest: dict, episode: dict, history_frames: int):
    import zarr

    from cosmos_framework.data.generator.action.datasets.umift_history_dataset import _resize_rgb

    indices_file = np.load(episode["frame_indices_path"], allow_pickle=False)
    source_indices = np.asarray(indices_file["source_indices"], dtype=np.int64)
    true_pts = np.asarray(indices_file["timestamps"], dtype=np.float64)
    if source_indices.ndim != 1 or true_pts.shape != source_indices.shape or len(source_indices) < 1:
        raise ValueError("frozen frame index/PTS arrays are malformed")
    if np.any(np.diff(source_indices) <= 0) or not np.isfinite(true_pts).all() or np.any(np.diff(true_pts) <= 0):
        raise ValueError("frozen source indices and true PTS must be strictly increasing")
    anchor_source_index = int(source_indices[0])
    group = zarr.open_group(manifest["zarr_path"], mode="r")["data"][f"episode_{episode['episode_id']}"]
    source_length = int(group["rgb_0"].shape[0])
    if not 0 <= anchor_source_index < source_length:
        raise ValueError("frozen suffix anchor is outside the source episode")
    requested = anchor_source_index - 2 * np.arange(history_frames - 1, -1, -1, dtype=np.int64)
    real_mask = requested >= 0
    history_indices = np.maximum(requested, 0)
    raw_history = np.asarray(group["rgb_0"].oindex[history_indices], dtype=np.uint8)
    history = _resize_rgb(raw_history).permute(1, 2, 3, 0).numpy()
    raw_timestamps = np.asarray(group["rgb_time_stamps_0"].oindex[history_indices], dtype=np.float64).reshape(-1)
    indices_file.close()
    return history, history_indices, raw_timestamps, real_mask, true_pts


def infer(args: argparse.Namespace) -> None:
    import torch

    from cosmos_framework.data.generator.action.datasets.umift_history_dataset import (
        get_umift_history_sft_dataset,
    )
    from cosmos_framework.inference.common.init import get_rank, init_script
    from examples.umift.history_infer import build_history_batch, load_history_model, run_history_prediction
    from examples.umift.infer import (
        _move_batch_to_cuda,
        validate_independent_parallelism,
        validate_launch_environment,
    )

    validate_launch_environment(dict(os.environ))
    init_script()
    manifest_path = args.root / "prepared/manifest.json"
    manifest_sha = file_sha(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("experiment_id") != "E2-H" or int(manifest.get("history_frames", -1)) != args.history_frames:
        raise ValueError("prepared manifest must match experiment_id=E2-H and CLI history_frames")
    checkpoint = Path(
        manifest["base_checkpoint"] if args.phase == "base" else manifest["selected_checkpoint"]
    )
    model, resolved, load_evidence = load_history_model(
        args.sft_toml, checkpoint, args.history_frames, independent_windows=True
    )
    validate_independent_parallelism(model.parallel_dims)
    dataset = get_umift_history_sft_dataset(
        manifest["zarr_path"],
        split="history",
        stage="e1",
        history_frames=args.history_frames,
        tokenizer_config=resolved.model.config.vlm_config.tokenizer,
        max_action_dim=int(resolved.model.config.max_action_dim),
    )

    rank = get_rank()
    output_root = args.root / ("history_smoke" if args.max_chunks else "history_inference") / f"H{args.history_frames}"
    results = []
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        for record_index, episode in enumerate(manifest["episodes"]):
            if record_index % 4 != rank:
                continue
            input_hashes = episode.get("input_files_sha256")
            if not isinstance(input_hashes, dict) or not input_hashes:
                raise ValueError("each suffix requires non-empty input_files_sha256")
            for filename, digest in input_hashes.items():
                if file_sha(Path(filename)) != digest:
                    raise ValueError(f"prepared input hash differs: {filename}")
            truth = np.load(episode["truth_path"], mmap_mode="r", allow_pickle=False)
            initial_history, history_indices, history_timestamps, real_mask, true_pts = (
                _initial_history_from_manifest(manifest, episode, args.history_frames)
            )
            if not np.array_equal(initial_history[-1], np.asarray(truth[0])):
                raise ValueError("initial history terminal frame differs from frozen suffix truth[0]")
            all_chunks = episode["chunks"]
            if any(int(chunk.get("index", -1)) != index for index, chunk in enumerate(all_chunks)):
                raise ValueError("frozen chunk indexes must be contiguous from zero")
            if any("noise_seed" not in chunk for chunk in all_chunks):
                raise ValueError("every frozen chunk requires noise_seed")
            chunks = all_chunks[: args.max_chunks] if args.max_chunks else all_chunks
            if not chunks:
                raise ValueError("suffix has no rollout chunks")
            count = 1 + sum(int(chunk["steps"]) for chunk in chunks)
            if not args.max_chunks and count != int(episode["frame_count"]):
                raise ValueError("frozen chunks do not cover the complete suffix")
            if len(true_pts) < count:
                raise ValueError("true PTS do not cover the requested rollout")
            actions_file = np.load(episode["actions_path"], allow_pickle=False)
            methods = ("B0",) if args.phase == "base" else (("E2-A",) if args.max_chunks else ("E2-A", "E2-Z", "E2-S"))
            for method in methods:
                action_key = "A" if method in ("B0", "E2-A") else method[-1]
                physical_actions = actions_file[action_key]
                expected_action_shape = (len(all_chunks), 16, 10)
                if physical_actions.shape != expected_action_shape:
                    raise ValueError(f"frozen {action_key} actions must have shape {expected_action_shape}")
                start_label = str(episode.get("start_percent", episode.get("initial_selected_frame", 0)))
                output_dir = output_root / method
                output_dir.mkdir(parents=True, exist_ok=True)
                output_path = output_dir / f"episode_{episode['episode_id']}_start_{start_label}.npy"
                if output_path.exists():
                    raise FileExistsError(f"refusing to overwrite rollout: {output_path}")
                output = np.lib.format.open_memmap(
                    output_path, mode="w+", dtype=np.float32, shape=(count, 256, 256, 3)
                )
                model_action_hashes: dict[int, str] = {}

                def predict(history: np.ndarray, chunk: dict) -> np.ndarray:
                    chunk_index = int(chunk["index"])
                    physical_action = physical_actions[chunk_index]
                    hash_key = f"{action_key}_physical_action_sha256"
                    if hash_key not in chunk or array_sha(physical_action) != chunk[hash_key]:
                        raise ValueError("physical action hash differs from frozen manifest")
                    sample = dataset.get_window(
                        int(episode["episode_id"]), 0, physical_action=physical_action
                    )
                    sample["video"].zero_()
                    sample["video"][:, : args.history_frames] = torch.from_numpy(history.copy()).permute(3, 0, 1, 2)
                    batch = build_history_batch(sample)
                    batch_video = batch["video"][0]
                    if torch.count_nonzero(batch_video[:, args.history_frames :]).item() != 0:
                        raise ValueError("future truth entered the history inference batch")
                    observed = batch_video[:, : args.history_frames].permute(1, 2, 3, 0).cpu().numpy()
                    if not np.array_equal(observed, history):
                        raise ValueError("rolling history changed while building the batch")
                    model_action_hashes[chunk_index] = array_sha(sample["action"].cpu().numpy())
                    noise_seed = int(chunk["noise_seed"])
                    return run_history_prediction(
                        model,
                        _move_batch_to_cuda(batch),
                        history_frames=args.history_frames,
                        noise_seed=noise_seed,
                        num_steps=int(manifest.get("num_steps", 30)),
                    )

                chunk_records = rollout(initial_history, chunks, predict, output)
                output.flush()
                del output
                for chunk_record in chunk_records:
                    chunk_record["padded_model_action_sha256"] = model_action_hashes[int(chunk_record["index"])]
                result = {
                    "protocol": "e2-history-open-loop-v1",
                    "episode_id": episode["episode_id"],
                    "raw_session": episode.get("raw_session"),
                    "start_percent": episode.get("start_percent"),
                    "initial_selected_frame": episode.get("initial_selected_frame"),
                    "initial_source_frame": int(history_indices[-1]),
                    "history_frames": args.history_frames,
                    "history_source_indices": history_indices.tolist(),
                    "history_timestamps": history_timestamps.tolist(),
                    "history_real_mask": real_mask.tolist(),
                    "history_padding_count": int((~real_mask).sum()),
                    "initial_history_sha256": array_sha(initial_history),
                    "method": method,
                    "frame_count": count,
                    "checkpoint_id": str(checkpoint),
                    "prediction_path": str(output_path),
                    "prediction_sha256": file_sha(output_path),
                    "frame_indices_path": episode["frame_indices_path"],
                    "true_pts": true_pts[:count].tolist(),
                    "manifest_sha256": manifest_sha,
                    "chunks": chunk_records,
                    "rank": rank,
                    "initial_history_only": True,
                    "num_steps": int(manifest.get("num_steps", 30)),
                    "sampling_seed": int(manifest.get("sampling_seed", 0)),
                    "complete_requested_suffix": count == int(episode["frame_count"]),
                }
                output_path.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
                results.append({"method": method, "episode": episode["episode_id"], "start": start_label})
            actions_file.close()
    if file_sha(manifest_path) != manifest_sha:
        raise ValueError("prepared manifest changed during inference")
    report = {
        "rank": rank,
        "phase": args.phase,
        "history_frames": args.history_frames,
        "results": results,
        "load_evidence": load_evidence,
        "wall_seconds": time.perf_counter() - started,
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "complete": True,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / f"run_{args.phase}_rank{rank}.json").write_text(json.dumps(report, indent=2) + "\n")
    torch.distributed.barrier()
    torch.distributed.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--history-frames", type=int, choices=SUPPORTED_HISTORY_FRAMES, required=True)
    parser.add_argument("--sft-toml", type=Path, required=True)
    parser.add_argument("--phase", choices=("base", "finetuned"), required=True)
    parser.add_argument("--max-chunks", type=int, default=0)
    args = parser.parse_args()
    if args.max_chunks < 0:
        parser.error("max-chunks must be nonnegative")
    infer(args)


if __name__ == "__main__":
    main()
