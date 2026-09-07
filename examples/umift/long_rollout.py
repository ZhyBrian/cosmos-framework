"""Open-loop, full-episode Edge FD rollout using the frozen E1 weights.

Only I0 is observed. Each later chunk conditions on the previous generated end
frame, rounded to uint8 as in the official image-based action FD example.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Callable

import numpy as np

from examples.umift.protocol import derive_noise_seed

EPISODES = (13, 43, 49)
IDENTITY = np.array([0, 0, 0, 1, 0, 0, 0, 1, 0, 0], dtype=np.float32)
BASE_CHECKPOINT = "/data/cosmos_models/Cosmos3-Edge-DCP-a9d944e/model"
BEST_CHECKPOINT = "/data/cosmos_runs/e1_main/cosmos3_action_fd_umift/action_sft/action_fd_umift_edge_e1/checkpoints/iter_000001000/model"


def array_sha(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def timestamp_statistics(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    intervals = np.diff(values)
    if len(values) < 2 or not np.isfinite(values).all() or np.any(intervals <= 0):
        raise ValueError("timestamps must be finite and strictly increasing")
    return {"frame_count": len(values), "first_timestamp": float(values[0]),
            "last_timestamp": float(values[-1]), "span_seconds": float(values[-1] - values[0]),
            "mean_rate_hz": float((len(values) - 1) / (values[-1] - values[0])),
            "inverse_median_interval_hz": float(1 / np.median(intervals)),
            "interval_seconds_min_p05_p50_p95_max": np.quantile(intervals, [0, .05, .5, .95, 1]).tolist()}


def chunk_plan(source_length: int, episode_id: int) -> list[dict]:
    frame_count = (source_length + 1) // 2
    if frame_count < 17:
        raise ValueError("episode needs at least one complete 17-frame source window")
    result = []
    for index, start in enumerate(range(0, frame_count - 1, 16)):
        steps = min(16, frame_count - 1 - start)
        anchor = min(start, frame_count - 17)
        label = f"episode_{episode_id}:s={2 * start}"
        result.append({"index": index, "output_start": start, "steps": steps,
                       "padding_steps": 16 - steps, "anchor_start": 2 * anchor,
                       "anchor_action_offset": start - anchor,
                       "chunk_id": label, "noise_seed": derive_noise_seed(label, 0)})
    return result


def padded_actions(anchor_actions: np.ndarray, offset: int, steps: int) -> np.ndarray:
    if anchor_actions.shape != (16, 10) or not 0 < steps <= 16 or not 0 <= offset <= 16 - steps:
        raise ValueError("invalid tail action slice")
    result = np.tile(IDENTITY, (16, 1))
    result[:steps] = anchor_actions[offset:offset + steps]
    return result


def quantize_condition(frame: np.ndarray) -> np.ndarray:
    frame = np.asarray(frame)
    if frame.shape != (256, 256, 3) or not np.isfinite(frame).all() or frame.min() < 0 or frame.max() > 1:
        raise ValueError("generated condition must be finite HWC RGB in [0,1]")
    return np.rint(frame * 255).astype(np.uint8)


def feedback_sample(sample: dict, condition: np.ndarray) -> dict:
    import torch

    if condition.shape != (256, 256, 3) or condition.dtype != np.uint8:
        raise ValueError("condition must be uint8 HWC")
    result = copy.copy(sample)
    result["video"] = torch.zeros_like(sample["video"], dtype=torch.uint8)
    result["video"][:, 0] = torch.from_numpy(condition.copy()).permute(2, 0, 1)
    return result


def rollout(first_frame: np.ndarray, chunks: list[dict], predict: Callable,
            output: np.ndarray) -> list[dict]:
    """Pure chain contract: predictor receives no GT frames except the first."""
    if first_frame.dtype != np.uint8 or first_frame.shape != (256, 256, 3):
        raise ValueError("invalid observed I0")
    if output.shape != (1 + sum(c["steps"] for c in chunks), 256, 256, 3):
        raise ValueError("output length differs from retained chunk steps")
    condition = first_frame.copy()
    output[0] = first_frame.astype(np.float32) / 255
    records = []
    for chunk in chunks:
        before = condition.copy()
        started = time.perf_counter()
        prediction = predict(condition.copy(), chunk)
        if prediction.shape != (17, 256, 256, 3) or not np.isfinite(prediction).all():
            raise ValueError("invalid generated chunk")
        if prediction.min() < 0 or prediction.max() > 1:
            raise ValueError("generated chunk outside [0,1]")
        start, steps = chunk["output_start"], chunk["steps"]
        output[start + 1:start + steps + 1] = prediction[1:steps + 1]
        condition = quantize_condition(prediction[steps])
        error = float(np.abs(condition.astype(np.float32) / 255 - prediction[steps]).max())
        if error > 0.5 / 255 + 1e-7:
            raise ValueError("unexpected feedback quantization error")
        record = {**chunk, "condition_sha256": array_sha(before),
                  "condition_source": "observed_I0" if chunk["index"] == 0 else "previous_generated_terminal",
                  "feedback_sha256": array_sha(condition),
                  "decoded_condition_sha256": array_sha(prediction[0]),
                  "feedback_max_quantization_error": error,
                  "seconds": time.perf_counter() - started}
        if records and record["condition_sha256"] != records[-1]["feedback_sha256"]:
            raise ValueError("broken generated-feedback chain")
        records.append(record)
        print(json.dumps({"chunk_done": chunk["index"], "end_frame": start + steps,
                          "seconds": record["seconds"]}), flush=True)
    return records


def prepare(args: argparse.Namespace) -> None:
    import zarr
    from cosmos_framework.data.generator.action.datasets.umift_zarr_dataset import UMIFTZarrIterableDataset

    destination = args.root / "prepared"
    destination.mkdir(parents=True, exist_ok=False)
    dataset = UMIFTZarrIterableDataset(str(args.zarr), split="history", stage="e1")
    zroot = zarr.open_group(str(args.zarr), mode="r")
    pairs_path = args.evaluation_root / "frozen_v2/history_S_pairs.json"
    pairs = json.loads(pairs_path.read_text())
    selection_path = args.evaluation_root / "metrics/checkpoint_selection.json"
    selection = json.loads(selection_path.read_text())["selected"]
    if selection["iteration"] != 1000 or selection["checkpoint"] != BEST_CHECKPOINT:
        raise ValueError("unexpected best checkpoint identity")
    manifest = {"protocol": "e1-long-open-loop-v1", "fps": 15, "sampling_seed": 0,
                "num_steps": 30, "guidance": 1.0, "initial_truth_only": True,
                "tail_rule": "remaining real actions then physical identity padding; retain only real steps",
                "S_rule": "original frozen chunk pairing; tail continues last donor window by 32 raw frames",
                "base_checkpoint": BASE_CHECKPOINT, "selected_checkpoint": BEST_CHECKPOINT,
                "zarr_path": str(args.zarr), "source_pair_sha256": file_sha(pairs_path),
                "checkpoint_selection_sha256": file_sha(selection_path), "episodes": []}
    for episode_id in EPISODES:
        group = zroot["data"][f"episode_{episode_id}"]
        length = int(group["rgb_0"].shape[0])
        if any(int(group[key].shape[0]) != length for key in
               ("ts_pose_fb_0", "rgb_time_stamps_0", "robot_time_stamps_0")):
            raise ValueError("RGB, pose and timestamp lengths differ")
        indices = np.arange(0, length, 2, dtype=np.int64)
        raw_timestamps = np.asarray(group["rgb_time_stamps_0"][:], dtype=np.float64).reshape(-1)
        timestamps = raw_timestamps[indices]
        robot_times = np.asarray(group["robot_time_stamps_0"].oindex[indices]).reshape(-1)
        if (not np.isfinite(timestamps).all() or np.any(np.diff(timestamps) <= 0)
                or not np.isfinite(robot_times).all() or np.max(np.abs(timestamps - robot_times)) > .020 + 1e-12):
            raise ValueError("invalid timestamp alignment")
        plan = chunk_plan(length, episode_id)
        truth_path = destination / f"episode_{episode_id}_truth.npy"
        truth = np.lib.format.open_memmap(truth_path, mode="w+", dtype=np.uint8,
                                         shape=(len(indices), 256, 256, 3))
        action_rows = {key: [] for key in ("A", "Z", "S")}
        for chunk in plan:
            sample = dataset.get_window(episode_id, chunk["anchor_start"])
            offset, steps, start = chunk["anchor_action_offset"], chunk["steps"], chunk["output_start"]
            video = sample["video"].permute(1, 2, 3, 0).numpy()[offset:offset + steps + 1]
            if start == 0:
                truth[0] = video[0]
            elif not np.array_equal(truth[start], video[0]):
                raise ValueError("truth overlap disagrees")
            truth[start + 1:start + steps + 1] = video[1:]
            correct = padded_actions(sample["physical_action"].numpy(), offset, steps)
            zero = np.tile(IDENTITY, (16, 1))
            if steps == 16:
                donor_id = pairs[chunk["chunk_id"]]
                chunk["donor_basis"] = "frozen_full_window_pair"
            else:
                previous = pairs[f"episode_{episode_id}:s={2 * (start - 16)}"]
                donor_ep, donor_start = previous.split(":s=")
                donor_id = f"{donor_ep}:s={int(donor_start) + 32}"
                chunk["donor_basis"] = "continuation_of_last_frozen_donor"
                chunk["previous_frozen_donor"] = previous
            donor_ep, donor_start = donor_id.split(":s=")
            if donor_id == chunk["chunk_id"]:
                raise ValueError("S donor cannot equal target chunk")
            donor = dataset.get_window(int(donor_ep.removeprefix("episode_")), int(donor_start))
            shuffled = padded_actions(donor["physical_action"].numpy(), 0, steps)
            chunk["donor_window_id"] = donor_id
            for key, actions in (("A", correct), ("Z", zero), ("S", shuffled)):
                action_rows[key].append(actions)
                chunk[f"{key}_physical_action_sha256"] = array_sha(actions)
        truth.flush()
        del truth
        actions_path = destination / f"episode_{episode_id}_actions.npz"
        np.savez(actions_path, **{key: np.stack(rows) for key, rows in action_rows.items()})
        times_path = destination / f"episode_{episode_id}_indices.npz"
        np.savez(times_path, source_indices=indices, timestamps=timestamps, robot_timestamps=robot_times)
        source_id = str(group.attrs["src"])
        record = {"episode_id": episode_id, "source_id": source_id, "raw_session": source_id.split("#")[0],
                  "source_length": length, "frame_count": len(indices), "fps": 15,
                  "omitted_off_grid_final_raw_frames": length - 1 - int(indices[-1]),
                  "actual_timestamp_span_seconds": float(timestamps[-1] - timestamps[0]),
                  "raw_timestamp_statistics": timestamp_statistics(raw_timestamps),
                  "sampled_timestamp_statistics": timestamp_statistics(timestamps),
                  "omitted_off_grid_final_seconds": float(raw_timestamps[-1] - timestamps[-1]),
                  "nominal_span_seconds": (len(indices) - 1) / 15,
                  "truth_path": str(truth_path), "actions_path": str(actions_path),
                  "frame_indices_path": str(times_path), "chunks": plan,
                  "input_files_sha256": {str(p): file_sha(p) for p in (truth_path, actions_path, times_path)}}
        manifest["episodes"].append(record)
        print(json.dumps({"prepared_episode": episode_id, "frames": len(indices), "chunks": len(plan),
                          "tail_steps": plan[-1]["steps"]}), flush=True)
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def infer(args: argparse.Namespace) -> None:
    import torch
    from cosmos_framework.data.generator.action.datasets.umift_zarr_dataset import get_umift_zarr_sft_dataset
    from cosmos_framework.inference.common.init import get_rank, init_script
    from examples.umift.infer import (
        _move_batch_to_cuda, build_singleton_batch, load_edge_fd_model, run_forward_dynamics,
        validate_independent_parallelism, validate_launch_environment,
    )

    validate_launch_environment(dict(os.environ))
    init_script()
    manifest_path = args.root / "prepared/manifest.json"
    manifest_sha = file_sha(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    checkpoint = Path(manifest["base_checkpoint"] if args.phase == "base" else manifest["selected_checkpoint"])
    model, resolved, load_evidence = load_edge_fd_model(args.sft_toml, checkpoint, independent_windows=True)
    validate_independent_parallelism(model.parallel_dims)
    dataset = get_umift_zarr_sft_dataset(manifest["zarr_path"], split="history", stage="e1",
                                        tokenizer_config=resolved.model.config.vlm_config.tokenizer,
                                        max_action_dim=int(resolved.model.config.max_action_dim))
    rank = get_rank()
    methods = ("B0",) if args.phase == "base" else (("E1-A",) if args.max_chunks else ("E1-A", "E1-Z", "E1-S"))
    output_root = args.root / ("smoke" if args.max_chunks else "inference")
    results = []
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    with torch.inference_mode():
        for ep_index, episode in enumerate(manifest["episodes"]):
            if ep_index % 4 != rank:
                continue
            for filename, digest in episode["input_files_sha256"].items():
                if file_sha(Path(filename)) != digest:
                    raise ValueError(f"prepared input hash differs: {filename}")
            if 1 + sum(c["steps"] for c in episode["chunks"]) != episode["frame_count"]:
                raise ValueError("prepared chunks do not cover the complete episode")
            truth = np.load(episode["truth_path"], mmap_mode="r", allow_pickle=False)
            actions_file = np.load(episode["actions_path"], allow_pickle=False)
            if any(actions_file[key].shape != (len(episode["chunks"]), 16, 10) for key in ("A", "Z", "S")):
                raise ValueError("prepared action dimensions differ from the chunk plan")
            plan = episode["chunks"][:args.max_chunks] if args.max_chunks else episode["chunks"]
            count = 1 + sum(chunk["steps"] for chunk in plan)
            for method in methods:
                output_dir = output_root / method
                output_dir.mkdir(parents=True, exist_ok=True)
                output_path = output_dir / f"episode_{episode['episode_id']}.npy"
                if output_path.exists():
                    raise FileExistsError(f"refusing to overwrite rollout: {output_path}")
                output = np.lib.format.open_memmap(output_path, mode="w+", dtype=np.float32,
                                                  shape=(count, 256, 256, 3))
                key = "A" if method in ("B0", "E1-A") else method[-1]
                physical = actions_file[key]
                normalizer_hashes = {}

                def predict(condition: np.ndarray, chunk: dict) -> np.ndarray:
                    action = physical[chunk["index"]]
                    if array_sha(action) != chunk[f"{key}_physical_action_sha256"]:
                        raise ValueError("physical action hash differs")
                    sample = dataset.get_window(episode["episode_id"], 0, physical_action=action)
                    sample = feedback_sample(sample, condition)
                    batch = build_singleton_batch(sample)
                    if torch.count_nonzero(batch["video"][0][:, 1:]).item() != 0:
                        raise ValueError("future truth entered the inference batch")
                    if not np.array_equal(batch["video"][0][:, 0].permute(1, 2, 0).numpy(), condition):
                        raise ValueError("condition changed in the batch")
                    normalizer_hashes[chunk["index"]] = array_sha(sample["action"].numpy())
                    return run_forward_dynamics(model, _move_batch_to_cuda(batch),
                                                noise_seed=chunk["noise_seed"], num_steps=30)

                records = rollout(np.asarray(truth[0]), plan, predict, output)
                output.flush()
                del output
                for record in records:
                    record["padded_model_action_sha256"] = normalizer_hashes[record["index"]]
                result = {"episode_id": episode["episode_id"], "raw_session": episode["raw_session"],
                          "method": method, "frame_count": count, "expected_total_frames": episode["frame_count"],
                          "complete_episode": count == episode["frame_count"], "checkpoint_id": str(checkpoint),
                          "prediction_path": str(output_path), "prediction_sha256": file_sha(output_path),
                          "manifest_sha256": manifest_sha, "chunks": records, "rank": rank,
                          "initial_truth_only": True, "num_steps": 30, "sampling_seed": 0,
                          "first_frame_matches_truth": True}
                if not args.max_chunks and not result["complete_episode"]:
                    raise ValueError("full inference did not cover the complete episode")
                (output_path.with_suffix(".json")).write_text(json.dumps(result, indent=2) + "\n")
                results.append({"method": method, "episode": episode["episode_id"], "frames": count})
                print(json.dumps({"rollout_done": results[-1]}), flush=True)
            actions_file.close()
    if file_sha(manifest_path) != manifest_sha:
        raise ValueError("prepared manifest changed during inference")
    report = {"rank": rank, "phase": args.phase, "results": results, "load_evidence": load_evidence,
              "wall_seconds": time.perf_counter() - started, "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
              "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
              "peak_reserved_bytes": torch.cuda.max_memory_reserved(), "complete": True}
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / f"run_{args.phase}_rank{rank}.json").write_text(json.dumps(report, indent=2) + "\n")
    torch.distributed.barrier()
    torch.distributed.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prep = subparsers.add_parser("prepare")
    prep.add_argument("--root", type=Path, required=True)
    prep.add_argument("--zarr", type=Path, required=True)
    prep.add_argument("--evaluation-root", type=Path, required=True)
    run = subparsers.add_parser("infer")
    run.add_argument("--root", type=Path, required=True)
    run.add_argument("--sft-toml", type=Path, required=True)
    run.add_argument("--phase", choices=("base", "finetuned"), required=True)
    run.add_argument("--max-chunks", type=int, default=0, help="positive: A-only smoke in a separate output tree")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args)
    else:
        if args.max_chunks < 0:
            parser.error("max-chunks must be nonnegative")
        infer(args)


if __name__ == "__main__":
    main()
