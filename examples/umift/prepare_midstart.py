"""Prepare 33%/67% observed-frame suffixes from immutable P8 inputs.

Physical A/Z/S action streams are sliced before regrouping. No model is loaded,
and no parent output, source Zarr, normalization or calibration is modified.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from examples.umift.long_rollout import (
    EPISODES, IDENTITY, array_sha, file_sha, timestamp_statistics,
)
from examples.umift.protocol import derive_noise_seed


def suffix_actions(episode: dict, actions: dict, percent: int) -> tuple[int, list[dict], dict]:
    """One action at global index s moves observed frame s to frame s+1."""
    if percent not in (0, 33, 67):
        raise ValueError("start percent must be 0 (regression), 33 or 67")
    count = episode["frame_count"]
    start = percent * (count - 1) // 100
    parent_chunks = episode["chunks"]
    if 1 + sum(c["steps"] for c in parent_chunks) != count:
        raise ValueError("parent chunk coverage differs")
    streams = {}
    for key in ("A", "Z", "S"):
        value = actions[key]
        if value.shape != (len(parent_chunks), 16, 10) or value.dtype != np.float32:
            raise ValueError("parent physical actions must be float32 K,16,10")
        for i, chunk in enumerate(parent_chunks):
            if chunk["index"] != i or chunk["output_start"] != 16 * i or not 0 < chunk["steps"] <= 16:
                raise ValueError("parent chunk layout differs")
            if array_sha(value[i]) != chunk[f"{key}_physical_action_sha256"]:
                raise ValueError("parent physical action hash differs")
        streams[key] = np.concatenate([value[i, :c["steps"]] for i, c in enumerate(parent_chunks)])
        if not np.isfinite(streams[key]).all():
            raise ValueError("nonfinite parent actions")
    if not np.array_equal(streams["Z"], np.tile(IDENTITY, (count - 1, 1))):
        raise ValueError("stationary stream must be physical identity")
    plan, rows = [], {key: [] for key in streams}
    for i, local_start in enumerate(range(0, count - start - 1, 16)):
        global_start = start + local_start
        steps = min(16, count - 1 - global_start)
        noise_label = f"episode_{episode['episode_id']}:s={2 * global_start}"
        segments = []
        for parent in parent_chunks:
            left = max(global_start, parent["output_start"])
            right = min(global_start + steps, parent["output_start"] + parent["steps"])
            if left < right:
                segments.append({"parent_chunk_index": parent["index"],
                                 "parent_chunk_id": parent["chunk_id"],
                                 "donor_window_id": parent["donor_window_id"],
                                 "donor_action_offset": left - parent["output_start"],
                                 "global_action_start": left, "steps": right - left})
        if sum(s["steps"] for s in segments) != steps:
            raise ValueError("S provenance does not cover new chunk")
        chunk = {"index": i, "output_start": local_start, "steps": steps,
                 "padding_steps": 16 - steps, "global_selected_start": global_start,
                 "global_source_start": 2 * global_start,
                 "chunk_id": f"episode_{episode['episode_id']}:p{percent}:s={2 * global_start}",
                 "noise_label": noise_label, "noise_seed": derive_noise_seed(noise_label, 0),
                 "S_source_segments": segments,
                 "donor_basis": "lossless_suffix_of_frozen_P8_S_stream"}
        for key, stream in streams.items():
            padded = np.tile(IDENTITY, (16, 1))
            padded[:steps] = stream[global_start:global_start + steps]
            rows[key].append(padded)
            chunk[f"{key}_physical_action_sha256"] = array_sha(padded)
        plan.append(chunk)
    return start, plan, {key: np.stack(value) for key, value in rows.items()}


def prepare(source_root: Path, root: Path, percent: int) -> dict:
    import zarr
    from cosmos_framework.data.generator.action.datasets.umift_zarr_dataset import (
        UMIFTZarrIterableDataset, _framewise_actions,
    )

    parent_path = source_root.resolve() / "prepared/manifest.json"
    parent_hash = file_sha(parent_path)
    parent = json.loads(parent_path.read_text())
    if parent["protocol"] != "e1-long-open-loop-v1" or tuple(e["episode_id"] for e in parent["episodes"]) != EPISODES:
        raise ValueError("expected original P8 three-episode preparation")
    destination = root.resolve() / "prepared"
    if destination.exists():
        raise FileExistsError(destination)
    dataset = UMIFTZarrIterableDataset(parent["zarr_path"], split="history", stage="e1")
    zroot = zarr.open_group(parent["zarr_path"], mode="r")
    manifest = {**parent, "protocol": "e1-midstart-open-loop-v1", "start_percent": percent,
                "start_rule": "floor(percent*(parent selected frame count-1)/100); suffix to last selected frame",
                "S_rule": "slice frozen P8 physical S stream at same global action index; regroup without resampling",
                "noise_rule": "unchanged seed derivation using original episode global raw chunk start and seed 0",
                "parent_manifest_path": str(parent_path), "parent_manifest_sha256": parent_hash,
                "episodes": []}
    destination.mkdir(parents=True, exist_ok=False)
    protected = {str(parent_path): parent_hash}
    for ep in parent["episodes"]:
        for filename, digest in ep["input_files_sha256"].items():
            if file_sha(Path(filename)) != digest:
                raise ValueError(f"parent input changed: {filename}")
            protected[filename] = digest
        with np.load(ep["actions_path"], allow_pickle=False) as archive:
            start, plan, rows = suffix_actions(ep, archive, percent)
        group = zroot["data"][f"episode_{ep['episode_id']}"]
        raw_times = np.asarray(group["rgb_time_stamps_0"][:], dtype=np.float64).reshape(-1)
        with np.load(ep["frame_indices_path"], allow_pickle=False) as archive:
            times = {key: archive[key][start:].copy() for key in ("source_indices", "timestamps", "robot_timestamps")}
        indices = times["source_indices"]
        if not np.array_equal(indices, np.arange(2 * start, ep["source_length"], 2)):
            raise ValueError("suffix source indices differ from original stride 2 grid")
        if not np.array_equal(raw_times[indices], times["timestamps"]):
            raise ValueError("suffix RGB time differs from actual Zarr")
        robot = np.asarray(group["robot_time_stamps_0"].oindex[indices]).reshape(-1)
        if not np.array_equal(robot, times["robot_timestamps"]):
            raise ValueError("suffix robot time differs from actual Zarr")
        poses = np.asarray(group["ts_pose_fb_0"].oindex[indices])
        direct_actions = _framewise_actions(poses)
        flat_correct = np.concatenate([rows["A"][i, :c["steps"]] for i, c in enumerate(plan)])
        if not np.array_equal(direct_actions, flat_correct):
            raise ValueError("suffix A does not match direct Zarr flange-delta conversion")
        truth_parent = np.load(ep["truth_path"], mmap_mode="r", allow_pickle=False)
        count = ep["frame_count"] - start
        if truth_parent.shape != (ep["frame_count"], 256, 256, 3) or truth_parent.dtype != np.uint8:
            raise ValueError("invalid parent truth")
        # Independently re-read the new initial RGB with the unchanged adapter.
        direct_first = dataset.get_window(ep["episode_id"], 2 * start)["video"][:, 0].permute(1, 2, 0).numpy()
        if not np.array_equal(truth_parent[start], direct_first):
            raise ValueError("new condition differs from actual source RGB")
        truth_path = destination / f"episode_{ep['episode_id']}_truth.npy"
        truth = np.lib.format.open_memmap(truth_path, mode="w+", dtype=np.uint8, shape=(count, 256, 256, 3))
        for offset in range(0, count, 64):
            truth[offset:offset + 64] = truth_parent[start + offset:start + min(offset + 64, count)]
        truth.flush()
        del truth, truth_parent
        actions_path = destination / f"episode_{ep['episode_id']}_actions.npz"
        times_path = destination / f"episode_{ep['episode_id']}_indices.npz"
        np.savez(actions_path, **rows)
        np.savez(times_path, **times)
        start_seconds = float(times["timestamps"][0] - raw_times[0])
        stats = timestamp_statistics(times["timestamps"])
        record = {**ep, "start_percent": percent, "initial_selected_frame": start,
                  "initial_source_frame": int(indices[0]), "parent_frame_count": ep["frame_count"],
                  "achieved_frame_fraction": start / (ep["frame_count"] - 1),
                  "original_episode_first_timestamp": float(raw_times[0]),
                  "initial_episode_elapsed_seconds": start_seconds,
                  "achieved_time_fraction": start_seconds / ep["actual_timestamp_span_seconds"],
                  "frame_count": count, "actual_timestamp_span_seconds": stats["span_seconds"],
                  "sampled_timestamp_statistics": stats, "nominal_span_seconds": (count - 1) / 15,
                  "truth_path": str(truth_path), "actions_path": str(actions_path),
                  "frame_indices_path": str(times_path), "chunks": plan,
                  "parent_input_files_sha256": ep["input_files_sha256"],
                  "verification": {"direct_zarr_A_exact": True, "direct_initial_rgb_exact": True,
                                   "direct_zarr_timestamps_exact": True},
                  "input_files_sha256": {str(p): file_sha(p) for p in (truth_path, actions_path, times_path)}}
        manifest["episodes"].append(record)
        print(json.dumps({"episode": ep["episode_id"], "percent": percent, "start": start,
                          "start_seconds": start_seconds, "span_seconds": stats["span_seconds"],
                          "frames": count, "chunks": len(plan), "tail_steps": plan[-1]["steps"]}), flush=True)
    if any(file_sha(Path(p)) != digest for p, digest in protected.items()):
        raise ValueError("parent inputs changed during preparation")
    manifest["parent_inputs_unchanged"] = True
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--start-percent", type=int, choices=(33, 67), required=True)
    args = parser.parse_args()
    prepare(args.source_root, args.root, args.start_percent)


if __name__ == "__main__":
    main()
