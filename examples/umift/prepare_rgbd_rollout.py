"""Bind frozen E2-H suffix fixtures to E3-Dout and materialize canonical RGBD truth."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import numpy as np

from examples.umift.long_rollout import BASE_CHECKPOINT, EPISODES
from examples.umift.rgbd_rollout import array_sha, file_sha


PARENT_MANIFESTS = {
    0: (
        Path("/data/cosmos_runs/e1_long_rollout_20260907/prepared/manifest.json"),
        "8f311b93c18e20d8ab752e5f86e33e0691296e10bcbdb565511ddd823ba151d9",
    ),
    33: (
        Path("/data/cosmos_runs/e1_midstart_rollout_20260907/start33/prepared/manifest.json"),
        "9c18fc449ed79a7230b047c0cff85f74837748409fe316c0301cb51f4dc96949",
    ),
    67: (
        Path("/data/cosmos_runs/e1_midstart_rollout_20260907/start67/prepared/manifest.json"),
        "aa3259970c9afd3e151972f439fff92e324dd7355957c3005b40bfbadea48645",
    ),
}


def bind_manifest(
    parent: dict[str, Any],
    selection: dict[str, Any],
    *,
    source_manifest: Path,
    source_sha256: str,
    selection_file: Path,
    selection_sha256: str,
    start_percent: int,
) -> dict[str, Any]:
    """Copy immutable suffix identities and add only E3 protocol/checkpoint bindings."""
    result = copy.deepcopy(parent)
    result.update(
        experiment_id="E3-Dout",
        protocol="e3-dout-rgbd-open-loop-v1",
        history_frames=5,
        future_frames=16,
        selected_iteration=int(selection["iteration"]),
        selected_checkpoint=str(selection["checkpoint"]),
        selection_protocol_file=str(selection["protocol_file"]),
        selection_protocol_sha256=str(selection["protocol_sha256"]),
        selection_rgb_weight=float(selection["rgb_weight"]),
        selection_joint_score=float(selection["joint_score"]),
        selection_uses_test_episodes=True,
        selection_file=str(selection_file),
        selection_sha256=selection_sha256,
        source_manifest=str(source_manifest),
        source_manifest_sha256=source_sha256,
        start_percent=int(start_percent),
        depth_units="metres",
        depth_representation="linear gray3: normalized=4*clip(depth_m,0,0.5)-1",
        feedback_dtype="float32",
        initial_h5_rgbd_only=True,
        future_gt_refresh_count=0,
        current_frame_display="true observed RGBD anchor; never decoder reconstruction",
        base_checkpoint_rgbd_pretrained=False,
        zarr_complete_manifest_sha256="308f4d46132885375eb00c5578d52529512f0bca2c4e422c13724bfef1c6b9e6",
        zarr_complete_manifest_file_count=15226,
        zarr_complete_manifest_bytes=13403184883,
    )
    result.pop("checkpoint_selection_sha256", None)
    for episode in result["episodes"]:
        anchor = int(episode.get("initial_source_frame", 0))
        episode.update(
            experiment_id="E3-Dout",
            start_percent=int(start_percent),
            initial_source_frame=anchor,
            initial_selected_frame=int(episode.get("initial_selected_frame", 0)),
            parent_frame_count=int(episode.get("parent_frame_count", episode["frame_count"])),
            initial_episode_elapsed_seconds=float(
                episode.get("initial_episode_elapsed_seconds", 0.0)
            ),
            selected_iteration=int(selection["iteration"]),
            history_frames=5,
            history_padding_count=int(sum(anchor - 2 * offset < 0 for offset in range(5))),
        )
    return result


def _validate_selection(path: Path) -> tuple[dict[str, Any], str, dict[str, Any]]:
    from examples.umift.rgbd_selection import choose_candidate, load_protocol

    selection_sha = file_sha(path)
    selection = json.loads(path.read_text())
    if (
        selection.get("experiment_id") != "E3-Dout"
        or selection.get("history_frames") != 5
        or selection.get("selection_uses_test_episodes") is not True
        or not isinstance(selection.get("candidates"), list)
        or len(selection["candidates"]) != 6
    ):
        raise ValueError("selection is not the frozen E3-Dout H5 result")
    protocol = Path(selection["protocol_file"])
    protocol_record, protocol_sha = load_protocol(protocol)
    if protocol_sha != selection["protocol_sha256"]:
        raise ValueError("E3-Dout selection protocol changed")
    reports = []
    for candidate in selection["candidates"]:
        metrics_path = Path(candidate["file"])
        if file_sha(metrics_path) != candidate["sha256"]:
            raise ValueError("E3-Dout candidate metrics changed after selection")
        report = json.loads(metrics_path.read_text())
        if candidate.get("iteration") != report.get("iteration"):
            raise ValueError("candidate iteration differs from its metrics report")
        if (
            not np.isclose(candidate.get("joint_score"), report.get("joint_score"), atol=1e-12, rtol=0)
            or candidate.get("metrics") != report.get("session_equal_aggregate", {}).get("overall")
        ):
            raise ValueError("candidate summary differs from its bound metrics report")
        reports.append(report)
    best = choose_candidate(
        reports, selection["protocol_sha256"], float(selection["rgb_weight"])
    )
    if (
        int(selection["iteration"]) != int(best["iteration"])
        or str(selection["checkpoint"]) != str(best["checkpoint"])
        or not np.isclose(
            float(selection["joint_score"]), float(best["joint_score"]), atol=1e-12, rtol=0
        )
    ):
        raise ValueError("selected E3-Dout checkpoint is not the frozen joint-score minimum")
    checkpoint = Path(selection["checkpoint"])
    if not (checkpoint / ".metadata").is_file():
        raise FileNotFoundError(checkpoint / ".metadata")
    return selection, selection_sha, protocol_record


def _materialize_truth(manifest: dict[str, Any], destination: Path) -> None:
    import zarr

    from cosmos_framework.data.generator.action.datasets.umift_history_dataset import _resize_rgb
    from cosmos_framework.data.generator.action.datasets.umift_rgbd_dataset import (
        _resize_depth_nearest,
    )

    zarr_root = zarr.open_group(manifest["zarr_path"], mode="r")["data"]
    for episode in manifest["episodes"]:
        for filename, expected in episode["input_files_sha256"].items():
            if file_sha(Path(filename)) != expected:
                raise ValueError(f"frozen suffix input changed: {filename}")
        frame_count = int(episode["frame_count"])
        index_path = Path(episode["frame_indices_path"])
        with np.load(index_path, allow_pickle=False) as archive:
            source_indices = np.asarray(archive["source_indices"], dtype=np.int64)
            timestamps = np.asarray(archive["timestamps"], dtype=np.float64)
        if (
            source_indices.shape != (frame_count,)
            or timestamps.shape != (frame_count,)
            or not np.array_equal(
                source_indices,
                int(episode["initial_source_frame"]) + 2 * np.arange(frame_count, dtype=np.int64),
            )
            or not np.isfinite(timestamps).all()
            or np.any(np.diff(timestamps) <= 0)
        ):
            raise ValueError("frozen suffix indices or PTS differ from the audited stride-2 fixture")
        group = zarr_root[f"episode_{episode['episode_id']}"]
        if tuple(group["depth_0"].shape) != (int(episode["source_length"]), 224, 224, 3):
            raise ValueError("canonical depth source shape differs from the audited episode")
        old_truth = np.load(episode["truth_path"], mmap_mode="r", allow_pickle=False)
        if old_truth.dtype != np.uint8 or old_truth.shape != (frame_count, 256, 256, 3):
            raise ValueError("frozen RGB-only truth fixture has unexpected shape or dtype")
        stem = destination / f"episode_{episode['episode_id']}_start_{episode['start_percent']}"
        rgb_path = stem.with_name(stem.name + "_truth_rgb.npy")
        depth_path = stem.with_name(stem.name + "_truth_depth_m.npy")
        rgb_output = np.lib.format.open_memmap(
            rgb_path, mode="w+", dtype=np.float32, shape=(frame_count, 256, 256, 3)
        )
        depth_output = np.lib.format.open_memmap(
            depth_path, mode="w+", dtype=np.float32, shape=(frame_count, 256, 256)
        )
        for start in range(0, frame_count, 32):
            indexes = source_indices[start : start + 32]
            source_rgb = np.asarray(group["rgb_0"].oindex[indexes], dtype=np.uint8)
            resized_rgb_u8 = _resize_rgb(source_rgb).permute(1, 2, 3, 0).numpy()
            if not np.array_equal(resized_rgb_u8, np.asarray(old_truth[start : start + len(indexes)])):
                raise ValueError("canonical Zarr RGB differs from the reused frozen suffix truth")
            rgb_output[start : start + len(indexes)] = resized_rgb_u8.astype(np.float32) / 255.0
            source_depth = np.asarray(group["depth_0"].oindex[indexes])
            if (
                not np.issubdtype(source_depth.dtype, np.floating)
                or not np.isfinite(source_depth).all()
                or np.any(source_depth < 0.0)
                or np.any(source_depth > 0.5)
                or not np.array_equal(source_depth[..., 0], source_depth[..., 1])
                or not np.array_equal(source_depth[..., 0], source_depth[..., 2])
            ):
                raise ValueError("canonical depth must be finite repeated-channel metres in [0,0.5]")
            resized_depth = _resize_depth_nearest(source_depth[..., 0]).cpu().numpy()
            depth_output[start : start + len(indexes)] = resized_depth
        rgb_output.flush()
        depth_output.flush()
        del rgb_output, depth_output
        episode.update(
            truth_rgb_path=str(rgb_path.resolve()),
            truth_rgb_sha256=file_sha(rgb_path),
            truth_depth_m_path=str(depth_path.resolve()),
            truth_depth_m_sha256=file_sha(depth_path),
            truth_source="canonical Zarr RGB/depth at reused frozen source_indices",
            truth_source_indices_sha256=array_sha(source_indices),
            true_pts=timestamps.tolist(),
        )


def prepare(args: argparse.Namespace) -> Path:
    source_manifest, expected_sha = PARENT_MANIFESTS[int(args.start_percent)]
    if args.source_manifest is not None:
        source_manifest = args.source_manifest.resolve()
    actual_source_sha = file_sha(source_manifest)
    if actual_source_sha != expected_sha:
        raise ValueError("source suffix manifest differs from the exact frozen audit SHA-256")
    parent = json.loads(source_manifest.read_text())
    if (
        parent.get("base_checkpoint") != BASE_CHECKPOINT
        or tuple(episode["episode_id"] for episode in parent.get("episodes", [])) != EPISODES
    ):
        raise ValueError("source suffix does not contain the frozen base identity and held-out episodes")
    selection, selection_sha, selection_protocol = _validate_selection(args.selection.resolve())
    if str(Path(selection_protocol["zarr_path"]).resolve()) != str(Path(parent["zarr_path"]).resolve()):
        raise ValueError("selection and long suffixes use different canonical Zarr roots")
    manifest = bind_manifest(
        parent,
        selection,
        source_manifest=source_manifest,
        source_sha256=actual_source_sha,
        selection_file=args.selection.resolve(),
        selection_sha256=selection_sha,
        start_percent=int(args.start_percent),
    )
    destination = args.root.resolve() / "prepared"
    destination.mkdir(parents=True, exist_ok=False)
    _materialize_truth(manifest, destination)
    checkpoint = Path(manifest["selected_checkpoint"])
    manifest["checkpoint_metadata_sha256"] = file_sha(checkpoint / ".metadata")
    path = destination / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--start-percent", type=int, choices=(0, 33, 67), required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--source-manifest",
        type=Path,
        help="Exact frozen manifest path; SHA remains fixed by --start-percent",
    )
    args = parser.parse_args()
    path = prepare(args)
    print(json.dumps({"manifest": str(path), "sha256": file_sha(path)}))


if __name__ == "__main__":
    main()
