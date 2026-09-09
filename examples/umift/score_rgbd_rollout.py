"""Read-only session-equal scoring for complete E3-Dout RGBD suffix rollouts."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

from examples.umift.rgbd_metrics import depth_video_metrics


MODEL_METHODS = ("B0", "E3-A", "E3-Z", "E3-S")
METHODS = ("P", *MODEL_METHODS)
SEGMENTS = ("full_suffix", "first_block", "early_third", "middle_third", "late_third")
RGB_METRICS = ("rgb_mse", "rgb_psnr", "rgb_ssim", "rgb_lpips", "rgb_temporal_l1")
DEPTH_METRICS = (
    "depth_mae_m",
    "depth_rmse_m",
    "depth_valid_fraction",
    "depth_zero_fraction",
    "depth_cap_fraction",
    "depth_out_of_range_fraction",
    "depth_zero_region_mean_m",
    "depth_cap_underprediction_m",
)
FRAME_METRICS = (*RGB_METRICS, *DEPTH_METRICS)


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verified_load_npy(
    path: Path,
    *,
    expected_sha256: str,
    expected_shape: tuple[int, ...],
    label: str,
) -> np.ndarray:
    """Memory-map one immutable float32 artifact after hash, shape, and finite checks."""
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    if file_sha(path) != expected_sha256:
        raise ValueError(f"{label} SHA-256 differs: {path}")
    value = np.load(path, mmap_mode="r", allow_pickle=False)
    if value.dtype != np.float32:
        raise ValueError(f"{label} must use float32, got {value.dtype}")
    if tuple(value.shape) != tuple(expected_shape):
        raise ValueError(f"{label} shape must be {expected_shape}, got {value.shape}")
    for start in range(0, len(value), 64):
        if not np.isfinite(value[start : start + 64]).all():
            raise ValueError(f"{label} must contain only finite values")
    return value


def segment_frame_indices(
    frame_count: int, first_block_steps: int
) -> dict[str, list[int]]:
    """Return generated-frame indexes; the observed anchor at index zero is always excluded."""
    if frame_count < 4:
        raise ValueError(
            "RGBD suffix scoring requires an anchor and at least three generated frames"
        )
    generated = np.arange(1, frame_count, dtype=np.int64)
    if not 1 <= int(first_block_steps) <= len(generated):
        raise ValueError("first rollout block has an invalid generated-frame count")
    thirds = np.array_split(generated, 3)
    return {
        "full_suffix": generated.tolist(),
        "first_block": generated[: int(first_block_steps)].tolist(),
        "early_third": thirds[0].tolist(),
        "middle_third": thirds[1].tolist(),
        "late_third": thirds[2].tolist(),
    }


def _validate_finite_range(
    value: np.ndarray,
    *,
    label: str,
    lower: float | None = None,
    upper: float | None = None,
) -> None:
    for start in range(0, len(value), 16):
        block = np.asarray(value[start : start + 16])
        if not np.isfinite(block).all():
            raise ValueError(f"{label} must contain only finite values")
        if lower is not None and np.any(block < lower):
            raise ValueError(f"{label} must be in [{lower},{upper}]")
        if upper is not None and np.any(block > upper):
            raise ValueError(f"{label} must be in [{lower},{upper}]")


def _validate_episode_arrays(
    truth_rgb: np.ndarray,
    truth_depth_m: np.ndarray,
    predictions: dict[str, tuple[np.ndarray, np.ndarray]],
) -> None:
    if truth_rgb.dtype != np.float32 or truth_depth_m.dtype != np.float32:
        raise ValueError("RGB and depth truth must use float32")
    if (
        truth_rgb.ndim != 4
        or truth_rgb.shape[-1] != 3
        or truth_depth_m.shape != truth_rgb.shape[:3]
    ):
        raise ValueError("truth must have matching RGB THWC and depth THW shapes")
    _validate_finite_range(truth_rgb, label="RGB truth", lower=0.0, upper=1.0)
    _validate_finite_range(truth_depth_m, label="depth truth", lower=0.0, upper=0.5)
    if tuple(predictions) != MODEL_METHODS:
        raise ValueError(f"predictions must contain methods in order {MODEL_METHODS}")
    for method, (rgb, depth_m) in predictions.items():
        if rgb.dtype != np.float32 or depth_m.dtype != np.float32:
            raise ValueError(f"{method} RGB and raw depth predictions must use float32")
        if rgb.shape != truth_rgb.shape or depth_m.shape != truth_depth_m.shape:
            raise ValueError(f"{method} prediction shape differs from GT")
        _validate_finite_range(
            rgb, label=f"{method} RGB prediction", lower=0.0, upper=1.0
        )
        _validate_finite_range(depth_m, label=f"{method} raw depth prediction")
        if not np.allclose(rgb[0], truth_rgb[0], atol=2e-7, rtol=0) or not np.allclose(
            depth_m[0], truth_depth_m[0], atol=2e-7, rtol=0
        ):
            raise ValueError(
                f"{method} anchor differs from canonical GT beyond float32 tolerance"
            )


def _rgb_frame_metrics(
    truth: np.ndarray,
    prediction: np.ndarray,
    *,
    evaluate_video_pair_fn: Callable[..., dict[str, Any]],
    lpips_metric: Any,
) -> list[dict[str, float | int]]:
    """Reuse the exact legacy 17-frame RGB evaluator in padded blocks, retaining real rows only."""
    rows: list[dict[str, float | int]] = []
    for start in range(1, len(truth), 16):
        stop = min(start + 16, len(truth))
        count = stop - start
        block_truth = np.concatenate(
            (truth[start - 1 : start], truth[start:stop]), axis=0
        )
        block_prediction = np.concatenate(
            (truth[start - 1 : start], prediction[start:stop]), axis=0
        )
        if count < 16:
            block_truth = np.concatenate(
                (block_truth, np.repeat(block_truth[-1:], 16 - count, axis=0)), axis=0
            )
            block_prediction = np.concatenate(
                (
                    block_prediction,
                    np.repeat(block_prediction[-1:], 16 - count, axis=0),
                ),
                axis=0,
            )
        legacy = evaluate_video_pair_fn(
            block_truth.astype(np.float32, copy=False),
            block_prediction.astype(np.float32, copy=False),
            include_lpips=True,
            lpips_metric=lpips_metric,
        )
        per_frame = legacy.get("per_frame")
        if not isinstance(per_frame, list) or len(per_frame) != 16:
            raise ValueError("legacy RGB evaluator did not return 16 frame rows")
        for offset, values in enumerate(per_frame[:count]):
            if set(values) != {"mse", "psnr", "ssim", "lpips"}:
                raise ValueError(
                    "legacy RGB evaluator returned an unexpected metric set"
                )
            rows.append(
                {
                    "frame_index": start + offset,
                    "rgb_mse": float(values["mse"]),
                    "rgb_psnr": float(values["psnr"]),
                    "rgb_ssim": float(values["ssim"]),
                    "rgb_lpips": float(values["lpips"]),
                }
            )

    temporal_values = []
    for start in range(1, len(truth), 16):
        stop = min(start + 16, len(truth))
        truth_pair = np.concatenate(
            (truth[start - 1 : start], truth[start:stop]), axis=0
        )
        prediction_anchor = truth[:1] if start == 1 else prediction[start - 1 : start]
        prediction_pair = np.concatenate(
            (prediction_anchor, prediction[start:stop]), axis=0
        )
        temporal_error = np.diff(truth_pair.astype(np.float64), axis=0) - np.diff(
            prediction_pair.astype(np.float64), axis=0
        )
        temporal_values.extend(np.sum(np.abs(temporal_error), axis=(1, 2, 3)).tolist())
    if len(rows) != len(temporal_values):
        raise ValueError("RGB frame and temporal metric lengths differ")
    for row, value in zip(rows, temporal_values, strict=True):
        row["rgb_temporal_l1"] = float(value)
    return rows


def _depth_frame_metrics(
    truth: np.ndarray, prediction: np.ndarray
) -> list[dict[str, Any]]:
    rows = []
    for start in range(1, len(truth), 16):
        stop = min(start + 16, len(truth))
        block_truth = np.concatenate(
            (truth[start - 1 : start], truth[start:stop]), axis=0
        )
        block_prediction = np.concatenate(
            (prediction[start - 1 : start], prediction[start:stop]), axis=0
        )
        result = depth_video_metrics(block_truth, block_prediction, exclude_first=True)
        for values in result["frames"]:
            row = dict(values)
            row["frame_index"] = start + int(row.pop("frame")) - 1
            rows.append(row)
    return rows


def _mean_frame_rows(rows: Iterable[dict[str, Any]]) -> dict[str, float | int | None]:
    values = list(rows)
    if not values:
        raise ValueError("cannot summarize an empty frame segment")
    result: dict[str, float | int | None] = {"frame_count": len(values)}
    for metric in FRAME_METRICS:
        present = [float(row[metric]) for row in values if row[metric] is not None]
        result[metric] = float(np.mean(present)) if present else None
    return result


def score_episode_arrays(
    truth_rgb: np.ndarray,
    truth_depth_m: np.ndarray,
    predictions: dict[str, tuple[np.ndarray, np.ndarray]],
    *,
    first_block_steps: int,
    evaluate_video_pair_fn: Callable[..., dict[str, Any]],
    lpips_metric: Any,
) -> dict[str, dict[str, Any]]:
    """Score P/B0/E3-A/E3-Z/E3-S for one complete suffix without changing input arrays."""
    _validate_episode_arrays(truth_rgb, truth_depth_m, predictions)
    segments = segment_frame_indices(len(truth_rgb), first_block_steps)
    all_predictions = {
        "P": (
            np.broadcast_to(truth_rgb[:1], truth_rgb.shape),
            np.broadcast_to(truth_depth_m[:1], truth_depth_m.shape),
        ),
        **predictions,
    }
    result: dict[str, dict[str, Any]] = {}
    for method, (rgb, depth_m) in all_predictions.items():
        rgb_rows = _rgb_frame_metrics(
            truth_rgb,
            rgb,
            evaluate_video_pair_fn=evaluate_video_pair_fn,
            lpips_metric=lpips_metric,
        )
        depth_rows = _depth_frame_metrics(truth_depth_m, depth_m)
        if [row["frame_index"] for row in rgb_rows] != [
            row["frame_index"] for row in depth_rows
        ]:
            raise ValueError(f"{method} RGB/depth frame indexes differ")
        frames = [
            {**rgb_row, **depth_row}
            for rgb_row, depth_row in zip(rgb_rows, depth_rows, strict=True)
        ]
        by_index = {int(row["frame_index"]): row for row in frames}
        segment_results = {}
        for name, indexes in segments.items():
            segment_results[name] = {
                "frame_indices": indexes,
                "mean": _mean_frame_rows(by_index[index] for index in indexes),
            }
        result[method] = {
            "frame_indices": segments["full_suffix"],
            "frames": frames,
            "segments": segment_results,
        }
    return result


def aggregate_session_equal_frame_rows(
    rows: Iterable[dict[str, Any]], metric_names: Iterable[str] = FRAME_METRICS
) -> dict[str, Any]:
    """Average frames within each session, then give every session equal weight."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["raw_session"])].append(row)
    if not grouped:
        raise ValueError("cannot aggregate an empty frame result")
    metric_names = tuple(metric_names)
    sessions: dict[str, dict[str, float | None]] = {}
    value_counts: dict[str, dict[str, int]] = {name: {} for name in metric_names}
    for session, members in sorted(grouped.items()):
        sessions[session] = {}
        for name in metric_names:
            values = [float(row[name]) for row in members if row[name] is not None]
            sessions[session][name] = float(np.mean(values)) if values else None
            value_counts[name][session] = len(values)
    overall = {}
    for name in metric_names:
        values = [
            float(metrics[name])
            for metrics in sessions.values()
            if metrics[name] is not None
        ]
        overall[name] = float(np.mean(values)) if values else None
    return {
        "sessions": sessions,
        "overall": overall,
        "frame_counts": {
            session: len(members) for session, members in sorted(grouped.items())
        },
        "value_counts": value_counts,
        "aggregation_order": "mean frames within raw_session, then mean raw_sessions equally",
    }


def _validate_chunks(episode: dict[str, Any]) -> int:
    chunks = episode.get("chunks")
    if not isinstance(chunks, list) or not chunks:
        raise ValueError("prepared rollout has no chunk inventory")
    expected_start = 0
    for index, chunk in enumerate(chunks):
        if (
            int(chunk["index"]) != index
            or int(chunk["output_start"]) != expected_start
            or not 1 <= int(chunk["steps"]) <= 16
        ):
            raise ValueError("prepared rollout chunks are not contiguous legal blocks")
        expected_start += int(chunk["steps"])
    if expected_start != int(episode["frame_count"]) - 1:
        raise ValueError("prepared rollout chunks do not cover the complete suffix")
    return int(chunks[0]["steps"])


def _load_prediction(
    root: Path,
    manifest: dict[str, Any],
    manifest_sha: str,
    episode: dict[str, Any],
    method: str,
    expected_rgb_shape: tuple[int, ...],
    expected_depth_shape: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    label = int(episode["start_percent"])
    stem = (
        root
        / "rgbd_inference"
        / method
        / f"episode_{episode['episode_id']}_start_{label}"
    )
    metadata_path = stem.with_suffix(".json")
    metadata = json.loads(metadata_path.read_text())
    expected_checkpoint = (
        manifest["base_checkpoint"]
        if method == "B0"
        else manifest["selected_checkpoint"]
    )
    if (
        metadata.get("method") != method
        or int(metadata.get("episode_id", -1)) != int(episode["episode_id"])
        or metadata.get("raw_session") != episode["raw_session"]
        or int(metadata.get("start_percent", -1)) != label
        or int(metadata.get("frame_count", -1)) != int(episode["frame_count"])
        or metadata.get("complete_requested_suffix") is not True
        or metadata.get("manifest_sha256") != manifest_sha
        or metadata.get("checkpoint_id") != expected_checkpoint
        or metadata.get("future_gt_refresh_count") != 0
    ):
        raise ValueError(
            f"{method} rollout metadata identity is invalid for episode {episode['episode_id']}"
        )
    rgb_path = Path(metadata["rgb_path"]).resolve()
    depth_path = Path(metadata["raw_depth_m_path"]).resolve()
    if (
        rgb_path != stem.with_name(stem.name + "_rgb.npy").resolve()
        or depth_path != stem.with_name(stem.name + "_depth_m.npy").resolve()
    ):
        raise ValueError(f"{method} rollout paths escape the expected method directory")
    rgb = verified_load_npy(
        rgb_path,
        expected_sha256=metadata["rgb_sha256"],
        expected_shape=expected_rgb_shape,
        label=f"{method} RGB prediction",
    )
    depth = verified_load_npy(
        depth_path,
        expected_sha256=metadata["raw_depth_m_sha256"],
        expected_shape=expected_depth_shape,
        label=f"{method} raw depth prediction",
    )
    metadata_chunks = metadata.get("chunks")
    if not isinstance(metadata_chunks, list) or len(metadata_chunks) != len(
        episode["chunks"]
    ):
        raise ValueError(f"{method} rollout metadata has missing or extra chunks")
    for chunk, record in zip(episode["chunks"], metadata_chunks, strict=True):
        for key in ("index", "output_start", "steps", "noise_seed"):
            if int(record[key]) != int(chunk[key]):
                raise ValueError(
                    f"{method} rollout chunk identity differs from the prepared manifest"
                )
        start = int(chunk["output_start"]) + 1
        stop = start + int(chunk["steps"])
        from examples.umift.rgbd_rollout import array_sha

        if array_sha(np.asarray(rgb[start:stop])) != record["retained_rgb_sha256"]:
            raise ValueError(f"{method} retained RGB chunk hash differs")
        if (
            array_sha(np.asarray(depth[start:stop]))
            != record["retained_raw_depth_m_sha256"]
        ):
            raise ValueError(f"{method} retained raw depth chunk hash differs")
    return (
        rgb,
        depth,
        {
            "method": method,
            "checkpoint_id": expected_checkpoint,
            "metadata_path": str(metadata_path.resolve()),
            "metadata_sha256": file_sha(metadata_path),
            "rgb_path": str(rgb_path),
            "rgb_sha256": metadata["rgb_sha256"],
            "raw_depth_m_path": str(depth_path),
            "raw_depth_m_sha256": metadata["raw_depth_m_sha256"],
        },
    )


def _npz_frame_columns(episode_results: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    records = []
    for episode in episode_results:
        for method_index, method in enumerate(METHODS):
            for row in episode["scored"][method]["frames"]:
                records.append(
                    {
                        **row,
                        "method_index": method_index,
                        "episode_id": int(episode["episode_id"]),
                        "start_percent": int(episode["start_percent"]),
                        "raw_session": str(episode["raw_session"]),
                    }
                )
    columns = {
        "method_names": np.asarray(METHODS),
        "method_index": np.asarray(
            [row["method_index"] for row in records], dtype=np.int16
        ),
        "episode_id": np.asarray(
            [row["episode_id"] for row in records], dtype=np.int16
        ),
        "start_percent": np.asarray(
            [row["start_percent"] for row in records], dtype=np.int16
        ),
        "raw_session": np.asarray([row["raw_session"] for row in records]),
        "frame_index": np.asarray(
            [row["frame_index"] for row in records], dtype=np.int32
        ),
    }
    for metric in FRAME_METRICS:
        columns[metric] = np.asarray(
            [np.nan if row[metric] is None else float(row[metric]) for row in records],
            dtype=np.float64,
        )
    return columns


def score_root(root: Path, output: Path | None = None) -> dict[str, Any]:
    """Validate and score one complete prepared/inferred rollout root without modifying inputs."""
    from examples.umift.evaluate import evaluate_video_pair
    from examples.umift.rgbd_rollout import _validate_manifest
    from examples.umift.score_refit_reference import _lpips_metric

    root = root.resolve()
    output = (root / "scoring" if output is None else output).resolve()
    if output.exists():
        raise FileExistsError(output)
    manifest_path = (root / "prepared" / "manifest.json").resolve()
    manifest_sha = file_sha(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    _validate_manifest(manifest, manifest_path)
    episodes = manifest.get("episodes", [])
    if len(episodes) != 3:
        raise ValueError("scoring requires the three prepared held-out sessions")

    lpips_metric = _lpips_metric()
    episode_results = []
    aggregate_rows: dict[str, dict[str, list[dict[str, Any]]]] = {
        method: {segment: [] for segment in SEGMENTS} for method in METHODS
    }
    for episode in episodes:
        input_hashes = episode.get("input_files_sha256")
        if not isinstance(input_hashes, dict) or not input_hashes:
            raise ValueError("prepared episode lacks its frozen input hash inventory")
        for filename, expected in input_hashes.items():
            path = Path(filename)
            if not path.is_file() or file_sha(path) != expected:
                raise ValueError(f"prepared frozen input hash differs: {filename}")
        frame_count = int(episode["frame_count"])
        rgb_shape = (frame_count, 256, 256, 3)
        depth_shape = (frame_count, 256, 256)
        truth_rgb = verified_load_npy(
            Path(episode["truth_rgb_path"]),
            expected_sha256=episode["truth_rgb_sha256"],
            expected_shape=rgb_shape,
            label="canonical RGB truth",
        )
        truth_depth = verified_load_npy(
            Path(episode["truth_depth_m_path"]),
            expected_sha256=episode["truth_depth_m_sha256"],
            expected_shape=depth_shape,
            label="canonical raw depth truth",
        )
        first_block_steps = _validate_chunks(episode)
        predictions = {}
        inventories = {}
        for method in MODEL_METHODS:
            rgb, depth, inventory = _load_prediction(
                root,
                manifest,
                manifest_sha,
                episode,
                method,
                rgb_shape,
                depth_shape,
            )
            predictions[method] = (rgb, depth)
            inventories[method] = inventory
        scored = score_episode_arrays(
            truth_rgb,
            truth_depth,
            predictions,
            first_block_steps=first_block_steps,
            evaluate_video_pair_fn=evaluate_video_pair,
            lpips_metric=lpips_metric,
        )
        episode_result = {
            "episode_id": int(episode["episode_id"]),
            "raw_session": str(episode["raw_session"]),
            "start_percent": int(episode["start_percent"]),
            "frame_count": frame_count,
            "first_block_steps": first_block_steps,
            "truth_rgb_path": str(Path(episode["truth_rgb_path"]).resolve()),
            "truth_rgb_sha256": episode["truth_rgb_sha256"],
            "truth_depth_m_path": str(Path(episode["truth_depth_m_path"]).resolve()),
            "truth_depth_m_sha256": episode["truth_depth_m_sha256"],
            "prediction_inventory": inventories,
            "scored": scored,
        }
        episode_results.append(episode_result)
        segment_indexes = segment_frame_indices(frame_count, first_block_steps)
        for method in METHODS:
            by_index = {
                int(row["frame_index"]): row for row in scored[method]["frames"]
            }
            for segment, indexes in segment_indexes.items():
                aggregate_rows[method][segment].extend(
                    {**by_index[index], "raw_session": episode["raw_session"]}
                    for index in indexes
                )

    aggregates = {
        method: {
            segment: aggregate_session_equal_frame_rows(rows)
            for segment, rows in aggregate_rows[method].items()
        }
        for method in METHODS
    }
    output.mkdir(parents=True, exist_ok=False)
    frame_path = output / "frame_metrics.npz"
    np.savez(frame_path, **_npz_frame_columns(episode_results))
    compact_episodes = []
    for episode in episode_results:
        compact_episodes.append(
            {key: value for key, value in episode.items() if key != "scored"}
            | {
                "methods": {
                    method: {"segments": values["segments"]}
                    for method, values in episode["scored"].items()
                }
            }
        )
    report = {
        "protocol": "e3-dout-rgbd-full-suffix-scoring-v1",
        "experiment_id": "E3-Dout",
        "complete": True,
        "methods": list(METHODS),
        "segments": {
            "full_suffix": "all generated frames; observed anchor index 0 excluded",
            "first_block": "generated frames retained from prepared chunk 0",
            "early_middle_late": "numpy.array_split of all generated frame indexes into three contiguous frame-count parts",
        },
        "aggregation_order": "per-frame metrics, mean frames within raw_session, mean raw_sessions equally",
        "rgb_metric_source": "examples.umift.evaluate.evaluate_video_pair",
        "rgb_evaluation_conversion": "float32 [0,1] passed directly; no uint8 round trip",
        "rgb_temporal_l1": "legacy temporal difference absolute-error sum per generated frame, continuous across blocks",
        "depth_metric_source": "examples.umift.rgbd_metrics.depth_video_metrics",
        "depth_mask": "fixed GT finite & 0<depth_m<0.5; prediction mask is never intersected",
        "depth_prediction": "raw metre-valued output; no clamp before metrics",
        "depth_diagnostics": {
            "depth_zero_fraction": "fraction of GT pixels exactly equal to 0m",
            "depth_cap_fraction": "fraction of GT pixels exactly equal to 0.5m",
            "depth_out_of_range_fraction": "fraction of raw predicted pixels below 0m or above 0.5m",
            "depth_zero_region_mean_m": "mean raw prediction where GT equals 0m; null when absent",
            "depth_cap_underprediction_m": "mean positive 0.5m-minus-prediction where GT equals 0.5m; null when absent",
        },
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_sha,
        "frame_metrics_npz": str(frame_path),
        "frame_metrics_npz_sha256": file_sha(frame_path),
        "frame_metrics_npz_schema": {
            "method_names": "unicode[METHODS] lookup table",
            "method_index": "int16 row index into method_names",
            "episode_id": "int16",
            "start_percent": "int16",
            "raw_session": "unicode",
            "frame_index": "int32 generated-frame index; anchor 0 absent",
            "metric_columns": f"float64 columns {list(FRAME_METRICS)}; absent GT strata are NaN",
        },
        "session_equal_aggregate": aggregates,
        "episodes": compact_episodes,
    }
    if file_sha(manifest_path) != manifest_sha:
        raise ValueError("prepared manifest changed during scoring")
    report_path = output / "metrics.json"
    report_path.write_text(json.dumps(report, indent=2, allow_nan=True) + "\n")
    return {
        **report,
        "metrics_path": str(report_path),
        "metrics_sha256": file_sha(report_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, help="New output directory; defaults to ROOT/scoring"
    )
    args = parser.parse_args()
    report = score_root(args.root, args.output)
    print(
        json.dumps(
            {
                "metrics": report["metrics_path"],
                "metrics_sha256": report["metrics_sha256"],
                "frame_metrics": report["frame_metrics_npz"],
                "frame_metrics_sha256": report["frame_metrics_npz_sha256"],
            }
        )
    )


if __name__ == "__main__":
    main()
