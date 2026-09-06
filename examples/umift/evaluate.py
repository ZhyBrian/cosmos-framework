"""Offline, session-balanced evaluation for the Cosmos3 UMI-FT E1 protocol.

The scorer consumes saved RGB predictions.  It intentionally does not pretend
to be a checkpoint launcher: Edge forward-dynamics loading and sampling need
the experiment-specific model builder and exported checkpoint to exist first.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable, Literal

import numpy as np
from scipy.optimize import linear_sum_assignment
from skimage.metrics import structural_similarity

from examples.umift.protocol import derive_noise_seed, persistence_prediction

Array = np.ndarray
ActionSpace = Literal["raw", "normalized"]


def _as_thwc01(video: Array, *, label: str) -> Array:
    value = np.asarray(video)
    if value.ndim == 5 and value.shape[0] == 1:
        value = value[0]
    if value.ndim != 4:
        raise ValueError(f"{label} must have four dimensions, got {value.shape}")
    if value.shape[-1] in (1, 3):
        pass
    elif value.shape[0] in (1, 3):
        value = np.moveaxis(value, 0, -1)
    else:
        raise ValueError(f"{label} must be THWC or CTHW RGB, got {value.shape}")
    value = value.astype(np.float32, copy=False)
    if not np.isfinite(value).all():
        raise ValueError(f"{label} contains non-finite values")
    low, high = float(value.min()), float(value.max())
    if low < 0.0 or high > 1.0:
        raise ValueError(f"{label} must already be in [0,1], got [{low},{high}]")
    return value


def _ssim(truth: Array, prediction: Array) -> float:
    smallest_side = min(truth.shape[0], truth.shape[1])
    win_size = min(7, smallest_side if smallest_side % 2 else smallest_side - 1)
    if win_size < 3:
        raise ValueError("SSIM requires spatial dimensions of at least 3")
    return float(
        structural_similarity(truth, prediction, data_range=1.0, channel_axis=-1, win_size=win_size)
    )


def _pixel_metrics(truth: Array, prediction: Array) -> dict[str, float]:
    mse = float(np.mean(np.square(truth.astype(np.float64) - prediction.astype(np.float64))))
    return {
        "mse": mse,
        "psnr": math.inf if mse == 0.0 else float(-10.0 * math.log10(mse)),
        "ssim": _ssim(truth, prediction),
    }


def lpips_frames(
    truth: Array,
    prediction: Array,
    *,
    metric: Callable[[Array, Array], Array] | None = None,
) -> list[float]:
    """Score THWC [0,1] frames after exactly one conversion to NCHW [-1,1].

    Supplying ``metric`` keeps the numerical contract testable without torch.
    The default lazily imports lpips and fixes alex/version=0.1.
    """
    truth_nchw = np.moveaxis(truth * 2.0 - 1.0, -1, 1).astype(np.float32)
    pred_nchw = np.moveaxis(prediction * 2.0 - 1.0, -1, 1).astype(np.float32)
    if metric is not None:
        values = np.asarray(metric(truth_nchw, pred_nchw), dtype=np.float64).reshape(-1)
        return values.tolist()
    try:
        import torch
        import lpips
    except ImportError as exc:
        raise RuntimeError(
            "LPIPS requires torch and lpips; dependencies were not installed by this evaluator"
        ) from exc
    model = lpips.LPIPS(net="alex", version="0.1")
    with torch.no_grad():
        values = model(torch.from_numpy(truth_nchw), torch.from_numpy(pred_nchw), normalize=False)
    return values.detach().cpu().numpy().reshape(-1).astype(float).tolist()


def evaluate_video_pair(
    truth: Array,
    prediction: Array,
    *,
    include_lpips: bool = False,
    lpips_metric: Callable[[Array, Array], Array] | None = None,
) -> dict[str, Any]:
    """Evaluate indices 1..16 of one 17-frame RGB prediction."""
    truth = _as_thwc01(truth, label="truth")
    prediction = _as_thwc01(prediction, label="prediction")
    if truth.shape != prediction.shape:
        raise ValueError(f"truth/prediction shapes differ: {truth.shape} vs {prediction.shape}")
    if truth.shape[0] != 17:
        raise ValueError(f"E1 evaluation requires exactly 17 frames, got {truth.shape[0]}")

    frame_rows = [_pixel_metrics(truth[index], prediction[index]) for index in range(1, 17)]
    if include_lpips:
        values = lpips_frames(truth[1:], prediction[1:], metric=lpips_metric)
        if len(values) != 16:
            raise ValueError(f"LPIPS metric returned {len(values)} values for 16 frames")
        for row, value in zip(frame_rows, values, strict=True):
            row["lpips"] = value
    names = tuple(frame_rows[0])
    mean = {name: float(np.mean([row[name] for row in frame_rows])) for name in names}

    # Predicted temporal difference 0->1 is anchored at the observed true I0.
    predicted_for_diff = np.concatenate([truth[:1], prediction[1:]], axis=0)
    true_diff = np.diff(truth, axis=0)
    predicted_diff = np.diff(predicted_for_diff, axis=0)
    temporal_error = true_diff.astype(np.float64) - predicted_diff.astype(np.float64)
    temporal_per_frame = np.mean(
        np.square(temporal_error), axis=(1, 2, 3)
    )
    temporal_per_frame_l1 = np.sum(np.abs(temporal_error), axis=(1, 2, 3))
    return {
        "frame_indices": list(range(1, 17)),
        "per_frame": frame_rows,
        "mean": mean,
        "last": dict(frame_rows[-1]),
        "horizons": {str(k): dict(frame_rows[k - 1]) for k in (4, 8, 12, 16)},
        "temporal": {
            "per_frame_mse": temporal_per_frame.astype(float).tolist(),
            "mean_mse": float(temporal_per_frame.mean()),
            "per_frame_l1": temporal_per_frame_l1.astype(float).tolist(),
            "mean_l1": float(temporal_per_frame_l1.sum() / 16.0),
        },
        "reconstructed_i0": _pixel_metrics(truth[0], prediction[0]),
    }


def _mean_numeric_tree(values: list[Any], *, path: str) -> Any:
    if len(values) == 1:
        return values[0]
    first = values[0]
    if isinstance(first, dict):
        expected_keys = set(first)
        if any(not isinstance(value, dict) or set(value) != expected_keys for value in values[1:]):
            raise ValueError(f"incompatible dictionary keys while averaging {path}")
        return {
            key: _mean_numeric_tree([value[key] for value in values], path=f"{path}.{key}")
            for key in first
        }
    if isinstance(first, list):
        expected_length = len(first)
        if any(not isinstance(value, list) or len(value) != expected_length for value in values[1:]):
            raise ValueError(f"incompatible list lengths while averaging {path}")
        return [
            _mean_numeric_tree([value[index] for value in values], path=f"{path}[{index}]")
            for index in range(expected_length)
        ]
    numeric_types = (int, float, np.integer, np.floating)
    if isinstance(first, bool) or any(
        isinstance(value, bool) or not isinstance(value, numeric_types) for value in values
    ):
        raise ValueError(f"non-numeric value while averaging {path}")
    return sum(float(value) for value in values) / len(values)


def collapse_sampling_seeds(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Average sampling seeds inside each logical window before aggregation."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["window_id"])].append(row)
    collapsed: list[dict[str, Any]] = []
    for window_id, members in sorted(grouped.items()):
        sessions = {str(row["raw_session"]) for row in members}
        if len(sessions) != 1:
            raise ValueError(f"window {window_id!r} maps to multiple raw sessions: {sorted(sessions)}")
        metric_names = set.intersection(*(set(row["metrics"]) for row in members))
        first = {key: value for key, value in members[0].items() if key not in {"metrics", "sampling_seed"}}
        first["window_id"] = window_id
        first["sampling_seeds"] = sorted(int(row["sampling_seed"]) for row in members if "sampling_seed" in row)
        first["metrics"] = {
            name: float(np.mean([row["metrics"][name] for row in members])) for name in sorted(metric_names)
        }
        for block in ("last", "horizons", "temporal", "reconstructed_i0"):
            present = [block in row for row in members]
            if any(present) and not all(present):
                raise ValueError(f"window {window_id!r} has inconsistent {block!r} blocks across seeds")
            if all(present):
                first[block] = _mean_numeric_tree(
                    [row[block] for row in members], path=f"window {window_id}.{block}"
                )
        collapsed.append(first)
    return collapsed


def aggregate_session_equal(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Average windows within raw session, then give each session equal weight."""
    grouped: dict[str, list[dict[str, float]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["raw_session"])].append(row["metrics"])
    if not grouped:
        raise ValueError("cannot aggregate an empty result set")
    metric_names = set.intersection(*(set(metric) for values in grouped.values() for metric in values))
    sessions: dict[str, dict[str, float]] = {}
    for session, metrics in sorted(grouped.items()):
        sessions[session] = {
            name: float(np.mean([metric[name] for metric in metrics])) for name in sorted(metric_names)
        }
    overall = {
        name: float(np.mean([metrics[name] for metrics in sessions.values()])) for name in sorted(metric_names)
    }
    return {"sessions": sessions, "overall": overall, "window_counts": {k: len(v) for k, v in grouped.items()}}


def prepare_model_actions(
    actions: Array,
    *,
    source_space: ActionSpace,
    normalizer: Callable[[Array], Array],
) -> Array:
    """Produce model-space actions, with normalized inputs passed through once."""
    value = np.asarray(actions, dtype=np.float32)
    if source_space == "normalized":
        return value.copy()
    if source_space == "raw":
        return np.asarray(normalizer(value), dtype=np.float32)
    raise ValueError(f"unknown action source space: {source_space}")


def make_physical_zero_action(num_actions: int, *, normalizer: Callable[[Array], Array]) -> Array:
    """Create raw [xyz, rotation-6D identity, gripper=0], then normalize once."""
    if num_actions < 1:
        raise ValueError("num_actions must be positive")
    row = np.array([0, 0, 0, 1, 0, 0, 0, 1, 0, 0], dtype=np.float32)
    return prepare_model_actions(np.tile(row, (num_actions, 1)), source_space="raw", normalizer=normalizer)


def select_session_previews(window_ids: Iterable[str], *, count: int = 4) -> list[str]:
    ordered = sorted(window_ids)
    if len(ordered) < count:
        raise ValueError(f"need at least {count} windows, got {len(ordered)}")
    indices = np.rint(np.linspace(0, len(ordered) - 1, count)).astype(int)
    return [ordered[index] for index in indices]


def sparse_window_starts(num_source_frames: int) -> list[int]:
    """Return fixed E1 evaluation starts s=0,32,... with s+32 < N."""
    if num_source_frames < 0:
        raise ValueError("num_source_frames must be non-negative")
    return list(range(0, max(0, num_source_frames - 32), 32))


def build_action_permutation(
    rows: Iterable[dict[str, Any]], *, output_path: str | Path | None = None
) -> dict[str, str]:
    """Make a minimum-cost within-split derangement of whole-trajectory magnitudes."""
    by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_split[str(row["split"])].append(row)
    pairs: dict[str, str] = {}
    for split, members in sorted(by_split.items()):
        if len(members) < 2:
            raise ValueError(f"split {split!r} needs at least two windows for permutation")
        ordered = sorted(members, key=lambda row: str(row["window_id"]))
        features = np.asarray(
            [
                [float(row["translation_magnitude"]), float(row["rotation_magnitude"])]
                for row in ordered
            ],
            dtype=np.float64,
        )
        scales = np.std(features, axis=0)
        scales = np.where(scales == 0.0, 1.0, scales)
        normalized = features / scales
        differences = normalized[:, None, :] - normalized[None, :, :]
        cost = np.sum(np.square(differences), axis=2)
        np.fill_diagonal(cost, np.inf)
        source_indices, replacement_indices = linear_sum_assignment(cost)
        if np.any(source_indices == replacement_indices):
            raise RuntimeError(f"split {split!r} assignment contains a self-pair")
        for source_index, replacement_index in zip(
            source_indices.tolist(), replacement_indices.tolist(), strict=True
        ):
            source = str(ordered[source_index]["window_id"])
            replacement = str(ordered[replacement_index]["window_id"])
            pairs[source] = replacement
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(pairs, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return pairs


def _load_array(path: str | Path) -> Array:
    value = np.load(Path(path), allow_pickle=False)
    if isinstance(value, np.lib.npyio.NpzFile):
        if value.files != ["video"]:
            raise ValueError(f"NPZ {path} must contain exactly one array named 'video'")
        return value["video"]
    return value


def score_manifest(manifest_path: str | Path, *, include_lpips: bool) -> dict[str, Any]:
    manifest_path = Path(manifest_path)
    base = manifest_path.parent
    scored: list[dict[str, Any]] = []
    with manifest_path.open(encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    for record in records:
        truth_path = Path(record["truth_path"])
        prediction_path = Path(record["prediction_path"])
        truth = _load_array(truth_path if truth_path.is_absolute() else base / truth_path)
        prediction = _load_array(prediction_path if prediction_path.is_absolute() else base / prediction_path)
        result = evaluate_video_pair(truth, prediction, include_lpips=include_lpips)
        scored.append(
            {
                **record,
                "metrics": result["mean"],
                "last": result["last"],
                "horizons": result["horizons"],
                "temporal": result["temporal"],
                "reconstructed_i0": result["reconstructed_i0"],
            }
        )
    windows = collapse_sampling_seeds(scored)
    aggregate = aggregate_session_equal(windows)
    return {
        "protocol": "E1-future1..16-seed-window-session-equal-v1",
        "samples": scored,
        "windows": windows,
        "aggregate": aggregate,
    }


def _write_json(path: str | Path, value: Any) -> None:
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=True) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    score = subparsers.add_parser("score", help="score saved 17-frame RGB predictions")
    score.add_argument("--manifest", required=True, help="JSONL with window_id/raw_session/truth_path/prediction_path")
    score.add_argument("--output", required=True)
    score.add_argument("--lpips", action="store_true", help="require LPIPS alex/version=0.1")
    args = parser.parse_args(argv)
    if args.command == "score":
        _write_json(args.output, score_manifest(args.manifest, include_lpips=args.lpips))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
