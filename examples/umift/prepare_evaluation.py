#!/usr/bin/env python3
"""Prepare frozen E1 evaluation rosters and action-permutation controls on CPU."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np


OUTPUT_NAMES = (
    "dev_windows.jsonl",
    "history_windows.jsonl",
    "dev_preview20.json",
    "dev_S_pairs.json",
    "history_S_pairs.json",
    "S_pair_diagnostics.json",
)


def validate_empty_output(output_dir: Path) -> None:
    existing = [output_dir / name for name in OUTPUT_NAMES if (output_dir / name).exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite existing evaluation artifacts: {existing}")


def rotation_angles_from_column_6d(rot6d: np.ndarray) -> np.ndarray:
    value = np.asarray(rot6d, dtype=np.float64)
    if value.ndim != 2 or value.shape[1] != 6:
        raise ValueError(f"rotation 6D values must have shape (T, 6), got {value.shape}")
    first = value[:, :3]
    second = value[:, 3:]
    first_norm = np.linalg.norm(first, axis=1, keepdims=True)
    if np.any(first_norm < 1e-12):
        raise ValueError("rotation 6D first column has zero norm")
    basis_1 = first / first_norm
    orthogonal = second - np.sum(basis_1 * second, axis=1, keepdims=True) * basis_1
    second_norm = np.linalg.norm(orthogonal, axis=1, keepdims=True)
    if np.any(second_norm < 1e-12):
        raise ValueError("rotation 6D columns are collinear")
    basis_2 = orthogonal / second_norm
    basis_3 = np.cross(basis_1, basis_2)
    matrices = np.stack((basis_1, basis_2, basis_3), axis=-1)
    cosine = np.clip((np.trace(matrices, axis1=1, axis2=2) - 1.0) / 2.0, -1.0, 1.0)
    return np.arccos(cosine)


def action_magnitudes(physical_action: np.ndarray) -> tuple[float, float]:
    action = np.asarray(physical_action, dtype=np.float64)
    if action.ndim != 2 or action.shape[1] != 10:
        raise ValueError(f"physical action must have shape (T, 10), got {action.shape}")
    translation = float(np.linalg.norm(action[:, :3], axis=1).sum())
    rotation = float(rotation_angles_from_column_6d(action[:, 3:9]).sum())
    return translation, rotation


def window_record(sample: dict[str, Any], split: str) -> dict[str, Any]:
    timestamps = np.asarray(sample["timestamps"], dtype=np.float64)
    translation, rotation = action_magnitudes(np.asarray(sample["physical_action"]))
    episode_id = int(sample["episode_id"])
    window_start = int(sample["window_start"])
    return {
        "window_id": f"episode_{episode_id}:s={window_start}",
        "split": split,
        "episode_id": episode_id,
        "window_start": window_start,
        "source_id": str(sample["source_id"]),
        "raw_session": str(sample["session_id"]),
        "timestamp_start": float(timestamps[0]),
        "timestamp_end": float(timestamps[-1]),
        "translation_magnitude": translation,
        "rotation_magnitude": rotation,
    }


def select_time_previews(rows: Iterable[dict[str, Any]], *, count: int = 4) -> dict[str, Any]:
    by_session: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_session[str(row["raw_session"])].append(row)
    sessions: dict[str, list[dict[str, Any]]] = {}
    for session, members in sorted(by_session.items()):
        ordered = sorted(members, key=lambda row: (float(row["timestamp_start"]), str(row["window_id"])))
        if len(ordered) < count:
            raise ValueError(f"session {session!r} has only {len(ordered)} windows, need {count}")
        targets = np.linspace(ordered[0]["timestamp_start"], ordered[-1]["timestamp_start"], count)
        selected: list[dict[str, Any]] = []
        used: set[str] = set()
        for target in targets:
            candidates = [row for row in ordered if str(row["window_id"]) not in used]
            chosen = min(
                candidates,
                key=lambda row: (
                    abs(float(row["timestamp_start"]) - float(target)),
                    float(row["timestamp_start"]),
                    str(row["window_id"]),
                ),
            )
            used.add(str(chosen["window_id"]))
            selected.append(
                {
                    "window_id": str(chosen["window_id"]),
                    "timestamp_start": float(chosen["timestamp_start"]),
                    "target_timestamp": float(target),
                    "absolute_time_error": abs(float(chosen["timestamp_start"]) - float(target)),
                }
            )
        sessions[session] = selected
    window_ids = [item["window_id"] for session in sorted(sessions) for item in sessions[session]]
    return {
        "method": "nearest unique windows to equally spaced actual start timestamps within each raw session",
        "sessions": sessions,
        "window_ids": window_ids,
    }


def pair_diagnostics(rows: list[dict[str, Any]], pairs: dict[str, str]) -> dict[str, Any]:
    by_id = {str(row["window_id"]): row for row in rows}
    features = np.asarray(
        [
            [float(row["translation_magnitude"]), float(row["rotation_magnitude"])]
            for row in rows
        ],
        dtype=np.float64,
    )
    scales = np.std(features, axis=0)
    scales = np.where(scales == 0.0, 1.0, scales)
    records = []
    for source, replacement in sorted(pairs.items()):
        left, right = by_id[source], by_id[replacement]
        records.append(
            {
                "source": source,
                "replacement": replacement,
                "translation_abs_difference": abs(
                    float(left["translation_magnitude"]) - float(right["translation_magnitude"])
                ),
                "rotation_abs_difference": abs(
                    float(left["rotation_magnitude"]) - float(right["rotation_magnitude"])
                ),
            }
        )
    diagnostics: dict[str, Any] = {
        "algorithm": "linear_sum_assignment with squared Euclidean cost and forbidden diagonal",
        "features": ["translation_magnitude", "rotation_magnitude"],
        "normalization_scales": {
            "translation_standard_deviation_or_one": float(scales[0]),
            "rotation_standard_deviation_or_one": float(scales[1]),
        },
        "pair_count": len(records),
    }
    warnings = []
    for label in ("translation", "rotation"):
        field = f"{label}_abs_difference"
        values = [float(row[field]) for row in records]
        median = float(np.median(values))
        worst_pair = max(records, key=lambda row: (float(row[field]), row["source"], row["replacement"]))
        diagnostics[f"{label}_difference"] = {
            "median": median,
            "p95": float(np.percentile(values, 95)),
            "max": float(worst_pair[field]),
            "worst_pair": worst_pair,
        }
        if float(worst_pair[field]) > 5.0 * max(median, 1e-12):
            warnings.append(f"worst-pair {field} exceeds 5x the median")
    diagnostics["heuristic_warnings"] = warnings
    diagnostics["heuristic_is_acceptance_gate"] = False
    return diagnostics


def prepare(zarr_path: Path, output_dir: Path) -> dict[str, Any]:
    from cosmos_framework.data.generator.action.datasets.umift_zarr_dataset import UMIFTZarrIterableDataset
    from examples.umift.evaluate import build_action_permutation

    validate_empty_output(output_dir)
    rows_by_split: dict[str, list[dict[str, Any]]] = {}
    for split in ("dev", "history"):
        dataset = UMIFTZarrIterableDataset(
            str(zarr_path), split=split, stage="e1", seed=42, resolution="256", fps=15.0, transform=None
        )
        rows_by_split[split] = [window_record(sample, split) for sample in dataset]
    expected = {"dev": 329, "history": 196}
    actual = {split: len(rows) for split, rows in rows_by_split.items()}
    if actual != expected:
        raise ValueError(f"evaluation roster count mismatch: expected={expected}, actual={actual}")
    preview = select_time_previews(rows_by_split["dev"])
    if len(preview["window_ids"]) != 20 or len(set(preview["window_ids"])) != 20:
        raise ValueError("dev preview roster must contain exactly 20 unique window IDs")
    all_pairs = build_action_permutation(rows_by_split["dev"] + rows_by_split["history"])
    pairs_by_split = {
        split: {row["window_id"]: all_pairs[row["window_id"]] for row in rows}
        for split, rows in rows_by_split.items()
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    for split, rows in rows_by_split.items():
        payload = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
        (output_dir / f"{split}_windows.jsonl").write_text(payload, encoding="utf-8")
    (output_dir / "dev_preview20.json").write_text(
        json.dumps(preview, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    for split, pairs in pairs_by_split.items():
        (output_dir / f"{split}_S_pairs.json").write_text(
            json.dumps(pairs, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    diagnostics = {
        split: pair_diagnostics(rows_by_split[split], pairs_by_split[split])
        for split in ("dev", "history")
    }
    (output_dir / "S_pair_diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {"window_counts": actual, "preview_count": 20, "pair_diagnostics": diagnostics}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zarr", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(prepare(args.zarr, args.output_dir), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
