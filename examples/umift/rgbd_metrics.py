"""Physical depth metrics for the clipped UMI-FT depth target; no GT-mask feedback."""
from __future__ import annotations

import math

import numpy as np


def depth_video_metrics(truth: np.ndarray, prediction: np.ndarray, *, exclude_first: bool = True) -> dict:
    """Frame-equal metrics on original GT support, including all raw prediction errors.

    The first item is the supplied observation, so the default scores only future
    frames. Empty zero/cap strata are reported as None, never fabricated as zero.
    The strict mask is a Zarr-only approximation: area-resize boundary mixtures
    cannot be identified without the original source validity footprints.
    """
    gt = np.asarray(truth, dtype=np.float64)
    pred = np.asarray(prediction, dtype=np.float64)
    if gt.shape != pred.shape or gt.ndim != 3:
        raise ValueError("truth and prediction must have matching THW shapes")
    if not np.isfinite(gt).all() or np.any(gt < 0) or np.any(gt > .5):
        raise ValueError("depth truth must be finite metres in [0,0.5]")
    if not np.isfinite(pred).all():
        raise ValueError("depth prediction must be finite; invalid values fail evaluation")
    first = 1 if exclude_first else 0
    if len(gt) <= first:
        raise ValueError("depth evaluation requires at least one future frame")
    rows = []
    for index in range(first, len(gt)):
        d, p = gt[index], pred[index]
        valid, zero, cap = (d > 0) & (d < .5), d == 0, d == .5
        if not valid.any():
            raise ValueError(f"depth truth frame {index} has no strict-interior metric support")
        error = p[valid] - d[valid]
        rows.append({
            "frame": index,
            "depth_mae_m": float(np.abs(error).mean()),
            "depth_rmse_m": float(np.sqrt(np.square(error).mean())),
            "depth_valid_fraction": float(valid.mean()),
            "depth_zero_fraction": float(zero.mean()),
            "depth_cap_fraction": float(cap.mean()),
            "depth_out_of_range_fraction": float(((p < 0) | (p > .5)).mean()),
            "depth_zero_region_mean_m": float(p[zero].mean()) if zero.any() else None,
            "depth_cap_underprediction_m": float(np.maximum(.5-p[cap], 0).mean()) if cap.any() else None,
        })
    means = {}
    for key in rows[0]:
        if key == "frame":
            continue
        values = [row[key] for row in rows if row[key] is not None]
        means[key] = float(np.mean(values)) if values else None
    return {"definition": "future-frame-equal; GT finite 0<d<0.5; raw prediction; metres",
            "mean": means, "frames": rows,
            "empty_stratum_frames": {k: sum(row[k] is None for row in rows) for k in
                                     ("depth_zero_region_mean_m", "depth_cap_underprediction_m")}}


def joint_selection_score(lpips: float, depth_mae_m: float, persistence_lpips: float,
                          persistence_depth_mae_m: float, *, rgb_weight: float) -> float:
    """Dimensionless score; the caller must freeze its weight before seeing candidates."""
    values = (lpips, depth_mae_m, persistence_lpips, persistence_depth_mae_m, rgb_weight)
    if not all(math.isfinite(float(x)) for x in values):
        raise ValueError("selection components must be finite")
    if min(lpips, depth_mae_m) < 0 or min(persistence_lpips, persistence_depth_mae_m) <= 1e-12:
        raise ValueError("errors must be nonnegative and persistence denominators positive")
    if not 0 <= rgb_weight <= 1:
        raise ValueError("rgb_weight must be in [0,1]")
    return float(rgb_weight*lpips/persistence_lpips
                 + (1-rgb_weight)*depth_mae_m/persistence_depth_mae_m)
