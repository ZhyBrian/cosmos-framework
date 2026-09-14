# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Minimal metric-depth auxiliary loss for the UMI-FT D1 experiment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch


@dataclass(frozen=True)
class DepthAuxLossResult:
    """Loss plus support metadata needed for a synchronized distributed check.

    When ``error_on_empty_support=False``, callers must all-reduce
    ``has_empty_support`` before backward and stop every rank together if any
    rank reports true. ``empty_support_indices`` uses future-relative frame
    indices in ``[0, 15]``.
    """

    loss: torch.Tensor
    per_frame_loss_m: torch.Tensor
    valid_counts: torch.Tensor
    has_empty_support: torch.Tensor
    empty_support_indices: torch.Tensor


def _as_batched_latent(tensor: torch.Tensor, name: str) -> tuple[torch.Tensor, bool]:
    if tensor.ndim == 4:  # [C,T,H,W]
        return tensor.unsqueeze(0), True
    if tensor.ndim == 5:  # [B,C,T,H,W]
        return tensor, False
    raise ValueError(f"{name} must have shape [C,T,H,W] or [B,C,T,H,W], got {tuple(tensor.shape)}")


def _broadcast_sigma(sigma: torch.Tensor, batch: int, timesteps: int) -> torch.Tensor:
    if sigma.ndim == 1 and sigma.shape[0] == timesteps:
        return sigma.view(1, 1, timesteps, 1, 1)
    if sigma.ndim == 2 and sigma.shape == (batch, timesteps):
        return sigma.view(batch, 1, timesteps, 1, 1)
    if sigma.ndim == 5 and sigma.shape[1:] == (1, timesteps, 1, 1) and sigma.shape[0] in (1, batch):
        return sigma
    raise ValueError(
        f"sigma must have shape [T], [B,T], or [B,1,T,1,1] for B={batch}, T={timesteps}; "
        f"got {tuple(sigma.shape)}"
    )


def _broadcast_condition_mask(mask: torch.Tensor, batch: int, timesteps: int) -> torch.Tensor:
    if mask.ndim == 3 and mask.shape == (timesteps, 1, 1):
        return mask.view(1, 1, timesteps, 1, 1)
    if mask.ndim == 4 and mask.shape[1:] == (timesteps, 1, 1) and mask.shape[0] in (1, batch):
        return mask.unsqueeze(1)
    if mask.ndim == 5 and mask.shape[1:] == (1, timesteps, 1, 1) and mask.shape[0] in (1, batch):
        return mask
    raise ValueError(
        "condition_mask must have shape [T,1,1], [B,T,1,1], or [B,1,T,1,1]; "
        f"got {tuple(mask.shape)}"
    )


def reconstruct_clean_latent(
    x0: torch.Tensor,
    xt: torch.Tensor,
    pred: torch.Tensor,
    sigma: torch.Tensor,
    condition_mask: torch.Tensor,
) -> torch.Tensor:
    """Recover clean latent and exactly refill conditioned history from ``x0``.

    Latents may be unbatched ``[C,T,H,W]`` (the E3 sample contract is
    ``[48,6,16,32]``) or batched ``[B,C,T,H,W]``. ``condition_mask`` uses
    1 for clean history and 0 for generated latent timesteps.
    """

    x0_b, unbatched = _as_batched_latent(x0, "x0")
    xt_b, xt_unbatched = _as_batched_latent(xt, "xt")
    pred_b, pred_unbatched = _as_batched_latent(pred, "pred")
    if xt_unbatched != unbatched or pred_unbatched != unbatched or x0_b.shape != xt_b.shape or x0_b.shape != pred_b.shape:
        raise ValueError(
            f"x0, xt, and pred must have identical rank and shape; got {tuple(x0.shape)}, "
            f"{tuple(xt.shape)}, {tuple(pred.shape)}"
        )

    batch, _, timesteps, _, _ = x0_b.shape
    sigma_b = _broadcast_sigma(sigma, batch, timesteps).to(device=pred.device, dtype=pred.dtype)
    mask = _broadcast_condition_mask(condition_mask, batch, timesteps).to(device=pred.device, dtype=pred.dtype)
    sigma_eff = (1.0 - mask) * sigma_b
    restored = mask * x0_b + (1.0 - mask) * (xt_b - sigma_eff * pred_b)
    return restored.squeeze(0) if unbatched else restored


def _as_batched_depth(tensor: torch.Tensor, name: str) -> torch.Tensor:
    if tensor.ndim == 3:  # [T,H,W]
        return tensor.unsqueeze(0)
    if tensor.ndim == 4:  # [B,T,H,W]
        return tensor
    raise ValueError(f"{name} must have shape [T,H,W] or [B,T,H,W], got {tuple(tensor.shape)}")


def compute_depth_aux_loss(
    *,
    x0: torch.Tensor,
    xt: torch.Tensor,
    pred: torch.Tensor,
    sigma: torch.Tensor,
    condition_mask: torch.Tensor,
    decoder: Callable[[torch.Tensor], torch.Tensor],
    depth_m: torch.Tensor,
    depth_metric_mask: torch.Tensor,
    error_on_empty_support: bool = True,
) -> DepthAuxLossResult:
    """Decode full latent context and compute normalized future metric-depth L1.

    The decoder must accept ``[B,C,6,H,W]`` and return an RGB canvas
    ``[B,3,T,H,512]`` with at least 21 frames. Its parameters are expected to
    be frozen, while its forward must run with autograd enabled for latent
    input gradients. Predicted depth is not clipped.

    Set ``error_on_empty_support=False`` under distributed training, all-reduce
    the returned scalar ``has_empty_support`` unconditionally on every rank,
    and enter backward only if the global flag is false.
    """

    clean = reconstruct_clean_latent(x0, xt, pred, sigma, condition_mask)
    clean_b = clean.unsqueeze(0) if clean.ndim == 4 else clean
    if clean_b.shape[2] != 6:
        raise ValueError(f"D1 expects exactly 6 latent timesteps, got {clean_b.shape[2]}")

    canvas = decoder(clean_b)
    if not isinstance(canvas, torch.Tensor) or canvas.ndim != 5:
        raise ValueError("decoder must return a tensor with shape [B,C,T,H,W]")
    if canvas.shape[0] != clean_b.shape[0] or canvas.shape[1] != 3 or canvas.shape[2] < 21 or canvas.shape[-1] != 512:
        raise ValueError(
            "decoder output must have shape [B,3,T,H,512] with T>=21; "
            f"got {tuple(canvas.shape)}"
        )

    # Metric supervision stays in FP32 even when the frozen decoder runs in BF16.
    predicted_depth_m = (canvas[:, :, 5:21, :, 256:].float().mean(dim=1) + 1.0) / 4.0
    target = _as_batched_depth(depth_m, "depth_m").to(device=pred.device, dtype=predicted_depth_m.dtype)
    supplied_mask = _as_batched_depth(depth_metric_mask, "depth_metric_mask").to(device=pred.device, dtype=torch.bool)
    if target.shape[1] == 21:
        target = target[:, 5:21]
        supplied_mask = supplied_mask[:, 5:21]
    elif target.shape[1] != 16:
        raise ValueError(f"depth_m time dimension must be 21 or 16, got {target.shape[1]}")
    if target.shape != predicted_depth_m.shape or supplied_mask.shape != target.shape:
        raise ValueError(
            f"decoded depth, depth_m, and depth_metric_mask must align; got {tuple(predicted_depth_m.shape)}, "
            f"{tuple(target.shape)}, {tuple(supplied_mask.shape)}"
        )

    valid = supplied_mask & (target > 0.0) & (target < 0.5)
    valid_counts = valid.sum(dim=(-2, -1))
    empty_indices = torch.nonzero(valid_counts == 0, as_tuple=False)
    has_empty = torch.any(valid_counts == 0)
    if error_on_empty_support and bool(has_empty.item()):
        locations = ", ".join(f"batch={b}, future_frame={t}" for b, t in empty_indices.tolist())
        raise ValueError(f"D1 depth supervision has empty valid support at {locations}")

    absolute_error = (predicted_depth_m - target).abs()
    per_frame_loss_m = (absolute_error * valid).sum(dim=(-2, -1)) / valid_counts.clamp_min(1)
    loss = per_frame_loss_m.mean() / 0.5
    return DepthAuxLossResult(
        loss=loss,
        per_frame_loss_m=per_frame_loss_m,
        valid_counts=valid_counts,
        has_empty_support=has_empty,
        empty_support_indices=empty_indices,
    )


__all__ = ["DepthAuxLossResult", "compute_depth_aux_loss", "reconstruct_clean_latent"]
