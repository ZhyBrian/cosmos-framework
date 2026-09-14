# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from __future__ import annotations

import unittest

import torch

from cosmos_framework.algorithm.loss.umift_depth_aux import (
    compute_depth_aux_loss,
    reconstruct_clean_latent,
)


class _FullCanvasDecoder(torch.nn.Module):
    """Small differentiable stand-in for the frozen full Wan decoder."""

    def __init__(self) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.0), requires_grad=False)
        self.input_shapes: list[tuple[int, ...]] = []

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        self.input_shapes.append(tuple(latent.shape))
        # Depend on every latent timestep, then emulate 5 history + 16 future frames.
        per_latent = latent.mean(dim=(1, 3, 4)) * self.scale  # [B,6]
        frame_values = torch.cat(
            [per_latent[:, :1].expand(-1, 5), per_latent[:, 1:].mean(dim=1, keepdim=True).expand(-1, 16)],
            dim=1,
        )
        return frame_values[:, None, :, None, None].expand(-1, 3, -1, 2, 512)


class _BFloat16CanvasDecoder(torch.nn.Module):
    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        value = latent.mean().to(torch.bfloat16)
        return value.expand(latent.shape[0], 3, 21, 1, 512)


class DepthAuxTest(unittest.TestCase):
    def test_clip_level_sigma_broadcast(self):
        x0 = torch.ones(1, 2, 6, 1, 1)
        noise = torch.zeros_like(x0)
        mask = torch.tensor([1, 1, 0, 0, 0, 0]).view(6, 1, 1)
        xt = x0 * 0.7
        pred = noise - x0
        result = reconstruct_clean_latent(x0, xt, pred, torch.tensor([0.3]), mask)
        torch.testing.assert_close(result, x0)

    def test_reconstruction_recovers_clean_and_refills_history(self) -> None:
        x0 = torch.arange(6.0).view(1, 6, 1, 1).expand(2, -1, 2, 2)
        noise = x0 + 3.0
        sigma = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.9])
        xt = (1.0 - sigma[None, :, None, None]) * x0 + sigma[None, :, None, None] * noise
        target_velocity = noise - x0
        pred = target_velocity.clone()
        pred[:, 0] = 99.0  # History prediction must be ignored.
        history_mask = torch.tensor([1, 0, 0, 0, 0, 0], dtype=torch.bool)[:, None, None]

        restored = reconstruct_clean_latent(x0, xt, pred, sigma, history_mask)

        torch.testing.assert_close(restored, x0)

    def test_loss_is_frame_equal_weighted_masks_strict_bounds_and_keeps_prediction_range(self) -> None:
        decoder = _FullCanvasDecoder()
        x0 = torch.zeros(1, 6, 1, 1)
        xt = torch.zeros_like(x0)
        pred = torch.full_like(x0, -8.0, requires_grad=True)
        sigma = torch.tensor([0.0, 0.5, 0.5, 0.5, 0.5, 0.5])
        history_mask = torch.tensor([1, 0, 0, 0, 0, 0], dtype=torch.bool)[:, None, None]
        # Decoder future value is 4, hence predicted depth is (4+1)/4 = 1.25 m.
        depth_m = torch.full((1, 21, 2, 256), 0.25)
        depth_m[:, 5, 0, 0] = 0.0
        depth_m[:, 5, 0, 1] = 0.5
        depth_m[:, 5, 1, 0] = 0.49
        strict_mask = torch.ones_like(depth_m, dtype=torch.bool)
        strict_mask[:, 5] = False
        strict_mask[:, 5, 0, :2] = True
        strict_mask[:, 5, 1, :2] = True

        result = compute_depth_aux_loss(
            x0=x0,
            xt=xt,
            pred=pred,
            sigma=sigma,
            condition_mask=history_mask,
            decoder=decoder,
            depth_m=depth_m,
            depth_metric_mask=strict_mask,
        )

        expected_first = ((1.25 - 0.49) + (1.25 - 0.25)) / 2
        expected_other = 1.0
        self.assertAlmostEqual(result.loss.item(), (expected_first + 15 * expected_other) / 16 / 0.5, places=6)
        self.assertEqual(decoder.input_shapes, [(1, 1, 6, 1, 1)])
        self.assertFalse(result.has_empty_support.item())
        result.loss.backward()
        self.assertEqual(pred.grad[:, 0].abs().sum().item(), 0.0)
        self.assertGreater(pred.grad[:, 1:].abs().sum().item(), 0.0)
        self.assertIsNone(decoder.scale.grad)

    def test_empty_future_frame_is_reported_or_raised(self) -> None:
        decoder = _FullCanvasDecoder()
        common = dict(
            x0=torch.zeros(1, 6, 1, 1),
            xt=torch.zeros(1, 6, 1, 1),
            pred=torch.zeros(1, 6, 1, 1),
            sigma=torch.ones(6),
            condition_mask=torch.tensor([1, 0, 0, 0, 0, 0])[:, None, None],
            decoder=decoder,
            depth_m=torch.full((1, 21, 2, 256), 0.25),
            depth_metric_mask=torch.ones(1, 21, 2, 256, dtype=torch.bool),
        )
        common["depth_metric_mask"][:, 7] = False

        with self.assertRaisesRegex(ValueError, r"batch=0, future_frame=2"):
            compute_depth_aux_loss(**common)

        result = compute_depth_aux_loss(**common, error_on_empty_support=False)
        self.assertTrue(result.has_empty_support.item())
        self.assertEqual(result.empty_support_indices.tolist(), [[0, 2]])

    def test_bfloat16_decoder_keeps_metric_depth_loss_in_float32(self) -> None:
        result = compute_depth_aux_loss(
            x0=torch.zeros(1, 6, 1, 1),
            xt=torch.zeros(1, 6, 1, 1),
            pred=torch.zeros(1, 6, 1, 1),
            sigma=torch.ones(6),
            # This is the exact unbatched mask shape emitted by PackedSequence.
            condition_mask=torch.tensor([1, 0, 0, 0, 0, 0])[:, None, None],
            decoder=_BFloat16CanvasDecoder(),
            depth_m=torch.full((21, 1, 256), 0.123456, dtype=torch.float32),
            depth_metric_mask=torch.ones(21, 1, 256, dtype=torch.bool),
        )

        self.assertEqual(result.loss.dtype, torch.float32)
        self.assertEqual(result.per_frame_loss_m.dtype, torch.float32)


if __name__ == "__main__":
    unittest.main()
