from __future__ import annotations

import pytest
import torch

from cosmos_framework.data.generator.sequence_packing import (
    SequencePlan,
    build_sequence_plans_from_data_batch,
)
from cosmos_framework.data.generator.sequence_packing.sequence import PackedSequenceBuilder


@pytest.mark.parametrize("history_frames", [1, 5, 9, 17])
def test_history_plan_drives_vision_loss_mask_and_action_mrope_offset(history_frames: int) -> None:
    condition_latents = (history_frames - 1) // 4 + 1
    latent_frames = condition_latents + 4
    plan = SequencePlan(
        has_text=True,
        has_vision=True,
        condition_frame_indexes_vision=list(range(condition_latents)),
        has_action=True,
        condition_frame_indexes_action=list(range(16)),
        action_start_frame_offset=history_frames,
    )
    assert build_sequence_plans_from_data_batch(
        {"video": [object()], "action": [object()], "sequence_plan": [plan]},
        input_video_key="video",
        input_image_key="images",
    ) == [plan]

    builder = PackedSequenceBuilder()
    builder.begin_sample(0)
    builder.pack_vision_tokens(
        input_vision_tokens=torch.zeros(1, 16, latent_frames, 2, 2),
        condition_frame_indexes_vision=plan.condition_frame_indexes_vision,
        input_timestep=0.5,
        latent_patch_size=1,
        vision_fps=15.0,
        enable_fps_modulation=True,
        base_fps=24.0,
        temporal_compression_factor=4,
        vision_temporal_positions=None,
        temporal_position_period=None,
    )
    vision_start_temporal_offset = 0
    builder.pack_action_tokens(
        input_action_tokens=torch.zeros(16, 64),
        condition_frame_indexes_action=plan.condition_frame_indexes_action,
        input_timestep=0.5,
        action_temporal_offset=vision_start_temporal_offset,
        enable_fps_modulation=True,
        base_fps=24.0,
        action_fps=15.0,
        base_temporal_compression_factor=4,
        action_start_frame_offset=plan.action_start_frame_offset,
    )

    assert builder.vision is not None
    assert builder.action is not None
    torch.testing.assert_close(
        builder.vision.condition_mask[0].flatten(),
        torch.tensor([1.0] * condition_latents + [0.0] * 4),
    )
    assert len(builder.vision.mse_loss_indexes) == 4 * 2 * 2
    assert builder.action.condition_mask[0].flatten().tolist() == [1.0] * 16
    assert builder.action.mse_loss_indexes == []

    all_positions = torch.cat(builder.position_ids, dim=1)
    expected_action_positions = [0.4 * frame for frame in range(history_frames, history_frames + 16)]
    assert all_positions[0, -16:].tolist() == pytest.approx(expected_action_positions)
