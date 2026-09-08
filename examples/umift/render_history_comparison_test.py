from pathlib import Path

import numpy as np
import pytest

from examples.umift.render_comparison import BOXES
from examples.umift.render_history_comparison import LABELS, METHODS, _encode, _verify_video


def test_encode_preserves_six_panels_and_true_pts(tmp_path: Path) -> None:
    pytest.importorskip("av")
    assert len(BOXES) == len(LABELS) == 6
    assert LABELS == (
        "真实视频", "首帧保持", "基础 Edge", "E2-H 真实动作", "E2-H 零动作", "E2-H 打乱动作",
    )
    elapsed = np.array([0.0, 0.071, 0.139], dtype=np.float64)
    truth = np.empty((3, 256, 256, 3), dtype=np.uint8)
    truth[0], truth[1], truth[2] = 15, 75, 135
    predictions = {}
    for method_index, method in enumerate(METHODS):
        value = np.empty((3, 256, 256, 3), dtype=np.float32)
        for frame_index in range(3):
            value[frame_index] = (35 + method_index * 40 + frame_index * 7) / 255.0
        predictions[method] = value
    episode = {"episode_id": 13, "start_percent": 0, "frame_count": 3,
               "selected_iteration": 1500, "history_padding_count": 4,
               "initial_episode_elapsed_seconds": 2.5}
    path = tmp_path / "history.mp4"

    _encode(path, truth, predictions, episode, elapsed, history_frames=17)
    verification = _verify_video(path, truth, predictions, episode, elapsed)

    assert verification["frame_count"] == 3
    assert verification["frame_pts_seconds"] == elapsed.tolist()
    assert verification["max_pts_error_seconds"] <= 0.001
    assert verification["max_panel_mae_0_255"] <= 3.0
    assert verification["last_frame_duration_seconds"] == pytest.approx(0.068, abs=0.001)
