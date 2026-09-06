from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from examples.umift.prepare_evaluation import (
    OUTPUT_NAMES,
    action_magnitudes,
    select_time_previews,
    validate_empty_output,
)


def test_action_magnitudes_use_column_rot6d_and_sum_step_lengths() -> None:
    action = np.zeros((2, 10), dtype=np.float32)
    action[:, :3] = [[3.0, 4.0, 0.0], [0.0, 0.0, 12.0]]
    identity = np.eye(3)
    quarter_turn = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    action[0, 3:9] = identity[:, :2].T.reshape(-1)
    action[1, 3:9] = quarter_turn[:, :2].T.reshape(-1)

    translation, rotation = action_magnitudes(action)

    assert translation == pytest.approx(17.0)
    assert rotation == pytest.approx(np.pi / 2)


def test_previews_follow_actual_timestamps_not_window_id_order() -> None:
    rows = [
        {"window_id": "episode_9:s=0", "raw_session": "s", "timestamp_start": 0.0},
        {"window_id": "episode_1:s=0", "raw_session": "s", "timestamp_start": 10.0},
        {"window_id": "episode_8:s=0", "raw_session": "s", "timestamp_start": 20.0},
        {"window_id": "episode_2:s=0", "raw_session": "s", "timestamp_start": 30.0},
    ]

    result = select_time_previews(rows, count=4)

    assert result["window_ids"] == [
        "episode_9:s=0",
        "episode_1:s=0",
        "episode_8:s=0",
        "episode_2:s=0",
    ]


def test_existing_output_refuses_before_writing(tmp_path: Path) -> None:
    existing = tmp_path / OUTPUT_NAMES[2]
    existing.write_text("frozen\n")

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        validate_empty_output(tmp_path)

    assert existing.read_text() == "frozen\n"
    assert not (tmp_path / OUTPUT_NAMES[0]).exists()
