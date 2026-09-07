import copy

import numpy as np
import pytest

from examples.umift.protocol import derive_noise_seed
from examples.umift.render_comparison import (
    METHODS,
    select_middle_windows,
    timeline,
    validate_arrays,
    validate_rows,
)


def test_selection_uses_time_middle_per_session_without_metric_or_visual_input():
    rows = [{"raw_session": session, "window_id": f"{session}:{i}", "timestamp_start": i,
             "split": "history"} for session in "abc" for i in range(4)]
    assert [r["window_id"] for r in select_middle_windows(rows[::-1])] == ["a:2", "b:2", "c:2"]


def test_playback_preserves_frame_order_and_slow_motion_only_repeats_frames():
    assert [k for k, _ in timeline(False)] == list(range(17))
    assert [k for k, _ in timeline(True)] == list(range(17)) * 3 + list(np.repeat(range(17), 4)) * 2
    assert len(timeline(True)) == 187


def test_different_decoded_i0_is_preserved_but_wrong_persistence_or_truth_is_rejected():
    truth = np.zeros((17, 4, 4, 3), dtype=np.float32)
    prediction = truth.copy()
    prediction[0] = 0.1
    predictions = [truth.copy()] + [prediction.copy() for _ in range(4)]
    validate_arrays([truth] * 5, predictions)
    changed = truth.copy()
    changed[16] = 0.2
    with pytest.raises(ValueError, match="truth arrays"):
        validate_arrays([truth] * 4 + [changed], predictions)
    with pytest.raises(ValueError, match="Persistence"):
        validate_arrays([truth] * 5, [prediction] + predictions[1:])
    predictions[4][0] = 0.2
    with pytest.raises(ValueError, match="condition reconstructions"):
        validate_arrays([truth] * 5, predictions)


@pytest.mark.parametrize("method,key,value", [
    ("E1-A", "checkpoint_id", "wrong-checkpoint"),
    ("E1-S", "noise_seed", 7),
    ("B0", "window_id", "different-window"),
    ("E1-Z", "sampling_seed", 1),
    ("E1-A", "raw_session", "different-session"),
])
def test_misaligned_comparisons_fail_before_rendering(method, key, value):
    frozen = {"window_id": "ep:0", "raw_session": "session", "source_id": "session#seg0"}
    rows = {m: {"window_id": "ep:0", "raw_session": "session", "episode": "session#seg0",
                "method": m, "split": "history", "sampling_seed": 0,
                "noise_seed": derive_noise_seed("ep:0", 0), "checkpoint_id": "best"} for m in METHODS}
    rows["B-Persistence"]["checkpoint_id"] = None
    rows["B0"]["checkpoint_id"] = "/data/cosmos_models/Cosmos3-Edge-DCP-a9d944e/model"
    validate_rows(rows, frozen, "best")
    broken = copy.deepcopy(rows)
    broken[method][key] = value
    with pytest.raises(ValueError):
        validate_rows(broken, frozen, "best")
