from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


_PATH = Path(__file__).with_name("history_rollout.py")
_SPEC = importlib.util.spec_from_file_location("history_rollout_under_test", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def _frame(value: int) -> np.ndarray:
    return np.full((4, 4, 3), value, dtype=np.uint8)


def _chunks(steps=(16, 16)) -> list[dict]:
    result = []
    output_start = 0
    for index, count in enumerate(steps):
        result.append({"index": index, "output_start": output_start, "steps": count, "noise_seed": index})
        output_start += count
    return result


def test_h1_matches_single_condition_rollout_contract() -> None:
    initial = _frame(7)[None]
    output = np.empty((33, 4, 4, 3), dtype=np.float32)

    def predict(history, chunk):
        values = [_frame(int(history[-1, 0, 0, 0]))]
        values.extend(_frame(20 + chunk["index"] * 16 + i) for i in range(16))
        return np.stack(values).astype(np.float32) / 255

    records = _MODULE.rollout(initial, _chunks(), predict, output)

    assert np.array_equal(output[0], initial[0].astype(np.float32) / 255)
    assert [int(output[i, 0, 0, 0] * 255) for i in range(1, 33)] == list(range(20, 52))
    assert records[1]["history_source"] == "generated_rolling_history"


def test_h17_second_chunk_keeps_anchor_plus_sixteen_generated_frames() -> None:
    initial = np.stack([_frame(i) for i in range(17)])
    seen: list[np.ndarray] = []
    output = np.empty((33, 4, 4, 3), dtype=np.float32)

    def predict(history, chunk):
        seen.append(history.copy())
        generated = np.stack([_frame(100 + chunk["index"] * 16 + i) for i in range(16)])
        return np.concatenate((history, generated), axis=0).astype(np.float32) / 255

    _MODULE.rollout(initial, _chunks(), predict, output)

    assert np.array_equal(seen[0], initial)
    assert int(seen[1][0, 0, 0, 0]) == 16
    assert [int(frame[0, 0, 0]) for frame in seen[1][1:]] == list(range(100, 116))
    assert _MODULE.rollout(initial, _chunks(), predict, np.empty_like(output))[1][
        "initial_observations_remaining"
    ] == 1


def test_tail_retains_only_real_steps() -> None:
    initial = np.stack([_frame(i) for i in range(5)])
    output = np.empty((20, 4, 4, 3), dtype=np.float32)

    def predict(history, chunk):
        generated = np.stack([_frame(50 + chunk["index"] * 20 + i) for i in range(16)])
        return np.concatenate((history, generated), axis=0).astype(np.float32) / 255

    records = _MODULE.rollout(initial, _chunks((16, 3)), predict, output)

    assert output.shape[0] == 20
    assert [int(output[i, 0, 0, 0] * 255) for i in range(17, 20)] == [70, 71, 72]
    assert records[-1]["steps"] == 3


def test_predictor_cannot_mutate_retained_history_prefix() -> None:
    initial = np.stack([_frame(i) for i in range(9)])
    original = initial.copy()
    output = np.empty((17, 4, 4, 3), dtype=np.float32)

    def predict(history, chunk):
        prediction = np.concatenate((history.copy(), np.stack([_frame(80 + i) for i in range(16)])))
        history[:] = 255
        return prediction.astype(np.float32) / 255

    records = _MODULE.rollout(initial, _chunks((16,)), predict, output)

    assert np.array_equal(initial, original)
    assert records[0]["history_sha256"] == _MODULE.array_sha(original)


def test_initial_history_clips_before_episode_start_and_marks_padding() -> None:
    sampled = np.stack([_frame(i) for i in range(10)])

    history, indices, mask = _MODULE.select_initial_history(sampled, anchor_source_index=4, history_frames=5)

    assert indices.tolist() == [0, 0, 0, 2, 4]
    assert mask.tolist() == [False, False, True, True, True]
    assert [int(frame[0, 0, 0]) for frame in history] == [0, 0, 0, 2, 4]


def test_rollout_rejects_bad_chunk_index_and_unretained_bad_prediction() -> None:
    initial = _frame(0)[None]
    output = np.empty((17, 4, 4, 3), dtype=np.float32)
    bad_index = [{"index": 2, "output_start": 0, "steps": 16, "noise_seed": 0}]

    try:
        _MODULE.rollout(initial, bad_index, lambda h, c: np.zeros((17, 4, 4, 3)), output)
    except ValueError as error:
        assert "index" in str(error)
    else:
        raise AssertionError("non-contiguous chunk index was accepted")

    def bad_tail(history, chunk):
        prediction = np.zeros((17, 4, 4, 3), dtype=np.float32)
        prediction[-1] = np.nan
        return prediction

    try:
        _MODULE.rollout(initial, _chunks((3,)), bad_tail, np.empty((4, 4, 4, 3), dtype=np.float32))
    except ValueError as error:
        assert "prediction" in str(error)
    else:
        raise AssertionError("invalid unretained prediction tail was accepted")
