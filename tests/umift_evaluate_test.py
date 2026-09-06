import json
import math
from pathlib import Path

import numpy as np
import pytest

from examples.umift.evaluate import (
    aggregate_session_equal,
    build_action_permutation,
    collapse_sampling_seeds,
    derive_noise_seed,
    evaluate_video_pair,
    lpips_frames,
    make_physical_zero_action,
    persistence_prediction,
    prepare_model_actions,
    select_session_previews,
    sparse_window_starts,
)


def _video(values: list[float], size: int = 16) -> np.ndarray:
    return np.stack([np.full((size, size, 3), value, np.float32) for value in values])


def test_perfect_future_ignores_a_wrong_reconstructed_condition_frame() -> None:
    truth = _video([0.25] + [0.5] * 16)
    prediction = truth.copy()
    prediction[0] = 1.0

    result = evaluate_video_pair(truth, prediction)

    assert result["frame_indices"] == list(range(1, 17))
    assert result["mean"]["mse"] == 0.0
    assert math.isinf(result["mean"]["psnr"])
    assert result["mean"]["ssim"] == pytest.approx(1.0)


def test_temporal_error_uses_true_i0_for_the_first_predicted_difference() -> None:
    truth = _video([0.0, 0.4] + [0.4] * 15)
    prediction = truth.copy()
    prediction[0] = 1.0  # Model reconstruction is deliberately wrong.

    result = evaluate_video_pair(truth, prediction)

    assert result["temporal"]["mean_mse"] == 0.0
    assert result["temporal"]["per_frame_mse"][0] == 0.0


def test_one_frame_misalignment_is_detected() -> None:
    truth = _video([0.0] + [i / 16 for i in range(1, 17)])
    shifted = np.concatenate([truth[:1], truth[2:], truth[-1:]], axis=0)

    result = evaluate_video_pair(truth, shifted)

    assert result["mean"]["mse"] > 0
    assert result["last"]["mse"] == 0.0


def test_persistence_repeats_observed_i0_for_all_future_frames() -> None:
    truth = _video([0.25] + [0.75] * 16)
    prediction = persistence_prediction(truth)

    np.testing.assert_array_equal(prediction, _video([0.25] * 17))


def test_lpips_receives_exactly_one_zero_one_to_minus_one_one_conversion() -> None:
    seen: list[tuple[np.ndarray, np.ndarray]] = []

    def metric(truth: np.ndarray, prediction: np.ndarray) -> np.ndarray:
        seen.append((truth.copy(), prediction.copy()))
        return np.zeros((truth.shape[0], 1, 1, 1), np.float32)

    values = lpips_frames(_video([0.0, 1.0]), _video([0.5, 0.25]), metric=metric)

    assert values == [0.0, 0.0]
    assert seen[0][0].shape == (2, 3, 16, 16)
    assert (seen[0][0][0] == -1).all() and (seen[0][0][1] == 1).all()
    assert (seen[0][1][0] == 0).all() and (seen[0][1][1] == -0.5).all()


def test_session_equal_aggregation_does_not_change_when_a_session_is_segmented() -> None:
    rows_a = [
        {"raw_session": "a", "episode": "a#seg0", "metrics": {"lpips": 1.0}},
        {"raw_session": "a", "episode": "a#seg0", "metrics": {"lpips": 3.0}},
        {"raw_session": "b", "episode": "b#seg0", "metrics": {"lpips": 8.0}},
    ]
    rows_b = [dict(row) for row in rows_a]
    rows_b[1]["episode"] = "a#seg1"

    assert aggregate_session_equal(rows_a)["overall"]["lpips"] == 5.0
    assert aggregate_session_equal(rows_a) == aggregate_session_equal(rows_b)


def test_sampling_seeds_are_averaged_per_window_before_session_weighting() -> None:
    rows = [
        {"window_id": "w0", "raw_session": "a", "sampling_seed": seed, "metrics": {"lpips": 0.0}}
        for seed in (0, 1, 2)
    ] + [{"window_id": "w1", "raw_session": "a", "sampling_seed": 0, "metrics": {"lpips": 10.0}}]

    windows = collapse_sampling_seeds(rows)

    assert aggregate_session_equal(windows)["overall"]["lpips"] == 5.0


def test_already_normalized_actions_are_not_normalized_twice() -> None:
    actions = np.full((16, 10), 0.25, np.float32)
    calls = 0

    def normalizer(value: np.ndarray) -> np.ndarray:
        nonlocal calls
        calls += 1
        return value + 10

    actual = prepare_model_actions(actions, source_space="normalized", normalizer=normalizer)

    assert calls == 0
    np.testing.assert_array_equal(actual, actions)


def test_physical_zero_is_identity_rotation_and_g0_before_one_normalization() -> None:
    seen: list[np.ndarray] = []

    def normalizer(value: np.ndarray) -> np.ndarray:
        seen.append(value.copy())
        return value + 2

    actual = make_physical_zero_action(16, normalizer=normalizer)

    expected_row = np.array([0, 0, 0, 1, 0, 0, 0, 1, 0, 0], np.float32)
    assert len(seen) == 1
    np.testing.assert_array_equal(seen[0], np.tile(expected_row, (16, 1)))
    np.testing.assert_array_equal(actual, np.tile(expected_row + 2, (16, 1)))


def test_action_permutation_stays_in_split_and_preserves_frozen_pair_ids(tmp_path: Path) -> None:
    rows = [
        {"window_id": "d0", "split": "dev", "translation_magnitude": 1.0, "rotation_magnitude": 3.0},
        {"window_id": "d1", "split": "dev", "translation_magnitude": 1.1, "rotation_magnitude": 3.1},
        {"window_id": "d2", "split": "dev", "translation_magnitude": 8.0, "rotation_magnitude": 9.0},
        {"window_id": "h0", "split": "history", "translation_magnitude": 1.0, "rotation_magnitude": 3.0},
        {"window_id": "h1", "split": "history", "translation_magnitude": 1.2, "rotation_magnitude": 3.1},
    ]
    path = tmp_path / "pairs.json"

    pairs = build_action_permutation(rows, output_path=path)

    by_id = {row["window_id"]: row for row in rows}
    assert all(source != target for source, target in pairs.items())
    assert all(by_id[source]["split"] == by_id[target]["split"] for source, target in pairs.items())
    assert json.loads(path.read_text()) == pairs


def test_four_previews_are_deterministic_approximately_equal_positions() -> None:
    ids = [f"w{i:02d}" for i in range(9)]
    assert select_session_previews(ids, count=4) == ["w00", "w03", "w05", "w08"]


def test_sparse_windows_use_stride_32_and_require_s_plus_32_below_n() -> None:
    assert sparse_window_starts(33) == [0]
    assert sparse_window_starts(65) == [0, 32]
    assert sparse_window_starts(32) == []


def test_noise_seed_is_stable_by_window_and_sampling_seed() -> None:
    first = derive_noise_seed("episode_50:s=0", 2)
    assert first == derive_noise_seed("episode_50:s=0", 2)
    assert first != derive_noise_seed("episode_50:s=32", 2)
    assert first != derive_noise_seed("episode_50:s=0", 1)
