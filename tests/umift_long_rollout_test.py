import numpy as np
import pytest

from examples.umift.long_rollout import IDENTITY, chunk_plan, padded_actions, quantize_condition, rollout, timestamp_statistics


@pytest.mark.parametrize("episode,length,frames,chunks,tail", [
    (13, 2695, 1348, 85, 3), (43, 1816, 908, 57, 11), (49, 1806, 903, 57, 6),
])
def test_full_episode_coverage_and_legal_tail_anchor(episode, length, frames, chunks, tail):
    plan = chunk_plan(length, episode)
    assert len(plan) == chunks and plan[-1]["steps"] == tail
    assert 1 + sum(row["steps"] for row in plan) == frames
    covered = [0]
    for row in plan:
        assert row["anchor_start"] + 32 < length
        assert row["anchor_start"] // 2 + row["anchor_action_offset"] == row["output_start"]
        covered.extend(range(row["output_start"] + 1, row["output_start"] + row["steps"] + 1))
    assert covered == list(range(frames))
    assert len({row["noise_seed"] for row in plan}) == chunks
    assert 2 * covered[-1] == length - 1 - (length % 2 == 0)


@pytest.mark.parametrize("offset,steps", [(13, 3), (5, 11), (10, 6)])
def test_tail_real_actions_preserved_then_physical_identity(offset, steps):
    original = np.arange(160, dtype=np.float32).reshape(16, 10)
    value = padded_actions(original, offset, steps)
    np.testing.assert_array_equal(value[:steps], original[offset:])
    np.testing.assert_array_equal(value[steps:], np.tile(IDENTITY, (16 - steps, 1)))


def test_open_loop_uses_generated_endpoint_and_discards_decoded_condition():
    first = np.full((256, 256, 3), 51, np.uint8)
    plan = chunk_plan(39, 13)  # 20 displayed frames: 16+3 future.
    received = []

    def predict(condition, chunk):
        received.append(condition.copy())
        prediction = np.full((17, 256, 256, 3), .3 if chunk["index"] == 0 else .7, np.float32)
        prediction[0] = 1.0  # A wrong reconstructed I0 must never enter stitched output.
        return prediction

    output = np.empty((20, 256, 256, 3), np.float32)
    records = rollout(first, plan, predict, output)
    np.testing.assert_array_equal(received[0], first)
    np.testing.assert_array_equal(received[1], quantize_condition(output[16]))
    np.testing.assert_array_equal(output[0], first.astype(np.float32) / 255)
    assert np.all(output[1:17] == .3) and np.all(output[17:] == .7)
    assert records[1]["condition_sha256"] == records[0]["feedback_sha256"]
    assert np.all(first == 51)


def test_feedback_does_not_mutate_actions_or_leave_future_images():
    torch = pytest.importorskip("torch")
    from examples.umift.long_rollout import feedback_sample
    video = torch.full((3, 17, 256, 256), 199, dtype=torch.uint8)
    action = torch.arange(16 * 64).reshape(16, 64)
    condition = np.full((256, 256, 3), 77, np.uint8)
    result = feedback_sample({"video": video, "action": action}, condition)
    assert torch.all(result["video"][:, 0] == 77) and torch.all(result["video"][:, 1:] == 0)
    assert result["action"] is action and torch.all(video == 199)


def test_generated_nonfinite_values_are_rejected():
    with pytest.raises(ValueError, match="finite"):
        quantize_condition(np.full((256, 256, 3), np.nan, np.float32))


def test_real_time_statistics_are_not_inferred_from_nominal_fps():
    stats = timestamp_statistics(np.array([7.0, 7.040, 7.080, 7.125]))
    assert stats["span_seconds"] == .125
    assert stats["mean_rate_hz"] == 24.0
    assert stats["frame_count"] == 4
