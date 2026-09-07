import copy

import numpy as np
import pytest

from examples.umift.long_rollout import IDENTITY, array_sha, chunk_plan
from examples.umift.prepare_midstart import suffix_actions
from examples.umift.protocol import derive_noise_seed


def parent_example(length, episode_id):
    plan = chunk_plan(length, episode_id)
    count = (length + 1) // 2
    # Distinct per-action values expose skipped/duplicated transitions or old pad.
    actions = {k: np.tile(IDENTITY, (len(plan), 16, 1)) for k in ("A", "Z", "S")}
    for i, c in enumerate(plan):
        c["donor_window_id"] = f"episode_0:s={i * 32}"
        for k, sign in (("A", 1), ("S", -1)):
            actions[k][i, :c["steps"], 0] = sign * (np.arange(c["steps"]) + c["output_start"] + 1)
        for k in actions:
            c[f"{k}_physical_action_sha256"] = array_sha(actions[k][i])
    return {"episode_id": episode_id, "frame_count": count, "chunks": plan}, actions


@pytest.mark.parametrize("ep,length,percent,start,nchunks,tail", [
    (13, 2695, 33, 444, 57, 7), (13, 2695, 67, 902, 28, 13),
    (43, 1816, 33, 299, 38, 16), (43, 1816, 67, 607, 19, 12),
    (49, 1806, 33, 297, 38, 13), (49, 1806, 67, 604, 19, 10),
])
def test_actual_suffix_coverage_action_alignment_and_donor_segments(ep, length, percent, start, nchunks, tail):
    parent, actions = parent_example(length, ep)
    before = copy.deepcopy(parent)
    original = {k: v.copy() for k, v in actions.items()}
    actual_start, chunks, result = suffix_actions(parent, actions, percent)
    assert actual_start == start and len(chunks) == nchunks and chunks[-1]["steps"] == tail
    assert 1 + sum(c["steps"] for c in chunks) == parent["frame_count"] - start
    for k in actions:
        expected = np.concatenate([actions[k][i, :c["steps"]] for i, c in enumerate(parent["chunks"])])[start:]
        actual = np.concatenate([result[k][i, :c["steps"]] for i, c in enumerate(chunks)])
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(result[k][-1, tail:], np.tile(IDENTITY, (16 - tail, 1)))
        np.testing.assert_array_equal(actions[k], original[k])
    assert result["A"][0, 0, 0] == start + 1  # first action s, not s-1
    for c in chunks:
        reconstructed = np.concatenate([
            actions["S"][segment["parent_chunk_index"], segment["donor_action_offset"]:
                         segment["donor_action_offset"] + segment["steps"]]
            for segment in c["S_source_segments"]
        ])
        np.testing.assert_array_equal(reconstructed, result["S"][c["index"], :c["steps"]])
        assert c["global_source_start"] == 2 * (start + c["output_start"])
        assert c["noise_seed"] == derive_noise_seed(f"episode_{ep}:s={c['global_source_start']}", 0)
    assert len(chunks[0]["S_source_segments"]) == 2
    assert parent == before


def test_zero_start_regression_matches_original_actions_and_seeds():
    parent, actions = parent_example(2695, 13)
    start, chunks, rows = suffix_actions(parent, actions, 0)
    assert start == 0
    for k in actions:
        np.testing.assert_array_equal(rows[k], actions[k])
    assert [c["noise_seed"] for c in chunks] == [c["noise_seed"] for c in parent["chunks"]]


def test_parent_corruption_rejected():
    parent, actions = parent_example(1816, 43)
    actions["A"][0, 0, 0] += 1
    with pytest.raises(ValueError, match="hash"):
        suffix_actions(parent, actions, 33)


def test_renderer_origin_metadata_and_persistence_use_new_input():
    from examples.umift.render_long_comparison import _panels_at, _validate_episode
    parent, actions = parent_example(1816, 43)
    start, chunks, _ = suffix_actions(parent, actions, 33)
    episode = {**parent, "start_percent": 33, "initial_selected_frame": start,
               "initial_source_frame": 2 * start, "parent_frame_count": parent["frame_count"],
               "frame_count": parent["frame_count"] - start, "chunks": chunks,
               "raw_session": "test", "source_id": "test", "source_length": 1816, "fps": 15,
               "truth_path": "unused", "frame_indices_path": "unused"}
    _validate_episode(episode)
    with pytest.raises(ValueError, match="origin"):
        _validate_episode({**episode, "initial_source_frame": 0})
    truth = np.full((2, 256, 256, 3), 77, np.uint8)
    truth[1] = 99
    preds = {m: np.full(truth.shape, .25, np.float32) for m in ("B0", "E1-A", "E1-Z", "E1-S")}
    panels = _panels_at(truth, preds, 1)
    assert np.all(np.asarray(panels[0]) == 99) and np.all(np.asarray(panels[1]) == 77)
    assert all(np.all(np.asarray(p) == 77) for p in _panels_at(truth, preds, 0))
