import copy
import json

import numpy as np
import pytest


def test_depth_freeze_uses_same_seventeen_pose_indices_as_dataset_window() -> None:
    from examples.umift.depth_selection import action_source_indices

    np.testing.assert_array_equal(action_source_indices(7), 7 + 2 * np.arange(17))


def _reports(arm: str = "d1") -> list[dict]:
    job = f"action_fd_umift_edge_rgbd_{arm}"
    return [
        dict(
            iteration=step,
            history_frames=5,
            experiment_id="E3-Depth-Aux",
            arm=arm,
            protocol_sha256="frozen",
            checkpoint=f"/data/cosmos_runs/e3_depth_aux_20260914/{arm}/train/"
            f"cosmos3_action_fd_umift/action_sft/{job}/checkpoints/iter_{step:09d}/model",
            rgb_weight=0.5,
            session_equal_aggregate={"overall": {"lpips": value, "depth_mae_m": 0.01}},
            persistence_session_equal_aggregate={
                "overall": {"lpips": 0.4, "depth_mae_m": 0.02}
            },
        )
        for step, value in zip((250, 500, 750, 1000), (0.3, 0.2, 0.1, 0.1))
    ]


def test_depth_selection_keeps_arm_identity_and_uses_early_tie() -> None:
    from examples.umift.depth_selection import choose_candidate

    assert choose_candidate(_reports(), "frozen", 0.5, "d1")["iteration"] == 750


def test_depth_selection_infer_preregisters_extension_steps() -> None:
    from examples.umift.depth_selection import EXTENSION_ITERATIONS, ITERATIONS

    assert EXTENSION_ITERATIONS == (1250, 1500)
    assert set(ITERATIONS).isdisjoint(EXTENSION_ITERATIONS)


def test_depth_selection_accepts_registered_extension_candidates() -> None:
    from examples.umift.depth_selection import choose_candidate

    reports = _reports()
    template = copy.deepcopy(reports[-1])
    extra = []
    for step, value in ((1250, 0.09), (1500, 0.08)):
        report = copy.deepcopy(template)
        report["iteration"] = step
        report["checkpoint"] = report["checkpoint"].replace("iter_000001000", f"iter_{step:09d}")
        report["session_equal_aggregate"]["overall"]["lpips"] = value
        extra.append(report)
    six = tuple(step for step, _ in zip((250, 500, 750, 1000, 1250, 1500), range(6)))
    chosen = choose_candidate(reports + extra, "frozen", 0.5, "d1", iterations=six)
    assert chosen["iteration"] == 1500
    with pytest.raises(ValueError):
        choose_candidate(reports, "frozen", 0.5, "d1", iterations=six)


@pytest.mark.parametrize("change", ["wrong_arm", "wrong_path", "wrong_protocol"])
def test_depth_selection_rejects_cross_arm_or_protocol_candidate(change: str) -> None:
    from examples.umift.depth_selection import choose_candidate

    reports = copy.deepcopy(_reports())
    if change == "wrong_arm":
        reports[0]["arm"] = "b_continue"
    elif change == "wrong_path":
        reports[0]["checkpoint"] = reports[0]["checkpoint"].replace(
            "action_fd_umift_edge_rgbd_d1", "action_fd_umift_edge_rgbd_h5"
        )
    else:
        reports[0]["protocol_sha256"] = "other"
    with pytest.raises(ValueError):
        choose_candidate(reports, "frozen", 0.5, "d1")


def test_depth_protocol_rejects_wrong_arm_and_old_e3_protocol(tmp_path) -> None:
    from examples.umift.depth_selection import expected_windows, load_protocol

    protocol = dict(
        protocol="e3-depth-aux-selection-v1",
        experiment_id="E3-Depth-Aux",
        arm="d1",
        history_frames=5,
        iterations=[250, 500, 750, 1000],
        rgb_weight=0.5,
        selection_metric="weighted_ratio_of_session_equal_lpips_and_depth_mae_to_persistence",
        aggregation_order="mean future frames, mean windows per session, mean sessions, then ratio to P",
        depth_mask="finite(GT) & 0<GT<0.5; never predicted-mask intersection",
        depth_units="metres",
        prediction_clamp_for_depth_metrics=False,
        tie_break="earlier_iteration",
        num_steps=30,
        selection_uses_test_episodes=True,
        zarr_path="/data/dataset.zarr",
        windows=[dict(window, physical_action_sha256=f"{index:064x}",
                      model_action_sha256=f"{index + 1:064x}")
                 for index, window in enumerate(expected_windows())],
    )
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(protocol))
    load_protocol(path, "d1")

    protocol["arm"] = "b_continue"
    path.write_text(json.dumps(protocol))
    with pytest.raises(ValueError):
        load_protocol(path, "d1")

    protocol["arm"] = "d1"
    protocol["protocol"] = "e3-rgbd-selection-v1"
    path.write_text(json.dumps(protocol))
    with pytest.raises(ValueError):
        load_protocol(path, "d1")


def test_depth_selection_rejects_model_action_that_differs_from_frozen_hash() -> None:
    from examples.umift.depth_selection import expected_model_action_sha256, validate_window_action_identity
    from examples.umift.rgbd_rollout import array_sha

    class TensorLike:
        def __init__(self, value): self.value = value
        def cpu(self): return self
        def numpy(self): return self.value
    physical = np.arange(160, dtype=np.float32).reshape(16, 10) / 100
    padded = np.zeros((16, 64), dtype=np.float32)
    window = dict(physical_action_sha256=array_sha(physical),
                  model_action_sha256=expected_model_action_sha256(physical))
    with pytest.raises(ValueError, match="model action"):
        validate_window_action_identity(
            window, {"physical_action": TensorLike(physical), "action": TensorLike(padded)}
        )
