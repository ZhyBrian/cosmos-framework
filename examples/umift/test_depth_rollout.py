import copy
from pathlib import Path

import numpy as np


def test_depth_manifest_binding_preserves_parent_and_adds_arm_identity() -> None:
    from examples.umift.depth_rollout import bind_manifest

    parent = {
        "protocol": "e2-history-open-loop-v1",
        "episodes": [{"episode_id": 13, "frame_count": 4, "chunks": []}],
    }
    original = copy.deepcopy(parent)
    selection = {
        "iteration": 750,
        "checkpoint": "/run/d1/model",
        "protocol_file": "/run/d1/protocol.json",
        "protocol_sha256": "1" * 64,
        "rgb_weight": 0.5,
        "joint_score": 0.7,
    }
    result = bind_manifest(
        parent,
        selection,
        arm="d1",
        source_manifest=Path("/source.json"),
        source_sha256="2" * 64,
        selection_file=Path("/selected.json"),
        selection_sha256="3" * 64,
        start_percent=0,
    )
    assert parent == original
    assert result["experiment_id"] == "E3-Depth-Aux"
    assert result["protocol"] == "e3-depth-aux-rgbd-open-loop-v1"
    assert result["arm"] == "d1"
    assert result["episodes"][0]["arm"] == "d1"


def test_depth_methods_keep_twelve_panel_layout_for_each_arm() -> None:
    from examples.umift.depth_rollout import model_methods
    from examples.umift.render_rgbd_comparison import panels_at

    truth_rgb = np.zeros((2, 256, 256, 3), np.float32)
    truth_depth = np.full((2, 256, 256), 0.2, np.float32)
    for arm in ("b_continue", "d1"):
        methods = model_methods(arm)
        predictions = {
            method: (truth_rgb.copy(), truth_depth.copy()) for method in methods
        }
        panels = panels_at(
            truth_rgb, truth_depth, predictions, 1, model_methods=methods
        )
        assert len(panels) == 12
        assert methods[1:] == (
            ("B-A", "B-Z", "B-S")
            if arm == "b_continue"
            else ("D1-A", "D1-Z", "D1-S")
        )


def test_original_e3_keyword_defaults_remain_unchanged() -> None:
    from examples.umift import render_rgbd_comparison as renderer
    from examples.umift import rgbd_rollout
    from examples.umift import score_rgbd_rollout

    assert rgbd_rollout.MODEL_METHODS == ("B0", "E3-A", "E3-Z", "E3-S")
    assert score_rgbd_rollout.MODEL_METHODS == rgbd_rollout.MODEL_METHODS
    assert renderer.MODEL_METHODS == rgbd_rollout.MODEL_METHODS
    assert renderer.PANEL_SPECS == tuple(
        (method, modality)
        for modality in ("rgb", "depth_m")
        for method in ("GT", "P", "B0", "E3-A", "E3-Z", "E3-S")
    )
