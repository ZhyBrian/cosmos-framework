import copy
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys

import numpy as np
import pytest


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


def test_depth_model_action_hash_is_derived_from_physical_action_and_rejects_tamper() -> None:
    from examples.umift.depth_rollout import (
        expected_model_action_sha256,
        validate_depth_action_identity,
    )

    physical = np.arange(160, dtype=np.float32).reshape(16, 10) / 100
    expected = expected_model_action_sha256(physical, max_action_dim=64)
    frozen = {
        "A_physical_action_sha256": "a" * 64,
        "A_model_action_sha256": expected,
    }
    actual = {
        "action_source": "A",
        "physical_action_sha256": "a" * 64,
        "padded_model_action_sha256": expected,
    }
    validate_depth_action_identity("D1-A", frozen, actual)

    tampered = dict(actual, padded_model_action_sha256="b" * 64)
    with pytest.raises(ValueError, match="model action"):
        validate_depth_action_identity("D1-A", frozen, tampered)


def test_rgbd_loader_accepts_explicit_job_and_records_experiment(monkeypatch) -> None:
    from examples.umift.rgbd_infer import load_rgbd_model

    config = SimpleNamespace(
        job=SimpleNamespace(name="action_fd_umift_edge_rgbd_d1"),
        dataloader_train=SimpleNamespace(
            _target_=SimpleNamespace(__name__="get_umift_rgbd_packing_dataloader"),
            dataloader=SimpleNamespace(
                datasets=SimpleNamespace(
                    umift=SimpleNamespace(
                        dataset=SimpleNamespace(
                            _target_=SimpleNamespace(
                                __name__="get_umift_rgbd_sft_dataset"
                            )
                        )
                    )
                )
            ),
        ),
    )
    config_module = ModuleType("cosmos_framework.configs.toml_config.sft_config")
    config_module.load_experiment_from_toml = lambda path: config
    history_module = ModuleType("examples.umift.history_infer")
    history_module.load_history_model = lambda *args, **kwargs: (
        "model",
        config,
        {"model_key_count": 549, "checkpoint_key_count": 549},
    )
    monkeypatch.setitem(
        sys.modules, "cosmos_framework.configs.toml_config.sft_config", config_module
    )
    monkeypatch.setitem(sys.modules, "examples.umift.history_infer", history_module)

    _, _, evidence = load_rgbd_model(
        Path("d1.toml"),
        Path("checkpoint"),
        expected_job_name="action_fd_umift_edge_rgbd_d1",
        experiment_id="E3-Depth-Aux",
    )
    assert evidence["experiment"] == "E3-Depth-Aux"
    assert evidence["job_name"] == "action_fd_umift_edge_rgbd_d1"

    with pytest.raises(ValueError, match="action_fd_umift_edge_rgbd_b_continue"):
        load_rgbd_model(
            Path("d1.toml"),
            Path("checkpoint"),
            expected_job_name="action_fd_umift_edge_rgbd_b_continue",
            experiment_id="E3-Depth-Aux",
        )
