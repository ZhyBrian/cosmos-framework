"""Failure-critical contracts for E3-Dout long RGBD rollouts and rendering."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import ImageFont

from examples.umift import render_rgbd_comparison as renderer
from examples.umift import rgbd_rollout


def _canvas(rgb: np.ndarray, depth_m: np.ndarray) -> np.ndarray:
    return rgbd_rollout.canvas_from_rgb_depth(rgb, depth_m)


def test_two_blocks_use_saved_float_rgbd_without_gt_refresh() -> None:
    height = width = 4
    initial_rgb = np.stack(
        [np.full((height, width, 3), 0.03125 * (index + 1), np.float32) for index in range(5)]
    )
    initial_depth = np.stack(
        [np.full((height, width), 0.02345 * (index + 1), np.float32) for index in range(5)]
    )
    initial_history = _canvas(initial_rgb, initial_depth)
    chunks = [
        {"index": 0, "output_start": 0, "steps": 16, "noise_seed": 101},
        {"index": 1, "output_start": 16, "steps": 3, "padding_steps": 13, "noise_seed": 202},
    ]
    rgb_output = np.empty((20, height, width, 3), np.float32)
    depth_output = np.empty((20, height, width), np.float32)
    observed_histories: list[np.ndarray] = []

    def predict(history: np.ndarray, chunk: dict) -> np.ndarray:
        observed_histories.append(history.copy())
        prediction = np.zeros((21, height, width * 2, 3), np.float32)
        prediction[:5] = history
        for step in range(16):
            global_step = int(chunk["output_start"]) + step
            rgb = np.full((height, width, 3), 0.12345 + global_step / 1000, np.float32)
            depth = np.full((height, width), 0.412345 + global_step / 100, np.float32)
            prediction[5 + step, :, :width] = 2 * rgb - 1
            prediction[5 + step, :, width:] = (4 * depth - 1)[..., None]
        history[:] = -0.777  # The predictor cannot mutate retained feedback state.
        return prediction

    records = rgbd_rollout.rollout(
        initial_history,
        chunks,
        predict,
        rgb_output,
        depth_output,
        initial_history_real_mask=np.array([False, False, False, False, True]),
    )

    assert len(observed_histories) == 2
    np.testing.assert_array_equal(observed_histories[0], initial_history)
    np.testing.assert_allclose(rgb_output[0], initial_rgb[-1], atol=2e-7, rtol=0)
    np.testing.assert_allclose(depth_output[0], initial_depth[-1], atol=2e-7, rtol=0)
    expected_second = _canvas(rgb_output[12:17], np.clip(depth_output[12:17], 0.0, 0.5))
    np.testing.assert_array_equal(observed_histories[1], expected_second)
    assert float(rgb_output[1, 0, 0, 0]) == pytest.approx(0.12345)
    assert float(depth_output[16, 0, 0]) == pytest.approx(0.562345)
    assert float(observed_histories[1][-1, 0, width, 0]) == pytest.approx(1.0)
    assert rgb_output.shape[0] == depth_output.shape[0] == 20
    assert [float(depth_output[index, 0, 0]) for index in range(17, 20)] == pytest.approx(
        [0.572345, 0.582345, 0.592345]
    )
    assert records[1]["history_source"] == "generated_rolling_rgbd"
    assert records[0]["initial_history_slots"] == 5
    assert records[0]["initial_real_slots"] == 1
    assert records[0]["initial_padding_slots"] == 4
    assert "initial_observations_in_history" not in records[0]
    assert records[1]["history_sha256"] == rgbd_rollout.array_sha(expected_second)
    assert records[0]["feedback_history_sha256"] == records[1]["history_sha256"]
    assert all("noise_seed" in record and "steps" in record for record in records)


def test_feedback_canvas_clips_saved_values_but_never_quantizes_rgb() -> None:
    rgb = np.array([[[[-0.1, 0.12345, 1.1]]]], np.float32)
    depth = np.array([[[-0.2]]], np.float32)

    canvas = _canvas(rgb, depth)

    np.testing.assert_allclose(canvas[0, 0, 0], [-1.0, 2 * 0.12345 - 1, 1.0], atol=2e-7)
    np.testing.assert_array_equal(canvas[0, 0, 1], [-1.0, -1.0, -1.0])
    assert float((canvas[0, 0, 0, 1] + 1) / 2) == pytest.approx(0.12345)
    assert float((canvas[0, 0, 0, 1] + 1) / 2) != pytest.approx(round(0.12345 * 255) / 255)


def test_prepare_binding_preserves_frozen_chunks_and_records_selected_identity() -> None:
    from examples.umift.prepare_rgbd_rollout import bind_manifest

    parent = {
        "protocol": "e2-history-open-loop-v1",
        "base_checkpoint": "/data/cosmos_models/Cosmos3-Edge-DCP-a9d944e/model",
        "zarr_path": "/data/cosmos_datasets/umift_us_all_source_308f4d46.zarr",
        "num_steps": 30,
        "sampling_seed": 0,
        "episodes": [
            {
                "episode_id": 13,
                "frame_count": 4,
                "initial_source_frame": 888,
                "chunks": [{"index": 0, "output_start": 0, "steps": 3, "noise_seed": 7}],
                "actions_path": "/frozen/actions.npz",
                "frame_indices_path": "/frozen/indices.npz",
            }
        ],
    }
    original = copy.deepcopy(parent)
    selection = {
        "experiment_id": "E3-Dout",
        "history_frames": 5,
        "iteration": 1500,
        "checkpoint": "/data/run/action_fd_umift_edge_rgbd_h5/checkpoints/iter_000001500/model",
        "protocol_file": "/data/run/protocol.json",
        "protocol_sha256": "1" * 64,
        "rgb_weight": 0.5,
        "joint_score": 0.82,
        "selection_uses_test_episodes": True,
    }

    result = bind_manifest(
        parent,
        selection,
        source_manifest=Path("/data/frozen/manifest.json"),
        source_sha256="2" * 64,
        selection_file=Path("/data/run/selected.json"),
        selection_sha256="3" * 64,
        start_percent=33,
    )

    assert parent == original
    assert result["protocol"] == "e3-dout-rgbd-open-loop-v1"
    assert result["selected_iteration"] == 1500
    assert result["selected_checkpoint"] == selection["checkpoint"]
    assert result["selection_joint_score"] == pytest.approx(0.82)
    assert result["selection_rgb_weight"] == pytest.approx(0.5)
    assert result["source_manifest_sha256"] == "2" * 64
    assert result["selection_sha256"] == "3" * 64
    assert result["episodes"][0]["chunks"] == parent["episodes"][0]["chunks"]
    assert result["episodes"][0]["history_padding_count"] == 0


def test_manifest_rejects_execution_fields_changed_after_source_binding(tmp_path: Path) -> None:
    source = {
        "base_checkpoint": "/data/cosmos_models/Cosmos3-Edge-DCP-a9d944e/model",
        "zarr_path": "/data/cosmos_datasets/umift_us_all_source_308f4d46.zarr",
        "num_steps": 30,
        "sampling_seed": 0,
        "guidance": 1.0,
        "episodes": [
            {
                "episode_id": episode_id,
                "source_id": f"session-{episode_id}#seg0",
                "raw_session": f"session-{episode_id}",
                "source_length": 40 + episode_id,
                "frame_count": 17,
                "start_percent": 33,
                "initial_selected_frame": 4,
                "initial_source_frame": 8,
                "parent_frame_count": 21,
                "initial_episode_elapsed_seconds": 0.53,
                "actions_path": f"/frozen/{episode_id}_actions.npz",
                "frame_indices_path": f"/frozen/{episode_id}_indices.npz",
                "truth_path": f"/frozen/{episode_id}_truth.npy",
                "input_files_sha256": {f"/frozen/{episode_id}_indices.npz": "a" * 64},
                "chunks": [
                    {
                        "index": 0,
                        "output_start": 0,
                        "steps": 16,
                        "padding_steps": 0,
                        "noise_seed": episode_id,
                    }
                ],
            }
            for episode_id in (13, 43, 49)
        ],
    }
    source_path = tmp_path / "source.json"
    source_path.write_text(json.dumps(source))
    selection_path = tmp_path / "selected.json"
    selection_path.write_text("{}")
    checkpoint = tmp_path / "action_fd_umift_edge_rgbd_h5" / "iter_000001500" / "model"
    checkpoint.mkdir(parents=True)
    (checkpoint / ".metadata").write_text("checkpoint")
    manifest = copy.deepcopy(source)
    manifest.update(
        experiment_id="E3-Dout",
        protocol="e3-dout-rgbd-open-loop-v1",
        history_frames=5,
        future_frames=16,
        source_manifest=str(source_path),
        source_manifest_sha256=rgbd_rollout.file_sha(source_path),
        selection_file=str(selection_path),
        selection_sha256=rgbd_rollout.file_sha(selection_path),
        selected_checkpoint=str(checkpoint),
        checkpoint_metadata_sha256=rgbd_rollout.file_sha(checkpoint / ".metadata"),
        zarr_complete_manifest_sha256=(
            "308f4d46132885375eb00c5578d52529512f0bca2c4e422c13724bfef1c6b9e6"
        ),
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    rgbd_rollout._validate_manifest(manifest, manifest_path)

    mutations = []
    changed_steps = copy.deepcopy(manifest)
    changed_steps["episodes"][0]["chunks"][0]["noise_seed"] += 1
    mutations.append(changed_steps)
    changed_num_steps = copy.deepcopy(manifest)
    changed_num_steps["num_steps"] = 29
    mutations.append(changed_num_steps)
    changed_source_identity = copy.deepcopy(manifest)
    changed_source_identity["episodes"][1]["raw_session"] = "other-session"
    mutations.append(changed_source_identity)
    changed_anchor = copy.deepcopy(manifest)
    changed_anchor["episodes"][2]["initial_source_frame"] += 2
    mutations.append(changed_anchor)
    for changed in mutations:
        with pytest.raises(ValueError, match="source manifest"):
            rgbd_rollout._validate_manifest(changed, manifest_path)

    source_start0 = copy.deepcopy(source)
    for episode in source_start0["episodes"]:
        for field in (
            "start_percent",
            "initial_selected_frame",
            "initial_source_frame",
            "parent_frame_count",
            "initial_episode_elapsed_seconds",
        ):
            episode.pop(field)
    source_start0_path = tmp_path / "source_start0.json"
    source_start0_path.write_text(json.dumps(source_start0))
    manifest_start0 = copy.deepcopy(source_start0)
    manifest_start0.update(manifest)
    manifest_start0["source_manifest"] = str(source_start0_path)
    manifest_start0["source_manifest_sha256"] = rgbd_rollout.file_sha(source_start0_path)
    manifest_start0["episodes"] = copy.deepcopy(source_start0["episodes"])
    for episode in manifest_start0["episodes"]:
        episode.update(
            start_percent=0,
            initial_selected_frame=0,
            initial_source_frame=0,
            parent_frame_count=episode["frame_count"],
            initial_episode_elapsed_seconds=0.0,
        )
    rgbd_rollout._validate_manifest(manifest_start0, manifest_path)
    manifest_start0["episodes"][0]["initial_source_frame"] = 2
    with pytest.raises(ValueError, match="source manifest"):
        rgbd_rollout._validate_manifest(manifest_start0, manifest_path)


def test_renderer_rejects_changed_index_file_or_manifest_pts(tmp_path: Path) -> None:
    path = tmp_path / "indices.npz"
    source_indices = np.array([10, 12, 14], np.int64)
    timestamps = np.array([4.0, 4.071, 4.139], np.float64)
    np.savez(path, source_indices=source_indices, timestamps=timestamps)
    expected_sha = renderer.file_sha(path)

    elapsed = renderer._load_elapsed(
        path,
        3,
        expected_sha256=expected_sha,
        expected_source_start=10,
        expected_timestamps=timestamps.tolist(),
    )

    assert elapsed.tolist() == pytest.approx([0.0, 0.071, 0.139])
    with pytest.raises(ValueError, match="PTS"):
        renderer._load_elapsed(
            path,
            3,
            expected_sha256=expected_sha,
            expected_source_start=10,
            expected_timestamps=[4.0, 4.070, 4.139],
        )
    np.savez(path, source_indices=source_indices, timestamps=[4.0, 4.072, 4.139])
    with pytest.raises(ValueError, match="SHA"):
        renderer._load_elapsed(
            path,
            3,
            expected_sha256=expected_sha,
            expected_source_start=10,
            expected_timestamps=timestamps.tolist(),
        )


def test_existing_run_evidence_is_never_overwritten(tmp_path: Path) -> None:
    expected = tmp_path / "run_base_rank0.json"
    assert rgbd_rollout._run_evidence_path(tmp_path, "base", 0) == expected
    expected.write_text("existing")
    with pytest.raises(FileExistsError):
        rgbd_rollout._run_evidence_path(tmp_path, "base", 0)


def test_renderer_binds_complete_scoring_files_and_row_identities(tmp_path: Path) -> None:
    scoring = tmp_path / "scoring"
    scoring.mkdir()
    manifest_path = tmp_path / "prepared" / "manifest.json"
    manifest_path.parent.mkdir()
    manifest_path.write_text("{}")
    manifest_sha = renderer.file_sha(manifest_path)
    episodes = [
        {"episode_id": 13, "raw_session": "s13", "start_percent": 0, "frame_count": 3},
        {"episode_id": 43, "raw_session": "s43", "start_percent": 0, "frame_count": 2},
        {"episode_id": 49, "raw_session": "s49", "start_percent": 0, "frame_count": 2},
    ]
    rows = [
        (method_index, episode["episode_id"], episode["start_percent"], episode["raw_session"], frame)
        for episode in episodes
        for method_index in range(5)
        for frame in range(1, episode["frame_count"])
    ]
    frame_path = scoring / "frame_metrics.npz"
    np.savez(
        frame_path,
        method_names=np.asarray(("P", "B0", "E3-A", "E3-Z", "E3-S")),
        method_index=np.asarray([row[0] for row in rows], np.int16),
        episode_id=np.asarray([row[1] for row in rows], np.int16),
        start_percent=np.asarray([row[2] for row in rows], np.int16),
        raw_session=np.asarray([row[3] for row in rows]),
        frame_index=np.asarray([row[4] for row in rows], np.int32),
        **{
            metric: np.zeros(len(rows), np.float64)
            for metric in renderer.FRAME_METRIC_COLUMNS
        },
    )
    metrics_path = scoring / "metrics.json"
    metrics_path.write_text(json.dumps({
        "protocol": "e3-dout-rgbd-full-suffix-scoring-v1",
        "experiment_id": "E3-Dout",
        "complete": True,
        "methods": ["P", "B0", "E3-A", "E3-Z", "E3-S"],
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_sha,
        "frame_metrics_npz": str(frame_path),
        "frame_metrics_npz_sha256": renderer.file_sha(frame_path),
    }))

    evidence, paths = renderer._validate_scoring(
        tmp_path, manifest_path, {"episodes": episodes}, manifest_sha
    )

    assert evidence["metrics_sha256"] == renderer.file_sha(metrics_path)
    assert evidence["frame_metrics_npz_sha256"] == renderer.file_sha(frame_path)
    assert evidence["frame_metric_rows"] == len(rows)
    assert paths == (metrics_path, frame_path)
    with np.load(frame_path, allow_pickle=False) as archive:
        columns = {key: archive[key] for key in archive.files}
    columns["episode_id"][0] = 49
    np.savez(frame_path, **columns)
    with pytest.raises(ValueError, match="SHA"):
        renderer._validate_scoring(
            tmp_path, manifest_path, {"episodes": episodes}, manifest_sha
        )


def _video_inputs(frame_count: int = 3):
    truth_rgb = np.empty((frame_count, 256, 256, 3), np.float32)
    truth_depth = np.empty((frame_count, 256, 256), np.float32)
    for frame in range(frame_count):
        truth_rgb[frame] = (20 + frame * 15) / 255
        truth_depth[frame] = 0.1 + frame * 0.03
    predictions = {}
    for method_index, method in enumerate(renderer.MODEL_METHODS):
        rgb = np.empty_like(truth_rgb)
        depth = np.empty_like(truth_depth)
        rgb[0], depth[0] = truth_rgb[0], truth_depth[0]
        for frame in range(1, frame_count):
            rgb[frame] = (60 + 25 * method_index + frame * 4) / 255
            depth[frame] = 0.18 + 0.05 * method_index + frame * 0.01
        predictions[method] = (rgb, depth)
    return truth_rgb, truth_depth, predictions


def test_twelve_panel_mapping_keeps_prediction_depth_independent_of_gt_mask() -> None:
    truth_rgb, truth_depth, predictions = _video_inputs(2)
    truth_depth[1, 100, 100] = 0.0
    predictions["B0"][1][1, 100, 100] = 0.25
    predictions["E3-A"][1][1, 100, 100] = 0.0

    panels = renderer.panels_at(truth_rgb, truth_depth, predictions, 1)

    assert len(renderer.PANEL_SPECS) == len(renderer.PANEL_BOXES) == len(panels) == 12
    assert renderer.PANEL_SPECS == tuple(
        (method, modality)
        for modality in ("rgb", "depth_m")
        for method in ("GT", "P", *renderer.MODEL_METHODS)
    )
    assert all(panel.size == (renderer.PANEL_SIZE, renderer.PANEL_SIZE) for panel in panels)
    b0_depth = panels[renderer.PANEL_SPECS.index(("B0", "depth_m"))]
    e3a_depth = panels[renderer.PANEL_SPECS.index(("E3-A", "depth_m"))]
    assert b0_depth.getpixel((100, 100)) == renderer.depth_color(0.25)
    assert e3a_depth.getpixel((100, 100)) == renderer.ZERO_SENTINEL_RGB
    assert b0_depth.getpixel((100, 100)) != renderer.ZERO_SENTINEL_RGB


def test_vfr_renderer_preserves_twelve_panels_dimensions_and_true_pts(tmp_path: Path) -> None:
    pytest.importorskip("av")
    truth_rgb, truth_depth, predictions = _video_inputs(3)
    elapsed = np.array([0.0, 0.071, 0.139], np.float64)
    episode = {
        "episode_id": 13,
        "start_percent": 33,
        "frame_count": 3,
        "selected_iteration": 1500,
        "history_padding_count": 0,
        "initial_episode_elapsed_seconds": 29.5,
        "chunks": [{"index": 0, "output_start": 0, "steps": 2, "noise_seed": 9}],
    }
    fonts = tuple(ImageFont.load_default() for _ in range(3))
    output = tmp_path / "rgbd.mp4"

    renderer.encode_video(
        output, truth_rgb, truth_depth, predictions, episode, elapsed, 1, fonts
    )
    result = renderer.verify_video(
        output, truth_rgb, truth_depth, predictions, episode, elapsed
    )

    assert result["frame_count"] == 3
    assert result["width"] == renderer.WIDTH
    assert result["height"] == renderer.HEIGHT
    assert result["frame_pts_seconds"] == elapsed.tolist()
    assert result["max_pts_error_seconds"] <= 0.001
    assert result["last_frame_duration_seconds"] == pytest.approx(0.068, abs=0.001)
    assert result["verified_panel_count"] == 12
