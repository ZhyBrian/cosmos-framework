import copy
import json
from pathlib import Path

import pytest

from examples.umift.long_rollout import BASE_CHECKPOINT, BEST_CHECKPOINT, file_sha, validate_reference_scope
from examples.umift.prepare_refit_rollout import bind_checkpoint, read_frozen_fixture


def fixture():
    return {"selected_checkpoint": BEST_CHECKPOINT, "base_checkpoint": BASE_CHECKPOINT,
            "checkpoint_selection_sha256": "old-dev-selection", "episodes": [
                {"episode_id": ep, "chunks": [{"noise_seed": 42}], "input_files_sha256": {"a": "b"}}
                for ep in (13, 43, 49)]}


def test_refit_binding_preserves_inputs_and_never_inherits_dev_selection():
    parent = fixture()
    before = copy.deepcopy(parent)
    path = Path("/data/cosmos_runs/test/action_fd_umift_edge_e1_refit/checkpoints/iter_000001000/model")
    result = bind_checkpoint(parent, path, 1000)
    assert parent == before
    assert "checkpoint_selection_sha256" not in result
    assert result["selected_checkpoint"] == str(path)
    assert result["experiment_id"] == "E1-R"
    for original, bound in zip(parent["episodes"], result["episodes"]):
        assert {k: v for k, v in bound.items() if k != "experiment_id"} == original


@pytest.mark.parametrize("path,iteration", [(BEST_CHECKPOINT, 1000),
    ("/data/cosmos_runs/test/action_fd_umift_edge_e1_refit/checkpoints/iter_000000500/model", 1000)])
def test_refit_rejects_old_job_or_wrong_iteration(path, iteration):
    with pytest.raises(ValueError):
        bind_checkpoint(fixture(), Path(path), iteration)


def test_previously_frozen_hash_rejects_noise_change_with_intact_input_hashes(tmp_path):
    path = tmp_path / "manifest.json"
    parent = fixture()
    path.write_text(json.dumps(parent))
    frozen_hash = file_sha(path)
    assert read_frozen_fixture(path, frozen_hash) == parent
    parent["episodes"][0]["chunks"][0]["noise_seed"] += 1
    path.write_text(json.dumps(parent))
    with pytest.raises(ValueError, match="previously audited"):
        read_frozen_fixture(path, frozen_hash)


def test_step500_cannot_start_full_generation_but_allows_bounded_reference():
    path = Path("/data/cosmos_runs/test/action_fd_umift_edge_e1_refit/checkpoints/iter_000000500/model")
    bound = bind_checkpoint(fixture(), path, 500)
    validate_reference_scope(bound, 2)
    with pytest.raises(ValueError, match="reference-only"):
        validate_reference_scope(bound, 0)
    validate_reference_scope({}, 0)  # original E1 behavior unchanged
