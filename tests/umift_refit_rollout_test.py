import copy
from pathlib import Path

import pytest

from examples.umift.long_rollout import BASE_CHECKPOINT, BEST_CHECKPOINT
from examples.umift.prepare_refit_rollout import bind_checkpoint


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
