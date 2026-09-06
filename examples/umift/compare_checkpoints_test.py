from __future__ import annotations

import pickle
import argparse
import hashlib
import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
import torch.distributed.checkpoint as dcp

from examples.umift.compare_checkpoints import compare_checkpoints, resolve_iteration
from examples.umift.collect_stateless_evidence import collect


def _write(root: Path, model_value: float, *, iteration: int = 10, rng_delta: int = 0) -> Path:
    checkpoint = root / f"iter_{iteration:09d}"
    states = {
        "model": {"weight": torch.tensor([model_value, 2.0]), "frozen": torch.tensor([7.0])},
        "optim": {
            "state": {
                "weight": {
                    "exp_avg": torch.tensor([0.1, 0.2]),
                    "exp_avg_sq": torch.tensor([0.01, 0.04]),
                }
            },
            "param_groups": [{"lr": 1e-4, "step": 10}],
        },
        "scheduler": {"last_epoch": 10, "_last_lr": [1e-4]},
        "trainer": {
            "iteration": iteration,
            "rng_state_0": {
                "torch": torch.tensor([1 + rng_delta, 2], dtype=torch.uint8),
                "torch_cuda": torch.tensor([3, 4], dtype=torch.uint8),
                "numpy_packed_len": torch.tensor(1),
                "numpy_packed_bytes": torch.tensor([5], dtype=torch.uint8),
                "random_packed_len": torch.tensor(1),
                "random_packed_bytes": torch.tensor([6], dtype=torch.uint8),
            },
        },
    }
    for component, state in states.items():
        dcp.save(state, checkpoint_id=checkpoint / component)
    (checkpoint / "dataloader").mkdir()
    with (checkpoint / "dataloader" / "rank_0.pkl").open("wb") as stream:
        pickle.dump({"next_draw_index": 10}, stream)
    root.mkdir(exist_ok=True)
    (root / "latest_checkpoint.txt").write_text(checkpoint.name + "\n")
    return checkpoint


def test_identical_dcp_components_and_rng_pass(tmp_path: Path) -> None:
    a = _write(tmp_path / "a", 1.0)
    b = _write(tmp_path / "b", 1.0)
    report = compare_checkpoints(a, b)
    assert report["pass"]
    assert report["model"]["all_parameters"]["bitwise_equal"]
    assert report["rng"]["per_rank"]["0"]["state_equal"]
    assert report["components"]["optim"]["categories"]["optimizer_first_moment"]["count"] == 1
    assert report["components"]["optim"]["categories"]["optimizer_second_moment"]["count"] == 1


def test_small_float_difference_reported_separately_from_bitwise(tmp_path: Path) -> None:
    a = _write(tmp_path / "a", 1.0)
    b = _write(tmp_path / "b", 1.0 + 5e-7)
    report = compare_checkpoints(a, b, atol=1e-6, rtol=0.0)
    model = report["model"]["all_parameters"]
    assert report["pass"]
    assert not model["bitwise_equal"]
    assert model["tolerance_pass"]
    assert model["max_abs"] > 0
    assert report["model"]["differing_parameter_names"] == ["weight"]


def test_metadata_shape_mismatch_fails_before_values(tmp_path: Path) -> None:
    a = _write(tmp_path / "a", 1.0)
    b = _write(tmp_path / "b", 1.0)
    dcp.save({"weight": torch.ones(3), "frozen": torch.tensor([7.0])}, checkpoint_id=b / "model")
    with pytest.raises(ValueError, match="metadata mismatch"):
        compare_checkpoints(a, b)


def test_rng_or_dataloader_mismatch_fails(tmp_path: Path) -> None:
    a = _write(tmp_path / "a", 1.0)
    b = _write(tmp_path / "b", 1.0, rng_delta=1)
    report = compare_checkpoints(a, b)
    assert not report["pass"]
    assert not report["rng"]["all_sequences_identical"]
    with (b / "dataloader" / "rank_0.pkl").open("wb") as stream:
        pickle.dump({"next_draw_index": 9}, stream)
    assert not compare_checkpoints(a, b)["dataloader"]["pass"]


def test_latest_marker_resolution(tmp_path: Path) -> None:
    checkpoint = _write(tmp_path / "job" / "checkpoints", 1.0)
    assert resolve_iteration(tmp_path / "job") == checkpoint.resolve()


def _stateless_evidence() -> dict:
    return {
        "schema_version": "umift-stateless-resume-v1",
        "stage": "smoke",
        "seed": 42,
        "world_size": 4,
        "grad_accum_iter": 4,
        "resume_from_iteration": 5,
        "derived_resume_microbatch": 20,
        "checkpoint5_copy": {"verified": True, "components": ["model", "optim", "scheduler", "trainer"]},
        "resolved_configs": {"semantic_fields_match": True, "reference_sha256": "a" * 64, "candidate_sha256": "a" * 64},
        "resume_log": {"loaded_iteration": 5, "derived_sample_offset": 20, "sha256": "b" * 64},
        "gloo_audit": {
            "stage": "e1", "world_size": 4, "resume_microbatch": 20,
            "seed": 42, "dataset": "/data/cosmos_datasets/fixture.zarr",
            "dataset_root_metadata_sha256": "c" * 64,
            "ranks": [{"rank": rank, "resume_offset_unit": "rank-local microbatch",
                       "resume_suffix_matches": True} for rank in range(4)],
        },
    }


def test_missing_dataloader_requires_explicit_verified_stateless_contract(tmp_path: Path) -> None:
    a = _write(tmp_path / "a", 1.0)
    b = _write(tmp_path / "b", 1.0)
    (a / "dataloader" / "rank_0.pkl").unlink(); (a / "dataloader").rmdir()
    (b / "dataloader" / "rank_0.pkl").unlink(); (b / "dataloader").rmdir()
    missing = compare_checkpoints(a, b)
    assert not missing["pass"]
    assert "stateless evidence" in missing["dataloader"]["error"]


def test_stateless_contract_rejects_incomplete_evidence(tmp_path: Path) -> None:
    a = _write(tmp_path / "a", 1.0); b = _write(tmp_path / "b", 1.0)
    for path in (a, b):
        (path / "dataloader" / "rank_0.pkl").unlink(); (path / "dataloader").rmdir()
    evidence = _stateless_evidence(); evidence["gloo_audit"]["ranks"][2]["resume_suffix_matches"] = False
    report = compare_checkpoints(a, b, umift_stateless_evidence=evidence)
    assert not report["pass"]
    assert "source is missing or changed" in report["dataloader"]["error"]
    assert "model" in report["components"]


def test_one_sided_dataloader_absence_cannot_use_stateless_evidence(tmp_path: Path) -> None:
    a = _write(tmp_path / "a", 1.0); b = _write(tmp_path / "b", 1.0)
    (b / "dataloader" / "rank_0.pkl").unlink(); (b / "dataloader").rmdir()
    report = compare_checkpoints(a, b)
    assert not report["pass"]
    assert report["dataloader"]["checkpoint_component_present"] == [True, False]
    assert report["model"]["all_parameters"]["tolerance_pass"]


def test_iteration_directory_must_match_trainer_and_optimizer_counts(tmp_path: Path) -> None:
    a = _write(tmp_path / "a", 1.0, iteration=9)
    b = _write(tmp_path / "b", 1.0, iteration=9)
    renamed_a = a.with_name("iter_000000010"); renamed_b = b.with_name("iter_000000010")
    a.rename(renamed_a); b.rename(renamed_b)
    report = compare_checkpoints(renamed_a, renamed_b)
    assert not report["pass"]
    assert "trainer iteration" in report["iteration_consistency"]["reference"]["error"]
    assert "model" in report["components"]


def test_stateless_evidence_is_collected_from_hashed_sources(tmp_path: Path) -> None:
    ref_config = tmp_path / "ref_config.yaml"; cand_config = tmp_path / "cand_config.yaml"
    base = {
        "model": {"config": {"parallelism": {"context_parallel_shard_degree": 1,
            "data_parallel_replicate_degree": 1, "data_parallel_shard_degree": 4}}},
        "optimizer": {"lr": 1e-5}, "scheduler": {"warmup": 0},
        "trainer": {"grad_accum_iter": 4, "seed": 42, "output_dir": "PLACEHOLDER"},
        "dataloader_train": {"max_samples_per_batch": 1, "dataloader": {"num_workers": 0,
            "generator": {"seed": 42}, "datasets": {"umift": {"dataset": {
                "stage": "smoke", "seed": 42, "split": "train",
                "zarr_path": "/data/cosmos_datasets/fixture.zarr"}}}}},
        "checkpoint": {"load_path": "/data/cosmos_models/base/model"},
    }
    import yaml
    ref = dict(base); ref["trainer"] = {**base["trainer"], "output_dir": "/data/cosmos_runs/ref"}
    cand = dict(base); cand["trainer"] = {**base["trainer"], "output_dir": "/data/cosmos_runs/cand"}
    ref_config.write_text(yaml.safe_dump(ref)); cand_config.write_text(yaml.safe_dump(cand))
    ref_launch = tmp_path / "ref_launch.yaml"; cand_launch = tmp_path / "cand_launch.yaml"
    launch = "cmd: torchrun\nargs_cfg_path: /tmp/config.py\nargs_override: []\n"
    ref_launch.write_text(launch); cand_launch.write_text(launch)
    ref_log = tmp_path / "ref.log"; resume_log = tmp_path / "resume.log"
    ranks = "\n".join(f"RankPartitionedDataLoader rank={rank} world_size=4" for rank in range(4))
    ref_log.write_text(ranks + "\n")
    resume_log.write_text("loaded checkpoint at iteration 5\n" + ranks + "\n")
    copy = tmp_path / "copy.json"
    copy_source = tmp_path / "source" / "iter_000000005"
    copy_target = tmp_path / "target" / "iter_000000005"
    records = []
    for component in ("model", "optim", "scheduler", "trainer"):
        for name in (".metadata", *(f"__{rank}_0.distcp" for rank in range(4))):
            artifact = copy_source / component / name
            artifact.parent.mkdir(parents=True, exist_ok=True)
            payload = f"{component}/{name}".encode()
            artifact.write_bytes(payload)
            records.append({"path": f"{component}/{name}", "bytes": len(payload),
                            "sha256": hashlib.sha256(payload).hexdigest()})
    copy.write_text(json.dumps({"source": str(copy_source), "target": str(copy_target),
        "files": records, "all_hashes_equal": True,
        "checkpoint_components": ["model", "optim", "scheduler", "trainer"],
        "dataloader_component_absent": True}))
    audit = tmp_path / "audit.json"
    audit.write_text(json.dumps(_stateless_evidence()["gloo_audit"]))
    args = argparse.Namespace(
        reference_config=ref_config, candidate_config=cand_config,
        reference_launch_info=ref_launch, candidate_launch_info=cand_launch,
        reference_driver_log=ref_log, candidate_driver_log=resume_log,
        checkpoint5_copy_evidence=copy, gloo_audit=audit, training_commit="9c81289",
    )
    evidence = collect(args)
    assert evidence["resume_log"]["loaded_iteration"] == 5
    assert evidence["resolved_configs"]["reference"]["path"] == str(ref_config.resolve())
    assert len(evidence["gloo_audit"]["source"]["sha256"]) == 64
    a = _write(tmp_path / "a", 1.0); b = _write(tmp_path / "b", 1.0)
    for checkpoint in (a, b):
        (checkpoint / "dataloader" / "rank_0.pkl").unlink(); (checkpoint / "dataloader").rmdir()
    assert compare_checkpoints(a, b, umift_stateless_evidence=evidence)["pass"]
    ref_log.write_text(ref_log.read_text() + "changed after collection\n")
    changed = compare_checkpoints(a, b, umift_stateless_evidence=evidence)
    assert not changed["pass"]
    assert "source is missing or changed" in changed["dataloader"]["error"]
    ref_log.write_text(ranks + "\n")

    wrong_workers = dict(cand)
    wrong_workers["dataloader_train"] = json.loads(json.dumps(cand["dataloader_train"]))
    wrong_workers["dataloader_train"]["dataloader"]["num_workers"] = 1
    cand_config.write_text(yaml.safe_dump(wrong_workers))
    with pytest.raises(ValueError, match="num_workers"):
        collect(args)
    cand_config.write_text(yaml.safe_dump(cand))

    wrong_audit = _stateless_evidence()["gloo_audit"]
    wrong_audit["dataset"] = "/data/cosmos_datasets/other.zarr"
    audit.write_text(json.dumps(wrong_audit))
    with pytest.raises(ValueError, match="dataset path/seed"):
        collect(args)
