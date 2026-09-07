"""Static contracts for the UMI-FT Cosmos3-Edge forward-dynamics recipe.

This test intentionally avoids importing the training stack so it remains useful on
the data-preparation host, where Hydra/Transformers/CUDA dependencies are absent.
"""

from __future__ import annotations

import ast
import os
from pathlib import Path
import shutil
import subprocess
import tomllib


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "cosmos_framework/configs/base/experiment/action/posttrain_config/action_fd_umift_edge.py"
TOML = ROOT / "examples/toml/sft_config/action_fd_umift_edge.toml"
LAUNCHER = ROOT / "examples/launch_sft_action_fd_umift_edge.sh"


def _source() -> str:
    return EXPERIMENT.read_text(encoding="utf-8")


def test_experiment_is_registered_and_uses_edge_umift_fd_contract() -> None:
    source = _source()
    ast.parse(source)
    assert "EDGE_MODEL_CONFIG" in source
    assert "NANO_MODEL_CONFIG" not in source
    assert "get_umift_zarr_sft_dataset" in source
    assert "get_umift_packing_dataloader" in source
    assert "get_umift_dataloader_generator" in source
    assert "dataloader_train=L(get_umift_packing_dataloader)(" in source
    assert "from cosmos_framework.data.generator.processors import build_processor_lazy" not in source
    assert 'tokenizer_type="${oc.env:EDGE_HF_SNAPSHOT_PATH}"' in source
    assert 'mode="forward_dynamics"' in source
    assert 'split="train"' in source
    assert 'stage="${oc.env:UMIFT_STAGE,e1}"' in source
    assert 'resolution="256"' in source
    assert "fps=15.0" in source
    assert 'tokenizer_config="${model.config.vlm_config.tokenizer}"' in source
    assert 'max_action_dim="${model.config.max_action_dim}"' in source
    assert "max_samples_per_batch=1" in source
    assert "max_sequence_length=None" in source
    assert "num_workers=0" in source
    assert "generator=L(get_umift_dataloader_generator)(seed=42)" in source
    assert "persistent_workers=False" in source
    assert "prefetch_factor" not in source

    registry = (ROOT / "cosmos_framework/configs/base/config.py").read_text(encoding="utf-8")
    assert "action_fd_umift_edge" in registry


def test_recipe_has_exact_edge_training_contract() -> None:
    source = _source()
    required = [
        'cfg["action_gen"] = True',
        'cfg["joint_attn_implementation"] = "two_way"',
        'cfg["resolution"] = "256"',
        'cfg["diffusion_expert_config"]["load_weights_from_pretrained"] = False',
        'cfg["activation_checkpointing"]["mode"] = "selective"',
        'cfg["compile"]["enabled"] = False',
        'cfg["ema"]["enabled"] = False',
        'cfg["parallelism"]["data_parallel_shard_degree"] = 4',
        'cfg["parallelism"]["fsdp_master_dtype"] = "float32"',
        'cfg["parallelism"]["fsdp_reduce_dtype"] = "bfloat16"',
        "grad_accum_iter=4",
        "straggler_detection=dict(enabled=False, report_freq=10)",
        "max_iter=1000",
        "max_consecutive_nan=1",
        "clip_norm=1.0",
        'warm_up_steps=[100]',
        'cycle_lengths=[1000]',
        'f_max=[1.0]',
        'f_min=[0.1]',
        "lr=1.0e-05",
        '"action2llm": 5.0',
        'load_training_state=False',
        'strict_resume=True',
    ]
    for fragment in required:
        assert fragment in source, fragment


def test_refit_recipe_changes_only_experiment_and_run_identity() -> None:
    original = tomllib.loads(TOML.read_text())
    refit = tomllib.loads(TOML.with_name("action_fd_umift_edge_refit.toml").read_text())
    assert refit["job"]["experiment"] == "action_fd_umift_edge_refit"
    assert refit["job"]["name"] == "action_fd_umift_edge_e1_refit"
    refit["job"]["experiment"] = original["job"]["experiment"]
    refit["job"]["name"] = original["job"]["name"]
    assert refit == original
    source = _source()
    assert "action_fd_umift_edge_refit = copy.deepcopy(action_fd_umift_edge)" in source
    assert 'action_fd_umift_edge_refit.dataloader_train.dataloader.datasets.umift.dataset.split = "refit_train"' in source


def test_toml_and_launcher_expose_safe_stage_profiles() -> None:
    raw = tomllib.loads(TOML.read_text(encoding="utf-8"))
    assert raw["job"]["experiment"] == "action_fd_umift_edge"
    assert raw["job"]["wandb_mode"] == "disabled"
    assert raw["model"]["precision"] == "bfloat16"
    assert raw["model"]["parallelism"]["data_parallel_shard_degree"] == 4
    assert raw["trainer"]["grad_accum_iter"] == 4
    assert raw["trainer"]["max_iter"] == 1000
    assert raw["dataloader_train"] == {"max_samples_per_batch": 1}
    assert raw["scheduler"] == {
        "cycle_lengths": [1000],
        "f_max": [1.0],
        "f_min": [0.1],
        "f_start": [0.0],
        "warm_up_steps": [100],
    }

    launcher = LAUNCHER.read_text(encoding="utf-8")
    for stage, updates in (("smoke", 10), ("overfit", 200), ("e1", 1000)):
        assert f'{stage}) MAX_UPDATES={updates}' in launcher
    assert 'NPROC_PER_NODE:=4' in launcher
    assert 'UMIFT_STAGE="$STAGE"' in launcher
    assert 'trainer.max_iter=$MAX_UPDATES' in launcher
    assert 'job.name=action_fd_umift_edge_${STAGE}' in launcher
    assert 'I4_ATTN_BACKENDS:=natten' in launcher
    assert 'I4_ATTN_BACKENDS_MULTIDIM' not in launcher
    assert 'EDGE_REVISION="a9d944e2c6a1bf9f48b92ad16348e70c5f1836ba"' in launcher
    assert 'basename "$EDGE_HF_SNAPSHOT_PATH"' in launcher


def test_launcher_uses_cosmos_paths_and_output_root_for_resume(tmp_path: Path) -> None:
    sandbox = tmp_path / "examples"
    sandbox.mkdir()
    shutil.copy2(LAUNCHER, sandbox / LAUNCHER.name)
    (sandbox / "_sft_launcher_common.sh").write_text(
        'printf "OUTPUT=%s\\nIMAGINAIRE=%s\\n" "$OUTPUT_ROOT" "$IMAGINAIRE_OUTPUT_ROOT"\n',
        encoding="utf-8",
    )

    revision = "a9d944e2c6a1bf9f48b92ad16348e70c5f1836ba"
    snapshot = tmp_path / revision
    snapshot.mkdir()
    output = tmp_path / "cosmos_runs/umift_edge_fd"
    wrong_output = tmp_path / "stale_output"
    wrong_latest = wrong_output / "cosmos3_action_fd_umift/action_sft/action_fd_umift_edge_e1/checkpoints/latest_checkpoint.txt"
    wrong_latest.parent.mkdir(parents=True)
    wrong_latest.write_text("iter_000000250\n", encoding="utf-8")
    env = {
        **os.environ,
        "OUTPUT_ROOT": str(output),
        "IMAGINAIRE_OUTPUT_ROOT": str(wrong_output),
        "EDGE_HF_SNAPSHOT_PATH": str(snapshot),
        "RUN_MODE": "resume",
    }

    rejected = subprocess.run(["bash", str(sandbox / LAUNCHER.name)], env=env, capture_output=True, text=True)
    assert rejected.returncode == 2
    assert f"no same-job checkpoint exists: {output}" in rejected.stderr

    correct_latest = output / "cosmos3_action_fd_umift/action_sft/action_fd_umift_edge_e1/checkpoints/latest_checkpoint.txt"
    correct_latest.parent.mkdir(parents=True)
    correct_latest.write_text("iter_000000250\n", encoding="utf-8")
    accepted = subprocess.run(["bash", str(sandbox / LAUNCHER.name)], env=env, capture_output=True, text=True)
    assert accepted.returncode == 0, accepted.stderr
    assert f"OUTPUT={output}" in accepted.stdout
    assert f"IMAGINAIRE={output}" in accepted.stdout
