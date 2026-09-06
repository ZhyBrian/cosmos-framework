from __future__ import annotations

import importlib.util
import re
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "examples/umift/preflight.py"
ENV_PATH = ROOT / "examples/umift/a40_env.sh"
SPEC = importlib.util.spec_from_file_location("umift_preflight", PATH)
assert SPEC and SPEC.loader
preflight = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(preflight)


def ns(**kwargs):
    return SimpleNamespace(**kwargs)


def test_a40_env_activates_fixed_conda_env_and_checks_actual_python() -> None:
    source = ENV_PATH.read_text(encoding="utf-8")
    assert '_umift_conda_sh="$_umift_conda_root/etc/profile.d/conda.sh"' in source
    assert "conda activate \"$_umift_env\"" in source
    assert "_umift_env=/data/miniconda3/envs/cosmos_edge_e1" in source
    assert '"${CONDA_PREFIX-}" != "$_umift_env"' in source
    assert "command -v python" in source
    assert "os.path.realpath(sys.executable)" in source
    assert '0|0,1,2,3)' in source
    assert 'CURAND_HOME="$_umift_site/nvidia/curand"' in source
    assert 'CUDNN_HOME="$_umift_site/nvidia/cudnn"' in source
    assert 'NVRTC_HOME="$_umift_site/nvidia/cuda_nvrtc"' in source
    assert 'I4_ATTN_BACKENDS="${I4_ATTN_BACKENDS:-natten}"' in source
    assert "unset I4_ATTN_BACKENDS_MULTIDIM" in source
    assert "VIRTUAL_ENV" not in source
    assert "UV_PYTHON_" not in source
    assert "/data/cosmos_envs" not in source


def test_a40_env_uses_only_cosmos_cache_namespaces() -> None:
    source = ENV_PATH.read_text(encoding="utf-8")
    assert "CONDA_PKGS_DIRS=/data/cosmos_conda/pkgs" in source
    assert "PIP_CACHE_DIR=/data/cosmos_conda/cache/pip" in source
    assert "UV_CACHE_DIR=/data/cosmos_conda/cache/uv" in source
    assert "HF_HOME=/data/cosmos_models/cache/huggingface" in source
    assert "TMPDIR=/data/cosmos_runs/tmp" in source
    assert (
        'EDGE_HF_SNAPSHOT_PATH="${EDGE_HF_SNAPSHOT_PATH:-/data/cosmos_models/Cosmos3-Edge/snapshots/'
        'a9d944e2c6a1bf9f48b92ad16348e70c5f1836ba}"' in source
    )
    assert (
        'WAN_VAE_PATH="${WAN_VAE_PATH:-/data/cosmos_models/Wan2.2-VAE-921dbaf/Wan2.2_VAE.pth}"'
        in source
    )
    assert "BASE_CHECKPOINT_PATH" not in source


def test_a40_env_checks_conda_prefix_python_and_component_paths(tmp_path: Path) -> None:
    conda_root = tmp_path / "miniconda3"
    env = conda_root / "envs/cosmos_edge_e1"
    (conda_root / "etc/profile.d").mkdir(parents=True)
    (env / "bin").mkdir(parents=True)
    for component in ("curand", "cudnn", "cuda_nvrtc"):
        (env / f"lib/python3.13/site-packages/nvidia/{component}").mkdir(parents=True)
    (conda_root / "etc/profile.d/conda.sh").write_text(
        'conda() { [[ "$1" == activate ]] || return 9; export CONDA_PREFIX="$2"; export PATH="$2/bin:$PATH"; }\n',
        encoding="utf-8",
    )
    python = env / "bin/python"
    python.write_text(f"#!/usr/bin/env bash\necho '{env}/bin/python3.13'\n", encoding="utf-8")
    python.chmod(0o755)
    rewritten = tmp_path / "a40_env.sh"
    replacements = {
        "/data/miniconda3": str(conda_root),
        "/data/cosmos_conda": str(tmp_path / "cosmos_conda"),
        "/data/cosmos_models": str(tmp_path / "cosmos_models"),
        "/data/cosmos_runs": str(tmp_path / "cosmos_runs"),
    }
    # Replace once: A40's tmp_path itself lives under /data/cosmos_runs.
    source = re.sub(
        "|".join(re.escape(path) for path in replacements),
        lambda match: replacements[match.group()],
        ENV_PATH.read_text(encoding="utf-8"),
    )
    rewritten.write_text(source, encoding="utf-8")
    result = subprocess.run(
        ["bash", "-c", f'export CUDA_VISIBLE_DEVICES=0; source "{rewritten}" && printf "%s" "$CONDA_PREFIX"'],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == str(env)

    wrong_conda = conda_root / "etc/profile.d/conda.sh"
    wrong_conda.write_text(
        'conda() { export CONDA_PREFIX="/outside/wrong"; export PATH="$2/bin:$PATH"; }\n', encoding="utf-8"
    )
    rejected = subprocess.run(
        ["bash", "-c", f'export CUDA_VISIBLE_DEVICES=0; source "{rewritten}" && echo SHOULD_NOT_RUN'],
        capture_output=True, text=True,
    )
    assert rejected.returncode == 2
    assert "wrong Conda environment active" in rejected.stderr
    assert "SHOULD_NOT_RUN" not in rejected.stdout


def test_local_artifacts_reject_remote_or_ambiguous_sources(tmp_path: Path) -> None:
    dcp = tmp_path / "dcp"
    hf = tmp_path / preflight.EDGE_REVISION
    vae = tmp_path / "vae.pth"
    dcp.mkdir()
    hf.mkdir()
    vae.write_bytes(b"x")
    (hf / "config.json").write_text("{}")
    (hf / "tokenizer.json").write_text("{}")
    (dcp / "model").mkdir()
    (dcp / "model" / ".metadata").write_bytes(b"x")

    result = preflight.validate_local_artifacts(dcp=dcp, hf_snapshot=hf, vae=vae)
    assert result["dcp"] == str(dcp.resolve())
    assert result["hf_snapshot"] == str(hf.resolve())

    with pytest.raises(ValueError, match="local filesystem"):
        preflight.validate_local_artifacts(dcp=Path("nvidia/Cosmos3-Edge"), hf_snapshot=hf, vae=vae)
    wrong = tmp_path / "main"
    wrong.mkdir()
    (wrong / "config.json").write_text("{}")
    (wrong / "tokenizer.json").write_text("{}")
    with pytest.raises(ValueError, match="revision"):
        preflight.validate_local_artifacts(dcp=dcp, hf_snapshot=wrong, vae=vae)


def test_config_contract_checks_training_semantics() -> None:
    config = ns(
        model=ns(config=ns(
            action_gen=True, vision_gen=True, sound_gen=False,
            joint_attn_implementation="two_way", resolution="256", precision="bfloat16",
            state_ch=48, max_action_dim=64,
            diffusion_expert_config=ns(load_weights_from_pretrained=False),
            tokenizer=ns(encode_exact_durations=[17]),
            vlm_config=ns(tokenizer=ns(tokenizer_type=f"/models/{preflight.EDGE_REVISION}")),
            compile=ns(enabled=False), ema=ns(enabled=False),
            activation_checkpointing=ns(mode="selective"),
            parallelism=ns(data_parallel_shard_degree=4, data_parallel_replicate_degree=1,
                           fsdp_master_dtype="float32", fsdp_reduce_dtype="bfloat16"),
        )),
        optimizer=ns(lr=1e-5, betas=[0.9, 0.99], eps=1e-8, weight_decay=0.05,
                     keys_to_select=preflight.EXPECTED_OPTIMIZER_KEYS,
                     lr_multipliers={key: 5.0 for key in preflight.ACTION_LR_KEYS}),
        scheduler=ns(cycle_lengths=[1000], warm_up_steps=[100], f_max=[1.0], f_min=[0.1], f_start=[0.0]),
        trainer=ns(grad_accum_iter=4, max_iter=1000, straggler_detection=ns(enabled=False)),
        checkpoint=ns(load_training_state=False, strict_resume=True),
        dataloader_train=ns(max_samples_per_batch=1, max_sequence_length=None),
    )
    report = preflight.assert_edge_fd_config(config)
    assert report["global_samples_per_update"] == 16
    config.model.config.action_gen = False
    with pytest.raises(AssertionError, match="action_gen"):
        preflight.assert_edge_fd_config(config)


def test_optimizer_report_rejects_unexpected_lr_and_nonfinite_grad() -> None:
    params = [
        (f"{key}.weight", ns(numel=lambda: 4, requires_grad=True, grad=ns(isfinite=True)))
        for key in preflight.EXPECTED_OPTIMIZER_KEYS
    ] + [("language_model.und.weight", ns(numel=lambda: 16, requires_grad=False, grad=None))]
    base_params = [p for name, p in params if not any(key in name for key in preflight.ACTION_LR_KEYS) and p.requires_grad]
    action_params = [p for name, p in params if any(key in name for key in preflight.ACTION_LR_KEYS)]
    groups = [
        {"lr": 1e-5, "params": base_params},
        {"lr": 5e-5, "params": action_params},
    ]
    report = preflight.summarize_optimizer(params, groups, finite_check=lambda grad: grad.isfinite)
    assert report["trainable_elements"] == 28
    assert report["lr_group_elements"] == {"1e-05": 16, "5e-05": 12}

    groups[1]["lr"] = 2e-5
    with pytest.raises(AssertionError, match="unexpected optimizer LR"):
        preflight.summarize_optimizer(params, groups, finite_check=lambda grad: grad.isfinite)


def test_optimizer_report_uses_scheduler_base_lr_after_zero_start() -> None:
    params = [
        (f"{key}.weight", ns(numel=lambda: 4, requires_grad=True, grad=None))
        for key in preflight.EXPECTED_OPTIMIZER_KEYS
    ]
    base_params = [p for name, p in params if not any(key in name for key in preflight.ACTION_LR_KEYS)]
    action_params = [p for name, p in params if any(key in name for key in preflight.ACTION_LR_KEYS)]
    groups = [
        {"lr": 0.0, "initial_lr": 1e-5, "params": base_params},
        {"lr": 0.0, "initial_lr": 5e-5, "params": action_params},
    ]
    report = preflight.summarize_optimizer(params, groups)
    assert report["lr_group_elements"] == {"1e-05": 16, "5e-05": 12}
    assert report["current_lrs"] == [0.0, 0.0]


def test_checkpoint_source_distinguishes_warmstart_and_resume() -> None:
    warm = ns(warm_start=True, path="/base")
    resume = ns(warm_start=False, path="/run/checkpoints/iter_000000005")
    assert preflight.assert_checkpoint_source({"model"}, warm, "warmstart")["kind"] == "warmstart"
    full = {"model", "optim", "scheduler", "trainer", "dataloader"}
    assert preflight.assert_checkpoint_source(full, resume, "resume")["kind"] == "resume"
    with pytest.raises(AssertionError, match="expected resume"):
        preflight.assert_checkpoint_source({"model"}, warm, "resume")
