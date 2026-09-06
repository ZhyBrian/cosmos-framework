from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "examples/umift/preflight.py"
SPEC = importlib.util.spec_from_file_location("umift_preflight", PATH)
assert SPEC and SPEC.loader
preflight = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(preflight)


def ns(**kwargs):
    return SimpleNamespace(**kwargs)


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
        trainer=ns(grad_accum_iter=4, max_iter=1000),
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
