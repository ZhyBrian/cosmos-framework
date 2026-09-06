"""E0 checks for the UMI-FT Cosmos3-Edge forward-dynamics experiment.

No subcommand enters the trainer loop. ``attention`` is a one-GPU SM86 BF16
ordinary-varlen NATTEN forward/backward probe. ``config`` is CPU-only.
``model`` performs exact config/model/optimizer/checkpoint construction under
torchrun, then exits before loading data or taking an optimizer step.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Iterable


EXPECTED_OPTIMIZER_KEYS = [
    "moe_gen", "time_embedder", "vae2llm", "llm2vae",
    "action2llm", "llm2action", "action_modality_embed",
]
ACTION_LR_KEYS = ["action2llm", "llm2action", "action_modality_embed"]
EDGE_REVISION = "a9d944e2c6a1bf9f48b92ad16348e70c5f1836ba"


def _get(obj: Any, dotted: str) -> Any:
    for part in dotted.split("."):
        obj = obj[part] if isinstance(obj, dict) else getattr(obj, part)
    return obj


def _expect(config: Any, dotted: str, expected: Any) -> None:
    actual = _get(config, dotted)
    assert actual == expected, f"{dotted}: expected {expected!r}, got {actual!r}"


def validate_local_artifacts(*, dcp: Path, hf_snapshot: Path, vae: Path) -> dict[str, str]:
    """Fail before CUDA init if any model input is not an explicit local artifact."""
    resolved: dict[str, str] = {}
    for label, value, kind in (("dcp", dcp, "dir"), ("hf_snapshot", hf_snapshot, "dir"), ("vae", vae, "file")):
        raw = str(value)
        if "://" in raw or not value.is_absolute():
            raise ValueError(f"{label} must be an absolute local filesystem path, got {raw!r}")
        if kind == "dir" and not value.is_dir():
            raise FileNotFoundError(f"{label} directory not found: {value}")
        if kind == "file" and not value.is_file():
            raise FileNotFoundError(f"{label} file not found: {value}")
        resolved[label] = str(value.resolve())

    for required in ("config.json", "tokenizer.json"):
        if not (hf_snapshot / required).is_file():
            raise FileNotFoundError(f"Edge HF snapshot lacks {required}: {hf_snapshot}")
    if hf_snapshot.name != EDGE_REVISION:
        raise ValueError(f"Edge HF snapshot directory must identify fixed revision {EDGE_REVISION}, got {hf_snapshot.name!r}")
    if not (dcp / "model" / ".metadata").is_file():
        raise FileNotFoundError(
            f"training warm-start expects an iteration/conversion root containing model/.metadata: {dcp}"
        )
    return resolved


def assert_edge_fd_config(config: Any) -> dict[str, Any]:
    checks = {
        "model.config.action_gen": True,
        "model.config.vision_gen": True,
        "model.config.sound_gen": False,
        "model.config.joint_attn_implementation": "two_way",
        "model.config.resolution": "256",
        "model.config.precision": "bfloat16",
        "model.config.state_ch": 48,
        "model.config.max_action_dim": 64,
        "model.config.diffusion_expert_config.load_weights_from_pretrained": False,
        "model.config.tokenizer.encode_exact_durations": [17],
        "model.config.compile.enabled": False,
        "model.config.ema.enabled": False,
        "model.config.activation_checkpointing.mode": "selective",
        "model.config.parallelism.data_parallel_shard_degree": 4,
        "model.config.parallelism.data_parallel_replicate_degree": 1,
        "model.config.parallelism.fsdp_master_dtype": "float32",
        "model.config.parallelism.fsdp_reduce_dtype": "bfloat16",
        "optimizer.lr": 1e-5,
        "optimizer.betas": [0.9, 0.99],
        "optimizer.eps": 1e-8,
        "optimizer.weight_decay": 0.05,
        "optimizer.keys_to_select": EXPECTED_OPTIMIZER_KEYS,
        "scheduler.cycle_lengths": [1000],
        "scheduler.warm_up_steps": [100],
        "scheduler.f_max": [1.0],
        "scheduler.f_min": [0.1],
        "scheduler.f_start": [0.0],
        "trainer.grad_accum_iter": 4,
        "checkpoint.load_training_state": False,
        "checkpoint.strict_resume": True,
        "dataloader_train.max_samples_per_batch": 1,
        "dataloader_train.max_sequence_length": None,
    }
    for dotted, expected in checks.items():
        _expect(config, dotted, expected)
    for key in ACTION_LR_KEYS:
        _expect(config, f"optimizer.lr_multipliers.{key}", 5.0)
    tokenizer_source = str(_get(config, "model.config.vlm_config.tokenizer.tokenizer_type"))
    assert Path(tokenizer_source).is_absolute(), f"Edge tokenizer source must be local absolute path: {tokenizer_source}"
    assert Path(tokenizer_source).name == EDGE_REVISION, (
        f"Edge tokenizer source must be fixed revision {EDGE_REVISION}, got {tokenizer_source}"
    )
    return {
        "max_optimizer_updates": int(_get(config, "trainer.max_iter")),
        "microbatches_per_rank_per_update": 4,
        "world_size": 4,
        "global_samples_per_update": 16,
    }


def summarize_optimizer(
    named_parameters: Iterable[tuple[str, Any]],
    param_groups: Iterable[dict[str, Any]],
    *,
    finite_check: Callable[[Any], bool] | None = None,
) -> dict[str, Any]:
    named = list(named_parameters)
    trainable = [(name, p) for name, p in named if p.requires_grad]
    unexpected = [name for name, _ in trainable if not any(key in name for key in EXPECTED_OPTIMIZER_KEYS)]
    assert not unexpected, f"trainable params outside optimizer contract: {unexpected[:20]}"
    missing_families = [key for key in EXPECTED_OPTIMIZER_KEYS if not any(key in name for name, _ in trainable)]
    assert not missing_families, f"optimizer parameter families absent: {missing_families}"

    lr_elements: dict[str, int] = {}
    current_lrs: list[float] = []
    valid_lrs = {1e-5, 5e-5}
    for group in param_groups:
        current_lrs.append(float(group["lr"]))
        # LambdaLR applies f_start=0 during construction.  initial_lr is the
        # configured peak/base LR and is therefore the value whose grouping
        # proves the 1x/5x optimizer contract.
        base_lr = float(group.get("initial_lr", group["lr"]))
        assert base_lr in valid_lrs, f"unexpected optimizer LR {base_lr}"
        key = f"{base_lr:g}"
        lr_elements[key] = lr_elements.get(key, 0) + sum(int(p.numel()) for p in group["params"])
    assert set(lr_elements) == {"1e-05", "5e-05"}, f"expected base/action LR groups, got {lr_elements}"

    if finite_check is not None:
        nonfinite = [name for name, p in trainable if p.grad is not None and not finite_check(p.grad)]
        assert not nonfinite, f"non-finite gradients: {nonfinite[:20]}"
    return {
        "parameter_tensors": len(named),
        "trainable_tensors": len(trainable),
        "trainable_elements": sum(int(p.numel()) for _, p in trainable),
        "lr_group_elements": lr_elements,
        "current_lrs": current_lrs,
    }


def assert_resume_rng_metadata(source: Any, *, rank: int) -> str:
    """Prove that a full resume contains the RNG leaf consumed by DCP.load."""
    from torch.distributed.checkpoint.filesystem import FileSystemReader

    trainer_path = Path(source.path) / "trainer"
    metadata = FileSystemReader(str(trainer_path)).read_metadata()
    rng_key = f"rng_state_{rank}"
    keys = metadata.state_dict_metadata
    assert any(key == rng_key or key.startswith(f"{rng_key}.") for key in keys), (
        f"resume trainer metadata lacks {rng_key}: {trainer_path}"
    )
    return rng_key


def assert_checkpoint_source(resume_keys: set[str], source: Any, expected: str) -> dict[str, Any]:
    assert source is not None, f"expected {expected} checkpoint source, got none"
    actual = "warmstart" if source.warm_start else "resume"
    assert actual == expected, f"expected {expected} checkpoint source, got {actual}: {source.path}"
    expected_keys = {"model"} if expected == "warmstart" else {"model", "optim", "scheduler", "trainer", "dataloader"}
    assert set(resume_keys) == expected_keys, f"{actual} expected keys {sorted(expected_keys)}, got {sorted(resume_keys)}"
    return {"kind": actual, "path": str(source.path), "keys": sorted(resume_keys)}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def environment_report(artifacts: dict[str, str] | None = None) -> dict[str, Any]:
    report: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "env": {key: os.environ.get(key) for key in (
            "CUDA_VISIBLE_DEVICES", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE",
            "I4_ATTN_BACKENDS", "I4_ATTN_BACKENDS_MULTIDIM",
        )},
    }
    try:
        import torch
        report["torch"] = torch.__version__
        report["cuda_runtime"] = torch.version.cuda
        report["cuda_devices"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        report["cuda_capabilities"] = [list(torch.cuda.get_device_capability(i)) for i in range(torch.cuda.device_count())]
    except Exception as exc:
        report["torch_error"] = repr(exc)
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,driver_version,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, check=True,
        )
        report["nvidia_smi"] = result.stdout.strip().splitlines()
    except Exception as exc:
        report["nvidia_smi_error"] = repr(exc)
    if artifacts:
        report["artifacts"] = artifacts
        hf = Path(artifacts["hf_snapshot"])
        vae = Path(artifacts["vae"])
        report["artifact_hashes"] = {
            "dcp_model_metadata": _sha256(Path(artifacts["dcp"]) / "model" / ".metadata"),
            "hf_config.json": _sha256(hf / "config.json"),
            "hf_tokenizer.json": _sha256(hf / "tokenizer.json"),
            "vae": _sha256(vae),
        }
    return report


def _offline_env() -> None:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"


def _load_config(args: argparse.Namespace, artifacts: dict[str, str]) -> Any:
    _offline_env()
    os.environ["BASE_CHECKPOINT_PATH"] = artifacts["dcp"]
    os.environ["WAN_VAE_PATH"] = artifacts["vae"]
    os.environ["EDGE_HF_SNAPSHOT_PATH"] = artifacts["hf_snapshot"]
    os.environ.setdefault("DATASET_PATH", "/data/cosmos_datasets/umift_us_all_source_308f4d46.zarr")
    from cosmos_framework.configs.toml_config.sft_config import load_experiment_from_toml
    config = load_experiment_from_toml(args.toml)
    return config


def run_config(args: argparse.Namespace) -> dict[str, Any]:
    artifacts = validate_local_artifacts(dcp=args.dcp, hf_snapshot=args.hf_snapshot, vae=args.vae)
    config = _load_config(args, artifacts)
    return {"config": assert_edge_fd_config(config), "environment": environment_report(artifacts)}


def run_attention() -> dict[str, Any]:
    os.environ["I4_ATTN_BACKENDS"] = "natten"
    assert "I4_ATTN_BACKENDS_MULTIDIM" not in os.environ, "unset I4_ATTN_BACKENDS_MULTIDIM for ordinary varlen"
    import torch
    assert torch.cuda.is_available(), "CUDA is unavailable"
    assert torch.cuda.device_count() == 1, "run as CUDA_VISIBLE_DEVICES=0; attention probe requires exactly one visible GPU"
    capability = torch.cuda.get_device_capability(0)
    assert capability == (8, 6), f"expected A40 SM86, got SM{capability[0]}{capability[1]}"
    assert torch.cuda.is_bf16_supported(), "SM86 runtime reports BF16 unsupported"

    from cosmos_framework.model.attention import attention
    from cosmos_framework.model.attention.varlen import generate_varlen_parameters
    torch.manual_seed(42)
    device = torch.device("cuda:0")
    q = torch.randn(1, 11, 4, 64, device=device, dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(1, 11, 4, 64, device=device, dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn(1, 11, 4, 64, device=device, dtype=torch.bfloat16, requires_grad=True)
    lengths = torch.tensor([5, 6], device=device, dtype=torch.int32)
    cu_q, cu_k, max_q, max_k = generate_varlen_parameters(q, k, v, lengths, lengths)
    output = attention(
        q, k, v,
        cumulative_seqlen_Q=cu_q, cumulative_seqlen_KV=cu_k,
        max_seqlen_Q=max_q, max_seqlen_KV=max_k,
        backend="natten",
    )
    assert isinstance(output, torch.Tensor) and output.shape == q.shape
    assert torch.isfinite(output).all().item(), "NATTEN forward produced non-finite values"
    output.float().square().mean().backward()
    for name, tensor in (("q", q), ("k", k), ("v", v)):
        assert tensor.grad is not None and torch.isfinite(tensor.grad).all().item(), f"{name} grad is non-finite"
        assert tensor.grad.abs().sum().item() > 0, f"{name} grad is all zero"
    return {"device": torch.cuda.get_device_name(0), "capability": list(capability), "dtype": "bfloat16",
            "backend": "natten", "layout": "ordinary_varlen_1d", "segments": [5, 6], "finite_grads": True}


def run_model(args: argparse.Namespace) -> dict[str, Any]:
    artifacts = validate_local_artifacts(dcp=args.dcp, hf_snapshot=args.hf_snapshot, vae=args.vae)
    config = _load_config(args, artifacts)
    config_report = assert_edge_fd_config(config)
    import torch
    from cosmos_framework.utils import distributed
    from cosmos_framework.utils.context_managers import distributed_init, model_init
    from cosmos_framework.utils.lazy_config import instantiate

    with distributed_init():
        distributed.init()
    assert torch.distributed.get_world_size() == 4, "model preflight requires torchrun --nproc_per_node=4"
    config.validate()
    config.freeze()
    trainer = config.trainer.type(config)
    resume_keys, source = trainer.checkpointer.keys_to_resume_during_load()
    checkpoint_report = assert_checkpoint_source(set(resume_keys), source, args.expect_load)
    with model_init():
        model = instantiate(config.model)
    model = model.to("cuda", memory_format=config.trainer.memory_format)
    model.on_train_start(config.trainer.memory_format)
    trainer.callbacks.on_optimizer_init_start()
    optimizer, scheduler = model.init_optimizer_scheduler(config.optimizer, config.scheduler)
    scaler = torch.amp.GradScaler("cuda", **config.trainer.grad_scaler_args)
    trainer.callbacks.on_optimizer_init_end()
    rng_key = None
    if args.expect_load == "resume":
        rng_key = assert_resume_rng_metadata(source, rank=torch.distributed.get_rank())
    loaded_iteration = trainer.checkpointer.load(model, optimizer, scheduler, scaler)
    inner_optimizers = list(optimizer.optimizers) if hasattr(optimizer, "optimizers") else [optimizer]
    param_groups = [group for inner in inner_optimizers for group in inner.param_groups]
    optimizer_report = summarize_optimizer(
        model.net.named_parameters(), param_groups,
        finite_check=lambda grad: bool(torch.isfinite(grad).all().item()),
    )
    checkpoint_report["rng_key"] = rng_key
    return {"config": config_report, "checkpoint": checkpoint_report, "optimizer": optimizer_report,
            "loaded_iteration": int(loaded_iteration), "environment": environment_report(artifacts)}


def _artifact_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--toml", type=Path, required=True)
    parser.add_argument("--dcp", type=Path, required=True)
    parser.add_argument("--hf-snapshot", type=Path, required=True)
    parser.add_argument("--vae", type=Path, required=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    _artifact_args(sub.add_parser("config", help="CPU compose and offline-artifact check"))
    sub.add_parser("attention", help="single-GPU SM86 BF16 ordinary-varlen NATTEN probe")
    model_parser = sub.add_parser("model", help="4-rank model/DCP/optimizer construction; no training")
    _artifact_args(model_parser)
    model_parser.add_argument("--expect-load", choices=("warmstart", "resume"), default="warmstart")
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args(argv)
    report = run_attention() if args.command == "attention" else (run_config(args) if args.command == "config" else run_model(args))
    payload = json.dumps(report, indent=2, sort_keys=True)
    is_primary = int(os.environ.get("RANK", "0")) == 0
    if is_primary:
        print(payload)
    if args.json_out and is_primary:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(payload + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
