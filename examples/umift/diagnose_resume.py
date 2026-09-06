"""Trace update 6 of the UMI Edge FD run without changing its computation.

Invoke this module with the same arguments passed to ``cosmos_framework.scripts.train``.
Each torchrun rank writes its own JSONL file below ``COSMOS_DIAG_DIR``.
"""

from __future__ import annotations

import dataclasses
import hashlib
import inspect
import json
import math
import os
import pickle
import random
import runpy
import sys
from pathlib import Path
from typing import Any, Callable


TRACE_ITERATION = int(os.environ.get("COSMOS_DIAG_ITERATION", "5"))
TRACE_DIR = Path(os.environ.get("COSMOS_DIAG_DIR", "/data/cosmos_runs/umift_edge_fd_resume_diagnostic"))
_MANIFEST: dict[str, Any] | None = None


def _digest_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _tensor_record(value: Any) -> dict[str, Any]:
    import torch

    local = value.to_local() if hasattr(value, "to_local") else value
    tensor = local.detach().contiguous()
    raw = tensor.reshape(-1).view(torch.uint8).cpu().numpy().tobytes()
    return {"shape": list(tensor.shape), "dtype": str(tensor.dtype), "sha256": _digest_bytes(raw)}


def _record(value: Any) -> Any:
    import torch

    if isinstance(value, torch.Tensor) or hasattr(value, "to_local"):
        return _tensor_record(value)
    if dataclasses.is_dataclass(value):
        return {field.name: _record(getattr(value, field.name)) for field in dataclasses.fields(value)}
    if isinstance(value, dict):
        return {str(key): _record(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_record(item) for item in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return {"type": type(value).__name__, "repr_sha256": _digest_bytes(repr(value).encode())}


def _rng_record() -> dict[str, Any]:
    import numpy as np
    import torch

    return {
        "python": _digest_bytes(pickle.dumps(random.getstate(), protocol=5)),
        "numpy": _digest_bytes(pickle.dumps(np.random.get_state(), protocol=5)),
        "torch_cpu": _tensor_record(torch.random.get_rng_state()),
        "torch_cuda": {
            "current_device": torch.cuda.current_device(),
            "state": _tensor_record(torch.cuda.get_rng_state()),
        },
    }


def _write(event: dict[str, Any]) -> None:
    rank = int(os.environ.get("RANK", "0"))
    path = TRACE_DIR / f"rank{rank}.jsonl"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"rank": rank, **event}, sort_keys=True) + "\n")


def _optimizer_record(model: Any, optimizer: Any) -> dict[str, Any]:
    names = {id(parameter): name for name, parameter in model.net.named_parameters() if parameter.requires_grad}
    groups = []
    for optimizer_index, inner in enumerate(optimizer.optimizers):
        for group_index, group in enumerate(inner.param_groups):
            parameters = []
            for parameter in group["params"]:
                state = inner.state.get(parameter, {})
                parameters.append({
                    "name": names.get(id(parameter), "<unknown>"),
                    "parameter": _tensor_record(parameter),
                    "gradient": None if parameter.grad is None else _tensor_record(parameter.grad),
                    "exp_avg": None if "exp_avg" not in state else _tensor_record(state["exp_avg"]),
                    "exp_avg_sq": None if "exp_avg_sq" not in state else _tensor_record(state["exp_avg_sq"]),
                    "step": _record(state.get("step")),
                })
            groups.append({
                "optimizer": optimizer_index,
                "group": group_index,
                "lr": _record(group["lr"]),
                "initial_lr": _record(group.get("initial_lr")),
                "group_state": {key: _record(value) for key, value in group.items() if key != "params"},
                "parameters": parameters,
            })
    return {"groups": groups}


def _gradient_record(model: Any) -> dict[str, Any]:
    import torch

    parameters = []
    total_sq = 0.0
    total_abs = 0.0
    for name, parameter in model.net.named_parameters():
        if parameter.grad is None:
            continue
        grad = parameter.grad.to_local() if hasattr(parameter.grad, "to_local") else parameter.grad
        grad_fp64 = grad.detach().double()
        sq = grad_fp64.square().sum()
        absolute = grad_fp64.abs().sum()
        total_sq += float(sq)
        total_abs += float(absolute)
        parameters.append({
            "name": name,
            "gradient": _tensor_record(parameter.grad),
            "l2": float(sq.sqrt()),
            "max_abs": float(grad_fp64.abs().max()),
            "sum_abs": float(absolute),
            "finite": bool(torch.isfinite(grad).all()),
        })
    return {"parameters": parameters, "local_l2": math.sqrt(total_sq), "local_sum_abs": total_abs}


def _wrap_method(cls: type, name: str, capture: Callable[[tuple[Any, ...], dict[str, Any], Any], None]) -> None:
    original = getattr(cls, name)

    def wrapped(self: Any, *args: Any, **kwargs: Any) -> Any:
        result = original(self, *args, **kwargs)
        if getattr(self, "_umift_diag_active", False):
            capture(args, kwargs, result)
        return result

    setattr(cls, name, wrapped)


def install_diagnostics() -> None:
    import cosmos_framework.callbacks.grad_clip as grad_clip_module
    from cosmos_framework.model.generator.omni_mot_model import OmniMoTModel
    from cosmos_framework.trainer import ImaginaireTrainer

    captures: dict[str, Any] = {}

    def capture_training_inputs(_args: tuple[Any, ...], _kwargs: dict[str, Any], result: Any) -> None:
        captures["training_inputs"] = {"value": _record(result), "rng_after": _rng_record()}

    def capture_noise_level(_args: tuple[Any, ...], _kwargs: dict[str, Any], result: Any) -> None:
        captures["vision_noise_level"] = {"value": _record(result), "rng_after": _rng_record()}

    noising_signature = inspect.signature(OmniMoTModel._add_noise_to_input)

    def capture_noising(args: tuple[Any, ...], kwargs: dict[str, Any], result: Any) -> None:
        bound = noising_signature.bind(None, *args, **kwargs)
        packed = bound.arguments["packed_sequence"]
        captures["noising"] = {
            "sigmas": _record(bound.arguments["sigmas"]),
            "condition_mask_vision": _record(packed.vision.condition_mask if packed.vision else None),
            "condition_mask_action": _record(packed.action.condition_mask if packed.action else None),
            "result": _record(result),
            "rng_after": _rng_record(),
        }

    _wrap_method(OmniMoTModel, "_get_training_inputs", capture_training_inputs)
    _wrap_method(OmniMoTModel, "_get_train_noise_level_vision", capture_noise_level)
    _wrap_method(OmniMoTModel, "_add_noise_to_input", capture_noising)
    _wrap_method(OmniMoTModel, "denoise", lambda _a, _k, out: captures.__setitem__("prediction", _record(out)))
    _wrap_method(OmniMoTModel, "_compute_losses", lambda _a, _k, out: captures.__setitem__("losses", _record(out)))

    original_training_step = OmniMoTModel.training_step
    trace_micro_slot = 0
    manifest_written = False

    def training_step(self: Any, data_batch: Any, iteration: int) -> Any:
        import torch

        nonlocal manifest_written, trace_micro_slot
        if not manifest_written:
            assert _MANIFEST is not None
            _write({**_MANIFEST, "cuda_current_device": torch.cuda.current_device()})
            manifest_written = True
        active = iteration == TRACE_ITERATION
        self._umift_diag_active = active
        if active:
            captures.clear()
            before = _rng_record()
            batch = _record(data_batch)
            buffers = {name: _record(value) for name, value in self.net.named_buffers()}
        try:
            result = original_training_step(self, data_batch, iteration)
        finally:
            self._umift_diag_active = False
        if active:
            _write({"event": "microbatch", "iteration": iteration, "micro_slot": trace_micro_slot,
                    "input": batch, "buffers_before": buffers, "rng_before": before,
                    "rng_after": _rng_record(), "captures": captures, "output": _record(result)})
            trace_micro_slot += 1
        return result

    OmniMoTModel.training_step = training_step

    backward_slot = 0
    original_after_backward = OmniMoTModel.on_after_backward

    def after_backward(self: Any, *args: Any, **kwargs: Any) -> Any:
        nonlocal backward_slot
        result = original_after_backward(self, *args, **kwargs)
        if getattr(self, "_umift_diag_backward_active", False):
            _write({"event": "backward", "iteration": TRACE_ITERATION, "micro_slot": backward_slot,
                    "cumulative_raw_gradient": _gradient_record(self), "rng_after": _rng_record()})
            backward_slot += 1
        return result

    OmniMoTModel.on_after_backward = after_backward

    original_trainer_step = ImaginaireTrainer.training_step

    def trainer_step(self: Any, model_ddp: Any, *args: Any, **kwargs: Any) -> Any:
        bound = inspect.signature(original_trainer_step).bind(self, model_ddp, *args, **kwargs)
        active = bound.arguments.get("iteration", 0) == TRACE_ITERATION
        model_ddp._umift_diag_backward_active = active
        try:
            return original_trainer_step(self, model_ddp, *args, **kwargs)
        finally:
            model_ddp._umift_diag_backward_active = False

    ImaginaireTrainer.training_step = trainer_step

    norm_capture: dict[str, Any] = {}
    norm_active = False
    original_total_norm = grad_clip_module._total_norm_by_mesh

    def total_norm(*args: Any, **kwargs: Any) -> Any:
        result = original_total_norm(*args, **kwargs)
        if norm_active:
            norm_capture["total"] = _record(result[0])
            norm_capture["total_value"] = float(result[0])
            norm_capture["per_mesh"] = _record(result[1])
        return result

    grad_clip_module._total_norm_by_mesh = total_norm
    original_grad_clip = grad_clip_module.GradClip.on_before_optimizer_step

    def grad_clip(self: Any, model: Any, *args: Any, **kwargs: Any) -> Any:
        nonlocal norm_active
        bound = inspect.signature(original_grad_clip).bind(self, model, *args, **kwargs)
        iteration = bound.arguments.get("iteration", 0)
        if iteration != TRACE_ITERATION:
            return original_grad_clip(self, model, *args, **kwargs)
        norm_capture.clear()
        before = _gradient_record(model)
        norm_active = True
        try:
            result = original_grad_clip(self, model, *args, **kwargs)
        finally:
            norm_active = False
        _write({"event": "grad_clip", "iteration": iteration, "before": before,
                "computed_norm": dict(norm_capture), "after": _gradient_record(model)})
        return result

    grad_clip_module.GradClip.on_before_optimizer_step = grad_clip

    original_optimizer_step = ImaginaireTrainer._optimizer_step

    def optimizer_step(self: Any, model: Any, optimizer: Any, scheduler: Any, grad_scaler: Any, iteration: int) -> None:
        if iteration != TRACE_ITERATION:
            return original_optimizer_step(self, model, optimizer, scheduler, grad_scaler, iteration)
        before = _optimizer_record(model, optimizer)
        result = original_optimizer_step(self, model, optimizer, scheduler, grad_scaler, iteration)
        _write({"event": "optimizer", "iteration": iteration, "before": before,
                "after": _optimizer_record(model, optimizer), "rng_after": _rng_record()})
        return result

    ImaginaireTrainer._optimizer_step = optimizer_step


def main() -> None:
    global _MANIFEST
    rank = int(os.environ.get("RANK", "0"))
    path = TRACE_DIR / f"rank{rank}.jsonl"
    if path.exists() and path.stat().st_size:
        raise FileExistsError(f"refusing to reuse non-empty diagnostic output: {path}")
    TRACE_DIR.mkdir(parents=True, exist_ok=True)
    try:
        import subprocess

        git_head = subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        git_head = "unavailable"
    import torch

    _MANIFEST = {
        "event": "manifest",
        "git_head": git_head,
        "argv": list(sys.argv),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "torch_version": torch.__version__,
        "python": sys.version,
        "trace_iteration": TRACE_ITERATION,
        "environment": {
            key: os.environ.get(key)
            for key in (
                "WORLD_SIZE", "RANK", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT",
                "I4_ATTN_BACKENDS", "I4_ATTN_BACKENDS_MULTIDIM", "COSMOS_DEVICE",
                "DATASET_PATH", "BASE_CHECKPOINT_PATH", "WAN_VAE_PATH", "EDGE_HF_SNAPSHOT_PATH",
            )
        },
    }
    install_diagnostics()
    sys.argv[0] = "cosmos_framework.scripts.train"
    runpy.run_module("cosmos_framework.scripts.train", run_name="__main__")


if __name__ == "__main__":
    main()
