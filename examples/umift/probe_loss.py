"""Evaluate the real E1 training loss on the frozen four-window overfit set.

Run once for the base model and once for iter200 under four-rank torchrun, then
use ``compare``. No backward pass or optimizer is constructed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import types
from pathlib import Path
from typing import Any

import numpy as np


def reset_probe_rng(draw: int, rank: int) -> int:
    """Reset the global RNGs used by the experiment's normal non-deterministic path."""
    import torch

    if torch.are_deterministic_algorithms_enabled():
        raise RuntimeError("loss probe must preserve the E1 default deterministic-algorithms=False mode")
    seed = draw * 65536 + rank
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return seed


def tensor_tree_sha256(value: Any) -> str:
    import torch

    digest = hashlib.sha256()

    def visit(item: Any) -> None:
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            digest.update(str(tensor.dtype).encode())
            digest.update(str(tuple(tensor.shape)).encode())
            digest.update(tensor.view(torch.uint8).numpy().tobytes())
        elif isinstance(item, (list, tuple)):
            digest.update(type(item).__name__.encode())
            for child in item:
                visit(child)
        elif item is None:
            digest.update(b"None")
        else:
            digest.update(repr(item).encode())

    visit(value)
    return digest.hexdigest()


def compare_reports(base: dict[str, Any], trained: dict[str, Any]) -> dict[str, Any]:
    base_rows = base["rows"]
    trained_rows = trained["rows"]
    if len(base_rows) != 16 or len(trained_rows) != 16:
        raise ValueError("each report must contain four ranks x four draws = 16 rows")
    identity = ("rank", "draw", "window_id", "x0_sha256", "xt_sha256", "sigma_sha256",
                "noise_sha256", "vision_mask_sha256", "action_mask_sha256")
    for index, (left, right) in enumerate(zip(base_rows, trained_rows, strict=True)):
        mismatch = [key for key in identity if left[key] != right[key]]
        if mismatch:
            raise ValueError(f"fixture mismatch at row {index}: {mismatch}")
    base_losses = np.asarray([row["loss"] for row in base_rows], dtype=np.float64)
    trained_losses = np.asarray([row["loss"] for row in trained_rows], dtype=np.float64)
    if not np.isfinite(base_losses).all() or not np.isfinite(trained_losses).all():
        raise ValueError("loss reports contain non-finite values")
    if (base_losses < 0).any() or (trained_losses < 0).any():
        raise ValueError("flow-matching loss must be non-negative")
    base_mean = float(base_losses.mean())
    trained_mean = float(trained_losses.mean())
    return {
        "fixture_match": True,
        "base_mean": base_mean,
        "trained_mean": trained_mean,
        "base_median": float(np.median(base_losses)),
        "trained_median": float(np.median(trained_losses)),
        "trained_over_base": None if base_mean == 0.0 else trained_mean / base_mean,
        "per_row": [None if base == 0.0 else float(trained / base)
                    for base, trained in zip(base_losses, trained_losses, strict=True)],
    }


def expected_overfit_window_id(rank: int, draw: int) -> str:
    return f"episode_0:s={64 * ((draw + rank) % 4)}"


def _first_int(value: Any) -> int:
    import torch
    while isinstance(value, list):
        value = value[0]
    return int(value.item()) if isinstance(value, torch.Tensor) else int(value)


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0,1,2,3":
        raise RuntimeError("loss probe requires CUDA_VISIBLE_DEVICES=0,1,2,3 exactly")
    import torch
    import torch.distributed as dist

    from cosmos_framework.configs.toml_config.sft_config import load_experiment_from_toml
    from cosmos_framework.utils import distributed, misc
    from cosmos_framework.utils.context_managers import data_loader_init, distributed_init, model_init
    from cosmos_framework.utils.lazy_config import instantiate

    if torch.are_deterministic_algorithms_enabled():
        raise RuntimeError("do not launch this probe with --deterministic or deterministic torch settings")
    if not (args.checkpoint / "model" / ".metadata").is_file():
        raise FileNotFoundError(f"checkpoint must be an iteration root containing model/.metadata: {args.checkpoint}")
    os.environ["BASE_CHECKPOINT_PATH"] = str(args.checkpoint.resolve())
    os.environ["UMIFT_STAGE"] = "overfit"
    os.environ["IMAGINAIRE_OUTPUT_ROOT"] = str(args.scratch.resolve())
    config = load_experiment_from_toml(args.toml, extra_overrides=[f"job.name=loss_probe_{args.label}"])
    config.validate()
    config.freeze()
    trainer = config.trainer.type(config)
    resume_keys, source = trainer.checkpointer.keys_to_resume_during_load()
    if source is None or not source.warm_start or set(resume_keys) != {"model"}:
        raise RuntimeError(f"probe requires isolated model-only warmstart, got source={source}, keys={resume_keys}")

    with distributed_init():
        distributed.init()
    if dist.get_world_size() != 4:
        raise RuntimeError("loss probe requires torchrun --nproc_per_node=4")
    rank = dist.get_rank()
    with model_init():
        model = instantiate(config.model)
    with data_loader_init():
        loader = instantiate(config.dataloader_train)
    model = model.to("cuda", memory_format=config.trainer.memory_format)
    model.on_train_start(config.trainer.memory_format)
    trainer.checkpointer.load(model)
    loader.set_start_iteration(0)
    iterator = iter(loader)
    model.train()

    local_rows = []
    for draw in range(4):
        batch = misc.to(next(iterator), device="cuda")
        seed = reset_probe_rng(draw, rank)
        captured: dict[str, Any] = {}
        original = model._add_noise_to_input

        def capture(self: Any, *call_args: Any, **call_kwargs: Any) -> Any:
            result = original(*call_args, **call_kwargs)
            captured["noised"] = result
            return result

        model._add_noise_to_input = types.MethodType(capture, model)
        try:
            with torch.no_grad():
                output, loss = model.training_step(batch, draw)
        finally:
            model._add_noise_to_input = original
        noised = captured["noised"]
        global_loss = loss.detach().float().clone()
        dist.all_reduce(global_loss, op=dist.ReduceOp.SUM)
        global_loss /= dist.get_world_size()
        vision_mask = output["condition_mask_vision"]
        action_mask = output["condition_mask_action"]
        if not all(bool(torch.all(mask[0] == 1).item()) and bool(torch.all(mask[1:] == 0).item())
                   for mask in vision_mask):
            raise RuntimeError("FD vision mask must condition only latent frame zero")
        if not all(bool(torch.all(mask == 1).item()) for mask in action_mask):
            raise RuntimeError("FD action mask must condition every action")
        window_id = f"episode_{_first_int(batch['episode_id'])}:s={_first_int(batch['window_start'])}"
        expected_window = expected_overfit_window_id(rank, draw)
        if window_id != expected_window:
            raise RuntimeError(f"unexpected overfit window: expected {expected_window}, got {window_id}")
        loss_value = float(loss.detach().float().item())
        if not np.isfinite(loss_value) or loss_value < 0:
            raise RuntimeError(f"invalid flow-matching loss: {loss_value}")
        local_rows.append({
            "rank": rank,
            "draw": draw,
            "seed": seed,
            "window_id": window_id,
            "loss": loss_value,
            "global_loss": float(global_loss.item()),
            "vision_loss": float(output["flow_matching_loss_vision"].detach().float().item()),
            "action_loss": float(output["flow_matching_loss_action"].detach().float().item()),
            "x0_sha256": tensor_tree_sha256(output["x0"]),
            "xt_sha256": tensor_tree_sha256(output["xt"]),
            "sigma_sha256": tensor_tree_sha256(output["sigma"]),
            "noise_sha256": tensor_tree_sha256(noised.epsilon_vision),
            "vision_mask_sha256": tensor_tree_sha256(vision_mask),
            "action_mask_sha256": tensor_tree_sha256(action_mask),
        })
    gathered: list[list[dict[str, Any]] | None] = [None] * 4
    dist.all_gather_object(gathered, local_rows)
    rows = sorted((row for rank_rows in gathered for row in rank_rows), key=lambda row: (row["rank"], row["draw"]))
    return {"label": args.label, "checkpoint": str(args.checkpoint.resolve()), "deterministic_algorithms": False,
            "rows": rows} if rank == 0 else {}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--toml", type=Path, required=True)
    run.add_argument("--checkpoint", type=Path, required=True)
    run.add_argument("--scratch", type=Path, required=True)
    run.add_argument("--label", choices=("base", "iter200"), required=True)
    run.add_argument("--output", type=Path, required=True)
    compare = sub.add_parser("compare")
    compare.add_argument("--base", type=Path, required=True)
    compare.add_argument("--trained", type=Path, required=True)
    compare.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.command == "run":
        report = run_probe(args)
        if report:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    else:
        report = compare_reports(json.loads(args.base.read_text()), json.loads(args.trained.read_text()))
        payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
        print(payload, end="")
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(payload, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
