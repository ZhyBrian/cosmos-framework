"""Run the E1-R trainer under a 24 GiB PyTorch allocator cap on one GPU.

Launch this wrapper with ``torchrun --nproc_per_node=1``.  It deliberately
hands control to the official training entry point via :mod:`runpy` and opts
into that entry point's guarded, post-success no-finalize exit; it does not
replace the trainer or monkeypatch ``os._exit``.
Use an external NVML sampler together with the training log for resource
evidence, because the allocator cap does not cover every CUDA allocation.
"""

from __future__ import annotations

import argparse
import json
import os
import runpy
import sys
from pathlib import Path


_TARGET_BYTES = 24 * 1024**3
_OUTPUT_BASE = Path("/data/cosmos_runs")
_TOML = Path(__file__).resolve().parents[1] / "toml/sft_config/action_fd_umift_edge_refit.toml"
_JOB_NAME = "action_fd_umift_edge_e1_refit_single24"
_OVERRIDES = (
    "model.config.parallelism.data_parallel_shard_degree=1",
    "model.config.parallelism.data_parallel_replicate_degree=1",
    "trainer.grad_accum_iter=16",
    "trainer.max_iter=2",
    "checkpoint.save_iter=2",
    f"job.name={_JOB_NAME}",
    "trainer.logging_iter=1",
    "trainer.callbacks.device_monitor.every_n=1",
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="New, dedicated output root below /data/cosmos_runs.",
    )
    return parser.parse_args(argv)


def _validate_launch(output_root: Path) -> Path:
    expected_env = {
        "CUDA_VISIBLE_DEVICES": "0",
        "WORLD_SIZE": "1",
        "RANK": "0",
        "LOCAL_RANK": "0",
    }
    mismatches = {
        key: {"expected": expected, "actual": os.environ.get(key)}
        for key, expected in expected_env.items()
        if os.environ.get(key) != expected
    }
    if mismatches:
        raise RuntimeError(f"single24 requires one-rank torchrun on visible GPU 0: {mismatches}")

    resolved = output_root.resolve()
    base = _OUTPUT_BASE.resolve()
    if resolved == base or base not in resolved.parents:
        raise ValueError(f"--output-root must be a child of {base}, got {resolved}")
    if resolved.exists():
        raise FileExistsError(
            f"single24 requires a new isolated output root; path already exists: {resolved}"
        )
    if not _TOML.is_file():
        raise FileNotFoundError(f"E1-R TOML not found: {_TOML}")
    return resolved


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    output_root = _validate_launch(args.output_root)

    import torch

    torch.cuda.set_device(0)
    properties = torch.cuda.get_device_properties(0)
    total_bytes = int(properties.total_memory)
    if _TARGET_BYTES >= total_bytes:
        raise RuntimeError(
            f"24 GiB cap requires a larger proxy GPU, got total_memory={total_bytes} bytes"
        )
    fraction = _TARGET_BYTES / total_bytes
    torch.cuda.set_per_process_memory_fraction(fraction, device=0)
    torch.cuda.reset_peak_memory_stats(0)

    os.environ["OUTPUT_ROOT"] = str(output_root)
    os.environ["IMAGINAIRE_OUTPUT_ROOT"] = str(output_root)
    os.environ["UMIFT_STAGE"] = "e1"
    os.environ["COSMOS_EXIT_WITHOUT_FINALIZE"] = "1"

    report = {
        "probe": "single24",
        "device": 0,
        "device_name": properties.name,
        "compute_capability": [properties.major, properties.minor],
        "hardware_total_bytes": total_bytes,
        "allocator_target_bytes": _TARGET_BYTES,
        "allocator_target_gib": _TARGET_BYTES / 1024**3,
        "allocator_fraction": fraction,
        "world_size": 1,
        "output_root": str(output_root),
        "toml": str(_TOML),
        "job_name": _JOB_NAME,
        "cosmos_exit_without_finalize": True,
        "overrides": list(_OVERRIDES),
        "resource_evidence": (
            "Capture whole-process GPU memory with an external NVML sampler and retain the "
            "training log. The PyTorch caching-allocator cap excludes some CUDA context, "
            "library workspace, communicator, and other external allocations."
        ),
    }
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)

    sys.argv = [
        "cosmos_framework.scripts.train",
        f"--sft-toml={_TOML}",
        "--",
        *_OVERRIDES,
    ]
    runpy.run_module("cosmos_framework.scripts.train", run_name="__main__")


if __name__ == "__main__":
    main()
