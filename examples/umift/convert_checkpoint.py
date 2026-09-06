"""Convert the fixed local Cosmos3-Edge HF snapshot to the E1 model-only DCP.

This experiment-local converter deliberately bypasses the public-config alias
round trip: the resolved E1 TOML contains the local ``build_processor`` target,
which has no public alias.  Model construction and conversion run on CPU and
all source dependencies must be absolute local paths.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

EDGE_REVISION = "a9d944e2c6a1bf9f48b92ad16348e70c5f1836ba"

# These must precede every framework/torch import.
os.environ["COSMOS_DEVICE"] = "cpu"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_manifest(snapshot: Path) -> dict[str, str]:
    """Hash every file that can define weights, architecture, or processing."""
    selected = []
    for path in snapshot.rglob("*"):
        if path.is_file() and (path.suffix in {".json", ".safetensors"} or path.name.startswith("tokenizer")):
            selected.append(path)
    if not any(path.suffix == ".safetensors" for path in selected):
        raise FileNotFoundError(f"no local safetensors weights found under {snapshot}")
    return {str(path.relative_to(snapshot)): sha256_file(path) for path in sorted(selected)}


def validate_inputs(snapshot: Path, vae: Path, output: Path) -> None:
    for label, path, is_dir in (("snapshot", snapshot, True), ("vae", vae, False)):
        if not path.is_absolute() or not (path.is_dir() if is_dir else path.is_file()):
            raise ValueError(f"{label} must be an existing absolute local {'directory' if is_dir else 'file'}: {path}")
    if snapshot.name != EDGE_REVISION:
        raise ValueError(f"snapshot must be fixed Edge revision {EDGE_REVISION}, got {snapshot.name}")
    for required in ("config.json", "tokenizer.json"):
        if not (snapshot / required).is_file():
            raise FileNotFoundError(snapshot / required)
    if not output.is_absolute():
        raise ValueError(f"output must be absolute: {output}")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite conversion output: {output}")


def state_evidence(state: dict[str, Any]) -> dict[str, Any]:
    import torch

    tensors = {key: value for key, value in state.items() if isinstance(value, torch.Tensor)}
    nonfinite = [key for key, value in tensors.items()
                 if (value.is_floating_point() or value.is_complex()) and not torch.isfinite(value).all().item()]
    if nonfinite:
        raise ValueError(f"loaded source model contains non-finite tensors: {nonfinite[:20]}")
    return {
        "key_count": len(state),
        "tensor_count": len(tensors),
        "parameter_count": sum(int(value.numel()) for value in tensors.values()),
        "all_tensors_finite": True,
        "shapes": {key: list(value.shape) for key, value in sorted(tensors.items())},
    }


def verify_dcp_metadata(state: dict[str, Any], model_dir: Path) -> dict[str, Any]:
    from torch.distributed.checkpoint.filesystem import FileSystemReader

    metadata = FileSystemReader(str(model_dir)).read_metadata().state_dict_metadata
    expected_keys = set(state)
    actual_keys = set(metadata)
    if expected_keys != actual_keys:
        raise ValueError(
            f"DCP key mismatch: missing={sorted(expected_keys - actual_keys)[:20]}, "
            f"unexpected={sorted(actual_keys - expected_keys)[:20]}"
        )
    for key, value in state.items():
        stored_size = getattr(metadata[key], "size", None)
        if stored_size is None or tuple(stored_size) != tuple(value.shape):
            raise ValueError(f"DCP shape mismatch for {key}: source={tuple(value.shape)}, stored={stored_size}")
    return {"strict_keys": True, "strict_shapes": True, "metadata_key_count": len(actual_keys)}


def convert(args: argparse.Namespace) -> dict[str, Any]:
    validate_inputs(args.snapshot, args.vae, args.output)
    manifest = source_manifest(args.snapshot)
    os.environ["BASE_CHECKPOINT_PATH"] = str(args.snapshot)
    os.environ["EDGE_HF_SNAPSHOT_PATH"] = str(args.snapshot)
    os.environ["WAN_VAE_PATH"] = str(args.vae)

    import torch
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.filesystem import FileSystemReader, FileSystemWriter
    from torch.distributed.checkpoint.state_dict import get_model_state_dict

    from cosmos_framework.checkpoint.dcp import CustomSavePlanner
    from cosmos_framework.configs.base.defaults.compile import CompileConfig
    from cosmos_framework.configs.base.defaults.parallelism import ParallelismConfig
    from cosmos_framework.configs.base.defaults.quantization import QuantizationConfig
    from cosmos_framework.configs.toml_config.sft_config import load_experiment_from_toml
    from cosmos_framework.inference.common.config import unstructure_config
    from cosmos_framework.inference.model import Cosmos3OmniConfig, Cosmos3OmniModel

    torch.set_grad_enabled(False)
    resolved = load_experiment_from_toml(args.toml)
    if resolved.model.config.ema.enabled:
        raise ValueError("E1 conversion requires EMA disabled")
    model_dict = unstructure_config(resolved.model, invalid="ignore")
    config = Cosmos3OmniConfig(model=model_dict)
    wrapper = Cosmos3OmniModel.from_pretrained_dcp(
        args.snapshot,
        config=config,
        parallelism_config=ParallelismConfig(
            data_parallel_shard_degree=1,
            data_parallel_replicate_degree=1,
            enable_inference_mode=True,
        ),
        compile_config=CompileConfig(enabled=False),
        quantization_config=QuantizationConfig(),
    )
    state = get_model_state_dict(wrapper.model)
    before = state_evidence(state)
    size_bytes = sum(value.numel() * value.element_size() for value in state.values() if isinstance(value, torch.Tensor))
    args.output.mkdir(parents=True)
    dcp.save(
        state_dict=state,
        storage_writer=FileSystemWriter(args.output / "model", thread_count=max(1, math.ceil(size_bytes / (5 * 1024**3)))),
        planner=CustomSavePlanner(),
    )
    verification = verify_dcp_metadata(state, args.output / "model")
    # Exercise a strict read of every key after the metadata comparison.
    dcp.load(state_dict=state, storage_reader=FileSystemReader(str(args.output / "model")))
    evidence = {
        "source": str(args.snapshot.resolve()),
        "source_files_sha256": manifest,
        "vae": str(args.vae.resolve()),
        "vae_sha256": sha256_file(args.vae),
        "output_model": str((args.output / "model").resolve()),
        "state": before,
        "verification": verification,
        "ema_enabled": False,
        "device": "cpu",
    }
    return evidence


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--toml", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--vae", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evidence", type=Path)
    args = parser.parse_args(argv)
    report = convert(args)
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    print(payload, end="")
    if args.evidence:
        args.evidence.parent.mkdir(parents=True, exist_ok=True)
        args.evidence.write_text(payload, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
