"""Four-rank probe for cold/warm reproducibility of the production UMI Edge VAE."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


def _tensor_summary(tensor: Any) -> dict[str, Any]:
    import torch

    value = tensor.detach().contiguous()
    raw = value.reshape(-1).view(torch.uint8).cpu().numpy().tobytes()
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "stride": list(tensor.stride()),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def _module_summary(module: Any) -> dict[str, Any]:
    return {
        "parameters": {name: _tensor_summary(value) for name, value in module.named_parameters()},
        "buffers": {name: _tensor_summary(value) for name, value in module.named_buffers()},
    }


def _normalized_fixed_video(zarr_path: Path, device: Any) -> Any:
    import torch
    from cosmos_framework.data.generator.action.datasets.umift_zarr_dataset import UMIFTZarrIterableDataset
    from cosmos_framework.model.generator.vision_encoder import normalize_uint8_item

    dataset = UMIFTZarrIterableDataset(
        str(zarr_path), split="train", stage="smoke", seed=42,
        resolution="256", fps=15.0,
    )
    dataset.shard_world_size = 1
    dataset.shard_rank = 0
    sample = next(iter(dataset))
    video = sample["video"]
    if isinstance(video, (list, tuple)):
        assert len(video) == 1
        video = video[0]
    assert tuple(video.shape[-3:]) == (17, 256, 256)
    if video.ndim == 4:
        video = video.unsqueeze(0)
    assert video.ndim == 5 and video.shape[:3] == (1, 3, 17)
    if video.dtype == torch.uint8:
        video = normalize_uint8_item(video, {"device": device, "dtype": torch.float32})
    else:
        video = video.to(device=device, dtype=torch.float32)
        assert bool(torch.all((video >= -1.0001) & (video <= 1.0001)).item())
    return video.contiguous()


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    import torch.distributed as dist
    from cosmos_framework.configs.toml_config.sft_config import load_experiment_from_toml
    from cosmos_framework.utils.lazy_config import instantiate

    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "0,1,2,3"
    assert int(os.environ.get("WORLD_SIZE", "0")) == 4
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    rank_dir = args.output / f"rank{rank}"

    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    reusable = [not args.output.exists() or not any(args.output.iterdir())] if rank == 0 else [None]
    dist.broadcast_object_list(reusable, src=0)
    if not reusable[0]:
        raise FileExistsError(f"refusing to overwrite non-empty output: {args.output}")
    if rank == 0:
        args.output.mkdir(parents=True, exist_ok=True)
    dist.barrier()
    rank_dir.mkdir()

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["WAN_VAE_PATH"] = str(args.vae.resolve())
    os.environ["EDGE_HF_SNAPSHOT_PATH"] = str(args.hf_snapshot.resolve())
    os.environ.setdefault("BASE_CHECKPOINT_PATH", str(args.output / "unused-dcp"))
    os.environ["DATASET_PATH"] = str(args.zarr.resolve())

    torch.backends.cudnn.benchmark = args.benchmark
    config = load_experiment_from_toml(args.toml)
    tokenizer = instantiate(config.model.config.tokenizer)
    tokenizer.reset_dtype() if hasattr(tokenizer, "reset_dtype") else None
    tokenizer = tokenizer.to(torch.device("cuda", local_rank)) if hasattr(tokenizer, "to") else tokenizer
    dist.barrier()

    module = tokenizer.model.model  # Wan2pt2VAEInterface -> WanVAE -> nn.Module
    weights = _module_summary(module)
    weights["latent_scale"] = [_tensor_summary(value) for value in tokenizer.model.scale]
    weight_digest = hashlib.sha256(json.dumps(weights, sort_keys=True).encode()).hexdigest()
    gathered: list[str | None] = [None] * 4
    dist.all_gather_object(gathered, weight_digest)
    assert len(set(gathered)) == 1, f"VAE state differs across ranks: {gathered}"

    video = _normalized_fixed_video(args.zarr, torch.device("cuda", local_rank))
    latents = []
    for repeat in range(3):
        with torch.no_grad():
            latent = tokenizer.encode(video)
        record = _tensor_summary(latent)
        torch.save(latent.detach().cpu(), rank_dir / f"latent_repeat{repeat}.pt")
        latents.append(record)
    summary = {
        "rank": rank,
        "local_rank": local_rank,
        "benchmark_requested": args.benchmark,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cuda_device": torch.cuda.current_device(),
        "input": _tensor_summary(video),
        "vae_state_sha256": weight_digest,
        "vae_state": weights,
        "latents": latents,
    }
    (rank_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    dist.barrier()
    dist.destroy_process_group()
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--toml", type=Path, required=True)
    parser.add_argument("--zarr", type=Path, required=True)
    parser.add_argument("--vae", type=Path, required=True)
    parser.add_argument("--hf-snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--benchmark", choices=("true", "false"), required=True)
    args = parser.parse_args()
    args.benchmark = args.benchmark == "true"
    run(args)


if __name__ == "__main__":
    main()
