"""Single-GPU, explicit-window UMI inference from a local DCP backup.

The frozen evaluation entry points retain their original launch and sampling
contracts. This separate entry point reuses their batch and decode helpers.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np

from examples.umift.infer import _numpy, strict_dcp_key_evidence, validate_checkpoint_path

HISTORY_FRAMES = 5
DEFAULT_NOISE_SEED = 495100992
DEFAULT_NUM_STEPS = 30
_LOCAL_ASSETS = ("WAN_VAE_PATH", "EDGE_HF_SNAPSHOT_PATH", "BASE_CHECKPOINT_PATH")


def local_asset_paths(environment: dict[str, str]) -> dict[str, str]:
    """Require explicit local assets instead of inheriting remote machine paths."""
    result = {}
    for name in _LOCAL_ASSETS:
        raw = environment.get(name)
        if not raw or not Path(raw).is_absolute():
            raise ValueError(f"{name} must name an existing absolute local path")
        path = Path(raw).resolve()
        if not path.exists():
            raise FileNotFoundError(f"{name}: {path}")
        result[name] = str(path)
    return result


def validate_model_contract(model_dict: dict[str, Any], dataset_target: Any, modality: str) -> None:
    cfg = model_dict["config"]
    if not cfg.get("action_gen") or not cfg.get("vision_gen") or cfg.get("sound_gen"):
        raise ValueError("resolved model must be the Edge vision+action FD graph")
    if str(cfg.get("resolution")) != "256" or cfg["tokenizer"]["encode_exact_durations"] != [21]:
        raise ValueError("local backup inference requires the frozen H5/21-frame/256 model")
    target_name = getattr(dataset_target, "__name__", str(dataset_target).rsplit(".", 1)[-1])
    expected = {
        "rgb": "get_umift_history_sft_dataset",
        "rgbd": "get_umift_rgbd_sft_dataset",
    }[modality]
    if target_name != expected:
        raise ValueError(f"{modality} requires {expected}, got {target_name}")


def single_gpu_runtime_options(cfg: dict[str, Any]) -> tuple[dict, dict, dict]:
    parallelism = dict(cfg["parallelism"])
    parallelism.update(
        data_parallel_shard_degree=1,
        data_parallel_replicate_degree=1,
        context_parallel_shard_degree=1,
        cfg_parallel_shard_degree=1,
        vae_load_balance_group_size=1,
        enable_inference_mode=True,
    )
    compile_options = dict(cfg["compile"])
    compile_options["enabled"] = False
    return parallelism, compile_options, dict(cfg.get("quantization", {}))


def load_local_model(sft_toml: Path, checkpoint: Path, modality: str) -> tuple[Any, Any, dict]:
    from cosmos_framework.configs.base.defaults.compile import CompileConfig
    from cosmos_framework.configs.base.defaults.parallelism import ParallelismConfig
    from cosmos_framework.configs.base.defaults.quantization import QuantizationConfig
    from cosmos_framework.configs.toml_config.sft_config import load_experiment_from_toml
    from cosmos_framework.inference.common.config import structure_config, unstructure_config
    from cosmos_framework.inference.model import Cosmos3OmniConfig, Cosmos3OmniModel

    validate_checkpoint_path(checkpoint)
    paths = local_asset_paths(dict(os.environ))
    resolved = load_experiment_from_toml(sft_toml)
    model_dict = unstructure_config(resolved.model, invalid="ignore")
    target = resolved.dataloader_train.dataloader.datasets.umift.dataset._target_
    validate_model_contract(model_dict, target, modality)
    if resolved.trainer.callbacks.compile_tokenizer.enabled:
        raise ValueError("local backup inference requires compile_tokenizer.enabled=false")
    parallelism, compile_options, quantization = single_gpu_runtime_options(model_dict["config"])
    wrapper = Cosmos3OmniModel.from_pretrained_dcp(
        checkpoint,
        config=Cosmos3OmniConfig(model=model_dict),
        parallelism_config=structure_config(parallelism, ParallelismConfig),
        compile_config=structure_config(compile_options, CompileConfig),
        quantization_config=structure_config(quantization, QuantizationConfig),
    )
    wrapper.eval()
    evidence = strict_dcp_key_evidence(wrapper.model, checkpoint)
    if evidence["model_key_count"] != 549 or evidence["checkpoint_key_count"] != 549:
        raise ValueError("local backup must preserve exactly the 549 Edge DCP/model keys")
    evidence.update(
        loader="Cosmos3OmniModel.from_pretrained_dcp",
        checkpoint=str(checkpoint.resolve()),
        sft_toml=str(sft_toml.resolve()),
        local_asset_paths=paths,
        model_config=model_dict,
        runtime_parallelism=parallelism,
        runtime_compile=compile_options,
        runtime_quantization=quantization,
        modality=modality,
        canvas_shape=[3, 21, 256, 512 if modality == "rgbd" else 256],
    )
    return wrapper.model, resolved, evidence


def check_sampling_reference(reference: dict[str, Any], *, noise_seed: int, num_steps: int) -> None:
    if not 0 <= noise_seed <= 0x7FFF_FFFF or num_steps < 1:
        raise ValueError("noise_seed must be a non-negative 31-bit integer and num_steps must be positive")
    for key, actual in (("noise_seed", noise_seed), ("num_steps", num_steps)):
        if key in reference and int(np.asarray(reference[key]).item()) != actual:
            raise ValueError(f"reference {key} differs from the requested sampling protocol")


def validate_causal_sample(sample: dict[str, Any], modality: str) -> None:
    width = 512 if modality == "rgbd" else 256
    if tuple(sample["video"].shape) != (3, 21, 256, width):
        raise ValueError(f"{modality} input must be [3,21,256,{width}]")
    history = _numpy(sample["history_source_indices"])
    source = _numpy(sample["source_indices"])
    if history.shape != (5,) or source.shape != (17,):
        raise ValueError("expected five history and 17 anchor/future source indices")
    if np.any(history > source[0]) or history[-1] != source[0] or np.any(np.diff(history) < 0):
        raise ValueError("history contains future leakage or does not end at the anchor")
    if not np.array_equal(source, source[0] + 2 * np.arange(17)):
        raise ValueError("anchor/future source indices must retain the original stride two")
    plan = sample["sequence_plan"]
    if list(plan.condition_frame_indexes_vision) != [0, 1] or plan.action_start_frame_offset != 5:
        raise ValueError("only the two causal H5 vision latents may be conditioned")


def anchor_prediction(truth: np.ndarray, decoded: np.ndarray) -> np.ndarray:
    if truth.shape[0] != 17 or decoded.shape[0] != 21 or truth.shape[1:] != decoded.shape[1:]:
        raise ValueError("expected 17 truth frames and H5+16 decoded frames with matching spatial shape")
    return np.concatenate((truth[:1], decoded[5:]), axis=0).astype(np.float32)


def validate_conditioned_video(video: Any, modality: str) -> None:
    value = _numpy(video)
    if modality == "rgbd":
        value = value[0]
    if np.any(value[:, 5:]):
        raise ValueError("future RGB/depth truth was not cleared from the model batch")


def choose_sample(dataset: Any, args: argparse.Namespace) -> dict[str, Any]:
    if args.action_mode == "A":
        return dataset.get_window(args.episode, args.start)
    if args.action_mode == "Z":
        physical = np.tile(np.array([0, 0, 0, 1, 0, 0, 0, 1, 0, 0], np.float32), (16, 1))
    else:
        if args.replacement_episode is None or args.replacement_start is None:
            raise ValueError("S requires --replacement-episode and --replacement-start")
        if (args.replacement_episode, args.replacement_start) == (args.episode, args.start):
            raise ValueError("S requires a different source window")
        replacement = dataset.get_window(args.replacement_episode, args.replacement_start)
        physical = _numpy(replacement["physical_action"]).copy()
    # get_window normalizes physical actions with the original dataset normalizer.
    return dataset.get_window(args.episode, args.start, physical_action=physical)


def resident_tensor_bytes(model: Any) -> dict[str, int]:
    sizes: dict[str, int] = {}
    seen = set()
    for tensor in (*model.parameters(), *model.buffers()):
        storage = tensor.untyped_storage()
        key = (str(tensor.device), storage.data_ptr())
        if key not in seen:
            sizes[key[0]] = sizes.get(key[0], 0) + storage.nbytes()
            seen.add(key)
    return sizes


def find_ffmpeg() -> str:
    binary = shutil.which("ffmpeg")
    if binary:
        return binary
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except (ImportError, RuntimeError, OSError) as error:
        raise RuntimeError("ffmpeg is unavailable in PATH and the existing imageio-ffmpeg environment") from error


def render_clip(path: Path, truth: np.ndarray, prediction: np.ndarray, label: str, fps: float) -> None:
    from PIL import Image, ImageDraw

    process = subprocess.Popen(
        [find_ffmpeg(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", "768x288", "-r", str(fps), "-i", "pipe:0", "-an", "-c:v", "libx264",
         "-pix_fmt", "yuv420p", str(path)],
        stdin=subprocess.PIPE,
    )
    try:
        for index in range(17):
            frame = Image.new("RGB", (768, 288), (20, 20, 20))
            draw = ImageDraw.Draw(frame)
            for column, (title, values) in enumerate((
                ("GT", truth[index]), ("Persistence", truth[0]), (label, prediction[index]),
            )):
                draw.text((column * 256 + 6, 8), title, fill="white")
                frame.paste(Image.fromarray(np.rint(np.clip(values, 0, 1) * 255).astype(np.uint8)),
                            (column * 256, 32))
            process.stdin.write(np.asarray(frame).tobytes())
    finally:
        process.stdin.close()
    if process.wait() != 0:
        raise RuntimeError("ffmpeg failed to render the GT/Persistence/prediction clip")


def infer(args: argparse.Namespace) -> None:
    import torch

    from cosmos_framework.data.generator.action.datasets.umift_history_dataset import get_umift_history_sft_dataset
    from cosmos_framework.data.generator.action.datasets.umift_rgbd_dataset import get_umift_rgbd_sft_dataset
    from examples.umift.history_infer import build_history_batch, run_history_prediction
    from examples.umift.infer import _move_batch_to_cuda
    from examples.umift.rgbd_infer import build_rgbd_batch, run_rgbd_prediction, split_prediction

    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("local backup entry point requires exactly one process")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError("expose exactly one GPU through CUDA_VISIBLE_DEVICES")
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to replace existing predictions: {output}")
    reference = {}
    if args.reference_npz:
        with np.load(args.reference_npz, allow_pickle=False) as archive:
            reference = {key: archive[key] for key in archive.files}
    check_sampling_reference(reference, noise_seed=args.noise_seed, num_steps=args.num_steps)
    torch.cuda.set_device(0)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    model, resolved, evidence = load_local_model(args.sft_toml, args.checkpoint, args.modality)
    dataset_factory = get_umift_rgbd_sft_dataset if args.modality == "rgbd" else get_umift_history_sft_dataset
    dataset = dataset_factory(
        str(args.zarr_path.resolve()), split="history", stage="e1", history_frames=5,
        tokenizer_config=resolved.model.config.vlm_config.tokenizer,
        max_action_dim=int(resolved.model.config.max_action_dim),
    )
    sample = choose_sample(dataset, args)
    validate_causal_sample(sample, args.modality)
    build_batch = build_rgbd_batch if args.modality == "rgbd" else build_history_batch
    batch = build_batch(sample)
    validate_conditioned_video(batch["video"][0], args.modality)
    truth_rgb = np.moveaxis(_numpy(sample["video"]), 0, -1)[4:]
    if args.modality == "rgbd":
        truth = np.clip((truth_rgb[:, :, :256] + 1) / 2, 0, 1).astype(np.float32)
    else:
        truth = truth_rgb.astype(np.float32) / 255
    inference_started = time.perf_counter()
    with torch.inference_mode():
        batch = _move_batch_to_cuda(batch)
        if args.modality == "rgbd":
            decoded, depth_decoded = split_prediction(run_rgbd_prediction(
                model, batch, noise_seed=args.noise_seed, num_steps=args.num_steps,
            ))
        else:
            decoded = run_history_prediction(
                model, batch, history_frames=5, noise_seed=args.noise_seed, num_steps=args.num_steps,
            )
    torch.cuda.synchronize()
    arrays = {
        "truth": truth, "prediction": anchor_prediction(truth, decoded),
        "physical_action": _numpy(sample["physical_action"]),
        "model_action": _numpy(sample["model_action"]), "action": _numpy(sample["action"]),
        "source_indices": _numpy(sample["source_indices"]),
        "history_source_indices": _numpy(sample["history_source_indices"]),
        "history_real_mask": _numpy(sample["history_real_mask"]),
        "timestamps": _numpy(sample["timestamps"]),
        "noise_seed": np.array(args.noise_seed), "num_steps": np.array(args.num_steps),
    }
    if args.modality == "rgbd":
        depth_truth = _numpy(sample["depth_m"])[4:]
        arrays.update(depth_truth=depth_truth, depth_prediction=anchor_prediction(depth_truth, depth_decoded))
    timestamps = arrays["timestamps"]
    if timestamps.shape != (17,) or not np.isfinite(timestamps).all() or np.any(np.diff(timestamps) <= 0):
        raise ValueError("the 17 observed Zarr timestamps must be finite and strictly increasing")
    duration = float(timestamps[-1] - timestamps[0])
    for key in ("truth", "depth_truth", "source_indices", "history_source_indices"):
        if key in reference and (key not in arrays or not np.array_equal(arrays[key], reference[key])):
            raise ValueError(f"local {key} differs from the supplied frozen reference")
    evidence.update(
        episode=args.episode, start=args.start, action_mode=args.action_mode,
        replacement_episode=args.replacement_episode, replacement_start=args.replacement_start,
        zarr_path=str(args.zarr_path.resolve()), noise_seed=args.noise_seed, num_steps=args.num_steps,
        noise_seed_source=args.noise_seed_source,
        reference_npz=str(args.reference_npz.resolve()) if args.reference_npz else None,
        guidance=1.0, has_negative_prompt=False, dataset_seed=dataset.seed,
        actual_duration_seconds=duration, output_fps=16 / duration, conditioning_fps=15.0,
        history_source_indices=arrays["history_source_indices"].tolist(),
        source_indices=arrays["source_indices"].tolist(), future_truth_cleared=True,
        anchor_frame_source="observed_ground_truth", resident_tensor_bytes=resident_tensor_bytes(model),
        gpu_name=torch.cuda.get_device_name(0),
        gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        load_seconds=inference_started - started,
        inference_seconds=time.perf_counter() - inference_started,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **arrays)
    output.with_suffix(".json").write_text(json.dumps(evidence, indent=2, default=str) + "\n")
    if not args.skip_video:
        render_clip(output.with_suffix(".mp4"), truth, arrays["prediction"], f"Pred {args.action_mode}", 16 / duration)
    print(json.dumps({"output": str(output), "inference_seconds": evidence["inference_seconds"],
                      "gpu_peak_allocated_bytes": evidence["gpu_peak_allocated_bytes"]}))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("sft-toml", "checkpoint", "zarr-path", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--episode", type=int, default=13)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--noise-seed", type=int, help=f"Default: frozen backup seed {DEFAULT_NOISE_SEED}.")
    parser.add_argument("--num-steps", type=int, default=DEFAULT_NUM_STEPS)
    parser.add_argument("--modality", choices=("rgb", "rgbd"), required=True)
    parser.add_argument("--reference-npz", type=Path)
    parser.add_argument("--action-mode", choices=("A", "Z", "S"), default="A")
    parser.add_argument("--mismatch-episode", "--replacement-episode", dest="replacement_episode", type=int, default=43)
    parser.add_argument("--mismatch-start", "--replacement-start", dest="replacement_start", type=int, default=0)
    parser.add_argument("--skip-video", action="store_true")
    args = parser.parse_args(argv)
    args.noise_seed_source = "CLI" if args.noise_seed is not None else "frozen_local_backup_default"
    if args.noise_seed is None:
        args.noise_seed = DEFAULT_NOISE_SEED
    if args.output.suffix != ".npz":
        parser.error("--output must end in .npz")
    infer(args)


if __name__ == "__main__":
    main()
