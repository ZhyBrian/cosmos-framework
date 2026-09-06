"""Generate Cosmos3 Edge UMI-FT FD predictions for offline evaluation.

Run this script inside the prepared Cosmos GPU environment.  Heavy framework
imports are deliberately lazy so its contracts can be tested on a CPU-only
machine.  One invocation handles one checkpoint/method/split and writes a
JSONL manifest accepted by :mod:`examples.umift.evaluate`.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
from pathlib import Path
from typing import Any, Callable, Iterable, Literal

import numpy as np

from examples.umift.protocol import derive_noise_seed, persistence_prediction

Variant = Literal["A", "Z", "S"]
_WINDOW_ID = re.compile(r"^episode_(?P<episode>\d+):s=(?P<start>\d+)$")
E0_OVERFIT_EPISODE = 0
E0_OVERFIT_STARTS = (0, 64, 128, 192)
_E0_OVERFIT_METHODS = {"B-VAE", "B0", "E1-A"}


def _numpy(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        return value.numpy()
    return np.asarray(value)


def _copy_value(value: Any) -> Any:
    return value.clone() if hasattr(value, "clone") else np.array(value, copy=True)


def validate_fd_sample(sample: dict[str, Any]) -> None:
    if sample.get("mode") != "forward_dynamics":
        raise ValueError(f"mode must be forward_dynamics, got {sample.get('mode')!r}")
    if sample.get("ai_caption") != "":
        raise ValueError("E1 requires the frozen empty prompt")
    video = _numpy(sample["video"])
    if video.shape != (3, 17, 256, 256):
        raise ValueError(f"video must be [3,17,256,256], got {video.shape}")
    action = _numpy(sample["action"])
    if action.shape[0] != 16:
        raise ValueError(f"action must contain 16 steps, got {action.shape}")
    fps = float(_numpy(sample["conditioning_fps"]).reshape(-1)[0])
    if fps != 15.0:
        raise ValueError(f"conditioning_fps must be 15.0, got {fps}")
    plan = sample["sequence_plan"]
    if list(plan.condition_frame_indexes_vision) != [0]:
        raise ValueError(
            f"future vision leakage: condition indexes are {list(plan.condition_frame_indexes_vision)}"
        )
    if list(plan.condition_frame_indexes_action) != list(range(16)):
        raise ValueError("all and only 16 action steps must be clean conditioning")
    if not (plan.has_text and plan.has_vision and plan.has_action):
        raise ValueError("FD sequence plan must include text, vision, and action streams")


def condition_only_video(video: Any) -> Any:
    """Keep observed I0 and blank future RGB before model inference."""
    result = _copy_value(video)
    result[:, 1:] = 0
    return result


def make_action_variant(
    sample: dict[str, Any],
    variant: Variant,
    *,
    normalizer: Callable[[Any], Any] | None = None,
    replacement: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Pure helper for A/Z/S; prefer dataset.get_window override in the CLI."""
    result = copy.copy(sample)
    if variant == "A":
        return result
    if variant == "Z":
        if normalizer is None:
            raise ValueError("Z requires the dataset's physical-space normalizer")
        physical = np.tile(np.array([0, 0, 0, 1, 0, 0, 0, 1, 0, 0], np.float32), (16, 1))
        model_action = _numpy(normalizer(physical)).astype(np.float32, copy=False)
        result["physical_action"] = physical
    elif variant == "S":
        if replacement is None:
            raise ValueError("S requires a replacement window")
        model_action = _numpy(replacement["model_action"]).astype(np.float32, copy=True)
        result["physical_action"] = _copy_value(replacement["physical_action"])
    else:
        raise ValueError(f"unknown action variant {variant!r}")
    padded = np.zeros_like(_numpy(sample["action"]), dtype=np.float32)
    padded[:, : model_action.shape[1]] = model_action
    result["model_action"] = model_action
    result["action"] = padded
    return result


def decoded_video_to_thwc01(decoded: Any) -> np.ndarray:
    value = _numpy(decoded).astype(np.float32, copy=False)
    if not np.isfinite(value).all():
        raise ValueError("decoded video contains non-finite values")
    if value.ndim == 5:
        if value.shape[0] != 1:
            raise ValueError(f"expected one decoded sample, got {value.shape}")
        value = value[0]
    if value.shape[0] != 3 or value.shape[1] != 17:
        raise ValueError(f"decoded video must be [1,3,17,H,W] or [3,17,H,W], got {value.shape}")
    value = np.moveaxis(value, 0, -1)
    return ((np.clip(value, -1.0, 1.0) + 1.0) / 2.0).astype(np.float32)


def run_forward_dynamics(model: Any, batch: dict[str, Any], *, noise_seed: int, num_steps: int) -> np.ndarray:
    outputs = model.generate_samples_from_batch(
        batch,
        seed=[int(noise_seed)],
        num_steps=int(num_steps),
        guidance=1.0,
        has_negative_prompt=False,
        upsample_task=None,
    )
    vision = outputs.get("vision")
    if not isinstance(vision, list) or len(vision) != 1:
        raise ValueError("model must return exactly one vision latent")
    return decoded_video_to_thwc01(model.decode(vision[0]))


def build_singleton_batch(sample: dict[str, Any]) -> dict[str, Any]:
    """Match the IterativeJointDataLoader's one-sample inference contract."""
    import torch

    validate_fd_sample(sample)
    video = condition_only_video(sample["video"])
    image_size = sample["image_size"]
    if image_size.ndim == 1:
        image_size = image_size.unsqueeze(0)
    batch: dict[str, Any] = {
        "video": [video],
        "action": [sample["action"]],
        "raw_action_dim": [sample["raw_action_dim"]],
        "mode": ["forward_dynamics"],
        "ai_caption": [""],
        "image_size": [image_size],
        "fps": torch.tensor([15.0], device=video.device),
        "conditioning_fps": torch.tensor([15.0], device=video.device),
        "num_frames": torch.tensor([17], device=video.device),
        "domain_id": [sample["domain_id"]],
        "sequence_plan": [sample["sequence_plan"]],
    }
    if "text_token_ids" in sample:
        batch["text_token_ids"] = [sample["text_token_ids"]]
    return batch


def load_edge_fd_model(sft_toml: Path, checkpoint: Path) -> tuple[Any, Any, dict[str, Any]]:
    """Build the inference wrapper from the resolved training model config."""
    from cosmos_framework.configs.base.defaults.compile import CompileConfig
    from cosmos_framework.configs.base.defaults.parallelism import ParallelismConfig
    from cosmos_framework.configs.base.defaults.quantization import QuantizationConfig
    from cosmos_framework.configs.toml_config.sft_config import load_experiment_from_toml
    from cosmos_framework.inference.common.config import unstructure_config
    from cosmos_framework.inference.model import Cosmos3OmniConfig, Cosmos3OmniModel

    validate_checkpoint_path(checkpoint)
    resolved = load_experiment_from_toml(sft_toml)
    model_dict = unstructure_config(resolved.model, invalid="ignore")
    cfg = model_dict["config"]
    if not cfg.get("action_gen") or not cfg.get("vision_gen") or cfg.get("sound_gen"):
        raise ValueError("resolved model is not the Edge vision+action FD graph")
    if str(cfg.get("resolution")) != "256" or cfg.get("tokenizer", {}).get("encode_exact_durations") != [17]:
        raise ValueError("resolved model does not satisfy the E1 256/17-frame contract")
    omni_config = Cosmos3OmniConfig(model=model_dict)
    parallelism = dict(cfg["parallelism"])
    parallelism["enable_inference_mode"] = True
    compile_options = dict(cfg["compile"])
    compile_options["enabled"] = False
    wrapper = Cosmos3OmniModel.from_pretrained_dcp(
        checkpoint,
        config=omni_config,
        parallelism_config=ParallelismConfig(**parallelism),
        compile_config=CompileConfig(**compile_options),
        quantization_config=QuantizationConfig(**cfg.get("quantization", {})),
    )
    wrapper.eval()
    evidence = strict_dcp_key_evidence(wrapper.model, checkpoint)
    evidence.update({"checkpoint": str(checkpoint.resolve()), "loader": "Cosmos3OmniModel.from_pretrained_dcp"})
    return wrapper.model, resolved, evidence


def validate_checkpoint_path(checkpoint: Path) -> None:
    if not checkpoint.is_dir():
        raise ValueError(f"checkpoint directory does not exist: {checkpoint}")
    if (checkpoint / "model").is_dir():
        raise ValueError(
            f"{checkpoint} is an iteration root; pass its model component directory: {checkpoint / 'model'}"
        )


def compare_checkpoint_keys(model_keys: set[str], checkpoint_keys: set[str]) -> dict[str, int]:
    missing = sorted(model_keys - checkpoint_keys)
    unexpected = sorted(checkpoint_keys - model_keys)
    if missing or unexpected:
        raise ValueError(
            "DCP/model state keys differ: "
            f"missing={missing[:20]} (total {len(missing)}), "
            f"unexpected={unexpected[:20]} (total {len(unexpected)})"
        )
    return {"model_key_count": len(model_keys), "checkpoint_key_count": len(checkpoint_keys)}


def strict_dcp_key_evidence(model: Any, checkpoint: Path) -> dict[str, int]:
    from torch.distributed.checkpoint.filesystem import FileSystemReader
    from torch.distributed.checkpoint.state_dict import get_model_state_dict

    model_keys = set(get_model_state_dict(model))
    checkpoint_keys = set(FileSystemReader(str(checkpoint)).read_metadata().state_dict_metadata)
    return compare_checkpoint_keys(model_keys, checkpoint_keys)


def _window_id(sample: dict[str, Any]) -> str:
    return f"episode_{int(sample['episode_id'])}:s={int(sample['window_start'])}"


def _parse_window_id(window_id: str) -> tuple[int, int]:
    match = _WINDOW_ID.fullmatch(window_id)
    if match is None:
        raise ValueError(f"invalid window id {window_id!r}")
    return int(match["episode"]), int(match["start"])


def _load_pairs(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in value.items()):
        raise ValueError("pair file must be a source-window to replacement-window JSON object")
    return value


def _save_prediction(output_dir: Path, method: str, window_id: str, seed: int, video: np.ndarray) -> Path:
    safe_id = window_id.replace(":", "_").replace("=", "-")
    path = output_dir / method / f"{safe_id}_seed{seed}.npy"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, video, allow_pickle=False)
    return path


def _iter_n(dataset: Iterable[dict[str, Any]]) -> Iterable[dict[str, Any]]:
    iterator = iter(dataset)
    for _ in range(len(dataset)):  # type: ignore[arg-type]
        yield next(iterator)


def resolve_dataset_protocol(split: str, stage: str) -> tuple[str, str]:
    """Map the closed E0 evaluation split onto the adapter's training subset."""
    if split == "overfit":
        return "train", "overfit"
    return split, stage


def iter_evaluation_windows(dataset: Any, split: str) -> Iterable[dict[str, Any]]:
    """Enumerate an evaluation split without exposing arbitrary training windows."""
    if split != "overfit":
        yield from _iter_n(dataset)
        return
    for start in E0_OVERFIT_STARTS:
        sample = dataset.get_window(E0_OVERFIT_EPISODE, start)
        actual = (int(sample["episode_id"]), int(sample["window_start"]))
        expected = (E0_OVERFIT_EPISODE, start)
        if actual != expected:
            raise ValueError(f"overfit dataset returned unexpected window {actual}; expected {expected}")
        yield sample


def validate_sampling_protocol(split: str, method: str, sampling_seeds: list[int]) -> None:
    if len(sampling_seeds) != len(set(sampling_seeds)):
        raise ValueError("sampling seeds must be unique")
    stochastic = method not in {"B-Persistence", "B-VAE"}
    if split == "overfit":
        if method not in _E0_OVERFIT_METHODS:
            raise ValueError("overfit split only supports B-VAE, B0, and E1-A")
        if sampling_seeds != [0]:
            raise ValueError("overfit E0 evaluation requires sampling seed [0]")
        return
    if split == "history" and stochastic and sampling_seeds != [0, 1, 2]:
        raise ValueError("history stochastic evaluation requires sampling seeds [0, 1, 2]")
    if split == "dev" and stochastic and sampling_seeds != [0]:
        raise ValueError("dev checkpoint selection requires sampling seed [0]")


def validate_launch_environment(environment: dict[str, str]) -> None:
    visible = environment.get("CUDA_VISIBLE_DEVICES")
    if visible != "0,1,2,3":
        raise ValueError(
            "model inference requires CUDA_VISIBLE_DEVICES=0,1,2,3 so GPUs 4-7 are never visible"
        )
    if environment.get("WORLD_SIZE") != "4":
        raise ValueError("model inference requires torchrun WORLD_SIZE=4")


def run_cli(args: argparse.Namespace) -> None:
    import torch

    from cosmos_framework.data.generator.action.datasets.umift_zarr_dataset import (
        get_umift_zarr_sft_dataset,
    )
    from cosmos_framework.inference.common.init import get_rank

    method = args.method
    validate_sampling_protocol(args.split, method, args.sampling_seeds)
    model = resolved = None
    load_evidence: dict[str, Any] | None = None
    if method != "B-Persistence":
        model, resolved, load_evidence = load_edge_fd_model(args.sft_toml, args.checkpoint)
    tokenizer_config = None if resolved is None else resolved.model.config.vlm_config.tokenizer
    max_action_dim = 64 if resolved is None else int(resolved.model.config.max_action_dim)
    dataset_split, dataset_stage = resolve_dataset_protocol(args.split, args.stage)
    dataset = get_umift_zarr_sft_dataset(
        str(args.zarr),
        split=dataset_split,
        stage=dataset_stage,
        seed=42,
        resolution="256",
        fps=15.0,
        mode="forward_dynamics",
        tokenizer_config=tokenizer_config,
        max_action_dim=max_action_dim,
    )
    pairs = _load_pairs(args.pairs)
    manifest_rows: list[dict[str, Any]] = []
    for original in iter_evaluation_windows(dataset, args.split):
        window_id = _window_id(original)
        sample = original
        if method == "E1-Z":
            physical_zero = np.tile(np.array([0, 0, 0, 1, 0, 0, 0, 1, 0, 0], np.float32), (16, 1))
            sample = dataset.get_window(original["episode_id"], original["window_start"], physical_action=physical_zero)
        elif method == "E1-S":
            replacement_id = pairs.get(window_id)
            if replacement_id is None:
                raise ValueError(f"missing frozen S pair for {window_id}")
            replacement_episode, replacement_start = _parse_window_id(replacement_id)
            replacement = dataset.get_window(replacement_episode, replacement_start)
            sample = dataset.get_window(
                original["episode_id"], original["window_start"], physical_action=replacement["physical_action"]
            )
        validate_fd_sample(sample)
        truth = np.moveaxis(_numpy(original["video"]), 0, -1).astype(np.float32) / 255.0
        seeds = args.sampling_seeds if method not in {"B-Persistence", "B-VAE"} else [0]
        for sampling_seed in seeds:
            if method == "B-Persistence":
                prediction = persistence_prediction(truth)
            elif method == "B-VAE":
                state = sample["video"].unsqueeze(0).to(device="cuda", dtype=torch.float32) / 127.5 - 1.0
                prediction = decoded_video_to_thwc01(model.decode(model.encode(state)))
            else:
                batch = build_singleton_batch(sample)
                batch = _move_batch_to_cuda(batch)
                prediction = run_forward_dynamics(
                    model,
                    batch,
                    noise_seed=derive_noise_seed(window_id, sampling_seed),
                    num_steps=args.num_steps,
                )
            if get_rank() == 0:
                truth_path = _save_prediction(args.output_dir, "truth", window_id, 0, truth)
                pred_path = _save_prediction(args.output_dir, method, window_id, sampling_seed, prediction)
                manifest_rows.append(
                    {
                        "window_id": window_id,
                        "raw_session": original["session_id"],
                        "episode": original["source_id"],
                        "split": args.split,
                        "method": method,
                        "checkpoint_id": None if args.checkpoint is None else str(args.checkpoint),
                        "sampling_seed": int(sampling_seed),
                        "noise_seed": derive_noise_seed(window_id, sampling_seed),
                        "truth_path": os.path.relpath(truth_path, args.output_dir),
                        "prediction_path": os.path.relpath(pred_path, args.output_dir),
                    }
                )
    if get_rank() == 0:
        manifest = args.output_dir / f"{args.split}_{method}.jsonl"
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in manifest_rows), encoding="utf-8")
        if load_evidence is not None:
            evidence_path = args.output_dir / f"{args.split}_{method}_load_evidence.json"
            evidence_path.write_text(json.dumps(load_evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _move_batch_to_cuda(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _move_batch_to_cuda(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_move_batch_to_cuda(item) for item in value]
    if hasattr(value, "cuda"):
        return value.cuda()
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sft-toml", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--zarr", type=Path, required=True)
    parser.add_argument("--split", choices=["dev", "history", "overfit"], required=True)
    parser.add_argument("--stage", choices=["smoke", "overfit", "e1"], default="e1")
    parser.add_argument(
        "--method",
        choices=["B-Persistence", "B-VAE", "B0", "E1-A", "E1-Z", "E1-S"],
        required=True,
    )
    parser.add_argument("--pairs", type=Path)
    parser.add_argument("--sampling-seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--num-steps", type=int, default=30)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.method != "B-Persistence" and args.checkpoint is None:
        parser.error("--checkpoint is required except for persistence")
    if args.method != "B-Persistence" and args.sft_toml is None:
        parser.error("--sft-toml is required except for persistence")
    if args.method == "E1-S" and args.pairs is None:
        parser.error("--pairs is required for S")
    if args.method != "B-Persistence":
        try:
            validate_launch_environment(dict(os.environ))
        except ValueError as exc:
            parser.error(str(exc))
    if args.method != "B-Persistence":
        from cosmos_framework.inference.common.init import init_script

        init_script()
    run_cli(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
