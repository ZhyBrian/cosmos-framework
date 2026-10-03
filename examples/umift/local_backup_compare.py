"""Score and render one local UMI anchor window, with optional A40 reference."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse

import numpy as np

from examples.umift.local_backup_infer import find_ffmpeg
from examples.umift.rgbd_metrics import depth_video_metrics, joint_selection_score

METHODS = ("GT", "P", "B0", "A", "Z", "S")
_MATCH_KEYS = ("truth", "history_source_indices", "source_indices", "timestamps", "noise_seed", "num_steps")


def validate_runs(runs: dict[str, dict[str, np.ndarray]]) -> bool:
    correct = runs["A"]
    has_depth = "depth_truth" in correct
    for method, run in runs.items():
        for key in _MATCH_KEYS:
            if key not in run or not np.array_equal(correct[key], run[key]):
                raise ValueError(f"{method}: {key} differs from the correct-action window")
        for key in ("truth", "prediction"):
            video = np.asarray(run[key])
            if video.shape != (17, 256, 256, 3) or not np.isfinite(video).all():
                raise ValueError(f"{method}: {key} must be finite [17,256,256,3]")
            if np.any((video < 0) | (video > 1)):
                raise ValueError(f"{method}: {key} must already be bounded RGB [0,1]")
        if not np.array_equal(run["prediction"][0], run["truth"][0]):
            raise ValueError(f"{method}: prediction must preserve the observed anchor")
        if ("depth_truth" in run) != has_depth or ("depth_prediction" in run) != has_depth:
            raise ValueError(f"{method}: depth modality differs from the correct-action window")
        if has_depth:
            if not np.array_equal(correct["depth_truth"], run["depth_truth"]):
                raise ValueError(f"{method}: depth truth differs")
            for key in ("depth_truth", "depth_prediction"):
                value = run[key]
                if value.shape != (17, 256, 256) or not np.isfinite(value).all():
                    raise ValueError(f"{method}: {key} must be finite [17,256,256] metres")
            if not np.array_equal(run["depth_prediction"][0], run["depth_truth"][0]):
                raise ValueError(f"{method}: depth prediction must preserve the observed anchor")
    if not np.array_equal(correct["physical_action"], runs["B0"]["physical_action"]):
        raise ValueError("B0 and A must use the same correct physical actions")
    expected_static = np.tile([0, 0, 0, 1, 0, 0, 0, 1, 0, 0], (16, 1))
    if not np.array_equal(runs["Z"]["physical_action"], expected_static):
        raise ValueError("Z must use physical static translation/rotation-identity/gripper-zero actions")
    times = np.asarray(correct["timestamps"])
    if times.shape != (17,) or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("17 Zarr timestamps must be finite and strictly increasing")
    return has_depth


def compare_reference(correct: dict[str, np.ndarray], reference: dict[str, np.ndarray]) -> dict:
    if not np.array_equal(correct["truth"], reference["truth"]):
        raise ValueError("A40 reference RGB truth differs from the local window")
    result = {"rgb_truth_equal": True, "anchor_convention": "local anchor is observed GT"}
    protocol_keys = ("history_source_indices", "source_indices", "timestamps", "noise_seed", "num_steps")
    for key in protocol_keys:
        if key in reference and not np.array_equal(correct[key], reference[key]):
            raise ValueError(f"A40 reference {key} differs from the local window")
    result["protocol_keys_unavailable_in_reference"] = [key for key in protocol_keys if key not in reference]
    for key, label in (("prediction", "rgb"), ("depth_prediction", "depth")):
        if key not in correct:
            continue
        truth_key = "truth" if label == "rgb" else "depth_truth"
        if key not in reference or truth_key not in reference:
            raise ValueError(f"A40 reference lacks {label} arrays")
        if not np.array_equal(correct[truth_key], reference[truth_key]):
            raise ValueError(f"A40 reference {label} truth differs")
        if correct[key].shape != reference[key].shape or not np.isfinite(reference[key]).all():
            raise ValueError(f"A40 reference {label} prediction shape/values invalid")
        difference = np.abs(correct[key].astype(np.float64) - reference[key].astype(np.float64))
        result[label] = {
            "all_17_max_abs_difference": float(difference.max()),
            "all_17_mean_abs_difference": float(difference.mean()),
            "future_16_max_abs_difference": float(difference[1:].max()),
            "future_16_mean_abs_difference": float(difference[1:].mean()),
        }
    return result


def offline_lpips_metric() -> tuple[Callable, dict]:
    import lpips
    import torch
    from torchvision.models import AlexNet_Weights

    filename = Path(urlparse(AlexNet_Weights.IMAGENET1K_V1.url).path).name
    cache = Path(torch.hub.get_dir()) / "checkpoints" / filename
    if not cache.is_file():
        raise FileNotFoundError(f"offline LPIPS requires the existing AlexNet weights: {cache}")
    model = lpips.LPIPS(net="alex", version="0.1").eval().cpu()

    def score(truth: np.ndarray, prediction: np.ndarray) -> np.ndarray:
        with torch.inference_mode():
            value = model(torch.from_numpy(truth), torch.from_numpy(prediction), normalize=False)
        return value.detach().cpu().numpy().reshape(-1)

    return score, {"net": "alex", "version": "0.1", "device": "cpu", "alexnet_weights": str(cache)}


def score_runs(runs: dict[str, dict], metric: Callable) -> dict:
    truth = runs["A"]["truth"]
    predictions = {"P": np.repeat(truth[:1], 17, axis=0)}
    predictions.update({method: runs[method]["prediction"] for method in ("B0", "A", "Z", "S")})
    truth_nchw = np.moveaxis(truth[1:] * 2 - 1, -1, 1).astype(np.float32)
    result = {}
    for method, prediction in predictions.items():
        pred_nchw = np.moveaxis(prediction[1:] * 2 - 1, -1, 1).astype(np.float32)
        values = np.asarray(metric(truth_nchw, pred_nchw), dtype=np.float64).reshape(-1)
        if values.shape != (16,) or not np.isfinite(values).all() or np.any(values < 0):
            raise ValueError("LPIPS must return 16 finite non-negative future-frame values")
        result[method] = {"rgb_lpips": float(values.mean()), "rgb_lpips_frames": values.tolist()}
        if "depth_truth" in runs["A"]:
            depth_truth = runs["A"]["depth_truth"]
            depth_prediction = (np.repeat(depth_truth[:1], 17, axis=0) if method == "P"
                                else runs[method]["depth_prediction"])
            depth = depth_video_metrics(depth_truth, depth_prediction)
            result[method].update(depth=depth, depth_mae_m=depth["mean"]["depth_mae_m"])
    if "depth_truth" in runs["A"]:
        baseline = result["P"]
        for method in result:
            result[method]["joint_50_50_relative_to_persistence"] = joint_selection_score(
                result[method]["rgb_lpips"], result[method]["depth_mae_m"],
                baseline["rgb_lpips"], baseline["depth_mae_m"], rgb_weight=0.5,
            )
    return result


def depth_pixels(depth: np.ndarray, *, ground_truth: bool) -> np.ndarray:
    """Fixed historical coolwarm anchors; clipping affects display only."""
    low = np.array([59, 76, 192], np.float32)
    middle = np.array([221, 221, 221], np.float32)
    high = np.array([180, 4, 38], np.float32)
    unit = np.clip(np.asarray(depth), 0, 0.5) / 0.5
    pixels = np.where((unit <= 0.5)[..., None],
                      low + (2 * unit)[..., None] * (middle - low),
                      middle + (2 * unit - 1)[..., None] * (high - middle))
    if ground_truth:
        pixels[np.asarray(depth) == 0] = 0
    return np.rint(pixels).clip(0, 255).astype(np.uint8)


def render_frame(runs: dict[str, dict], index: int, fps: float):
    from PIL import Image, ImageDraw

    correct = runs["A"]
    has_depth = "depth_truth" in correct
    image = Image.new("RGB", (768, 48 + (1152 if has_depth else 576)), (20, 20, 20))
    draw = ImageDraw.Draw(image)
    elapsed = float(correct["timestamps"][index] - correct["timestamps"][0])
    draw.text((6, 5), f"One anchor window, frame {index}/16, t={elapsed:.3f}s, playback={fps:.3f}fps", fill="white")
    draw.text((6, 24), "Depth: fixed 0.0m blue / 0.25m gray / 0.5m red; GT missing=black" if has_depth
              else "GT / Persistence / Base / Correct / Static / Mismatched action", fill="white")
    for modality_index, modality in enumerate(("rgb", "depth") if has_depth else ("rgb",)):
        truth_key = "truth" if modality == "rgb" else "depth_truth"
        prediction_key = "prediction" if modality == "rgb" else "depth_prediction"
        for method_index, method in enumerate(METHODS):
            if method == "GT":
                value = correct[truth_key][index]
            elif method == "P":
                value = correct[truth_key][0]
            else:
                value = runs[method][prediction_key][index]
            pixels = (np.rint(value * 255).astype(np.uint8) if modality == "rgb"
                      else depth_pixels(value, ground_truth=method in ("GT", "P")))
            x = (method_index % 3) * 256
            y = 48 + (2 * modality_index + method_index // 3) * 288
            draw.text((x + 6, y + 8), f"{method} | {modality}" + (" [m]" if modality == "depth" else ""), fill="white")
            image.paste(Image.fromarray(pixels), (x, y + 32))
    return image


def render_video(path: Path, runs: dict[str, dict], fps: float) -> None:
    first = render_frame(runs, 0, fps)
    process = subprocess.Popen(
        [find_ffmpeg(), "-n", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{first.width}x{first.height}", "-r", str(fps), "-i", "pipe:0", "-an",
         "-c:v", "libx264", "-preset", "fast", "-crf", "14", "-pix_fmt", "yuv444p",
         "-movflags", "+faststart", str(path)],
        stdin=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    write_error = None
    try:
        for index in range(17):
            if process.poll() is not None:
                raise BrokenPipeError("ffmpeg exited before all frames were written")
            process.stdin.write(np.asarray(render_frame(runs, index, fps)).tobytes())
    except BrokenPipeError as error:
        write_error = str(error)
    finally:
        try:
            process.stdin.close()
        except BrokenPipeError:
            pass
        process.stdin = None
        _, stderr = process.communicate()
    if process.returncode != 0 or write_error:
        raise RuntimeError(f"ffmpeg failed: {write_error or ''} {stderr.decode(errors='replace')}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("correct", "static", "mismatch", "base", "output-prefix"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--reference-npz", type=Path)
    args = parser.parse_args(argv)
    if not args.output_prefix.is_absolute():
        parser.error("--output-prefix must be absolute")
    outputs = {suffix: Path(str(args.output_prefix) + suffix) for suffix in (".json", ".png", ".mp4")}
    for path in outputs.values():
        if path.exists():
            raise FileExistsError(f"refusing to overwrite: {path}")
    runs, paths, metadata = {}, {}, {}
    for method, path in (("A", args.correct), ("Z", args.static), ("S", args.mismatch), ("B0", args.base)):
        with np.load(path, allow_pickle=False) as archive:
            runs[method] = {key: archive[key] for key in archive.files}
        paths[method] = str(path.resolve())
        metadata[method] = json.loads(path.with_suffix(".json").read_text())
    has_depth = validate_runs(runs)
    for method, evidence in metadata.items():
        for key in ("episode", "start"):
            if evidence[key] != metadata["A"][key]:
                raise ValueError(f"{method}: metadata {key} differs")
    report: dict[str, Any] = {
        "scope": "one 17-frame anchor window; future 16 frames scored; no checkpoint selection or long rollout",
        "inputs": paths, "run_metadata": metadata, "panels": list(METHODS),
        "depth_display_range_m": [0, 0.5] if has_depth else None,
        "prediction_depth_policy": "raw predictions scored; display clipping only; no GT masking",
        "poster_frame": 16,
        "video_encoding": {"codec": "libx264", "pixel_format": "yuv444p", "crf": 14},
    }
    if args.reference_npz:
        with np.load(args.reference_npz, allow_pickle=False) as archive:
            report["a40_reference_difference"] = compare_reference(runs["A"], dict(archive))
        report["a40_reference_path"] = str(args.reference_npz.resolve())
    metric, report["lpips_protocol"] = offline_lpips_metric()
    report["metrics"] = score_runs(runs, metric)
    times = runs["A"]["timestamps"]
    duration = float(times[-1] - times[0])
    report.update(actual_duration_seconds=duration, output_fps=16 / duration, conditioning_fps=15.0)
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    render_video(outputs[".mp4"], runs, 16 / duration)
    render_frame(runs, 16, 16 / duration).save(outputs[".png"])
    outputs[".json"].write_text(json.dumps(report, indent=2) + "\n")
    summary = {method: {key: value for key, value in row.items()
                        if key in ("rgb_lpips", "depth_mae_m", "joint_50_50_relative_to_persistence")}
               for method, row in report["metrics"].items()}
    print(json.dumps({"outputs": {key: str(path) for key, path in outputs.items()}, "metrics": summary}))


if __name__ == "__main__":
    main()
