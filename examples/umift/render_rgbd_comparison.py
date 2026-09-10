"""Render audited 12-panel RGB/depth comparisons for E3-Dout suffix rollouts."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageFont


FPS = 15
MODEL_METHODS = ("B0", "E3-A", "E3-Z", "E3-S")
METHODS = ("GT", "P", *MODEL_METHODS)
SCORING_METHODS = ("P", *MODEL_METHODS)
FRAME_METRIC_COLUMNS = (
    "rgb_mse",
    "rgb_psnr",
    "rgb_ssim",
    "rgb_lpips",
    "rgb_temporal_l1",
    "depth_mae_m",
    "depth_rmse_m",
    "depth_valid_fraction",
    "depth_zero_fraction",
    "depth_cap_fraction",
    "depth_out_of_range_fraction",
    "depth_zero_region_mean_m",
    "depth_cap_underprediction_m",
)
MODALITIES = ("rgb", "depth_m")
PANEL_SIZE = 256
WIDTH, HEIGHT = 1648, 900
PANEL_SPECS = tuple((method, modality) for modality in MODALITIES for method in METHODS)
PANEL_BOXES = tuple(
    (16 + column * 272, 154 + row * 322)
    for row in range(2)
    for column in range(6)
)
ZERO_SENTINEL_RGB = (24, 27, 31)
_COOLWARM_LOW = np.array([59, 76, 192], np.float32)
_COOLWARM_MID = np.array([221, 221, 221], np.float32)
_COOLWARM_HIGH = np.array([180, 4, 38], np.float32)


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def depth_color(depth_m: float) -> tuple[int, int, int]:
    """Return the fixed 0–0.5 m coolwarm color for a positive depth value."""
    value = float(np.clip(depth_m, 0.0, 0.5)) / 0.5
    if value <= 0.5:
        color = _COOLWARM_LOW + (2.0 * value) * (_COOLWARM_MID - _COOLWARM_LOW)
    else:
        color = _COOLWARM_MID + (2.0 * value - 1.0) * (_COOLWARM_HIGH - _COOLWARM_MID)
    return tuple(np.rint(color).astype(np.uint8).tolist())


def _depth_pixels(depth_m: np.ndarray) -> np.ndarray:
    depth = np.asarray(depth_m, dtype=np.float32)
    if depth.shape != (256, 256) or not np.isfinite(depth).all():
        raise ValueError("depth display requires finite 256x256 metres")
    clipped = np.clip(depth, 0.0, 0.5)
    unit = clipped / np.float32(0.5)
    pixels = np.empty((256, 256, 3), np.float32)
    lower = unit <= 0.5
    alpha_low = (2.0 * unit)[..., None]
    alpha_high = (2.0 * unit - 1.0)[..., None]
    pixels[:] = _COOLWARM_MID
    pixels[lower] = (
        _COOLWARM_LOW + alpha_low * (_COOLWARM_MID - _COOLWARM_LOW)
    )[lower]
    pixels[~lower] = (
        _COOLWARM_MID + alpha_high * (_COOLWARM_HIGH - _COOLWARM_MID)
    )[~lower]
    pixels[depth == 0.0] = ZERO_SENTINEL_RGB
    return np.rint(pixels).clip(0, 255).astype(np.uint8)


def _rgb_panel(rgb: np.ndarray) -> Image.Image:
    value = np.asarray(rgb, dtype=np.float32)
    if value.shape != (256, 256, 3) or not np.isfinite(value).all():
        raise ValueError("RGB display requires finite 256x256x3")
    pixels = np.rint(np.clip(value, 0.0, 1.0) * 255).astype(np.uint8)
    return Image.fromarray(pixels)


def _depth_panel(depth_m: np.ndarray) -> Image.Image:
    return Image.fromarray(_depth_pixels(depth_m))


def panels_at(
    truth_rgb: np.ndarray,
    truth_depth_m: np.ndarray,
    predictions: dict[str, tuple[np.ndarray, np.ndarray]],
    frame_index: int,
) -> list[Image.Image]:
    """Build six RGB and six depth panels without applying a GT mask to predictions."""
    sources: dict[str, tuple[np.ndarray, np.ndarray]] = {
        "GT": (truth_rgb[frame_index], truth_depth_m[frame_index]),
        "P": (truth_rgb[0], truth_depth_m[0]),
    }
    for method in MODEL_METHODS:
        if method not in predictions:
            raise ValueError(f"missing RGBD prediction method {method}")
        if frame_index == 0:
            sources[method] = (truth_rgb[0], truth_depth_m[0])
        else:
            sources[method] = (
                predictions[method][0][frame_index],
                predictions[method][1][frame_index],
            )
    panels = []
    for method, modality in PANEL_SPECS:
        rgb, depth = sources[method]
        panels.append(_rgb_panel(rgb) if modality == "rgb" else _depth_panel(depth))
    return panels


def _chunk_for_frame(chunks: list[dict[str, Any]], frame_index: int) -> int | None:
    if frame_index == 0:
        return None
    generated_index = frame_index - 1
    for chunk in chunks:
        start = int(chunk["output_start"])
        if start <= generated_index < start + int(chunk["steps"]):
            return int(chunk["index"])
    raise ValueError(f"frame {frame_index} is not covered by a rollout block")


def _draw_colorbar(draw: ImageDraw.ImageDraw, font: ImageFont.ImageFont) -> None:
    left, top, width, height = 1110, 775, 480, 18
    for offset in range(width):
        value = 0.5 * offset / (width - 1)
        draw.line((left + offset, top, left + offset, top + height), fill=depth_color(value))
    draw.rectangle((left, top, left + width, top + height), outline="#8A9BAE", width=1)
    draw.text((left, top + 22), "0 m", font=font, fill="#CBD5E1")
    draw.text((left + width // 2 - 28, top + 22), "0.25 m", font=font, fill="#CBD5E1")
    draw.text((left + width - 42, top + 22), "0.5 m", font=font, fill="#CBD5E1")


def make_frame(
    truth_rgb: np.ndarray,
    truth_depth_m: np.ndarray,
    predictions: dict[str, tuple[np.ndarray, np.ndarray]],
    frame_index: int,
    elapsed_seconds: float,
    episode: dict[str, Any],
    sample_index: int,
    fonts: tuple[ImageFont.ImageFont, ImageFont.ImageFont, ImageFont.ImageFont],
) -> Image.Image:
    canvas = Image.new("RGB", (WIDTH, HEIGHT), "#101827")
    draw = ImageDraw.Draw(canvas)
    title, normal, small = fonts
    block = _chunk_for_frame(episode["chunks"], frame_index)
    block_text = "真实 H5 RGBD 当前帧" if block is None else f"block {block} · 生成 RGBD 自反馈"
    selected_iteration = int(episode["selected_iteration"])
    draw.text((16, 10), "Cosmos3 E3-Dout · RGB / 米制深度十二格全后缀比较", font=title, fill="#F4F7FB")
    draw.text(
        (16, 58),
        f"样本 {sample_index}/3 · episode {episode['episode_id']} · 起点 {episode['start_percent']}% · "
        f"H5 前补 {episode['history_padding_count']} 帧 · {block_text}",
        font=normal,
        fill="#CBD5E1",
    )
    headings = ("GT", "P 首帧保持", "B0 基础 Edge", "E3-A 正确动作", "E3-Z 静止动作", "E3-S 错配动作")
    for column, heading in enumerate(headings):
        x = PANEL_BOXES[column][0]
        detail = (
            "未微调；并非 RGBD 预训练" if column == 2 else
            f"选择步 {selected_iteration}" if column >= 3 else
            "Zarr 真值" if column == 0 else "真实起点 RGBD 固定"
        )
        draw.text((x, 105), heading, font=normal, fill="#F4F7FB")
        draw.text((x, 132), detail, font=small, fill="#AABAD0")
    panels = panels_at(truth_rgb, truth_depth_m, predictions, frame_index)
    for panel, (method, modality), (x, y) in zip(panels, PANEL_SPECS, PANEL_BOXES, strict=True):
        canvas.paste(panel, (x, y))
        draw.rectangle((x - 1, y - 1, x + PANEL_SIZE, y + PANEL_SIZE), outline="#597187", width=1)
        if method == "GT":
            row_label = "RGB" if modality == "rgb" else "Depth (m)"
            draw.text((x, y + PANEL_SIZE + 4), row_label, font=small, fill="#F5D28B")
    total = int(episode["frame_count"])
    episode_elapsed = elapsed_seconds + float(episode.get("initial_episode_elapsed_seconds", 0.0))
    draw.text(
        (16, 775),
        f"帧 {frame_index}/{total - 1} · 本段真实 t={elapsed_seconds:.3f}s · 原 episode t={episode_elapsed:.3f}s · 模型参数 15 Hz",
        font=normal,
        fill="#F5D28B",
    )
    draw.text(
        (16, 812),
        "深度：固定蓝（近）—灰—红（远），0–0.5 米；越界只在显示时夹到端点。",
        font=small,
        fill="#CBD5E1",
    )
    draw.text(
        (16, 840),
        "灰黑只表示该格深度恰为 0；预测没有独立有效性 mask，也不使用 GT mask。",
        font=small,
        fill="#CBD5E1",
    )
    _draw_colorbar(draw, small)
    return canvas


def encode_video(
    path: Path,
    truth_rgb: np.ndarray,
    truth_depth_m: np.ndarray,
    predictions: dict[str, tuple[np.ndarray, np.ndarray]],
    episode: dict[str, Any],
    elapsed: np.ndarray,
    sample_index: int,
    fonts: tuple[ImageFont.ImageFont, ImageFont.ImageFont, ImageFont.ImageFont],
) -> None:
    import av

    clock = Fraction(1, 1_000_000)
    durations = np.r_[np.diff(elapsed), elapsed[-1] - elapsed[-2]]
    with av.open(str(path), mode="w", options={"movflags": "+faststart"}) as container:
        stream = container.add_stream("libx264", rate=FPS, options={"preset": "fast", "crf": "16", "bf": "0"})
        stream.width, stream.height = WIDTH, HEIGHT
        stream.pix_fmt = "yuv444p"
        stream.time_base = stream.codec_context.time_base = clock

        def mux(packet: av.Packet) -> None:
            if packet.pts is None or packet.time_base is None:
                raise ValueError("encoder returned packet without presentation time")
            packet_time = float(packet.pts * packet.time_base)
            index = int(np.argmin(np.abs(elapsed - packet_time)))
            if abs(float(elapsed[index]) - packet_time) > 0.001:
                raise ValueError("encoder changed a source presentation timestamp")
            packet.duration = max(1, round(float(durations[index]) / float(packet.time_base)))
            container.mux(packet)

        for frame_index in range(int(episode["frame_count"])):
            image = make_frame(
                truth_rgb,
                truth_depth_m,
                predictions,
                frame_index,
                float(elapsed[frame_index]),
                episode,
                sample_index,
                fonts,
            )
            frame = av.VideoFrame.from_image(image)
            frame.time_base = clock
            frame.pts = round(float(elapsed[frame_index]) * 1_000_000)
            for packet in stream.encode(frame):
                mux(packet)
        for packet in stream.encode():
            mux(packet)


def _verification_indices(episode: dict[str, Any]) -> set[int]:
    frame_count = int(episode["frame_count"])
    result = {0, frame_count // 2, frame_count - 1}
    for chunk in episode["chunks"]:
        first = int(chunk["output_start"]) + 1
        result.add(first)
        result.add(first + int(chunk["steps"]) - 1)
    return result


def verify_video(
    path: Path,
    truth_rgb: np.ndarray,
    truth_depth_m: np.ndarray,
    predictions: dict[str, tuple[np.ndarray, np.ndarray]],
    episode: dict[str, Any],
    elapsed: np.ndarray,
) -> dict[str, Any]:
    import av

    expected_count = int(episode["frame_count"])
    inspect = _verification_indices(episode)
    decoded_count = 0
    last_duration = None
    max_pts_error = 0.0
    max_panel_mae = 0.0
    frame_pts = []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        observed = (stream.width, stream.height, stream.codec_context.name, stream.codec_context.format.name)
        if observed != (WIDTH, HEIGHT, "h264", "yuv444p"):
            raise ValueError(f"unexpected RGBD video properties: {observed}")
        for decoded_count, frame in enumerate(container.decode(video=0), start=1):
            frame_index = decoded_count - 1
            if frame_index >= expected_count or frame.pts is None:
                raise ValueError("unexpected decoded RGBD frame or missing PTS")
            pts = float(frame.pts * frame.time_base)
            frame_pts.append(pts)
            error = abs(pts - float(elapsed[frame_index]))
            max_pts_error = max(max_pts_error, error)
            if error > 0.001:
                raise ValueError(f"PTS differs from source timestamp at frame {frame_index}: {error}")
            if frame_index in inspect:
                decoded = frame.to_ndarray(format="rgb24")
                expected_panels = panels_at(truth_rgb, truth_depth_m, predictions, frame_index)
                for expected, (x, y) in zip(expected_panels, PANEL_BOXES, strict=True):
                    patch = decoded[y : y + PANEL_SIZE, x : x + PANEL_SIZE].astype(np.float32)
                    mae = float(np.abs(patch - np.asarray(expected, dtype=np.float32)).mean())
                    max_panel_mae = max(max_panel_mae, mae)
                    if mae > 4.0:
                        raise ValueError(f"encoded RGBD panel MAE {mae:.4f} exceeds 4/255")
            if frame_index == expected_count - 1 and frame.duration is not None:
                last_duration = float(frame.duration * frame.time_base)
    if decoded_count != expected_count:
        raise ValueError(f"decoded {decoded_count} RGBD frames, expected {expected_count}")
    expected_last_duration = float(elapsed[-1] - elapsed[-2])
    if last_duration is None or abs(last_duration - expected_last_duration) > 0.001:
        raise ValueError("last RGBD frame duration differs from the final source interval")
    return {
        "frame_count": decoded_count,
        "frame_pts_seconds": frame_pts,
        "model_parameter_fps": FPS,
        "actual_timestamp_span_seconds": float(elapsed[-1]),
        "last_frame_duration_seconds": last_duration,
        "expected_container_duration_seconds": float(elapsed[-1] + expected_last_duration),
        "width": WIDTH,
        "height": HEIGHT,
        "codec": "h264",
        "pixel_format": "yuv444p",
        "verified_frame_indices": sorted(inspect),
        "verified_panel_count": len(PANEL_SPECS),
        "max_pts_error_seconds": max_pts_error,
        "max_panel_encoding_mae_0_255": max_panel_mae,
        "sha256": file_sha(path),
        "bytes": path.stat().st_size,
    }


def _load_array(path: Path, shape: tuple[int, ...], *, bounded: bool) -> np.ndarray:
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    if array.dtype != np.float32 or array.shape != shape:
        raise ValueError(f"invalid float32 RGBD array {path}: {array.dtype} {array.shape}")
    for start in range(0, len(array), 64):
        block = np.asarray(array[start : start + 64])
        if not np.isfinite(block).all() or (bounded and (block.min() < 0.0 or block.max() > 1.0)):
            raise ValueError(f"invalid RGBD values: {path}")
    return array


def _bound_input_sha(episode: dict[str, Any], path: Path) -> str:
    inputs = episode.get("input_files_sha256")
    if not isinstance(inputs, dict):
        raise ValueError("prepared episode lacks frozen input file identities")
    resolved = path.resolve()
    matches = [
        digest
        for filename, digest in inputs.items()
        if Path(filename).resolve() == resolved
    ]
    if len(matches) != 1 or not isinstance(matches[0], str) or len(matches[0]) != 64:
        raise ValueError(f"prepared input SHA binding is missing or ambiguous: {path}")
    return matches[0]


def _load_elapsed(
    path: Path,
    frame_count: int,
    *,
    expected_sha256: str,
    expected_source_start: int,
    expected_timestamps: list[float],
) -> np.ndarray:
    if file_sha(path) != expected_sha256:
        raise ValueError("frozen frame-index file SHA differs from the prepared manifest")
    with np.load(path, allow_pickle=False) as archive:
        source_indices = np.asarray(archive["source_indices"], dtype=np.int64)
        timestamps = np.asarray(archive["timestamps"], dtype=np.float64)
    expected_pts = np.asarray(expected_timestamps, dtype=np.float64)
    expected_indices = int(expected_source_start) + 2 * np.arange(frame_count, dtype=np.int64)
    if (
        source_indices.shape != (frame_count,)
        or timestamps.shape != (frame_count,)
        or expected_pts.shape != (frame_count,)
    ):
        raise ValueError("frozen index/PTS arrays have the wrong suffix length")
    if not np.array_equal(source_indices, expected_indices):
        raise ValueError("frozen suffix source indices differ from anchor + 2*arange")
    if (
        not np.isfinite(timestamps).all()
        or not np.isfinite(expected_pts).all()
        or np.any(np.diff(timestamps) <= 0)
        or not np.array_equal(timestamps, expected_pts)
    ):
        raise ValueError("frozen suffix PTS differ from the prepared manifest true_pts")
    return timestamps - timestamps[0]


def _iter_run_evidence(root: Path) -> Iterable[Path]:
    return sorted((root / "rgbd_inference").glob("run_*_rank*.json"))


def _validate_scoring(
    root: Path,
    manifest_path: Path,
    manifest: dict[str, Any],
    manifest_sha: str,
) -> tuple[dict[str, Any], tuple[Path, Path]]:
    scoring_root = (root / "scoring").resolve()
    metrics_path = scoring_root / "metrics.json"
    frame_path = scoring_root / "frame_metrics.npz"
    report = _read_json(metrics_path)
    if (
        report.get("protocol") != "e3-dout-rgbd-full-suffix-scoring-v1"
        or report.get("experiment_id") != "E3-Dout"
        or report.get("complete") is not True
        or report.get("methods") != list(SCORING_METHODS)
        or Path(report.get("manifest_path", "")).resolve() != manifest_path.resolve()
        or report.get("manifest_sha256") != manifest_sha
        or Path(report.get("frame_metrics_npz", "")).resolve() != frame_path
    ):
        raise ValueError("renderer requires complete scoring bound to this prepared manifest")
    frame_sha = file_sha(frame_path)
    if report.get("frame_metrics_npz_sha256") != frame_sha:
        raise ValueError("scoring frame_metrics NPZ SHA differs from metrics.json")

    expected_rows = []
    for episode in manifest.get("episodes", []):
        for method_index in range(len(SCORING_METHODS)):
            for frame_index in range(1, int(episode["frame_count"])):
                expected_rows.append(
                    (
                        method_index,
                        int(episode["episode_id"]),
                        int(episode["start_percent"]),
                        str(episode["raw_session"]),
                        frame_index,
                    )
                )
    expected_keys = {
        "method_names",
        "method_index",
        "episode_id",
        "start_percent",
        "raw_session",
        "frame_index",
        *FRAME_METRIC_COLUMNS,
    }
    with np.load(frame_path, allow_pickle=False) as archive:
        if set(archive.files) != expected_keys:
            raise ValueError("scoring frame_metrics NPZ schema differs from the scoring contract")
        if tuple(archive["method_names"].tolist()) != SCORING_METHODS:
            raise ValueError("scoring frame_metrics method lookup differs from the scoring contract")
        columns = {
            "method_index": np.asarray(archive["method_index"]),
            "episode_id": np.asarray(archive["episode_id"]),
            "start_percent": np.asarray(archive["start_percent"]),
            "raw_session": np.asarray(archive["raw_session"]),
            "frame_index": np.asarray(archive["frame_index"]),
        }
        expected_dtypes = {
            "method_index": np.dtype(np.int16),
            "episode_id": np.dtype(np.int16),
            "start_percent": np.dtype(np.int16),
            "frame_index": np.dtype(np.int32),
        }
        for name, dtype in expected_dtypes.items():
            if columns[name].dtype != dtype:
                raise ValueError(f"scoring frame_metrics {name} dtype differs from the contract")
        if columns["raw_session"].dtype.kind != "U":
            raise ValueError("scoring frame_metrics raw_session must be Unicode")
        for name in FRAME_METRIC_COLUMNS:
            values = np.asarray(archive[name])
            if values.dtype != np.dtype(np.float64) or values.shape != (len(expected_rows),):
                raise ValueError(f"scoring frame_metrics {name} shape/dtype differs from the contract")
    actual_rows = list(
        zip(
            columns["method_index"].tolist(),
            columns["episode_id"].tolist(),
            columns["start_percent"].tolist(),
            columns["raw_session"].tolist(),
            columns["frame_index"].tolist(),
            strict=True,
        )
    )
    if actual_rows != expected_rows:
        raise ValueError("scoring frame_metrics row identities differ from the prepared suffixes")
    return (
        {
            "protocol": report["protocol"],
            "complete": True,
            "metrics_path": str(metrics_path),
            "metrics_sha256": file_sha(metrics_path),
            "frame_metrics_npz": str(frame_path),
            "frame_metrics_npz_sha256": frame_sha,
            "frame_metric_rows": len(expected_rows),
        },
        (metrics_path, frame_path),
    )


def _validate_metadata(
    path: Path, episode: dict[str, Any], method: str, manifest_sha: str, checkpoint: str
) -> tuple[dict[str, Any], Path, Path]:
    record = _read_json(path)
    frame_indices_sha = _bound_input_sha(episode, Path(episode["frame_indices_path"]))
    true_pts_sha = array_sha(np.asarray(episode["true_pts"], dtype=np.float64))
    for field, expected in (
        ("experiment_id", "E3-Dout"),
        ("method", method),
        ("episode_id", episode["episode_id"]),
        ("start_percent", episode["start_percent"]),
        ("frame_count", episode["frame_count"]),
        ("manifest_sha256", manifest_sha),
        ("checkpoint_id", checkpoint),
        ("future_gt_refresh_count", 0),
        ("selected_iteration", episode["selected_iteration"]),
        ("frame_indices_sha256", frame_indices_sha),
        ("true_pts_sha256", true_pts_sha),
    ):
        if record.get(field) != expected:
            raise ValueError(f"{method}: rollout metadata mismatch for {field}")
    if record.get("complete_requested_suffix") is not True or record.get("initial_h5_rgbd_only") is not True:
        raise ValueError(f"{method}: rollout is incomplete or refreshed from future GT")
    rgb_path = Path(record["rgb_path"])
    depth_path = Path(record["raw_depth_m_path"])
    if not rgb_path.is_absolute() or not depth_path.is_absolute():
        raise ValueError("prediction array paths must be absolute")
    if record.get("rgb_sha256") != file_sha(rgb_path) or record.get("raw_depth_m_sha256") != file_sha(depth_path):
        raise ValueError(f"{method}: prediction hashes differ from rollout metadata")
    chunks = record.get("chunks")
    if not isinstance(chunks, list) or len(chunks) != len(episode["chunks"]):
        raise ValueError(f"{method}: block count differs from frozen suffix")
    previous = None
    for block_index, (expected_chunk, actual) in enumerate(
        zip(episode["chunks"], chunks, strict=True)
    ):
        for field in ("index", "output_start", "steps", "noise_seed"):
            if actual.get(field) != expected_chunk[field]:
                raise ValueError(f"{method}: frozen block identity differs for {field}")
        action_key = "A" if method in ("B0", "E3-A") else method[-1]
        expected_action_hash = expected_chunk[f"{action_key}_physical_action_sha256"]
        if actual.get("physical_action_sha256") != expected_action_hash:
            raise ValueError(f"{method}: physical action hash differs from the frozen fixture")
        expected_history_source = (
            "initial_observed_h5_rgbd" if block_index == 0 else "generated_rolling_rgbd"
        )
        if (
            actual.get("history_source") != expected_history_source
            or actual.get("feedback_projection")
            != "clip RGB to [0,1]; clip depth to [0,0.5]m; gray3; float32"
        ):
            raise ValueError(f"{method}: block history source or float feedback projection differs")
        for digest_field in (
            "history_sha256",
            "feedback_history_sha256",
            "retained_rgb_sha256",
            "retained_raw_depth_m_sha256",
            "padded_model_action_sha256",
        ):
            digest = actual.get(digest_field)
            if not isinstance(digest, str) or len(digest) != 64:
                raise ValueError(f"{method}: missing block digest {digest_field}")
        if previous is not None and actual.get("history_sha256") != previous:
            raise ValueError(f"{method}: generated RGBD history chain is broken")
        previous = actual.get("feedback_history_sha256")
    return record, rgb_path, depth_path


def _font_set(font_path: Path) -> tuple[ImageFont.ImageFont, ImageFont.ImageFont, ImageFont.ImageFont]:
    return tuple(ImageFont.truetype(str(font_path), size, index=2) for size in (36, 22, 17))


def _write_html(destination: Path, inventory: dict[str, Any]) -> Path:
    cards = []
    for item in inventory["episodes"]:
        video = html.escape(item["outputs"]["video"]["file"])
        poster = html.escape(item["outputs"]["posters"]["mid"]["file"])
        cards.append(
            f'<section><h2>episode {item["episode_id"]} · 起点 {item["start_percent"]}%</h2>'
            f'<p>{item["frame_count"]} 帧；真实时间戳跨度 {item["actual_timestamp_span_seconds"]:.3f} 秒。</p>'
            f'<video controls preload="metadata" poster="{poster}" src="{video}"></video></section>'
        )
    content = (
        '<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><title>E3-Dout RGBD</title>'
        '<style>body{background:#101827;color:#f4f7fb;font:17px system-ui;max-width:1500px;margin:30px auto;'
        'padding:0 20px;line-height:1.65}video{width:100%;background:#000}section{margin:38px 0}</style></head><body>'
        '<h1>E3-Dout RGB / depth 十二格全后缀比较</h1><p>六种方法按列，上行为 RGB，下行为米制 depth。'
        'B0 是未微调 Edge，并非 RGBD 预训练模型。所有生成方法仅用初始真实 H5 RGBD，后续跨块使用自身浮点预测反馈。'
        '深度固定为蓝（近）—灰—红（远），范围 0–0.5 米，越界值仅在显示时夹到端点。'
        '灰黑只表示该格深度恰为 0；预测没有独立有效性 mask，也不使用 GT mask。'
        '播放 PTS 使用 stride-2 后保存的真实 Zarr 时间戳。</p>'
        + "".join(cards)
        + '<p><a href="inventory.json">输入、checkpoint、逐块链和媒体 SHA-256 清单</a></p></body></html>'
    )
    path = destination / "index.html"
    path.write_text(content)
    return path


def render(root: Path, destination: Path, font_path: Path) -> dict[str, Any]:
    root, destination, font_path = root.resolve(), destination.resolve(), font_path.resolve()
    if destination.exists():
        raise FileExistsError(destination)
    if not font_path.is_file():
        raise FileNotFoundError(font_path)
    manifest_path = root / "prepared" / "manifest.json"
    manifest = _read_json(manifest_path)
    manifest_sha = file_sha(manifest_path)
    if manifest.get("experiment_id") != "E3-Dout" or manifest.get("protocol") != "e3-dout-rgbd-open-loop-v1":
        raise ValueError("renderer requires an E3-Dout RGBD rollout manifest")
    episodes = manifest.get("episodes")
    if not isinstance(episodes, list) or len(episodes) != 3:
        raise ValueError("renderer requires exactly three frozen held-out episodes")
    run_evidence = list(_iter_run_evidence(root))
    run_records = [_read_json(path) for path in run_evidence]
    phases = {record.get("phase") for record in run_records}
    rank_phase = {(record.get("phase"), record.get("rank")) for record in run_records}
    expected_rank_phase = {(phase, rank) for phase in ("base", "finetuned") for rank in range(4)}
    if phases != {"base", "finetuned"} or rank_phase != expected_rank_phase or len(run_records) != 8:
        raise ValueError("renderer requires base and finetuned strict-load evidence")
    for record in run_records:
        checkpoint = (
            manifest["base_checkpoint"] if record["phase"] == "base" else manifest["selected_checkpoint"]
        )
        if (
            record.get("complete") is not True
            or record.get("load_evidence", {}).get("checkpoint") != checkpoint
        ):
            raise ValueError("run evidence is incomplete or identifies the wrong strict-loaded checkpoint")
    scoring_evidence, scoring_paths = _validate_scoring(
        root, manifest_path, manifest, manifest_sha
    )
    protected = {manifest_path, font_path, *run_evidence, *scoring_paths}
    prepared = []
    for episode in episodes:
        frame_count = int(episode["frame_count"])
        rgb_truth_path = Path(episode["truth_rgb_path"])
        depth_truth_path = Path(episode["truth_depth_m_path"])
        index_path = Path(episode["frame_indices_path"])
        truth_rgb = _load_array(rgb_truth_path, (frame_count, 256, 256, 3), bounded=True)
        truth_depth = _load_array(depth_truth_path, (frame_count, 256, 256), bounded=False)
        if (
            episode.get("truth_rgb_sha256") != file_sha(rgb_truth_path)
            or episode.get("truth_depth_m_sha256") != file_sha(depth_truth_path)
        ):
            raise ValueError("canonical RGBD truth differs from the prepared manifest hashes")
        if truth_depth.min() < 0.0 or truth_depth.max() > 0.5:
            raise ValueError("canonical Zarr depth truth must remain in [0,0.5] metres")
        elapsed = _load_elapsed(
            index_path,
            frame_count,
            expected_sha256=_bound_input_sha(episode, index_path),
            expected_source_start=int(episode["initial_source_frame"]),
            expected_timestamps=episode["true_pts"],
        )
        predictions = {}
        metadata = {}
        source_paths = {}
        for method in MODEL_METHODS:
            checkpoint = manifest["base_checkpoint"] if method == "B0" else manifest["selected_checkpoint"]
            stem = root / "rgbd_inference" / method / f"episode_{episode['episode_id']}_start_{episode['start_percent']}"
            record, rgb_path, depth_path = _validate_metadata(
                stem.with_suffix(".json"), episode, method, manifest_sha, checkpoint
            )
            rgb = _load_array(rgb_path, (frame_count, 256, 256, 3), bounded=True)
            depth = _load_array(depth_path, (frame_count, 256, 256), bounded=False)
            if (
                not np.allclose(rgb[0], truth_rgb[0], atol=2e-7, rtol=0)
                or not np.allclose(depth[0], truth_depth[0], atol=2e-7, rtol=0)
            ):
                raise ValueError(f"{method}: displayed current frame differs from the true RGBD anchor")
            predictions[method] = (rgb, depth)
            metadata[method] = record
            source_paths[method] = (stem.with_suffix(".json"), rgb_path, depth_path)
            protected.update(source_paths[method])
        protected.update((rgb_truth_path, depth_truth_path, index_path))
        prepared.append((episode, truth_rgb, truth_depth, elapsed, predictions, metadata, source_paths))
    before = {path: file_sha(path) for path in sorted(protected)}
    destination.mkdir(parents=True)
    videos = destination / "videos"
    posters = destination / "posters"
    videos.mkdir()
    posters.mkdir()
    fonts = _font_set(font_path)
    inventory = {
        "experiment_id": "E3-Dout",
        "protocol": manifest["protocol"],
        "panel_order": [list(spec) for spec in PANEL_SPECS],
        "panel_layout": "six method columns; RGB top row; depth_m bottom row; no letterbox mask",
        "depth_display": "fixed blue-near through gray to red-far over 0-0.5m; display clips out-of-range values to endpoints; exact-zero gray-black sentinel; predictions have no independent validity mask and use no GT mask",
        "selected_iteration": manifest["selected_iteration"],
        "base_checkpoint_note": "B0 is unfinetuned Edge and is not an RGBD-pretrained checkpoint",
        "time_semantics": "every frame PTS is frozen stride-2 Zarr timestamp minus suffix first timestamp; 15Hz is model parameter",
        "scoring_evidence": scoring_evidence,
        "load_evidence": [
            {"path": str(path), "sha256": before[path], "json": record}
            for path, record in zip(run_evidence, run_records, strict=True)
        ],
        "episodes": [],
    }
    for sample_index, item in enumerate(prepared, start=1):
        episode, truth_rgb, truth_depth, elapsed, predictions, metadata, source_paths = item
        stem = f"episode_{episode['episode_id']}_start_{episode['start_percent']}"
        video_path = videos / f"{stem}_rgbd_comparison.mp4"
        encode_video(video_path, truth_rgb, truth_depth, predictions, episode, elapsed, sample_index, fonts)
        video_record = verify_video(video_path, truth_rgb, truth_depth, predictions, episode, elapsed)
        video_record["file"] = str(video_path.relative_to(destination))
        poster_records = {}
        for label, frame_index in (
            ("start", 0),
            ("mid", int(episode["frame_count"]) // 2),
            ("end", int(episode["frame_count"]) - 1),
        ):
            path = posters / f"{stem}_{label}.png"
            make_frame(
                truth_rgb,
                truth_depth,
                predictions,
                frame_index,
                float(elapsed[frame_index]),
                episode,
                sample_index,
                fonts,
            ).save(path)
            poster_records[label] = {
                "file": str(path.relative_to(destination)),
                "frame_index": frame_index,
                "sha256": file_sha(path),
                "bytes": path.stat().st_size,
            }
        inventory["episodes"].append({
            **episode,
            "actual_timestamp_span_seconds": float(elapsed[-1]),
            "sources": {
                "truth_rgb": {"path": str(episode["truth_rgb_path"]), "sha256": before[Path(episode["truth_rgb_path"])]},
                "truth_depth_m": {"path": str(episode["truth_depth_m_path"]), "sha256": before[Path(episode["truth_depth_m_path"])]},
                "frame_indices": {"path": str(episode["frame_indices_path"]), "sha256": before[Path(episode["frame_indices_path"])]},
                "predictions": {
                    method: {
                        "metadata": metadata[method],
                        "metadata_sha256": before[source_paths[method][0]],
                        "rgb_sha256": before[source_paths[method][1]],
                        "raw_depth_m_sha256": before[source_paths[method][2]],
                    }
                    for method in MODEL_METHODS
                },
            },
            "outputs": {"video": video_record, "posters": poster_records},
        })
        print(json.dumps({"episode": episode["episode_id"], "video": video_record}), flush=True)
    after = {path: file_sha(path) for path in sorted(protected)}
    changed = [str(path) for path in before if before[path] != after[path]]
    if changed:
        raise ValueError(f"protected RGBD inputs changed during rendering: {changed}")
    inventory["protected_inputs_unchanged"] = True
    inventory_path = destination / "inventory.json"
    inventory_path.write_text(json.dumps(inventory, indent=2, ensure_ascii=False) + "\n")
    html_path = _write_html(destination, inventory)
    inventory["artifacts"] = {
        "html": {"file": html_path.name, "sha256": file_sha(html_path)},
        "inventory": {"file": inventory_path.name},
    }
    inventory_path.write_text(json.dumps(inventory, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"inventory": str(inventory_path), "sha256": file_sha(inventory_path)}))
    return inventory


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--font", type=Path, required=True)
    args = parser.parse_args()
    render(args.root, args.output, args.font)


if __name__ == "__main__":
    main()
