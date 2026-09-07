"""Render audited full-episode or midstart-suffix Cosmos3 E1 comparisons.

CPU only.  This consumes immutable prepared truth and saved inference arrays; it
never imports a model, runs inference, or modifies its inputs.
"""

from __future__ import annotations

import argparse
import html
import json
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from examples.umift.render_comparison import BOXES, HEIGHT, PANEL_SIZE, WIDTH, file_sha


FPS = 15
METHODS = ("B0", "E1-A", "E1-Z", "E1-S")
LABELS = (
    ("真实视频 · GT", "数据集 RGB 真值"),
    ("一直复制第一帧", "Persistence · 真实首帧保持"),
    ("未微调的 Edge", "B0 · 正确动作"),
    ("微调 Edge · 正确动作", "E1-A · 第 1000 步"),
    ("微调 Edge · 静止动作", "E1-Z · 第 1000 步"),
    ("微调 Edge · 逐块错配动作", "E1-S · 第 1000 步"),
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _absolute_file(value: Any, field: str) -> Path:
    path = Path(str(value))
    if not path.is_absolute() or not path.is_file():
        raise ValueError(f"{field} must be an existing absolute file: {path}")
    return path


def _validate_truth(path: Path, frame_count: int) -> np.ndarray:
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    if array.dtype != np.uint8 or array.shape != (frame_count, 256, 256, 3):
        raise ValueError(f"invalid truth array {path}: {array.dtype} {array.shape}")
    return array


def _validate_prediction(path: Path, frame_count: int) -> np.ndarray:
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    if array.dtype != np.float32 or array.shape != (frame_count, 256, 256, 3):
        raise ValueError(f"invalid prediction array {path}: {array.dtype} {array.shape}")
    # Scan in bounded temporal slices so validation does not materialize a full episode.
    for start in range(0, frame_count, 64):
        block = np.asarray(array[start : start + 64])
        if not np.isfinite(block).all() or block.min() < 0.0 or block.max() > 1.0:
            raise ValueError(f"prediction contains non-finite or out-of-range RGB: {path}")
    return array


def _normalise_checkpoint_id(value: Any) -> str:
    return str(value).rstrip("/")


def _validate_prediction_metadata(
    path: Path,
    episode: dict[str, Any],
    method: str,
    manifest_sha256: str,
    base_checkpoint: str,
    selected_checkpoint: str,
) -> dict[str, Any]:
    record = _read_json(path)
    if record.get("method") != method or record.get("episode_id") != episode["episode_id"]:
        raise ValueError(f"{method}: method or episode identity mismatch: {path}")
    frame_count = int(episode["frame_count"])
    if "start_percent" in episode:
        if record.get("complete_requested_suffix") is not True:
            raise ValueError(f"{method}: requested suffix is incomplete")
        for field in ("start_percent", "initial_selected_frame", "initial_source_frame", "parent_frame_count"):
            if record.get(field) != episode[field]:
                raise ValueError(f"{method}: suffix origin mismatch: {field}")
    if (("start_percent" not in episode and record.get("complete_episode") is not True)
            or int(record.get("frame_count", -1)) != frame_count):
        raise ValueError(f"{method}: incomplete or wrong-length episode metadata: {path}")
    if record.get("manifest_sha256") != manifest_sha256:
        raise ValueError(f"{method}: inference did not use the current prepared manifest")
    prediction_path = _absolute_file(record.get("prediction_path"), "prediction_path")
    expected = path.with_suffix(".npy").resolve()
    if prediction_path.resolve() != expected:
        raise ValueError(f"{method}: metadata prediction_path does not match {expected}")
    checkpoint = _normalise_checkpoint_id(record.get("checkpoint_id"))
    expected_checkpoint = selected_checkpoint if method.startswith("E1-") else base_checkpoint
    if checkpoint != _normalise_checkpoint_id(expected_checkpoint):
        raise ValueError(f"{method}: unexpected checkpoint_id: {checkpoint}")
    declared_prediction_sha = record.get("prediction_sha256")
    if declared_prediction_sha != file_sha(prediction_path):
        raise ValueError(f"{method}: prediction file differs from inference metadata SHA-256")
    chunks = record.get("chunks")
    if not isinstance(chunks, list) or len(chunks) != len(episode["chunks"]):
        raise ValueError(f"{method}: inference/prepared chunk count differs")
    previous_feedback = None
    for expected_chunk, actual_chunk in zip(episode["chunks"], chunks):
        if int(actual_chunk.get("index", -1)) != int(expected_chunk["index"]):
            raise ValueError(f"{method}: chunk index mismatch")
        for key in ("condition_sha256", "feedback_sha256"):
            digest = actual_chunk.get(key)
            if not isinstance(digest, str) or len(digest) != 64:
                raise ValueError(f"{method}: missing {key} for chunk {expected_chunk['index']}")
        if previous_feedback is not None and actual_chunk["condition_sha256"] != previous_feedback:
            raise ValueError(f"{method}: generated feedback chain is broken")
        previous_feedback = actual_chunk["feedback_sha256"]
    return record


def _validate_episode(episode: dict[str, Any]) -> None:
    required = {
        "episode_id", "raw_session", "source_id", "source_length", "frame_count",
        "fps", "truth_path", "chunks", "frame_indices_path",
    }
    missing = required - episode.keys()
    if missing:
        raise ValueError(f"episode manifest is missing {sorted(missing)}")
    frame_count = int(episode["frame_count"])
    if frame_count < 2 or float(episode["fps"]) != FPS:
        raise ValueError("each episode must contain at least two frames and use model sampling parameter fps=15")
    chunks = episode["chunks"]
    if not isinstance(chunks, list) or not chunks:
        raise ValueError("episode chunks must be a non-empty list")
    covered = 0
    for index, chunk in enumerate(chunks):
        for key in ("output_start", "steps", "index", "noise_seed"):
            if key not in chunk:
                raise ValueError(f"chunk {index} is missing {key}")
        if "start_percent" in episode:
            segments = chunk.get("S_source_segments", [])
            if sum(segment["steps"] for segment in segments) != chunk["steps"]:
                raise ValueError("suffix S source segments do not cover chunk")
        elif "donor_window_id" not in chunk:
            raise ValueError("full episode chunk is missing donor_window_id")
        steps = int(chunk["steps"])
        if int(chunk["index"]) != index or int(chunk["output_start"]) != 16 * index:
            raise ValueError(f"non-canonical chunk index/output_start at chunk {index}")
        if not 1 <= steps <= 16 or int(chunk["output_start"]) != covered:
            raise ValueError(f"invalid or discontinuous chunk {index}")
        covered += steps
    if covered != frame_count - 1:
        raise ValueError(f"chunks cover {covered} predictions, expected {frame_count - 1}")
    if "start_percent" in episode:
        start = episode["initial_selected_frame"]
        parent_count = episode["parent_frame_count"]
        if (start != episode["start_percent"] * (parent_count - 1) // 100
                or episode["initial_source_frame"] != 2 * start
                or frame_count != parent_count - start):
            raise ValueError("suffix origin and frame count disagree")


def _font_set(font_path: Path) -> tuple[ImageFont.FreeTypeFont, ...]:
    return tuple(ImageFont.truetype(str(font_path), size, index=2) for size in (40, 29, 23))


def _load_timestamps(path: Path, frame_count: int) -> np.ndarray:
    with np.load(path, allow_pickle=False) as archive:
        if "timestamps" not in archive:
            raise ValueError(f"frame index archive has no timestamps: {path}")
        timestamps = np.asarray(archive["timestamps"], dtype=np.float64)
    if timestamps.shape != (frame_count,) or not np.isfinite(timestamps).all():
        raise ValueError(f"invalid timestamps in {path}: {timestamps.shape}")
    elapsed = timestamps - timestamps[0]
    if elapsed[0] != 0.0 or np.any(np.diff(elapsed) <= 0):
        raise ValueError(f"timestamps must be strictly increasing: {path}")
    return elapsed


def _to_panel(frame: np.ndarray, *, float_input: bool) -> Image.Image:
    if float_input:
        pixels = np.rint(np.asarray(frame, dtype=np.float32) * 255.0).clip(0, 255).astype(np.uint8)
    else:
        pixels = np.asarray(frame, dtype=np.uint8)
    return Image.fromarray(pixels).resize((PANEL_SIZE, PANEL_SIZE), Image.Resampling.NEAREST)


def _panels_at(truth: np.ndarray, predictions: dict[str, np.ndarray], frame_index: int) -> list[Image.Image]:
    true_first = truth[0]
    if frame_index == 0:
        return [_to_panel(true_first, float_input=False) for _ in LABELS]
    return [
        _to_panel(truth[frame_index], float_input=False),
        _to_panel(true_first, float_input=False),
        *[_to_panel(predictions[method][frame_index], float_input=True) for method in METHODS],
    ]


def _chunk_for_frame(chunks: list[dict[str, Any]], frame_index: int) -> int | None:
    if frame_index == 0:
        return None
    prediction_index = frame_index - 1
    for chunk in chunks:
        start = int(chunk["output_start"])
        if start <= prediction_index < start + int(chunk["steps"]):
            return int(chunk["index"])
    raise ValueError(f"frame {frame_index} is not covered by any chunk")


def _make_frame(
    truth: np.ndarray,
    predictions: dict[str, np.ndarray],
    frame_index: int,
    elapsed_seconds: float,
    episode: dict[str, Any],
    sample_index: int,
    fonts: tuple[ImageFont.FreeTypeFont, ...],
) -> Image.Image:
    canvas = Image.new("RGB", (WIDTH, HEIGHT), "#101827")
    draw = ImageDraw.Draw(canvas)
    title, normal, small = fonts
    total = int(episode["frame_count"])
    block = _chunk_for_frame(episode["chunks"], frame_index)
    block_text = "初始真实条件" if block is None else f"block {block} · 生成自反馈"
    midstart = "start_percent" in episode
    heading = (f"Cosmos3 Edge · 从 {episode['start_percent']}% 帧开始预测至末尾" if midstart
               else "Cosmos3 Edge · 全长开放环六宫格对比")
    draw.text((16, 10), heading, font=title, fill="#F4F7FB")
    draw.text(
        (16, 66),
        f"样本 {sample_index}/3 | episode {episode['episode_id']} | {episode['raw_session']}",
        font=normal,
        fill="#CBD5E1",
    )
    labels = list(LABELS)
    if midstart:
        labels[1] = ("一直复制新输入帧", "Persistence · 33% / 67% 起点保持")
        labels[5] = ("微调 Edge · 错配动作", "E1-S · 原冻结动作流的后缀")
    for (heading, caption), panel, (x, y) in zip(labels, _panels_at(truth, predictions, frame_index), BOXES):
        draw.text((x, y - 66), heading, font=normal, fill="#F4F7FB")
        draw.text((x, y - 29), caption, font=small, fill="#AABAD0")
        canvas.paste(panel, (x, y))
        draw.rectangle((x - 1, y - 1, x + PANEL_SIZE, y + PANEL_SIZE), outline="#597187", width=1)
    timeline = f"帧 {frame_index}/{total - 1} | 实际 t={elapsed_seconds:.3f} 秒 | 模型 15 Hz | {block_text}"
    if midstart:
        episode_time = elapsed_seconds + episode["initial_episode_elapsed_seconds"]
        timeline = (f"本段帧 {frame_index}/{total - 1} | 本段 {elapsed_seconds:.3f}s | "
                    f"原 episode {episode_time:.3f}s | 模型 15 Hz | {block_text}")
    draw.text(
        (16, 1310),
        timeline,
        font=small,
        fill="#F5D28B",
    )
    note = (
        "首帧：六格共享同一真实 I0。"
        if frame_index == 0
        else "此后模型格均为逐块生成结果；块间仅传递上一块末帧，不重置为真实帧。"
    )
    if midstart and frame_index == 0:
        note = (f"新观测：原采样帧 {episode['initial_selected_frame']} / {episode['parent_frame_count'] - 1}，"
                f"Zarr 原始帧 {episode['initial_source_frame']}；六格共享，随后仅生成自反馈。")
    draw.text((16, 1354), note, font=small, fill="#CBD5E1")
    return canvas


def _encode_video(
    path: Path,
    truth: np.ndarray,
    predictions: dict[str, np.ndarray],
    episode: dict[str, Any],
    elapsed: np.ndarray,
    sample_index: int,
    fonts: tuple[ImageFont.FreeTypeFont, ...],
) -> None:
    import av

    clock = Fraction(1, 1_000_000)
    durations = np.r_[np.diff(elapsed), elapsed[-1] - elapsed[-2]]
    with av.open(str(path), mode="w", options={"movflags": "+faststart"}) as container:
        stream = container.add_stream(
            "libx264", rate=FPS, options={"preset": "fast", "crf": "16", "bf": "0"}
        )
        stream.width = WIDTH
        stream.height = HEIGHT
        stream.pix_fmt = "yuv420p"
        stream.time_base = clock
        stream.codec_context.time_base = clock

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
            image = _make_frame(
                truth, predictions, frame_index, float(elapsed[frame_index]), episode, sample_index, fonts
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


def _verify_video(
    path: Path,
    truth: np.ndarray,
    predictions: dict[str, np.ndarray],
    episode: dict[str, Any],
    elapsed: np.ndarray,
) -> dict[str, Any]:
    import av

    expected_count = int(episode["frame_count"])
    inspect = _verification_indices(episode)
    max_mae = 0.0
    max_pts_error = 0.0
    last_duration = None
    decoded_count = 0
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        observed = (
            stream.width, stream.height, stream.average_rate,
            stream.codec_context.name, stream.codec_context.format.name,
        )
        if observed[:2] != (WIDTH, HEIGHT) or observed[3:] != ("h264", "yuv420p"):
            raise ValueError(f"unexpected video properties: {observed}")
        for decoded_count, frame in enumerate(container.decode(video=0), start=1):
            frame_index = decoded_count - 1
            if frame_index >= expected_count or frame.pts is None:
                raise ValueError("unexpected decoded frame or missing PTS")
            pts_seconds = float(frame.pts * frame.time_base)
            pts_error = abs(pts_seconds - float(elapsed[frame_index]))
            max_pts_error = max(max_pts_error, pts_error)
            if pts_error > 0.001:
                raise ValueError(f"PTS differs from source time by {pts_error:.6f}s at frame {frame_index}")
            if frame_index not in inspect:
                continue
            rgb = frame.to_ndarray(format="rgb24")
            expected_panels = _panels_at(truth, predictions, frame_index)
            for expected, (x, y) in zip(expected_panels, BOXES):
                patch = rgb[y : y + PANEL_SIZE, x : x + PANEL_SIZE].astype(np.float32)
                mae = float(np.abs(patch - np.asarray(expected, dtype=np.float32)).mean())
                max_mae = max(max_mae, mae)
                if mae > 3.0:
                    raise ValueError(f"encoded panel MAE {mae:.4f} > 3 on 0-255 scale at frame {frame_index}")
            if frame_index == expected_count - 1 and frame.duration is not None:
                last_duration = float(frame.duration * frame.time_base)
    if decoded_count != expected_count:
        raise ValueError(f"decoded {decoded_count} frames, expected {expected_count}")
    expected_last_duration = float(elapsed[-1] - elapsed[-2])
    if last_duration is None or abs(last_duration - expected_last_duration) > 0.001:
        raise ValueError(f"last frame duration {last_duration} differs from source dt {expected_last_duration}")
    return {
        "frame_count": decoded_count,
        "model_parameter_fps": FPS,
        "actual_timestamp_span_seconds": float(elapsed[-1]),
        "last_frame_duration_seconds": last_duration,
        "expected_container_duration_seconds": float(elapsed[-1] + elapsed[-1] - elapsed[-2]),
        "width": WIDTH,
        "height": HEIGHT,
        "codec": "h264",
        "pixel_format": "yuv420p",
        "verified_frame_indices": sorted(inspect),
        "max_pts_error_seconds": max_pts_error,
        "pts_error_limit_seconds": 0.001,
        "max_panel_encoding_mae_0_255": max_mae,
        "mae_limit_0_255": 3.0,
        "sha256": file_sha(path),
        "bytes": path.stat().st_size,
    }


def _iter_run_evidence(root: Path) -> Iterable[Path]:
    paths = set((root / "inference").glob("**/run_rank*.json"))
    paths.update((root / "inference").glob("**/run_*_rank*.json"))
    return sorted(paths)


def _write_html(destination: Path, inventory: dict[str, Any]) -> Path:
    cards = []
    for index, sample in enumerate(inventory["episodes"], start=1):
        video = html.escape(sample["outputs"]["video"]["file"])
        poster = html.escape(sample["outputs"]["posters"]["mid"]["file"])
        episode_id = html.escape(str(sample["episode_id"]))
        raw_session = html.escape(str(sample["raw_session"]))
        cards.append(
            f'<section><h2>样本 {index} · episode {episode_id}</h2>'
            f'<p>原始 session：{raw_session}；共 {sample["frame_count"]} 帧；实际时间戳跨度 '
            f'{sample["actual_timestamp_span_seconds"]:.3f} 秒，模型采样参数 15 Hz。</p>'
            f'<video controls preload="metadata" poster="{poster}" src="{video}"></video>'
            f'<p><a href="{video}">下载原速单遍 MP4</a> · '
            f'<a href="{html.escape(sample["outputs"]["posters"]["start"]["file"])}">起始图</a> · '
            f'<a href="{html.escape(sample["outputs"]["posters"]["end"]["file"])}">结束图</a></p></section>'
        )
    content = (
        '<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width"><title>Cosmos3 E1 全长六宫格评估</title>'
        '<style>body{background:#101827;color:#f4f7fb;font:17px system-ui;max-width:1200px;margin:36px auto;'
        'padding:0 20px;line-height:1.7}video{width:100%;max-height:85vh;background:#000}section{margin:40px 0}'
        'a{color:#9ad5ff}h1{font-size:30px}</style></head><body>'
        '<h1>Cosmos3 Edge · 三个 history episode 全长开放环比较</h1>'
        '<p>每个视频与对应 GT 具有相同帧数，并按 Zarr 真实时间戳进行变帧率播放；只播放一次且默认不自动循环。'
        '六格在全局第 0 帧共享同一张真实 I0；此后模型逐块生成，并把上一块末帧反馈到下一块，'
        '不在块边界重置为 GT，也不显示每块的 VAE 条件帧重建。</p>'
        '<p>这里必须区分三件事：采集帧率来自原始 Zarr 时间戳；本实验对原始序列 stride=2 取样；'
        '15 Hz 是模型配置中的采样参数。页面和视频时轴使用 stride=2 后保留下来的真实时间戳，'
        '不把帧数除以 15 当作实际时长。</p>'
        '<p>微调模型均使用第 1000 步权重。E1-A 输入正确动作，E1-Z 输入物理静止动作，'
        'E1-S 输入逐块错配动作；B0 使用未微调 Edge。旧 17 帧短窗指标不能直接解释这些开放环长视频。</p>'
        + "".join(cards)
        + '<p>MP4 使用 H.264/yuv420p 有损编码，仅用于目视评估；预测源数组未被覆盖。'
        '<a href="inventory.json">查看输入输出哈希与逐块审计信息</a>。</p></body></html>'
    )
    if "start_percent" in inventory:
        percent = inventory["start_percent"]
        content = content.replace("全长六宫格评估", f"{percent}% 新起点六宫格评估")
        content = content.replace("三个 history episode 全长开放环比较", f"三个 history episode 从 {percent}% 帧至末尾的开放环比较")
        content = content.replace("六格在全局第 0 帧共享同一张真实 I0", "六格在本段第 0 帧共享同一张新的真实观测帧")
        content = content.replace("E1-S 输入逐块错配动作", "E1-S 输入原 P8 冻结错配动作流的对应后缀（新块可能跨两个旧 donor）")
        details = "".join(
            f"<li>episode {e['episode_id']}：原采样帧 {e['initial_selected_frame']}/{e['parent_frame_count'] - 1}，"
            f"Zarr 原始帧 {e['initial_source_frame']}；新起点位于原 episode t={e['initial_episode_elapsed_seconds']:.3f}s，"
            f"剩余实际跨度 {e['actual_timestamp_span_seconds']:.3f}s。</li>"
            for e in inventory["episodes"]
        )
        content = content.replace("</h1>", "</h1><p>百分比按 stride=2 帧索引进度向下取整，"
                                  "不按名义 15 Hz 推算采集时间。GT 只显示同一剩余段。</p><ul>" + details + "</ul>", 1)
    path = destination / "index.html"
    path.write_text(content)
    return path


def render(root: Path, destination: Path, font_path: Path) -> dict[str, Any]:
    root = root.resolve()
    destination = destination.resolve()
    font_path = font_path.resolve()
    if destination.exists():
        raise FileExistsError(f"output directory already exists: {destination}")
    if not font_path.is_file():
        raise FileNotFoundError(font_path)
    manifest_path = root / "prepared" / "manifest.json"
    manifest = _read_json(manifest_path)
    manifest_sha256 = file_sha(manifest_path)
    base_checkpoint = str(manifest.get("base_checkpoint", ""))
    selected_checkpoint = str(manifest.get("selected_checkpoint", ""))
    if not base_checkpoint or not selected_checkpoint or "000001000" not in selected_checkpoint:
        raise ValueError("manifest does not identify the fixed base and selected step-1000 checkpoints")
    episodes = manifest.get("episodes")
    if not isinstance(episodes, list) or len(episodes) != 3:
        raise ValueError("prepared manifest must contain exactly three episodes")
    if len({str(ep.get("episode_id")) for ep in episodes}) != 3:
        raise ValueError("episode_id values must be unique")

    protected_paths: set[Path] = {manifest_path, font_path}
    run_evidence = list(_iter_run_evidence(root))
    if not run_evidence:
        raise ValueError("no inference/**/run_rank*.json load evidence found")
    evidence_phases = set()
    for evidence_path in run_evidence:
        evidence = _read_json(evidence_path)
        if not evidence.get("load_evidence"):
            raise ValueError(f"run evidence has no checkpoint load_evidence: {evidence_path}")
        evidence_phases.add(evidence.get("phase"))
    if not {"base", "finetuned"}.issubset(evidence_phases):
        raise ValueError(f"checkpoint load evidence does not cover base and finetuned phases: {evidence_phases}")
    protected_paths.update(run_evidence)
    prepared: list[dict[str, Any]] = []
    for episode in episodes:
        _validate_episode(episode)
        truth_path = _absolute_file(episode["truth_path"], "truth_path")
        frame_indices_path = _absolute_file(episode["frame_indices_path"], "frame_indices_path")
        protected_paths.update((truth_path, frame_indices_path))
        frame_count = int(episode["frame_count"])
        elapsed = _load_timestamps(frame_indices_path, frame_count)
        if "start_percent" in episode:
            with np.load(frame_indices_path, allow_pickle=False) as times:
                expected_indices = np.arange(episode["initial_source_frame"], episode["source_length"], 2)
                if not np.array_equal(times["source_indices"], expected_indices):
                    raise ValueError("suffix timestamps have different source indices")
                initial_elapsed = float(times["timestamps"][0] - episode["original_episode_first_timestamp"])
                if abs(initial_elapsed - episode["initial_episode_elapsed_seconds"]) > 1e-9:
                    raise ValueError("suffix displayed episode time differs from source timestamp")
        declared_span = episode.get("actual_timestamp_span_seconds")
        if declared_span is not None and abs(float(declared_span) - float(elapsed[-1])) > 1e-9:
            raise ValueError("manifest actual_timestamp_span_seconds differs from timestamp archive")
        truth = _validate_truth(truth_path, frame_count)
        predictions: dict[str, np.ndarray] = {}
        metadata: dict[str, dict[str, Any]] = {}
        prediction_paths: dict[str, Path] = {}
        metadata_paths: dict[str, Path] = {}
        for method in METHODS:
            base = root / "inference" / method / f"episode_{episode['episode_id']}"
            metadata_path = base.with_suffix(".json")
            record = _validate_prediction_metadata(
                metadata_path,
                episode,
                method,
                manifest_sha256,
                base_checkpoint,
                selected_checkpoint,
            )
            prediction_path = Path(record["prediction_path"]).resolve()
            predictions[method] = _validate_prediction(prediction_path, frame_count)
            if not np.array_equal(predictions[method][0], truth[0].astype(np.float32) / 255):
                raise ValueError(f"{method}: first frame differs from observed I0")
            metadata[method] = record
            prediction_paths[method] = prediction_path
            metadata_paths[method] = metadata_path
            protected_paths.update((prediction_path, metadata_path))
        prepared.append(
            {
                "manifest": episode,
                "truth": truth,
                "truth_path": truth_path,
                "frame_indices_path": frame_indices_path,
                "elapsed": elapsed,
                "predictions": predictions,
                "prediction_paths": prediction_paths,
                "metadata": metadata,
                "metadata_paths": metadata_paths,
            }
        )

    before_hashes = {path: file_sha(path) for path in sorted(protected_paths)}
    destination.mkdir(parents=True, exist_ok=False)
    videos_dir = destination / "videos"
    posters_dir = destination / "posters"
    videos_dir.mkdir()
    posters_dir.mkdir()
    fonts = _font_set(font_path)
    inventory: dict[str, Any] = {
        "protocol": "full_episode_open_loop_action_fd",
        "model_parameter_fps": FPS,
        "panel_order": ["GT", "B-Persistence", *METHODS],
        "selected_checkpoint_iteration": 1000,
        "conditioning": "all panels share real I0; generated panels are never reset to GT after frame 0",
        "chunk_display": "generated frames only; per-chunk conditioning reconstruction is not displayed",
        "video_playback": "single-pass VFR from Zarr timestamps; encoder nominal rate 15; HTML has no autoplay/loop",
        "time_semantics": {
            "capture": "actual source Zarr timestamps",
            "sampling": "source frame indices use stride 2",
            "model_parameter_fps": FPS,
            "playback_pts": "sampled timestamp minus first sampled timestamp",
        },
        "font": {"path": str(font_path), "face_index": 2, "sha256": before_hashes[font_path]},
        "load_evidence": [
            {"path": str(path), "sha256": before_hashes[path], "json": _read_json(path)} for path in run_evidence
        ],
        "episodes": [],
    }
    if "start_percent" in manifest:
        if any(e.get("start_percent") != manifest["start_percent"] for e in episodes):
            raise ValueError("mixed suffix origins within one video group")
        inventory.update(protocol="midstart_suffix_open_loop_action_fd", start_percent=manifest["start_percent"],
                         conditioning="all panels share the new observed start frame; no later GT reset",
                         parent_manifest_sha256=manifest["parent_manifest_sha256"])
    for sample_index, item in enumerate(prepared, start=1):
        episode = item["manifest"]
        safe_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in str(episode["episode_id"]))
        video_path = videos_dir / f"episode_{safe_id}_long_comparison.mp4"
        _encode_video(
            video_path, item["truth"], item["predictions"], episode, item["elapsed"], sample_index, fonts
        )
        video_record = _verify_video(
            video_path, item["truth"], item["predictions"], episode, item["elapsed"]
        )
        video_record["file"] = str(video_path.relative_to(destination))
        poster_records: dict[str, Any] = {}
        poster_indices = {"start": 0, "mid": int(episode["frame_count"]) // 2, "end": int(episode["frame_count"]) - 1}
        for name, frame_index in poster_indices.items():
            poster_path = posters_dir / f"episode_{safe_id}_{name}.png"
            _make_frame(
                item["truth"], item["predictions"], frame_index, float(item["elapsed"][frame_index]),
                episode, sample_index, fonts
            ).save(poster_path)
            poster_records[name] = {
                "file": str(poster_path.relative_to(destination)),
                "frame_index": frame_index,
                "sha256": file_sha(poster_path),
                "bytes": poster_path.stat().st_size,
            }
        source_records = {
            "truth": {"path": str(item["truth_path"]), "sha256": before_hashes[item["truth_path"]]},
            "frame_indices": {
                "path": str(item["frame_indices_path"]),
                "sha256": before_hashes[item["frame_indices_path"]],
            },
            "predictions": {
                method: {
                    "path": str(item["prediction_paths"][method]),
                    "sha256": before_hashes[item["prediction_paths"][method]],
                    "metadata_path": str(item["metadata_paths"][method]),
                    "metadata_sha256": before_hashes[item["metadata_paths"][method]],
                    "checkpoint_id": item["metadata"][method]["checkpoint_id"],
                    "chunks": item["metadata"][method]["chunks"],
                }
                for method in METHODS
            },
        }
        inventory["episodes"].append(
            {
                **episode,
                "truth_path": str(item["truth_path"]),
                "frame_indices_path": str(item["frame_indices_path"]),
                "actual_timestamp_span_seconds": float(item["elapsed"][-1]),
                "sources": source_records,
                "outputs": {"video": video_record, "posters": poster_records},
            }
        )
        print(json.dumps({"episode_id": episode["episode_id"], "video": video_record}, ensure_ascii=False), flush=True)

    after_hashes = {path: file_sha(path) for path in sorted(protected_paths)}
    changed = [str(path) for path in before_hashes if before_hashes[path] != after_hashes[path]]
    if changed:
        raise ValueError(f"protected source files changed during rendering: {changed}")
    inventory["protected_inputs_unchanged"] = True
    inventory["protected_inputs"] = {
        str(path): {"before_sha256": before_hashes[path], "after_sha256": after_hashes[path]}
        for path in sorted(protected_paths)
    }
    inventory_path = destination / "inventory.json"
    inventory_path.write_text(json.dumps(inventory, indent=2, ensure_ascii=False) + "\n")
    html_path = _write_html(destination, inventory)
    inventory["artifacts"] = {
        "inventory": {"file": inventory_path.name, "sha256": file_sha(inventory_path)},
        "html": {"file": html_path.name, "sha256": file_sha(html_path)},
    }
    # Refresh inventory once so it includes the HTML hash. Its own hash is intentionally
    # reported by the CLI after the final write rather than recursively embedded.
    inventory["artifacts"]["inventory"].pop("sha256")
    inventory_path.write_text(json.dumps(inventory, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"inventory": str(inventory_path), "sha256": file_sha(inventory_path)}, ensure_ascii=False))
    return inventory


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="long-rollout root containing prepared/ and inference/")
    parser.add_argument("--output", type=Path, required=True, help="new output directory")
    parser.add_argument("--font", type=Path, required=True, help="NotoSansCJK-Regular.ttc; Chinese face index 2")
    args = parser.parse_args()
    render(args.root, args.output, args.font)


if __name__ == "__main__":
    main()
