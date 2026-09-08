"""Render audited E2-H six-panel comparisons for one H/start manifest."""

from __future__ import annotations

import argparse
import html
import json
from fractions import Fraction
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from examples.umift.long_rollout import file_sha
from examples.umift.render_long_comparison import _load_timestamps, _validate_prediction, _validate_truth
from examples.umift.render_comparison import BOXES, HEIGHT, PANEL_SIZE, WIDTH


METHODS = ("B0", "E2-A", "E2-Z", "E2-S")
LABELS = ("真实视频", "首帧保持", "基础 Edge", "E2-H 真实动作", "E2-H 零动作", "E2-H 打乱动作")
FONT_PATH = Path("/data/cosmos_runs/e1_final_eval/video_assets_20260907/NotoSansCJK-Regular.ttc")


def _font(size: int):
    if FONT_PATH.is_file():
        return ImageFont.truetype(str(FONT_PATH), size, index=2)
    fallback = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    return ImageFont.truetype(str(fallback), size) if fallback.is_file() else ImageFont.load_default()


def _panel(frame, floating: bool) -> Image.Image:
    value = np.asarray(frame)
    if floating:
        value = np.rint(value * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(value.astype(np.uint8)).resize((PANEL_SIZE, PANEL_SIZE), Image.Resampling.NEAREST)


def compose_frame(truth, predictions, index: int, elapsed: float, episode: dict, history_frames: int):
    panels = [_panel(truth[index], False), _panel(truth[0], False)]
    panels.extend(_panel(predictions[method][index], True) for method in METHODS)
    image = Image.new("RGB", (WIDTH, HEIGHT), "#101827")
    draw = ImageDraw.Draw(image)
    iteration = int(episode["selected_iteration"])
    padding = int(episode["history_padding_count"])
    real_history = history_frames - padding
    episode_elapsed = float(episode.get("initial_episode_elapsed_seconds", 0.0)) + elapsed
    draw.text((16, 8), f"Cosmos3 Edge E2-H · H={history_frames} · 选中训练步 {iteration}",
              font=_font(35), fill="white")
    draw.text(
        (16, 55),
        f"episode {episode['episode_id']} · 起点 {episode.get('start_percent', 0)}% · "
        f"帧 {index}/{episode['frame_count'] - 1} · 本段真实 elapsed {elapsed:.3f}s · "
        f"原 episode elapsed {episode_elapsed:.3f}s",
        font=_font(22),
        fill="#CBD5E1",
    )
    draw.text((16, 92),
              f"模型参数 15 Hz（视频时长按真实 PTS） · 起点历史：真实 {real_history}/{history_frames}，"
              f"首帧填充 {padding}", font=_font(22), fill="#F5D28B")
    for label, panel, (x, y) in zip(LABELS, panels, BOXES, strict=True):
        draw.text((x, y - 42), label, font=_font(24), fill="white")
        image.paste(panel, (x, y))
    block = 0 if index == 0 else (index - 1) // 16
    remaining = max(0, history_frames - 16 * block)
    if block == 0:
        source = f"初始历史条件（不足部分重复 episode 首帧填充 {padding} 帧）"
    else:
        source = f"生成帧自反馈滚动历史；仍保留初始观测 {remaining} 帧"
    draw.text((16, 1348), source, font=_font(22), fill="#F5D28B")
    return image


def _panels_at(truth, predictions, index: int) -> list[np.ndarray]:
    return [truth[index], truth[0], *(predictions[method][index] for method in METHODS)]


def _encode(path: Path, truth, predictions, episode, elapsed, history_frames):
    import av

    clock = Fraction(1, 1_000_000)
    durations = np.r_[np.diff(elapsed), elapsed[-1] - elapsed[-2]]
    with av.open(str(path), "w", options={"movflags": "+faststart"}) as container:
        stream = container.add_stream("libx264", rate=15, options={"crf": "16", "bf": "0"})
        stream.width, stream.height, stream.pix_fmt, stream.time_base = WIDTH, HEIGHT, "yuv420p", clock
        stream.codec_context.time_base = clock

        def mux(packet) -> None:
            if packet.pts is None or packet.time_base is None:
                raise ValueError("encoder returned packet without presentation time")
            packet_time = float(packet.pts * packet.time_base)
            index = int(np.argmin(np.abs(elapsed - packet_time)))
            if abs(float(elapsed[index]) - packet_time) > 0.001:
                raise ValueError("encoder changed a source presentation timestamp")
            packet.duration = max(1, round(float(durations[index]) / float(packet.time_base)))
            container.mux(packet)

        for index in range(len(elapsed)):
            frame = av.VideoFrame.from_image(
                compose_frame(truth, predictions, index, float(elapsed[index]), episode, history_frames)
            )
            frame.pts, frame.time_base = round(float(elapsed[index]) * 1_000_000), clock
            for packet in stream.encode(frame):
                mux(packet)
        for packet in stream.encode():
            mux(packet)


def _verify_video(path: Path, truth, predictions, episode, elapsed) -> dict:
    import av

    expected_count = int(episode["frame_count"])
    inspect = {0, expected_count // 2, expected_count - 1}
    decoded_count = 0
    max_pts_error = 0.0
    max_panel_mae = 0.0
    last_duration = None
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        observed = (stream.width, stream.height, stream.codec_context.name, stream.codec_context.format.name)
        if observed != (WIDTH, HEIGHT, "h264", "yuv420p"):
            raise ValueError(f"unexpected history video properties: {observed}")
        for decoded_count, frame in enumerate(container.decode(video=0), start=1):
            index = decoded_count - 1
            if index >= expected_count or frame.pts is None:
                raise ValueError("unexpected decoded frame or missing PTS")
            pts_error = abs(float(frame.pts * frame.time_base) - float(elapsed[index]))
            max_pts_error = max(max_pts_error, pts_error)
            if pts_error > 0.001:
                raise ValueError(f"PTS differs from source by {pts_error:.6f}s at frame {index}")
            if index in inspect:
                rgb = frame.to_ndarray(format="rgb24")
                for expected, (x, y) in zip(_panels_at(truth, predictions, index), BOXES, strict=True):
                    expected_rgb = np.asarray(expected)
                    if np.issubdtype(expected_rgb.dtype, np.floating):
                        expected_rgb = np.rint(expected_rgb * 255).clip(0, 255)
                    expected_rgb = np.asarray(
                        Image.fromarray(expected_rgb.astype(np.uint8)).resize(
                            (PANEL_SIZE, PANEL_SIZE), Image.Resampling.NEAREST
                        )
                    )
                    patch = rgb[y : y + PANEL_SIZE, x : x + PANEL_SIZE].astype(np.float32)
                    mae = float(np.abs(patch - expected_rgb.astype(np.float32)).mean())
                    max_panel_mae = max(max_panel_mae, mae)
                    if mae > 3.0:
                        raise ValueError(f"encoded history panel MAE {mae:.4f} at frame {index}")
            if index == expected_count - 1 and frame.duration is not None:
                last_duration = float(frame.duration * frame.time_base)
    if decoded_count != expected_count:
        raise ValueError(f"decoded {decoded_count} frames, expected {expected_count}")
    expected_last_duration = float(elapsed[-1] - elapsed[-2])
    if last_duration is None or abs(last_duration - expected_last_duration) > 0.001:
        raise ValueError(f"last frame duration {last_duration} differs from source {expected_last_duration}")
    return {
        "frame_count": decoded_count,
        "frame_pts_seconds": np.asarray(elapsed, dtype=float).tolist(),
        "actual_timestamp_span_seconds": float(elapsed[-1]),
        "last_frame_duration_seconds": last_duration,
        "expected_container_duration_seconds": float(elapsed[-1] + expected_last_duration),
        "max_pts_error_seconds": max_pts_error,
        "max_panel_mae_0_255": max_panel_mae,
        "codec": "h264",
        "pixel_format": "yuv420p",
        "width": WIDTH,
        "height": HEIGHT,
    }


def render(root: Path, history_frames: int, output_dir: Path) -> dict:
    manifest_path = root / "prepared/manifest.json"
    manifest_sha = file_sha(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("experiment_id") != "E2-H" or manifest.get("history_frames") != history_frames:
        raise ValueError("manifest does not match E2-H/history_frames")
    if not manifest.get("selected_checkpoint") or int(manifest.get("selected_iteration", -1)) < 0:
        raise ValueError("manifest must freeze selected_checkpoint and selected_iteration")
    if output_dir.exists():
        raise FileExistsError(output_dir)
    episodes = manifest.get("episodes")
    if not isinstance(episodes, list) or len(episodes) != 3:
        raise ValueError("renderer requires exactly three episodes")
    output_dir.mkdir(parents=True)
    (output_dir / "videos").mkdir()
    (output_dir / "posters").mkdir()
    inventory = {"experiment_id": "E2-H", "history_frames": history_frames, "episodes": []}
    cards = []
    inference_root = root / "history_inference" / f"H{history_frames}"
    for episode in episodes:
        count = int(episode["frame_count"])
        truth = _validate_truth(Path(episode["truth_path"]), count)
        elapsed = _load_timestamps(Path(episode["frame_indices_path"]), count)
        predictions = {}
        sources = {}
        label = str(episode.get("start_percent", 0))
        for method in METHODS:
            base = inference_root / method / f"episode_{episode['episode_id']}_start_{label}"
            metadata = json.loads(base.with_suffix(".json").read_text())
            expected_checkpoint = manifest["base_checkpoint"] if method == "B0" else manifest["selected_checkpoint"]
            if (metadata.get("checkpoint_id") != expected_checkpoint
                    or metadata.get("manifest_sha256") != manifest_sha
                    or metadata.get("method") != method or metadata.get("episode_id") != episode["episode_id"]
                    or metadata.get("start_percent") != episode["start_percent"]):
                raise ValueError(f"{method} checkpoint or suffix identity differs from the manifest")
            if metadata.get("history_frames") != history_frames or not metadata.get("complete_requested_suffix"):
                raise ValueError(f"incomplete/mismatched {method} metadata")
            if len(metadata["chunks"]) != len(episode["chunks"]):
                raise ValueError(f"{method} does not cover all frozen chunks")
            true_pts = np.asarray(metadata.get("true_pts"), dtype=np.float64)
            if true_pts.shape != (count,) or not np.allclose(true_pts - true_pts[0], elapsed, atol=1e-9):
                raise ValueError(f"{method} metadata true PTS differ from the frozen archive")
            previous = None
            for chunk in metadata["chunks"]:
                if previous is not None and chunk["history_sha256"] != previous:
                    raise ValueError(f"broken {method} history hash chain")
                previous = chunk["feedback_history_sha256"]
            prediction_path = base.with_suffix(".npy")
            if metadata.get("prediction_sha256") != file_sha(prediction_path):
                raise ValueError(f"changed prediction: {prediction_path}")
            predictions[method] = _validate_prediction(prediction_path, count)
            if not np.array_equal(predictions[method][0], truth[0].astype(np.float32) / 255):
                raise ValueError(f"{method} first output frame differs from the frozen anchor")
            sources[method] = {
                "file": str(prediction_path),
                "sha256": file_sha(prediction_path),
                "metadata": str(base.with_suffix(".json")),
                "metadata_sha256": file_sha(base.with_suffix(".json")),
            }
        video = output_dir / "videos" / f"episode_{episode['episode_id']}_start_{label}.mp4"
        _encode(video, truth, predictions, episode, elapsed, history_frames)
        video_verification = _verify_video(video, truth, predictions, episode, elapsed)
        posters = {}
        for name, index in (("start", 0), ("mid", count // 2), ("end", count - 1)):
            path = output_dir / "posters" / f"episode_{episode['episode_id']}_{name}.png"
            compose_frame(truth, predictions, index, float(elapsed[index]), episode, history_frames).save(path)
            posters[name] = {"file": str(path.relative_to(output_dir)), "sha256": file_sha(path)}
        item = {
            **episode,
            "sources": sources,
            "video": {"file": str(video.relative_to(output_dir)), "sha256": file_sha(video),
                      "verification": video_verification},
            "posters": posters,
        }
        inventory["episodes"].append(item)
        cards.append(
            f'<h2>episode {episode["episode_id"]}</h2><video controls '
            f'poster="{posters["mid"]["file"]}" src="{item["video"]["file"]}"></video>'
        )
    inventory_path = output_dir / "inventory.json"
    inventory_path.write_text(json.dumps(inventory, indent=2) + "\n")
    page = (
        "<!doctype html><meta charset=utf-8><title>E2-H comparison</title>"
        "<style>body{background:#101827;color:white;max-width:1200px;margin:auto}"
        "video{width:100%}</style><h1>E2-H history rollout</h1>"
        + "".join(cards)
        + f'<p><a href="{html.escape(inventory_path.name)}">inventory</a></p>'
    )
    (output_dir / "index.html").write_text(page)
    return inventory


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--history-frames", type=int, choices=(1, 5, 9, 17), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    render(args.root.resolve(), args.history_frames, args.output_dir.resolve())


if __name__ == "__main__":
    main()
