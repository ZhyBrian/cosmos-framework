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
LABELS = ("GT", "Persistence", "Base Edge", "E2-H action", "E2-H zero", "E2-H shuffled")


def _font(size: int):
    path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    return ImageFont.truetype(str(path), size) if path.is_file() else ImageFont.load_default()


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
    draw.text((16, 12), f"Cosmos3 Edge E2-H · H={history_frames}", font=_font(40), fill="white")
    draw.text(
        (16, 68),
        f"episode {episode['episode_id']} · start {episode.get('start_percent', 0)}% · "
        f"frame {index}/{episode['frame_count'] - 1} · true PTS {elapsed:.3f}s",
        font=_font(24),
        fill="#CBD5E1",
    )
    for label, panel, (x, y) in zip(LABELS, panels, BOXES, strict=True):
        draw.text((x, y - 42), label, font=_font(24), fill="white")
        image.paste(panel, (x, y))
    block = 0 if index == 0 else (index - 1) // 16
    remaining = max(0, history_frames - 16 * block)
    source = "initial history" if block == 0 else f"rolling feedback; {remaining} initial frame(s) remain"
    draw.text((16, 1348), source, font=_font(22), fill="#F5D28B")
    return image


def _encode(path: Path, truth, predictions, episode, elapsed, history_frames):
    import av

    clock = Fraction(1, 1_000_000)
    with av.open(str(path), "w", options={"movflags": "+faststart"}) as container:
        stream = container.add_stream("libx264", rate=15, options={"crf": "16", "bf": "0"})
        stream.width, stream.height, stream.pix_fmt, stream.time_base = WIDTH, HEIGHT, "yuv420p", clock
        for index in range(len(elapsed)):
            frame = av.VideoFrame.from_image(
                compose_frame(truth, predictions, index, float(elapsed[index]), episode, history_frames)
            )
            frame.pts, frame.time_base = round(float(elapsed[index]) * 1_000_000), clock
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def render(root: Path, history_frames: int, output_dir: Path) -> dict:
    manifest_path = root / "prepared/manifest.json"
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
            if metadata.get("history_frames") != history_frames or not metadata.get("complete_requested_suffix"):
                raise ValueError(f"incomplete/mismatched {method} metadata")
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
        posters = {}
        for name, index in (("start", 0), ("mid", count // 2), ("end", count - 1)):
            path = output_dir / "posters" / f"episode_{episode['episode_id']}_{name}.png"
            compose_frame(truth, predictions, index, float(elapsed[index]), episode, history_frames).save(path)
            posters[name] = {"file": str(path.relative_to(output_dir)), "sha256": file_sha(path)}
        item = {
            **episode,
            "sources": sources,
            "video": {"file": str(video.relative_to(output_dir)), "sha256": file_sha(video)},
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
