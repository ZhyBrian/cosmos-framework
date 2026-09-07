"""Render three frozen E1 history windows as audited six-panel MP4 comparisons.

CPU only: consumes saved predictions, never imports a model or edits source arrays.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from examples.umift.protocol import derive_noise_seed, persistence_prediction

METHODS = ("B-Persistence", "B0", "E1-A", "E1-Z", "E1-S")
LABELS = (
    ("真实视频 · GT", "数据集 RGB 真值 · 未经 VAE 重建"),
    ("一直复制第一帧", "Persistence | 每一帧都保持真实首帧"),
    ("未微调的 Edge", "B0 | 基础权重 + 正确动作"),
    ("微调 Edge · 正确动作", "E1-A | 最佳权重：第 1000 步"),
    ("微调 Edge · 静止动作", "E1-Z | 同一最佳权重 + 物理静止动作"),
    ("微调 Edge · 错配动作", "E1-S | 同一最佳权重 + 冻结错配轨迹"),
)
FPS = 15
WIDTH, HEIGHT = 1600, 1400
PANEL_SIZE = 512
BOXES = tuple((16 + 528 * (i % 3), 174 + 606 * (i // 3)) for i in range(6))


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def select_middle_windows(rows: list[dict]) -> list[dict]:
    if len({row["window_id"] for row in rows}) != len(rows):
        raise ValueError("duplicate frozen window")
    sessions = sorted({row["raw_session"] for row in rows})
    if len(sessions) != 3 or any(row["split"] != "history" for row in rows):
        raise ValueError("expected exactly three history sessions")
    selected = []
    for session in sessions:
        members = sorted((r for r in rows if r["raw_session"] == session),
                         key=lambda r: (r["timestamp_start"], r["window_id"]))
        selected.append(members[len(members) // 2])
    return selected


def index_seed_zero(path: Path) -> dict[str, dict]:
    result = {}
    for row in read_jsonl(path):
        if row["sampling_seed"] != 0:
            continue
        key = row["window_id"]
        if key in result:
            raise ValueError(f"duplicate window/seed in {path}: {key}")
        result[key] = row
    return result


def validate_rows(rows: dict[str, dict], frozen: dict, selected_checkpoint: str) -> None:
    for method in METHODS:
        row = rows[method]
        for key, expected in (("method", method), ("window_id", frozen["window_id"]),
                              ("raw_session", frozen["raw_session"]),
                              ("episode", frozen["source_id"]), ("split", "history"),
                              ("sampling_seed", 0),
                              ("noise_seed", derive_noise_seed(frozen["window_id"], 0))):
            if row[key] != expected:
                raise ValueError(f"{method}: mismatched {key}")
        checkpoint = row["checkpoint_id"]
        if method.startswith("E1-") and checkpoint != selected_checkpoint:
            raise ValueError(f"{method}: not the selected checkpoint")
        if method == "B0" and checkpoint != "/data/cosmos_models/Cosmos3-Edge-DCP-a9d944e/model":
            raise ValueError("B0: not the fixed base Edge checkpoint")
        if method == "B-Persistence" and checkpoint is not None:
            raise ValueError("Persistence must not load a model checkpoint")


def load_array(path: Path) -> np.ndarray:
    array = np.load(path, allow_pickle=False)
    if array.shape != (17, 256, 256, 3) or not np.isfinite(array).all():
        raise ValueError(f"invalid video shape or values: {path}")
    if array.min() < 0 or array.max() > 1:
        raise ValueError(f"RGB outside [0,1]: {path}")
    return array


def validate_arrays(truths: list[np.ndarray], predictions: list[np.ndarray]) -> None:
    if any(not np.array_equal(truths[0], truth) for truth in truths[1:]):
        raise ValueError("truth arrays are not identical")
    if not np.array_equal(predictions[0], persistence_prediction(truths[0])):
        raise ValueError("Persistence is not exact repetition of truth[0]")
    if any(not np.array_equal(predictions[1][0], prediction[0]) for prediction in predictions[2:]):
        raise ValueError("model condition reconstructions differ")


def timeline(review: bool) -> list[tuple[int, str]]:
    if not review:
        return [(k, "原速 1.00×") for k in range(17)]
    normal = [(k, f"原速 1.00× · 第 {loop}/3 遍") for loop in range(1, 4) for k in range(17)]
    slow = [(k, f"四倍慢放 0.25× · 第 {loop}/2 遍")
            for loop in range(1, 3) for k in range(17) for _ in range(4)]
    return normal + slow


def panel_images(arrays: list[np.ndarray]) -> list[list[Image.Image]]:
    return [[Image.fromarray(np.rint(array[k] * 255).astype(np.uint8)).resize(
        (PANEL_SIZE, PANEL_SIZE), Image.Resampling.NEAREST) for array in arrays] for k in range(17)]


def make_frame(panels: list[Image.Image], k: int, label: str, sample: dict,
               index: int, fonts: tuple) -> Image.Image:
    canvas = Image.new("RGB", (WIDTH, HEIGHT), "#101827")
    draw = ImageDraw.Draw(canvas)
    title, normal, small = fonts
    draw.text((16, 10), "Cosmos3 Edge · 六宫格同步视频对比", font=title, fill="#F4F7FB")
    draw.text((16, 66), f"样本 {index}/3 | {sample['window_id']} | 模型生成 seed=0", font=normal, fill="#CBD5E1")
    for i, ((heading, caption), image, (x, y)) in enumerate(zip(LABELS, panels, BOXES)):
        draw.text((x, y - 66), heading, font=normal, fill="#F4F7FB")
        draw.text((x, y - 29), caption, font=small, fill="#AABAD0")
        canvas.paste(image, (x, y))
        draw.rectangle((x - 1, y - 1, x + PANEL_SIZE, y + PANEL_SIZE), outline="#597187", width=1)
    draw.text((16, 1310), f"{label} | 帧 k={k:02d}/16 | 名义相对时间 {k / FPS:.3f} 秒", font=normal, fill="#F5D28B")
    note = ("k=0：GT/复制格为原始首帧；四个模型格为条件帧重建。"
            if k == 0 else "k=1–16：未来预测。六格同步；完整画面；慢放只重复帧，不插值、不补帧。")
    draw.text((16, 1354), note, font=small, fill="#CBD5E1")
    return canvas


def encode_video(path: Path, schedule: list[tuple[int, str]], panels: list[list[Image.Image]],
                 sample: dict, index: int, fonts: tuple) -> None:
    import imageio_ffmpeg

    command = [imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-n",
               "-f", "rawvideo", "-vcodec", "rawvideo", "-pix_fmt", "rgb24",
               "-s", f"{WIDTH}x{HEIGHT}", "-r", str(FPS), "-i", "-", "-an",
               "-c:v", "libx264", "-preset", "fast", "-crf", "16", "-pix_fmt", "yuv420p",
               "-movflags", "+faststart", str(path)]
    with subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE) as process:
        try:
            for k, label in schedule:
                process.stdin.write(make_frame(panels[k], k, label, sample, index, fonts).tobytes())
            process.stdin.close()
            error = process.stderr.read().decode()
            if process.wait() != 0:
                raise RuntimeError(error)
        except BaseException:
            process.terminate()
            process.wait()
            raise


def verify_video(path: Path, schedule: list[tuple[int, str]], panels: list[list[Image.Image]]) -> dict:
    import av

    max_mae = 0.0
    count = 0
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        if (stream.width, stream.height, stream.average_rate, stream.codec_context.name,
                stream.codec_context.format.name) != (WIDTH, HEIGHT, FPS, "h264", "yuv420p"):
            raise ValueError("unexpected video dimensions, rate or codec")
        for count, frame in enumerate(container.decode(video=0), start=1):
            if count > len(schedule) or abs(float(frame.pts * frame.time_base) - (count - 1) / FPS) > 1e-6:
                raise ValueError("unexpected frame count or presentation timestamp")
            k = schedule[count - 1][0]
            rgb = frame.to_ndarray(format="rgb24")
            for image, (x, y) in zip(panels[k], BOXES):
                patch = rgb[y:y + PANEL_SIZE, x:x + PANEL_SIZE].astype(np.float32)
                mae = float(np.abs(patch - np.asarray(image, dtype=np.float32)).mean())
                max_mae = max(max_mae, mae)
                if mae > 3.0:
                    raise ValueError(f"decoded panel differs from expected source frame: MAE={mae}")
        if count != len(schedule):
            raise ValueError(f"truncated video: {count} != {len(schedule)}")
    return {"frame_count": count, "fps": FPS, "duration_seconds": count / FPS,
            "width": WIDTH, "height": HEIGHT, "codec": "h264", "pixel_format": "yuv420p",
            "max_panel_encoding_mae_0_255": max_mae, "sha256": file_sha(path), "bytes": path.stat().st_size}


def render(root: Path, destination: Path, font_path: Path) -> dict:
    selected = json.loads((root / "metrics/checkpoint_selection.json").read_text())["selected"]
    if selected["iteration"] != 1000:
        raise ValueError("this E1 visualization requires the audited selected iteration 1000")
    frozen_rows = read_jsonl(root / "frozen_v2/history_windows.jsonl")
    samples = select_middle_windows(frozen_rows)
    pairs_path = root / "frozen_v2/history_S_pairs.json"
    pairs = json.loads(pairs_path.read_text())
    frozen_ids = {row["window_id"] for row in frozen_rows}
    sources = {method: root / "history" / f"history_{method}.jsonl" for method in METHODS}
    indexes = {method: index_seed_zero(path) for method, path in sources.items()}
    fonts = tuple(ImageFont.truetype(str(font_path), size, index=2) for size in (40, 29, 23))
    destination.mkdir(parents=True, exist_ok=False)
    inventory = {"selection_rule": "middle by (timestamp_start, window_id) in each of three history sessions",
                 "sampling_seed": 0, "selected_checkpoint": selected, "panel_order": ["GT", *METHODS],
                 "condition_frame": "saved model reconstruction retained; scored future is k=1..16",
                 "display": "2x nearest-neighbor; uint8 rounding; no crop or temporal interpolation",
                 "font_path": str(font_path), "font_sha256": file_sha(font_path), "samples": []}
    protected = {path: file_sha(path) for path in [*sources.values(), pairs_path,
                  root / "frozen_v2/history_windows.jsonl", root / "metrics/checkpoint_selection.json"]}
    for index, sample in enumerate(samples, start=1):
        window = sample["window_id"]
        rows = {method: indexes[method][window] for method in METHODS}
        validate_rows(rows, sample, selected["checkpoint"])
        replacement = pairs[window]
        if replacement == window or replacement not in frozen_ids:
            raise ValueError("invalid frozen shuffled-action replacement")
        predictions, truths, inputs = [], [], []
        for method in METHODS:
            row = rows[method]
            paths = {key: sources[method].parent / row[key] for key in ("prediction_path", "truth_path")}
            for path in paths.values():
                protected[path] = file_sha(path)
            truths.append(load_array(paths["truth_path"]))
            predictions.append(load_array(paths["prediction_path"]))
            inputs.append({**row, "source_manifest": str(sources[method]),
                           "files": {key: {"path": str(path), "sha256": protected[path]} for key, path in paths.items()}})
        validate_arrays(truths, predictions)
        panels = panel_images([truths[0], *predictions])
        stem = f"sample{index}_" + window.replace(":", "_").replace("=", "-")
        outputs = {}
        for review in (False, True):
            name = "review" if review else "native15fps"
            path = destination / f"{stem}_{name}.mp4"
            schedule = timeline(review)
            encode_video(path, schedule, panels, sample, index, fonts)
            outputs[name] = {"file": path.name, **verify_video(path, schedule, panels),
                             "source_frame_indices": [k for k, _ in schedule]}
        poster = destination / f"{stem}_poster.png"
        make_frame(panels[8], 8, "静态预览 · 中间帧", sample, index, fonts).save(poster)
        outputs["poster"] = {"file": poster.name, "sha256": file_sha(poster)}
        inventory["samples"].append({**sample, "replacement_window_id": replacement,
                                     "noise_seed": rows["E1-A"]["noise_seed"], "sources": inputs,
                                     "outputs": outputs})
        print(json.dumps({"window_id": window, "outputs": outputs}, ensure_ascii=False), flush=True)
    for path, before in protected.items():
        if file_sha(path) != before:
            raise ValueError(f"source changed during rendering: {path}")
    inventory["source_files_unchanged"] = True
    inventory["protected_inputs"] = {str(path): digest for path, digest in protected.items()}
    (destination / "inventory.json").write_text(json.dumps(inventory, indent=2, ensure_ascii=False) + "\n")
    cards = []
    for i, sample in enumerate(inventory["samples"], start=1):
        outputs = sample["outputs"]
        cards.append(f'<section><h2>样本 {i} · {sample["window_id"]}</h2>'
                     f'<p>原始 session：{sample["raw_session"]}；错配动作来自 {sample["replacement_window_id"]}</p>'
                     f'<video controls loop preload="metadata" poster="{outputs["poster"]["file"]}" '
                     f'src="{outputs["review"]["file"]}"></video><p><a href="{outputs["native15fps"]["file"]}">'
                     '原速单遍视频</a> · 观看版：3 遍原速 + 2 遍四倍慢放。</p></section>')
    html = ('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width">'
            '<title>Cosmos3 E1 六宫格视频对比</title><style>body{background:#101827;color:#f4f7fb;font:17px system-ui;'
            'max-width:1200px;margin:36px auto;padding:0 20px;line-height:1.7}video{width:100%;max-height:85vh;'
            'background:#000}section{margin:36px 0}a{color:#9ad5ff}h1{font-size:28px}</style>'
            '<h1>Cosmos3 Edge · 三个历史留出样本</h1><p>上排：真实视频 / 复制首帧 / 未微调 Edge；'
            '下排：微调后正确动作 / 静止动作 / 错配动作。每个原始 session 取时间排序中间窗，未按效果挑选。</p>'
            '<p>最佳权重：第 1000 步；生成 seed=0。每段17帧、名义15Hz，首尾跨度约1.067秒。'
            '第0帧是条件诊断，未来预测为第1–16帧。保留原预测，不插帧；慢放只重复展示帧。</p>'
            + ''.join(cards) + '<p>视频采用H.264有损压缩，只作观看；原数值评价来自未修改的float32数组。'
            '历史留出不是新患者盲测；这三个短片不替代完整评价。<a href="inventory.json">可核验来源清单</a></p></html>')
    (destination / "index.html").write_text(html)
    return inventory


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--font", type=Path, required=True, help="NotoSansCJK-Regular.ttc; Simplified Chinese face index 2")
    args = parser.parse_args()
    render(args.root, args.output, args.font)
