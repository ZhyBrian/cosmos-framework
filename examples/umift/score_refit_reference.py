"""Score the preregistered three-session, first-window E1-R reference panel."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np

from examples.umift.evaluate import aggregate_session_equal, evaluate_video_pair
from examples.umift.long_rollout import BASE_CHECKPOINT, BEST_CHECKPOINT, EPISODES, array_sha, file_sha
from examples.umift.protocol import persistence_prediction

DEFAULT_ROOT = Path("/data/cosmos_runs/e1_refit_20260907")
DEFAULT_OLD_ROOT = Path("/data/cosmos_runs/e1_long_rollout_20260907")


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _episodes(manifest: dict[str, Any], *, label: str) -> dict[int, dict[str, Any]]:
    records = manifest.get("episodes")
    if not isinstance(records, list):
        raise ValueError(f"{label} has no episode list")
    result = {int(record["episode_id"]): record for record in records}
    if tuple(result) != EPISODES:
        raise ValueError(f"{label} must contain history episodes {EPISODES} in order")
    sessions = [str(result[episode_id]["raw_session"]) for episode_id in EPISODES]
    if len(set(sessions)) != len(sessions):
        raise ValueError(f"{label} episodes do not represent three distinct raw sessions")
    return result


def _validate_input_files(episodes: dict[int, dict[str, Any]], *, label: str) -> None:
    for episode_id, episode in episodes.items():
        hashes = episode.get("input_files_sha256")
        if not isinstance(hashes, dict) or not hashes:
            raise ValueError(f"{label} episode {episode_id} has no input file hashes")
        for filename, expected in hashes.items():
            path = Path(filename)
            if not path.is_file() or file_sha(path) != expected:
                raise ValueError(f"{label} input hash differs: {filename}")
        truth_path = str(episode["truth_path"])
        if truth_path not in hashes:
            raise ValueError(f"{label} truth_path is absent from input_files_sha256")


def _same_frozen_inputs(
    left: dict[int, dict[str, Any]], right: dict[int, dict[str, Any]], *, label: str
) -> None:
    for episode_id in EPISODES:
        if left[episode_id]["input_files_sha256"] != right[episode_id]["input_files_sha256"]:
            raise ValueError(f"{label} episode {episode_id} is not bound to the same frozen inputs")
        if left[episode_id]["chunks"] != right[episode_id]["chunks"]:
            raise ValueError(f"{label} episode {episode_id} has a different frozen chunk plan")


def _validate_manifest_identity(
    manifest: dict[str, Any], *, iteration: int | None, reference_only: bool | None, label: str
) -> str:
    if manifest.get("base_checkpoint") != BASE_CHECKPOINT:
        raise ValueError(f"{label} has an unexpected base checkpoint")
    checkpoint = str(manifest["selected_checkpoint"])
    if iteration is not None:
        expected_parent = f"iter_{iteration:09d}"
        path = Path(checkpoint)
        if (manifest.get("experiment_id") != "E1-R"
                or manifest.get("checkpoint_iteration") != iteration
                or manifest.get("reference_only") is not reference_only
                or path.name != "model" or path.parent.name != expected_parent
                or "action_fd_umift_edge_e1_refit" not in path.parts):
            raise ValueError(f"{label} has an unexpected E1-R checkpoint identity")
    elif checkpoint != BEST_CHECKPOINT:
        raise ValueError(f"{label} has an unexpected original E1 checkpoint identity")
    return checkpoint


def _validate_prediction(
    path: Path,
    *,
    episode: dict[str, Any],
    method: str,
    checkpoint: str,
    manifest_sha256: str,
    require_two_chunks: bool,
) -> np.ndarray:
    metadata_path = path.with_suffix(".json")
    metadata = _read_json(metadata_path)
    expected = {
        "episode_id": int(episode["episode_id"]),
        "raw_session": str(episode["raw_session"]),
        "method": method,
        "checkpoint_id": checkpoint,
        "manifest_sha256": manifest_sha256,
        "sampling_seed": 0,
        "initial_truth_only": True,
        "first_frame_matches_truth": True,
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"{metadata_path}: unexpected {key}: {metadata.get(key)!r}")
    if Path(metadata.get("prediction_path", "")).resolve() != path.resolve():
        raise ValueError(f"{metadata_path}: prediction_path does not identify {path}")
    if metadata.get("prediction_sha256") != file_sha(path):
        raise ValueError(f"{metadata_path}: prediction SHA256 differs")
    prediction = np.load(path, mmap_mode="r", allow_pickle=False)
    if prediction.dtype != np.float32 or prediction.ndim != 4 or prediction.shape[1:] != (256, 256, 3):
        raise ValueError(f"{path}: unexpected prediction array {prediction.dtype} {prediction.shape}")
    if int(metadata.get("frame_count", -1)) != len(prediction) or len(prediction) < 17:
        raise ValueError(f"{metadata_path}: frame count differs or lacks a complete first window")
    chunks = metadata.get("chunks")
    if not isinstance(chunks, list) or not chunks:
        raise ValueError(f"{metadata_path}: no rollout chunk evidence")
    if require_two_chunks and (len(chunks) != 2 or len(prediction) != 33):
        raise ValueError(f"{metadata_path}: step500 reference must contain exactly two chunks")
    plan = episode["chunks"]
    if not require_two_chunks and (
        metadata.get("complete_episode") is not True
        or len(prediction) != int(episode["frame_count"])
        or len(chunks) != len(plan)
    ):
        raise ValueError(f"{metadata_path}: formal rollout is not a complete episode")
    if len(chunks) > len(plan):
        raise ValueError(f"{metadata_path}: too many rollout chunks")
    for index, record in enumerate(chunks):
        frozen = plan[index]
        for key in ("index", "output_start", "steps", "chunk_id", "noise_seed"):
            if record.get(key) != frozen.get(key):
                raise ValueError(f"{metadata_path}: chunk {index} differs at {key}")
        if index and record.get("condition_sha256") != chunks[index - 1].get("feedback_sha256"):
            raise ValueError(f"{metadata_path}: broken generated-feedback chain at chunk {index}")
    truth = np.load(episode["truth_path"], mmap_mode="r", allow_pickle=False)
    if chunks[0].get("condition_sha256") != array_sha(np.asarray(truth[0])):
        raise ValueError(f"{metadata_path}: first condition is not the observed truth frame")
    return prediction


def _lpips_metric() -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    try:
        import lpips
        import torch
    except ImportError as exc:
        raise RuntimeError("CPU scoring requires the existing torch and lpips environment") from exc
    model = lpips.LPIPS(net="alex", version="0.1").eval().cpu()

    def score(truth: np.ndarray, prediction: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            values = model(torch.from_numpy(truth), torch.from_numpy(prediction), normalize=False)
        return values.detach().cpu().numpy()

    return score


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--old-root", type=Path, default=DEFAULT_OLD_ROOT)
    args = parser.parse_args()
    root, old_root = args.root.resolve(), args.old_root.resolve()
    start0 = root / "evaluation/start0"
    reference500 = root / "evaluation/reference500"
    paths = {
        "new1000": start0 / "prepared/manifest.json",
        "new500": reference500 / "prepared/manifest.json",
        "old1000": old_root / "prepared/manifest.json",
    }
    manifests = {key: _read_json(path) for key, path in paths.items()}
    manifest_shas = {key: file_sha(path) for key, path in paths.items()}
    episode_sets = {key: _episodes(value, label=key) for key, value in manifests.items()}
    for key, episodes in episode_sets.items():
        _validate_input_files(episodes, label=key)
    _same_frozen_inputs(episode_sets["new1000"], episode_sets["new500"], label="step500")
    _same_frozen_inputs(episode_sets["new1000"], episode_sets["old1000"], label="old E1")
    new1000_checkpoint = _validate_manifest_identity(
        manifests["new1000"], iteration=1000, reference_only=False, label="new1000"
    )
    new500_checkpoint = _validate_manifest_identity(
        manifests["new500"], iteration=500, reference_only=True, label="new500"
    )
    old_checkpoint = _validate_manifest_identity(
        manifests["old1000"], iteration=None, reference_only=None, label="old1000"
    )

    sources = {
        "Persistence": None,
        "B0": (start0 / "inference/B0", "B0", BASE_CHECKPOINT, "new1000", False),
        "old_E1-A": (old_root / "inference/E1-A", "E1-A", old_checkpoint, "old1000", False),
        "new500_E1-A": (reference500 / "smoke/E1-A", "E1-A", new500_checkpoint, "new500", True),
        "E1-A": (start0 / "inference/E1-A", "E1-A", new1000_checkpoint, "new1000", False),
        "E1-Z": (start0 / "inference/E1-Z", "E1-Z", new1000_checkpoint, "new1000", False),
        "E1-S": (start0 / "inference/E1-S", "E1-S", new1000_checkpoint, "new1000", False),
    }
    lpips_metric = _lpips_metric()
    episode_results: list[dict[str, Any]] = []
    aggregate_rows: dict[str, list[dict[str, Any]]] = {key: [] for key in sources}
    for episode_id in EPISODES:
        episode = episode_sets["new1000"][episode_id]
        truth_u8 = np.load(episode["truth_path"], mmap_mode="r", allow_pickle=False)
        if truth_u8.dtype != np.uint8 or truth_u8.ndim != 4 or truth_u8.shape[1:] != (256, 256, 3):
            raise ValueError(f"episode {episode_id}: unexpected truth array")
        truth = np.asarray(truth_u8[:17], dtype=np.float32) / 255.0
        methods: dict[str, Any] = {}
        for result_key, source in sources.items():
            if source is None:
                prediction = persistence_prediction(truth)
            else:
                directory, method, checkpoint, manifest_key, require_two_chunks = source
                prediction_all = _validate_prediction(
                    directory / f"episode_{episode_id}.npy",
                    episode=episode_sets[manifest_key][episode_id],
                    method=method,
                    checkpoint=checkpoint,
                    manifest_sha256=manifest_shas[manifest_key],
                    require_two_chunks=require_two_chunks,
                )
                if not np.array_equal(np.asarray(prediction_all[0]), truth[0]):
                    raise ValueError(f"{result_key} episode {episode_id}: first frame differs from truth")
                prediction = np.asarray(prediction_all[:17])
            metrics = evaluate_video_pair(
                truth, prediction, include_lpips=True, lpips_metric=lpips_metric
            )
            methods[result_key] = metrics
            aggregate_rows[result_key].append({
                "raw_session": episode["raw_session"], "metrics": metrics["mean"]
            })
        episode_results.append({
            "episode_id": episode_id,
            "raw_session": episode["raw_session"],
            "methods": methods,
        })

    aggregates = {
        method: aggregate_session_equal(rows) for method, rows in aggregate_rows.items()
    }
    report = {
        "protocol": "e1-refit-three-session-first-window-reference-v1",
        "sample_count": 3,
        "seed0": True,
        "sampling_seed": 0,
        "firstwindow_only": True,
        "fixed1000_no_selection": True,
        "reference500_quantitative_scope": "first chunk only; second chunk checked only as feedback-chain evidence",
        "comparison_scope": "standalone three-session reference; not comparable to the old 196-window aggregate",
        "psnr_exact_match_json_encoding": "Infinity (Python json allow_nan=True)",
        "manifest_sha256": manifest_shas,
        "checkpoint_identities": {
            "base": BASE_CHECKPOINT,
            "old_E1": old_checkpoint,
            "new_step500_reference_only": new500_checkpoint,
            "new_fixed_step1000": new1000_checkpoint,
        },
        "episode_results": episode_results,
        "session_equal_aggregate": aggregates,
    }
    output = root / "evaluation/reference_metrics.json"
    if output.exists():
        raise FileExistsError(f"refusing to overwrite reference metrics: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=True) + "\n")
    summary = {
        method: {name: values["overall"][name] for name in ("psnr", "ssim", "lpips")}
        for method, values in aggregates.items()
    }
    print(json.dumps({"output": str(output), "session_equal_overall": summary}, allow_nan=True))


if __name__ == "__main__":
    main()
