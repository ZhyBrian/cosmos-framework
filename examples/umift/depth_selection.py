"""Frozen B-continuation/D1 candidate evaluation with the E3 RGBD metrics."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from examples.umift import rgbd_selection as core
from examples.umift.long_rollout import EPISODES, file_sha
from examples.umift.protocol import derive_noise_seed
from examples.umift.rgbd_metrics import joint_selection_score


ITERATIONS = (250, 500, 750, 1000)
ARMS = ("b_continue", "d1")
PROTOCOL = "e3-depth-aux-selection-v1"
EXPERIMENT_ID = "E3-Depth-Aux"
expected_windows = core.expected_windows


def action_source_indices(start: int) -> np.ndarray:
    """Return the 17 source poses used by the dataset's forward window."""
    return int(start) + 2 * np.arange(17, dtype=np.int64)


def expected_model_action_sha256(physical_action: np.ndarray, max_action_dim: int = 64) -> str:
    """Derive the model-space action digest from the canonical physical action."""
    physical = np.asarray(physical_action, dtype=np.float32)
    if physical.shape != (16, 10):
        raise ValueError("physical action must have shape [16,10]")
    stats_path = (Path(__file__).parents[2] / "cosmos_framework/data/generator/action/normalizer_stats/umi_lerobot_stats.json")
    stats = json.loads(stats_path.read_text())
    q01 = np.asarray(stats["q01"][:10], dtype=np.float32)
    q99 = np.asarray(stats["q99"][:10], dtype=np.float32)
    scale = np.maximum((q99 - q01) / np.float32(2), np.float32(1e-8))
    normalized = (physical - (q99 + q01) / np.float32(2)) / scale
    if max_action_dim < 10:
        raise ValueError("max_action_dim cannot be smaller than 10")
    model = np.pad(normalized, ((0, 0), (0, max_action_dim - 10))).astype(np.float32)
    return hashlib.sha256(np.ascontiguousarray(model).tobytes()).hexdigest()


def validate_window_action_identity(window: dict, sample: dict) -> None:
    from examples.umift.rgbd_rollout import array_sha

    if array_sha(sample["physical_action"].cpu().numpy()) != window.get("physical_action_sha256"):
        raise ValueError("selection physical action differs from frozen source action")
    if array_sha(sample["action"].cpu().numpy()) != window.get("model_action_sha256"):
        raise ValueError("selection model action differs from independently frozen padding")


def _identity(arm: str) -> dict[str, str]:
    if arm not in ARMS:
        raise ValueError(f"unknown depth-selection arm: {arm}")
    return {"experiment_id": EXPERIMENT_ID, "arm": arm}


def validate_candidate_identity(checkpoint: Path, iteration: int, arm: str) -> None:
    _identity(arm)
    job = f"action_fd_umift_edge_rgbd_{arm}"
    parts = checkpoint.parts
    try:
        root_index = parts.index("e3_depth_aux_20260914")
    except ValueError as exc:
        raise ValueError("checkpoint is outside the frozen depth-aux run") from exc
    if (
        checkpoint.name != "model"
        or checkpoint.parent.name != f"iter_{iteration:09d}"
        or parts[root_index + 1 : root_index + 2] != (arm,)
        or job not in parts
    ):
        raise ValueError(f"checkpoint is not the requested {arm} iteration")


def freeze(zarr_path: Path, output: Path, rgb_weight: float, arm: str) -> None:
    import zarr

    identity = _identity(arm)
    if output.exists():
        raise FileExistsError(output)
    joint_selection_score(1, 1, 1, 1, rgb_weight=rgb_weight)
    root = zarr.open_group(str(zarr_path), mode="r")
    windows = []
    for episode_id in EPISODES:
        group = root["data"][f"episode_{episode_id}"]
        last = int(group["rgb_0"].shape[0]) - 33
        starts = np.rint(np.linspace(0, last, 8)).astype(int)
        if len(set(starts.tolist())) != 8 or last < 0:
            raise ValueError("insufficient legal windows")
        for start in starts:
            window_id = f"episode_{episode_id}:s={start}"
            from cosmos_framework.data.generator.action.datasets.umift_zarr_dataset import _framewise_actions

            indices = action_source_indices(int(start))
            physical_action = _framewise_actions(
                np.asarray(group["ts_pose_fb_0"].oindex[indices], dtype=np.float64)
            )
            windows.append(
                dict(
                    episode_id=episode_id,
                    start=int(start),
                    raw_session=str(group.attrs["src"]).split("#")[0],
                    window_id=window_id,
                    noise_seed=derive_noise_seed(window_id, 0),
                    physical_action_sha256=hashlib.sha256(
                        np.ascontiguousarray(physical_action).tobytes()
                    ).hexdigest(),
                    model_action_sha256=expected_model_action_sha256(physical_action),
                )
            )
    if [{k: v for k, v in window.items() if not k.endswith("action_sha256")} for window in windows] != expected_windows():
        raise ValueError("source differs from the audited fixed 24 windows")
    report = dict(
        protocol=PROTOCOL,
        **identity,
        history_frames=5,
        iterations=list(ITERATIONS),
        rgb_weight=rgb_weight,
        selection_metric="weighted_ratio_of_session_equal_lpips_and_depth_mae_to_persistence",
        aggregation_order="mean future frames, mean windows per session, mean sessions, then ratio to P",
        depth_mask="finite(GT) & 0<GT<0.5; never predicted-mask intersection",
        depth_units="metres",
        prediction_clamp_for_depth_metrics=False,
        tie_break="earlier_iteration",
        num_steps=30,
        selection_uses_test_episodes=True,
        zarr_path=str(zarr_path),
        windows=windows,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")


def load_protocol(path: Path, arm: str) -> tuple[dict, str]:
    protocol = json.loads(path.read_text())
    identity = _identity(arm)
    if (
        protocol.get("protocol") != PROTOCOL
        or any(protocol.get(key) != value for key, value in identity.items())
        or protocol.get("history_frames") != 5
        or protocol.get("selection_uses_test_episodes") is not True
        or protocol.get("iterations") != list(ITERATIONS)
        or [{k: v for k, v in window.items() if not k.endswith("action_sha256")}
            for window in protocol.get("windows", [])] != expected_windows()
        or any(
            not isinstance(window.get(field), str) or len(window[field]) != 64
            for window in protocol.get("windows", [])
            for field in ("physical_action_sha256", "model_action_sha256")
        )
        or protocol.get("prediction_clamp_for_depth_metrics") is not False
        or protocol.get("num_steps") != 30
        or protocol.get("depth_units") != "metres"
        or protocol.get("selection_metric")
        != "weighted_ratio_of_session_equal_lpips_and_depth_mae_to_persistence"
        or protocol.get("aggregation_order")
        != "mean future frames, mean windows per session, mean sessions, then ratio to P"
        or protocol.get("depth_mask")
        != "finite(GT) & 0<GT<0.5; never predicted-mask intersection"
        or protocol.get("tie_break") != "earlier_iteration"
    ):
        raise ValueError(f"invalid frozen depth-selection protocol for {arm}")
    joint_selection_score(1, 1, 1, 1, rgb_weight=protocol["rgb_weight"])
    return protocol, file_sha(path)


def choose_candidate(
    reports: list[dict], protocol_sha: str, rgb_weight: float, arm: str
) -> dict:
    identity = _identity(arm)
    return core.choose_candidate(
        reports,
        protocol_sha,
        rgb_weight,
        iterations=ITERATIONS,
        candidate_validator=lambda checkpoint, iteration: validate_candidate_identity(
            checkpoint, iteration, arm
        ),
        identity=identity,
    )


def infer(args) -> None:
    identity = _identity(args.arm)
    core.infer(
        args,
        protocol_loader=lambda path: load_protocol(path, args.arm),
        iterations=ITERATIONS,
        candidate_validator=lambda checkpoint, iteration: validate_candidate_identity(
            checkpoint, iteration, args.arm
        ),
        identity=identity,
        window_action_validator=validate_window_action_identity,
        expected_job_name=f"action_fd_umift_edge_rgbd_{args.arm}",
        allowed_visible_devices=("0,1,2,3", "4,5,6,7"),
    )


def score(args) -> None:
    identity = _identity(args.arm)
    core.score(
        args,
        protocol_loader=lambda path: load_protocol(path, args.arm),
        candidate_validator=lambda checkpoint, iteration: validate_candidate_identity(
            checkpoint, iteration, args.arm
        ),
        identity=identity,
    )


def select(args) -> None:
    protocol, protocol_sha = load_protocol(args.protocol, args.arm)
    paths = [args.input / f"iter_{step:09d}" / "metrics.json" for step in ITERATIONS]
    reports = [json.loads(path.read_text()) for path in paths]
    best = choose_candidate(reports, protocol_sha, protocol["rgb_weight"], args.arm)
    result = dict(
        **_identity(args.arm),
        history_frames=5,
        iteration=best["iteration"],
        checkpoint=best["checkpoint"],
        protocol_file=str(args.protocol.resolve()),
        protocol_sha256=protocol_sha,
        rgb_weight=protocol["rgb_weight"],
        selection_uses_test_episodes=True,
        joint_score=best["joint_score"],
        candidates=[
            dict(
                iteration=report["iteration"],
                metrics=report["session_equal_aggregate"]["overall"],
                joint_score=report["joint_score"],
                file=str(path),
                sha256=file_sha(path),
            )
            for report, path in zip(reports, paths, strict=True)
        ],
    )
    output = args.input / "selected.json"
    if output.exists():
        raise FileExistsError(output)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    freeze_parser = subparsers.add_parser("freeze")
    freeze_parser.add_argument("--zarr", type=Path, required=True)
    freeze_parser.add_argument("--output", type=Path, required=True)
    freeze_parser.add_argument("--rgb-weight", type=float, required=True)
    freeze_parser.add_argument("--arm", choices=ARMS, required=True)
    infer_parser = subparsers.add_parser("infer")
    for name in ("protocol", "checkpoint", "sft-toml", "output"):
        infer_parser.add_argument(f"--{name}", type=Path, required=True)
    infer_parser.add_argument("--iteration", type=int, required=True)
    infer_parser.add_argument("--arm", choices=ARMS, required=True)
    for command in ("score", "select"):
        command_parser = subparsers.add_parser(command)
        command_parser.add_argument("--protocol", type=Path, required=True)
        command_parser.add_argument("--input", type=Path, required=True)
        command_parser.add_argument("--arm", choices=ARMS, required=True)
    args = parser.parse_args()
    if args.command == "freeze":
        freeze(args.zarr, args.output, args.rgb_weight, args.arm)
    elif args.command == "infer":
        infer(args)
    elif args.command == "score":
        score(args)
    else:
        select(args)


if __name__ == "__main__":
    main()
