"""B-continuation/D1 adapters for the frozen E3 RGBD long-rollout pipeline."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from examples.umift import depth_selection
from examples.umift import prepare_rgbd_rollout as prepare_core
from examples.umift import render_rgbd_comparison as render_core
from examples.umift import rgbd_rollout as rollout_core
from examples.umift import score_rgbd_rollout as score_core


ARMS = ("b_continue", "d1")
EXPERIMENT_ID = "E3-Depth-Aux"
ROLLOUT_PROTOCOL = "e3-depth-aux-rgbd-open-loop-v1"
SCORING_PROTOCOL = "e3-depth-aux-rgbd-full-suffix-scoring-v1"
expected_model_action_sha256 = depth_selection.expected_model_action_sha256


def validate_depth_action_identity(method: str, frozen: dict, actual: dict) -> None:
    action_key = "A" if method == "B0" else method[-1]
    if actual.get("action_source", action_key) != action_key:
        raise ValueError("depth rollout action source differs from method")
    if actual.get("physical_action_sha256") != frozen.get(f"{action_key}_physical_action_sha256"):
        raise ValueError("depth rollout physical action differs from frozen fixture")
    if actual.get("padded_model_action_sha256") != frozen.get(f"{action_key}_model_action_sha256"):
        raise ValueError("depth rollout model action differs from independently frozen padding")


def _arm_label(arm: str) -> str:
    if arm == "b_continue":
        return "B"
    if arm == "d1":
        return "D1"
    raise ValueError(f"unknown depth rollout arm: {arm}")


def model_methods(arm: str) -> tuple[str, ...]:
    label = _arm_label(arm)
    return ("B0", f"{label}-A", f"{label}-Z", f"{label}-S")


def bind_manifest(parent, selection, *, arm: str, **kwargs):
    _arm_label(arm)
    return prepare_core.bind_manifest(
        parent,
        selection,
        experiment_id=EXPERIMENT_ID,
        protocol=ROLLOUT_PROTOCOL,
        arm=arm,
        **kwargs,
    )


def _validate_selection(path: Path, arm: str):
    selection_sha = rollout_core.file_sha(path)
    selection = json.loads(path.read_text())
    if (
        selection.get("experiment_id") != EXPERIMENT_ID
        or selection.get("arm") != arm
        or selection.get("history_frames") != 5
        or selection.get("selection_uses_test_episodes") is not True
        or not isinstance(selection.get("candidates"), list)
        or len(selection["candidates"]) != 4
    ):
        raise ValueError(f"selection is not the frozen {arm} depth result")
    protocol_path = Path(selection["protocol_file"])
    protocol, protocol_sha = depth_selection.load_protocol(protocol_path, arm)
    if protocol_sha != selection["protocol_sha256"]:
        raise ValueError("depth selection protocol changed")
    reports = []
    for candidate in selection["candidates"]:
        metrics_path = Path(candidate["file"])
        if rollout_core.file_sha(metrics_path) != candidate["sha256"]:
            raise ValueError("depth candidate metrics changed after selection")
        report = json.loads(metrics_path.read_text())
        if candidate.get("iteration") != report.get("iteration"):
            raise ValueError("candidate iteration differs from its metrics report")
        if (
            not np.isclose(
                candidate.get("joint_score"), report.get("joint_score"), atol=1e-12, rtol=0
            )
            or candidate.get("metrics")
            != report.get("session_equal_aggregate", {}).get("overall")
        ):
            raise ValueError("candidate summary differs from its metrics report")
        reports.append(report)
    best = depth_selection.choose_candidate(
        reports, selection["protocol_sha256"], float(selection["rgb_weight"]), arm
    )
    if (
        int(selection["iteration"]) != int(best["iteration"])
        or str(selection["checkpoint"]) != str(best["checkpoint"])
        or not np.isclose(
            float(selection["joint_score"]), float(best["joint_score"]), atol=1e-12, rtol=0
        )
    ):
        raise ValueError("selected depth checkpoint is not the frozen joint-score minimum")
    checkpoint = Path(selection["checkpoint"])
    if not (checkpoint / ".metadata").is_file():
        raise FileNotFoundError(checkpoint / ".metadata")
    return selection, selection_sha, protocol


def prepare(args: argparse.Namespace) -> Path:
    path = prepare_core.prepare(
        args,
        selection_validator=lambda path: _validate_selection(path, args.arm),
        experiment_id=EXPERIMENT_ID,
        protocol=ROLLOUT_PROTOCOL,
        arm=args.arm,
    )
    manifest = json.loads(path.read_text())
    for episode in manifest["episodes"]:
        with np.load(episode["actions_path"], allow_pickle=False) as archive:
            actions = {key: np.asarray(archive[key], dtype=np.float32) for key in ("A", "Z", "S")}
        for chunk in episode["chunks"]:
            index = int(chunk["index"])
            for key, values in actions.items():
                chunk[f"{key}_model_action_sha256"] = expected_model_action_sha256(values[index])
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    return path


def infer(args: argparse.Namespace) -> None:
    rollout_core.infer(
        args,
        experiment_id=EXPERIMENT_ID,
        protocol=ROLLOUT_PROTOCOL,
        arm=args.arm,
        model_methods=model_methods(args.arm),
        expected_job_name=f"action_fd_umift_edge_rgbd_{args.arm}",
        allowed_visible_devices=("0,1,2,3", "4,5,6,7"),
    )


def score(args: argparse.Namespace) -> dict:
    return score_core.score_root(
        args.root,
        args.output,
        model_methods=model_methods(args.arm),
        experiment_id=EXPERIMENT_ID,
        rollout_protocol=ROLLOUT_PROTOCOL,
        scoring_protocol=SCORING_PROTOCOL,
        arm=args.arm,
    )


def render(args: argparse.Namespace) -> dict:
    label = _arm_label(args.arm)
    return render_core.render(
        args.root,
        args.output,
        args.font,
        model_methods=model_methods(args.arm),
        experiment_id=EXPERIMENT_ID,
        rollout_protocol=ROLLOUT_PROTOCOL,
        scoring_protocol=SCORING_PROTOCOL,
        arm=args.arm,
        title_text=f"{label} continuation",
        headings=(
            "GT",
            "P 首帧保持",
            "B0 基础 Edge",
            f"{label}-A 正确动作",
            f"{label}-Z 静止动作",
            f"{label}-S 错配动作",
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--arm", choices=ARMS, required=True)
    prepare_parser.add_argument("--selection", type=Path, required=True)
    prepare_parser.add_argument("--start-percent", type=int, choices=(0, 33, 67), required=True)
    prepare_parser.add_argument("--root", type=Path, required=True)
    prepare_parser.add_argument("--source-manifest", type=Path)
    infer_parser = subparsers.add_parser("infer")
    infer_parser.add_argument("--arm", choices=ARMS, required=True)
    infer_parser.add_argument("--root", type=Path, required=True)
    infer_parser.add_argument("--sft-toml", type=Path, required=True)
    infer_parser.add_argument("--phase", choices=("base", "finetuned"), required=True)
    infer_parser.add_argument("--max-chunks", type=int, default=0)
    score_parser = subparsers.add_parser("score")
    score_parser.add_argument("--arm", choices=ARMS, required=True)
    score_parser.add_argument("--root", type=Path, required=True)
    score_parser.add_argument("--output", type=Path)
    render_parser = subparsers.add_parser("render")
    render_parser.add_argument("--arm", choices=ARMS, required=True)
    render_parser.add_argument("--root", type=Path, required=True)
    render_parser.add_argument("--output", type=Path, required=True)
    render_parser.add_argument("--font", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        path = prepare(args)
        print(json.dumps({"manifest": str(path), "sha256": rollout_core.file_sha(path)}))
    elif args.command == "infer":
        if args.max_chunks < 0:
            parser.error("max-chunks must be nonnegative")
        infer(args)
    elif args.command == "score":
        report = score(args)
        print(json.dumps({"metrics": report["metrics_path"], "metrics_sha256": report["metrics_sha256"]}))
    else:
        render(args)


if __name__ == "__main__":
    main()
