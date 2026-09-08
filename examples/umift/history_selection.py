"""Freeze and evaluate the common E2-H held-out checkpoint-selection windows."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np

from examples.umift.long_rollout import EPISODES, array_sha, file_sha
from examples.umift.protocol import derive_noise_seed


def validate_candidate_identity(checkpoint: Path, history_frames: int, iteration: int) -> None:
    if (checkpoint.name != "model" or checkpoint.parent.name != f"iter_{iteration:09d}"
            or f"action_fd_umift_edge_h{history_frames}" not in checkpoint.parts):
        raise ValueError("checkpoint does not belong to the requested H/iteration")


def freeze(zarr_path: Path, output: Path) -> None:
    import zarr

    if output.exists():
        raise FileExistsError(output)
    root = zarr.open_group(str(zarr_path), mode="r")
    windows = []
    for episode_id in EPISODES:
        group = root["data"][f"episode_{episode_id}"]
        last_start = int(group["rgb_0"].shape[0]) - 33
        starts = np.rint(np.linspace(0, last_start, 8)).astype(int)
        if len(set(starts.tolist())) != 8 or last_start < 0:
            raise ValueError("episode cannot supply eight distinct complete windows")
        for start in starts:
            window_id = f"episode_{episode_id}:s={start}"
            windows.append({"episode_id": episode_id, "start": int(start),
                            "raw_session": str(group.attrs["src"]).split("#")[0],
                            "window_id": window_id, "noise_seed": derive_noise_seed(window_id, 0)})
    report = {"protocol": "e2-history-selection-v1", "histories": [1, 5, 9, 17],
              "iterations": [500, 1000, 1500, 2000, 2500, 3000],
              "selection_metric": "session_equal_future16_lpips_alex_v0.1",
              "tie_break": "earlier_iteration", "num_steps": 30, "sampling_seed": 0,
              "selection_uses_test_episodes": True, "zarr_path": str(zarr_path), "windows": windows}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")


def infer(args: argparse.Namespace) -> None:
    import torch

    from cosmos_framework.data.generator.action.datasets.umift_history_dataset import get_umift_history_sft_dataset
    from cosmos_framework.inference.common.init import init_script
    from examples.umift.history_infer import build_history_batch, load_history_model, run_history_prediction
    from examples.umift.infer import _move_batch_to_cuda, validate_independent_parallelism, validate_launch_environment

    validate_launch_environment(os.environ)
    init_script()
    protocol_sha = file_sha(args.protocol)
    protocol = json.loads(args.protocol.read_text())
    if (protocol.get("protocol") != "e2-history-selection-v1" or args.history_frames not in protocol["histories"]
            or protocol["selection_uses_test_episodes"] is not True):
        raise ValueError("unexpected E2-H selection protocol")
    if args.iteration not in protocol["iterations"]:
        raise ValueError("iteration is not a preregistered candidate")
    validate_candidate_identity(args.checkpoint, args.history_frames, args.iteration)
    model, resolved, evidence = load_history_model(args.sft_toml, args.checkpoint, args.history_frames)
    validate_independent_parallelism(model.parallel_dims)
    dataset = get_umift_history_sft_dataset(
        protocol["zarr_path"], split="history", history_frames=args.history_frames,
        tokenizer_config=resolved.model.config.vlm_config.tokenizer,
        max_action_dim=int(resolved.model.config.max_action_dim),
    )
    rank = torch.distributed.get_rank()
    if rank == 0:
        args.output.mkdir(parents=True, exist_ok=False)
    torch.distributed.barrier()
    rows = []
    with torch.inference_mode():
        for index, window in enumerate(protocol["windows"]):
            if index % 4 != rank:
                continue
            sample = dataset.get_window(window["episode_id"], window["start"])
            truth = sample["video"].permute(1, 2, 3, 0).numpy()[args.history_frames - 1:].astype(np.float32) / 255
            batch = _move_batch_to_cuda(build_history_batch(sample))
            full_prediction = run_history_prediction(
                model, batch, history_frames=args.history_frames,
                noise_seed=window["noise_seed"], num_steps=protocol["num_steps"],
            )
            # Old evaluator scores positions 1..16, anchored at the current real observation.
            prediction = np.concatenate((truth[:1], full_prediction[args.history_frames:]), axis=0)
            path = args.output / f"window_{index:02d}.npz"
            np.savez(path, truth=truth, prediction=prediction)
            rows.append({**window, "window_index": index, "file": path.name, "sha256": file_sha(path),
                         "physical_action_sha256": array_sha(sample["physical_action"].numpy()),
                         "history_source_indices": sample["history_source_indices"].tolist(),
                         "history_padding_count": sample["history_padding_count"]})
            print(json.dumps({"window_done": index, "history_frames": args.history_frames}), flush=True)
    if file_sha(args.protocol) != protocol_sha:
        raise ValueError("selection protocol changed during inference")
    (args.output / f"rank_{rank}.json").write_text(json.dumps({
        "history_frames": args.history_frames, "iteration": args.iteration,
        "protocol_sha256": protocol_sha, "load_evidence": evidence,
        "rank": rank, "rows": rows, "complete": True,
    }, indent=2) + "\n")
    torch.distributed.barrier()
    torch.distributed.destroy_process_group()


def score(args: argparse.Namespace) -> None:
    from examples.umift.evaluate import aggregate_session_equal, evaluate_video_pair
    from examples.umift.score_refit_reference import _lpips_metric

    protocol_sha = file_sha(args.protocol)
    protocol = json.loads(args.protocol.read_text())
    reports = [json.loads((args.input / f"rank_{rank}.json").read_text()) for rank in range(4)]
    identity = {(r["history_frames"], r["iteration"], r["load_evidence"]["checkpoint"]) for r in reports}
    if len(identity) != 1 or any(r["protocol_sha256"] != protocol_sha or not r["complete"] for r in reports):
        raise ValueError("rank reports disagree about candidate/protocol/completion")
    rows = sorted([row for r in reports for row in r["rows"]], key=lambda r: r["window_index"])
    if [r["window_index"] for r in rows] != list(range(len(protocol["windows"]))):
        raise ValueError("selection result does not contain each frozen window exactly once")
    metric = _lpips_metric()
    results = []
    for row, window in zip(rows, protocol["windows"], strict=True):
        if any(row[k] != v for k, v in window.items()):
            raise ValueError("selection window differs from frozen protocol")
        path = args.input / row["file"]
        if file_sha(path) != row["sha256"]:
            raise ValueError("selection prediction hash differs")
        with np.load(path, allow_pickle=False) as data:
            metrics = evaluate_video_pair(data["truth"], data["prediction"], include_lpips=True, lpips_metric=metric)
        results.append({**row, "metrics": metrics})
    aggregate = aggregate_session_equal([{"raw_session": r["raw_session"], "metrics": {
        **r["metrics"]["mean"], "temporal_l1": r["metrics"]["temporal"]["mean_l1"]}} for r in results])
    history_frames, iteration, checkpoint = next(iter(identity))
    validate_candidate_identity(Path(checkpoint), history_frames, iteration)
    report = {"history_frames": history_frames, "iteration": iteration, "checkpoint": checkpoint,
              "protocol_sha256": protocol_sha, "selection_uses_test_episodes": True,
              "session_equal_aggregate": aggregate, "windows": results}
    output = args.input / "metrics.json"
    if output.exists():
        raise FileExistsError(output)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"metrics": str(output), "overall": aggregate["overall"]}))


def choose_candidate(reports: list[dict], protocol_sha: str, history_frames: int) -> dict:
    if sorted(r["iteration"] for r in reports) != [500, 1000, 1500, 2000, 2500, 3000]:
        raise ValueError("selection requires all six distinct preregistered candidates")
    for report in reports:
        validate_candidate_identity(Path(report["checkpoint"]), history_frames, report["iteration"])
        if report["protocol_sha256"] != protocol_sha or report["history_frames"] != history_frames:
            raise ValueError("candidate H/protocol differs")
        if not math.isfinite(report["session_equal_aggregate"]["overall"]["lpips"]):
            raise ValueError("non-finite selection LPIPS")
    return min(reports, key=lambda r: (r["session_equal_aggregate"]["overall"]["lpips"], r["iteration"]))


def choose(args: argparse.Namespace) -> None:
    protocol_sha = file_sha(args.protocol)
    paths = [args.input / f"iter_{iteration:09d}" / "metrics.json" for iteration in
             (500, 1000, 1500, 2000, 2500, 3000)]
    reports = [json.loads(path.read_text()) for path in paths]
    selected = choose_candidate(reports, protocol_sha, args.history_frames)
    output = args.input / "selected.json"
    if output.exists():
        raise FileExistsError(output)
    result = {"history_frames": args.history_frames, "iteration": selected["iteration"],
              "checkpoint": selected["checkpoint"], "protocol_sha256": protocol_sha,
              "protocol_file": str(args.protocol.resolve()),
              "selection_uses_test_episodes": True,
              "candidates": [{"iteration": r["iteration"], "metrics": r["session_equal_aggregate"]["overall"],
                               "file": str(p), "sha256": file_sha(p)} for p, r in zip(paths, reports, strict=True)]}
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("freeze")
    prep.add_argument("--zarr", type=Path, required=True)
    prep.add_argument("--output", type=Path, required=True)
    predict = commands.add_parser("infer")
    predict.add_argument("--protocol", type=Path, required=True)
    predict.add_argument("--history-frames", type=int, choices=(1, 5, 9, 17), required=True)
    predict.add_argument("--iteration", type=int, required=True)
    predict.add_argument("--checkpoint", type=Path, required=True)
    predict.add_argument("--sft-toml", type=Path, required=True)
    predict.add_argument("--output", type=Path, required=True)
    evaluate = commands.add_parser("score")
    evaluate.add_argument("--protocol", type=Path, required=True)
    evaluate.add_argument("--input", type=Path, required=True)
    selection = commands.add_parser("choose")
    selection.add_argument("--protocol", type=Path, required=True)
    selection.add_argument("--input", type=Path, required=True)
    selection.add_argument("--history-frames", type=int, choices=(1, 5, 9, 17), required=True)
    args = parser.parse_args()
    if args.command == "freeze":
        freeze(args.zarr, args.output)
    elif args.command == "infer":
        infer(args)
    elif args.command == "score":
        score(args)
    else:
        choose(args)


if __name__ == "__main__":
    main()
