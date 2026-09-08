"""Bind the immutable E1 suffix fixtures to an E2-H selected checkpoint."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np

from examples.umift.long_rollout import BASE_CHECKPOINT, EPISODES, file_sha


PARENTS = {
    0: ("/data/cosmos_runs/e1_long_rollout_20260907", "8f311b93c18e20d8ab752e5f86e33e0691296e10bcbdb565511ddd823ba151d9"),
    33: ("/data/cosmos_runs/e1_midstart_rollout_20260907/start33", "9c18fc449ed79a7230b047c0cff85f74837748409fe316c0301cb51f4dc96949"),
    67: ("/data/cosmos_runs/e1_midstart_rollout_20260907/start67", "aa3259970c9afd3e151972f439fff92e324dd7355957c3005b40bfbadea48645"),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--start-percent", type=int, choices=(0, 33, 67), required=True)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    selection = json.loads(args.selection.read_text())
    history_frames = selection["history_frames"]
    iteration = selection["iteration"]
    checkpoint = Path(selection["checkpoint"])
    if history_frames not in (1, 5, 9, 17) or iteration not in (500, 1000, 1500, 2000, 2500, 3000):
        raise ValueError("unexpected selected H/iteration")
    if (checkpoint.name != "model" or checkpoint.parent.name != f"iter_{iteration:09d}"
            or f"action_fd_umift_edge_h{history_frames}" not in checkpoint.parts):
        raise ValueError("selected checkpoint identity does not match this history experiment")
    if not (checkpoint / ".metadata").is_file():
        raise FileNotFoundError(checkpoint / ".metadata")
    source_root, parent_sha = PARENTS[args.start_percent]
    source = Path(source_root) / "prepared/manifest.json"
    if file_sha(source) != parent_sha:
        raise ValueError("immutable parent fixture manifest has changed")
    parent = json.loads(source.read_text())
    if parent["base_checkpoint"] != BASE_CHECKPOINT or tuple(e["episode_id"] for e in parent["episodes"]) != EPISODES:
        raise ValueError("parent base checkpoint or held-out episodes differ")
    result = copy.deepcopy(parent)
    result.pop("checkpoint_selection_sha256", None)
    result.update(experiment_id="E2-H", protocol="e2-history-open-loop-v1",
                  history_frames=history_frames, selected_iteration=iteration,
                  selected_checkpoint=str(checkpoint), start_percent=args.start_percent,
                  checkpoint_metadata_sha256=file_sha(checkpoint / ".metadata"),
                  selection_file=str(args.selection.resolve()), selection_sha256=file_sha(args.selection),
                  selection_uses_test_episodes=True, initial_history_only=True,
                  e1_fixture_manifest=str(source), e1_fixture_manifest_sha256=parent_sha)
    result.pop("initial_truth_only", None)
    for episode in result["episodes"]:
        for filename, expected in episode["input_files_sha256"].items():
            if file_sha(Path(filename)) != expected:
                raise ValueError(f"immutable input changed: {filename}")
        with np.load(episode["frame_indices_path"], allow_pickle=False) as archive:
            anchor = int(archive["source_indices"][0])
        episode.update(experiment_id="E2-H", history_frames=history_frames, selected_iteration=iteration)
        if args.start_percent == 0:
            episode.update(start_percent=0, initial_selected_frame=0, initial_source_frame=anchor,
                           initial_episode_elapsed_seconds=0.0, parent_frame_count=episode["frame_count"])
        elif episode["start_percent"] != args.start_percent or episode["initial_source_frame"] != anchor:
            raise ValueError("suffix anchor differs from source indices")
        episode["history_padding_count"] = int(sum(anchor - 2 * k < 0 for k in range(history_frames)))
    destination = args.root.resolve() / "prepared"
    destination.mkdir(parents=True, exist_ok=False)
    output = destination / "manifest.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"manifest": str(output), "sha256": file_sha(output), "H": history_frames,
                      "iteration": iteration, "start_percent": args.start_percent}))


if __name__ == "__main__":
    main()
