"""Bind immutable E1 video/action fixtures to E1-R weights without regenerating fixtures."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from examples.umift.long_rollout import BASE_CHECKPOINT, BEST_CHECKPOINT, EPISODES, file_sha


def read_frozen_fixture(path: Path, expected_sha256: str) -> dict:
    if file_sha(path) != expected_sha256:
        raise ValueError("parent manifest differs from the previously audited fixture SHA256")
    return json.loads(path.read_text())


def bind_checkpoint(parent: dict, checkpoint: Path, iteration: int) -> dict:
    if iteration not in (500, 1000):
        raise ValueError("only reference step500 and preregistered final step1000 are used")
    if checkpoint.name != "model" or checkpoint.parent.name != f"iter_{iteration:09d}":
        raise ValueError("checkpoint path does not identify the requested iteration/model")
    if "action_fd_umift_edge_e1_refit" not in checkpoint.parts:
        raise ValueError("checkpoint must belong to the separate E1-R training job")
    if parent.get("selected_checkpoint") != BEST_CHECKPOINT or parent.get("base_checkpoint") != BASE_CHECKPOINT:
        raise ValueError("expected original E1 frozen fixtures and base identity")
    if tuple(e["episode_id"] for e in parent["episodes"]) != EPISODES:
        raise ValueError("only history episodes 13,43,49 are evaluated")
    result = copy.deepcopy(parent)
    result.pop("checkpoint_selection_sha256", None)
    result["selected_checkpoint"] = str(checkpoint)
    result["experiment_id"] = "E1-R"
    result["checkpoint_iteration"] = iteration
    result["reference_only"] = iteration == 500
    result["checkpoint_selection_rule"] = "fixed1000; step500 reference only; no test-driven selection or tuning"
    for episode in result["episodes"]:
        episode["experiment_id"] = "E1-R"
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--checkpoint-model", type=Path, required=True)
    parser.add_argument("--expected-fixture-sha256", required=True,
                        help="Previously audited parent manifest hash (including chunk plan and noise seeds)")
    parser.add_argument("--iteration", type=int, choices=(500, 1000), default=1000)
    args = parser.parse_args()
    source = args.source_root.resolve() / "prepared/manifest.json"
    checkpoint = args.checkpoint_model.resolve()
    if not (checkpoint / ".metadata").is_file():
        raise FileNotFoundError(checkpoint / ".metadata")
    parent = read_frozen_fixture(source, args.expected_fixture_sha256)
    manifest = bind_checkpoint(parent, checkpoint, args.iteration)
    for episode in manifest["episodes"]:
        for filename, expected in episode["input_files_sha256"].items():
            if file_sha(Path(filename)) != expected:
                raise ValueError(f"frozen input changed: {filename}")
    manifest["e1_fixture_manifest"] = str(source)
    manifest["e1_fixture_manifest_sha256"] = file_sha(source)
    manifest["checkpoint_metadata_sha256"] = file_sha(checkpoint / ".metadata")
    destination = args.root.resolve() / "prepared"
    destination.mkdir(parents=True, exist_ok=False)
    path = destination / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"manifest": str(path), "sha256": file_sha(path),
                      "checkpoint": str(checkpoint), "iteration": args.iteration}))


if __name__ == "__main__":
    main()
