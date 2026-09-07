"""Read-only E1-R population, sampling and resolved-recipe audit; no CUDA work."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
from itertools import islice
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import os
    import yaml
    from cosmos_framework.configs.toml_config.sft_config import load_experiment_from_toml
    from cosmos_framework.data.generator.action.datasets.umift_zarr_dataset import UMIFTZarrIterableDataset
    from cosmos_framework.utils.serialization import to_yaml
    from examples.umift.preflight import assert_edge_fd_config

    configs = [load_experiment_from_toml(f"examples/toml/sft_config/{name}.toml") for name in
               ("action_fd_umift_edge", "action_fd_umift_edge_refit")]
    for config in configs:
        assert_edge_fd_config(config)
        config.validate()
    serialized = [yaml.safe_load(to_yaml(config)) for config in configs]

    def differences(a, b, path=""):
        if isinstance(a, dict) and isinstance(b, dict):
            assert a.keys() == b.keys(), path
            return [row for key in a for row in differences(a[key], b[key], f"{path}.{key}".lstrip("."))]
        return [] if a == b else [{"path": path, "e1": a, "refit": b}]

    delta = differences(*serialized)
    assert {item["path"] for item in delta} == {
        "job.name", "dataloader_train.dataloader.datasets.umift.dataset.split",
        "trainer.callbacks.sampled_media.output_uri",
    }, delta
    media = next(item for item in delta if item["path"] == "trainer.callbacks.sampled_media.output_uri")
    assert media["refit"] == media["e1"].replace(
        "/action_fd_umift_edge_e1/", "/action_fd_umift_edge_e1_refit/"
    ), media  # derived run path, not an independent training change
    datasets = {split: UMIFTZarrIterableDataset(os.environ["DATASET_PATH"], split=split, seed=42)
                for split in ("train", "dev", "history", "refit_train")}
    roster = {split: [vars(ep) for ep in ds._episodes] for split, ds in datasets.items()}
    sessions = {split: {ep.session_id for ep in ds._episodes} for split, ds in datasets.items()}
    ids = {split: {ep.episode_id for ep in ds._episodes} for split, ds in datasets.items()}
    assert ids["refit_train"] == set(range(59)) - {13, 43, 49}
    assert ids["train"] == set(range(50)) - {13, 43, 49}
    assert ids["dev"] == set(range(50, 59))
    assert ids["history"] == {13, 43, 49}
    assert sessions["refit_train"].isdisjoint(sessions["history"])
    assert len(sessions["refit_train"]) == 37 and len(sessions["history"]) == 3
    draws = [(i, ep.episode_id, ep.session_id, start)
             for i, ep, start in islice(datasets["refit_train"]._training_draws(0), 16000)]
    assert {row[1] for row in draws} == ids["refit_train"]
    assert all(row[1] not in ids["history"] for row in draws)
    report = {
        "protocol": "e1-refit-56-fixed-1000-v1", "dataset": os.environ["DATASET_PATH"],
        "config_differences": delta, "roster": roster,
        "episode_counts": {key: len(value) for key, value in ids.items()},
        "session_counts": {key: len(value) for key, value in sessions.items()},
        "dense_window_counts": {key: ds.total_images for key, ds in datasets.items()},
        "draw_count": len(draws), "draw_sha256": hashlib.sha256(json.dumps(draws).encode()).hexdigest(),
        "draws_per_episode": dict(Counter(row[1] for row in draws)),
        "first_16_draws": draws[:16], "test_session_overlap": [],
        "selection": "fixed iter1000; historical test diagnostics do not select or tune",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("config_differences", "episode_counts", "session_counts",
                                                  "dense_window_counts", "draw_count", "draw_sha256")}, indent=2))


if __name__ == "__main__":
    main()
