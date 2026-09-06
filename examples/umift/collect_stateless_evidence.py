#!/usr/bin/env python3
"""Collect traceable evidence for UMI's iteration-seeded resume contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any


COMPONENTS = {"model", "optim", "scheduler", "trainer"}
TRAINING_TREES = ("model", "optimizer", "scheduler", "trainer", "dataloader_train", "checkpoint")
_ALLOWED_PATH_KEYS = {"output_dir", "output_root", "job_dir", "log_dir", "save_dir", "save_path", "experiment_dir"}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _yaml(path: Path) -> Any:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to read resolved Cosmos config YAML") from exc
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _walk(value: Any, prefix: str = ""):
    if isinstance(value, dict):
        for key, child in value.items():
            yield from _walk(child, f"{prefix}.{key}" if prefix else str(key))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _walk(child, f"{prefix}[{index}]")
    else:
        yield prefix, value


def _named_values(value: Any, name: str) -> list[tuple[str, Any]]:
    return [(path, item) for path, item in _walk(value) if path.rsplit(".", 1)[-1] == name]


def _at(value: Any, dotted: str) -> Any:
    current = value
    for part in dotted.split("."):
        if not isinstance(current, dict) or part not in current:
            raise ValueError(f"resolved config lacks exact field {dotted}")
        current = current[part]
    return current


def _require_named_value(value: Any, name: str, expected: Any, source: Path) -> str:
    matches = [(path, item) for path, item in _named_values(value, name) if item == expected]
    if not matches:
        raise ValueError(f"{source} has no {name}={expected!r}")
    return matches[0][0]


def _semantic_config(path: Path) -> dict[str, Any]:
    parsed = _yaml(path)
    if not isinstance(parsed, dict):
        raise ValueError(f"{path} is not a mapping")
    missing = [key for key in TRAINING_TREES if key not in parsed]
    if missing:
        raise ValueError(f"{path} lacks resolved training trees: {missing}")
    def scrub(value: Any, prefix: str = "") -> Any:
        if isinstance(value, dict):
            return {key: ("<allowed-job-path>"
                          if key in _ALLOWED_PATH_KEYS or f"{prefix}.{key}" == "trainer.callbacks.sampled_media.output_uri"
                          else scrub(child, f"{prefix}.{key}"))
                    for key, child in value.items()}
        if isinstance(value, list):
            return [scrub(child) for child in value]
        return value
    trees = {key: scrub(parsed[key], key) for key in TRAINING_TREES}
    tree_digest = hashlib.sha256(json.dumps(trees, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()
    expected = {
        "trainer.seed": 42,
        "trainer.grad_accum_iter": 4,
        "model.config.parallelism.context_parallel_shard_degree": 1,
        "model.config.parallelism.data_parallel_replicate_degree": 1,
        "model.config.parallelism.data_parallel_shard_degree": 4,
        "dataloader_train.max_samples_per_batch": 1,
        "dataloader_train.dataloader.num_workers": 0,
        "dataloader_train.dataloader.generator.seed": 42,
        "dataloader_train.dataloader.datasets.umift.dataset.seed": 42,
        "dataloader_train.dataloader.datasets.umift.dataset.split": "train",
        "dataloader_train.dataloader.datasets.umift.dataset.stage": "smoke",
    }
    fields = {name: _at(parsed, name) for name in expected}
    wrong = {name: {"expected": expected[name], "actual": value} for name, value in fields.items() if value != expected[name]}
    if wrong:
        raise ValueError(f"{path} violates UMI stateless E0 config contract: {wrong}")
    dataset_path = str(_at(parsed, "dataloader_train.dataloader.datasets.umift.dataset.zarr_path"))
    return {"path": str(path.resolve()), "sha256": _sha(path), "parsed_fields": fields,
            "dataset_path": str(Path(dataset_path).resolve()),
            "training_trees_sha256": tree_digest, "training_trees": trees}


def _launch(path: Path) -> dict[str, Any]:
    parsed = _yaml(path)
    if not isinstance(parsed, dict) or not {"cmd", "args_cfg_path", "args_override"}.issubset(parsed):
        raise ValueError(f"{path} lacks real launch_info cmd/args_cfg_path/args_override fields")
    return {"path": str(path.resolve()), "sha256": _sha(path),
            "cmd": parsed["cmd"], "args_cfg_path": parsed["args_cfg_path"],
            "args_override": parsed["args_override"]}


def _driver(path: Path, *, require_loaded_five: bool) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace")
    loaded = bool(re.search(r"(?:loaded|load(?:ing|ed)?).*iteration[^0-9]*5\b", text, re.IGNORECASE))
    if require_loaded_five and not loaded:
        raise ValueError(f"{path} has no parsed checkpoint iteration-5 load event")
    world_four = bool(re.search(r"world(?:_size| size)?\s*[=: ]\s*4\b", text, re.IGNORECASE))
    ranks = sorted({int(x) for x in re.findall(r"\brank\s*[=: ]\s*([0-3])\b", text, re.IGNORECASE)})
    return {"path": str(path.resolve()), "sha256": _sha(path), "loaded_iteration": 5 if loaded else None,
            "world_size_4_observed": world_four, "observed_ranks": ranks}


def _copy_evidence(path: Path) -> dict[str, Any]:
    parsed = json.loads(path.read_text(encoding="utf-8"))
    if parsed.get("all_hashes_equal") is not True or parsed.get("dataloader_component_absent") is not True:
        raise ValueError(f"{path} is not a successful copy operation log")
    if set(parsed.get("checkpoint_components", ())) != COMPONENTS:
        raise ValueError(f"{path} has the wrong checkpoint component set")
    source = Path(str(parsed.get("source", "")))
    target = Path(str(parsed.get("target", "")))
    if source.name != "iter_000000005" or target.name != "iter_000000005" or source == target:
        raise ValueError(f"{path} has invalid checkpoint-5 source/target")
    files = parsed.get("files")
    if not isinstance(files, list) or len(files) != 20:
        raise ValueError(f"{path} must contain exactly 20 copy records")
    expected_paths = {f"{component}/.metadata" for component in COMPONENTS}
    expected_paths |= {f"{component}/__{rank}_0.distcp" for component in COMPONENTS for rank in range(4)}
    records = {str(row.get("path")): row for row in files if isinstance(row, dict)}
    if set(records) != expected_paths:
        raise ValueError(f"{path} copy record paths do not match four components x four ranks plus metadata")
    for relative, row in records.items():
        size, digest = int(row.get("bytes", -1)), str(row.get("sha256", ""))
        if size < 0 or len(digest) != 64:
            raise ValueError(f"{path} has invalid record for {relative}")
        # Re-read every small source artifact. Multi-GB model/optimizer shards are
        # represented by the contemporaneous operation log and are not rehashed.
        if size <= 2 * 1024 * 1024:
            actual = source / relative
            if not actual.is_file() or actual.stat().st_size != size or _sha(actual) != digest:
                raise ValueError(f"copy operation source artifact is missing or changed: {actual}")
    return {"path": str(path.resolve()), "sha256": _sha(path), "verified": True,
            "components": sorted(COMPONENTS), "record_count": 20,
            "meaning": "validated copy operation log; small source artifacts rehashed, large shards not independently rehashed"}


def collect(args: argparse.Namespace) -> dict[str, Any]:
    reference_config = _semantic_config(args.reference_config)
    candidate_config = _semantic_config(args.candidate_config)
    if reference_config["training_trees_sha256"] != candidate_config["training_trees_sha256"]:
        raise ValueError("resolved model/optimizer/scheduler/trainer/dataloader/checkpoint trees differ")
    reference_launch = _launch(args.reference_launch_info)
    candidate_launch = _launch(args.candidate_launch_info)
    reference_log = _driver(args.reference_driver_log, require_loaded_five=False)
    resume_log = _driver(args.candidate_driver_log, require_loaded_five=True)
    for label, report in (("reference", reference_log), ("candidate", resume_log)):
        if not report["world_size_4_observed"] or report["observed_ranks"] != [0, 1, 2, 3]:
            raise ValueError(f"{label} driver log does not prove world size 4 with ranks 0..3")
    audit = json.loads(args.gloo_audit.read_text(encoding="utf-8"))
    audit_dataset = str(Path(str(audit.get("dataset", ""))).resolve())
    if audit_dataset != reference_config["dataset_path"] or int(audit.get("seed", -1)) != 42:
        raise ValueError("Gloo audit dataset path/seed differs from the resolved training config")
    audit["source"] = {"path": str(args.gloo_audit.resolve()), "sha256": _sha(args.gloo_audit)}
    return {
        "schema_version": "umift-stateless-resume-v1", "stage": "smoke", "seed": 42,
        "world_size": 4, "grad_accum_iter": 4, "resume_from_iteration": 5,
        "derived_resume_microbatch": 20,
        "checkpoint5_copy": _copy_evidence(args.checkpoint5_copy_evidence),
        "resolved_configs": {"semantic_fields_match": True,
            "reference_sha256": reference_config["sha256"], "candidate_sha256": candidate_config["sha256"],
            "training_trees_sha256": reference_config["training_trees_sha256"],
            "reference": reference_config, "candidate": candidate_config},
        "launch_info": {"reference": reference_launch, "candidate": candidate_launch},
        "training_code_identity": {"commit": args.training_commit,
            "provenance": "operator-recorded pre-launch git rev-parse/pull observation; original launch_info does not record git SHA"},
        "reference_log": reference_log,
        "resume_log": {**resume_log, "derived_sample_offset": 20},
        "gloo_audit": audit,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("reference-config", "candidate-config", "reference-launch-info", "candidate-launch-info",
                 "reference-driver-log", "candidate-driver-log", "checkpoint5-copy-evidence", "gloo-audit"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--training-commit", default="9c81289")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    evidence = collect(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(evidence, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
