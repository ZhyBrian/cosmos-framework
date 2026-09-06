#!/usr/bin/env python3
"""Compare two Cosmos distributed training checkpoints without model construction.

The comparator reads DCP metadata and individual storage records directly.  It
therefore works with the real ``model/optim/scheduler/trainer`` component
directories and does not assume a monolithic ``torch.save`` checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import pickle
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


DEFAULT_ATOL = 1.0e-7
DEFAULT_RTOL = 1.0e-6
REQUIRED_DCP_COMPONENTS = ("model", "optim", "scheduler", "trainer")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass
class TensorStats:
    count: int = 0
    changed_count: int = 0
    element_count: int = 0
    changed_element_count: int = 0
    sum_diff_sq: float = 0.0
    sum_ref_sq: float = 0.0
    max_abs: float = 0.0
    max_rel: float = 0.0
    bitwise_equal: bool = True
    tolerance_pass: bool = True

    def add(self, other: "TensorStats") -> None:
        for name in ("count", "changed_count", "element_count", "changed_element_count"):
            setattr(self, name, getattr(self, name) + getattr(other, name))
        self.sum_diff_sq += other.sum_diff_sq
        self.sum_ref_sq += other.sum_ref_sq
        self.max_abs = max(self.max_abs, other.max_abs)
        self.max_rel = max(self.max_rel, other.max_rel)
        self.bitwise_equal &= other.bitwise_equal
        self.tolerance_pass &= other.tolerance_pass

    def report(self) -> dict[str, Any]:
        out = asdict(self)
        out["relative_l2"] = math.sqrt(self.sum_diff_sq) / max(math.sqrt(self.sum_ref_sq), 1.0e-30)
        del out["sum_diff_sq"]
        del out["sum_ref_sq"]
        return out


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("compare_checkpoints requires PyTorch with torch.distributed.checkpoint") from exc
    return torch


def resolve_iteration(path: str | Path) -> Path:
    """Resolve an iteration directory or a job/checkpoints directory via latest marker."""
    root = Path(path).expanduser().resolve()
    if root.name.startswith("iter_"):
        return root
    candidates = (root / "latest_checkpoint.txt", root / "checkpoints" / "latest_checkpoint.txt")
    marker = next((p for p in candidates if p.is_file()), None)
    if marker is None:
        raise FileNotFoundError(f"no latest_checkpoint.txt under {root} or {root / 'checkpoints'}")
    name = marker.read_text(encoding="utf-8").strip()
    if not name or Path(name).name != name or not name.startswith("iter_"):
        raise ValueError(f"invalid latest checkpoint marker {marker}: {name!r}")
    iteration = marker.parent / name
    if not iteration.is_dir():
        raise FileNotFoundError(f"latest marker {marker} points to missing directory {iteration}")
    return iteration


def _index_key(index: Any) -> tuple[str, tuple[int, ...] | None, int | None]:
    offset = getattr(index, "offset", None)
    return (index.fqn, None if offset is None else tuple(offset), getattr(index, "index", None))


def _metadata_signature(meta: Any) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in meta.state_dict_metadata.items():
        if hasattr(value, "size") and hasattr(value, "chunks"):
            result[key] = {
                "kind": "tensor",
                "shape": tuple(value.size),
                "dtype": str(value.properties.dtype),
                "layout": str(value.properties.layout),
                "chunks": sorted((tuple(c.offsets), tuple(c.sizes)) for c in value.chunks),
            }
        else:
            result[key] = {"kind": "bytes"}
    return result


class DCPComponent:
    def __init__(self, path: Path):
        torch = _torch()
        from torch.distributed.checkpoint.filesystem import FileSystemReader

        if not (path / ".metadata").is_file():
            raise FileNotFoundError(f"not a DCP component (missing .metadata): {path}")
        self.path = path
        self.metadata = FileSystemReader(path).read_metadata()
        self.signature = _metadata_signature(self.metadata)
        self.storage = {_index_key(index): info for index, info in self.metadata.storage_data.items()}
        del torch

    def read(self, key: tuple[str, tuple[int, ...] | None, int | None]) -> Any:
        torch = _torch()
        info = self.storage[key]
        file_path = self.path / info.relative_path
        with file_path.open("rb") as stream:
            stream.seek(info.offset)
            payload = stream.read(info.length)
        if len(payload) != info.length:
            raise EOFError(f"short DCP read for {key} from {file_path}")
        return torch.load(io.BytesIO(payload), map_location="cpu", weights_only=False)


def _tensor_stats(reference: Any, candidate: Any, atol: float, rtol: float) -> TensorStats:
    torch = _torch()
    if tuple(reference.shape) != tuple(candidate.shape) or reference.dtype != candidate.dtype:
        raise ValueError("tensor payload disagrees with metadata preflight")
    exact = bool(torch.equal(reference, candidate))
    count = reference.numel()
    if reference.is_floating_point() or reference.is_complex():
        a = reference.detach().to(torch.float64)
        b = candidate.detach().to(torch.float64)
        finite_same = bool(torch.equal(torch.isfinite(a), torch.isfinite(b)))
        both_nan = torch.isnan(a) & torch.isnan(b)
        diff = torch.where(both_nan, torch.zeros_like(a), (b - a).abs())
        scale = a.abs()
        close = finite_same and bool(torch.all(diff <= atol + rtol * scale))
        max_abs = float(diff.max().item()) if count else 0.0
        rel = diff / torch.clamp(scale, min=1.0e-30)
        max_rel = float(rel.max().item()) if count else 0.0
        sum_diff_sq = float(torch.sum(diff.square()).item())
        sum_ref_sq = float(torch.sum(torch.where(torch.isfinite(a), a, torch.zeros_like(a)).square()).item())
        changed_elements = int(torch.count_nonzero(diff).item())
    else:
        neq = reference != candidate
        changed_elements = int(torch.count_nonzero(neq).item())
        max_abs = float((reference.to(torch.float64) - candidate.to(torch.float64)).abs().max().item()) if count else 0.0
        max_rel = math.inf if changed_elements else 0.0
        sum_diff_sq = max_abs * max_abs
        sum_ref_sq = float(torch.sum(reference.to(torch.float64).square()).item())
        close = exact
    return TensorStats(
        count=1,
        changed_count=0 if exact else 1,
        element_count=count,
        changed_element_count=changed_elements,
        sum_diff_sq=sum_diff_sq,
        sum_ref_sq=sum_ref_sq,
        max_abs=max_abs,
        max_rel=max_rel,
        bitwise_equal=exact,
        tolerance_pass=close,
    )


def _category(component: str, fqn: str) -> str:
    lower = fqn.lower()
    if component == "model":
        return "model_all"
    if component == "optim":
        if "exp_avg_sq" in lower or "second_moment" in lower:
            return "optimizer_second_moment"
        if "exp_avg" in lower or "first_moment" in lower:
            return "optimizer_first_moment"
        return "optimizer_other"
    return component


def _stable_object(value: Any) -> bytes:
    return pickle.dumps(value, protocol=4)


def _object_equal(a: Any, b: Any) -> bool:
    torch = _torch()
    if torch.is_tensor(a) or torch.is_tensor(b):
        return torch.is_tensor(a) and torch.is_tensor(b) and a.dtype == b.dtype and tuple(a.shape) == tuple(b.shape) and bool(torch.equal(a, b))
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(_object_equal(a[key], b[key]) for key in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(_object_equal(x, y) for x, y in zip(a, b, strict=True))
    try:
        return bool(a == b)
    except Exception:
        return _stable_object(a) == _stable_object(b)


def compare_component(
    path_a: Path,
    path_b: Path,
    component: str,
    atol: float,
    rtol: float,
    *,
    optimizer_parameter_names: set[str] | None = None,
) -> dict[str, Any]:
    left, right = DCPComponent(path_a), DCPComponent(path_b)
    if left.signature != right.signature:
        missing = sorted(set(left.signature) - set(right.signature))
        extra = sorted(set(right.signature) - set(left.signature))
        mismatched = sorted(k for k in set(left.signature) & set(right.signature) if left.signature[k] != right.signature[k])
        raise ValueError(f"{component} metadata mismatch: missing={missing[:20]}, extra={extra[:20]}, shape/type={mismatched[:20]}")
    if set(left.storage) != set(right.storage):
        raise ValueError(f"{component} DCP chunk-index mismatch")

    categories: dict[str, TensorStats] = defaultdict(TensorStats)
    changed_names: set[str] = set()
    tensor_names: set[str] = set()
    object_mismatches: list[str] = []
    scalar_values: dict[str, dict[str, Any]] = {}
    for key in sorted(left.storage, key=repr):
        a, b = left.read(key), right.read(key)
        if _torch().is_tensor(a):
            tensor_names.add(key[0])
            stats = _tensor_stats(a, b, atol, rtol)
            categories[_category(component, key[0])].add(stats)
            if component == "model" and optimizer_parameter_names and key[0] in optimizer_parameter_names:
                categories["model_optimizer_parameters"].add(stats)
            if not stats.bitwise_equal:
                changed_names.add(key[0])
                if component == "model":
                    categories["model_updated"].add(stats)
            if component == "scheduler" and a.numel() <= 16:
                scalar_values[key[0]] = {
                    "reference": a.tolist(),
                    "candidate": b.tolist(),
                    "dtype": str(a.dtype),
                }
        elif not _object_equal(a, b):
            object_mismatches.append(key[0])
            scalar_values[key[0]] = {"reference": repr(a), "candidate": repr(b)}
        elif component in ("scheduler", "trainer"):
            scalar_values[key[0]] = {"reference": repr(a), "candidate": repr(b)}

    output = {
        "metadata_key_count": len(left.signature),
        "chunk_count": len(left.storage),
        "categories": {name: stats.report() for name, stats in sorted(categories.items())},
        "changed_tensor_names": sorted(changed_names),
        "tensor_names": sorted(tensor_names),
        "object_mismatches": sorted(set(object_mismatches)),
        "scalar_values": scalar_values,
    }
    output["pass"] = not object_mismatches and all(x["tolerance_pass"] for x in output["categories"].values())
    return output


def _validate_umift_stateless_evidence(evidence: dict[str, Any]) -> dict[str, Any]:
    required = {
        "schema_version", "stage", "seed", "world_size", "grad_accum_iter",
        "resume_from_iteration", "derived_resume_microbatch", "checkpoint5_copy",
        "resolved_configs", "resume_log", "gloo_audit",
    }
    missing = sorted(required - evidence.keys())
    if missing:
        raise ValueError(f"UMI stateless evidence is missing fields: {missing}")
    if evidence["schema_version"] != "umift-stateless-resume-v1":
        raise ValueError("unsupported UMI stateless evidence schema")
    sources = [
        evidence.get("resolved_configs", {}).get("reference", {}),
        evidence.get("resolved_configs", {}).get("candidate", {}),
        evidence.get("launch_info", {}).get("reference", {}),
        evidence.get("launch_info", {}).get("candidate", {}),
        evidence.get("reference_log", {}), evidence.get("resume_log", {}),
        evidence.get("checkpoint5_copy", {}), evidence.get("gloo_audit", {}).get("source", {}),
    ]
    for source in sources:
        path, recorded = Path(str(source.get("path", ""))), str(source.get("sha256", ""))
        if not path.is_file() or _file_sha256(path) != recorded:
            raise ValueError(f"UMI stateless evidence source is missing or changed: {path}")
    if evidence["stage"] != "smoke" or int(evidence["seed"]) != 42 or int(evidence["world_size"]) != 4:
        raise ValueError("UMI stateless evidence must identify smoke/seed42/world4")
    resume_iteration = int(evidence["resume_from_iteration"])
    accumulation = int(evidence["grad_accum_iter"])
    derived = int(evidence["derived_resume_microbatch"])
    if resume_iteration != 5 or accumulation != 4 or derived != resume_iteration * accumulation:
        raise ValueError("UMI stateless evidence has an invalid iteration-to-microbatch derivation")
    copied = evidence["checkpoint5_copy"]
    if copied.get("verified") is not True or set(copied.get("components", ())) != set(REQUIRED_DCP_COMPONENTS):
        raise ValueError("UMI stateless evidence lacks a verified four-component checkpoint-5 copy")
    configs = evidence["resolved_configs"]
    if configs.get("semantic_fields_match") is not True:
        raise ValueError("UMI stateless evidence says resolved training configs differ")
    for label in ("reference_sha256", "candidate_sha256"):
        digest = str(configs.get(label, ""))
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest.lower()):
            raise ValueError(f"UMI stateless evidence lacks {label}")
    resume_log = evidence["resume_log"]
    if int(resume_log.get("loaded_iteration", -1)) != resume_iteration:
        raise ValueError("UMI stateless evidence resume log did not load iteration 5")
    log_digest = str(resume_log.get("sha256", ""))
    if int(resume_log.get("derived_sample_offset", -1)) != derived or len(log_digest) != 64 or any(
        char not in "0123456789abcdef" for char in log_digest.lower()
    ):
        raise ValueError("UMI stateless evidence lacks the derived resume offset or log identity")
    audit = evidence["gloo_audit"]
    ranks = audit.get("ranks", [])
    offset_units = {row.get("resume_offset_unit") for row in ranks}
    dataset_identity = {
        "path": audit.get("dataset"),
        "root_metadata_sha256": audit.get("dataset_root_metadata_sha256"),
        "seed": audit.get("seed"),
    }
    if (
        audit.get("stage") != "e1" or int(audit.get("world_size", -1)) != 4
        or int(audit.get("resume_microbatch", -1)) != derived
        or offset_units != {"rank-local microbatch"}
        or not dataset_identity["path"] or not dataset_identity["root_metadata_sha256"]
        or int(dataset_identity["seed"]) != 42 or len(ranks) != 4
        or {int(row.get("rank", -1)) for row in ranks} != set(range(4))
        or not all(row.get("resume_suffix_matches") is True for row in ranks)
    ):
        raise ValueError("UMI stateless evidence has an invalid four-rank Gloo audit")
    canonical_evidence = json.dumps(evidence, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {
        "mode": "umift-iteration-seeded-v1",
        "checkpoint_component_present": False,
        "resume_from_iteration": resume_iteration,
        "grad_accum_iter": accumulation,
        "derived_resume_microbatch": derived,
        "world_size": 4,
        "dataset_identity": dataset_identity,
        "evidence_sha256": hashlib.sha256(canonical_evidence).hexdigest(),
        "basis": "trainer iteration x grad_accum_iter, verified checkpoint-5 copy, matching resolved configs, resume log, and four-rank E1 Gloo suffix audit",
        "pass": True,
    }


def compare_dataloader(a: Path, b: Path, *, umift_stateless_evidence: dict[str, Any] | None = None) -> dict[str, Any]:
    present_a, present_b = a.is_dir(), b.is_dir()
    if present_a != present_b:
        return {"checkpoint_component_present": [present_a, present_b], "pass": False,
                "error": "dataloader checkpoint component exists on only one side"}
    if not present_a:
        if umift_stateless_evidence is None:
            return {"checkpoint_component_present": False, "pass": False,
                    "error": "both dataloader components are absent; explicit verified UMI stateless evidence is required"}
        try:
            return _validate_umift_stateless_evidence(umift_stateless_evidence)
        except Exception as exc:
            return {"mode": "umift-iteration-seeded-v1", "checkpoint_component_present": False,
                    "pass": False, "error": f"{type(exc).__name__}: {exc}"}
    if umift_stateless_evidence is not None:
        return {"checkpoint_component_present": True, "pass": False,
                "error": "UMI stateless evidence conflicts with present dataloader checkpoint components"}
    files_a = {p.name: p for p in a.glob("rank_*.pkl")} if a.is_dir() else {}
    files_b = {p.name: p for p in b.glob("rank_*.pkl")} if b.is_dir() else {}
    if set(files_a) != set(files_b):
        return {"mode": "checkpointed", "checkpoint_component_present": True, "pass": False,
                "error": f"dataloader rank-file mismatch: {sorted(files_a)} != {sorted(files_b)}"}
    mismatches = []
    hashes = {}
    for name in sorted(files_a):
        raw_a, raw_b = files_a[name].read_bytes(), files_b[name].read_bytes()
        obj_a, obj_b = pickle.loads(raw_a), pickle.loads(raw_b)
        equal = _object_equal(obj_a, obj_b)
        if not equal:
            mismatches.append(name)
        hashes[name] = {"reference": hashlib.sha256(raw_a).hexdigest(), "candidate": hashlib.sha256(raw_b).hexdigest(), "semantic_equal": equal}
    if not files_a:
        return {"mode": "checkpointed", "checkpoint_component_present": True, "pass": False,
                "error": "dataloader checkpoint directories contain no rank state files"}
    return {"mode": "checkpointed", "checkpoint_component_present": True, "rank_files": hashes, "mismatches": mismatches, "pass": not mismatches}


def _iteration_number(path: Path) -> int:
    try:
        return int(path.name.removeprefix("iter_"))
    except ValueError as exc:
        raise ValueError(f"invalid iteration directory name: {path.name}") from exc


def _read_component_values(path: Path) -> dict[str, Any]:
    component = DCPComponent(path)
    values: dict[str, Any] = {}
    keys_by_fqn: dict[str, list[tuple[str, tuple[int, ...] | None, int | None]]] = defaultdict(list)
    for key in component.storage:
        keys_by_fqn[key[0]].append(key)
    for fqn, keys in sorted(keys_by_fqn.items()):
        # Objects and scalar tensors each have one storage record. Larger tensors
        # may be sharded; no individual shard is a valid global counter.
        if len(keys) == 1:
            value = component.read(keys[0])
            if not _torch().is_tensor(value) or value.numel() == 1:
                values[fqn] = value
    return values


def _scalar_int(value: Any) -> int | None:
    torch = _torch()
    if torch.is_tensor(value) and value.numel() == 1:
        return int(value.item())
    if isinstance(value, (int, float)) and int(value) == value:
        return int(value)
    return None


def validate_iteration_counters(root: Path) -> dict[str, Any]:
    expected = _iteration_number(root)
    trainer_values = _read_component_values(root / "trainer")
    trainer_candidates = {name: _scalar_int(value) for name, value in trainer_values.items() if name.endswith("iteration")}
    trainer_candidates = {name: value for name, value in trainer_candidates.items() if value is not None}
    if not trainer_candidates or set(trainer_candidates.values()) != {expected}:
        raise ValueError(f"trainer iteration does not match directory iteration {expected}: {trainer_candidates}")

    optim_values = _read_component_values(root / "optim")
    optimizer_steps: dict[str, int] = {}
    def visit(value: Any, prefix: str) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                visit(child, f"{prefix}.{key}" if prefix else str(key))
        elif isinstance(value, (list, tuple)):
            for index, child in enumerate(value):
                visit(child, f"{prefix}[{index}]")
        elif prefix.endswith("step"):
            scalar = _scalar_int(value)
            if scalar is not None:
                optimizer_steps[prefix] = scalar
    for name, value in optim_values.items():
        visit(value, name)
    if not optimizer_steps or set(optimizer_steps.values()) != {expected}:
        raise ValueError(f"optimizer step counters do not match directory iteration {expected}: {optimizer_steps}")

    scheduler_values = _read_component_values(root / "scheduler")
    scheduler_counts = {
        name: scalar for name, value in scheduler_values.items()
        if (name.endswith("last_epoch") or name.endswith("_step_count")) and (scalar := _scalar_int(value)) is not None
    }
    if not scheduler_counts:
        raise ValueError("scheduler checkpoint exposes no recognized iteration counter")
    # Scheduler implementations differ in whether _step_count is last_epoch or last_epoch+1.
    # Require an exact last_epoch anchor when present; report auxiliary counters without
    # imposing a universal +1 convention.
    last_epochs = {value for name, value in scheduler_counts.items() if name.endswith("last_epoch")}
    if last_epochs and last_epochs != {expected}:
        raise ValueError(f"scheduler last_epoch does not match directory iteration {expected}: {scheduler_counts}")
    return {"directory_iteration": expected, "trainer": trainer_candidates, "optimizer_steps": optimizer_steps, "scheduler_counts": scheduler_counts, "pass": True}


def _iteration_consistency_report(root: Path) -> dict[str, Any]:
    try:
        return validate_iteration_counters(root)
    except Exception as exc:
        return {"directory_iteration": _iteration_number(root), "pass": False, "error": f"{type(exc).__name__}: {exc}"}


def _rng_report(trainer: dict[str, Any]) -> dict[str, Any]:
    changed = set(trainer["changed_tensor_names"]) | set(trainer["object_mismatches"])
    fqns = set(trainer.get("tensor_names", ())) | set(trainer.get("scalar_values", {})) | changed
    ranks: dict[str, dict[str, Any]] = {}
    for fqn in sorted(k for k in fqns if k.startswith("rng_state_")):
        rank = fqn.split(".", 1)[0].removeprefix("rng_state_")
        ranks.setdefault(rank, {"state_equal": True, "changed_fields": []})
        if fqn in changed:
            ranks[rank]["state_equal"] = False
            ranks[rank]["changed_fields"].append(fqn)
    return {"per_rank": ranks, "all_sequences_identical": bool(ranks) and all(v["state_equal"] for v in ranks.values()), "basis": "exact equality of serialized Torch CPU/CUDA, NumPy, and Python RNG states; equal states define identical subsequent sequences"}


def compare_checkpoints(reference: str | Path, candidate: str | Path, *, atol: float = DEFAULT_ATOL, rtol: float = DEFAULT_RTOL, umift_stateless_evidence: dict[str, Any] | None = None) -> dict[str, Any]:
    if atol < 0 or rtol < 0:
        raise ValueError("tolerances must be non-negative")
    root_a, root_b = resolve_iteration(reference), resolve_iteration(candidate)
    for component in REQUIRED_DCP_COMPONENTS:
        if not (root_a / component).is_dir() or not (root_b / component).is_dir():
            raise FileNotFoundError(f"both checkpoints must contain DCP component {component!r}")
    result: dict[str, Any] = {
        "reference": str(root_a),
        "candidate": str(root_b),
        "tolerances_fixed_before_comparison": {"atol": atol, "rtol": rtol, "rule": "max_abs <= atol + rtol * abs(reference), elementwise"},
        "components": {},
    }
    result["iteration_consistency"] = {
        "reference": _iteration_consistency_report(root_a),
        "candidate": _iteration_consistency_report(root_b),
    }
    # Optimizer state uses canonical parameter FQNs. Compare it first so those
    # names can identify the updated/trainable subset of the model state.
    result["components"]["optim"] = compare_component(root_a / "optim", root_b / "optim", "optim", atol, rtol)
    optimizer_parameter_names = set()
    for name in result["components"]["optim"]["tensor_names"]:
        if name.startswith("state."):
            for suffix in (".exp_avg_sq", ".exp_avg", ".second_moment", ".first_moment", ".step"):
                if name.endswith(suffix):
                    optimizer_parameter_names.add(name[len("state.") : -len(suffix)])
                    break
    result["components"]["model"] = compare_component(
        root_a / "model",
        root_b / "model",
        "model",
        atol,
        rtol,
        optimizer_parameter_names=optimizer_parameter_names,
    )
    for component in ("scheduler", "trainer"):
        result["components"][component] = compare_component(root_a / component, root_b / component, component, atol, rtol)
    result["dataloader"] = compare_dataloader(root_a / "dataloader", root_b / "dataloader", umift_stateless_evidence=umift_stateless_evidence)
    result["rng"] = _rng_report(result["components"]["trainer"])
    model_all = result["components"]["model"]["categories"].get("model_all", {})
    result["model"] = {
        "all_parameters": model_all,
        "optimizer_updated_parameters": result["components"]["model"]["categories"].get(
            "model_optimizer_parameters", TensorStats().report()
        ),
        "optimizer_parameter_names": sorted(optimizer_parameter_names),
        "differing_parameters": result["components"]["model"]["categories"].get("model_updated", TensorStats().report()),
        "differing_parameter_names": result["components"]["model"]["changed_tensor_names"],
    }
    result["pass"] = (
        all(c["pass"] for c in result["components"].values())
        and all(item["pass"] for item in result["iteration_consistency"].values())
        and result["dataloader"]["pass"] and result["rng"]["all_sequences_identical"]
    )
    return result


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", help="continuous-run iter directory, checkpoints directory, or job directory")
    parser.add_argument("candidate", help="resumed-run iter directory, checkpoints directory, or job directory")
    parser.add_argument("--atol", type=float, default=DEFAULT_ATOL)
    parser.add_argument("--rtol", type=float, default=DEFAULT_RTOL)
    parser.add_argument("--umift-stateless-evidence", type=Path, help="JSON evidence for the UMI iteration-seeded dataloader contract")
    parser.add_argument("--output", type=Path, required=True, help="JSON report path (written even when numeric comparison fails)")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        evidence = None if args.umift_stateless_evidence is None else json.loads(args.umift_stateless_evidence.read_text())
        report = compare_checkpoints(args.reference, args.candidate, atol=args.atol, rtol=args.rtol, umift_stateless_evidence=evidence)
    except Exception as exc:
        report = {"pass": False, "error": f"{type(exc).__name__}: {exc}", "tolerances_fixed_before_comparison": {"atol": args.atol, "rtol": args.rtol}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report.get("pass") else 1


if __name__ == "__main__":
    sys.exit(main())
