from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


PATH = Path(__file__).resolve().parents[1] / "examples/umift/probe_vae_repro.py"
SPEC = importlib.util.spec_from_file_location("probe_vae_repro", PATH)
assert SPEC and SPEC.loader
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


def test_tensor_summary_preserves_scalar_and_stride() -> None:
    torch = pytest.importorskip("torch")
    scalar = torch.tensor(1.0)
    matrix = torch.arange(6, dtype=torch.bfloat16).reshape(2, 3).T
    assert probe._tensor_summary(scalar)["shape"] == []
    assert probe._tensor_summary(matrix)["stride"] == [1, 3]
    assert probe._tensor_summary(matrix)["dtype"] == "torch.bfloat16"
