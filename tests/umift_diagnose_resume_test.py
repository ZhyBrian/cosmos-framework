from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path

import pytest


PATH = Path(__file__).resolve().parents[1] / "examples/umift/diagnose_resume.py"
SPEC = importlib.util.spec_from_file_location("diagnose_resume", PATH)
assert SPEC and SPEC.loader
diag = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(diag)


def test_wrap_calls_original_once_and_only_captures_when_active() -> None:
    calls = []
    captures = []

    class Subject:
        _umift_diag_active = False

        def operation(self, value):
            calls.append(value)
            return value + 1

    diag._wrap_method(Subject, "operation", lambda args, kwargs, result: captures.append((args, kwargs, result)))
    subject = Subject()
    assert subject.operation(2) == 3
    subject._umift_diag_active = True
    assert subject.operation(4) == 5
    assert calls == [2, 4]
    assert captures == [((4,), {}, 5)]


def test_default_trace_is_only_sixth_update() -> None:
    assert diag.TRACE_ITERATION == 5


def test_noising_signature_supports_positional_and_keyword_calls() -> None:
    def noising(self, gen_data_clean, packed_sequence, sigmas, sigmas_action=None):
        return None

    signature = inspect.signature(noising)
    positional = signature.bind(None, "clean", "packed", "sigma")
    keyword = signature.bind(None, gen_data_clean="clean", packed_sequence="packed", sigmas="sigma")
    assert positional.arguments["packed_sequence"] == keyword.arguments["packed_sequence"] == "packed"
    assert positional.arguments["sigmas"] == keyword.arguments["sigmas"] == "sigma"


def test_tensor_record_handles_scalar_and_bfloat16() -> None:
    torch = pytest.importorskip("torch")
    scalar = torch.tensor(1.25, dtype=torch.float32)
    bf16 = torch.tensor([1.0, 2.0], dtype=torch.bfloat16)
    assert diag._tensor_record(scalar)["shape"] == []
    assert diag._tensor_record(bf16)["dtype"] == "torch.bfloat16"

    class LocalTensor:
        def to_local(self):
            return bf16

    assert diag._tensor_record(LocalTensor())["sha256"] == diag._tensor_record(bf16)["sha256"]
