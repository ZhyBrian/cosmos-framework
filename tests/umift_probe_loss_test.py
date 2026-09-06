from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "examples/umift/probe_loss.py"
SPEC = importlib.util.spec_from_file_location("umift_probe_loss", SCRIPT)
assert SPEC and SPEC.loader
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


def test_rng_reset_preserves_normal_training_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, int | None]] = []
    fake_torch = SimpleNamespace(
        are_deterministic_algorithms_enabled=lambda: False,
        manual_seed=lambda seed: calls.append(("cpu", seed)),
        cuda=SimpleNamespace(manual_seed_all=lambda seed: calls.append(("cuda", seed))),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    assert probe.reset_probe_rng(3, 2) == 3 * 65536 + 2
    assert calls == [("cpu", 3 * 65536 + 2), ("cuda", 3 * 65536 + 2)]
    assert not hasattr(fake_torch, "use_deterministic_algorithms")


def _report(*, mutate: str | None = None) -> dict:
    rows = []
    for rank in range(4):
        for draw in range(4):
            row = {
                "rank": rank, "draw": draw, "window_id": probe.expected_overfit_window_id(rank, draw), "loss": 2.0,
                "x0_sha256": "x0", "xt_sha256": "xt", "sigma_sha256": "sigma",
                "noise_sha256": "noise", "vision_mask_sha256": "vm", "action_mask_sha256": "am",
            }
            rows.append(row)
    if mutate:
        rows[7][mutate] = "different"
    return {"rows": rows}


def test_compare_requires_all_fixture_hashes_before_loss_comparison() -> None:
    result = probe.compare_reports(_report(), _report())
    assert result["fixture_match"] is True
    assert result["trained_over_base"] == 1.0
    for field in ("window_id", "x0_sha256", "xt_sha256", "sigma_sha256", "noise_sha256",
                  "vision_mask_sha256", "action_mask_sha256"):
        with pytest.raises(ValueError, match="fixture mismatch"):
            probe.compare_reports(_report(), _report(mutate=field))


def test_overfit_fixed_rank_windows_and_invalid_loss_contract() -> None:
    assert [[probe.expected_overfit_window_id(rank, draw) for draw in range(4)] for rank in range(4)] == [
        ["episode_0:s=0"] * 4,
        ["episode_0:s=64"] * 4,
        ["episode_0:s=128"] * 4,
        ["episode_0:s=192"] * 4,
    ]
    negative = _report()
    negative["rows"][0]["loss"] = -1.0
    with pytest.raises(ValueError, match="non-negative"):
        probe.compare_reports(_report(), negative)
    nonfinite = _report()
    nonfinite["rows"][0]["loss"] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        probe.compare_reports(_report(), nonfinite)
    zero = _report()
    for row in zero["rows"]:
        row["loss"] = 0.0
    result = probe.compare_reports(zero, zero)
    assert result["trained_over_base"] is None
    assert all(value is None for value in result["per_row"])


def test_runtime_uses_real_model_loss_without_optimizer_or_backward() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert "model.training_step(batch, draw)" in source
    assert "model._add_noise_to_input" in source
    assert "torch.no_grad()" in source
    assert "init_optimizer_scheduler" not in source
    assert ".backward(" not in source
    assert "use_deterministic_algorithms(" not in source
    assert "trainer.checkpointer.load(model)" in source
    assert 'CUDA_VISIBLE_DEVICES") != "0,1,2,3"' in source
    assert source.index('CUDA_VISIBLE_DEVICES") != "0,1,2,3"') < source.index("    import torch\n", source.index("def run_probe"))
    assert "torch.all(mask[0] == 1)" in source
    assert "torch.all(mask[1:] == 0)" in source
