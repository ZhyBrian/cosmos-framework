from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "examples/umift/convert_checkpoint.py"
SPEC = importlib.util.spec_from_file_location("umift_convert_checkpoint", SCRIPT)
assert SPEC and SPEC.loader
converter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(converter)


def _inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    snapshot = tmp_path / converter.EDGE_REVISION
    snapshot.mkdir()
    (snapshot / "config.json").write_text("{}", encoding="utf-8")
    (snapshot / "tokenizer.json").write_text("{}", encoding="utf-8")
    (snapshot / "model-00001-of-00001.safetensors").write_bytes(b"weights")
    vae = tmp_path / "Wan2.2_VAE.pth"
    vae.write_bytes(b"vae")
    return snapshot, vae, tmp_path / "converted"


def test_converter_forces_cpu_and_offline_before_framework_imports() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert 'os.environ["COSMOS_DEVICE"] = "cpu"' in source
    assert 'os.environ["HF_HUB_OFFLINE"] = "1"' in source
    assert "build_public_model_config" not in source
    assert "Cosmos3OmniConfig(model=model_dict)" in source
    assert "data_parallel_shard_degree=1" in source
    assert "CompileConfig(enabled=False)" in source


def test_local_inputs_and_source_manifest(tmp_path: Path) -> None:
    snapshot, vae, output = _inputs(tmp_path)
    converter.validate_inputs(snapshot, vae, output)
    manifest = converter.source_manifest(snapshot)
    assert set(manifest) == {
        "config.json", "tokenizer.json", "model-00001-of-00001.safetensors"
    }
    assert all(len(value) == 64 for value in manifest.values())

    with pytest.raises(ValueError, match="fixed Edge revision"):
        wrong = tmp_path / "main"
        wrong.mkdir()
        converter.validate_inputs(wrong, vae, output)
    output.mkdir()
    with pytest.raises(FileExistsError, match="overwrite"):
        converter.validate_inputs(snapshot, vae, output)


def test_converter_has_full_key_shape_and_strict_read_contract() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert "expected_keys != actual_keys" in source
    assert "DCP shape mismatch" in source
    assert "dcp.load(state_dict=state" in source
    assert '"parameter_count"' in source
    assert '"source_files_sha256"' in source
    assert '"vae_sha256"' in source
