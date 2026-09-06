from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest

from examples.umift.infer import (
    E0_OVERFIT_STARTS,
    _window_id,
    _structure_runtime_configs,
    condition_only_video,
    compare_checkpoint_keys,
    decoded_video_to_thwc01,
    iter_rank_windows,
    make_action_variant,
    merge_rank_payloads,
    run_forward_dynamics,
    iter_evaluation_windows,
    resolve_dataset_protocol,
    validate_sampling_protocol,
    validate_checkpoint_path,
    validate_fd_sample,
    validate_independent_parallelism,
    validate_launch_environment,
)


def _sample() -> dict:
    action = np.arange(16 * 64, dtype=np.float32).reshape(16, 64)
    model_action = action[:, :10].copy()
    video = np.zeros((3, 17, 256, 256), dtype=np.uint8)
    video[:, 1:] = 1
    return {
        "ai_caption": "",
        "video": video,
        "action": action,
        "model_action": model_action,
        "physical_action": np.zeros((16, 10), np.float32),
        "conditioning_fps": np.float32(15.0),
        "mode": "forward_dynamics",
        "domain_id": 6,
        "session_id": "session-a",
        "source_id": "session-a#seg0",
        "episode_id": 50,
        "window_start": 32,
        "sequence_plan": SimpleNamespace(
            has_text=True,
            has_vision=True,
            has_action=True,
            condition_frame_indexes_vision=[0],
            condition_frame_indexes_action=list(range(16)),
        ),
    }


def test_validation_rejects_future_vision_conditioning() -> None:
    sample = _sample()
    sample["sequence_plan"].condition_frame_indexes_vision = [0, 1]

    with pytest.raises(ValueError, match="future vision leakage"):
        validate_fd_sample(sample)


def test_validation_rejects_non_256_canvas() -> None:
    sample = _sample()
    sample["video"] = sample["video"][:, :, :128, :]
    with pytest.raises(ValueError, match=r"\[3,17,256,256\]"):
        validate_fd_sample(sample)


def test_condition_batch_keeps_i0_and_erases_all_future_rgb() -> None:
    video = _sample()["video"]

    conditioned = condition_only_video(video)

    np.testing.assert_array_equal(conditioned[:, 0], video[:, 0])
    assert np.count_nonzero(conditioned[:, 1:]) == 0
    assert np.count_nonzero(video[:, 1:]) > 0


def test_z_variant_builds_physical_identity_then_normalizes_once() -> None:
    calls: list[np.ndarray] = []

    def normalizer(value: np.ndarray) -> np.ndarray:
        calls.append(value.copy())
        return value + 5

    result = make_action_variant(_sample(), "Z", normalizer=normalizer)

    expected = np.array([0, 0, 0, 1, 0, 0, 0, 1, 0, 0], np.float32)
    assert len(calls) == 1
    np.testing.assert_array_equal(calls[0][0], expected)
    np.testing.assert_array_equal(result["model_action"][0], expected + 5)
    np.testing.assert_array_equal(result["action"][0, :10], expected + 5)
    assert np.count_nonzero(result["action"][:, 10:]) == 0


def test_s_variant_uses_replacement_model_action_without_renormalizing() -> None:
    replacement = _sample()
    replacement["model_action"] = np.full((16, 10), -0.25, np.float32)

    result = make_action_variant(_sample(), "S", replacement=replacement)

    np.testing.assert_array_equal(result["model_action"], replacement["model_action"])
    np.testing.assert_array_equal(result["action"][:, :10], replacement["model_action"])


def test_runner_passes_fixed_noise_seed_and_decodes_one_video() -> None:
    calls: list[dict] = []

    class FakeModel:
        def generate_samples_from_batch(self, batch, **kwargs):
            calls.append({"batch": batch, **kwargs})
            return {"vision": [np.full((1, 3, 17, 2, 2), 0.5, np.float32)]}

        def decode(self, latent):
            return latent

    batch = {"video": [np.ones((1, 3, 17, 2, 2), np.float32)]}
    result = run_forward_dynamics(FakeModel(), batch, noise_seed=1234, num_steps=30)

    assert calls[0]["seed"] == [1234]
    assert calls[0]["num_steps"] == 30
    assert calls[0]["guidance"] == 1.0
    assert result.shape == (17, 2, 2, 3)
    assert (result == 0.75).all()


def test_decode_contract_converts_minus_one_one_once() -> None:
    decoded = np.stack(
        [np.full((17, 2, 2), value, np.float32) for value in (-1.0, 0.0, 1.0)], axis=0
    )[None]

    actual = decoded_video_to_thwc01(decoded)

    assert actual.shape == (17, 2, 2, 3)
    np.testing.assert_array_equal(actual[0, 0, 0], [0.0, 0.5, 1.0])


def test_decode_rejects_nonfinite_model_output() -> None:
    decoded = np.zeros((1, 3, 17, 2, 2), np.float32)
    decoded[0, 0, 1, 0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        decoded_video_to_thwc01(decoded)


def test_history_generation_requires_all_three_preregistered_seeds() -> None:
    with pytest.raises(ValueError, match=r"\[0, 1, 2\]"):
        validate_sampling_protocol("history", "E1-A", [0])
    validate_sampling_protocol("history", "E1-A", [0, 1, 2])


def test_training_iteration_root_is_rejected_in_favor_of_model_component(tmp_path: Path) -> None:
    (tmp_path / "model").mkdir()
    with pytest.raises(ValueError, match="/model"):
        validate_checkpoint_path(tmp_path)


def test_checkpoint_key_comparison_rejects_missing_and_unexpected_keys() -> None:
    assert compare_checkpoint_keys({"a", "b"}, {"a", "b"}) == {"model_key_count": 2, "checkpoint_key_count": 2}
    with pytest.raises(ValueError, match=r"missing=\['b'\].*unexpected=\['c'\]"):
        compare_checkpoint_keys({"a", "b"}, {"a", "c"})


def test_runtime_configs_roundtrip_type_metadata_and_apply_inference_overrides() -> None:
    pytest.importorskip("torch")
    from cosmos_framework.configs.base.defaults.compile import CompileConfig
    from cosmos_framework.configs.base.defaults.parallelism import ParallelismConfig
    from cosmos_framework.configs.base.defaults.quantization import QuantizationConfig
    from cosmos_framework.inference.common.config import unstructure_config

    parallelism = unstructure_config(ParallelismConfig(enable_inference_mode=False))
    compile_options = unstructure_config(CompileConfig(enabled=True))
    quantization = unstructure_config(QuantizationConfig())
    assert all("_type" in value for value in (parallelism, compile_options, quantization))

    restored_parallelism, restored_compile, restored_quantization = _structure_runtime_configs({
        "parallelism": parallelism,
        "compile": compile_options,
        "quantization": quantization,
    })

    assert isinstance(restored_parallelism, ParallelismConfig)
    assert restored_parallelism.enable_inference_mode is True
    assert isinstance(restored_compile, CompileConfig)
    assert restored_compile.enabled is False
    assert isinstance(restored_quantization, QuantizationConfig)

    independent_parallelism, independent_compile, _ = _structure_runtime_configs(
        {"parallelism": parallelism, "compile": compile_options, "quantization": quantization},
        independent_windows=True,
    )
    assert isinstance(independent_parallelism, ParallelismConfig)
    assert independent_parallelism.enable_inference_mode is True
    assert independent_parallelism.data_parallel_shard_degree == 1
    assert independent_parallelism.data_parallel_replicate_degree == 4
    assert independent_parallelism.context_parallel_shard_degree == 1
    assert independent_parallelism.cfg_parallel_shard_degree == 1
    assert independent_parallelism.vae_load_balance_group_size == 1
    assert independent_compile.enabled is False


def test_model_launch_requires_only_gpu_zero_through_three_visible() -> None:
    validate_launch_environment({"CUDA_VISIBLE_DEVICES": "0,1,2,3", "WORLD_SIZE": "4"})
    with pytest.raises(ValueError, match="CUDA_VISIBLE_DEVICES=0,1,2,3"):
        validate_launch_environment({"CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7", "WORLD_SIZE": "4"})


def test_overfit_split_maps_only_to_training_episode_and_overfit_stage() -> None:
    assert resolve_dataset_protocol("overfit", "e1") == ("train", "overfit")
    assert resolve_dataset_protocol("dev", "smoke") == ("dev", "smoke")
    assert resolve_dataset_protocol("history", "e1") == ("history", "e1")


def test_overfit_enumeration_uses_exact_four_frozen_windows() -> None:
    calls: list[tuple[int, int]] = []

    class Dataset:
        def get_window(self, episode: int, start: int) -> dict:
            calls.append((episode, start))
            return {"episode_id": episode, "window_start": start}

    windows = list(iter_evaluation_windows(Dataset(), "overfit"))

    assert E0_OVERFIT_STARTS == (0, 64, 128, 192)
    assert calls == [(0, 0), (0, 64), (0, 128), (0, 192)]
    assert [(row["episode_id"], row["window_start"]) for row in windows] == calls
    assert [_window_id(row) for row in windows] == [
        "episode_0:s=0",
        "episode_0:s=64",
        "episode_0:s=128",
        "episode_0:s=192",
    ]


def test_overfit_enumeration_rejects_dataset_window_identity_leakage() -> None:
    class Dataset:
        def get_window(self, episode: int, start: int) -> dict:
            return {"episode_id": 1, "window_start": start}

    with pytest.raises(ValueError, match="overfit dataset returned unexpected window"):
        list(iter_evaluation_windows(Dataset(), "overfit"))


@pytest.mark.parametrize("method", ["B-VAE", "B0", "E1-A"])
def test_overfit_protocol_accepts_only_seed_zero_for_e0_methods(method: str) -> None:
    validate_sampling_protocol("overfit", method, [0])
    with pytest.raises(ValueError, match=r"sampling seed \[0\]"):
        validate_sampling_protocol("overfit", method, [1])
    with pytest.raises(ValueError, match=r"sampling seed \[0\]"):
        validate_sampling_protocol("overfit", method, [0, 1])


def test_overfit_protocol_does_not_open_arbitrary_training_evaluation() -> None:
    for method in ("B-Persistence", "E1-Z", "E1-S"):
        with pytest.raises(ValueError, match="only supports B-VAE, B0, and E1-A"):
            validate_sampling_protocol("overfit", method, [0])


def test_independent_window_partition_has_exact_union_no_overlap_and_unequal_tails() -> None:
    windows = [f"w{i}" for i in range(10)]

    shards = [list(iter_rank_windows(windows, rank=rank, world_size=4)) for rank in range(4)]

    assert shards == [
        [(0, "w0"), (4, "w4"), (8, "w8")],
        [(1, "w1"), (5, "w5"), (9, "w9")],
        [(2, "w2"), (6, "w6")],
        [(3, "w3"), (7, "w7")],
    ]
    assert sorted(index for shard in shards for index, _ in shard) == list(range(10))


def test_rank_manifest_merge_restores_window_then_seed_order() -> None:
    payloads = [
        {"rank": 0, "rows": [{"window_index": 4, "seed_index": 0}, {"window_index": 0, "seed_index": 1}]},
        {"rank": 1, "rows": [{"window_index": 1, "seed_index": 1}, {"window_index": 1, "seed_index": 0}]},
        {"rank": 2, "rows": [{"window_index": 0, "seed_index": 0}]},
    ]

    rows = merge_rank_payloads(payloads)

    assert [(row["window_index"], row["seed_index"]) for row in rows] == [
        (0, 0), (0, 1), (1, 0), (1, 1), (4, 0)
    ]


def test_rank_manifest_merge_rejects_duplicate_window_seed_work() -> None:
    duplicate = {"window_index": 2, "seed_index": 0}
    with pytest.raises(ValueError, match="duplicate independent-window result"):
        merge_rank_payloads([
            {"rank": 0, "rows": [duplicate]},
            {"rank": 1, "rows": [dict(duplicate)]},
        ])


def test_rank_manifest_merge_rejects_missing_rank_payload() -> None:
    with pytest.raises(ValueError, match=r"rank payloads.*\[0, 2\]"):
        merge_rank_payloads(
            [{"rank": 0, "rows": []}, {"rank": 2, "rows": []}],
            expected_world_size=3,
        )


def test_independent_parallelism_rejects_any_collective_generation_axis() -> None:
    valid = SimpleNamespace(dp_replicate=4, dp_shard=1, cp=1, cfgp=1, lb=1, enable_inference_mode=True)
    validate_independent_parallelism(valid)
    for field in ("dp_shard", "cp", "cfgp", "lb"):
        invalid = SimpleNamespace(**vars(valid))
        setattr(invalid, field, 2)
        with pytest.raises(ValueError, match="independent-window parallelism"):
            validate_independent_parallelism(invalid)


def test_rank_manifest_merge_rejects_missing_seed_from_cartesian_product() -> None:
    with pytest.raises(ValueError, match=r"missing=.*\(1, 1\)"):
        merge_rank_payloads(
            [{"rank": 0, "rows": [
                {"window_index": 0, "seed_index": 0, "window_id": "w0"},
                {"window_index": 0, "seed_index": 1, "window_id": "w0"},
                {"window_index": 1, "seed_index": 0, "window_id": "w1"},
            ]}],
            expected_window_count=2,
            expected_seed_count=2,
        )


def test_rank_manifest_merge_rejects_out_of_range_cartesian_key() -> None:
    with pytest.raises(ValueError, match=r"unexpected=.*\(2, 0\)"):
        merge_rank_payloads(
            [{"rank": 0, "rows": [
                {"window_index": 0, "seed_index": 0},
                {"window_index": 1, "seed_index": 0},
                {"window_index": 2, "seed_index": 0},
            ]}],
            expected_window_count=2,
            expected_seed_count=1,
        )


def test_rank_manifest_merge_rejects_two_ids_for_same_window_index() -> None:
    with pytest.raises(ValueError, match="window index 0 maps to multiple window IDs"):
        merge_rank_payloads([{"rank": 0, "rows": [
            {"window_index": 0, "seed_index": 0, "window_id": "w0"},
            {"window_index": 0, "seed_index": 1, "window_id": "other"},
        ]}])
