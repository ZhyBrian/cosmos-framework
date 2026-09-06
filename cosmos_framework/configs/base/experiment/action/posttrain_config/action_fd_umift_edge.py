# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Cosmos3-Edge forward dynamics for the UMI-FT RGB/action E1 protocol."""

import copy

from hydra.core.config_store import ConfigStore

from cosmos_framework.configs.base.experiment.sft.models.edge_model_config import EDGE_MODEL_CONFIG
from cosmos_framework.data.generator.action.datasets.umift_zarr_dataset import (
    get_umift_packing_dataloader,
    get_umift_zarr_sft_dataset,
)
from cosmos_framework.data.generator.processors import build_processor
from cosmos_framework.data.generator.joint_dataloader import RankPartitionedDataLoader
from cosmos_framework.utils.lazy_config import LazyCall as L
from cosmos_framework.utils.lazy_config import LazyDict


def _make_model_config() -> dict:
    """Combine the released Edge graph with the Nano FD task deltas."""
    cfg = copy.deepcopy(EDGE_MODEL_CONFIG)
    cfg["action_gen"] = True
    cfg["sound_gen"] = False
    cfg["joint_attn_implementation"] = "two_way"
    cfg["resolution"] = "256"
    cfg["max_num_tokens_after_packing"] = 45056
    cfg["diffusion_expert_config"]["load_weights_from_pretrained"] = False
    cfg["tokenizer"]["encode_exact_durations"] = [17]
    cfg["activation_checkpointing"]["mode"] = "selective"
    cfg["compile"]["enabled"] = False
    cfg["ema"]["enabled"] = False
    cfg["parallelism"]["data_parallel_shard_degree"] = 4
    cfg["parallelism"]["fsdp_master_dtype"] = "float32"
    cfg["parallelism"]["fsdp_reduce_dtype"] = "bfloat16"
    # Direct local artifact: unlike build_processor_lazy(repository=...), this
    # never invokes the HF CLI. The launcher pins the snapshot directory name
    # to the approved immutable Edge revision.
    cfg["vlm_config"]["tokenizer"] = L(build_processor)(
        tokenizer_type="${oc.env:EDGE_HF_SNAPSHOT_PATH}",
    )
    return cfg


action_fd_umift_edge = LazyDict(
    dict(
        defaults=[
            {"override /data_train": None},
            {"override /data_val": None},
            {"override /model": "mot_fsdp"},
            {"override /optimizer": "fusedadamw"},
            {"override /scheduler": "lambdalinear"},
            {"override /tokenizer": "wan2pt2_tokenizer"},
            {"override /sound_tokenizer": None},
            {"override /vlm_config": None},
            {"override /checkpoint": "gcp"},
            {"override /callbacks": ["basic", "optimization", "job_monitor", "training_stats"]},
            {"override /ema": "power"},
            {"override /ckpt_type": "dcp"},
            "_self_",
        ],
        job=dict(
            project="cosmos3_action_fd_umift",
            group="action_sft",
            name="action_fd_umift_edge_e1",
            wandb_mode="disabled",
        ),
        model=dict(config=_make_model_config()),
        optimizer=dict(
            betas=[0.9, 0.99],
            eps=1.0e-08,
            fused=True,
            keys_to_select=[
                "moe_gen",
                "time_embedder",
                "vae2llm",
                "llm2vae",
                "action2llm",
                "llm2action",
                "action_modality_embed",
            ],
            lr=1.0e-05,
            lr_multipliers={
                "action2llm": 5.0,
                "llm2action": 5.0,
                "action_modality_embed": 5.0,
            },
            optimizer_type="FusedAdam",
            weight_decay=0.05,
        ),
        scheduler=dict(
            cycle_lengths=[1000],
            f_max=[1.0],
            f_min=[0.1],
            f_start=[0.0],
            lr_scheduler_type="LambdaLinear",
            verbosity_interval=0,
            warm_up_steps=[100],
        ),
        trainer=dict(
            distributed_parallelism="fsdp",
            grad_accum_iter=4,
            logging_iter=10,
            max_iter=1000,
            max_val_iter=None,
            run_validation=False,
            run_validation_on_start=False,
            save_zero_checkpoint=False,
            seed=42,
            timeout_period=999999999,
            validation_iter=100,
            compile_config=dict(recompile_limit=8, use_duck_shape=False),
            cudnn=dict(benchmark=True, deterministic=False),
            ddp=dict(broadcast_buffers=True, find_unused_parameters=False, static_graph=True),
            grad_scaler_args=dict(enabled=False),
            straggler_detection=dict(enabled=True, report_freq=10),
            callbacks=dict(
                compile_tokenizer=dict(enabled=False, warmup_resolutions=None),
                dataloader_speed=dict(every_n=10, save_s3=False, step_size=1),
                device_monitor=dict(every_n=10, log_memory_detail=True, save_s3=False, step_size=1),
                grad_clip=dict(clip_norm=1.0, force_finite=True),
                heart_beat=dict(every_n=20, save_s3=False, step_size=1, update_interval_in_minute=20),
                iter_speed=dict(every_n=10, hit_thres=50, save_s3=False, save_s3_every_log_n=500),
                low_precision=dict(update_iter=1),
                manual_gc=dict(every_n=20, gc_level=1, warm_up=1),
                norm_monitor=dict(every_n=10),
                param_count=dict(save_s3=False),
                sigma_loss_analysis=dict(every_n=250, every_n_viz=250, save_s3=False),
                skip_nan_step=dict(max_consecutive_nan=1),
                training_stats=dict(log_freq=10),
            ),
        ),
        checkpoint=dict(
            dcp_async_mode_enabled=False,
            enable_gcs_patch_in_boto3=True,
            keys_not_to_resume=[],
            keys_to_skip_loading=[],
            load_ema_to_reg=False,
            load_from_object_store=dict(bucket="", credentials="", enabled=False),
            save_to_object_store=dict(bucket="", credentials="", enabled=False),
            load_path="???",
            load_training_state=False,
            only_load_scheduler_state=False,
            save_iter=250,
            strict_resume=True,
            verbose=True,
        ),
        dataloader_train=L(get_umift_packing_dataloader)(
            audio_sample_rate=48000,
            dataset_name="umift_zarr",
            max_samples_per_batch=1,
            max_sequence_length=None,
            patch_spatial="${model.config.diffusion_expert_config.patch_spatial}",
            sound_latent_fps="${model.config.sound_latent_fps}",
            tokenizer_spatial_compression_factor="${model.config.tokenizer.spatial_compression_factor}",
            tokenizer_temporal_compression_factor="${model.config.tokenizer.temporal_compression_factor}",
            dataloader=L(RankPartitionedDataLoader)(
                batch_size=1,
                in_order=True,
                num_workers=0,
                persistent_workers=False,
                pin_memory=True,
                sampler=None,
                datasets=dict(
                    umift=dict(
                        ratio=1,
                        dataset=L(get_umift_zarr_sft_dataset)(
                            zarr_path="${oc.env:DATASET_PATH}",
                            split="train",
                            stage="${oc.env:UMIFT_STAGE,e1}",
                            seed=42,
                            resolution="256",
                            fps=15.0,
                            mode="forward_dynamics",
                            tokenizer_config="${model.config.vlm_config.tokenizer}",
                            max_action_dim="${model.config.max_action_dim}",
                        ),
                    )
                ),
            ),
        ),
        dataloader_val=None,
        upload_reproducible_setup=False,
    ),
    flags={"allow_objects": True},
)


ConfigStore.instance().store(
    group="experiment",
    package="_global_",
    name="action_fd_umift_edge",
    node=action_fd_umift_edge,
)
