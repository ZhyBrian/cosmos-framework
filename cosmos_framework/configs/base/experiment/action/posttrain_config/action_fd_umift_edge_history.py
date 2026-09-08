# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Cosmos3-Edge UMI-FT forward dynamics with RGB history prefixes."""

import copy

from hydra.core.config_store import ConfigStore

from cosmos_framework.configs.base.experiment.action.posttrain_config.action_fd_umift_edge import (
    action_fd_umift_edge_refit,
)
from cosmos_framework.data.generator.action.datasets.umift_history_dataset import (
    get_umift_history_sft_dataset,
)
from cosmos_framework.utils.lazy_config import LazyCall as L


def _make_history_experiment(history_frames: int):
    experiment = copy.deepcopy(action_fd_umift_edge_refit)
    experiment.job.name = f"action_fd_umift_edge_h{history_frames}"
    experiment.scheduler.cycle_lengths = [3000]
    experiment.scheduler.warm_up_steps = [100]
    experiment.trainer.max_iter = 3000
    experiment.checkpoint.save_iter = 500
    experiment.model.config.tokenizer.encode_exact_durations = [history_frames + 16]

    # Preserve the E1 recipe; checkpoint selection compares the same 16 future RGB frames.
    # With False, clean history remains in the vision-loss denominator and longer H is down-weighted.
    experiment.model.config.rectified_flow_training_config.normalize_loss_by_active = False

    experiment.dataloader_train.dataloader.datasets.umift.dataset = L(get_umift_history_sft_dataset)(
        zarr_path="${oc.env:DATASET_PATH}",
        split="refit_train",
        stage="${oc.env:UMIFT_STAGE,e1}",
        seed=42,
        resolution="256",
        fps=15.0,
        history_frames=history_frames,
        mode="forward_dynamics",
        tokenizer_config="${model.config.vlm_config.tokenizer}",
        max_action_dim="${model.config.max_action_dim}",
    )
    return experiment


for _history_frames in (1, 5, 9, 17):
    _name = f"action_fd_umift_edge_h{_history_frames}"
    _experiment = _make_history_experiment(_history_frames)
    globals()[_name] = _experiment
    ConfigStore.instance().store(
        group="experiment",
        package="_global_",
        name=_name,
        node=_experiment,
    )


__all__ = [f"action_fd_umift_edge_h{history_frames}" for history_frames in (1, 5, 9, 17)]
