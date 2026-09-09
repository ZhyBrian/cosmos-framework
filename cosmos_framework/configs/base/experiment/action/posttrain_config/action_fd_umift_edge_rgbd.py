# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Cosmos3-Edge UMI-FT H5 forward dynamics with a joint RGBD canvas."""

import copy

from hydra.core.config_store import ConfigStore

from cosmos_framework.configs.base.experiment.action.posttrain_config.action_fd_umift_edge_history import (
    action_fd_umift_edge_h5,
)
from cosmos_framework.data.generator.action.datasets.umift_rgbd_dataset import (
    get_umift_rgbd_packing_dataloader,
    get_umift_rgbd_sft_dataset,
)


action_fd_umift_edge_rgbd_h5 = copy.deepcopy(action_fd_umift_edge_h5)
action_fd_umift_edge_rgbd_h5.job.name = "action_fd_umift_edge_rgbd_h5"
action_fd_umift_edge_rgbd_h5.dataloader_train._target_ = (
    get_umift_rgbd_packing_dataloader
)
action_fd_umift_edge_rgbd_h5.dataloader_train.dataloader.datasets.umift.dataset._target_ = (
    get_umift_rgbd_sft_dataset
)

ConfigStore.instance().store(
    group="experiment",
    package="_global_",
    name="action_fd_umift_edge_rgbd_h5",
    node=action_fd_umift_edge_rgbd_h5,
)


__all__ = ["action_fd_umift_edge_rgbd_h5"]
