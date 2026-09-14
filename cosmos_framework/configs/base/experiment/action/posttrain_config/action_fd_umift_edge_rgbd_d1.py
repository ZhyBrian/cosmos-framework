# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Independent B-continuation and D1 recipes initialized from E3@3000."""

import copy

from hydra.core.config_store import ConfigStore

from cosmos_framework.configs.base.experiment.action.posttrain_config.action_fd_umift_edge_rgbd import (
    action_fd_umift_edge_rgbd_h5,
)


def _make_continuation(name: str):
    experiment = copy.deepcopy(action_fd_umift_edge_rgbd_h5)
    experiment.job.name = name
    experiment.scheduler.cycle_lengths = [1000]
    experiment.scheduler.f_min = [0.1]
    experiment.scheduler.warm_up_steps = [100]
    experiment.trainer.max_iter = 1000
    experiment.checkpoint.load_training_state = False
    experiment.checkpoint.save_iter = 250
    return experiment


action_fd_umift_edge_rgbd_b_continue = _make_continuation(
    "action_fd_umift_edge_rgbd_b_continue"
)

action_fd_umift_edge_rgbd_d1 = _make_continuation("action_fd_umift_edge_rgbd_d1")
action_fd_umift_edge_rgbd_d1.model._target_ = (
    "cosmos_framework.model.generator.umift_depth_model.UMIFTDepthModel"
)
action_fd_umift_edge_rgbd_d1.model.depth_aux_weight = (
    "${oc.env:UMIFT_DEPTH_AUX_WEIGHT}"
)

for _name, _experiment in (
    ("action_fd_umift_edge_rgbd_b_continue", action_fd_umift_edge_rgbd_b_continue),
    ("action_fd_umift_edge_rgbd_d1", action_fd_umift_edge_rgbd_d1),
):
    ConfigStore.instance().store(
        group="experiment",
        package="_global_",
        name=_name,
        node=_experiment,
    )


__all__ = [
    "action_fd_umift_edge_rgbd_b_continue",
    "action_fd_umift_edge_rgbd_d1",
]
