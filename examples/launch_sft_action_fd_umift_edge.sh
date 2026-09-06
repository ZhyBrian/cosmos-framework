#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# Usage: STAGE=smoke|overfit|e1 RUN_MODE=warmstart|resume bash examples/launch_sft_action_fd_umift_edge.sh
# smoke/overfit/e1 mean 10/200/1000 optimizer updates. With accumulation=4 and
# four ranks, each update consumes 16 clips globally. The E1 scheduler remains
# fixed at warmup=100, total=1000 for every profile so smokes exercise the same LR.

set -uo pipefail

TOML_FILE="examples/toml/sft_config/action_fd_umift_edge.toml"
: "${DATASET_PATH:=/data/cosmos_datasets/umift_us_all_source_308f4d46.zarr}"
: "${BASE_CHECKPOINT_PATH:=/data/cosmos_models/Cosmos3-Edge-DCP-a9d944e}"
: "${WAN_VAE_PATH:=/data/cosmos_models/Wan2.2-VAE-921dbaf/Wan2.2_VAE.pth}"
: "${EDGE_HF_SNAPSHOT_PATH:=/data/cosmos_models/Cosmos3-Edge/snapshots/a9d944e2c6a1bf9f48b92ad16348e70c5f1836ba}"
: "${OUTPUT_ROOT:=/data/cosmos_runs/umift_edge_fd}"
IMAGINAIRE_OUTPUT_ROOT="$OUTPUT_ROOT"
: "${NPROC_PER_NODE:=4}"
: "${STAGE:=e1}"
: "${RUN_MODE:=warmstart}"
: "${I4_ATTN_BACKENDS:=natten}"
EDGE_REVISION="a9d944e2c6a1bf9f48b92ad16348e70c5f1836ba"

[[ "$EDGE_HF_SNAPSHOT_PATH" = /* && -d "$EDGE_HF_SNAPSHOT_PATH" ]] || { echo "ERROR: EDGE_HF_SNAPSHOT_PATH must be an existing absolute local snapshot" >&2; exit 2; }
[[ "$(basename "$EDGE_HF_SNAPSHOT_PATH")" == "$EDGE_REVISION" ]] || { echo "ERROR: Edge snapshot must be pinned to revision $EDGE_REVISION: $EDGE_HF_SNAPSHOT_PATH" >&2; exit 2; }

case "$STAGE" in
    smoke) MAX_UPDATES=10; SAVE_ITER=5 ;;
    overfit) MAX_UPDATES=200; SAVE_ITER=200 ;;
    e1) MAX_UPDATES=1000; SAVE_ITER=250 ;;
    *) echo "ERROR: STAGE must be smoke, overfit, or e1; got: $STAGE" >&2; exit 2 ;;
esac

RUN_DIR="$OUTPUT_ROOT/cosmos3_action_fd_umift/action_sft/action_fd_umift_edge_${STAGE}"
LATEST="$RUN_DIR/checkpoints/latest_checkpoint.txt"
case "$RUN_MODE" in
    warmstart)
        [[ ! -e "$LATEST" ]] || { echo "ERROR: warmstart refused because a same-job checkpoint exists: $LATEST; use RUN_MODE=resume" >&2; exit 2; }
        ;;
    resume)
        [[ -f "$LATEST" ]] || { echo "ERROR: resume requested but no same-job checkpoint exists: $LATEST" >&2; exit 2; }
        ;;
    *) echo "ERROR: RUN_MODE must be warmstart or resume; got: $RUN_MODE" >&2; exit 2 ;;
esac

export DATASET_PATH BASE_CHECKPOINT_PATH WAN_VAE_PATH EDGE_HF_SNAPSHOT_PATH OUTPUT_ROOT IMAGINAIRE_OUTPUT_ROOT NPROC_PER_NODE
export UMIFT_STAGE="$STAGE" I4_ATTN_BACKENDS

TAIL_OVERRIDES=(
    "job.name=action_fd_umift_edge_${STAGE}"
    "trainer.max_iter=$MAX_UPDATES"
    "checkpoint.save_iter=$SAVE_ITER"
    ${EXTRA_TAIL_OVERRIDES:-}
)

source "$(dirname "${BASH_SOURCE[0]}")/_sft_launcher_common.sh"
