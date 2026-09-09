#!/usr/bin/env bash
# Fixed-budget E3 joint RGBD training; candidate evaluation is a separate frozen-protocol step.
set -euo pipefail
cd /data/cosmos-framework
E3_ROOT=/data/cosmos_runs/e3_rgbd_20260909
E3_TRAIN="$E3_ROOT/train"
E3_LOGS="$E3_ROOT/logs"
test -f "$E3_ROOT/preflight/finished_at_utc.txt"
test ! -e "$E3_TRAIN"
test ! -e "$E3_LOGS/train_started_at_utc.txt"
mkdir -p "$E3_LOGS"
export CUDA_VISIBLE_DEVICES=0,1,2,3
source examples/umift/a40_env.sh
export DATASET_PATH=/data/cosmos_datasets/umift_us_all_source_308f4d46.zarr
export BASE_CHECKPOINT_PATH=/data/cosmos_models/Cosmos3-Edge-DCP-a9d944e
export OMP_NUM_THREADS=1 UMIFT_STAGE=e1 PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1
nvidia-smi -i 0,1,2,3 --query-gpu=memory.used --format=csv,noheader,nounits \
    | awk '$1 > 256 { busy=1 } END { exit busy }'
git rev-parse HEAD > "$E3_LOGS/training_source_commit.txt"
date -u +%FT%TZ > "$E3_LOGS/train_started_at_utc.txt"
set +e
OUTPUT_ROOT="$E3_TRAIN" IMAGINAIRE_OUTPUT_ROOT="$E3_TRAIN" COSMOS_EXIT_WITHOUT_FINALIZE=1 \
    torchrun --standalone --nproc-per-node=4 -m cosmos_framework.scripts.train \
    --sft-toml=examples/toml/sft_config/action_fd_umift_edge_rgbd_h5.toml \
    > "$E3_LOGS/train.log" 2>&1
E3_RC=$?
set -e
printf '%s\n' "$E3_RC" > "$E3_LOGS/train_exit_code.txt"
date -u +%FT%TZ > "$E3_LOGS/train_finished_at_utc.txt"
exit "$E3_RC"
