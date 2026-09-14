#!/usr/bin/env bash
# Resume one E3-Depth-Aux arm (b_continue or d1) in place for the registered
# +500-update extension (iter 1001..1500, constant 0.1x LR second cycle).
# Requires the stage-2 run to have finished at exactly iter_000001000.
set -euo pipefail

cd /data/cosmos-framework

ARM="${1:-}"
case "$ARM" in
    b_continue|d1) ;;
    *) echo "usage: $0 {b_continue|d1}" >&2; exit 2 ;;
esac

: "${BASE_CHECKPOINT_PATH:?set BASE_CHECKPOINT_PATH to the E3@3000 model checkpoint}"
if [[ "$ARM" == "d1" ]]; then
    : "${UMIFT_DEPTH_AUX_WEIGHT:?set UMIFT_DEPTH_AUX_WEIGHT to the calibrated stage-2 value}"
fi

DEPTH_ROOT=/data/cosmos_runs/e3_depth_aux_20260914
ARM_ROOT="$DEPTH_ROOT/$ARM"
TRAIN_ROOT="$ARM_ROOT/train"
LOG_ROOT="$ARM_ROOT/logs"
TOML="examples/toml/sft_config/action_fd_umift_edge_rgbd_${ARM}_ext.toml"
CKPT_DIR="$TRAIN_ROOT/cosmos3_action_fd_umift/action_sft/action_fd_umift_edge_rgbd_${ARM}/checkpoints"

# In-place resume guards: stage-2 evidence must exist and sit exactly at 1000.
test -d "$CKPT_DIR/iter_000001000"
test "$(cat "$CKPT_DIR/latest_checkpoint.txt")" = "iter_000001000"
test ! -e "$CKPT_DIR/iter_000001250"
test ! -e "$LOG_ROOT/train_ext_exit_code.txt"
test "$(cat "$LOG_ROOT/train_exit_code.txt")" = "0"

if [[ "$ARM" == "d1" ]]; then
    export CUDA_VISIBLE_DEVICES=0,1,2,3
else
    export CUDA_VISIBLE_DEVICES=4,5,6,7
fi
source examples/umift/a40_env.sh
export DATASET_PATH=/data/cosmos_datasets/umift_us_all_source_308f4d46.zarr
export OMP_NUM_THREADS=1 UMIFT_STAGE=e1 PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1

nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=memory.used --format=csv,noheader,nounits \
    | awk '$1 > 256 { busy=1 } END { exit busy }'
git rev-parse HEAD > "$LOG_ROOT/training_ext_source_commit.txt"
date -u +%FT%TZ > "$LOG_ROOT/train_ext_started_at_utc.txt"

set +e
OUTPUT_ROOT="$TRAIN_ROOT" IMAGINAIRE_OUTPUT_ROOT="$TRAIN_ROOT" COSMOS_EXIT_WITHOUT_FINALIZE=1 \
    torchrun --standalone --nproc-per-node=4 -m cosmos_framework.scripts.train \
    --sft-toml="$TOML" > "$LOG_ROOT/train_ext.log" 2>&1
TRAIN_RC=$?
set -e

printf '%s\n' "$TRAIN_RC" > "$LOG_ROOT/train_ext_exit_code.txt"
date -u +%FT%TZ > "$LOG_ROOT/train_ext_finished_at_utc.txt"
exit "$TRAIN_RC"
