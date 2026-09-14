#!/usr/bin/env bash
# Evaluate the registered +500-update extension candidates (1250/1500) for one
# arm and re-select over all six steps; writes selected_ext.json alongside the
# untouched stage-2 selected.json.
set -euo pipefail

cd /data/cosmos-framework

ARM="${1:-}"
case "$ARM" in
    b_continue|d1) ;;
    *) echo "usage: $0 {b_continue|d1}" >&2; exit 2 ;;
esac

: "${BASE_CHECKPOINT_PATH:?set BASE_CHECKPOINT_PATH to the original E3@3000 iteration root}"
test -f "$BASE_CHECKPOINT_PATH/model/.metadata"

DEPTH_ROOT=/data/cosmos_runs/e3_depth_aux_20260914
ARM_ROOT="$DEPTH_ROOT/$ARM"
TRAIN_LOGS="$ARM_ROOT/logs"
PROTOCOL="$ARM_ROOT/protocol/selection.json"
SELECTION="$ARM_ROOT/selection"
EVAL_LOGS="$ARM_ROOT/evaluation_logs"
JOB="action_fd_umift_edge_rgbd_${ARM}"
CHECKPOINTS="$ARM_ROOT/train/cosmos3_action_fd_umift/action_sft/$JOB/checkpoints"
TOML="examples/toml/sft_config/${JOB}.toml"

test "$(cat "$TRAIN_LOGS/train_ext_exit_code.txt")" = 0
test -f "$PROTOCOL"
test -f "$TOML"
test -d "$SELECTION"
test ! -e "$SELECTION/selected_ext.json"
for STEP in 1250 1500; do
    ITERATION="$(printf '%09d' "$STEP")"
    test -f "$CHECKPOINTS/iter_${ITERATION}/model/.metadata"
    test ! -e "$SELECTION/iter_${ITERATION}"
done

if [[ "$ARM" == "d1" ]]; then
    export CUDA_VISIBLE_DEVICES=0,1,2,3
else
    export CUDA_VISIBLE_DEVICES=4,5,6,7
fi

source examples/umift/a40_env.sh
export DATASET_PATH=/data/cosmos_datasets/umift_us_all_source_308f4d46.zarr
export OMP_NUM_THREADS=1 UMIFT_STAGE=e1 PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1

if [[ "$ARM" == "d1" ]]; then
    D1_CALIBRATION_PATH="${D1_CALIBRATION_PATH:-$DEPTH_ROOT/preflight/calibration.json}"
    test -f "$D1_CALIBRATION_PATH"
    UMIFT_DEPTH_AUX_WEIGHT="$(python - "$D1_CALIBRATION_PATH" "$BASE_CHECKPOINT_PATH" <<'PY'
import json
import math
import sys

record = json.load(open(sys.argv[1]))
weight = float(record["lambda_depth"])
if record.get("protocol") != "d1-train-gradient-calibration-v1":
    raise ValueError("invalid D1 calibration protocol")
if record.get("checkpoint") != sys.argv[2]:
    raise ValueError("D1 calibration used a different E3 checkpoint")
if not math.isfinite(weight) or weight <= 0:
    raise ValueError("D1 calibration lambda must be finite and positive")
print(repr(weight))
PY
    )"
    export UMIFT_DEPTH_AUX_WEIGHT
fi

python - "$PROTOCOL" "$ARM" <<'PY'
import sys
from pathlib import Path
from examples.umift.depth_selection import load_protocol

load_protocol(Path(sys.argv[1]), sys.argv[2])
PY

nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=memory.used --format=csv,noheader,nounits \
    | awk '$1 > 256 { busy=1 } END { exit busy }'

git rev-parse HEAD > "$EVAL_LOGS/evaluation_ext_source_commit.txt"
date -u +%FT%TZ > "$EVAL_LOGS/evaluation_ext_started_at_utc.txt"

run_step() {
    local name="$1"
    shift
    date -u +%FT%TZ > "$EVAL_LOGS/${name}_started_at_utc.txt"
    set +e
    "$@" > "$EVAL_LOGS/${name}.log" 2>&1
    local rc=$?
    set -e
    printf '%s\n' "$rc" > "$EVAL_LOGS/${name}_exit_code.txt"
    date -u +%FT%TZ > "$EVAL_LOGS/${name}_finished_at_utc.txt"
    if [[ "$rc" -ne 0 ]]; then
        printf '%s\n' "$rc" > "$EVAL_LOGS/evaluation_ext_exit_code.txt"
        date -u +%FT%TZ > "$EVAL_LOGS/evaluation_ext_finished_at_utc.txt"
        exit "$rc"
    fi
}

for STEP in 1250 1500; do
    ITERATION="$(printf '%09d' "$STEP")"
    CANDIDATE="$SELECTION/iter_${ITERATION}"
    CHECKPOINT="$CHECKPOINTS/iter_${ITERATION}/model"
    run_step "iter_${ITERATION}_infer" \
        torchrun --standalone --nproc-per-node=4 -m examples.umift.depth_selection infer \
        --arm "$ARM" --protocol "$PROTOCOL" --iteration "$STEP" \
        --checkpoint "$CHECKPOINT" --sft-toml "$TOML" --output "$CANDIDATE"
    run_step "iter_${ITERATION}_score" \
        python -m examples.umift.depth_selection score \
        --arm "$ARM" --protocol "$PROTOCOL" --input "$CANDIDATE"
done

run_step select_ext python -m examples.umift.depth_selection select \
    --arm "$ARM" --protocol "$PROTOCOL" --input "$SELECTION" \
    --iterations 250,500,750,1000,1250,1500 --output-name selected_ext.json

printf '0\n' > "$EVAL_LOGS/evaluation_ext_exit_code.txt"
date -u +%FT%TZ > "$EVAL_LOGS/evaluation_ext_finished_at_utc.txt"
