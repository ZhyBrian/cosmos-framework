#!/usr/bin/env bash
# Render the nine frozen long RGBD comparison videos for one selected arm.
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
SELECTION="$ARM_ROOT/selection/selected.json"
EVAL_LOGS="$ARM_ROOT/evaluation_logs"
VIDEO_LOGS="$ARM_ROOT/video_logs"
ROLLOUT_ROOT="$DEPTH_ROOT/rollout/$ARM"
OUTPUT_ROOT="$DEPTH_ROOT/cosmos_output/$ARM"
FONT=/data/cosmos_runs/e1_final_eval/video_assets_20260907/NotoSansCJK-Regular.ttc
JOB="action_fd_umift_edge_rgbd_${ARM}"
TOML="examples/toml/sft_config/${JOB}.toml"

test -f "$SELECTION"
test "$(cat "$EVAL_LOGS/evaluation_exit_code.txt")" = 0

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

mkdir "$VIDEO_LOGS"
git rev-parse HEAD > "$VIDEO_LOGS/video_source_commit.txt"
date -u +%FT%TZ > "$VIDEO_LOGS/videos_started_at_utc.txt"

run_stage() {
    local name="$1"
    shift
    date -u +%FT%TZ > "$VIDEO_LOGS/${name}_started_at_utc.txt"
    set +e
    "$@" > "$VIDEO_LOGS/${name}.log" 2>&1
    local rc=$?
    set -e
    printf '%s\n' "$rc" > "$VIDEO_LOGS/${name}_exit_code.txt"
    date -u +%FT%TZ > "$VIDEO_LOGS/${name}_finished_at_utc.txt"
    if [[ "$rc" -ne 0 ]]; then
        printf '%s\n' "$rc" > "$VIDEO_LOGS/videos_exit_code.txt"
        date -u +%FT%TZ > "$VIDEO_LOGS/videos_finished_at_utc.txt"
        exit "$rc"
    fi
}

for PERCENT in 0 33 67; do
    ROOT="$ROLLOUT_ROOT/start${PERCENT}"
    OUTPUT="$OUTPUT_ROOT/start${PERCENT}"
    run_stage "start${PERCENT}_prepare" \
        python -m examples.umift.depth_rollout prepare \
        --arm "$ARM" --selection "$SELECTION" --start-percent "$PERCENT" --root "$ROOT"
    run_stage "start${PERCENT}_base_infer" \
        torchrun --standalone --nproc-per-node=4 -m examples.umift.depth_rollout infer \
        --arm "$ARM" --root "$ROOT" --sft-toml "$TOML" --phase base
    run_stage "start${PERCENT}_finetuned_infer" \
        torchrun --standalone --nproc-per-node=4 -m examples.umift.depth_rollout infer \
        --arm "$ARM" --root "$ROOT" --sft-toml "$TOML" --phase finetuned
    run_stage "start${PERCENT}_score" \
        python -m examples.umift.depth_rollout score --arm "$ARM" --root "$ROOT"
    run_stage "start${PERCENT}_render" \
        python -m examples.umift.depth_rollout render \
        --arm "$ARM" --root "$ROOT" --output "$OUTPUT" --font "$FONT"
done

printf '0\n' > "$VIDEO_LOGS/videos_exit_code.txt"
date -u +%FT%TZ > "$VIDEO_LOGS/videos_finished_at_utc.txt"
