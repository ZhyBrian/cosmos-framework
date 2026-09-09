#!/usr/bin/env bash
# E3-only fixed-candidate selection and complete RGBD suffix evaluation.
set -euo pipefail
cd /data/cosmos-framework
E3_ROOT=/data/cosmos_runs/e3_rgbd_20260909
E3_PROTOCOL="$E3_ROOT/protocol/selection.json"
E3_SELECTION="$E3_ROOT/selection"
E3_LOGS="$E3_ROOT/logs"
E3_TOML=examples/toml/sft_config/action_fd_umift_edge_rgbd_h5.toml
E3_CHECKPOINTS="$E3_ROOT/train/cosmos3_action_fd_umift/action_sft/action_fd_umift_edge_rgbd_h5/checkpoints"
E3_FONT=/data/cosmos_runs/e1_final_eval/video_assets_20260907/NotoSansCJK-Regular.ttc
test "$(cat "$E3_LOGS/train_exit_code.txt")" = 0
test -f "$E3_CHECKPOINTS/iter_000003000/model/.metadata"
test -f "$E3_ROOT/evidence/rollout_cpu_tests_passed.txt"
test -f "$E3_PROTOCOL"
test -f "$E3_FONT"
test ! -e "$E3_SELECTION"
test ! -e "$E3_LOGS/evaluation_started_at_utc.txt"
export CUDA_VISIBLE_DEVICES=0,1,2,3
source examples/umift/a40_env.sh
export DATASET_PATH=/data/cosmos_datasets/umift_us_all_source_308f4d46.zarr
export BASE_CHECKPOINT_PATH=/data/cosmos_models/Cosmos3-Edge-DCP-a9d944e
export PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 UMIFT_STAGE=e1
python - "$E3_PROTOCOL" <<'PY'
import sys
from pathlib import Path
from examples.umift.rgbd_selection import load_protocol
load_protocol(Path(sys.argv[1]))
PY
nvidia-smi -i 0,1,2,3 --query-gpu=memory.used --format=csv,noheader,nounits \
    | awk '$1 > 256 { busy=1 } END { exit busy }'
mkdir "$E3_SELECTION"
git rev-parse HEAD > "$E3_LOGS/evaluation_source_commit.txt"
date -u +%FT%TZ > "$E3_LOGS/evaluation_started_at_utc.txt"
for E3_STEP in 500 1000 1500 2000 2500 3000; do
    E3_ITER=$(printf '%09d' "$E3_STEP")
    E3_CANDIDATE="$E3_SELECTION/iter_${E3_ITER}"
    E3_CHECKPOINT="$E3_CHECKPOINTS/iter_${E3_ITER}/model"
    test -f "$E3_CHECKPOINT/.metadata"
    torchrun --standalone --nproc-per-node=4 -m examples.umift.rgbd_selection infer \
        --protocol "$E3_PROTOCOL" --iteration "$E3_STEP" --checkpoint "$E3_CHECKPOINT" \
        --sft-toml "$E3_TOML" --output "$E3_CANDIDATE" \
        > "$E3_LOGS/selection_iter_${E3_ITER}_infer.log" 2>&1
    python -m examples.umift.rgbd_selection score --protocol "$E3_PROTOCOL" --input "$E3_CANDIDATE" \
        > "$E3_LOGS/selection_iter_${E3_ITER}_score.log" 2>&1
done
python -m examples.umift.rgbd_selection choose --protocol "$E3_PROTOCOL" --input "$E3_SELECTION" \
    > "$E3_LOGS/selection_choose.log" 2>&1
for E3_PERCENT in 0 33 67; do
    E3_ROLLOUT="$E3_ROOT/long/start${E3_PERCENT}"
    python -m examples.umift.prepare_rgbd_rollout --selection "$E3_SELECTION/selected.json" \
        --start-percent "$E3_PERCENT" --root "$E3_ROLLOUT" \
        > "$E3_LOGS/start${E3_PERCENT}_prepare.log" 2>&1
    if [[ "$E3_PERCENT" == 0 ]]; then
        torchrun --standalone --nproc-per-node=4 -m examples.umift.rgbd_rollout \
            --root "$E3_ROLLOUT" --sft-toml "$E3_TOML" --phase finetuned --max-chunks 2 \
            > "$E3_LOGS/start0_finetuned_smoke.log" 2>&1
    fi
    for E3_PHASE in base finetuned; do
        torchrun --standalone --nproc-per-node=4 -m examples.umift.rgbd_rollout \
            --root "$E3_ROLLOUT" --sft-toml "$E3_TOML" --phase "$E3_PHASE" \
            > "$E3_LOGS/start${E3_PERCENT}_${E3_PHASE}.log" 2>&1
    done
    python -m examples.umift.score_rgbd_rollout --root "$E3_ROLLOUT" \
        > "$E3_LOGS/start${E3_PERCENT}_score.log" 2>&1
    python -m examples.umift.render_rgbd_comparison --root "$E3_ROLLOUT" \
        --output "$E3_ROOT/cosmos_output/start${E3_PERCENT}" --font "$E3_FONT" \
        > "$E3_LOGS/start${E3_PERCENT}_render.log" 2>&1
done
date -u +%FT%TZ > "$E3_LOGS/evaluation_finished_at_utc.txt"
