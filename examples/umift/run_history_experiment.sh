#!/usr/bin/env bash
# Sequential four-GPU E2-H training, checkpoint selection, rollout, and rendering runner.
set -euo pipefail

cd /data/cosmos-framework
ROOT=/data/cosmos_runs/e2_history_20260908
BASE=/data/cosmos_models/Cosmos3-Edge-DCP-a9d944e
DATA=/data/cosmos_datasets/umift_us_all_source_308f4d46.zarr
PROTOCOL="$ROOT/protocol/selection.json"

test -f "$ROOT/protocol/model_gate.json"
test -f "$ROOT/evidence/finished_at_utc.txt"
test -f "$ROOT/evidence/data_audit.json"
test -f "$PROTOCOL"
test -f "$BASE/model/.metadata"
for H in 1 5 9 17; do
    test ! -e "$ROOT/H${H}"
done
test ! -e "$ROOT/cosmos_output"

python3 -c '
import json
from pathlib import Path
root = Path("/data/cosmos_runs/e2_history_20260908")
gate = json.loads((root / "protocol/model_gate.json").read_text())
audit = json.loads((root / "evidence/data_audit.json").read_text())
assert gate.get("passed") is True
assert audit.get("passed") is True
assert all(
    audit["configs"][str(history_frames)]["normalize_loss_by_active"] is False
    for history_frames in (1, 5, 9, 17)
)
'

export CUDA_VISIBLE_DEVICES=0,1,2,3
source examples/umift/a40_env.sh
export DATASET_PATH="$DATA"
export BASE_CHECKPOINT_PATH="$BASE"
export OMP_NUM_THREADS=1
export UMIFT_STAGE=e1

nvidia-smi -i 0,1,2,3 --query-gpu=memory.used --format=csv,noheader,nounits \
    | awk '$1 > 256 { busy=1 } END { exit busy }'

git rev-parse HEAD > "$ROOT/evidence/experiment_source_commit.txt"
date -u +%FT%TZ > "$ROOT/evidence/experiment_started_at_utc.txt"

for H in 1 5 9 17; do
    TOML="examples/toml/sft_config/action_fd_umift_edge_h${H}.toml"
    GROUP="$ROOT/H${H}"
    TRAIN_OUTPUT="$GROUP/train"
    CHECKPOINT_ROOT="$TRAIN_OUTPUT/cosmos3_action_fd_umift/action_sft/action_fd_umift_edge_h${H}/checkpoints"
    SELECTION="$GROUP/selection"
    mkdir -p "$GROUP/logs" "$SELECTION"
    date -u +%FT%TZ > "$GROUP/logs/train_started_at_utc.txt"

    set +e
    PYTHONPATH=. OUTPUT_ROOT="$TRAIN_OUTPUT" IMAGINAIRE_OUTPUT_ROOT="$TRAIN_OUTPUT" \
        COSMOS_EXIT_WITHOUT_FINALIZE=1 \
        torchrun --standalone --nproc-per-node=4 -m cosmos_framework.scripts.train \
        --sft-toml="$TOML" > "$GROUP/logs/train.log" 2>&1
    TRAIN_RC=$?
    set -e
    printf '%s\n' "$TRAIN_RC" > "$GROUP/logs/train_exit_code.txt"
    date -u +%FT%TZ > "$GROUP/logs/train_finished_at_utc.txt"
    if [[ "$TRAIN_RC" != 0 ]]; then
        echo "H=${H} training failed; inspect $GROUP/logs/train.log" >&2
        exit "$TRAIN_RC"
    fi

    for ITERATION in 500 1000 1500 2000 2500 3000; do
        ITER=$(printf '%09d' "$ITERATION")
        CHECKPOINT="$CHECKPOINT_ROOT/iter_${ITER}/model"
        CANDIDATE="$SELECTION/iter_${ITER}"
        test -f "$CHECKPOINT/.metadata"
        test ! -e "$CANDIDATE"
        PYTHONPATH=. torchrun --standalone --nproc-per-node=4 \
            -m examples.umift.history_selection infer \
            --protocol "$PROTOCOL" --history-frames "$H" --iteration "$ITERATION" \
            --checkpoint "$CHECKPOINT" --sft-toml "$TOML" --output "$CANDIDATE" \
            > "$GROUP/logs/selection_iter_${ITER}_infer.log" 2>&1
        PYTHONPATH=. python -m examples.umift.history_selection score \
            --protocol "$PROTOCOL" --input "$CANDIDATE" \
            > "$GROUP/logs/selection_iter_${ITER}_score.log" 2>&1
    done
    PYTHONPATH=. python -m examples.umift.history_selection choose \
        --protocol "$PROTOCOL" --input "$SELECTION" --history-frames "$H" \
        > "$GROUP/logs/selection_choose.log" 2>&1

    for PERCENT in 0 33 67; do
        ROLLOUT_ROOT="$GROUP/long/start${PERCENT}"
        PYTHONPATH=. python -m examples.umift.prepare_history_rollout \
            --selection "$SELECTION/selected.json" --start-percent "$PERCENT" \
            --root "$ROLLOUT_ROOT" > "$GROUP/logs/start${PERCENT}_prepare.log" 2>&1

        if [[ "$PERCENT" == 0 ]]; then
            PYTHONPATH=. torchrun --standalone --nproc-per-node=4 \
                -m examples.umift.history_rollout --root "$ROLLOUT_ROOT" \
                --history-frames "$H" --sft-toml "$TOML" --phase finetuned --max-chunks 2 \
                > "$GROUP/logs/start0_finetuned_smoke.log" 2>&1
        fi
        for PHASE in base finetuned; do
            PYTHONPATH=. torchrun --standalone --nproc-per-node=4 \
                -m examples.umift.history_rollout --root "$ROLLOUT_ROOT" \
                --history-frames "$H" --sft-toml "$TOML" --phase "$PHASE" \
                > "$GROUP/logs/start${PERCENT}_${PHASE}.log" 2>&1
        done
        PYTHONPATH=. python -m examples.umift.render_history_comparison \
            --root "$ROLLOUT_ROOT" --history-frames "$H" \
            --output-dir "$ROOT/cosmos_output/H${H}/start${PERCENT}" \
            > "$GROUP/logs/start${PERCENT}_render.log" 2>&1
    done
    date -u +%FT%TZ > "$GROUP/logs/finished_at_utc.txt"
done

date -u +%FT%TZ > "$ROOT/evidence/experiment_finished_at_utc.txt"
