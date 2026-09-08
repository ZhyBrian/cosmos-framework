#!/usr/bin/env bash
# Sequential A40 preflight for E2 history data, model loading, short training, and 24 GiB proxy.
set -euo pipefail

cd /data/cosmos-framework
ROOT=/data/cosmos_runs/e2_history_20260908
BASE=/data/cosmos_models/Cosmos3-Edge-DCP-a9d944e
DATA=/data/cosmos_datasets/umift_us_all_source_308f4d46.zarr

test ! -e "$ROOT/preflight"
test ! -e "$ROOT/protocol"
test -d "$DATA"
test -f "$BASE/model/.metadata"
mkdir -p "$ROOT/evidence" "$ROOT/protocol" "$ROOT/preflight"

export CUDA_VISIBLE_DEVICES=0,1,2,3
source examples/umift/a40_env.sh
export DATASET_PATH="$DATA"
export BASE_CHECKPOINT_PATH="$BASE"
export OMP_NUM_THREADS=1
export UMIFT_STAGE=e1

git rev-parse HEAD > "$ROOT/evidence/source_commit.txt"
date -u +%FT%TZ > "$ROOT/evidence/started_at_utc.txt"

TESTS=(
    cosmos_framework/data/generator/action/datasets/umift_history_dataset_test.py
    examples/umift/history_contract_test.py
    examples/umift/history_selection_test.py
    examples/umift/history_rollout_test.py
    examples/umift/render_history_comparison_test.py
)
if [[ -f examples/umift/history_infer_test.py ]]; then
    TESTS+=(examples/umift/history_infer_test.py)
fi
PYTHONPATH=. python -m pytest -c /dev/null --noconftest "${TESTS[@]}" \
    > "$ROOT/evidence/cpu_tests.log" 2>&1

PYTHONPATH=. python -m examples.umift.audit_history_data \
    --zarr "$DATA" --output "$ROOT/evidence/data_audit.json" \
    > "$ROOT/evidence/data_audit.log" 2>&1
PYTHONPATH=. python -m examples.umift.history_selection freeze \
    --zarr "$DATA" --output "$ROOT/protocol/selection.json" \
    > "$ROOT/evidence/history_selection.log" 2>&1

nvidia-smi -i 0,1,2,3 --query-gpu=memory.used --format=csv,noheader,nounits \
    | awk '$1 > 256 { busy=1 } END { exit busy }'

for H in 1 5 9 17; do
    HH=$(printf '%02d' "$H")
    TOML="examples/toml/sft_config/action_fd_umift_edge_h${H}.toml"
    MODEL_OUTPUT="$ROOT/preflight/model_H${HH}"
    TRAIN_OUTPUT="$ROOT/preflight/train_H${HH}"
    test -f "$TOML"
    test ! -e "$MODEL_OUTPUT"
    test ! -e "$TRAIN_OUTPUT"

    CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=. \
        torchrun --standalone --nproc-per-node=4 -m examples.umift.check_history_model \
        --zarr "$DATA" --sft-toml "$TOML" --checkpoint "$BASE/model" \
        --history-frames "$H" --output "$MODEL_OUTPUT" \
        > "$ROOT/evidence/model_H${HH}.log" 2>&1

    set +e
    CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=. \
        OUTPUT_ROOT="$TRAIN_OUTPUT" IMAGINAIRE_OUTPUT_ROOT="$TRAIN_OUTPUT" \
        COSMOS_EXIT_WITHOUT_FINALIZE=1 \
        torchrun --standalone --nproc-per-node=4 -m cosmos_framework.scripts.train \
        --sft-toml="$TOML" -- \
        trainer.max_iter=3 checkpoint.save_iter=3 trainer.logging_iter=1 \
        > "$ROOT/evidence/train_H${HH}.log" 2>&1
    TRAIN_RC=$?
    set -e
    printf '%s\n' "$TRAIN_RC" > "$ROOT/evidence/train_H${HH}_exit_code.txt"
    if [[ "$TRAIN_RC" != 0 ]]; then
        echo "H=${H} short training failed; inspect $ROOT/evidence/train_H${HH}.log" >&2
        exit "$TRAIN_RC"
    fi
done

PROBE_OUTPUT="$ROOT/probe24_acc1"
test ! -e "$PROBE_OUTPUT"
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. torchrun --standalone --nproc-per-node=1 \
    -m examples.umift.probe_single24 --output-root "$PROBE_OUTPUT" --grad-accum-iter 1 \
    > "$ROOT/evidence/probe24_acc1.log" 2>&1 &
PROBE_PID=$!
trap 'kill "$PROBE_PID" 2>/dev/null || true; wait "$PROBE_PID" 2>/dev/null || true' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
while kill -0 "$PROBE_PID" 2>/dev/null; do
    nvidia-smi -i 0 --query-gpu=timestamp,memory.used,utilization.gpu \
        --format=csv,noheader,nounits >> "$ROOT/evidence/probe24_acc1_nvml.csv"
    sleep 1
done
set +e
wait "$PROBE_PID"
PROBE_RC=$?
set -e
trap - EXIT INT TERM
printf '%s\n' "$PROBE_RC" > "$ROOT/evidence/probe24_acc1_exit_code.txt"
if [[ "$PROBE_RC" == 0 ]]; then
    printf '%s\n' success > "$ROOT/evidence/probe24_acc1_outcome.txt"
elif grep -qi 'out of memory' "$ROOT/evidence/probe24_acc1.log"; then
    printf '%s\n' expected_oom > "$ROOT/evidence/probe24_acc1_outcome.txt"
else
    printf '%s\n' unexpected_failure > "$ROOT/evidence/probe24_acc1_outcome.txt"
    echo 'single24 acc1 failed for a reason other than memory' >&2
    exit "$PROBE_RC"
fi

date -u +%FT%TZ > "$ROOT/evidence/finished_at_utc.txt"
