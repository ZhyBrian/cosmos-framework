#!/usr/bin/env bash
# A40 E1-R only: run after main exit0; never modify or select training weights.
set -euo pipefail
cd /data/cosmos-framework
ROOT=/data/cosmos_runs/e1_refit_20260907
test "$(cat "$ROOT/main/exit_code.txt")" = 0
export CUDA_VISIBLE_DEVICES=0,1,2,3
source examples/umift/a40_env.sh
export DATASET_PATH=/data/cosmos_datasets/umift_us_all_source_308f4d46.zarr
export BASE_CHECKPOINT_PATH=/data/cosmos_models/Cosmos3-Edge-DCP-a9d944e
export OMP_NUM_THREADS=1
export UMIFT_STAGE=e1
TOML=examples/toml/sft_config/action_fd_umift_edge_refit.toml
CHECKPOINTS="$ROOT/main/cosmos3_action_fd_umift/action_sft/action_fd_umift_edge_e1_refit/checkpoints"
PARENT0=/data/cosmos_runs/e1_long_rollout_20260907
PARENT_MID=/data/cosmos_runs/e1_midstart_rollout_20260907
FONT=/data/cosmos_runs/e1_final_eval/video_assets_20260907/NotoSansCJK-Regular.ttc
test -f "$CHECKPOINTS/iter_000001000/model/.metadata"
test -f "$CHECKPOINTS/iter_000000500/model/.metadata"
nvidia-smi -i 0,1,2,3 --query-gpu=memory.used --format=csv,noheader,nounits \
    | awk '$1 > 256 { busy=1 } END { exit busy }'
test ! -e "$ROOT/evaluation"
mkdir -p "$ROOT/evaluation" "$ROOT/cosmos_output"
git rev-parse HEAD > "$ROOT/evaluation/source_commit.txt"
date -u +%FT%TZ > "$ROOT/evaluation/started_at_utc.txt"

# GPU0 proxy is sequential with the four-GPU jobs. The external reading includes
# context/library memory that PyTorch's 24GiB caching allocator cap does not.
CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc-per-node=1 \
    -m examples.umift.probe_single24 --output-root "$ROOT/single24" \
    > "$ROOT/evaluation/single24.log" 2>&1 &
PROBE_PID=$!
trap 'kill "$PROBE_PID" 2>/dev/null || true; wait "$PROBE_PID" 2>/dev/null || true' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
while kill -0 "$PROBE_PID" 2>/dev/null; do
    nvidia-smi -i 0 --query-gpu=timestamp,memory.used,utilization.gpu --format=csv,noheader,nounits \
        >> "$ROOT/evaluation/single24_nvml.csv"
    sleep 1
done
if wait "$PROBE_PID"; then PROBE_RC=0; else PROBE_RC=$?; fi
trap - EXIT INT TERM
printf '%s\n' "$PROBE_RC" > "$ROOT/evaluation/single24_exit_code.txt"
if [[ "$PROBE_RC" != 0 ]] && ! grep -qi 'out of memory' "$ROOT/evaluation/single24.log"; then
    echo 'single24 failed for a reason other than memory: inspect before proceeding' >&2
    exit "$PROBE_RC"
fi

python -m examples.umift.prepare_refit_rollout --source-root "$PARENT0" \
    --root "$ROOT/evaluation/reference500" --iteration 500 \
    --checkpoint-model "$CHECKPOINTS/iter_000000500/model" \
    --expected-fixture-sha256 8f311b93c18e20d8ab752e5f86e33e0691296e10bcbdb565511ddd823ba151d9
torchrun --standalone --nproc-per-node=4 -m examples.umift.long_rollout infer \
    --root "$ROOT/evaluation/reference500" --sft-toml "$TOML" --phase finetuned --max-chunks 2 \
    > "$ROOT/evaluation/reference500.log" 2>&1

for PERCENT in 0 33 67; do
    case "$PERCENT" in
        0) PARENT="$PARENT0"; SHA=8f311b93c18e20d8ab752e5f86e33e0691296e10bcbdb565511ddd823ba151d9 ;;
        33) PARENT="$PARENT_MID/start33"; SHA=9c18fc449ed79a7230b047c0cff85f74837748409fe316c0301cb51f4dc96949 ;;
        67) PARENT="$PARENT_MID/start67"; SHA=aa3259970c9afd3e151972f439fff92e324dd7355957c3005b40bfbadea48645 ;;
    esac
    GROUP="$ROOT/evaluation/start$PERCENT"
    python -m examples.umift.prepare_refit_rollout --source-root "$PARENT" --root "$GROUP" \
        --checkpoint-model "$CHECKPOINTS/iter_000001000/model" --expected-fixture-sha256 "$SHA"
    if [[ "$PERCENT" == 0 ]]; then
        torchrun --standalone --nproc-per-node=4 -m examples.umift.long_rollout infer \
            --root "$GROUP" --sft-toml "$TOML" --phase finetuned --max-chunks 2 \
            > "$GROUP/reference1000.log" 2>&1
    fi
    for PHASE in base finetuned; do
        torchrun --standalone --nproc-per-node=4 -m examples.umift.long_rollout infer \
            --root "$GROUP" --sft-toml "$TOML" --phase "$PHASE" > "$GROUP/$PHASE.log" 2>&1
    done
done

for PERCENT in 0 33 67; do
    python -m examples.umift.render_long_comparison --root "$ROOT/evaluation/start$PERCENT" \
        --output "$ROOT/cosmos_output/start$PERCENT" --font "$FONT" \
        > "$ROOT/evaluation/render$PERCENT.log" 2>&1
done
date -u +%FT%TZ > "$ROOT/evaluation/finished_at_utc.txt"
