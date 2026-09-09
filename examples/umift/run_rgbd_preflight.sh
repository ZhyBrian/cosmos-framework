#!/usr/bin/env bash
# E3-only correctness checks; preserves all prior E1/E2 runs and source data.
set -euo pipefail
cd /data/cosmos-framework
E3_ROOT=/data/cosmos_runs/e3_rgbd_20260909
E3_BASE=/data/cosmos_models/Cosmos3-Edge-DCP-a9d944e
E3_DATA=/data/cosmos_datasets/umift_us_all_source_308f4d46.zarr
E3_TOML=examples/toml/sft_config/action_fd_umift_edge_rgbd_h5.toml
E3_PREFLIGHT="$E3_ROOT/preflight"
test ! -e "$E3_PREFLIGHT"
mkdir -p "$E3_PREFLIGHT" "$E3_ROOT/evidence"
export CUDA_VISIBLE_DEVICES=0,1,2,3
source examples/umift/a40_env.sh
export DATASET_PATH="$E3_DATA" BASE_CHECKPOINT_PATH="$E3_BASE" OMP_NUM_THREADS=1 UMIFT_STAGE=e1
export PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1
git rev-parse HEAD > "$E3_PREFLIGHT/source_commit.txt"
date -u +%FT%TZ > "$E3_PREFLIGHT/started_at_utc.txt"
python -m pytest -c /dev/null --noconftest \
    cosmos_framework/data/generator/action/datasets/umift_history_dataset_test.py \
    examples/umift/history_contract_test.py examples/umift/history_selection_test.py \
    examples/umift/history_rollout_test.py examples/umift/render_history_comparison_test.py \
    examples/umift/test_rgbd_dataset.py examples/umift/test_rgbd_infer.py \
    examples/umift/test_rgbd_selection.py > "$E3_PREFLIGHT/cpu_tests.log" 2>&1
nvidia-smi -i 0,1,2,3 --query-gpu=memory.used --format=csv,noheader,nounits \
    | awk '$1 > 256 { busy=1 } END { exit busy }'
torchrun --standalone --nproc-per-node=4 -m examples.umift.check_rgbd_model \
    --zarr "$E3_DATA" --sft-toml "$E3_TOML" --checkpoint "$E3_BASE/model" \
    --output "$E3_PREFLIGHT/base_model" > "$E3_PREFLIGHT/base_model.log" 2>&1
set +e
OUTPUT_ROOT="$E3_PREFLIGHT/train" IMAGINAIRE_OUTPUT_ROOT="$E3_PREFLIGHT/train" \
    COSMOS_EXIT_WITHOUT_FINALIZE=1 \
    torchrun --standalone --nproc-per-node=4 -m cosmos_framework.scripts.train \
    --sft-toml="$E3_TOML" -- trainer.max_iter=3 checkpoint.save_iter=3 trainer.logging_iter=1 \
    > "$E3_PREFLIGHT/train.log" 2>&1
E3_RC=$?
set -e
printf '%s\n' "$E3_RC" > "$E3_PREFLIGHT/train_exit_code.txt"
if [[ "$E3_RC" != 0 ]]; then exit "$E3_RC"; fi
E3_CP="$E3_PREFLIGHT/train/cosmos3_action_fd_umift/action_sft/action_fd_umift_edge_rgbd_h5/checkpoints/iter_000000003/model"
test -f "$E3_CP/.metadata"
torchrun --standalone --nproc-per-node=4 -m examples.umift.check_rgbd_model \
    --zarr "$E3_DATA" --sft-toml "$E3_TOML" --checkpoint "$E3_CP" \
    --output "$E3_PREFLIGHT/reloaded_model" > "$E3_PREFLIGHT/reloaded_model.log" 2>&1
date -u +%FT%TZ > "$E3_PREFLIGHT/finished_at_utc.txt"
