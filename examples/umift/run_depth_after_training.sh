#!/usr/bin/env bash
# Continue the authorized evaluation/video workflow after this arm finishes.
set -euo pipefail
cd /data/cosmos-framework
ARM="${1:-}"
case "$ARM" in
    b_continue|d1) ;;
    *) echo "usage: $0 {b_continue|d1}" >&2; exit 2 ;;
esac
ARM_ROOT="/data/cosmos_runs/e3_depth_aux_20260914/$ARM"
test -f "$ARM_ROOT/launcher.pid"
TRAIN_PID="$(cat "$ARM_ROOT/launcher.pid")"
while [[ ! -f "$ARM_ROOT/logs/train_exit_code.txt" ]]; do
    if ! kill -0 "$TRAIN_PID" 2>/dev/null; then
        echo "training launcher exited without an exit record" >&2
        exit 1
    fi
    sleep 30
done
test "$(cat "$ARM_ROOT/logs/train_exit_code.txt")" = 0
bash examples/umift/run_depth_evaluation.sh "$ARM"
bash examples/umift/run_depth_videos.sh "$ARM"
