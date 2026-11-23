#!/bin/sh
# Qwen3-30B-A3B: wait for preceding experiments; start a fresh, guarded run.
set -eu
: "${ARCHIVIST_PYTHON:?Set ARCHIVIST_PYTHON to the Mac MLX Python}"
: "${ARCHIVIST_RUN_DIR:?Set ARCHIVIST_RUN_DIR to a persistent output directory}"
: "${WAIT_PID:?Set WAIT_PID to the preceding queue PID}"
export PYTHONPATH=.
export MLX_ADAPTERS_DIR="$ARCHIVIST_RUN_DIR/adapters"
export MLX_MAX_TOKENS=8192
export MLX_EVAL_MAX_MEMORY_GB=60
mkdir -p "$ARCHIVIST_RUN_DIR" "$MLX_ADAPTERS_DIR"
lock="$ARCHIVIST_RUN_DIR/queue.lock"
if ! mkdir "$lock" 2>/dev/null; then
    echo "Queue already running or stale lock: $lock" >&2
    exit 1
fi
trap 'rmdir "$lock" 2>/dev/null || :' EXIT
trap 'exit 1' HUP INT TERM

while ps -p "$WAIT_PID" -o command= 2>/dev/null | grep -F -q 'gemma_floor4_queue.sh'; do
    echo "$(date): waiting for preceding Gemma queue $WAIT_PID"
    sleep 60
done

wait_for_memory() {
    while :; do
        pages=$(/usr/bin/vm_stat | awk '/Pages free:/{gsub(/[.]/,"",$3); print $3}')
        page_size=$(/usr/sbin/sysctl -n hw.pagesize)
        level=$(/usr/sbin/sysctl -n kern.memorystatus_level)
        case "$pages:$page_size:$level" in
            *[!0-9:]*|:*|*::*|*:) echo "Cannot read memory state" >&2; exit 1;;
        esac
        [ "$((pages * page_size))" -ge 34359738368 ] && [ "$level" -ge 60 ] && return 0
        echo "$(date): waiting for 32 GiB free pages and pressure level >= 60%"
        sleep 60
    done
}

config=training/configs/archivist_qwen3.yaml
wait_for_memory
echo "$(date): starting Qwen3 longest-scene smoke test"
"$ARCHIVIST_PYTHON" -u -m training.archivist.train "$config" --smoke --reserve-gib 4
wait_for_memory
echo "$(date): starting Qwen3 varied-scene stability test"
"$ARCHIVIST_PYTHON" -u -m training.archivist.train "$config" --smoke --smoke-varied \
    --reserve-gib 4
wait_for_memory
echo "$(date): starting fresh Qwen3 full fine-tune"
"$ARCHIVIST_PYTHON" -u -m training.archivist.train "$config" --reserve-gib 4
wait_for_memory
echo "$(date): evaluating fine-tuned Qwen3 on validation"
"$ARCHIVIST_PYTHON" -u -m evaluation.archivist \
    mlx:Qwen3-30B-A3B-4bit+archivist-v1-qwen3 \
    data/interim/archivist_sft/valid.jsonl "$ARCHIVIST_RUN_DIR/evaluation/tuned-valid" \
    --memory-pressure-only
wait_for_memory
echo "$(date): evaluating base Qwen3 on test"
"$ARCHIVIST_PYTHON" -u -m evaluation.archivist mlx:Qwen3-30B-A3B-4bit \
    data/interim/archivist_sft/test.jsonl "$ARCHIVIST_RUN_DIR/evaluation/base-test" \
    --memory-pressure-only
wait_for_memory
echo "$(date): evaluating fine-tuned Qwen3 on test"
"$ARCHIVIST_PYTHON" -u -m evaluation.archivist \
    mlx:Qwen3-30B-A3B-4bit+archivist-v1-qwen3 \
    data/interim/archivist_sft/test.jsonl "$ARCHIVIST_RUN_DIR/evaluation/tuned-test" \
    --memory-pressure-only
echo "$(date): Qwen3 experiment complete"
