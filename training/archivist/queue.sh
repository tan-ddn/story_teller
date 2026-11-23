#!/bin/sh
# Sequential baseline, memory smoke, fine-tune, and final comparison on a shared Mac.
set -eu

: "${ARCHIVIST_PYTHON:?Set ARCHIVIST_PYTHON to the Mac MLX Python}"
export PYTHONPATH=.
export MLX_MAX_TOKENS=8192
export MLX_EVAL_MAX_MEMORY_GB=60
export MLX_ADAPTERS_DIR="${MLX_ADAPTERS_DIR:-$PWD/queue-adapters}"
root="${ARCHIVIST_OUTPUT_DIR:-$PWD/benchmark}"
mkdir -p "$root" "$MLX_ADAPTERS_DIR"
lock="$root/queue.lock"
if ! mkdir "$lock" 2>/dev/null; then
    echo "Queue already running or stale lock: $lock" >&2
    exit 1
fi
trap 'rmdir "$lock"' EXIT HUP INT TERM

wait_for_memory() {
    # 8 GiB free-page reserve plus 24 GiB of model headroom. Fail closed on parse errors.
    while :; do
        pages=$(/usr/bin/vm_stat | awk '/Pages free:/{gsub(/[.]/,"",$3); print $3}')
        page_size=$(/usr/sbin/sysctl -n hw.pagesize)
        case "$pages:$page_size" in *[!0-9:]*|:*) echo "Cannot read free memory" >&2; exit 1;; esac
        if [ "$((pages * page_size))" -ge 34359738368 ]; then break; fi
        echo "$(date): waiting for 32 GiB free pages (includes 8 GiB reserve)"
        sleep 60
    done
}

wait_for_pressure() {
    while :; do
        level=$(/usr/sbin/sysctl -n kern.memorystatus_level)
        case "$level" in ''|*[!0-9]*) echo "Cannot read memory pressure" >&2; exit 1;; esac
        if [ "$level" -ge 60 ]; then break; fi
        echo "$(date): waiting for memory-pressure level of at least 60%"
        sleep 60
    done
}

evaluate() {
    wait_for_pressure
    echo "$(date): evaluating $1 on $2"
    "$ARCHIVIST_PYTHON" -u -m evaluation.archivist "$1" \
        "data/interim/archivist_sft/$2.jsonl" "$root/$3" --memory-pressure-only
}

train() {
    if [ "$2" = "--smoke" ]; then wait_for_pressure; else wait_for_memory; fi
    echo "$(date): training $1 $2"
    if [ "$2" = "--smoke" ]; then
        "$ARCHIVIST_PYTHON" -u -m training.archivist.train "training/configs/$1.yaml" \
            --smoke --memory-pressure-only
    else
        "$ARCHIVIST_PYTHON" -u -m training.archivist.train "training/configs/$1.yaml"
    fi
}

evaluate mlx:Qwen3-30B-A3B-4bit valid base-qwen3-valid
evaluate mlx:Qwen3.6-27B-4bit valid base-qwen36-valid
evaluate mlx:gemma-4-26b-a4b-it-4bit valid base-gemma4-valid

# The three full-length smoke runs must all pass before any full training is launched.
train archivist_qwen3 --smoke
train archivist_qwen36 --smoke
train archivist_gemma4 --smoke
train archivist_qwen3 ""
train archivist_qwen36 ""
train archivist_gemma4 ""

evaluate mlx:Qwen3-30B-A3B-4bit+archivist-v1-qwen3 valid tuned-qwen3-valid
evaluate mlx:Qwen3.6-27B-4bit+archivist-v1-qwen36 valid tuned-qwen36-valid
evaluate mlx:gemma-4-26b-a4b-it-4bit+archivist-v1-gemma4 valid tuned-gemma4-valid
evaluate mlx:Qwen3-30B-A3B-4bit test base-qwen3-test
evaluate mlx:Qwen3.6-27B-4bit test base-qwen36-test
evaluate mlx:gemma-4-26b-a4b-it-4bit test base-gemma4-test
evaluate mlx:Qwen3-30B-A3B-4bit+archivist-v1-qwen3 test tuned-qwen3-test
evaluate mlx:Qwen3.6-27B-4bit+archivist-v1-qwen36 test tuned-qwen36-test
evaluate mlx:gemma-4-26b-a4b-it-4bit+archivist-v1-gemma4 test tuned-gemma4-test
echo "$(date): all benchmark stages completed"
