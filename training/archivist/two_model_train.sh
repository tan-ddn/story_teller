#!/bin/sh
# Train the two selected models sequentially; never resume or overwrite an adapter.
set -u
: "${ARCHIVIST_PYTHON:?Set ARCHIVIST_PYTHON to the Mac MLX Python}"
: "${ARCHIVIST_RUN_DIR:?Set ARCHIVIST_RUN_DIR to a persistent output directory}"
export PYTHONPATH=.
export MLX_MAX_TOKENS=8192
export MLX_EVAL_MAX_MEMORY_GB=60
export MLX_ADAPTERS_DIR="$ARCHIVIST_RUN_DIR/adapters"
mkdir -p "$ARCHIVIST_RUN_DIR" "$MLX_ADAPTERS_DIR"
lock="$ARCHIVIST_RUN_DIR/queue.lock"
if ! mkdir "$lock" 2>/dev/null; then
    echo "Queue already running or stale lock: $lock" >&2
    exit 1
fi
trap 'rmdir "$lock" 2>/dev/null || :' EXIT
trap 'exit 1' HUP INT TERM

wait_for_memory() {
    while :; do
        pages=$(/usr/bin/vm_stat | awk '/Pages free:/{gsub(/[.]/,"",$3); print $3}') || return 1
        page_size=$(/usr/sbin/sysctl -n hw.pagesize) || return 1
        pressure=$(/usr/sbin/sysctl -n kern.memorystatus_level) || return 1
        case "$pages:$page_size:$pressure" in
            *[!0-9:]*|:*|*::*|*:) echo "Cannot read memory state" >&2; return 1;;
        esac
        if [ "$((pages * page_size))" -ge 34359738368 ] && [ "$pressure" -ge 60 ]; then
            return 0
        fi
        echo "$(date): waiting for 32 GiB free pages and pressure level >= 60%"
        sleep 60
    done
}

train_one() {
    name=$1
    adapter=$2
    if [ -e "$MLX_ADAPTERS_DIR/$adapter" ]; then
        echo "$(date): $adapter exists; refusing to overwrite or claim success" >&2
        return 1
    fi
    wait_for_memory || return 1
    echo "$(date): starting full fine-tune $name"
    if "$ARCHIVIST_PYTHON" -u -m training.archivist.train "training/configs/$name.yaml"; then
        echo "$(date): completed $name"
        return 0
    else
        status=$?
        echo "$(date): failed $name (exit $status); checkpoints retained" >&2
        return "$status"
    fi
}

evaluate() {
    model=$1
    split=$2
    label=$3
    wait_for_memory || return 1
    echo "$(date): evaluating $label on $split"
    "$ARCHIVIST_PYTHON" -u -m evaluation.archivist "$model" \
        "data/interim/archivist_sft/$split.jsonl" "$ARCHIVIST_RUN_DIR/evaluation/$label" \
        --memory-pressure-only
}

gemma_ok=0
qwen_ok=0
if train_one archivist_gemma4 archivist-v1-gemma4; then gemma_ok=1; fi
if train_one archivist_qwen3 archivist-v1-qwen3; then qwen_ok=1; fi

if [ "$gemma_ok" -eq 1 ]; then
    evaluate mlx:gemma-4-26b-a4b-it-4bit+archivist-v1-gemma4 valid tuned-gemma4-valid || :
    evaluate mlx:gemma-4-26b-a4b-it-4bit test base-gemma4-test || :
    evaluate mlx:gemma-4-26b-a4b-it-4bit+archivist-v1-gemma4 test tuned-gemma4-test || :
fi
if [ "$qwen_ok" -eq 1 ]; then
    evaluate mlx:Qwen3-30B-A3B-4bit+archivist-v1-qwen3 valid tuned-qwen3-valid || :
    evaluate mlx:Qwen3-30B-A3B-4bit test base-qwen3-test || :
    evaluate mlx:Qwen3-30B-A3B-4bit+archivist-v1-qwen3 test tuned-qwen3-test || :
fi
echo "$(date): queue finished; gemma_trained=$gemma_ok qwen_trained=$qwen_ok"
