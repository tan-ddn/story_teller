#!/bin/sh
# Explicitly invoked only after a confirmed failed adapter transfer; no automatic retry.
set -eu
: "${ARCHIVIST_PYTHON:?Set ARCHIVIST_PYTHON to the Mac MLX Python}"
: "${ARCHIVIST_RUN_DIR:?Set ARCHIVIST_RUN_DIR to a new persistent output directory}"
export PYTHONPATH=.
export MLX_ADAPTERS_DIR="$ARCHIVIST_RUN_DIR/adapters"
export MLX_MAX_TOKENS=8192 MLX_EVAL_MAX_MEMORY_GB=60
models=${MLX_MODELS_DIR:-/Volumes/Untitled/Shared/models}
model=DeepSeek-R1-Distill-Qwen-14B-4bit
data=data/interim/archivist_sft
config=training/configs/archivist_deepseek14_mlx.yaml
mkdir -p "$MLX_ADAPTERS_DIR"
"$ARCHIVIST_PYTHON" -u -m training.archivist.deepseek_mlx_preflight "$models/$model" "$data"
wait_for_memory() {
    while :; do
        pages=$(/usr/bin/vm_stat | awk '/Pages free:/{gsub(/[.]/,"",$3); print $3}')
        page_size=$(/usr/sbin/sysctl -n hw.pagesize)
        level=$(/usr/sbin/sysctl -n kern.memorystatus_level)
        case "$pages:$page_size:$level" in *[!0-9:]*|:*|*::*|*:) exit 1;; esac
        [ "$((pages * page_size))" -ge 34359738368 ] && [ "$level" -ge 60 ] && return 0
        sleep 60
    done
}
wait_for_memory
"$ARCHIVIST_PYTHON" -u -m training.archivist.train "$config" --smoke --reserve-gib 4
wait_for_memory
"$ARCHIVIST_PYTHON" -u -m training.archivist.train "$config" --smoke --smoke-varied --reserve-gib 4
wait_for_memory
"$ARCHIVIST_PYTHON" -u -m training.archivist.train "$config" --reserve-gib 4
wait_for_memory
"$ARCHIVIST_PYTHON" -u -m evaluation.archivist "mlx:$model+archivist-deepseek14-mlx-scratch" "$data/valid.jsonl" "$ARCHIVIST_RUN_DIR/tuned-valid"
