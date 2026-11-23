#!/bin/sh
# Run only diagnostic smoke tests, after the selected prior worker exits.
set -u
: "${ARCHIVIST_PYTHON:?Set ARCHIVIST_PYTHON to the Mac MLX Python}"
: "${WAIT_PID:=${GEMMA_PID:-}}"
: "${WAIT_PID:?Set WAIT_PID (or GEMMA_PID) to the prior worker PID}"
WAIT_MATCH="${WAIT_MATCH:-smoke-varied}"
export PYTHONPATH=.
MAX_MEMORY_GB="${MAX_MEMORY_GB:-60}"
mkdir -p fix-tests

while ps -p "$WAIT_PID" -o command= 2>/dev/null | grep -F -q "$WAIT_MATCH"; do
    echo "$(date): waiting for prior worker $WAIT_PID"
    sleep 30
done

for layers in ${LAYER_COUNTS:-8 4}; do
    while :; do
        pressure=$(/usr/sbin/sysctl -n kern.memorystatus_level) || exit 1
        case "$pressure" in ''|*[!0-9]*) echo "Cannot read memory pressure" >&2; exit 1;; esac
        [ "$pressure" -ge 60 ] && break
        echo "$(date): waiting for pressure level >= 60%"
        sleep 60
    done
    config="fix-tests/qwen36-windows-l${layers}.yaml"
    sed -e "s@dataset: data/interim/archivist_sft@dataset: data/interim/archivist_sft_windows_v1@" \
        -e 's/max_seq_length: 10240/max_seq_length: 8192/' \
        -e "s/num_layers: 16/num_layers: ${layers}/" \
        training/configs/archivist_qwen36.yaml > "$config"
    echo "$(date): starting diagnostic window smoke, ${layers} layers"
    if MLX_ADAPTERS_DIR="$PWD/fix-tests/window-l${layers}-cap${MAX_MEMORY_GB}" \
        "$ARCHIVIST_PYTHON" -u -c \
        'import mlx.core as mx, runpy; mx.disable_compile(); runpy.run_module("training.archivist.train", run_name="__main__")' \
        "$config" --smoke --memory-pressure-only --max-memory-gb "$MAX_MEMORY_GB"; then
        echo "$(date): ${layers}-layer smoke passed"
    else
        status=$?
        echo "$(date): ${layers}-layer smoke failed (exit $status)"
    fi
done
echo "$(date): diagnostic window tests finished"
