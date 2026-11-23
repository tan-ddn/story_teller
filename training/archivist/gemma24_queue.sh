#!/bin/sh
# Run a Gemma layer-count experiment only after the existing benchmark queue ends.
set -eu
: "${ARCHIVIST_PYTHON:?Set ARCHIVIST_PYTHON to the Mac MLX Python}"
: "${ARCHIVIST_RUN_DIR:?Set ARCHIVIST_RUN_DIR to a persistent output directory}"
: "${WAIT_PID:?Set WAIT_PID to the current benchmark queue PID}"
export PYTHONPATH=.
export MLX_ADAPTERS_DIR="$ARCHIVIST_RUN_DIR/adapters"
export MLX_MAX_TOKENS=8192
export MLX_EVAL_MAX_MEMORY_GB=60
layers="${GEMMA_LAYERS:-24}"
reserve_gib="${FULL_RESERVE_GIB:-8}"
wait_match="${WAIT_MATCH:-two_model_train.sh}"
case "$layers" in 8|20|24) ;; *) echo "Unsupported layer count: $layers" >&2; exit 1;; esac
adapter="archivist-v1-gemma4-l${layers}"
mkdir -p "$ARCHIVIST_RUN_DIR" "$MLX_ADAPTERS_DIR"
lock="$ARCHIVIST_RUN_DIR/queue.lock"
if ! mkdir "$lock" 2>/dev/null; then
    echo "Queue already running or stale lock: $lock" >&2
    exit 1
fi
trap 'rmdir "$lock" 2>/dev/null || :' EXIT
trap 'exit 1' HUP INT TERM

while ps -p "$WAIT_PID" -o command= 2>/dev/null | grep -F -q "$wait_match"; do
    echo "$(date): waiting for existing benchmark queue $WAIT_PID"
    sleep 60
done

wait_for_pressure() {
    while :; do
        level=$(/usr/sbin/sysctl -n kern.memorystatus_level)
        case "$level" in ''|*[!0-9]*) echo "Cannot read memory pressure" >&2; exit 1;; esac
        [ "$level" -ge 60 ] && return 0
        echo "$(date): waiting for pressure level >= 60%"
        sleep 60
    done
}

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

config="training/configs/archivist_gemma4_l${layers}.yaml"
if [ "${SKIP_VERIFIED_SMOKE:-0}" != 1 ]; then
    wait_for_pressure
    echo "$(date): starting ${layers}-layer longest-scene smoke test"
    if [ -n "${SMOKE_RESERVE_GIB:-}" ]; then
        "$ARCHIVIST_PYTHON" -u -m training.archivist.train "$config" --smoke \
            --reserve-gib "$SMOKE_RESERVE_GIB"
    else
        "$ARCHIVIST_PYTHON" -u -m training.archivist.train "$config" --smoke --memory-pressure-only
    fi
    wait_for_pressure
    echo "$(date): starting ${layers}-layer varied-scene stability test"
    if [ -n "${SMOKE_RESERVE_GIB:-}" ]; then
        "$ARCHIVIST_PYTHON" -u -m training.archivist.train "$config" --smoke --smoke-varied \
            --reserve-gib "$SMOKE_RESERVE_GIB"
    else
        "$ARCHIVIST_PYTHON" -u -m training.archivist.train "$config" --smoke --smoke-varied \
            --memory-pressure-only
    fi
else
    : "${SMOKE_VERIFIED_RUN_DIR:?Set SMOKE_VERIFIED_RUN_DIR to the prior smoke outputs}"
    for suffix in smoke smoke-varied; do
        prior="$SMOKE_VERIFIED_RUN_DIR/adapters/$adapter-$suffix"
        [ -s "$prior/adapters.safetensors" ] || { echo "Missing verified $suffix adapter" >&2; exit 1; }
        cmp -s "$config" "$prior/training_config.yaml" || {
            echo "Prior $suffix config does not match" >&2; exit 1;
        }
        cmp -s data/interim/archivist_sft/manifest.json "$prior/data_manifest.json" || {
            echo "Prior $suffix dataset manifest does not match" >&2; exit 1;
        }
    done
    echo "$(date): using previously verified smoke results"
fi
wait_for_memory
echo "$(date): starting ${layers}-layer full fine-tune"
"$ARCHIVIST_PYTHON" -u -m training.archivist.train "$config" --reserve-gib "$reserve_gib"
wait_for_pressure
echo "$(date): evaluating ${layers}-layer adapter on validation"
"$ARCHIVIST_PYTHON" -u -m evaluation.archivist \
    "mlx:gemma-4-26b-a4b-it-4bit+$adapter" \
    data/interim/archivist_sft/valid.jsonl "$ARCHIVIST_RUN_DIR/evaluation/valid" \
    --memory-pressure-only
wait_for_pressure
echo "$(date): evaluating ${layers}-layer adapter on test"
"$ARCHIVIST_PYTHON" -u -m evaluation.archivist \
    "mlx:gemma-4-26b-a4b-it-4bit+$adapter" \
    data/interim/archivist_sft/test.jsonl "$ARCHIVIST_RUN_DIR/evaluation/test" \
    --memory-pressure-only
echo "$(date): ${layers}-layer experiment complete"
