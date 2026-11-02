#!/bin/bash
# Supervised timber training run on euler: resumes after crashes, stops at the completion marker.
#
# Usage (on euler): bash ~/work/ctlog-eval/scripts/euler_run_timber.sh <config> <run_name> [extra train args...]
# MLflow credentials are read once from ~/work/ctlog-eval/mlflow_cred.sh (staged by euler_watch.sh
# stage_creds or by hand) and the file is shredded right after.

set -u
REPO=/home/jovyan/work/ctlog-eval
cd "$REPO"
CONFIG="${1:?config required}"
RUN_NAME="${2:?run name required}"
shift 2
LOG_DIR="$REPO/logs/$RUN_NAME"
LOG="$LOG_DIR/run.log"
SUP_LOG="$LOG_DIR/supervisor.log"
MAX_RESTARTS=${MAX_RESTARTS:-20}
MIN_ALIVE_SECONDS=120
PYTHON="$REPO/.venv/bin/python"
[ -x "$PYTHON" ] || PYTHON=python
mkdir -p "$LOG_DIR"

if [ -f "$REPO/mlflow_cred.sh" ]; then
    set -a
    source "$REPO/mlflow_cred.sh"
    set +a
    shred -u "$REPO/mlflow_cred.sh" 2>/dev/null || rm -f "$REPO/mlflow_cred.sh"
fi
export CTLOG_GIT_COMMIT="${CTLOG_GIT_COMMIT:-$(cat "$REPO/TIMBER_COMMIT" 2>/dev/null || echo unknown)}"

restarts=0
while true; do
    if grep -q "Best foreground mean IoU" "$LOG" 2>/dev/null; then
        echo "$(date '+%F %T') run already complete" >> "$SUP_LOG"
        break
    fi
    started=$(date +%s)
    echo "$(date '+%F %T') starting (restart #$restarts)" >> "$SUP_LOG"
    "$PYTHON" -u -m src.train_timber --config "$CONFIG" --run_name "$RUN_NAME" --local_log_dir "$LOG_DIR" --resume "$@" \
        >> "$LOG" 2>&1
    rc=$?
    elapsed=$(( $(date +%s) - started ))
    if grep -q "Best foreground mean IoU" "$LOG" 2>/dev/null; then
        echo "$(date '+%F %T') completed (rc=$rc)" >> "$SUP_LOG"
        break
    fi
    restarts=$(( restarts + 1 ))
    if [ "$restarts" -ge "$MAX_RESTARTS" ] || [ "$elapsed" -lt "$MIN_ALIVE_SECONDS" ]; then
        echo "$(date '+%F %T') giving up after ${elapsed}s, restart #$restarts (rc=$rc)" >> "$SUP_LOG"
        exit 1
    fi
    echo "$(date '+%F %T') died after ${elapsed}s (rc=$rc); resuming in 30s" >> "$SUP_LOG"
    sleep 30
done
