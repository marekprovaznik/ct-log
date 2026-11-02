#!/bin/bash
# Upload the DagsHub MLflow credentials to euler as ~/work/ctlog-eval/mlflow_cred.sh.
#
# Usage (from repo root): with-secrets scripts/euler_stage_mlflow_creds.sh
# Reads MLFLOW_TRACKING_USERNAME / MLFLOW_TRACKING_PASSWORD from the environment, or from .env when
# they are unset there. Never prints them. The training runner (euler_run_timber.sh, kwp run scripts)
# sources and shreds the remote file at start-up, so stage right before a launch.

set -u
REPO="$(cd "$(dirname "$0")/.." && pwd)"
if [ -z "${MLFLOW_TRACKING_USERNAME:-}" ] || [ -z "${MLFLOW_TRACKING_PASSWORD:-}" ]; then
    if [ -f "$REPO/.env" ]; then
        set -a
        source <(grep -E '^(export )?MLFLOW_TRACKING_(USERNAME|PASSWORD)=' "$REPO/.env" | sed 's/^export //')
        set +a
    fi
fi
if [ -z "${MLFLOW_TRACKING_USERNAME:-}" ] || [ -z "${MLFLOW_TRACKING_PASSWORD:-}" ]; then
    echo "MLFLOW_TRACKING_USERNAME/PASSWORD not set (run under with-secrets)" >&2
    exit 1
fi

umask 077
tmp=$(mktemp)
trap 'shred -u "$tmp" 2>/dev/null || rm -f "$tmp"' EXIT
printf "MLFLOW_TRACKING_USERNAME=%s\nMLFLOW_TRACKING_PASSWORD=%s\n" "$MLFLOW_TRACKING_USERNAME" "$MLFLOW_TRACKING_PASSWORD" > "$tmp"
timeout 120 uv run --no-project "$REPO/scripts/euler.py" put "$tmp" work/ctlog-eval/mlflow_cred.sh > /dev/null 2>&1
rc=$?
[ "$rc" -eq 0 ] && echo "staged work/ctlog-eval/mlflow_cred.sh" || echo "upload failed (rc=$rc)" >&2
exit "$rc"
