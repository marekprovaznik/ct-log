#!/bin/bash
# Event-only watchdog for a supervised training job on euler.
#
# Polls the job every POLL seconds with a bash probe (no model tokens) and exits, which wakes the
# Claude session that started it in the background, only when a decision is needed:
#   FINISHED   completion marker "Best foreground mean IoU" in the job log
#   FAILED     job exited without the marker (supervisor gave up after its restarts)
#   BUG        a new traceback that a small model (Sonnet, headless, no tools) classified as a
#              deterministic code/config/data error, so further restarts would only repeat it
#   MLFLOW     MLflow logging disabled/failed (training continues, metrics are lost)
#   STALL      job log silent for STALL_SEC
#   UNREACHABLE euler unreachable for UNREACH_POLLS consecutive polls
#   RECOVERY   auto-recovery failed MAX_RECOVERIES times
# Handled without waking anyone:
#   - supervisor restarts after a TRANSIENT traceback (the supervisor retries with --resume)
#   - container restart / recreation (e.g. admin changing --shm-size): re-stages the MLflow
#     credentials from the local .env and relaunches the job, which resumes from ~/work
#
# Usage (from repo root, as a background command of the session that should be woken):
#   scripts/euler_watch.sh <job> <remote_run_script> <remote_train_info_yaml>
# e.g.
#   scripts/euler_watch.sh hires784x40 /home/jovyan/work/ctlog-eval/scripts/run_784_40ep.sh \
#       /home/jovyan/work/ctlog-eval/logs/kwp_v1_capped_nopith_784_40ep/train_info.yaml
# The remote run script must source and shred ~/work/ctlog-eval/mlflow_cred.sh and resume
# (--resume); see TRAINING_AGENTS.md.

JOB=$1
RUN_SCRIPT=$2
INFO=$3
POLL=${POLL:-300}
STALL_SEC=${STALL_SEC:-2700}
UNREACH_POLLS=${UNREACH_POLLS:-6}
MAX_RECOVERIES=${MAX_RECOVERIES:-3}
TRIAGE_MODEL=${TRIAGE_MODEL:-sonnet}
IGNORE_MLFLOW=${IGNORE_MLFLOW:-}

if [ -z "$JOB" ] || [ -z "$RUN_SCRIPT" ] || [ -z "$INFO" ]; then
    echo "usage: scripts/euler_watch.sh <job> <remote_run_script> <remote_train_info_yaml>"
    exit 2
fi

cd "$(dirname "$0")/.." || exit 2
REPO=$(pwd)
eval "$(grep '^export EULER_' ~/.bashrc)"
EVENTS="$REPO/logs/euler_watch_${JOB}.log"
mkdir -p "$REPO/logs"

euler() {
    timeout "${2:-120}" uv run --no-project "$REPO/scripts/euler.py" exec "$1" 2>/dev/null
}

note() {
    echo "$(date '+%F %T') $*" | tee -a "$EVENTS"
}

wake() {
    note "WAKE $*"
    exit 0
}

probe() {
    euler "
L=~/jobs/$JOB/log; I=$INFO
cstart=\$(stat -c %Y /proc/1)
if [ ! -f \$L ]; then echo \"STATE ep=-9 att=0 tb=0 mlf=0 age=0 exit=nolog done=0 cstart=\$cstart\"; exit 0; fi
ep=\$(grep -m1 'current_epoch:' \$I 2>/dev/null | awk '{print \$2}'); ep=\${ep:--1}
att=\$(grep -c 'SUPERVISOR attempt' \$L); tb=\$(grep -c Traceback \$L)
mlf=\$(grep -ciE 'MLflow (logging disabled|metric logging failed|model logging skipped)' \$L)
done_marker=\$(grep -c 'Best foreground mean IoU' \$L)
age=\$(( \$(date +%s) - \$(stat -c %Y \$L) ))
ex=\$(cat ~/jobs/$JOB/exit 2>/dev/null || echo none)
echo \"STATE ep=\$ep att=\$att tb=\$tb mlf=\$mlf age=\$age exit=\$ex done=\$done_marker cstart=\$cstart\"" 90 | grep '^STATE'
}

field() {
    echo "$1" | sed -n "s/.* $2=\([^ ]*\).*/\1/p"
}

stage_creds() {
    "$REPO/scripts/euler_stage_mlflow_creds.sh" > /dev/null 2>&1
}

relaunch() {
    stage_creds || return 1
    timeout 120 uv run --no-project "$REPO/scripts/euler.py" bg "$JOB" "bash $RUN_SCRIPT" --min-free-mb 8000 2>&1 \
        | grep -q "spuštěno"
}

triage() {
    local tail_text prompt
    tail_text=$(euler "tail -n 60 ~/jobs/$JOB/log" 90)
    prompt="You triage a crashed PyTorch training job. Reply with exactly one line: TRANSIENT: <reason> or BUG: <reason>. TRANSIENT = would likely succeed if simply restarted (CUDA OOM caused by another GPU user, network/HTTP/SSL error, driver hiccup, killed by signal). BUG = deterministic code/config/data error (KeyError, shape mismatch, FileNotFoundError, ValueError from config, syntax or import error, NaN loss that recurs).

Log tail:
$tail_text"
    (cd /tmp && echo "$prompt" | timeout 180 claude -p --model "$TRIAGE_MODEL" --tools "" --strict-mcp-config \
        --no-session-persistence 2>/dev/null | head -1)
}

base=$(probe)
[ -z "$base" ] && { note "cannot reach euler at start"; exit 1; }
note "start $base"
b_att=$(field "$base" att); b_tb=$(field "$base" tb); b_mlf=$(field "$base" mlf); b_cs=$(field "$base" cstart)
fails=0
recoveries=0
while true; do
    sleep "$POLL"
    s=$(probe)
    if [ -z "$s" ]; then
        fails=$((fails + 1))
        [ "$fails" -ge "$UNREACH_POLLS" ] && wake "UNREACHABLE for $fails polls"
        continue
    fi
    fails=0

    if [ "$(field "$s" cstart)" != "$b_cs" ] || [ "$(field "$s" exit)" = "nolog" ]; then
        if [ "$(field "$s" done)" = "1" ]; then
            wake "FINISHED $s"
        fi
        recoveries=$((recoveries + 1))
        [ "$recoveries" -gt "$MAX_RECOVERIES" ] && wake "RECOVERY limit reached: $s"
        note "container restarted or job log gone ($s); relaunching (recovery $recoveries/$MAX_RECOVERIES)"
        if relaunch; then
            sleep 60
            s=$(probe)
            note "relaunched: $s"
            b_att=$(field "$s" att); b_tb=$(field "$s" tb); b_mlf=$(field "$s" mlf); b_cs=$(field "$s" cstart)
        else
            note "relaunch failed (GPU guard or upload); will retry next poll"
        fi
        continue
    fi

    [ "$(field "$s" done)" = "1" ] && [ "$(field "$s" exit)" != "none" ] && wake "FINISHED $s"
    [ "$(field "$s" exit)" != "none" ] && wake "FAILED job exited without completion marker: $s"
    if [ "$(field "$s" mlf)" -gt "$b_mlf" ]; then
        [ -z "$IGNORE_MLFLOW" ] && wake "MLFLOW failure: $s"
        note "MLflow failure ignored (IGNORE_MLFLOW set): $s"
        b_mlf=$(field "$s" mlf)
    fi

    if [ "$(field "$s" tb)" -gt "$b_tb" ]; then
        verdict=$(triage)
        note "traceback triage ($TRIAGE_MODEL): ${verdict:-no verdict}"
        case "$verdict" in
            TRANSIENT*) b_tb=$(field "$s" tb); b_att=$(field "$s" att) ;;
            *) wake "BUG or untriaged traceback: ${verdict:-no verdict} | $s" ;;
        esac
    fi

    [ "$(field "$s" age)" -gt "$STALL_SEC" ] && wake "STALL log silent $(field "$s" age)s: $s"
done
