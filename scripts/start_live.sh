#!/bin/sh
# Always-on service: web UI on :8080 all the time, counting 6:00-19:00 (local time).
# Day totals are kept per day in runs/live_daily/. Stop with Ctrl-C
# (or `tmux send-keys -t mitafes-live C-c`): a running session finishes and writes its summary.
cd "$(dirname "$0")/.." || exit 1
exec .venv/bin/python live_service.py --view "${VIEW:-plaza}" --hours "${HOURS:-6-19}" --web-port "${PORT:-8080}" "$@" \
  2>&1 | tee -ia "runs/live_service.log"
