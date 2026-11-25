#!/usr/bin/env bash
# Restart scripts/run_schedule.sh at a given time today (default 19:00) so a changed START/END
# takes effect without stopping the counter earlier (bash keeps the running loop in memory).
# The running live_fused.py is stopped right after one of its 10-s log flushes, so at most the
# reconciliation's pending events are lost.
# Usage: setsid nohup bash scripts/switch_schedule_at.sh 19:00 </dev/null >>runs/live/switch.log 2>&1 &
cd "$(dirname "$0")/.."
target=$(date -d "today ${1:-19:00}" +%s)
echo "$(date '+%F %T') waiting until $(date -d @$target '+%T')"
while (( $(date +%s) < target )); do sleep $(( target-$(date +%s) > 30 ? 30 : 1 )); done
P=$(pgrep -f "live_fused.py --setup" | head -1)
if [ -n "$P" ]; then
  L=$(ls -t runs/live/2*_*.log | head -1); m=$(stat -c %y "$L")
  for i in $(seq 1 240); do [ "$(stat -c %y "$L")" != "$m" ] && break; sleep 0.05; done
fi
kill -TERM "$(cat runs/live/schedule.pid)" 2>/dev/null
[ -n "$P" ] && kill -TERM "$P"
while [ -n "$P" ] && kill -0 "$P" 2>/dev/null; do sleep 0.2; done
setsid nohup bash scripts/run_schedule.sh </dev/null >/dev/null 2>&1 &
echo "$(date '+%F %T') schedule restarted (live_fused $P stopped)" | tee -a runs/live/restarts.log
