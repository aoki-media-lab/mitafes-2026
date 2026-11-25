#!/usr/bin/env bash
# Keep the live glass counter running with the browser view.
# Restarts with a new output folder if the process exits abnormally (counts then restart
# from 0 in the browser; every run's events stay in its own folder).
# Usage: scripts/run_live.sh   (env: SETUP, HOURS, WEB_HOST, WEB_PORT)
# Stop:  kill the loop (its PID is in runs/live/run_live.pid), then the python child.
cd "$(dirname "$0")/.."
SETUP=${SETUP:-reference/setups/glass_view3_tuned_20261123.json}
HOURS=${HOURS:-12}
WEB_HOST=${WEB_HOST:-$(tailscale ip -4 2>/dev/null | head -1)}   # Tailscale only: the page has no authentication
WEB_HOST=${WEB_HOST:-127.0.0.1}   # no Tailscale: localhost only
WEB_PORT=${WEB_PORT:-8080}
# Link-only access: the page is served at http://WEB_HOST:WEB_PORT/<token>/ (token in ~/.mitafes_web_token).
mkdir -p runs/live;echo $$ > runs/live/run_live.pid
end=$(( $(date +%s) + HOURS*3600 ))
while (( $(date +%s) < end-60 )); do
  out=runs/live/$(date +%Y%m%d_%H%M%S)
  # Web view sized for a weak uplink: 960 px, JPEG 60, up to ~6 fps, slight smoothing (~2.7 Mbps), capped at 4.5 Mbps per viewer
  # (low-data mode in the page: one still every 2 s ~ 0.3 Mbps).
  .venv/bin/python live_fused.py --setup "$SETUP" --output "$out" --duration $(( end-$(date +%s) )) \
    --web-host "$WEB_HOST" --web-port "$WEB_PORT" --web-width 960 --web-quality 60 --web-fps 7 --web-max-mbps 4.5 --web-soften 0.7 \
    --web-token-file ~/.mitafes_web_token --event-snapshots >> "$out.log" 2>&1 && break
  echo "$(date '+%F %T') live_fused exited ($?), restarting" >> runs/live/restarts.log
  sleep 10
done
