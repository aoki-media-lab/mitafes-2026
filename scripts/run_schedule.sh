#!/usr/bin/env bash
# Daily schedule for the glass counter (runs until killed). Since 2026-11-23 both directions are
# counted (in = right-to-left, out = left-to-right; the page shows in, out and in-out). Since 11/23
# the tuned setup keeps detections from 0.03, starts tracks from 0.10 and limits the gap-filling rules.
#   START-END (default 06:00-19:00): live_fused.py counts and serves the browser view;
#                                    restarted with a new run folder if it exits early.
#   other hours: scripts/idle_web.py keeps the same page/URL up with the latest day's
#                totals and chart; no camera frames are read or sent.
# Page: http://WEB_HOST:WEB_PORT/<token>/  (token in ~/.mitafes_web_token; Tailscale only).
# Stop: kill $(cat runs/live/schedule.pid), then the python child.
cd "$(dirname "$0")/.."
SETUP=${SETUP:-reference/setups/glass_view3_tuned_20261123.json}
START=${START:-06:00}
END=${END:-19:00}
WEB_HOST=${WEB_HOST:-$(tailscale ip -4 2>/dev/null | head -1)}   # Tailscale only: the page has no authentication
WEB_HOST=${WEB_HOST:-127.0.0.1}   # no Tailscale: localhost only
WEB_PORT=${WEB_PORT:-8080}
TOKEN_FILE=${TOKEN_FILE:-$HOME/.mitafes_web_token}
# Browser stream: 1280 px (720p), JPEG 60, up to 10 fps, capped at 8 Mbps per viewer
# (the cap lowers the frame rate on busy scenes). Weak uplink: WEB_WIDTH=960 WEB_FPS=7 WEB_MBPS=4.5.
WEB_WIDTH=${WEB_WIDTH:-1280}
WEB_QUALITY=${WEB_QUALITY:-60}
WEB_FPS=${WEB_FPS:-10}
WEB_MBPS=${WEB_MBPS:-8}
# Record the annotated view (12.5 fps H.264, one file per clock hour) to <run>/video; RECORD=0 turns it off.
RECORD=${RECORD:-1}
mkdir -p runs/live;echo $$ > runs/live/schedule.pid
while true; do
  now=$(date +%s); s=$(date -d "today $START" +%s); e=$(date -d "today $END" +%s)
  if (( now >= s && now < e-60 )); then
    out=runs/live/$(date +%Y%m%d_%H%M%S)
    REC=(); [ "$RECORD" = 1 ] && REC=(--record)
    # Slight smoothing (0.7) keeps JPEGs ~1/3 smaller; low-data mode in the page: one still every 2 s.
    .venv/bin/python live_fused.py --setup "$SETUP" --output "$out" --duration $(( e-now )) \
      --web-host "$WEB_HOST" --web-port "$WEB_PORT" --web-width "$WEB_WIDTH" --web-quality "$WEB_QUALITY" --web-fps "$WEB_FPS" \
      --web-max-mbps "$WEB_MBPS" --web-soften 0.7 --web-token-file "$TOKEN_FILE" --event-snapshots "${REC[@]}" >> "$out.log" 2>&1 \
      || { echo "$(date '+%F %T') live_fused exited ($?), restarting" >> runs/live/restarts.log; sleep 10; }
  else
    if (( now < s )); then until=$s; else until=$(date -d "tomorrow $START" +%s); fi
    .venv/bin/python scripts/idle_web.py --setup "$SETUP" --runs runs/live --until "$until" --end-time "$END" \
      --web-host "$WEB_HOST" --web-port "$WEB_PORT" --web-token-file "$TOKEN_FILE" >> runs/live/idle.log 2>&1 \
      || { echo "$(date '+%F %T') idle_web exited ($?), restarting" >> runs/live/restarts.log; sleep 10; }
  fi
done
