#!/bin/bash
# Drive the bench page in headless Chrome and print what it reported.
#
#   ./serve.py 8099 &
#   ./run_bench.sh '[{"threads":14},{"threads":5}]' [max_wait_s] [port]
#
# The plan is a JSON array of configuration overrides; see DEFAULTS in index.html for the fields.
# Firefox cannot run this: it hangs in initThreadPool inside a nested module worker, which is how
# QSMbly's pipeline worker calls it. Do not add --virtual-time-budget, it distorts timers.
set -eo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
PLAN_JSON="${1:?usage: run_bench.sh '<json plan>' [max_wait_s] [port]}"
MAXW="${2:-3600}"
PORT="${3:-8099}"
PROFILE="$(mktemp -d)"

rm -f "$HERE/results.jsonl"
PLAN=$(python3 -c 'import sys,urllib.parse;print(urllib.parse.quote(sys.argv[1]))' "$PLAN_JSON")
URL="http://localhost:$PORT/scripts/bench-dl-threading/index.html?auto=1&plan=$PLAN"

trap 'pkill -f "user-data-dir=$PROFILE" 2>/dev/null; rm -rf "$PROFILE"' EXIT
(setsid google-chrome-stable --headless=new --user-data-dir="$PROFILE" \
  --disable-background-timer-throttling --disable-renderer-backgrounding \
  --disable-backgrounding-occluded-windows --no-first-run --no-default-browser-check \
  "$URL" > "$HERE/chrome.log" 2>&1 &)

waited=0
until grep -q all_done "$HERE/results.jsonl" 2>/dev/null || [ "$waited" -ge "$MAXW" ]; do
  sleep 5
  waited=$((waited + 5))
done
[ "$waited" -ge "$MAXW" ] && echo "WARNING: gave up after ${MAXW}s; partial results below" >&2

python3 - "$HERE/results.jsonl" <<'PY'
import json, sys
for line in open(sys.argv[1]):
    o = json.loads(line)
    if o.get("type") == "line":
        print(o["s"])
PY
