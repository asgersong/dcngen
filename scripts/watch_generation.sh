#!/usr/bin/env bash
# Live progress of a generation run: completed scenario count,
# throughput, and ETA, appended one line every two minutes so the pane keeps
# a readable history. The driver itself prints only its final report, so this
# is the thing to watch while a multi-day run is in flight.
#
#   scripts/watch_generation.sh [out_dir] [total_rows]
set -u
OUT="${1:-data/v1}"
TOTAL="${2:-3016}"

count() { find "$OUT" -maxdepth 2 -name card.json 2>/dev/null | wc -l; }

START_TS=$(date +%s)
START_N=$(count)
echo "watching $OUT — $START_N/$TOTAL complete at start ($(date '+%F %H:%M'))"

while true; do
  N=$(count)
  ELAPSED=$(( $(date +%s) - START_TS ))
  DONE=$(( N - START_N ))
  LINE=$(printf '%s  %5d/%-5d scenarios' "$(date '+%m-%d %H:%M')" "$N" "$TOTAL")
  if [ "$DONE" -gt 0 ] && [ "$ELAPSED" -gt 300 ]; then
    LINE+=$(python3 -c "
done_, el, rem = $DONE, $ELAPSED, $TOTAL - $N
rate = done_ / (el / 3600.0)
eta = rem / rate if rate > 0 else float('inf')
print(f'  +{done_} this session  {rate:6.1f} rows/h  ETA {eta:5.1f} h')
")
  fi
  echo "$LINE"
  [ "$N" -ge "$TOTAL" ] && { echo "== all $TOTAL scenarios present =="; break; }
  sleep 120
done
