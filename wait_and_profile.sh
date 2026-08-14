#!/usr/bin/env bash
# Wait for the GPU to go quiet, then take a clean per-stage baseline.
#
# Every timing in PLAN.md was measured while an Isaac Lab job held ~7 GB and the GPU sat
# at 68-90%, so they are worst-case. This re-runs the profile once that clears, and leaves
# the result in outputs/baseline_profile.txt for whoever looks next.
#
#   nohup ./wait_and_profile.sh > /dev/null 2>&1 &

cd "$(dirname "$0")" || exit 1
OUT=outputs/baseline_profile.txt
VIDEO=${1:-outputs/input_capture.mp4}

{
  echo "waiting for the GPU to free up — started $(date '+%F %H:%M')"
} > "$OUT"

# quiet means: no isaac process holding memory, and utilisation low for 60 s straight
quiet=0
while [ "$quiet" -lt 6 ]; do
  sleep 10
  busy=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)
  util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits)
  if [ "$busy" -eq 0 ] && [ "$util" -lt 20 ]; then
    quiet=$((quiet + 1))
  else
    quiet=0
  fi
done

{
  echo "GPU quiet at $(date '+%F %H:%M') — running the profile"
  nvidia-smi --query-gpu=utilization.gpu,memory.used,clocks.sm --format=csv,noheader
  echo
} >> "$OUT"

GEM-X/.venv/bin/python replay_delay.py "$VIDEO" outputs/baseline.mp4 \
  --ik k1 --profile 2>&1 | grep -E "inferences|delay|ms  \(n=|TOTAL|IK in-loop" >> "$OUT"

{
  echo
  echo "for comparison, the same run under contention (2026-08-12, Isaac at 68%):"
  echo "  vitpose 25.9 | predict 23.1 | fk 9.2 | ik 8.1 | TOTAL 69.5 ms -> 14.4 FPS"
  echo "  skeleton delay median 133 ms"
} >> "$OUT"
