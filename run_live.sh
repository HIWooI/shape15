#!/usr/bin/env bash
# Camera -> retarget -> Isaac, in one command.
#
# The ordering is the whole point. Isaac takes 60-90 s to come up and the reference is
# UDP, so anything the camera sends before the receiver binds is simply gone -- an early
# run lost 434 frames that way. This waits for the port to be listening, then hands the
# terminal to the camera so `c` (calibrate) and `q` (quit) still work. Isaac is killed on
# the way out, whether that is `q`, Ctrl-C, or the script failing.
#
#   ./run_live.sh                     # local window on :1
#   ./run_live.sh --stream 8080       # browser instead, camera on 8080
#
# Watch the robot at http://<this machine>:8100.
set -euo pipefail
cd "$(dirname "$0")"

CKPT=${CKPT:-logs/rsl_rl/cyclo_mimic_k1_rev1_multi/2026-08-11_20-07-44/model_14000.pt}
SEED=${SEED:-/motions/take5b/take5b.npz}
CONTAINER=${CONTAINER:-cyclo_lab_shape14_eval}
REF_PORT=${REF_PORT:-9411}
VIEW_PORT=${VIEW_PORT:-8100}
TAKE=${TAKE:-outputs/live_$(date +%m%d_%H%M)}

log() { printf '\033[36m[run_live]\033[0m %s\n' "$*"; }

cleanup() {
    log "stopping Isaac"
    docker exec "$CONTAINER" pkill -f live_play.py 2>/dev/null || true
}
trap cleanup EXIT

log "staging the runner into $CONTAINER"
docker cp live_play.py "$CONTAINER:/tools/live_play.py" >/dev/null
docker cp mjpeg.py "$CONTAINER:/tools/mjpeg.py" >/dev/null

log "starting Isaac (60-90 s)"
docker exec "$CONTAINER" bash -lc "cd /workspace/cyclo_lab_private && \
    ./third_party/IsaacLab/_isaac_sim/python.sh /tools/live_play.py \
      --checkpoint '$CKPT' --seed_clip '$SEED' --num_envs 1 \
      --port $REF_PORT --stream $VIEW_PORT" &

# The receiver binds the UDP port after the policy is loaded, so a listening port is the
# honest readiness signal -- not a fixed sleep, which is either too short or wasted time.
for _ in $(seq 120); do
    ss -lun 2>/dev/null | grep -q ":$REF_PORT " && break
    sleep 2
done
ss -lun 2>/dev/null | grep -q ":$REF_PORT " || { log "Isaac never bound :$REF_PORT"; exit 1; }

IP=$(hostname -I | awk '{print $1}')
log "ready. robot view: http://$IP:$VIEW_PORT"
log "press 'c' in the camera window to calibrate, 'q' to quit"
log "raw take -> $TAKE"

DISPLAY=${DISPLAY:-:1} GEM-X/.venv/bin/python demo_webcam.py \
    --flip --unmirror --robot k1 --mlp models/k1_retarget.pt --smooth 2.0 \
    --ref_stream "127.0.0.1:$REF_PORT" \
    --save_raw "$TAKE" --motion_command "$TAKE/mc.npz" "$@"
