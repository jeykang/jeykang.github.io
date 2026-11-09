#!/usr/bin/env bash
# rerun_warmup.sh — HUGSIM closed-loop episodes with AlpaSim's protocol (2026-10-07).
#
#   ./rerun_warmup.sh <short> [<short> ...]       e.g. ./rerun_warmup.sh bb4394e7 a07e81de
#
# For each clip's LiDAR-seeded twin and its recorded-traffic scenario (configs/pai/<short>_lidar-rec-02.yaml):
#   constant velocity with 3.0 s of recorded driving and the handover speed held   -> runs/pai_cvwu_ltf/
#   VaVAM, seeds 0-4, 100-degree front view, 3.0 s of recorded driving              -> runs/pai_vavam_front100wu_s<seed>_ltf/
# The original runs (control from t = 0) stay where they are. Finished episodes are skipped.
set -uo pipefail
H=$(cd "$(dirname "$0")" && pwd)
log(){ echo "[$(date '+%F %T')] $*"; }
for short in "$@"; do
  SCEN=$H/configs/pai/${short}_lidar-rec-02.yaml
  [ -f "$SCEN" ] || { log "$short: no scenario yet, skipped"; continue; }
  out=$H/runs/pai_cvwu_ltf/${short}_lidar_rec_02
  if [ ! -f "$out/eval.json" ]; then
    log "$short: constant velocity (warm-up)"; "$H/run_closed_loop.sh" "$SCEN" ltf "$H/configs/pai_base_local_cv_wu.yaml" > "$H/runs/twin_queue/$short.cvwu.log" 2>&1
  fi
  for s in 0 1 2 3 4; do
    out=$H/runs/pai_vavam_front100wu_s${s}_ltf/${short}_lidar_rec_02
    [ -f "$out/eval.json" ] && continue
    log "$short: VaVAM seed $s (warm-up)"
    CAMERA="$H/configs/pai_camera_front100.yaml" TAG=front100wu VAVAM_ARGS="--warmup 3.0" \
      "$H/run_vavam.sh" "$SCEN" "$s" > "$H/runs/twin_queue/$short.vavamwu_s$s.log" 2>&1
  done
  log "$short: done"
done
