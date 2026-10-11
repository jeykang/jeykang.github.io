#!/usr/bin/env bash
# rerun_after_fixes.sh — the reruns the 2026-10-07 protocol fixes call for, once the A10 is free.
#
#   nohup setsid ./rerun_after_fixes.sh > hugsim/runs/twin_queue/after_fixes.log 2>&1 &
#
# 1. refuses to start while the A10 (second GPU by PCI order) has more than 2 GB in use;
# 2. HUGSIM warm-up episodes for the clips the queue built last (hugsim/rerun_warmup.sh);
# 3. AlpaSim constant velocity on NVIDIA's scene of each twin clip with the fixed policy bridge
#    (recorded routes, as in the twin A/B) -> alpasim/runs/<short>_nvidia_cv2/per_clip.parquet;
#    the first run's driver log must show the bridge's pose-history speed, or the rest is skipped;
# 4. rollouts.py, so compare_simulators.py reads everything.
set -uo pipefail
E=$(cd "$(dirname "$0")" && pwd); A=$E/alpasim
log(){ echo "[$(date '+%F %T')] $*"; }
CLIPS="ac73935a-548f-402f-8a6b-16688261b219 bb4394e7-7485-43f3-877c-f83e40d48a13 a07e81de-291b-4c47-bab8-7216e1e2d348 0ec48454-b00b-4966-a847-1156eb4f8bcc a2bd8a78-0011-4790-9981-cb48bceb3896 abd45a30-146a-40c0-9f78-9ce697ac3ade 44c3b4d5-de26-4ba6-b748-c3a22730a717 e848c843-d275-4f60-a194-61cc7f699124 ba91fe2c-c4bc-458f-8708-4ce345e675c7"
# (a pgrep-based wait for the twin queue matched unrelated processes and stalled for 26 h on
# 2026-10-07; start this only when the A10 is free instead)
used=$(nvidia-smi --query-gpu=pci.bus_id,memory.used --format=csv,noheader,nounits | sort | awk -F', ' 'NR==2{print $2}')
if [ "${used:-0}" -gt 2000 ]; then log "the A10 has ${used} MiB in use: not starting"; exit 3; fi
log "2. HUGSIM warm-up episodes for the remaining clips"
"$E/hugsim/rerun_warmup.sh" ac73935a bb4394e7 a07e81de 0ec48454 a2bd8a78 abd45a30 44c3b4d5 e848c843 ba91fe2c
log "3. AlpaSim constant velocity, fixed bridge"
first=1
for C in $CLIPS; do
  short=${C:0:8}; R=$A/runs/${short}_nvidia_cv2; D=$A/local_scenes_${short}_nvidia
  [ -f "$R/per_clip.parquet" ] && continue
  ls "$D"/*.usdz >/dev/null 2>&1 || { log "  $short: no local NVIDIA scene in $D, skipped"; continue; }
  log "  $short"
  DRIVER=harness EXTRA_ARGS="runtime.simulation_config.route_generator_type=RECORDED" LOCAL_USDZ_DIR=$D RUN_DIR=$R \
    "$A/run_scene.sh" constant_velocity > "$R.log" 2>&1
  "$A/repo/.venv/bin/python" "$A/per_clip.py" "$R" -o "$R/per_clip.parquet" > "$R.per_clip.txt" 2>&1
  if [ $first -eq 1 ]; then
    first=0
    if grep -rqa "from the pose history" "$R" 2>/dev/null; then log "  bridge fallback confirmed in the driver log"
    else log "  bridge fallback NOT found in $R: stopping before the other clips"; exit 4; fi
  fi
done
log "4. rollouts"
"$E/.skip_venv/bin/python" "$E/rollouts.py" | head -1
log "done"
