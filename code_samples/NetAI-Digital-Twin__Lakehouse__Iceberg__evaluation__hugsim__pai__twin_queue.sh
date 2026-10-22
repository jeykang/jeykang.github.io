#!/usr/bin/env bash
# twin_queue.sh — run pai/twin_hugsim.sh over several clips, one after another (one A10).
#
#   pai/twin_queue.sh <clip-uuid> [<clip-uuid> ...]
#
# A clip that fails is logged and the queue moves on; re-running the same command resumes
# (finished steps are skipped). Stops early if fewer than MIN_FREE_GB (default 15) are free.
set -uo pipefail
H=$(cd "$(dirname "$0")/.." && pwd)
log(){ echo "[$(date '+%F %T')] [queue] $*"; }
log "start: $# clips"
for C in "$@"; do
  free=$(df -BG --output=avail "$H" | tail -1 | tr -dc '0-9')
  if [ "$free" -lt "${MIN_FREE_GB:-15}" ]; then log "only ${free} GB free: stopping before ${C:0:8}"; exit 3; fi
  "$H/pai/twin_hugsim.sh" "$C" || log "${C:0:8} failed (exit $?), continuing"
done
log "queue finished"
