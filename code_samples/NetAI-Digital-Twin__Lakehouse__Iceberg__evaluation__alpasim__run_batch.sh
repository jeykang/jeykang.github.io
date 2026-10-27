#!/usr/bin/env bash
# run_batch.sh — a closed-loop batch over catalog NuRec scenes, in resumable chunks.
#
#   run_batch.sh <scene-list.txt> <run-prefix> [chunk=16]
#
# The list holds one scene id per line (clipgt-<clip-uuid>; other lines are ignored). Each
# chunk is one AlpaSim launch (run_scene.sh) over up to <chunk> scenes, written to
# runs/<run-prefix>-cNN/ and summarised by per_clip.py into runs/<run-prefix>-cNN/per_clip.parquet.
# A chunk whose per_clip.parquet exists is skipped, so re-running the same command resumes.
#
# Disk: every scene the wizard downloads stays in repo/data/nre-artifacts/all-usdzs (~1.7 GB
# each). Before each chunk the script checks for ~chunk x 2 GB + 5 GB free and stops (exit 3)
# if it is not there. It deletes nothing itself: freeing the scene cache is the user's call
# (an assistant-initiated prune was refused by the permission classifier on 2026-10-06).
#
# Environment, passed through to run_scene.sh: DRIVER (default here: vavam, AlpaSim's native
# VaVAM driver, as in batches 1-3), N_ROLLOUTS (default 1), EXTRA_ARGS.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
LIST=$(readlink -f "${1:?usage: run_batch.sh <scene-list.txt> <run-prefix> [chunk]}")
PREFIX=${2:?run prefix}; CHUNK=${3:-16}
export DRIVER=${DRIVER:-vavam} N_ROLLOUTS=${N_ROLLOUTS:-1}
mapfile -t IDS < <(grep -E '^clipgt-[0-9a-f-]+$' "$LIST")
n=${#IDS[@]}; nc=$(( (n + CHUNK - 1) / CHUNK ))
log(){ echo "[$(date '+%F %T')] $*"; }
log "batch $PREFIX: $n scenes, $nc chunks of <= $CHUNK, driver=$DRIVER, n_rollouts=$N_ROLLOUTS"
for ((c = 0; c < nc; c++)); do
  R=$HERE/runs/$(printf '%s-c%02d' "$PREFIX" $((c + 1)))
  chunk=("${IDS[@]:c*CHUNK:CHUNK}")
  if [ -f "$R/per_clip.parquet" ]; then log "chunk $((c + 1))/$nc already done"; continue; fi
  avail=$(df --output=avail -BG "$HERE" | tail -1 | tr -dc '0-9')
  need=$(( ${#chunk[@]} * 2 + 5 ))
  if [ "$avail" -lt "$need" ]; then log "only ${avail} GB free, chunk needs ~${need} GB: stopping"; exit 3; fi
  mkdir -p "$R"; printf '%s\n' "${chunk[@]}" > "$R.scenes.txt"
  log "chunk $((c + 1))/$nc: ${#chunk[@]} scenes -> $R (${avail} GB free)"
  RUN_DIR=$R "$HERE/run_scene.sh" "$DRIVER" "${chunk[@]}" > "$R.log" 2>&1
  log "  run_scene exit $?; aggregate: $(ls "$R/aggregate" 2>/dev/null | tr '\n' ' ')"
  "$HERE/repo/.venv/bin/python" "$HERE/per_clip.py" "$R" -o "$R/per_clip.parquet" > "$R.per_clip.txt" 2>&1
  log "  per_clip: $(grep -E 'validation|wrote' "$R.per_clip.txt" | tr '\n' ' ')"
done
log "batch $PREFIX finished"
