#!/usr/bin/env bash
# twin_hugsim.sh — one PhysicalAI-AV clip -> HUGSIM twin (LiDAR-seeded) -> closed loop, replayed as recorded.
#
#   pai/twin_hugsim.sh <clip-uuid> [--keep-intermediates]
#
# The HUGSIM counterpart of nurec/twin_pipeline.sh, for the cross-simulator comparison at n = 9
# (HUGSIM.md, "Same policy, same clip"). The A10 is GPU 1; the RTX 6000 (GPU 0) decodes video
# for the converter and runs the VaVAM client. Steps:
#   1. NCore v4 store with camera streams   ncore/staged/<clip> -> ncore/out_hugsim/pai_<clip>   (temporary; ~3 min)
#      (twin_pipeline.sh dropped the decoded streams of ncore/out/pai_<clip>; its aux files are reused)
#   2. HUGSIM input                         pai/load_ncore.py --aux-sseg                         (~1 min)
#   3. HUGSIM preprocessing                 pai/prepare.sh: masks, UniDepth, merged clouds       (~10 min, A10)
#   4. LiDAR seeding                        pai/lidar_points.py -> data/pai/<short>_lidar         (~3 min)
#   5. reconstruction + export              pai/train.sh -> data/scenes/pai/<short>_lidar        (~2 h 20 min, A10)
#   6. recorded vehicles as actor assets    pai/export_actors.py -> <scene>/actors/
#   7. scenario                             pai/make_scenario.py -> configs/pai/<short>_lidar-rec-02.yaml
#   8. closed loop                          VaVAM seeds 0-4 on the 100-degree front view (run_vavam.sh),
#                                           constant velocity (run_closed_loop.sh)
#   9. summary                              summarize_runs.py -> runs/twin_queue/<short>.summary.txt
#  10. cleanup                              only what THIS run created: the temporary NCore store,
#                                           training checkpoints and point-cloud renders, depth maps,
#                                           masks, rectified images and semantic labels (all regenerable
#                                           with steps 1-3; the export is self-contained). Kept: the
#                                           export, meta_data.json, test renders, results.json, the runs.
# Disk guard: a clip whose scene is not built yet starts only with MIN_FREE_GB (default 12) free;
# its peak is ~8 GB, so the volume never drops below ~4 GB (it also hosts the catalog database).
# Re-running skips finished steps. Per-step timings: runs/twin_queue/<short>.json; logs beside it.
set -uo pipefail
H=$(cd "$(dirname "$0")/.." && pwd); E=$(cd "$H/.." && pwd); NCORE=$E/ncore
C=${1:?usage: twin_hugsim.sh <clip-uuid> [--keep-intermediates]}; shift
KEEP=0; for a in "$@"; do [ "$a" = --keep-intermediates ] && KEEP=1; done
short=${C:0:8}; NAME=${short}_lidar
SRC=$H/data/pai/$short; LSRC=$H/data/pai/$NAME; MODEL=$H/data/models/pai_$NAME; SCENE=$H/data/scenes/pai/$NAME
TMP=$NCORE/out_hugsim/pai_$C; STORE=$TMP/pai_$C.json; AUX=$NCORE/out/pai_$C/pai_$C.aux.sseg.zarr.itar
SCEN=$H/configs/pai/$NAME-rec-02.yaml
Q=$H/runs/twin_queue; mkdir -p "$Q"; L=$Q/$short; J=$Q/$short.json
log(){ echo "[$(date '+%F %T')] [$short] $*"; }
t0=$(date +%s); tp=$t0
mark(){ local now; now=$(date +%s); python3 - "$J" "$C" "$1" "$((now - t0))" "$((now - tp))" "${2:-ok}" <<'PY'
import json, os, sys
p, clip, step, t, dt, st = sys.argv[1:]
d = json.load(open(p)) if os.path.exists(p) else {"clip": clip, "steps": {}}
old = d["steps"].get(step)
# a re-run skips finished steps: keep their original timing instead of recording ~0 s
if not (old and old.get("status") == "ok" and st in ("ok", "skipped")):
    d["steps"][step] = {"t_s": int(t), "step_s": int(dt), "status": st}
json.dump(d, open(p, "w"), indent=1)
PY
tp=$now; }
fail(){ log "FAILED at $1 (see $L.$1.log)"; mark "$1" failed; exit 1; }
MADE_STORE=0; PREPARED=0; TRAINED=0
[ -f "$AUX" ] || fail aux-missing
if [ ! -f "$SCENE/scene.pth" ]; then
  free=$(df -BG --output=avail "$H" | tail -1 | tr -dc '0-9')
  if [ "$free" -lt "${MIN_FREE_GB:-12}" ]; then log "only ${free} GB free (< ${MIN_FREE_GB:-12}): not starting"; mark disk-guard failed; exit 3; fi
fi

# 1-4 only matter until the scene is exported
if [ ! -f "$SCENE/scene.pth" ]; then
  if [ ! -f "$LSRC/lidar_points.json" ]; then
    if ! ls "$TMP"/pai_$C.ncore4-camera_front_wide_120fov*.zarr.itar >/dev/null 2>&1; then
      log "1. NCore store with camera streams"; mkdir -p "$(dirname "$TMP")"
      (cd "$NCORE/repo" && CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$NCORE/repo" "$NCORE/.venv/bin/python" -m tools.data_converter.pai.converter \
         --root-dir "$NCORE/staged" --output-dir "$(dirname "$TMP")" pai-v4 --clip-id "$C") > "$L.convert.log" 2>&1 || fail convert
      MADE_STORE=1
    fi
    [ -f "$STORE" ] || fail convert; mark convert
    if [ ! -f "$SRC/meta_data.json" ]; then
      log "2. load_ncore (rectify to pinhole, aux semantics, boxes)"
      (cd "$NCORE" && PYTHONPATH=repo .venv/bin/python "$H/pai/load_ncore.py" --ncore "$STORE" --out "$SRC" --aux-sseg "$AUX") > "$L.load.log" 2>&1 || fail load
    fi
    mark load
    if [ ! -f "$SRC/ground_points3d.ply" ]; then
      log "3. prepare (dynamic masks, UniDepth, clouds)"; "$H/pai/prepare.sh" "$SRC" > "$L.prepare.log" 2>&1 || fail prepare
      PREPARED=1
    fi
    mark prepare
    log "4. LiDAR seeding"
    (cd "$NCORE" && PYTHONPATH=repo .venv/bin/python "$H/pai/lidar_points.py" --ncore "$STORE" --src "$SRC" --out "$LSRC") > "$L.lidar.log" 2>&1 || fail lidar
    [ -f "$LSRC/lidar_points.json" ] || fail lidar
  fi
  mark lidar
  # the NCore store is not needed for training: free its ~3 GB before the longest step
  if [ $KEEP -eq 0 ] && [ $MADE_STORE -eq 1 ]; then rm -rf "$TMP"; fi
  log "5. reconstruction (ground model, scene, export)"
  "$H/pai/train.sh" "$LSRC" "$MODEL" "$SCENE" > "$L.train.log" 2>&1 || fail train
  TRAINED=1
fi
[ -f "$SCENE/scene.pth" ] || fail train; mark train
PSNR=$(python3 -c "import json; r = json.load(open('$MODEL/results.json'))['test']['30000']; print(f\"{r['psnr']:.2f} dB, SSIM {r['ssim']:.3f}, LPIPS {r['lpips']:.3f}\")" 2>/dev/null)
log "   held-out: ${PSNR:-n/a}"

if [ ! -f "$SCENE/actors/actors.json" ]; then
  log "6. actor assets"
  docker run --rm --gpus '"device=1"' -v "$H:$H" -w "$H/repo" hugsim-dev:cu118 \
    pixi run python "$H/pai/export_actors.py" "$SCENE" > "$L.actors.log" 2>&1 || fail actors
fi
mark actors

log "7. scenario"
python3 "$H/pai/make_scenario.py" "$SCENE" "$SRC" "$SCEN" > "$L.scenario.log" 2>&1 || fail scenario
log "   $(tail -1 "$L.scenario.log" | sed 's#.*: start_velo#start_velo#')"; mark scenario

for s in 0 1 2 3 4; do
  out=$H/runs/pai_vavam_front100_s${s}_ltf/${NAME}_rec_02
  [ -f "$out/eval.json" ] && continue
  log "8. VaVAM seed $s (100-degree front view)"
  CAMERA="$H/configs/pai_camera_front100.yaml" TAG=front100 "$H/run_vavam.sh" "$SCEN" "$s" > "$L.vavam_s$s.log" 2>&1
  [ -f "$out/eval.json" ] || log "   no eval.json for seed $s (see $L.vavam_s$s.log)"
done
out=$H/runs/pai_cv_ltf/${NAME}_rec_02
if [ ! -f "$out/eval.json" ]; then
  log "8. constant velocity"; "$H/run_closed_loop.sh" "$SCEN" ltf "$H/configs/pai_base_local_cv.yaml" > "$L.cv.log" 2>&1
fi
mark rollouts

docker run --rm -v "$H:$H" -w "$H/repo" hugsim-dev:cu118 pixi run python "$H/summarize_runs.py" \
  "runs/pai_vavam_front100_s*_ltf/${NAME}_rec_02" "runs/pai_cv_ltf/${NAME}_rec_02" > "$Q/$short.summary.txt" 2>&1
grep -E "rec_02" "$Q/$short.summary.txt" | sed "s#^#[$short]   #"; mark summary

if [ $KEEP -eq 0 ]; then
  [ $MADE_STORE -eq 1 ] && rm -rf "$TMP"
  [ $TRAINED -eq 1 ] && rm -rf "$MODEL/ckpts" "$MODEL/point_cloud_vis"
  [ $PREPARED -eq 1 ] && rm -rf "$SRC/depth" "$SRC/masks" "$SRC/images" "$SRC/semantics" \
                                 "$LSRC/depth" "$LSRC/masks" "$LSRC/images" "$LSRC/semantics"
  mark cleanup
fi
log "DONE in $(( ($(date +%s) - t0) / 60 )) min; $(df -BG --output=avail "$H" | tail -1 | tr -d ' ') free"
