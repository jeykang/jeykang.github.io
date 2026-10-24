#!/usr/bin/env bash
# twin_pipeline.sh — one PhysicalAI-AV clip -> our NuRec twin -> AlpaSim A/B against NVIDIA's scene.
#
#   twin_pipeline.sh <clip-uuid> [--skip-rollouts] [--keep-intermediates]
#
# Steps (all on this host; the A10 is GPU 1, NVDEC for the converter is GPU 0):
#   1. stage + NCore v4 convert          ncore/stage_pai_clip.py, NVIDIA's pai converter   (~5 min)
#   2. Instant NuRec initialisation       nurec/instant_nurec (merged PLY)                  (~5 min)
#   3. aux store                          nre-tools-ga ncore-aux-data (Mask2Former ...)     (~2 h)
#   4. NRE training, prod config          nre-ga, 30k steps, artifacts every 10k           (~2 h)
#   5. post-hoc ground mesh               nre-ga export-ground-mesh
#   6. NVIDIA's scene of the same clip    HF sample_set/26.04_release (gated token)
#   7. re-base onto NVIDIA's time axis    nurec/alpasim_bundle.sh (JSON/USDA/metadata + checkpoint)
#   8. A/B                                ours_map (our scene + NVIDIA's map layers) vs nvidia,
#                                         constant_velocity and VaVAM; per_clip tables
#   9. cleanup                            keeps the twin (ours_map.usdz), last.usdz, the reference,
#                                         aux + staged inputs; drops checkpoints/intermediate artifacts
#                                         and the 3 GB NCore zarr store (regenerable in minutes)
# Outputs: nurec/out/<short>_a10_prod/, alpasim/local_scenes_<short>_{ours_map,nvidia}/,
#          alpasim/runs/<short>_{ours_map,nvidia}_{cv,vavam}/, nurec/out/<short>.twin.json (timings).
set -uo pipefail
E=$(cd "$(dirname "$0")/.." && pwd); NUREC=$E/nurec; NCORE=$E/ncore; ALPA=$E/alpasim
ROOT=${PAI_ROOT:-$E/../netai-e2e/nvidia-physicalai-av-subset}
NRE=${NRE_IMAGE:-nvcr.io/nvidia/nre/nre-ga:26.04}; NRETOOLS=${NRE_TOOLS_IMAGE:-nvcr.io/nvidia/nre/nre-tools-ga:26.04}
CFG=configs/apps/prod/Hyperion-8.1/car2sim_6cam.yaml
C=${1:?clip uuid}; shift; SKIP_ROLLOUTS=0; KEEP=0
for a in "$@"; do case $a in --skip-rollouts) SKIP_ROLLOUTS=1;; --keep-intermediates) KEEP=1;; esac; done
short=${C:0:8}; RUN=${short}_a10_prod; T=$NUREC/usdz_tools.py
log(){ echo "[$(date +%H:%M:%S)] [$short] $*"; }
J=$NUREC/out/$short.twin.json; declare -A TM; t0=$(date +%s)
mark(){ TM[$1]=$(( $(date +%s) - t0 )); python3 - "$J" "$C" "$1" "${TM[$1]}" "${2:-ok}" <<'PY'
import json, sys, os
p, clip, step, t, st = sys.argv[1:]; d = json.load(open(p)) if os.path.exists(p) else {"clip": clip, "steps": {}}
old = d["steps"].get(step)
# a re-run skips steps that are already done; keep their original timing instead of recording ~0 s
if not (old and old.get("status") == "ok" and st == "ok"):
    d["steps"][step] = {"t_s": int(t), "status": st}
json.dump(d, open(p, "w"), indent=1)
PY
}
fail(){ log "FAILED at $1"; mark "$1" failed; exit 1; }
mkdir -p $NUREC/out $NCORE/staged $NCORE/out

# 1. stage + convert
if [ ! -f $NCORE/out/pai_$C/pai_$C.json ]; then
  log "stage + convert"
  $NCORE/.venv/bin/python $NCORE/stage_pai_clip.py --root "$ROOT" --clip $C --out $NCORE/staged --allow-empty > $NUREC/out/$short.stage.log 2>&1 || fail stage
  (cd $NCORE/repo && PYTHONPATH=$NCORE/repo $NCORE/.venv/bin/python -m tools.data_converter.pai.converter --root-dir $NCORE/staged --output-dir $NCORE/out pai-v4 --clip-id $C) > $NUREC/out/$short.convert.log 2>&1 || fail convert
fi
[ -f $NCORE/out/pai_$C/pai_$C.json ] || fail convert; mark convert

# 6 (early, network only). NVIDIA's scene of the same clip
REFDIR=$ALPA/local_scenes_ref_$C; mkdir -p $REFDIR
if ! ls $REFDIR/*.usdz >/dev/null 2>&1; then
  log "downloading NVIDIA's scene"
  HF_TOKEN=$(cat ~/.cache/huggingface/token) $NCORE/.venv/bin/python - "$C" "$REFDIR" <<'PY' > $NUREC/out/$short.ref.log 2>&1 || fail reference
import os, sys; from huggingface_hub import hf_hub_download
c, d = sys.argv[1:]
p = hf_hub_download("nvidia/PhysicalAI-Autonomous-Vehicles-NuRec", f"sample_set/26.04_release/{c}/{c}.usdz", repo_type="dataset", token=os.environ["HF_TOKEN"], local_dir=f"{d}/.hf")
os.link(p, f"{d}/{c}.usdz"); print(p)
PY
fi
REF=$(ls $REFDIR/*.usdz | head -1); [ -n "$REF" ] || fail reference; mark reference

# 2. Instant NuRec (only needed to start training; cleanup removes it afterwards)
PLY=$(find $NUREC/out/instant_$C -name "pai_$C.ply" 2>/dev/null | head -1)
if [ -z "$PLY" ] && [ ! -f $NUREC/out/$RUN/artifacts/last.usdz ]; then
  log "Instant NuRec"
  (cd $NUREC/instant_nurec && CUDA_VISIBLE_DEVICES=1 .venv/bin/python run_inference.py --ncore-path $NCORE/out/pai_$C/pai_$C.json --output-dir $NUREC/out/instant_$C --merge) > $NUREC/out/$short.instant.log 2>&1
  PLY=$(find $NUREC/out/instant_$C -name "pai_$C.ply" 2>/dev/null | head -1)
fi
[ -n "$PLY" ] || [ -f $NUREC/out/$RUN/artifacts/last.usdz ] || fail instant; mark instant

# 3. aux store (beside the NCore store, where the prod config's road-semantic base looks for it)
if [ ! -f $NCORE/out/pai_$C/pai_$C.aux.sseg.zarr.itar ] && [ ! -f $NUREC/out/$RUN/artifacts/last.usdz ]; then
  log "aux store (nre-tools-ga)"; mkdir -p $NUREC/aux_$C
  docker run --rm --gpus '"device=1"' --shm-size=16g -v $NCORE/out/pai_$C:/workdir/dataset:ro -v $NUREC/aux_$C:/workdir/output --name nre-aux-$short $NRETOOLS \
    ncore-aux-data --dataset-path=/workdir/dataset/pai_$C.json --output-dir=/workdir/output \
    --camera-id=camera_cross_left_120fov --camera-id=camera_cross_right_120fov --camera-id=camera_front_tele_30fov --camera-id=camera_front_wide_120fov --camera-id=camera_rear_left_70fov --camera-id=camera_rear_right_70fov --lidar-id=lidar_top_360fov \
    --segmentation-backend=mask2former --no-seg-logits --lidar-seg-camvis --store-meta > $NUREC/out/$short.aux.log 2>&1
  sudo -n chown -R $(id -u):$(id -g) $NUREC/aux_$C 2>/dev/null
  cp $NUREC/aux_$C/pai_$C.aux* $NCORE/out/pai_$C/ 2>/dev/null
fi
[ -f $NCORE/out/pai_$C/pai_$C.aux.sseg.zarr.itar ] || [ -f $NUREC/out/$RUN/artifacts/last.usdz ] || fail aux; mark aux

# 4. training
if [ ! -f $NUREC/out/$RUN/artifacts/last.usdz ]; then
  log "training $RUN ($CFG)"
  docker run --rm --gpus '"device=1"' --shm-size=64g \
    -v $NCORE/out/pai_$C:/workdir/dataset:ro -v $NUREC/out:/workdir/output -v $(dirname "$PLY"):/workdir/instant_nurec:ro \
    --name nre-train-$RUN $NRE mode=trainval out_dir=/workdir/output logger.run_id=$RUN --config-name=$CFG \
    dataset.path=/workdir/dataset/pai_$C.json dataset.lidar_ids=[lidar_top_360fov] \
    checkpoint.every_n_train_steps=10000 checkpoint.artifact.nrend.enabled=true checkpoint.artifact.mesh.ground.enabled=false \
    model/gaussians/initialization@model.layers.background.initialization=nrm_ply \
    model.layers.background.initialization.path=/workdir/instant_nurec/pai_$C.ply \
    model.layers.background.initialization.num_point_cloud_points=2000000 dataset.n_samples_per_epoch=30000 \
    model.layers.background.optimizers.0.params.positions.args.lr=5.06e-5 \
    model.layers.dynamic_rigids.optimizers.0.params.positions.args.lr=4e-5 > $NUREC/out/$RUN.log 2>&1
  log "training exit $?"; sudo -n chown -R $(id -u):$(id -g) $NUREC/out/$RUN 2>/dev/null
fi
[ -f $NUREC/out/$RUN/artifacts/last.usdz ] || fail train; mark train
PSNR=$(grep -A2 "^  test/psnr:" $NUREC/out/$RUN/val/metrics.yaml 2>/dev/null | grep value | awk '{print $2}'); log "test/psnr ${PSNR:-n/a}"

# 5. ground mesh
if [ ! -f $NUREC/out/$RUN/ground/mesh_ground.ply ]; then
  log "ground mesh"
  docker run --rm --gpus '"device=1"' --shm-size=16g -v $NCORE/out/pai_$C:/workdir/dataset:ro -v $NUREC/out:/workdir/output \
    $NRE export-ground-mesh --config-name=/workdir/output/$RUN/config/parsed.yaml --output-dir=/workdir/output/$RUN/ground > $NUREC/out/$RUN.ground.log 2>&1
  sudo -n chown -R $(id -u):$(id -g) $NUREC/out/$RUN 2>/dev/null
fi
MESH=$(ls $NUREC/out/$RUN/ground/*.ply 2>/dev/null | head -1); [ -n "$MESH" ] && mark ground || mark ground missing

# 7. re-base (+ mesh) and build the variants
B=$NUREC/out/$RUN/ab.usdz
if [ ! -f $ALPA/local_scenes_${short}_ours_map/ours_map.usdz ]; then
  if [ ! -f $B ]; then
    $NUREC/alpasim_bundle.sh $NUREC/out/$RUN/artifacts/last.usdz $B --from-reference "$REF" ${MESH:+--add "$MESH"} > $NUREC/out/$short.bundle.log 2>&1 || fail bundle
  fi
  for v in ours_map nvidia; do d=$ALPA/local_scenes_${short}_$v; rm -rf $d; mkdir -p $d; done
  python3 $T inject $B "$REF" $ALPA/local_scenes_${short}_ours_map/ours_map.usdz >> $NUREC/out/$short.bundle.log 2>&1 || fail inject
  ln -f "$REF" $ALPA/local_scenes_${short}_nvidia/$(basename "$REF")
  python3 $T fix-meta $ALPA/local_scenes_${short}_ours_map/ours_map.usdz --scene-id clipgt-$C >/dev/null 2>&1
fi
mark bundle

# 8. A/B
if [ $SKIP_ROLLOUTS -eq 0 ]; then
  for pol in cv vavam; do
    if [ $pol = cv ]; then DRV=harness; SPEC=constant_velocity; else DRV=vavam; SPEC=vavam; fi
    for v in ours_map nvidia; do
      R=$ALPA/runs/${short}_${v}_$pol; [ -f $R/per_clip.parquet ] && continue
      log "AlpaSim $SPEC on $v"; rm -rf $R
      # recorded-waypoint routes on both twins: the map-based generator can fold back once a policy leaves the
      # recorded line (seen with VaVAM), and the A/B wants the same route logic on both; the map still scores
      DRIVER=$DRV EXTRA_ARGS="runtime.simulation_config.route_generator_type=RECORDED" LOCAL_USDZ_DIR=$ALPA/local_scenes_${short}_$v RUN_DIR=$R $ALPA/run_scene.sh $SPEC > $R.log 2>&1
      log "  exit $? ; $(ls $R/aggregate 2>/dev/null | tr '\n' ' ')"
      $ALPA/repo/.venv/bin/python $ALPA/per_clip.py $R -o $R/per_clip.parquet 2>&1 | tail -1
    done
  done
  mark rollouts
fi

# 9. cleanup
if [ $KEEP -eq 0 ]; then
  rm -f $B; rm -rf $NUREC/out/$RUN/checkpoints; ls $NUREC/out/$RUN/artifacts/ | grep -v "^last.usdz$" | sed "s#^#$NUREC/out/$RUN/artifacts/#" | xargs -r rm -f
  rm -f $NCORE/out/pai_$C/pai_$C.ncore4*.zarr.itar; rm -rf $NUREC/out/instant_$C
  mark cleanup
fi
log "DONE in $(( ($(date +%s)-t0)/60 )) min"
