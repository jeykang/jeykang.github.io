#!/usr/bin/env bash
# prepare.sh <source_dir> — HUGSIM's own preprocessing after load_ncore.py (nuScenes branch):
# dynamic masks, UniDepth V2 depth, merged non-ground and ground point clouds. Runs in
# hugsim-dev:cu118 on the A10. (Semantics come from load_ncore.py --aux-sseg, not InverseForm.)
set -euo pipefail
H=$(cd "$(dirname "$0")/.." && pwd); OUT=$(readlink -f "$1")
docker run --rm --gpus '"device=1"' --shm-size=16g --name hugsim-prep-$(basename "$OUT") \
  -v "$H:$H" -e HF_HOME="$H/.hf-cache" -e HF_HUB_OFFLINE=1 -e PYTHONPATH="$H/repo/data" -w "$H/repo/data" hugsim-dev:cu118 bash -c "
  set -e; P='pixi run --manifest-path $H/repo/pixi.toml python'
  echo '[prep] dynamic masks';  \$P utils/create_dynamic_mask.py --data_path $OUT --data_type nuscenes
  echo '[prep] depth';          \$P $H/pai/estimate_depth_noxf.py --out $OUT   # xformers off, see that file
  echo '[prep] points3d';       \$P utils/merge_depth_wo_ground.py --out $OUT --total 200000
  echo '[prep] ground points';  \$P utils/merge_depth_ground.py --out $OUT --total 200000 --datatype nuscenes
  echo '[prep] DONE'"
