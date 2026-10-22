#!/usr/bin/env bash
# run_closed_loop.sh — one HUGSIM closed-loop episode with the LTF client, on the A10.
#
#   run_closed_loop.sh <scenario.yaml> [ad=ltf] [base=configs/nuscenes_base_local.yaml] [camera=repo/configs/sim/nuscenes_camera.yaml]
#
# Runs inside hugsim-dev:cu118 (see Dockerfile / build_env.sh). The simulator launches the
# policy client itself (`zsh <ltf_path> <ad_cuda> <output>`, named pipes in the output dir);
# with --gpus device=1 the A10 is cuda:0 for both. Output: runs/nusc_<ad>/<scene>_<mode>/
# (data.pkl, video.mp4, eval.json with the HD-Score terms).
set -euo pipefail
H=$(cd "$(dirname "$0")" && pwd)
SCEN=$(readlink -f "$1"); AD=${2:-ltf}
BASE=$(readlink -f "${3:-$H/configs/nuscenes_base_local.yaml}")
CAM=$(readlink -f "${4:-$H/repo/configs/sim/nuscenes_camera.yaml}")
docker run --rm --gpus '"device=1"' --shm-size=16g --name "hugsim-run-$(basename "$SCEN" .yaml)" \
  -v "$H:$H" -w "$H/repo" hugsim-dev:cu118 \
  pixi run python closed_loop.py --scenario_path "$SCEN" --base_path "$BASE" --camera_path "$CAM" \
    --kinematic_path "$H/repo/configs/sim/kinematic.yaml" --ad "$AD" --ad_cuda 0
