#!/usr/bin/env bash
# train.sh <source_dir> <model_dir> [export_dir] — HUGSIM reconstruction of a converted clip on
# the A10: ground model, then the full scene (configs/nusc.yaml: semantics, affine exposure,
# unicycle vehicles; 30k iterations), then the export the simulator loads (scenes/<name>/).
set -euo pipefail
H=$(cd "$(dirname "$0")/.." && pwd); SRC=$(readlink -f "$1"); MODEL=$(readlink -m "$2"); EXP=$(readlink -m "${3:-$H/data/scenes/pai/$(basename "$SRC")}")
mkdir -p "$MODEL" "$(dirname "$EXP")"
docker run --rm --gpus '"device=1"' --shm-size=16g --name "hugsim-train-$(basename "$SRC")" \
  -v "$H:$H" -e HF_HOME="$H/.hf-cache" -e HF_HUB_OFFLINE=1 -w "$H/repo" hugsim-dev:cu118 bash -c "
  set -e; P='pixi run python -u'
  echo '[train] ground model';  \$P train_ground.py --data_cfg ./configs/nusc.yaml --source_path $SRC --model_path $MODEL
  echo '[train] scene';         \$P train.py --data_cfg ./configs/nusc.yaml --source_path $SRC --model_path $MODEL
  echo '[train] export';        \$P eval_render/export_scene.py --model_path $MODEL --output_path $EXP --iteration 30000
  echo '[train] DONE'"
