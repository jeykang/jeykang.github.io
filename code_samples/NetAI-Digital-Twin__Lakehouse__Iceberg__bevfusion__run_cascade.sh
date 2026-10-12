#!/usr/bin/env bash
# Launch the BEVFusion cascade perception run over the metadata-preselected
# cohort, sharded one container per GPU. Resumable (runner --resume default on):
# re-running this script continues from the existing shard parquets.
#
#   ./run_cascade.sh
#
# Output: <NFS>/.perception_bevfusion/bevfusion_shard_NN_of_02.parquet
# (kept separate from the retired YOLO scorer's .perception/ — see §12).
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/.."          # -> Lakehouse/Iceberg
ROOT="$PWD"
COHORT="${1:-$ROOT/bevfusion/cohort/cascade_cohort.csv}"
N="${N:-20}"
THR="${THR:-0.05}"
OUTDIR=/mnt/netai-e2e/nvidia-physicalai-av-subset/.perception_bevfusion
CFG=/workspace/mmdetection3d/projects/BEVFusion/configs/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py
CKPT=/workspace/checkpoints/bevfusion_lidar-cam_spconv2.pth

run_shard () {  # $1=device  $2=shard_id
  docker run -d --name "bevfusion_cascade_$2" --runtime=nvidia --gpus "device=$1" \
    -v "$ROOT/netai-e2e:/mnt/netai-e2e" \
    -v "$ROOT/bevfusion/checkpoints:/workspace/checkpoints:ro" \
    -v "$ROOT/bevfusion/cohort:/work/cohort:ro" \
    --entrypoint bash netai/bevfusion-runner:latest -c "
      setpriv --reuid=1000 --regid=1007 --clear-groups env HOME=/tmp \
        python /workspace/runner.py \
          --clip-list /work/cohort/$(basename "$COHORT") \
          --config $CFG --checkpoint $CKPT \
          --frames-per-clip $N --score-thr $THR \
          --shard-id $2 --n-shards 2 --resume \
          --output-dir $OUTDIR \
          --nfs-root /mnt/netai-e2e/nvidia-physicalai-av-subset 2>&1
    "
}

# remove any prior cascade containers (resume picks up their parquet output)
docker rm -f bevfusion_cascade_0 bevfusion_cascade_1 >/dev/null 2>&1 || true
run_shard 0 0
run_shard 1 1
echo "launched cascade shards:"; docker ps --filter name=bevfusion_cascade --format '{{.Names}} {{.Status}}'
