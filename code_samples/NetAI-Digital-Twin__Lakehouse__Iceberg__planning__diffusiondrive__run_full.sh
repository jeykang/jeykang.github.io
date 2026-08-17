#!/usr/bin/env bash
# Full rung-1 run: DiffusionDrive mode_spread over all lidar-covered clips,
# sharded one detached container per GPU. Resumable (re-run to continue).
# Writes to a STAGING dir on NFS; finalize.sh swaps it into .planning/ (after
# archiving the rung-0 CV parquet) once both shards finish.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/../.."          # -> Lakehouse/Iceberg
ROOT="$PWD"
STAGE=/mnt/netai-e2e/nvidia-physicalai-av-subset/.planning_dd_staging

run_shard () {  # $1=device  $2=shard_id
  docker run -d --name "dd_run_$2" --runtime=nvidia --gpus "device=$1" \
    -v "$ROOT/netai-e2e:/mnt/netai-e2e" \
    --user 1000:1007 -e HOME=/tmp \
    --entrypoint python netai/diffusiondrive-runner:latest \
      /workspace/DiffusionDrive/runner.py \
        --output-dir "$STAGE" --shard-id "$2" --n-shards 2 --resume
}

docker rm -f dd_run_0 dd_run_1 >/dev/null 2>&1 || true
run_shard 0 0
run_shard 1 1
echo "launched:"; docker ps --filter name=dd_run --format '{{.Names}} {{.Status}}'
