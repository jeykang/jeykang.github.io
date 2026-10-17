#!/usr/bin/env bash
# Download Cosmos-Transfer1 7B checkpoints via the SIF (download_checkpoints.py needs
# torch). Run detached on the login node; HF_TOKEN passed via env (not embedded).
# Writes checkpoints to /scratch/autodr_test/cosmos-transfer1/checkpoints.
set -eo pipefail
H=/scratch/autodr_test
mkdir -p $H/hf_cache $H/cosmos-transfer1/checkpoints
singularity exec -B $H:$H --env HF_HOME=$H/hf_cache \
  $H/cosmos_transfer1.sif \
  bash -c "cd $H/cosmos-transfer1 && PYTHONPATH=\$PWD python scripts/download_checkpoints.py \
           --output_dir \$PWD/checkpoints --hf_token $HF_TOKEN --model 7b"
echo ">>> WEIGHTS DOWNLOAD DONE"
