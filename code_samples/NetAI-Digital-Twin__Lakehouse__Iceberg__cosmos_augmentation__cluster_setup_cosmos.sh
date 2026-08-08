#!/usr/bin/env bash
# Build the cosmos-transfer1 conda env on /scratch (bare conda; cu128 wheels + the
# node's driver 575 -> no container needed). Follows INSTALL.md order exactly.
# Run detached on the login node; poll /scratch/autodr_test/cosmos_env_setup.log.
set -eo pipefail
cd /scratch/autodr_test
log(){ echo "[setup $(date +%H:%M:%S)] $*"; }

# 1. miniconda on /scratch
if [ ! -x /scratch/autodr_test/miniconda3/bin/conda ]; then
  log "installing miniconda"
  wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /scratch/autodr_test/mc.sh
  bash /scratch/autodr_test/mc.sh -b -p /scratch/autodr_test/miniconda3
fi
source /scratch/autodr_test/miniconda3/etc/profile.d/conda.sh

cd /scratch/autodr_test/cosmos-transfer1
git submodule update --init --recursive 2>&1 | tail -2 || true

# 2. conda env (python 3.12)
if ! conda env list | grep -q cosmos-transfer1; then
  log "creating conda env"; conda env create --file cosmos-transfer1.yaml
fi
conda activate cosmos-transfer1

# 3. deps (exact INSTALL.md sequence)
B=https://github.com/nvidia-cosmos/cosmos-dependencies/releases/download/v1.1.0
log "pip requirements"; pip install -q -r requirements.txt
log "flashinfer/vllm/decord"
pip install -q "https://download.pytorch.org/whl/cu128/flashinfer/flashinfer_python-0.2.5%2Bcu128torch2.7-cp38-abi3-linux_x86_64.whl"
pip install -q vllm==0.9.2
pip install -q decord==0.6.0
log "cosmos wheels (apex/flash_attn/natten/te/torch)"
pip install -q "$B/apex-0.1+cu128.torch271-cp312-cp312-linux_x86_64.whl"
pip install -q "$B/flash_attn-2.6.3+cu128.torch271-cp312-cp312-linux_x86_64.whl"
pip install -q "$B/natten-0.21.0+cu128.torch271-cp312-cp312-linux_x86_64.whl"
pip install -q "$B/transformer_engine-1.13.0+cu128.torch271-cp312-cp312-linux_x86_64.whl"
pip install -q "$B/torch-2.7.1+cu128-cp312-cp312-manylinux_2_28_x86_64.whl"
pip install -q "$B/torchvision-0.22.1+cu128-cp312-cp312-manylinux_2_28_x86_64.whl"

# 4. TE include symlink patch
ln -sf $CONDA_PREFIX/lib/python3.12/site-packages/nvidia/*/include/* $CONDA_PREFIX/include/ 2>/dev/null || true
ln -sf $CONDA_PREFIX/lib/python3.12/site-packages/nvidia/*/include/* $CONDA_PREFIX/include/python3.12 2>/dev/null || true

log "torch check"; python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)" || true
log ">>> COSMOS ENV SETUP DONE"
