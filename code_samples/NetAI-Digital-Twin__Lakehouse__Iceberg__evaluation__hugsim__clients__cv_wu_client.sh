#!/usr/bin/zsh
# launched by HUGSIM's closed_loop.py as: zsh cv_wu_client.sh <cuda_id> <output_dir>
# constant velocity with AlpaSim's protocol: 3.0 s of recorded driving, then the handover speed held
cd /home/netai/jeykang/NetAI-Digital-Twin/Lakehouse/Iceberg/evaluation/hugsim/repo && pixi run python /home/netai/jeykang/NetAI-Digital-Twin/Lakehouse/Iceberg/evaluation/hugsim/clients/cv_client.py $2 --warmup 3.0 --hold-speed
