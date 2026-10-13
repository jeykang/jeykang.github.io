#!/usr/bin/zsh
# launched by HUGSIM's closed_loop.py as: zsh cv_client.sh <cuda_id> <output_dir>
cd /home/netai/jeykang/NetAI-Digital-Twin/Lakehouse/Iceberg/evaluation/hugsim/repo && pixi run python /home/netai/jeykang/NetAI-Digital-Twin/Lakehouse/Iceberg/evaluation/hugsim/clients/cv_client.py $2
