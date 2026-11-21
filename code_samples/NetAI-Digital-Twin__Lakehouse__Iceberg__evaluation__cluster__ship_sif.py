#!/usr/bin/env python3
"""Transfer the locally-built SIF to the cluster (login node cannot build images)."""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "cosmos_augmentation"))
import cluster  # noqa: E402

SIF = sys.argv[1] if len(sys.argv) > 1 else "/tmp/alpamayo_eval.sif"
env = cluster.load_env()
remote = f"{env['CLUSTER_HOME']}/alpamayo_eval/alpamayo_eval.sif"
sz = os.path.getsize(SIF) / 1e9
print(f"[ship] {SIF} -> {remote}  ({sz:.2f} GB)")
t0 = time.time()
cluster.run(f"mkdir -p {env['CLUSTER_HOME']}/alpamayo_eval")
cluster.put(SIF, remote)
print(f"[ship] uploaded in {time.time()-t0:.0f}s ({sz/max(1e-9,(time.time()-t0))*1000:.0f} MB/s)")
o, e = cluster.run(f"ls -la {remote} && singularity exec {remote} python -c "
                   f"'import torch,transformers;print(torch.__version__, transformers.__version__)'",
                   timeout=300)
print(o or e)
