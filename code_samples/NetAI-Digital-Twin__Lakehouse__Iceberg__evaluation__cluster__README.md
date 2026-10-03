# Running the evaluation on the A100 cluster

> **STATUS 2026-08-24: CLUSTER UNAVAILABLE — do not attempt access.**
> Access to the previously assigned nodes has ended and no nodes are currently
> usable. The final blocker on pod12/pod15 was that uid 1064 has **no passwd entry
> on those nodes** (`NO_PASSWD_ENTRY`; pod09 had one), so Singularity aborts in its
> host-side starter before any bind applies — plain, `--no-home`, `--userns` and
> `--fakeroot` all fail identically. That is an admin-side fix, not a job-side one.
> Everything needed is already local: results, logs, the 60-clip slice, VaVAM's
> weights and the SIF. See ../BENCHMARKS.md. This directory is kept for when
> cluster access returns.

For models that do not fit the local 23 GB A10. Confirmed cluster facts:
`slurm-master2` login node, partition `jobs`, pods are **A100-SXM4-40GB** (driver
575.57.08), 8 GPUs/node, `/scratch` 130 TB free, `$CLUSTER_HOME=/scratch/autodr_test`.
pod09 was IDLE at time of writing; the `dgx-a100-n[1-4]` nodes were all allocated, so
whether they carry 40 GB or 80 GB cards is still unknown — worth checking, because
80 GB would let Alpamayo2-Super run on a single GPU.

## Why these models need the cluster

| model | weights | fits 40 GB? | note |
|---|---|---|---|
| Alpamayo-1.5-10B | ~21 GB | yes, 1 GPU | already validated locally at 23.19 GB peak on an A10 |
| Alpamayo-R1-10B | ~22 GB | yes, 1 GPU | **OOMs locally**: no SDPA support -> eager attention |
| Alpamayo2-Super | ~68 GB | no | 72 GB peak measured by NVIDIA; needs >=2, use 4 |

## Runbook

```bash
# 1. environment on /scratch (detached; ~20 min)
python cosmos_augmentation/cluster.py "nohup bash $E/setup_env.sh > $E/env.log 2>&1 &"

# 2. weights (detached; 22 GB for R1, ~68 GB for Alpamayo2)
python cosmos_augmentation/cluster.py \
  "nohup bash $E/download_weights.sh 'nvidia/Alpamayo-R1-10B' > $E/w.log 2>&1 &"

# 3. stage a slice (build first, inspect size, then upload)
python evaluation/cluster/stage_slice.py --limit 60 --cameras 4
python evaluation/cluster/stage_slice.py --limit 60 --cameras 4 --upload

# 4. ship the harness itself (pure python + pyarrow + cv2)
#    scp evaluation/*.py -> $E/evaluation/

# 5. run
sbatch --export=ALL,MODEL=r1,LIMIT=60 evaluation/cluster/eval.sbatch
```

## Confirmed working (2026-08-19)

| step | job | result |
|---|---|---|
| SIF built locally, shipped | — | 9.5 GB, `torch 2.8.0+cu128` verified on-node |
| weights + sources | 175028 / 175184 | R1 21 GB in 44 s; 1.5 likewise |
| slice from HuggingFace | 175179 | 60 clips / 6 chunks in 6:52, 3.3 GB extracted |
| R1 evaluation | 175182 | 55 clips, 23.9 s/clip, peak 25.83 GB |
| 1.5 evaluation | 175186 | 55 clips, 22.4 s/clip, peak 23.19 GB |

The local NFS mount was down throughout — the cluster path needs nothing from it.

## Adding another model

Proven twice (Alpamayo family, then VaVAM from a different lab):

1. weights + sources -> `fetch_weights.sbatch` (or a model-specific fetch job)
2. extra python deps -> venv with `--system-site-packages` on /scratch, NOT
   `pip install --user` (the SIF sets `PYTHONNOUSERSITE=1`)
3. a `Policy` subclass in `evaluation/`, wired into `eval.sbatch`'s `case`
4. `sbatch --export=ALL,MODEL=<name>,LIMIT=60[,EVAL_HORIZON_S=...,TAG=...]`

Use `TAG` whenever the horizon differs, or runs overwrite each other's parquet.

## Notes and traps

- **No container.** The Cosmos work showed a bare conda env on `/scratch` is enough
  here; the login node cannot build images, so avoiding a SIF removes the
  build-locally-then-transfer step entirely.
- **flash-attn is not installed** (needs nvcc, absent on the login node). 1.5 falls
  back to SDPA, R1 to eager. Verify Alpamayo2's attention support *before* booking a
  node: if it also lacks SDPA, eager attention will push peak memory above the 72 GB
  figure and change the GPU count.
- **Staging is minimal by construction.** `stage_slice.py` rebuilds each per-chunk
  obstacle zip with only the selected clips, so a 60-clip / 4-camera slice is ~5-6 GB
  rather than the whole chunk set. `AV_ROOT` makes the adapter read it unmodified.
- **`--env` DOES NOT PROPAGATE into the container here.** This cost three debugging
  cycles, each surfacing at a different stage: the HF token for the data fetch, then
  `PYTHONPATH` (`ModuleNotFoundError: alpamayo_r1`), then the HF token again for
  model loading. Export everything inside the `bash -c` instead, and for
  `huggingface_hub` prefer passing `token=` explicitly — `$HF_HOME/token` was not
  picked up either. Every job here now does this.
- **Gated repos are not just the checkpoints.** The dataset
  `nvidia/PhysicalAI-Autonomous-Vehicles` is gated (the Alpamayo model repos are
  not), and **Alpamayo 1.5 pulls a gated base model, `nvidia/Cosmos-Reason2-8B`, at
  load time**. That never appears locally because the cache is already warm, so it
  only bites on fresh hardware. The token lives at `$E/.hf_token`, mode 600, and is
  read by the job — not passed via `--export`, which would expose it in
  `scontrol show job`.
- **Per-chunk packaging drives slice design.** The dataset ships one ZIP per
  (sensor, chunk), ~1.35 GB per camera-chunk holding ~100 clips. A random 60-clip
  draw would touch ~60 chunks (~330 GB); `plan_slice.py` instead takes 10 clips from
  each of 6 chunks (~33 GB downloaded, 3.3 GB kept). The result is chunk-clustered,
  so report it as such — it is not a representative cohort sample.
- **Scale reference.** Measured: 22-24 s/clip for a 10B model on one A100-40GB
  (~160 clips/h), against 26.9 s/clip on the local A10. Less speedup than the GPU gap
  suggests, because the run is partly I/O and CoC-generation bound. Budget Alpamayo2
  from ~24 s/clip, not from a compute ratio.
