# Evaluation runs — recorded metrics

All rows: NVIDIA PhysicalAI AV, 60-clip slice, `--require-sensors`, horizon 4.0 s,
dt 0.5 s, 3 decision points per clip. Raw records in `.results_*.runmeta.json`
(written automatically by `run_eval.py`).

## Scores

| policy | MF-PDMS | NC | TTC | EP | HC | EC | clips w/ collision |
|---|---|---|---|---|---|---|---|
| `replay_human` *(oracle)* | 0.949 | 0.994 | 0.871 | 1.000 | 0.997 | 0.997 | 1.8% |
| **`alpamayo_1_5`** | **0.884** | 0.988 | 0.825 | 0.877 | 0.993 | 0.991 | 3.5% |
| `constant_velocity` | 0.853 | 0.942 | 0.801 | 0.880 | 1.000 | 1.000 | 15.8% |
| `stationary` | 0.421 | 0.690 | 0.591 | 0.035 | 1.000 | 1.000 | 38.6% |

Alpamayo-1.5 reproduced to 3 decimal places across two independent runs
(0.884 / 0.884), which is the expected behaviour given a fixed seed.

## Cost and resources

| | `alpamayo_1_5` | track-only baselines |
|---|---|---|
| clips scored | 57 of 60 | 57 of 60 |
| decisions scored | 171 | 171 |
| wall total | **1534.9 s** | ~26 s |
| per clip | **26.93 s** | 0.46 s |
| per decision | **8.98 s** | 0.15 s |
| throughput | **133.7 clips/h** | ~7,800 clips/h |
| per-clip p50 / p95 / max | 23.9 / 32.0 / 47.4 s | 0.04 / 0.08 / 0.08 s |
| peak VRAM allocated | **23.19 GB** | — |
| peak VRAM reserved | 23.29 GB | — |
| workers | 1 (GPU-serialised) | 8 |

Host: A10 23.68 GB (sm_86) + Quadro RTX 6000 25.19 GB (sm_75, unusable — Turing has
no BF16). Model ran on the A10 at **98% of card capacity**; `PYTORCH_CUDA_ALLOC_CONF=
expandable_segments:True` was required.

Extrapolation: a 1,000-clip model evaluation is **~7.5 h** single-GPU. The 3 clips
lost of 60 had no decision point with a full 4 s horizon inside sensor coverage.

Trajectory accuracy (spot check, 7 clips): mean ADE 2.01 m at the 4 s horizon,
best 0.07 m. NVIDIA reports minADE_6 = 1.22 m for this model on challenging samples;
ours is single-sample at a fixed horizon, so 2.01 m is the expected neighbourhood.

## Alpamayo-R1-10B — attempted, does not fit locally

R1 is the **1.0 release**; NVlabs/alpamayo states 1.5 supersedes it (R1 lacks the RL
post-training and navigation conditioning). Attempted anyway as a second data point,
because a metric that ranked R1 above 1.5 would be suspect.

**Result: CUDA OOM on the A10, confirmed empirically.**

- checkpoint 22.2 GB (vs 1.5's ~21 GB), 8.2 B backbone + 2.3 B action expert
- **R1 does not support SDPA** — `AlpamayoR1 does not support an attention
  implementation through scaled_dot_product_attention` — so it falls back to
  **eager** attention, which materialises the full attention matrix. 1.5 avoided
  this by running SDPA.
- failure point: 21.48 GB of weights allocated, 227 MiB free of 22.06 GB usable,
  OOM trying to allocate a further 552 MiB inside `eager_attention_forward`.

This matches NVIDIA's stated 24 GB minimum. Not an evaluator limitation — the policy
adapter loaded and ran up to the attention kernel. `AlpamayoR1Policy` in
`policy_alpamayo.py` is complete and will run unchanged on a ≥24 GB GPU.

## Reference ladder on the cluster slice, 4 s (2026-08-19)

Model scores are unreadable without the ladder on the *same* clips. All 55 shared:

| policy (4 s horizon) | MF-PDMS | NC | TTC | EP | collisions |
|---|---|---|---|---|---|
| `replay_human` *(oracle)* | 0.942 | 0.982 | 0.879 | 1.000 | 3.6% |
| **Alpamayo-1.5-10B** | 0.855 | 0.952 | 0.794 | 0.895 | 10.9% |
| **Alpamayo-R1-10B** | 0.851 | 0.964 | 0.776 | 0.885 | 9.1% |
| `constant_velocity` | 0.800 | 0.903 | 0.739 | 0.853 | 23.6% |
| `reactive_idm` | 0.794 | 0.915 | 0.715 | 0.835 | 23.6% |
| `stationary` | 0.401 | 0.655 | 0.545 | 0.055 | 45.5% |

Both learned models sit **+0.05 above the naive rules and -0.09 below the oracle**,
and roughly halve constant_velocity's collision rate (9-11% vs 23.6%). So MF-PDMS
does register learned actor-avoidance, even though it cannot separate one Alpamayo
generation from the next (paired test below).

**`reactive_idm` does not beat `constant_velocity`** (0.794 vs 0.800) despite being
the only non-learned policy that reacts to other actors. It buys a little safety (NC 0.915
vs 0.903) and pays for it in progress and TTC, with an identical collision count
(13/55). Braking for a lead vehicle in a narrow forward corridor does not help when
collisions arrive laterally, and EP's weight of 5 punishes the caution. That
strengthens the reading above: the learned models do something a simple reactive
heuristic does not reproduce.

## Cluster runs — Alpamayo R1 vs 1.5, head to head (2026-08-19)

Both models on the **same 60-clip slice**, A100-SXM4-40GB (pod09), SLURM+Singularity.
The slice was pulled straight from HuggingFace onto /scratch (the local NFS mount was
down), 10 clips from each of 6 chunks; 55 of 60 had a full decision window.

| | Alpamayo-1.5-10B | Alpamayo-R1-10B |
|---|---|---|
| MF-PDMS | 0.855 | 0.851 |
| NC | 0.952 | **0.964** |
| TTC | **0.794** | 0.776 |
| EP | **0.895** | 0.885 |
| HC / EC | 0.982 / 0.982 | 0.980 / 0.974 |
| clips w/ collision | 10.9% | 9.1% |
| s/clip | 22.4 | 23.9 |
| peak VRAM | 23.19 GB | **25.83 GB** |

### Paired test: no difference on any sub-metric

`compare_runs.py`, 10,000 bootstrap resamples over the 55 shared clips:

| metric | diff (1.5 − R1) | 95% CI | verdict |
|---|---|---|---|
| mf_pdms | +0.004 | [-0.016, +0.023] | no difference |
| nc | -0.012 | [-0.036, +0.012] | no difference |
| ttc | +0.018 | [-0.012, +0.055] | no difference |
| ep | +0.010 | [-0.009, +0.029] | no difference |
| hc | +0.002 | [-0.012, +0.015] | no difference |
| ec | +0.008 | [-0.006, +0.023] | no difference |

Per-clip, **R1 wins 30 and 1.5 wins 21** (4 ties) — the opposite sign to the mean
gap, which is what a null looks like.

### Correction: the earlier apparent ordering was a slice artifact

An earlier note here reported 1.5 at 0.884 against R1 at 0.851 and observed that this
matched NVIDIA's claim that 1.5 supersedes R1. **Those were different slices.** On the
shared slice 1.5 scores 0.855, not 0.884. The slice effect (0.029) is roughly **7x the
model effect** (0.004), so the ordering carried no information about the models. Any
model-vs-model claim from this evaluator has to be paired on identical clips.

### What this says about the instrument

The version-ordering test proposed earlier does not discriminate, so it cannot be used
to validate MF-PDMS. Two readings, and this run cannot separate them:
  * the models genuinely are close on 4 s open-loop trajectory geometry — plausible,
    since NVIDIA's claimed 1.5 gains are in RL post-training, navigation conditioning,
    CoC reasoning and closed-loop AlpaSim score, none of which this measures;
  * or MF-PDMS is too coarse to resolve one model generation.

What the metric *does* resolve is coarser and still useful: oracle 0.949 vs
constant-velocity 0.853 vs stationary 0.421. So treat it as discriminating
**competent / naive / degenerate**, not adjacent model versions.

## Out-of-lab transferability: VaVAM (2026-08-19)

The point of running VaVAM (valeoai VideoActionModel) is **not its score**. It is an
out-of-lab, out-of-architecture policy — an autoregressive video GPT plus a diffusion
action expert, from a different group, trained on OpenDV/nuPlan/nuScenes — and,
crucially, it consumes a **different input representation entirely**: VQ token
indices rather than pixels or actor tracks. Getting it through the same `Policy`
contract as the Alpamayo family is direct evidence the evaluator is transferable
rather than an Alpamayo-shaped wrapper. It ran unmodified.

Its action expert emits exactly 6 steps at 2 Hz = **3.0 s**, so the whole ladder was
re-run at `EVAL_HORIZON_S=3.0` rather than extrapolating a model past what it
predicts. This table is therefore separate from the 4 s tables above and the two are
not comparable.

| policy (3 s horizon, 55 clips) | MF-PDMS | NC | TTC | EP | HC | collisions |
|---|---|---|---|---|---|---|
| `replay_human` *(oracle)* | 0.941 | 0.982 | 0.873 | 1.000 | 0.994 | 3.6% |
| `constant_velocity` | 0.849 | 0.945 | 0.788 | 0.880 | 1.000 | 12.7% |
| **`vavam`** (seeded) | **0.611** | 0.848 | 0.661 | 0.885 | **0.627** | 27.3% |
| `stationary` | 0.448 | 0.727 | 0.606 | 0.073 | 1.000 | 43.6% |

### VaVAM is stochastic — a single number carries ~0.03 of sampling noise

Its action expert is a **diffusion sampler**, and the first runs did not seed it.
Three observations of the same 55 clips:

| run | seed | MF-PDMS |
|---|---|---|
| cluster, A100 | unseeded | 0.632 |
| local, A10 | unseeded | 0.660 |
| local, A10 (x2) | seeded | **0.611**, 0.611 |

Two unseeded runs differ by **0.028**, and the full observed range is ~0.05 — larger
than the entire gap between Alpamayo generations (0.005) and comparable to the
learned-vs-naive gap (0.05). The policy now seeds per decision and reproduces
exactly, so 0.611 is the reported value; note it is *one deterministic draw*, not a
converged mean. Anything requiring precision from a stochastic policy should average
over seeds.

This also reframes the Alpamayo tie: three generations within 0.005 is **below the
noise floor a stochastic policy exhibits on this benchmark**, which is a cleaner
statement of why MF-PDMS cannot separate them.

Found only by running the same model on two machines — an argument for keeping the
evaluation reproducible off-cluster.

**Read this as a domain-gap measurement, not a model ranking.** VaVAM was trained on
other datasets and scored here on NVIDIA PhysicalAI with no adaptation, so it sits
below the naive baseline — the expected shape. Two details are still informative:

* **Progress is competitive** (EP 0.885 vs constant_velocity's 0.880) — it drives
  forward sensibly. What costs it is safety (29.1% collisions) and, distinctively,
  **comfort: HC 0.627 / EC ~0.63 against ~0.98-1.00 for every other policy.** Its
  trajectories are rough. That is not a resampling artefact: at a 3 s horizon its 6
  outputs map 1:1 onto the evaluator's time grid with no interpolation.
* **It is by far the cheapest model tested** — 4.01 GB peak VRAM and a 1.75 GB
  checkpoint, against 23-26 GB for the Alpamayo family.

### Integration notes (all verified upstream, none guessed)

| detail | value | source |
|---|---|---|
| resize | factor 3.75 -> 512x288 | VaVAM's own **nuplan** preset; nuPlan frames are 1920x1080, exactly this dataset's resolution, so no geometry is invented |
| normalisation | `2.0 * x - 1.0` -> [-1,1] | `vam/datalib/token_creator.py` |
| tokeniser | `VQ_ds16_16384_llamagen_encoder.jit`, output (8, 18, 32) int64 | released asset; the action model takes tokens, not pixels |
| history | 8 frames @ 2 Hz | `sequence_length`, matching the action rate |
| command | STRAIGHT (RIGHT=0/LEFT=1/STRAIGHT=2) | no route available; same assumption as the DiffusionDrive integration |

Two environment traps worth reusing:

* **`torch>=2.6` defaults `torch.load(weights_only=True)`** and this checkpoint embeds
  omegaconf objects, so the strict unpickler rejects it. The upstream loader does not
  expose the flag; the policy relaxes it only around that one call. A version-skew
  problem, not a VaVAM problem.
* **The SIF sets `PYTHONNOUSERSITE=1`** (my own doing in `alpamayo.def`), which makes
  `pip install --user` impossible inside it. Extra deps go in a venv created with
  `--system-site-packages` on /scratch, which inherits the image's torch. That is the
  reusable pattern for adding future out-of-lab models without rebuilding the image.

## Alpamayo2-Super — cluster feasibility

There is no "Alpamayo R2". The successor is **Alpamayo2-Super** (released
2026-08-04): 34 B total — a 32 B VLM backbone built on Cosmos 3 Super Reasoner with
RL post-training, plus the same 2.3 B diffusion action decoder. Weights OpenMDW-1.1,
code Apache-2.0, commercial use permitted. Published: minADE_6 @6.4 s = 0.911 m,
AlpaSim 1.50 ± 0.13, LingoQA 79.2.

**Verdict: feasible on the A100 cluster, not on any local GPU. Requires sharding.**

| | value | implication |
|---|---|---|
| peak VRAM measured by NVIDIA | **72,115 MiB** on one H100 80 GB | 1.8x a whole A100-40GB |
| weights (BF16) | ~68 GB | must span >= 2 GPUs; 4 for headroom |
| cluster node | 8x A100-40GB, 320 GB total | 4 GPUs -> ~17 GB weights each |
| validated hardware | H100 only; "other architectures not yet validated" | Ampere is BF16-capable but unproven here |
| deps | torch >=2.8, transformers >=4.57.1, **DeepSpeed >=0.17.4** | DeepSpeed already a declared dep -> sharding path exists |
| I/O contract | 6-7 cameras, 4 frames, ego translation + rotation | same family; our adapter already builds this |
| output | 64 waypoints, 0.1-6.4 s @ 0.1 s | **identical to 1.5/R1** — resampling code unchanged |

### Work required

1. **Container.** SIF with torch 2.8 / transformers 4.57.1 / DeepSpeed + the
   `alpamayo2` package. Prior art: `cosmos_augmentation/cosmos_transfer1.def` and
   the hard-won lesson that the login node cannot build images — build locally with
   apptainer fakeroot, transfer the SIF.
2. **Weights.** ~68-72 GB pulled on the cluster. Prior art:
   `cosmos_augmentation/cluster_download_weights.sh` (113 GB for Cosmos-Transfer1).
3. **Data staging.** One camera mp4 is ~23 MB per 20 s clip, so ~92 MB/clip for the
   4-camera config and ~160 MB/clip for 7 cameras, plus KB-scale egomotion and
   obstacle labels. A 60-clip validation slice is **~10 GB**; 500 clips ~80 GB.
4. **Policy.** A third subclass of `_AlpamayoBase` — `_load` + `_message` only. The
   input construction and output resampling are already shared and verified.
5. **Sharding.** `device_map="auto"` across 4 GPUs is the low-effort route;
   DeepSpeed inference is the documented one. Note attention: if Alpamayo2 also
   lacks SDPA support and flash-attn cannot be built on the cluster (no `nvcc`
   historically), eager attention will inflate the 72 GB figure further — budget
   accordingly, and verify attention support before booking the node.

### RAN 2026-08-20 as nf4 — and all three Alpamayo generations tie

Quantisation cleared the blocker. The ladder of attempts, each eliminating one cause:

| attempt | outcome |
|---|---|
| bf16, 4 GPUs, `device_map=auto` | KV cache device mismatch (`cuda:3` vs `cuda:0`) |
| bf16 + `_no_split_modules` declared | unchanged — blocks were never the problem |
| int8, all modules | `silu_cuda not implemented for 'Char'` — the custom expert is not bnb-safe |
| int8, backbone only | loaded, then **OOM at 39.28/39.49 GB** — short by ~1 GB |
| **nf4, backbone only** | **ran**: 28.47 GB peak on one A100-40GB |

Quantising the 32B Qwen3-VL backbone while leaving the 2.3B diffusion expert in bf16
restores the single-device semantics the released code assumes (NVIDIA tested on one
H100-80GB), which is what actually fixed it — the memory saving was the means, not
the point.

**Report this as `Alpamayo2-Super nf4` — NOT the released model.** The backbone is
4-bit; numerics differ.

| | value |
|---|---|
| MF-PDMS | **0.850** |
| NC / TTC / EP / HC / EC | 0.952 / 0.788 / 0.883 / 0.983 / 0.983 |
| clips w/ collision | 10.9% |
| throughput | 55.9 s/clip (18.6 s/decision), 64.4 clips/h |
| peak VRAM | 28.47 GB on 1 GPU |
| normalised cost | 15.5 GPU-h / 1k clips (vs 6.6 for R1) |
| checkpoint | 71.65 GB |

nf4 costs **2.3x the wall-clock of R1** (55.9 vs 23.9 s/clip) for dequantisation.

### RESOLVED 2026-08-24: the generations differ; the metric is blind to it

Published closed-loop AlpaSim scores settle the ambiguity below without running
anything: R1 0.73 +/- 0.01, Alpamayo-1.5 1.37 +/- 0.10, Alpamayo2-Super 1.50 +/- 0.13
— a **2.05x spread**, with R1 vs 1.5 roughly 6 sigma apart — against our MF-PDMS
spread of 1.006x, ranked in a different order. So the reading is "MF-PDMS is too
coarse", not "the models are equivalent". Full analysis and what to do about it:
`ALPASIM.md`.

### The headline: the metric cannot separate any Alpamayo generation

Paired over the same 55 clips, 10,000 bootstrap resamples:

| pair | diff | 95% CI | verdict |
|---|---|---|---|
| A2-nf4 - 1.5 | -0.005 | [-0.025, +0.013] | no difference |
| A2-nf4 - R1 | -0.000 | [-0.023, +0.022] | no difference |
| 1.5 - R1 | +0.004 | [-0.016, +0.023] | no difference |

Per-clip win counts are coin-flips throughout (A2 27-25 over 1.5; R1 26-25 over A2).

So **R1 (10B), 1.5 (10B) and Alpamayo2-Super (34B) all land within 0.005 of each
other** — across a 3.4x parameter jump and two model generations. Meanwhile the same
metric cleanly separates oracle (0.942) from naive (0.800) from degenerate (0.401).

That bounds what MF-PDMS is for: it discriminates **competence classes**, not model
generations. Whether the generations genuinely are equivalent on 4 s open-loop
geometry — plausible, since NVIDIA's claimed gains for the newer models are in RL
post-training, navigation conditioning, CoC reasoning and closed-loop AlpaSim score,
none of which this measures — or whether the metric is too coarse, this design still
cannot separate. The nf4 caveat does not rescue the comparison either way: a
quantised 34B matching an unquantised 10B is consistent with both readings.

### Previously blocked (2026-08-19), kept for the record

Weights fetched (70 GB in 1:27) and the policy written; the model never ran. Four
submissions, each clearing one barrier:

| # | failure | fix |
|---|---|---|
| 1 | `source data must include ordered camera_names` | supply `camera_names` |
| 2 | ring check: source must contain the **canonical 7-camera ring in order** | stage a 7th camera (`rear_tele`), supply all 7 — the driving profile selects 6 of them but all must be present |
| 3 | `ego_t0_frame_idx must be a tensor` | supply it (value is overwritten; only dtype/device are read) |
| 4 | **`RuntimeError: tensors is on cuda:3, different from other tensors on cuda:0`** in `cache_utils.update` | **unresolved** |

Barriers 1-3 are Alpamayo2's stricter input contract versus 1.5/R1 — reasonable
design (it makes silent camera-ordering bugs impossible), just stricter than the API
the adapter was written against. Those are fixed and the policy is complete.

**Barrier 4 is the blocker.** The KV cache is not device-aware across a shard
boundary. Declaring the block classes did not help:

    Alpamayo2Super._no_split_modules = ["Qwen3VLTextDecoderLayer", "Qwen3VLVisionBlock"]

The Qwen3-VL backbone declares those itself, but the `Alpamayo2Super` wrapper does
not re-export them, so `device_map="auto"` had no block boundaries. Adding them left
the error unchanged, so blocks were not the problem — the cache is allocated on one
device and concatenated on another during the rollout.

This is consistent with NVIDIA testing the model on **one H100-80GB**: multi-GPU is
our configuration, not a supported path. Both nodes assigned to this project
(`hpc-pr-a-pod09`, `hpc-pr-a-pod17`) are **A100-SXM4-40GB**, and 68 GB of BF16
weights cannot sit on one, so sharding is not optional here.

### Options, none free

1. **8-bit quantisation** (bitsandbytes) — ~34 GB, fits one 40 GB A100, restores the
   single-device semantics the code assumes. Changes numerics, so it must be reported
   as *Alpamayo2-Super int8*, not the released model. Needs bitsandbytes added to the
   SIF (rebuild + re-ship).
2. **An 80 GB GPU** — matches NVIDIA's tested configuration exactly and needs no code
   changes. The `dgx-a100-n[1-4]` nodes may carry 80 GB cards but are not assigned to
   this project.
3. **Tensor parallelism** — correct in principle, not wired in the released code;
   the largest effort of the three.

CPU offload was considered and rejected: it would restore single-device semantics,
but streaming 68 GB per forward pass through a generative rollout (~256 CoC tokens
per decision) is hours per clip.

### Recommendation

Run **one cluster campaign covering both R1 and Alpamayo2-Super**. They share the
container, the staging, and the policy scaffolding; R1 additionally fits on a single
A100-40GB, so it is the cheap smoke test that de-risks the Alpamayo2 run. Expected
output is a 4-row benchmark (oracle / Alpamayo2 / 1.5 / R1) on an identical slice,
which is the first thing here that would test whether MF-PDMS orders models the same
way their published benchmarks do.
