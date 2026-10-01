# alpasim/ — closed-loop validation layer (detachable)

**Why this exists.** MF-PDMS is open-loop and dataset-agnostic, which is the point:
it runs on any dataset exposing actor tracks and ego poses. But it cannot separate
model generations — published closed-loop AlpaSim scores span **2.05x** across the
three Alpamayo models where MF-PDMS spans **1.006x**, in a different order (see
`../ALPASIM.md`). AlpaSim is the external anchor that quantifies that gap.

**What it is NOT.** Not the primary evaluator. AlpaSim consumes NuRec USDZ scenes,
which exist for 1,607 clips of one dataset; making it primary would relock the system
to NVIDIA PhysicalAI. It is a calibration layer, enabled only where NuRec exists.

## Detaching
Delete this directory. Nothing outside it imports it; the evaluator has no AlpaSim
dependency, and `../policies.py` / `../policy_*.py` are untouched.

## Design: an AlpaSim plugin, not a fork

AlpaSim discovers components through Python entry points (`alpasim.models`,
`alpasim.configs`). `plugin/` is a standalone package registering one model,
`harness`, that delegates to this project's `Policy` objects. The AlpaSim checkout in
`repo/` is unmodified, so it can be updated with `git pull`.

The policy spec uses the same syntax as `run_eval.py --policy`, so one string names
the same policy in both systems:

```bash
HARNESS_DIR=/abs/path/to/evaluation \
HARNESS_POLICY=policy_vavam:VaVAMPolicy \
uv run alpasim ... driver=harness
```

## What transfers, and what cannot

AlpaSim's `PredictionInput` gives cameras, command, speed, acceleration and ego pose
history. It deliberately withholds ground-truth actor boxes — closed-loop expects the
policy to perceive them. Our `Observation` carries `agent_history`. Therefore:

| policy | transfers? |
|---|---|
| `constant_velocity`, `constant_turn_rate`, `stationary` | yes — ego-only |
| `policy_vavam:VaVAMPolicy`, Alpamayo policies | yes — camera-based |
| `reactive_idm` | **no** — reads ground-truth tracks |
| `replay_human` | **no** — oracle, needs the recorded future |

The two that cannot transfer are rejected explicitly rather than silently handed an
empty actor set, which would look like a working run producing a meaningless score.

## Prerequisites (verified on this host 2026-08-24)

| requirement | status |
|---|---|
| NVIDIA driver >= 570 | 580.159.03 |
| `uv` >= 0.9.17 | 0.11.24 |
| Rust / `cargo` (builds `utils_rs`) | installed 1.98.0 |
| Docker + compose + buildx, rootless | 29.6.1 / v5.2.0 / v0.35.0 |
| NVIDIA Container Toolkit | GPUs visible in container |
| HF token **with gated NuRec access** | required — scenes 404 without it |

## The VRAM problem, and the L40S

AlpaSim runs the NuRec renderer **and** the driver on the same GPU. Its tutorial cites
**~40 GB for Alpamayo 1/1.5** and ~60 GB for the CFG variant. This host has a 23 GB
A10 (the 24 GB Quadro RTX 6000 is Turing, no BF16), so locally this layer pairs only
with a small policy:

| policy | standalone peak | viable on the 23 GB A10 alongside a renderer? |
|---|---|---|
| `constant_velocity` etc. | 0 GB | yes |
| VaVAM | 4.01 GB | yes |
| Alpamayo-1.5 / R1 | 23.2 / 25.8 GB | no |
| Alpamayo2-Super nf4 | 28.5 GB | no |

A **48 GB L40S** would fit the renderer plus a 10B driver, which is the configuration
that produces closed-loop numbers for the models NVIDIA already publishes AlpaSim
scores for — i.e. the direct check on whether our pipeline reproduces theirs. Nothing
here assumes SLURM, so moving is a path change, not a rewrite; the only cost is that
job orchestration would be manual.

Disk is the other bound: NuRec scenes are ~1.79 GB each and ~146 GB is free, so ~80
scenes maximum.

## Local scenes (our own USDZ, no catalog)

`LOCAL_USDZ_DIR=<dir> ./run_scene.sh <policy-spec>` runs every `*.usdz` under
`<dir>` (recursively) through AlpaSim's `local` artifact repository instead of the
HuggingFace catalog — the path a self-reconstructed Gold clip would take. The
directory becomes the scene cache bind-mounted into the containers, so files that
live elsewhere must be **hardlinked** in (`ln`), not symlinked; `ego-hoods/` is
mounted separately from `defines.sensordata` and is unaffected. `scene_ids` is
nulled explicitly because `base_config.yaml` carries a default scene.

`EXTRA_ARGS="key=value …"` appends wizard overrides to `run_scene.sh`. The one that matters
for our own twins: `runtime.simulation_config.route_generator_type=RECORDED` drives a scene
that has no `map.xodr` along its recorded route (the default `MAP` generator fails on it);
the evaluator still needs the map for off-road / lane metrics and refuses the scene score
without them — see `evaluation/nurec/README.md` ("time base" and the v3 results) for what
our exports need before AlpaSim loads them at all.


Verified 2026-09-21: one cached NuRec scene run this way reproduced the
catalog-path per-clip metrics identically on all 16 columns (`runs/ws1-local-cv`
vs `runs/20260914-235934`, `constant_velocity`). This closes FEASIBILITY.md risk 1.

## Status

Running end to end locally, in both directions:

- **Policy bridge** (`driver=harness`) — our `Policy` objects drive AlpaSim. All five
  reference policies work; `constant_velocity` scored over 10 NuRec scenes.
- **Native driver** (`DRIVER=vavam`) — an upstream AlpaSim driver runs through the same
  runner, over 40 scenes.

`per_clip.py` extracts per-clip scores (AlpaSim publishes only a run-level mean) and
`../correlate_alpasim.py` joins them to MF-PDMS. Results in `../ALPASIM.md`.

Scene artifacts land in `repo/data/nre-artifacts/` at ~1.6 GB/scene and are **not**
pruned between batches; 40 scenes is ~65 GB. Watch disk before large batches.
