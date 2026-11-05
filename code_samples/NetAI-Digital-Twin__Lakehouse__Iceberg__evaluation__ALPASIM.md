# Should AlpaSim be added as a validation layer? (2026-08-24)

**Short answer: yes, but strictly as a calibration anchor on the NuRec subset — not
as the primary evaluator.** The open-loop, dataset-agnostic design is the right
default and should stay. What AlpaSim buys is the external yardstick MF-PDMS
currently lacks.

## The evidence is already available, at zero cost

NVIDIA publishes closed-loop AlpaSim scores for all three Alpamayo models we ran.
Placing them beside our numbers answers the question left open in BENCHMARKS.md —
*are the generations genuinely tied, or is MF-PDMS too coarse?*

| model | AlpaSim (closed-loop) | rank | minADE@6.4s | rank | MF-PDMS (ours) | rank |
|---|---|---|---|---|---|---|
| Alpamayo-R1-10B | 0.73 ± 0.01 | 1 | 1.220 m | 1 | 0.851 | 2 |
| Alpamayo-1.5-10B | 1.37 ± 0.10 | 2 | 0.916 m | 2 | 0.855 | 3 |
| Alpamayo2-Super | 1.50 ± 0.13 | 3 | 0.911 m | 3 | 0.850 | 1 |

| metric | spread across the three models |
|---|---|
| AlpaSim (closed-loop) | 0.73 -> 1.50 = **2.05x** |
| minADE (open-loop, NVIDIA's own) | 1.220 -> 0.911 m = 1.34x |
| **MF-PDMS (ours, open-loop)** | 0.850 -> 0.855 = **1.006x** |

**The generational difference is real and large — closed-loop nearly doubles from R1
to Alpamayo2 (0.73 -> 1.50, R1 vs 1.5 alone is ~6 sigma apart) — and MF-PDMS is blind
to it.** Our ranking is not merely compressed, it is uncorrelated: MF-PDMS puts
Alpamayo2 *last*.

So the earlier ambiguity resolves in favour of "the metric is too coarse", **not**
"the models are equivalent".

### But open-loop coarseness is not our bug alone

NVIDIA's own open-loop metric saturates on the same pair: minADE 0.916 vs 0.911 m for
1.5 vs Alpamayo2 — a **0.5% difference** across a 3.4x parameter jump — while their
closed-loop metric separates the same pair (1.37 vs 1.50). minADE does separate R1
(1.220 m), where MF-PDMS does not, so MF-PDMS is *less* sensitive than minADE on that
pair. That is not purely a defect: minADE measures geometric agreement with the
recorded human path, while MF-PDMS measures safety/progress/comfort properties of the
trajectory. A policy can deviate from the human line while remaining collision-free
and comfortable, and MF-PDMS deliberately does not punish that.

The general lesson holds regardless: **4-second open-loop scoring cannot resolve one
model generation from the next, because errors that matter compound over a closed-loop
rollout that open-loop scoring never performs.**

## What adding AlpaSim would and would not do

**Would:**
* give a per-clip correlation between MF-PDMS and a closed-loop score on the *same*
  scenes — the actual validation, far stronger than the 3 aggregate points above;
* produce closed-loop numbers for policies NVIDIA does not publish (VaVAM,
  DiffusionDrive, the reference ladder), which is where our evaluator adds value;
* calibrate how much open-loop coarseness costs, i.e. tell a consumer when the cheap
  metric is sufficient and when it is not.

**Would not:**
* replace the open-loop path. AlpaSim consumes **NuRec USDZ scenes**, which exist for
  1,607 clips of one dataset. Making it primary would relock the system to NVIDIA
  PhysicalAI — precisely what the open-loop design avoids.

## Cost, against current constraints (no cluster)

| item | figure |
|---|---|
| AlpaSim | Apache-2.0, `NVlabs/alpasim`, gRPC policy interface |
| scenes | NuRec USDZ, gated, ~1.79 GB/scene, 1,607 scenes (2.87 TB total) |
| local disk free | ~146 GB -> ~80 scenes maximum |
| local GPU | one A10, 23 GB |

The binding constraint is VRAM. AlpaSim's renderer plus a policy must share 23 GB, so
locally it pairs only with a small policy — VaVAM at 4.01 GB fits comfortably;
Alpamayo-1.5 at 23.19 GB does not fit alongside a renderer at all. Practical local
scope is therefore **VaVAM (and the reference ladder) over a few dozen NuRec scenes**,
which is enough for a correlation estimate but not for closed-loop numbers on the
10B/34B models.

## Recommendation

1. **Record the published-score comparison now** (done, above). It already answers
   the question that motivated this, and costs nothing.
2. **Keep MF-PDMS as the dataset-agnostic primary**, and re-scope how it is described:
   it separates competence classes (oracle / learned / naive / degenerate) and is
   **not** a model-selection metric between close variants. That is a legitimate and
   useful role, but it must be stated rather than implied.
3. **Add AlpaSim as an optional validation adapter**, gated on NuRec availability, and
   report the MF-PDMS-vs-AlpaSim per-clip correlation as a property *of the metric*.
   Do this locally with VaVAM + the ladder when disk allows; a cluster with 80 GB
   cards would be needed for closed-loop numbers on the large models.
4. If only one thing is done: **2**. The single largest risk in this work is a
   consumer reading a 0.005 MF-PDMS gap as a model improvement.

---

# Per-clip correlation: first results

The recommendation above (item 3) is now implemented. Two pieces of tooling:

- **`alpasim/per_clip.py`** — per-clip closed-loop scores. AlpaSim publishes only a
  run-level mean, but its aggregation pipeline computes a per-trajectory table on the
  way there, so this calls that pipeline rather than reimplementing it.
- **`correlate_alpasim.py`** — joins per-clip MF-PDMS to per-clip AlpaSim on `clip_id`
  and reports Spearman with a bootstrap CI.

## Getting per-clip numbers right

A naive `max`/`last` aggregation over `metrics_unprocessed.parquet` does **not**
reproduce AlpaSim's own numbers. The modifier chain materially changes them:

| stage | effect |
|---|---|
| `RemoveTimestepsBeforeEvent(eval_relevant > 0)` | drops lead-in |
| `RemoveTimestepsAfterEvent(offroad_or_collision > 0)` | truncates at first failure |
| `RemoveTimestepsAfterEvent(dist_to_gt_trajectory >= 4.0)` | truncates once ego leaves the GT path — **from wizard config, not the defaults** |

Skipping the chain inflated `dist_traveled_m` 36.5 -> 37.2 and `offroad` 0.60 -> 0.70
(one clip only went off-road *after* already deviating 4 m). `per_clip.py` therefore
validates its per-clip means against the run's own summary parquet and reports
`MISMATCH` if the chain ever drifts.

## Batch 1: `constant_velocity`, 10 NuRec scenes

| metric | value |
|---|---|
| `progress` | 0.19 |
| `dist_traveled_m` | 36.5 (GT 189.7) |
| `offroad` | 0.60 |
| `collision_any` / `collision_at_fault` | 0.30 / **0.00** |
| `duration_frac_20s` | 0.095 |

(Numbers here are post-fix; an earlier pass that skipped the 4 m truncation
reported `offroad` 0.70 and 10/10 failures.)

Correlation against MF-PDMS on the same 10 clips and the same policy: **no metric
reached significance**, |rho| <= 0.21, CIs spanning roughly [-0.8, +0.8].

**This is not evidence that the metrics disagree.** `offroad_or_collision` is 1.0 on
9 of 10 clips — 6 off-road, 3 rear-ended, none at fault. The closed-loop outcome is
very nearly a constant, so there is almost nothing for the open-loop score to
correlate *with*, and n=10 leaves the CIs uninformative either way. The comparison
needs a policy that does not fail everywhere, which is what the VaVAM batch is for.

It is worth noting separately that closed-loop is far harsher than open-loop on the
same policy: MF-PDMS gives `constant_velocity` 0.880 over these clips while AlpaSim
has it leaving the road or being struck in 10/10. Open-loop scoring resets the ego to
ground truth at every decision point, so heading error never compounds; closed-loop
lets it run. That is the structural difference the correlation is meant to quantify.

## Runner

`run_scene.sh` now takes a `DRIVER` env var, so native AlpaSim drivers run through the
same path as the policy bridge (the `harness` driver):

    ./run_scene.sh constant_velocity <scene_id> ...        # policy bridge
    DRIVER=vavam ./run_scene.sh vavam <scene_id> ...       # native driver

Native drivers need their weights staged under `repo/data/drivers/<driver>/`, which is
bind-mounted to `/mnt/drivers/`. VaVAM needs both `VAM_width_1024_pretrained_139k.pt`
and `VQ_ds16_16384_llamagen_encoder.jit`; hardlink them from `.vavam/ckpt/` rather than
symlinking, as a symlink does not resolve inside the bind mount.

## Batch 2: VaVAM on the same 10 scenes

Same scenes, same renderer, native AlpaSim `driver=vavam`. The point is the paired
comparison — identical scenes, so nothing but the policy differs.

| metric | `constant_velocity` | VaVAM |
|---|---|---|
| `progress` | 0.190 | **0.471** |
| `dist_traveled_m` | 36.5 | **75.4** |
| `duration_frac_20s` | 0.095 | **0.273** |
| `offroad` | 0.600 | **0.300** |
| `collision_any` | 0.300 | 0.500 |
| `collision_at_fault` | **0.000** | 0.300 |
| `offroad_or_collision` | 0.900 | 0.800 |
| clips with no incident | 1 | 2 |

VaVAM travels further and survives longer on **10 of 10 clips** (Wilcoxon p=0.002 for
both `progress` and `dist_traveled_m` — the floor at n=10). It halves the off-road
rate, which is what a policy that can actually steer should do.

> **Correction (2026-10-07): this `constant_velocity` did not hold its velocity.** While AlpaSim
> forces the recorded trajectory (the first 3.0 s, `force_gt_duration_us`), its driver reports
> zero speed, and our policy bridge passed that on (AlpaSim's `_get_speed_and_acceleration`
> documents a finite-difference fallback for that case but does not implement it). Constant
> velocity therefore planned to stand still at handover, AlpaSim braked, and the policy then held
> whatever speed was left. Rollout logs show the speed handed to the policy at 0.00 from 0.5 to
> 3.0 s, and in every constant-velocity rollout the speed after handover is 3–6 m/s below the
> speed before it; slow clips crawl (1920170b 19.2 → 14.5 m/s, 2ce64e22 4.2 → 0.7 m/s here).
> The bridge now falls back to the ego's pose history (`plugin/alpasim_harness/harness_model.py`).
> So the table above compares VaVAM with "brake at handover, then hold", not with constant
> velocity; VaVAM's 10-of-10 lead in progress and distance is at least partly that braking, and
> the policy-level reversal below needs a rerun of these ten scenes with the fixed bridge (they
> are no longer cached; the rerun waits on disk space). VaVAM's own numbers are unaffected: its
> AlpaSim wrapper does not read the reported speed.

It also collides *more*, and at fault where `constant_velocity` never was. That is not
a contradiction: `constant_velocity` leaves the road almost immediately, and the run is
truncated at that point, so it is never alive long enough to hit anything. Surviving
longer exposes a policy to more opportunities to fail. This is a good illustration of
why single closed-loop metrics are misread in isolation, and why `offroad_or_collision`
(0.900 vs 0.800) moves far less than `progress` (2.5x).


## Per-clip correlation at n=40 (the result)

VaVAM over 40 NuRec scenes, correlating its per-clip open-loop MF-PDMS against its
per-clip closed-loop AlpaSim score. Both axes now have real spread (MF-PDMS
0.569 +/- 0.255; 14 of 40 clips finish with no incident, 26 with one).

| AlpaSim metric | rho | 95% CI | p |
|---|---|---|---|
| `progress_rel` | +0.320 | [-0.01, 0.60] | 0.044 |
| `progress` | +0.234 | [-0.10, 0.53] | 0.146 |
| `dist_traveled_m` | -0.213 | [-0.53, 0.15] | 0.186 |
| `offroad_or_collision_at_fault` | +0.198 | [-0.14, 0.51] | 0.220 |
| `dist_to_gt_trajectory` | +0.157 | [-0.18, 0.47] | 0.333 |
| `collision_any` | -0.044 | [-0.41, 0.32] | 0.788 |
| `offroad` | -0.030 | [-0.32, 0.27] | 0.854 |
| `duration_frac_20s` | -0.013 | [-0.32, 0.30] | 0.934 |

**Nothing survives correction for 8 comparisons** (Bonferroni 0.05/8 = 0.00625).
`progress_rel` at p=0.044 is the strongest signal and it is marginal.

The direct test is cleaner than any correlation. Split the clips by whether VaVAM
actually had an incident in closed-loop:

| closed-loop outcome | n | mean MF-PDMS | Mann-Whitney p |
|---|---|---|---|
| no incident | 14 | 0.593 | 0.660 |
| offroad or collision | 26 | 0.556 | |
| no at-fault collision | 31 | 0.546 | 0.095 |
| at-fault collision | 9 | **0.649** | |

MF-PDMS does not distinguish the clips where VaVAM drives cleanly from the clips where
it leaves the road or crashes. The at-fault split, if anything, points the wrong way.

**An n=10 result that did not replicate.** On the first 10 scenes `dist_traveled_m`
correlated at rho = -0.842 (p=0.002, surviving Bonferroni), which looked like strong
evidence of *anti*-correlation. At n=40 it is -0.213 (p=0.186). The n=10 finding was
noise dressed up by a small sample and 8 simultaneous tests. It is recorded here
because it is a good example of why these numbers need n in the tens, not a handful.

## What this means

Open-loop MF-PDMS and closed-loop AlpaSim measure different things, and the open-loop
metric carries **little to no per-clip information** about closed-loop outcomes. The
two also disagree at the policy level, in the direction that matters:

| policy | MF-PDMS (open-loop, 137 clips) | AlpaSim `progress` (closed-loop) |
|---|---|---|
| `constant_velocity` | **0.857** | 0.190 |
| VaVAM | 0.489 | **0.471** |

Open-loop ranks a policy that cannot steer *above* a real driving model. Closed-loop
reverses it. (Unconfirmed since 2026-10-07: the closed-loop `constant_velocity` here braked at
handover because of a zero-speed bug in our bridge; see the correction under Batch 2.) The mechanism is not subtle: open-loop resets the ego to ground truth at
every decision point, so heading error never compounds, and it scores a policy by
similarity to the human trajectory rather than by whether the result is drivable.

This does not retire MF-PDMS. It is ~28x cheaper per clip than VaVAM open-loop and
does not need NuRec artifacts at all, and it still separates competence classes
(oracle 0.968 / naive ~0.86 / degenerate 0.241). But it should be described as a
cheap, dataset-agnostic screen, **not** as a predictor of closed-loop driving quality
per clip, and it must not be used to rank policies against each other.

**This is the transferability result.** A second, independent driving policy — trained by a
different group, on different data, through a different driver path (native AlpaSim
rather than our policy bridge) — runs end to end and produces a coherent, separable
score. The evaluator is not tied to one model.

## Batch 3: 40 more scenes, n=80 (2026-09-21)

Forty further NuRec scenes, chosen by a seeded shuffle of the 97 open-loop-scored clips
that had no closed-loop result (not by any score, so the labelled set stays unbiased for
the rollout-triage work in `SKIP.md`). Native `driver=vavam`, same renderer, local A10:
40 scenes in 25 min wall including downloads, ~38 s per scene of simulation.

| VaVAM, closed-loop | n=40 (batches 1-2) | n=80 (batches 1-3) |
|---|---|---|
| `offroad_or_collision` | 0.650 | 0.625 |
| `offroad_or_collision_at_fault` | 0.550 | 0.513 |
| `collision_at_fault` | 0.225 | 0.188 |
| `offroad` | 0.325 | 0.325 |
| `progress_rel` | — | 0.911 |

The rates are stable between halves, so the n=40 picture of the policy was not a small-
sample artefact; what *was* is the strength of any per-clip predictor of these outcomes,
which is the subject of `SKIP.md`. Per-clip rows for all 80 are in
`.cl_vavam_perclip.parquet` (`alpasim/per_clip.py -o` over the three run directories).
