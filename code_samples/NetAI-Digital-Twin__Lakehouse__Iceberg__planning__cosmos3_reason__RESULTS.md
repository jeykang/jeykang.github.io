# Cosmos3-Edge difficulty axis — measured result (2026-08-10)

Re-test of the axis `planning/alpamayo/FINDINGS.md` shelved in June. Same 452-clip
OOD label set, same battery, same baselines. Two paths measured: `cold` (one forward
pass) and `reasoned` (free-text rationale, then logit-EV conditioned on it).

## Headline

**Compute was not the only thing holding this back, and the reasoned path is the
first VLM axis in this repo to clear the bar.**

- `cold` — 0.636, beats every Alpamayo variant at 1/25th the cost, but adds nothing
  to production. Dead end, cleanly.
- `reasoned` — **0.670 alone, and it makes the production union 0.630 -> 0.690 with
  a bootstrap CI that excludes zero.** Nearly uncorrelated with `conflict`
  (Spearman +0.047), so it is additive rather than redundant.
- out-of-family control (Gemma-4-E4B) — reproduces the direction (+0.033, ns) and,
  decisively, carries **independent** private signal (residual AUC 0.616 after
  Cosmos3 is projected out). Circularity cannot be the whole explanation.
- **downstream — nothing.** Against a camera-only consumer detector chosen to be
  independent of every production axis, cosmos3 correlates **-0.024** with consumer
  difficulty and improves the selection at no fraction. The axis is real against the
  labels and useless against the job. **This closes the line.**

**Status: CLOSED — not adopted.** The axis is real (scene-grounded, out-of-family
replicated, +0.06 on the production union against `ood_reasoning`) but has **no
downstream utility**: correlation with camera-only consumer difficulty is -0.024, and
adding it to the selection beats PRODUCTION at no fraction tested. The label-side
lift was real and irrelevant. See "Downstream utility test".

## Battery, both paths (N=452)

| | cold | reasoned | Alpamayo-1.5 (10B) |
|---|---|---|---|
| OOD AUC | 0.636 | **0.670** | 0.437 / 0.565 / 0.604 |
| s/clip (scoring only) | **1.3** | 2.3 | 6.6 / ~8 / 66.5 |
| s/clip (incl. swap control) | 2.68 | 4.69 | — |
| 33k cohort | ~12 h | **~21 h** | ~610 h |
| frames used | 4 @ 640x360 | 4 @ 640x360 | 1 (VRAM-capped) |
| VRAM peak | 5.28 GB | 5.28 GB | 22-23 GB |
| parse success | 100% | 100% | 100% |
| determinism | 0.0000 / 10 | 0.0000 / 10 | deterministic |
| score spread (sd) | 0.134 | 0.237 | — |

The reasoning step reproduces the Alpamayo progression (0.437 -> 0.565 -> 0.604 there;
0.636 -> 0.670 here) — but from a much higher floor, and this time it crosses conflict.

## Negative controls

**Swapped frames (the control the June notes asked for and never ran).** Score each
clip against a *different* clip's video, keep its label:

| | real | swapped | scene-attributable |
|---|---|---|---|
| cold | 0.636 | 0.473 | +0.163 |
| reasoned | 0.670 | **0.501** | **+0.169** |

The signal collapses to exact chance when the video is wrong. This is the strongest
grounding evidence any VLM path here has produced — compare Addendum 2's `mistake`,
where blanked frames still scored 0.609 against a real 0.706, i.e. most of that
signal was prior, not scene.

**Blank frames — the control used throughout the June battery is inverted for the
cold path.** Cold real-blank = **-0.222**, with **0/40** clips above their blank
version: a zeroed frame is not an absence of scene, it is a plausible *pitch-dark,
zero-visibility* scene, and the model rates it 0.943. The reasoned path does not
have this failure (+0.599, blank mean 0.006 — with a rationale in front of it the
model says it cannot see anything and scores 0). So blank-frame margins are only
meaningful for paths that verbalise first; the June cold/CoC margins (+0.10, +0.03)
were measured against a control pushing the wrong way. Their conclusions were
negative anyway, but the control cannot be cited *for* grounding.

## Does it earn a place? (`analyze_axis.py`, 2000 bootstrap resamples)

Axes rank-normalized and combined by noisy-OR exactly as `edge_case_scorer` does,
with the full production perceptual leg `max(darkness, camera_gated)`:

| combination | cold | reasoned |
|---|---|---|
| cosmos3 alone | 0.636 | 0.670 |
| conflict alone | 0.645 | 0.645 |
| camera_gated alone | 0.579 | 0.579 |
| darkness alone | 0.533 | 0.533 |
| PRODUCTION = OR(conflict, perceptual) | 0.630 | 0.630 |
| OR(conflict, cosmos3) | 0.661 | **0.709** |
| PRODUCTION + cosmos3 | 0.641 | **0.690** |

95% CI on the AUC delta:

| delta | cold | reasoned |
|---|---|---|
| cosmos3 - conflict | [-0.070, +0.050] ns | [-0.049, +0.094] ns |
| OR(conflict, cosmos3) - conflict | [-0.022, +0.052] ns | **[+0.020, +0.106] SIG** |
| PRODUCTION+cosmos3 - PRODUCTION | [-0.023, +0.043] ns | **[+0.024, +0.096] SIG** |

Note what this does and does not say. As a *standalone* axis it does **not** beat
conflict (CI straddles zero, both paths). Its value is entirely **additive** — which
is the right question, because production combines axes by noisy-OR, not by picking
one.

**It is not just re-deriving darkness.** That was the obvious way for this to be a
strawman: the model plainly reacts to lighting and weather, and production already
gets darkness free from `hour_of_day`. The comparison above includes darkness in the
production baseline, and the lift survives it (darkness alone is only 0.533 here).

**The disagreement pocket is enriched, and only for the reasoned path:**

| region | cold n / OOD | reasoned n / OOD | base |
|---|---|---|---|
| cosmos3-hard / conflict-easy | 21 / 0.38 | **25 / 0.52** | 0.45 |
| conflict-hard / cosmos3-easy | 8 / 0.38 | 30 / 0.43 | 0.45 |

Clips the reasoned axis uniquely flags are *above* base rate for human-hard; the cold
path's are below. That is the mechanism behind the union lift, and it is the check
that failed in Addendum 2.

## Circularity control: Gemma 4 E4B (2026-08-10)

The test proposed below was run. Backend: `google/gemma-4-E4B-it` (Apache-2.0,
4B-effective, 15.9 GB BF16, 17.3 GB peak on the A10) — different lab, different
corpus, size-matched. Battery byte-identical: same frames, prompts, steer, logit-EV
readout; only the checkpoint changed (verified by a regression check that Cosmos3
still reproduces its first-clip score to 1e-9 after the refactor).

| reasoned path | Cosmos3-Edge (4B) | Gemma-4-E4B (4B-eff) |
|---|---|---|
| OOD AUC | 0.670 | 0.642 |
| swapped-frame AUC | 0.501 | 0.496 |
| scene-attributable | +0.169 | +0.146 |
| Spearman vs conflict | +0.047 | +0.080 |
| s/clip | 2.3 | 10.1 |
| PRODUCTION + model | 0.690 | 0.663 |
| CI on the union lift | **[+0.024, +0.096] SIG** | [-0.001, +0.067] ns |

**The direction replicates out-of-family; the significance does not.** Gemma is
independently scene-grounded (swap collapses to 0.496), independently near-orthogonal
to conflict (+0.080), and lifts the production union by +0.033 — half of Cosmos3's
+0.060, with a CI whose lower bound sits exactly on zero. On its own this is
ambiguous: it is equally consistent with "the effect is real and Cosmos3 is simply
better at driving scenes" and with "part of Cosmos3's margin is family leakage".

### What separates those two readings

If Cosmos3's margin were label leakage, its *private* component — the part Gemma
cannot explain — would carry the signal, and Gemma's private component would be
noise. Measured (`analyze_axis.py --compare`):

| | AUC |
|---|---|
| agreement between the two backends (Spearman) | **+0.155** |
| Cosmos3 residual, after removing Gemma | 0.653 |
| Gemma residual, after removing Cosmos3 | **0.616** |
| ensemble (mean rank) | **0.705** |
| PRODUCTION + ensemble | **0.694** |
| CI on the ensemble union lift | **[+0.041, +0.090] SIG** |

Three things follow.

1. **The out-of-family model has real private signal.** Gemma's residual predicts
   human-hard at 0.616 after Cosmos3's contribution is projected out. Leakage from
   an Alpamayo-lineage labeller cannot explain a Google model's independent
   component. Circularity is therefore not the whole story — which was the open
   question blocking adoption.
2. **The two VLMs barely agree (+0.155)** yet both beat chance by a similar margin.
   They are capturing different partially-valid slices of "hard", not a shared prior.
3. **The ensemble is better than either alone** (0.705 vs 0.670 / 0.642) and its
   union lift has the strongest CI in this whole line of work, with the lower bound
   well clear of zero.

Cosmos3 still beats Gemma by 0.028 while being 4x smaller and 4x faster, which is
consistent with NVIDIA's published driving-reasoning gap rather than requiring a
leakage explanation. But that residual 0.028 is the one quantity this design cannot
cleanly attribute, and it should not be leaned on.

### Caveat on the ensemble result

The ensemble was **not** the pre-registered test — the pre-registered test was "does
an out-of-family model reproduce the lift", and its honest answer is *direction yes,
significance no*. The ensemble was constructed after seeing both runs, so its CI is
not corrected for that choice. Given this repo's history with post-hoc point
estimates (Addendum 2: 0.706 -> 0.535), the ensemble should be treated as a strong
hypothesis requiring its own confirmation, not as a settled number.

## The caveat that motivated the control (kept for the record)

**Circularity.** The `ood_reasoning` labels are Alpamayo-lineage (Cosmos-Reason2
backbone), and Cosmos3-Edge is the same model family, so a VLM-family scorer is
structurally favoured — the June notes already flagged the AUC here as "generous, not
independent". Unlike Addendum 2's failure this is **not** fixable by more clips:
N=452 is the entire label set, and the bootstrap addresses sampling noise, not label
bias. Hence the out-of-family control above, which is the only design that could
speak to it.

## Human spot-review (2026-08-11) — INCONCLUSIVE, and the design is why

60 clips, blind, shuffled: A = cosmos3-hard/conflict-easy (25), B = both-easy (25),
plus 5 both-hard and 5 reverse-pocket. No scores, labels or rationales shown.

| stratum | n | mean human rating (0-3) | OOD base rate |
|---|---|---|---|
| A cosmos3-hard / conflict-easy | 25 | 1.56 | 0.52 |
| B both-easy | 25 | 1.36 | 0.16 |
| both-hard anchor | 5 | 1.40 | — |
| reverse (conflict-hard / cosmos3-easy) | 5 | 1.20 | — |

**Decisive contrast: not confirmed.** A - B = **+0.20**, in the predicted direction,
bootstrap 95% CI [-0.36, +0.76], Mann-Whitney p=0.50, effect size 0.56.

### The design was under-powered, and that is the finding

| | |
|---|---|
| pooled sd of human ratings | 1.040 |
| observed difference | +0.20 (Cohen d = 0.19) |
| **smallest difference this design could detect** | **0.82** |
| n per stratum needed to detect +0.20 at 80% power | **424** (we had 25) |

At n=25 per stratum on a 4-point scale, this instrument could only ever have
resolved a *large* effect. The true effect, if +0.20 is near it, is small enough that
**no feasible human review resolves it** — 424 clips per stratum is ~850 blind
ratings for one axis check. This is a limitation of the instrument I built, not a
property of the axis, and it should not be read as evidence against cosmos3.

The scale guard passed only technically: the both-hard anchor (1.40) exceeded
both-easy (1.36) by 0.04 on n=5. That margin is noise. The guard was too small to
validate anything and should have been 15-20 clips.

### Two things it did establish

**The reviewer is not the problem.** The stated worry was that a non-driver cannot
judge driving difficulty. The ratings say otherwise: AUC(human -> ood_reasoning) =
**0.640** overall and **0.821** within stratum B — comparable to cosmos3's 0.698 on
the same 60 clips. These ratings carry real signal about the label set; they are not
noise.

**The likelier instrument flaw is static frames, not the reviewer.** Human ratings
correlate **-0.222** with the production `conflict` axis. `conflict` measures
agent-interaction and closing dynamics — cut-ins, closing speed, occlusion over time
— which four still frames physically cannot show. A contact sheet can display
clutter and visibility but not motion, so it under-represents exactly the construct
conflict encodes. Any future review of a temporal axis needs short video, not stills.

### The tension worth recording

The OOD labels separate A from B strongly (0.52 vs 0.16 base rate). The human barely
separates them (1.56 vs 1.36). Two readings, and this design cannot choose between
them: either the human cannot see from stills what makes A hard, or the labels'
A/B separation is itself partly VLM-lineage artifact. Given the reviewer tracks the
labels well overall (0.640), the first reading is more plausible — but that is an
argument, not a measurement.

## Recommended next step (as written before the downstream test — superseded)

Both remaining anchors have now been tried, and the human one is closed as
impractical. That changes the recommendation.

**Stop trying to validate this axis against difficulty *labels*.** The measured
effect (union lift ~+0.06 AUC; human-visible difference d=0.19) is real enough to
survive bootstrap on 452 clips but too small for human review to confirm at any
tractable sample size, and the only label set available is lineage-contaminated.
Further label-side work has poor expected yield.

**Validate downstream instead — the anchor the pipeline actually cares about.**
The reason a difficulty axis exists is to mine clips that expose failures in the
camera-only consumer model. That is directly measurable and needs no human panel and
no OOD labels: take the top-N by PRODUCTION vs top-N by PRODUCTION+cosmos3, and
compare the camera-only detector's failure rate on the two selections. If the
cosmos3-augmented selection surfaces materially more camera failures, the axis earns
its place on the merit that matters; if not, the +0.06 against lineage labels was
never worth acting on. The machinery already exists
(`planning/camera_perception_runner.py` scores all 33,767 clips).

**Cost, if adopted.** Cosmos3 alone is ~21 h for the 33k cohort and already
significant. The ensemble is ~145 h (Gemma is 4x slower) for roughly +0.004 on the
union. Cosmos3 alone is the sane production choice; the ensemble is evidence, not a
pipeline.

## Downstream utility test (2026-08-11) — CLOSED: no utility

The anchor that needs no labels and no panel. Pre-registered before any outcome was
computed (see `downstream_utility.py` docstring). Random cohort sample, n=1500 (the
452 gate clips are label-enriched at 45% OOD and would not resemble production
selection). Consumer proxy deliberately independent of the axis: **yolov8n** at
frame fractions 0.15/0.45/0.85, where `camera_low_conf` was built with **yolo11x** at
0.3/0.5/0.7. Failure = agents present (behavioral) AND consumer confidence < 0.5.

Cohort baseline: 80% of clips have agents; failure rate given agents = **0.356**.

Failure rate given agents, by selection and fraction:

| top-N | PRODUCTION | PROD+cosmos3 | cosmos3 alone | conflict alone | perceptual alone |
|---|---|---|---|---|---|
| 5% | 0.145 | 0.238 | 0.333 | 0.013 | 0.604 |
| 10% | 0.226 | 0.216 | 0.307 | 0.013 | 0.651 |
| 20% | 0.283 | 0.270 | 0.317 | 0.043 | 0.677 |
| 30% | 0.312 | 0.275 | 0.327 | 0.058 | 0.623 |
| *cohort base* | *0.356* | | | | |

**cosmos3 has no downstream utility.** Its correlation with consumer difficulty is
**-0.024** — zero. Selection-level: the clips it swaps into the top 10% fail at
0.300 vs 0.308 for the ones it pushes out (CI [-0.228, +0.088]). At no fraction does
adding it beat PRODUCTION, and alone it tracks the cohort base rate. This is
unambiguous and it does not depend on any label set.

A trap worth recording: on raw mean confidence the swapped-in clips looked *harder*
(0.418 vs 0.579), which reads as a win. It is the empty-scene confound this repo
already documented for the raw camera axis — swapped-in clips are only 70%
agents-present vs 91% pushed-out, so they score low confidence by being **empty**,
not by being hard. Gating on agent presence erases the entire effect. Any future
camera-difficulty work must gate before reading a confidence number.

### The finding that is not about cosmos3

Correlations with the independent consumer detector, over the 1207 agent-present clips:

| axis | Spearman vs consumer confidence | circular? |
|---|---|---|
| `conflict` (behavioral) | **+0.689** | no — different detector, different frames |
| `camera_gated` (perceptual) | -0.705 | **yes, largely** — shares a detector lineage |
| `darkness` (perceptual) | -0.150 | no |
| cosmos3 | -0.024 | no |

**The behavioral axis strongly anti-selects camera failures, and this is not an
artifact.** `conflict` correlates +0.689 with consumer *confidence*: agent-dense
scenes contain large, close, unambiguous objects that a camera detector finds easily.
Its top-10% selection fails at **0.013** against a 0.356 base. Because the production
composite noisy-ORs behavioral with perceptual, that leg drags the union *below the
cohort base rate* at every fraction (0.145-0.312 vs 0.356) — the union is worse than
random sampling at finding clips that break a camera-only consumer, while the
perceptual leg alone is nearly 2x base.

This is not a bug: `conflict` measures agent-interaction difficulty, a different and
deliberately-included construct, and the union exists to keep clips hard on *either*
axis. But if the mining goal is the camera-only endgame the consumer actually ships,
unioning behavioral in halves the yield. Note also that perceptual's 0.65 is inflated
by the -0.705 shared-detector correlation; its clean, non-circular component is
darkness at -0.150. Recorded in `nvidia_ingestion/VALIDITY_BATTERY_FINDINGS.md`.

## What is settled regardless

1. The June shelving was **partly over-attributed to hardware**. At the full config
   the 10B model could never afford, a 4B model reaches 0.670 — but note the cold
   path, run at that same full config, still lands at 0.636 and adds nothing. So
   hardware was masking a real signal in the *reasoned* path only.
2. There is now a cheap, deterministic, **verifiably scene-grounded** driving VLM on
   this host: 2.3 s/clip, 5.3 GB, one A10, no cluster booking. Independent of the
   difficulty question, that makes it a credible instrument for the augmentation
   **hallucination gate**, which currently infers added agents from YOLO detection
   counts and which Edge could answer directly (its reasoning output supports
   bounding boxes and 2D/3D point localization).

## Reproduce
```bash
./c3_venv/bin/python gate_runner.py 452 --mode cold
./c3_venv/bin/python gate_runner.py 452 --mode reasoned
./c3_venv/bin/python gate_runner.py 452 --mode reasoned --model gemma4-e4b
./c3_venv/bin/python analyze_axis.py .gate_cosmos3-edge_cold_452.json \
                                    .gate_cosmos3-edge_reasoned_452.json \
                                    .gate_gemma4-e4b_reasoned_452.json
./c3_venv/bin/python analyze_axis.py --compare .gate_cosmos3-edge_reasoned_452.json \
                                               .gate_gemma4-e4b_reasoned_452.json
```
Gemma's `cold` path was not run: the cold path was already established as a dead end
in-family (adds nothing, CI [-0.023, +0.043]), so an out-of-family cold control has
nothing to confirm or refute.
