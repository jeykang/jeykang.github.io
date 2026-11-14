# Alpamayo-1.5 VLM difficulty scorer — SHELVED (2026-06-26)

**Idea**: have a reasoning VLM (`nvidia/Alpamayo-1.5-10B`, Apache-2.0, built on the
Cosmos-Reason2 backbone) *judge* per-clip driving difficulty directly, instead of
running a planner. This sidesteps the planner-**transfer** failure that sank
SparseDrive/DiffusionDrive (no driving — just judging), and Alpamayo is native to
the NVIDIA PhysicalAI dataset family.

**Verdict: SHELVED.** It works and the reasoning is genuinely good, but the
difficulty *signal* does not beat the existing `conflict` axis, is only weakly
scene-grounded, and is wildly impractical on this hardware. Production stays the
validated **conflict + darkness** noisy-OR union (see `nvidia_ingestion/VALIDITY_BATTERY_FINDINGS.md`).

## Setup (reproducible)
- Vendored repo `NVlabs/alpamayo1.5` + `uv` venv `a1_5_venv` (Python 3.12, torch
  2.8+cu128, transformers 4.57, **SDPA** — no `nvcc` on this host), both gitignored.
- Weights pulled from HF (`nvidia/Alpamayo-1.5-10B`, ~20 GB) via the existing
  `jeykang-gist` login (model not gated; dataset access granted).
- Frames streamed by `physical_ai_av` by `clip_id` (same dataset as our subset).
- Run on RTX 6000 (`CUDA_VISIBLE_DEVICES=1`, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`).

## What was tried, vs the production baseline
Gate = the 452-clip OOD-labelled set (`/tmp/conf/clips.txt`); circularity caveat:
`ood_reasoning` is Alpamayo-lineage, so AUC here is generous, not independent.

| Approach | OOD AUC | ρ vs conflict | neg-control | throughput |
|---|---|---|---|---|
| cold VQA digit (logit-EV) | 0.437 | −0.158 | +0.10 | 6.6 s/clip |
| reasoned VQA (free reason → score) | 0.565 | +0.04 | +0.21 | ~8 s/clip |
| **native CoC rollout** (best) | **0.604** | +0.15 | **+0.03** | **66.5 s/clip** |
| minADE (trajectory prediction error) | 0.350 | — | — | — |
| **conflict (production)** | **0.651** | — | 0.10→0.003 | GPU-free |

## Key findings
1. **Output-format problem solved** — reading the next-token distribution over
   digits 0–9 (logit expected-value) gives a continuous, 100%-parseable,
   *deterministic* score (single forward, no sampling). This is the reusable bit.
2. **The model sees scenes well** — free descriptions are accurate ("stopped truck
   blocking the lane", "construction cones blocking the center", "cut-in vehicle
   merging"). The CoC reasoning is high quality.
3. **But the difficulty scalar is weak**: best (CoC) AUC 0.604 < conflict 0.651,
   and the cold snap-judgment is *anti-aligned* (0.437) because it defaults to a
   "dark = hard" prior. Reasoning helps (0.437→0.565→0.604) but never clears conflict.
4. **Grounding weakens with reasoning**: CoC neg-control is only **+0.03** — the
   model hallucinates a plausible chain-of-causation even on blanked frames, so the
   score is substantially prior-driven. Real validity red flag.
5. **`minADE` (planning error) is anti-aligned (0.350)** — re-confirms prediction
   error tracks ego-kinematics, not difficulty (same lesson as rung-0/DiffusionDrive).
6. **Infeasible on 24 GB**: the 10B model is 22 GB resident; the CoC rollout
   (diffusion expert + generation) only fits at a degraded **1-frame / 64-token**
   config (~23 GB peak), and runs at **66.5 s/clip → ~610 h for the 33k sample**.

## If revisited (≥40 GB GPU)
The CoC reasoning quality justifies a retry on bigger hardware (H100/A100):
full-config rollout (4 frames, 256-token reasoning, batched), stronger negative
control (shuffled-frame, not just blank), and an independent validation anchor
(not `ood_reasoning`). Even then it must beat conflict's 0.651 to earn a place.

## Files
- `difficulty_qa.py` — model load + logit-EV scorer (VQA path).
- `gate_runner.py` — VQA gate (cold logit-EV).
- `reasoned_gate.py` — reason-then-extract (VQA) gate.
- `coc_gate.py` — native CoC-rollout gate (+ minADE).
- Env/weights/vendored repo are gitignored (`alpamayo1.5/`, `*venv*`).

## Addendum — "model struggle" signals (2026-06-26)

Followup hypothesis: instead of asking Alpamayo to *judge* difficulty, score clips
by *how hard a time the model had* — i.e. uncertainty in the VLA pipeline (the VLM
feeds a diffusion action expert). Tested 3 internal-struggle signals
(`struggle_gate.py`, N=40, K=3 trajectory samples):

| Signal | OOD AUC | ρ vs conflict | neg-control |
|---|---|---|---|
| trajectory spread (action-expert multimodality) | 0.448 | −0.295 | moved −10.8 (20/20) |
| reasoning entropy (VLM generation) | 0.501 | +0.05 | moved +0.31 |
| minADE (prediction error) | 0.335 | −0.13 | — |

**Result: closed.** All three are anti-aligned or null vs human-hard, and spread
is *negatively* correlated with the validated conflict signal. The decisive
insight is the negative control: Alpamayo's trajectory spread **does** respond to
the scene (moves a lot on blanked frames — unlike DiffusionDrive's scene-blind
`mode_spread`), so the native model fixed scene-grounding. But it measures the
**inverse construct** — trajectory spread = the controller's *freedom*, which is
highest on easy/open/empty scenes and lowest on constrained/hard ones. So **model
uncertainty ≈ scene openness ≈ inverse of difficulty**, which explains why the
whole family (mode_spread → minADE → action-expert spread) fails. Not a transfer
problem; a construct problem. The "struggle" framing is conclusively shelved.

## Addendum 2 — consequential-failure ("model struggle done right") (2026-06-27)

Final, strongest version of "score clips by how hard a time the model had": does
ALPAMAYO'S planned trajectory come unsafe vs the recorded agents (NAVSIM-PDMS over
a native planner's actual output) — `pdms_planner_gate.py`, `mistake_confirm.py`.
Three signals: collision severity, path proximity, and **`mistake`** = how much
closer Alpamayo's path gets to an agent than the human path did (the consequential
model-error part, with scene-density cancelled out).

**n=40 gate looked like a breakthrough**: `mistake` OOD AUC 0.706 (> conflict 0.651)
AND ρ=+0.05 with conflict (independent axis). **N=200 confirmation killed it:**

| | n=40 gate | N=200 confirm |
|---|---|---|
| mistake OOD AUC | 0.706 | **0.535** |
| ρ vs conflict | +0.05 | −0.06 |
| neg-control real AUC (first 40) | — | 0.706 |
| neg-control BLANK AUC (first 40) | — | **0.609** |
| determinism \|Δ\| | — | 0.000 |

Two failures: (1) **small-sample luck** — AUC 0.706→0.535 at scale (barely above
chance, below conflict); (2) **negative control partially fails** — blanked frames
still give 0.609, so most of the signal isn't scene-grounded, and real-mean (2.10) >
blank-mean (1.86) means scene-informed planning gets *closer* to agents (normal
interactive driving, not error). Construct is muddy + weak. **Closed.**

This is the 4th confirmation of the agent-interaction ~0.65 ceiling (static proximity
/ feasibility sim / planner collision / planner deviation all land there). It's a
construct+label-set property, not a missing metric. The validity battery (larger-N +
neg-control) caught an appealing n=40 false positive — same discipline that caught
mode_spread. Driving-agent difficulty scoring is conclusively exhausted; production
stays conflict + darkness (perception-confidence IS the real "model struggle" leg).

## Addendum 3 — RE-OPENED on Cosmos3-Edge (2026-08-10)

The "if revisited (>=40 GB GPU)" note above has been acted on, from the other
direction: instead of a bigger GPU, a smaller model. NVIDIA's Cosmos3-Edge (4B,
released 2026-07-20) is 4.87 GB of reasoner weights vs Alpamayo's 22 GB, so the
FULL config this section could never afford — 4 frames, full reasoning budget —
now runs on the A10 at 5.28 GB peak. New detachable module:
`planning/cosmos3_reason/` (FEASIBILITY.md, RESULTS.md).

Same 452-clip gate, same battery, same conflict baseline:

| path | OOD AUC | s/clip | 33k | vs this section |
|---|---|---|---|---|
| alpamayo cold VQA (10B) | 0.437 | 6.6 | ~55h | — |
| alpamayo reasoned VQA (10B) | 0.565 | ~8 | ~73h | — |
| alpamayo CoC rollout (10B) | 0.604 | 66.5 | ~610h | — |
| conflict (production) | 0.651 | — | — | — |
| **cosmos3-edge cold (4B)** | 0.636 | 1.3 | ~12h | beats all 3 |
| **cosmos3-edge reasoned (4B)** | **0.670** | 2.3 | ~21h | beats conflict |

Three things this changes.

**1. The negative control used in this section is inverted for cold paths.** Zeroing
frames does not remove the scene, it creates a *pitch-dark, zero-visibility* scene,
which a model with the dark=hard prior scores near the top (0.943; 0/40 clips beat
their own blank). The +0.10 / +0.21 / +0.03 blank margins above, and the 0.609 blank
AUC in Addendum 2, were measured against a control pushing the wrong way. The
conclusions there were negative regardless, but blank-frame margins must not be
cited as evidence *for* grounding. Use the shuffled-frame control this section asked
for — now implemented (`gate_runner.py --mode`, one extra forward pass).

**2. Under that control, the VLM axis IS scene-grounded** — the thing this section
doubted. Scoring each clip against a *different* clip's video collapses AUC to
0.501 (exact chance) from 0.670, i.e. +0.169 is scene-attributable. Contrast
Addendum 2's `mistake`, where blanked frames still scored 0.609 vs a real 0.706.

**3. The reasoned axis is ADDITIVE to production, significantly.** Not as a
replacement — standalone it does not beat conflict (CI straddles zero). But
noisy-OR'd into the production union (conflict OR max(darkness, camera_gated)):
0.630 -> **0.690**, bootstrap 95% CI **[+0.024, +0.096]**. Spearman vs conflict is
only +0.047, i.e. a near-independent axis, and its unique-flag pocket is enriched
for human-hard (n=25, OOD 0.52 vs base 0.45) — the exact check that failed in
Addendum 2. The lift survives including darkness in the baseline, so it is not
re-deriving `hour_of_day`.

**Not adopted yet.** The blocker is the circularity this section already flagged:
`ood_reasoning` is Alpamayo-lineage and Cosmos3-Edge is the same family, so these
labels structurally favour it. Unlike Addendum 2 this cannot be settled with more
clips (N=452 is the whole label set). Decisive test proposed in RESULTS.md: re-run
the identical battery with a non-NVIDIA VLM (Qwen3-2B-VL-Instruct, ~1 GPU-hour). If
an out-of-family model reproduces the union lift, adopt; if the lift is
Cosmos-family-only, it is an artifact and this line closes for good.

So the June verdict stands as written for the *cold* path (0.636, adds nothing:
CI [-0.023, +0.043]) and the ~0.65 ceiling holds for it. It does NOT stand for the
reasoned path, which was never run at full config on this hardware.

### Addendum 3b — circularity control run (Gemma 4 E4B, 2026-08-10)

The blocker in Addendum 3 was that `ood_reasoning` is Alpamayo-lineage and
Cosmos3-Edge is the same family. Control: identical battery, out-of-family backend
(`google/gemma-4-E4B-it`, Apache-2.0, 4B-effective, different lab and corpus). Only
the checkpoint changed — Cosmos3 reproduces its prior score to 1e-9 after the
refactor, so the comparison is exact.

| reasoned | Cosmos3-Edge | Gemma-4-E4B |
|---|---|---|
| OOD AUC | 0.670 | 0.642 |
| swapped-frame AUC | 0.501 | 0.496 |
| rho vs conflict | +0.047 | +0.080 |
| PRODUCTION + model | 0.690 | 0.663 |
| CI on union lift | [+0.024, +0.096] SIG | [-0.001, +0.067] ns |

Direction replicates, significance does not — ambiguous on its own. The
disambiguating measurement is the residual: if Cosmos3's margin were leakage, only
its private component would predict, and the out-of-family model's private component
would be noise. Instead **Gemma's residual, after projecting out Cosmos3, still
scores AUC 0.616**, and the two backends agree only at rho=+0.155. An out-of-family
model has genuine independent signal on these labels, so lineage leakage is not the
whole story. Their ensemble reaches 0.705 and lifts the production union with the
tightest CI yet, [+0.041, +0.090] — but it was constructed post-hoc, so per the
Addendum 2 lesson it is a hypothesis, not a number to bank.

Net: circularity is substantially — not entirely — cleared. The remaining
unattributable quantity is Cosmos3's 0.028 margin over Gemma, which is equally
consistent with NVIDIA's published driving-reasoning gap. Remaining anchor that
sidesteps the label set entirely: human spot-review of the 25 cosmos3-hard /
conflict-easy clips. Full detail: `planning/cosmos3_reason/RESULTS.md`.

### Addendum 3c — human spot-review: inconclusive, and human review retired as an anchor (2026-08-11)

Blind 60-clip review (`spot_review.py` / `score_review.py`), stratified so the
contrast isolates cosmos3: A = cosmos3-hard/conflict-easy (25) vs B = both-easy (25),
both conflict-easy. A-B = **+0.20** on a 0-3 scale, CI [-0.36, +0.76], p=0.50 —
**not confirmed**.

The diagnostics matter more than the verdict. Pooled sd 1.04, so the smallest
difference this design could detect was **0.82**; detecting the observed +0.20 at 80%
power needs **424 clips per stratum**, not 25. The effect is too small for human
review to resolve at any tractable sample size. This is an instrument limitation, not
evidence against the axis, and the both-hard anchor (n=5, margin 0.04) was too small
to validate the scale at all — both are design errors worth not repeating.

Two useful by-products:
- **The reviewer flagged themselves as a non-driver; the data says that was not the
  binding problem.** AUC(human -> ood_reasoning) = 0.640 overall, 0.821 within
  stratum B, vs cosmos3's 0.698 on the same clips. The ratings carry real signal.
- **Static frames are the likelier flaw.** Human ratings correlate **-0.222** with
  `conflict`, which encodes closing/cut-in dynamics that four stills cannot show. Any
  future review of a temporal axis must use short video, not contact sheets.

**Consequence for this line of work:** both label-side anchors are now exhausted —
the only OOD label set is Alpamayo-lineage, and human review cannot resolve an effect
this size. Further validation against *difficulty labels* has poor expected yield.
The remaining anchor is downstream utility: compare camera-only detector failure
rates on top-N selected by PRODUCTION vs PRODUCTION+cosmos3. That needs no panel and
no labels, and it is the merit the axis exists to serve.

### Addendum 3d — downstream utility: none. Line CLOSED (2026-08-11)

The label-free anchor, pre-registered. Random cohort sample n=1500 (not the
label-enriched gate set), consumer proxy chosen independent of every production axis
(`yolov8n` @ 0.15/0.45/0.85 vs the axis's `yolo11x` @ 0.3/0.5/0.7). Failure = agents
present AND consumer confidence < 0.5; cohort failure|agents = 0.356.

**cosmos3 correlates -0.024 with consumer difficulty.** Zero. Its top-10% selection
fails at 0.307 (vs base 0.356); clips it swaps into the PRODUCTION top-10% fail at
0.300 vs 0.308 for those it displaces (CI [-0.228, +0.088]); it beats PRODUCTION at
no fraction from 5% to 30%.

So the axis is simultaneously **real and useless**: scene-grounded under a
swapped-frame control, replicated out-of-family, +0.06 on the production union
against `ood_reasoning` with a CI excluding zero — and worth nothing for the job the
axis exists to do. The label-side lift was real and irrelevant. That reconciles this
whole line: every prior verdict here was "doesn't beat conflict"; the correct verdict
was "the target itself doesn't predict downstream failure".

Trap recorded for reuse: on raw mean confidence, cosmos3's swapped-in clips looked
harder (0.418 vs 0.579). They are 70% agents-present vs 91% — they score low
confidence by being **empty**, not hard. Gating on agent presence erases the effect
entirely. Same empty-scene confound as the raw camera axis; gate before reading any
confidence number.

**Line closed.** Five model generations (SparseDrive -> DiffusionDrive -> Alpamayo-1.5
10B -> Cosmos3-Edge 4B -> Gemma-4-E4B), three anchors (OOD labels, human review,
downstream utility). Bigger and better models moved the label-side metric and never
moved the outcome. The one thing this round did surface that changes production is
about the existing axes, not the candidate: `conflict` anti-selects camera failures
at +0.689, dragging the union below cohort base rate — see
`nvidia_ingestion/VALIDITY_BATTERY_FINDINGS.md`.
