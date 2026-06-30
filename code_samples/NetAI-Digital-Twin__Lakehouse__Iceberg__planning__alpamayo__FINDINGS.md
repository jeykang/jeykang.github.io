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
