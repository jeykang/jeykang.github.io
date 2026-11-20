# Rollout triage — which curated clips earn a closed-loop rollout (2026-09-21)

The question from the Sep 4 call: can the proving ground skip what will predictably
fail, and can that skipping be a tool rather than a per-run judgement? `skip.py` is the
tool; this file is what it measures so far. Policy under test: VaVAM, closed-loop in
AlpaSim over NuRec scenes, n = 80 clips (batches 1–3, `ALPASIM.md`).

## What the screen is

A per-clip predictor of closed-loop failure fitted on features that cost seconds, not
minutes: the three curation axes scored on the same clips (`ax_*`: conflict load,
behavioral axes, camera-only perception, 143 clips via `NFS_ROOT=.av_slice_nurec`),
and the open-loop reference ladder (`ol_<policy>_*`: NC, TTC, EP, HC, EC, MF-PDMS for
replay_human, constant_velocity, constant_turn_rate, reactive_idm, stationary and VaVAM
itself; ~0.5 s per clip per policy). Leave-one-out logistic regression, standardised,
`C = 0.5`, balanced classes; AUC with a 2,000-resample bootstrap; recall of the real
failures when only the top *b* fraction of clips is rolled out. Baselines are single
scores used as a ranking.

## Result at n = 80

Target `offroad_or_collision_at_fault` (41 of 80 positive):

| ranking | AUC | 95% CI | recall @20% | @50% | @80% |
|---|---|---|---|---|---|
| random | 0.500 | | 0.20 | 0.50 | 0.80 |
| **screen, open-loop ladder only** (43 features) | **0.657** | [0.53, 0.78] | 0.27 | **0.66** | 0.90 |
| screen, all features (60) | 0.628 | [0.50, 0.75] | 0.32 | 0.59 | 0.80 |
| screen, VaVAM's own open-loop only (7) | 0.595 | [0.47, 0.72] | 0.29 | 0.56 | 0.83 |
| screen, curation axes only (17) | 0.486 | [0.35, 0.61] | 0.27 | 0.44 | 0.78 |
| open-loop MF-PDMS, worst first | 0.460 | [0.33, 0.60] | 0.22 | 0.39 | 0.83 |
| camera gated low-conf, highest first | 0.480 | [0.35, 0.61] | 0.17 | 0.56 | 0.71 |
| behavioral score, highest first | 0.414 | [0.30, 0.54] | 0.20 | 0.46 | 0.76 |
| conflict load, highest first | 0.371 | [0.25, 0.49] | 0.15 | 0.44 | 0.76 |

Target `offroad_or_collision` (50 of 80 positive): ladder-only screen 0.673 [0.54, 0.80],
recall 0.60 at a 50% budget; all-features 0.660; MF-PDMS alone 0.538.

## Reading it

1. **The open-loop ladder is the screen, not the difficulty score.** The policy's own
   open-loop pass, read against five track-only reference policies, says more about
   whether it will fail closed-loop on a clip than any curation axis does. (Corrected
   2026-10-06: the signal is in the policy's own pass, 11.6 s per clip for VaVAM on a GPU;
   the track-only policies alone are at chance, next section.) The axes alone are
   at chance for this target, and the single strongest curation signal, conflict load,
   is *anti*-correlated (0.37): the densest scenes are where VaVAM is struck from
   behind or truncated early rather than at fault.
2. **The dial works but buys less than the n = 40 run promised.** At a 50% rollout
   budget the ladder screen recovers 66% of the at-fault failures (random: 50%); at 80%
   it recovers 90%. The n = 40 fit reported AUC 0.79–0.80 with the same code; at n = 80
   it is 0.63–0.66 with a CI whose lower edge is near chance. Record it the way
   `ALPASIM.md` records its n = 10 correlation: the small sample flattered the model.
3. **Open-loop MF-PDMS alone remains useless as a screen** (0.46), consistent with the
   per-clip null in `ALPASIM.md`. What carries signal is the *pattern* across the
   ladder — how the oracle, the naive rules and the policy differ on the same clip —
   not any one aggregate.
4. **Cost** (corrected 2026-10-06, next section). Closed-loop VaVAM is ~38 s per scene
   here. The track-only rungs cost ~0.4 s per clip each, but the screen's signal comes from
   VaVAM's own open-loop pass, 11.6 s per clip on one GPU, not the ~0.5 s per rung assumed
   here before. A 50% budget on a 3,176-clip Gold saves 16.8 of 33.5 GPU-hours per policy
   per re-curation if that open-loop pass is run anyway, and 6.5 (19 %) if it is run only to
   triage, at the cost of missing a third of the at-fault failures. The dial exposes that
   trade.

## Where the screen's signal comes from, and what it costs (2026-10-06)

Leave-one-out at n = 80, at-fault target, same model:

| features | n | AUC | 95% CI | cost per clip |
|---|---|---|---|---|
| full ladder (five track-only policies + VaVAM's own open-loop) | 43 | 0.657 | [0.53, 0.78] | 11.6 s GPU + ~2.5 s CPU |
| VaVAM's own open-loop only | 7 | 0.595 | [0.47, 0.72] | 11.6 s GPU |
| **the five track-only policies only** | 36 | **0.476** | [0.34, 0.60] | ~2.5 s CPU |

The track-only rungs carry nothing on their own; what they add is context for the policy's own
open-loop numbers. So the screen is not "six trivial policies at ~3 s": it is the policy under
test run open-loop (11.6 s per clip for VaVAM on one GPU, `.results_nurec_vavam.runmeta.json`)
against a closed-loop rollout at ~38 s. Whether triage pays therefore depends on whether that
open-loop pass is a cost of triage or a product the proving ground delivers anyway. It is the
latter in the design (open-loop scoring is the first rung for every curated clip), and then a
50 % budget saves half of the closed-loop GPU time; counted as a triage cost, it saves 19 %.
The model is in `nvidia_ingestion/STORAGE_SIZING.md` (validation cost section). Making the
screen cheaper (one decision point instead of three, batched inference) is roadmap item P2.1.

## Per-decision features (2026-10-06)

The evaluator scores three decision points per clip and `run_eval.py` used to keep only their
mean. `run_eval.py --per-decision` now writes each one, keyed by the same `episode_id` as the
decision windows in `nvidia_gold.episode` (`.decisions_nurec_<policy>.parquet`; the per-clip
outputs reproduce the old ones exactly). `skip.py features --per-decision` adds, per policy and
metric, the worst decision (`_min`) and the earliest one (`_first`). For the five track-only
policies (written to `.skip_features_decisions.parquet`; `.skip_features.parquet`, whose hash
the frozen screen records, is untouched):

| feature set | n features | LOO AUC | 95% CI |
|---|---|---|---|
| per-clip means (the screen above) | 43 | 0.657 | [0.53, 0.78] |
| + worst decision | 73 | 0.629 | [0.50, 0.75] |
| + earliest decision | 73 | 0.640 | [0.51, 0.77] |
| + both | 103 | 0.624 | [0.49, 0.75] |
| per-decision only (track-only policies) | 60 | 0.453 | [0.32, 0.58] |

No gain: with 80 labels the extra columns add variance, not signal, consistent with the
track-only rungs being uninformative on their own.

VaVAM's own per-decision scores (`.decisions_nurec_vavam.parquet`, a rerun on the RTX 6000; the
`openloop+` rows combine the original A10 run's per-clip means with the rerun's decisions):

| feature set | n features | LOO AUC | 95% CI |
|---|---|---|---|
| per-clip means (the screen) | 43 | 0.657 | [0.53, 0.78] |
| + worst decision, all six policies | 79 | 0.680 | [0.56, 0.80] |
| + earliest decision, all six | 79 | 0.624 | [0.49, 0.76] |
| + both | 115 | 0.647 | [0.52, 0.77] |
| track-only ladder + VaVAM's worst decision | 42 | 0.670 | [0.55, 0.79] |
| track-only ladder + VaVAM's earliest decision | 42 | 0.595 | [0.46, 0.73] |
| VaVAM's worst decision only | 6 | 0.628 | [0.50, 0.75] |
| VaVAM's earliest decision only (a third of the cost) | 6 | 0.547 | [0.42, 0.67] |

The worst of VaVAM's three decisions carries its signal about as well as the mean does; every
difference above sits inside the intervals, and the best of eight variants tried on the same 80
labels is biased upward, so none replaces the frozen screen. Scoring only the earliest decision,
which would cut the open-loop pass to a third, loses most of the signal; which decision is the
informative one varies by clip. Batched inference is the cost lever left (roadmap P2.1).

**VaVAM's open-loop scores do not reproduce across GPUs.** The rerun on the RTX 6000, with the
same seed as the original A10 run, moved MF-PDMS by more than 0.01 on 37 of 137 clips (by up to
0.27; a decision's collision verdict flips on some). The screen survives it: on the rerun's
features its LOO AUC is 0.664 [0.54, 0.78], and refitted on them it ranks the 63 frozen clips with
Spearman 0.976 against the frozen ranking, keeping 31 of the dial's 32.

## Prospective test, prepared (2026-10-06)

Every number above is a leave-one-out estimate on the clips the model was fitted on, and the
n = 40 run already showed how much that flatters. The next batch is set up as a prospective test:

1. **Frozen before any rollout.** `skip.py predict --feature-set openloop` fitted the ladder-only
   screen on the 80 labelled clips and recorded its failure probability for all 63 unlabelled
   NuRec-slice clips (`.skip_frozen_batch4.parquet` + `.json`: frozen 2026-10-06T00:29:58Z,
   sha256 `9ce825a6…`, feature file sha256 `e60e66c7…`; write-protected). Its top half is
   exactly the 32 clips in `.skip_next_rollouts.txt`. Six of the 63 have no open-loop features
   and were ranked on imputed medians (all six fell in the top half), so results are reported on
   all 63 and on the 57 complete ones.
2. **All 63 are rolled out, not only the dial's 32**, since recall cannot be measured on a set
   the dial chose (`.skip_batch4_scenes.txt`, a seeded shuffle; `alpasim/run_batch.sh` in chunks
   of 16, same configuration as batch 3).
3. **Three rollouts per scene.** VaVAM is unseeded in AlpaSim, so they are independent draws:
   the lowest rollout id is the single-draw label comparable to batches 1–3, the majority is the
   robust label, and their disagreement bounds any screen. `skip.py prospective` reports AUC and
   recall per label and subset, the dial as frozen, the share of clips whose rollouts disagree,
   and the AUC a screen knowing each clip's true failure propensity would reach against
   single-rollout labels (beta-binomial fit, with a bootstrap interval).

Status: not run. It needs ~107 GB of scene downloads in chunks and the root volume had 28 GB
free; freeing the 64 GB AlpaSim scene cache is the user's decision.

```bash
N_ROLLOUTS=3 alpasim/run_batch.sh .skip_batch4_scenes.txt vavam-batch4 16
.skip_venv/bin/python skip.py prospective --frozen .skip_frozen_batch4.parquet \
    --closed-loop alpasim/runs/vavam-batch4-c0*/per_clip.parquet --out-prefix .skip_prospective_batch4
```

## What would move it

- More labelled scenes: 63 more open-loop-scored NuRec clips exist (`batch4`), and the
  disk holds room for one more batch after pruning the scene cache.
- ~~Per-decision features instead of per-clip means~~: tried 2026-10-06, no gain (above).
- A cheaper screen: the policy's own open-loop pass carries the signal; fewer decision points
  or batched inference would cut its 11.6 s per clip.
- A second policy: everything above is one policy on one dataset; a screen that holds
  for Alpamayo-1.5 closed-loop (needs an L40S) is the generality claim.

## Reproduce

```bash
.skip_venv/bin/python skip.py features --closed-loop vavam .cl_vavam_perclip.parquet
.skip_venv/bin/python skip.py fit --feature-set openloop
.skip_venv/bin/python skip.py select --feature-set openloop --budget 0.5 \
    --exclude-labelled --scene-ids --out next_rollouts.txt   # the next batch, by the dial
```
Outputs: `.skip_features.parquet`, `.skip*_curve.csv`, `.skip*_fit.json` (coefficients).
