# Rollout triage — which curated clips earn a closed-loop rollout (2026-09-21)

The question from the Sep 4 call: can the validation ground skip what will predictably
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

1. **The cheap reference ladder is the screen, not the difficulty score.** Running six
   trivial policies open-loop on a clip (about 3 s) says more about whether a real
   policy will fail closed-loop on it than any curation axis does. The axes alone are
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
4. **Cost.** Closed-loop VaVAM is ~38 s per scene here; the ladder is ~3 s. A 50%
   budget on a 3,176-clip Gold saves ~17 GPU-hours per policy per re-curation at the
   cost of missing a third of the at-fault failures. The dial exposes exactly that
   trade, which is what was asked for.

## What would move it

- More labelled scenes: 63 more open-loop-scored NuRec clips exist (`batch4`), and the
  disk holds room for one more batch after pruning the scene cache.
- Per-decision features instead of per-clip means (the evaluator has them; `run_eval.py`
  aggregates).
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
