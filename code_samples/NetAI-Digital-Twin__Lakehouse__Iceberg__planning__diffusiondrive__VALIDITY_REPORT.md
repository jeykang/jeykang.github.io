# DiffusionDrive planning signal — validity report (2026-06-23)

**Question**: is the `mode_spread` planning_score a valid *scene-difficulty*
signal? **Answer: no.** It passed the bounded gate (discriminative + additive to
ego_dynamics) but fails validity on multiple independent axes. Documented here
(not in MEDALLION_PROGRESS.md) per the standalone-writeup convention.

## Method
`planning/diffusiondrive/validate.py` on 150 fully-covered clips (one A10), plus
an external-label test against `reasoning/ood_reasoning.parquet` (1,740
human-flagged hard-event clips) over the full 31,812-clip DiffusionDrive run.

## Results

| Test | Result | Reading |
|---|---|---|
| Negative control — blank lidar | Spearman(real, blank-lidar) = **0.967** | ranking ~unchanged with NO lidar |
| Negative control — blank camera | Spearman(real, blank-cam) = **0.973** | ranking ~unchanged with NO camera |
| External labels (OOD hard events) | **AUC 0.373** (OOD mean 0.382 vs 0.484); PEDESTRIAN_DENSITY lowest (0.314) | anti-aligned with human difficulty |
| Convergent — bev_occupancy | Spearman **−0.045** | no relation to scene clutter |
| Convergent — perception (BEVFusion) | Spearman **0.068** | no relation to perceptual difficulty |
| Convergent — ego_dynamics | Spearman **−0.383** | only tracks ego kinematics (negatively) |
| Determinism | max |real−rerun| = **0.38** (between-clip stdev 2.16) | diffusion sampling not seeded → noisy |
| Frame stability | within-clip stdev 0.88 vs between 2.16 (ratio **0.40**) | single-frame noise is large |
| Plan-vs-GT (open-loop L2) | median **3.58 m**, mean 9.04 m | planner partially works; long bad tail |

## Conclusion
The decisive result is the **negative control**: blanking the entire lidar BEV
*and* the camera leaves the per-clip ranking ~97% intact. So the clip-to-clip
variation in `mode_spread` is driven by the **ego status + the model's learned
trajectory priors, not by perception of the scene**. That explains the rest: it
anti-correlates with ego_dynamics, has ~zero correlation with occupancy/
perception, and points the wrong way against human hard-event labels. On top of
that, a large fraction of its variance is non-determinism + single-frame noise.

`mode_spread` therefore measures *trajectory openness given ego state* (high on
fast/open roads, low on constrained interactive ones) — roughly the **opposite**
of safety-critical difficulty.

## Implications
- **`mode_spread` should not be used as a Gold difficulty signal.** It is
  currently wired into production Gold (rung-1 primary) and should be detached.
- **rung-0 (CV open-loop L2) is suspect for the same reason**: it is
  ego-kinematics by construction (gate ρ=0.69 vs ego_dynamics) and was never
  negative-control-tested. Likely also not scene-driven.
- **Methodology lesson**: the bounded gate (discriminative + additive) is
  necessary but NOT sufficient. Negative-control ablation + an external label
  are what distinguish a real signal from an ego-kinematics/noise artifact.
  Perception (BEVFusion) and the metadata sub-scores were never subjected to
  these tests either and warrant the same scrutiny.
- **Path to a valid planning-difficulty signal**: it must be **scene-content
  based** — e.g. rung-2 map-free PDMS (collision / TTC / progress against the
  detected agents), not trajectory spread or open-loop L2 (both ego-dominated).
  That requires agent boxes (BEVFusion detections or obstacle.offline) and is
  future work.
