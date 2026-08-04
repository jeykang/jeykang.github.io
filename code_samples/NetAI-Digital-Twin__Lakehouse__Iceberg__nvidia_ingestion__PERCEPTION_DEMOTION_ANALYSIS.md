# Perception Demotion Analysis — BEVFusion vs Metadata-only Gold cohort

**Date**: 2026-06-22
**Scope**: effect of the BEVFusion perception signal on the Gold top-10% cohort.
**Inputs**: metadata-only `clip_scores` baseline (306,152 clips) vs
perception-active `clip_scores` (same run with the 3,338-clip BEVFusion cascade
cohort blended in). Standalone analysis — not part of `MEDALLION_PROGRESS.md`.

> Note: this is the *first genuine* application of perception to Gold scoring.
> The prior §10 v3→v5 "perception integration" predates a loader bug
> (`_load_perception_scores` imported pyarrow, absent in the spark-submit driver
> Python, and silently returned `{}`), so those earlier numbers almost certainly
> reflect metadata-only scoring. Fixed 2026-06-22 (read via Spark).

## Method

Both score sets are ranked; the **top-10% = exact rank-based 10%** (30,615 of
306,152). A clip is **demoted** if it was in the metadata-only top-10% but not
the perception-active top-10% *and* carries a perception score (i.e. it is in the
cascade cohort, so perception actually moved it). "Dominant metadata factor" =
the highest-valued metadata sub-score among `{time_of_day, season_geography,
sensor_coverage, ego_dynamics}`.

## Headline

| metric | value |
|---|---|
| Spearman (metadata vs perception), overall | 0.9996 |
| top-10% Jaccard | 0.985 |
| demoted out of top-10% (cohort) | **228** |
| promoted into top-10% (cohort) | 67 |
| top-10% set size | 30,615 |

Overall ranking barely moves — only 1.1% of clips carry perception — but the
**cohort boundary**, which is exactly where Gold selection happens, is
meaningfully reshaped: 228 clips (0.74% of the top-10%) are swapped out.

## Why these 228 were demoted: metadata over-stated their difficulty

Demoted clips score **higher than the cohort average on every metadata
dimension** — they looked especially hard by metadata — yet have **lower
perception** scores: the visual/LiDAR scene is comparatively empty.

| sub-score | demoted mean | cohort mean |
|---|---|---|
| sensor_coverage | 0.955 | 0.888 |
| season_geography | 0.848 | 0.731 |
| time_of_day | 0.780 | 0.711 |
| ego_dynamics | 0.264 | 0.233 |
| **perception** | **0.468** | **0.551** |

Perception cleanly separates the swap: mean perception **demoted 0.468 vs
promoted 0.691** (cohort 0.551; demoted median 0.474). Low perception is the
mechanism of demotion.

## Dominant metadata factor among the 228 demoted

| dominant factor | count | share |
|---|---|---|
| sensor_coverage | 108 | 47% |
| time_of_day | 63 | 28% |
| season_geography | 57 | 25% |
| ego_dynamics | 0 | 0% |

This matches the §10 ordering (sensor_coverage, time_of_day, season_geography).
These are the "environmental/sensor-profile-says-hard" signals; perception acts
as the corrective when the actual scene content is sparse. `ego_dynamics` never
dominates (its values are low across the cohort).

## Where the demoted clips cluster

- **Season**: winter 187 (82%), fall 32, summer 5, spring 4 — heavily winter,
  consistent with the season_geography boost firing on northern-hemisphere
  winter clips that turn out visually quiet.
- **Country**: United States 78, France 34, Sweden 17, Estonia 17, Spain 15,
  Luxembourg 15.

## Representative demoted clips (lowest perception)

| clip | perception | difficulty | hour | season | country | dominant | sub-scores (tod/season/sensor/ego) |
|---|---|---|---|---|---|---|---|
| `10ce392f…` | 0.247 | 0.526 | 12 | winter | United States | sensor_coverage | 0.20 / 0.90 / 1.00 / 0.55 |
| `fc1b737a…` | 0.247 | 0.562 | 21 | winter | United States | sensor_coverage | 0.80 / 0.90 / 1.00 / 0.28 |
| `ebe1c6cc…` | 0.248 | 0.534 | 20 | winter | United States | season_geography | 0.70 / 0.90 / 0.58 / 0.46 |

Each is a winter US clip that metadata flags as hard (full/degraded sensor
profile, winter boost, some at night) but BEVFusion finds visually sparse —
exactly the "metadata-says-hard-but-actually-empty-road" false positive the
perception signal is meant to damp.

## Interpretation

Perception behaves as a **damper**, not a re-ranker: it leaves the global order
almost untouched (Spearman 0.9996) and instead corrects the *tail* — pulling
metadata-inflated-but-visually-empty clips out of the Gold cohort and admitting
genuinely complex scenes in their place. For a top-N curated subset, that tail
correction is the entire value: it sharpens precisely the membership boundary.

## Caveats

- **Placeholder calibration / domain shift**: BEVFusion runs with nuScenes-style
  intrinsics + identity extrinsics on PhysicalAI, so perception is a *relative*
  signal, not absolute detection accuracy.
- **Coverage ceiling**: only the 3,338-clip cascade cohort (top-30% by metadata
  among the 11,128 fully-camera-covered on-disk clips) carries perception;
  non-cohort clips are unaffected by construction.
- **Magnitude**: 228 swaps in a 30,615-clip top-10% — real but small; the
  effect is concentrated at the boundary, as intended.
