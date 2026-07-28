# Progress Fact Sheet — May 28 → June 23, 2026
NVIDIA PhysicalAI medallion lakehouse (Apache Iceberg + Polaris REST catalog +
MinIO + Spark). All concrete numbers + context for building the meeting slides and
figures. (Activity: 31 commits, all June 19–23.)

---

## 0. Context / framing
- **System**: Bronze → Silver → Gold medallion pipeline over the NVIDIA PhysicalAI
  autonomous-vehicle dataset.
- **End goal (the "why")**: *edge-case mining* — produce a subset of the dataset
  with the trivially-easy training cases stripped out, on the assumption that a
  model able to handle the hard cases handles easy ones trivially. So Gold = the
  hardest clips.
- **Data scale**: full catalog **305,724** clips (Silver). The on-disk working set
  is a deliberate **10 TB random sample** (of the full ~120 TB / ~300k-clip
  dataset) = **31,737** clips with actual sensor data ("sensor-covered"). Sensor-
  based difficulty signals apply to the sample; the difficulty tier is scoped to it.
- **One-line story**: this month we turned Gold difficulty scoring from an
  unvalidated metadata heuristic into a **measured, validated** hard-clip selector —
  and in doing so found the old score was *worse than random* and fixed it.

---

## 1. Headline result — difficulty scoring is now validated
**Before:** Gold difficulty was a weighted average of metadata heuristics
(time-of-day, season, sensor-coverage, ego-dynamics) + perception. Never validated.

**Finding:** tested against **1,740 human-flagged hard clips** (the dataset's
`ood_reasoning` labels), the old composite scored **AUC 0.450 — anti-aligned**
(below the 0.5 chance line; it was effectively selecting *night* clips, not
*hard-to-drive* clips).

**Fix:** a two-axis **noisy-OR union** built for edge-case mining (keep a clip if
hard on *either* axis):
`difficulty = 1 − (1 − behavioral) · (1 − perceptual)`
- **behavioral** = agent-interaction "conflict" from the dataset's own 3D labels
- **perceptual** = darkness / perception-degradation

**Key numbers:**
| Metric | Value |
|---|---|
| Old composite OOD AUC | **0.450** (anti-aligned) |
| Behavioral axis (conflict) OOD AUC | **0.651** |
| Conflict AUC on pedestrian-density clips | **0.866** |
| Composite vs darkness, rank-corr (old → new) | **−0.14 → +0.61** (dark clips now kept, not stripped) |
| Gold output | **3,176 clips** = top 10% of the 31,737-clip sample |
| Gold score spread (std), old → new | 0.087 → ~0.21 (now discriminative) |

---

## 2. Validity methodology (reusable asset)
Every candidate difficulty signal must pass a **validity battery** before reaching
Gold:
1. **Negative control** — blank the scene inputs; the score *must* change (proves
   it's scene-driven, not an artifact). Dataset-agnostic, always available.
2. **External label** — does it rank a held-out human hard-clip set higher? (here:
   `ood_reasoning`, measured by AUC).
3. **Convergent / discriminant** — correlates with independent proxies, not
   reducible to one trivial factor.
4. **Reproducibility** — deterministic + stable across frames.
- Several candidate signals were tested; only those that passed were kept. Tooling:
  `validity_battery.py` (reusable on future datasets).
- AUC reading: 0.5 = random; >0.5 ranks hard clips above easy; <0.5 = backwards.

---

## 3. Validity battery results — per-signal OOD AUC  *(FIGURE 1 data)*
Full scored set: **1,737** OOD-overlap clips vs **305,724** total.

| Signal | OOD AUC | verdict |
|---|---|---|
| sensor_coverage | 0.432 | anti-aligned (miscalibrated) |
| **composite (OLD)** | **0.450** | **anti-aligned** |
| time_of_day | 0.477 | ~chance |
| ego_dynamics | 0.498 | ~chance |
| season_geography | 0.507 | ~chance |
| perception | 0.564 | modest (n=29, low power) |
| **conflict (behavioral)** | **0.651** | **valid** |
| **composite (NEW, covered tier)** | **0.655** | **valid** |

Why the metadata dims failed (diagnostics): composite was dominated by
`time_of_day` (Spearman ρ=0.797 with composite); `season_geography` is 92% one
value and `ego_dynamics` is 89% the neutral default (both near-constant);
`sensor_coverage` scored 47% of clips as maximally sensor-deprived (miscalibrated).

---

## 4. Behavioral axis — agent-conflict from the dataset's 3D labels
- **Source**: downloaded `obstacle.offline` (the dataset's own auto-labeled 3D
  boxes) for our 340 chunks — **16 GB, 0 failures**. Per-track 3D boxes in the ego
  frame with reliable **class** (person / rider / stroller / automobile / …) and
  **track_id**. Also populates the canonical `DynamicObject` table.
- **Signal**: forward-zone, inverse-distance agent load, multi-frame, rank-
  normalized over the 31,812 clips. GPU-free.
- **Validated AUC 0.651 overall**, and it concentrates exactly where it should:

### Per-event-cluster conflict AUC  *(FIGURE 2 data)*
(positives = clips in that human cluster; n = on-disk clips available)
| Event cluster | conflict AUC | n |
|---|---|---|
| PEDESTRIAN_DENSITY | **0.866** | 52 |
| SPECIAL_VEHICLE | 0.654 | 34 |
| ROAD_DEBRIS | 0.646 | 1 |
| CYCLISTS | 0.621 | 9 |
| EMERGENCY_INCIDENT | 0.588 | 3 |
| WORK_ZONES | 0.561 | 88 |
| ANIMALS | 0.541 | 4 |
| COMPLEX_INTERSECTION | 0.499 | 5 |
| OTHER_LONGTAIL | 0.212 | 4 |

(Small n for several clusters → indicative, not precise; the two largest —
pedestrian 0.866 (n=52) and work-zones 0.561 (n=88) — are the most reliable.)

Full OOD label set = 1,740 clips across 9 clusters (all *behavioral*, predominantly
daytime): WORK_ZONES 856, PEDESTRIAN_DENSITY 380, SPECIAL_VEHICLE 260, CYCLISTS 79,
COMPLEX_INTERSECTION 50, OTHER_LONGTAIL 39, EMERGENCY 30, ROAD_DEBRIS 23, ANIMALS 23.

---

## 5. Perceptual axis — darkness is real difficulty we were discarding
The OOD labels are entirely behavioral/daytime, so they can't validate perceptual
difficulty. Validated it directly instead, via BEVFusion detection stats
(n=3,334 clips):

### Darkness measurably degrades perception  *(FIGURE 3 data)*
| | Day (low time-of-day) | Dark (high) | change |
|---|---|---|---|
| mean detection confidence | 0.505 | 0.456 | **−10%** |
| mean detections / frame | 11.53 | 8.72 | **−24%** |

Correlations: spearman(darkness, detection confidence) = **−0.131**;
spearman(darkness, detection count) = **−0.174**.

**The bug this exposed**: the *old* scoring rated dark clips *easier* on every axis
— spearman(darkness, conflict) = −0.142 (fewer agents detected in the dark) and
spearman(darkness, perception_score) = −0.096 (the perception damper read sparse
dark detections as "emptier → easier"). So perceptually-hard dark clips were being
**stripped out** — the opposite of the goal.

### The fix, quantified  *(FIGURE 4 data)*
Rank-correlation of the score with darkness:
| Score | corr with darkness |
|---|---|
| conflict-only (behavioral) | **−0.14** (dark clips ranked low) |
| union composite (final) | **+0.61** (dark clips now ranked high) |

Perceptual axis = `max(darkness, 1 − detection_confidence)`, rank-normalized so it
shares the conflict axis's scale (without this, night saturated the union and
vetoed everything → 89% dark Gold; after normalizing, balanced — see §6).

---

## 6. Gold composition — both axes contribute  *(optional FIGURE 5 data)*
Effect of rank-normalizing the perceptual axis (top-10% Gold = 3,176 clips):
| Gold composition | raw darkness | rank-normalized (final) |
|---|---|---|
| dark (high time-of-day) | 89% | **78%** |
| high-conflict | 38% | **70%** |
| perceptual-rescued (low conflict, kept by darkness) | 70% | **46%** |

Interpretation: ~70% of Gold is behaviorally hard, ~78% perceptually hard (large
both-hard overlap), ~46% are clips kept purely on the perceptual axis that a
conflict-only score would have discarded.

---

## 7. Supporting deliverables
- **Perception operational (BEVFusion)**: mmdet3d BEVFusion running end-to-end on
  lidar + 6 cameras (256×704, N=20 frames/clip); cascade-scored a **3,338-clip**
  cohort; integrated into Gold. Per-clip stats: detection count, max-confidence,
  class diversity, driving-object count. Fixed a loader bug (pyarrow absent in the
  Spark driver) that had silently disabled perception entirely.
- **Persistent catalog**: Polaris moved from an in-memory metastore (lost all
  table registrations on restart) to a **Postgres-backed** metastore. Footprint
  ~47 MB.
- **Silver quality hardening**: `missing_sensors` check rewritten to use
  `feature_presence` (expected-sensor truth) instead of a hardcoded radar floor →
  **99.86% retention** (428 excluded of 306,152). The prior hardcoded floor falsely
  failed ~93% of clips.
- **Repo cleanup**: archived early-setup work (nuScenes/KAIST/Nessie tests,
  benchmarks, paper drafts) into `archive/`; active tree is now just the NVIDIA
  pipeline + live infra. History preserved (git renames).

---

## 8. Next steps
- Grow the on-disk sample to extend coverage of the validated signals.
- Add and validate further hardness axes (e.g. adverse weather) under the same battery.
- Tune the behavioral/perceptual balance against downstream training outcomes.

---

## 9. Figure inventory (recommended, non-filler)
1. **Per-signal validity AUC** (§3) — bar chart, 0.5 chance line; the headline:
   old composite 0.450 (red) vs conflict/new 0.65 (green), metadata dims clustered
   near/below chance.
2. **Per-cluster conflict AUC** (§4) — horizontal bars; pedestrian 0.866 stands out;
   annotate n.
3. **Darkness degrades perception** (§5) — paired Day/Dark bars for confidence
   (−10%) and detections (−24%).
4. **The inversion fix** (§5) — two bars, −0.14 → +0.61 across the zero line:
   "dark clips went from discarded to kept."
5. *(optional)* **Gold composition** (§6) — before/after rank-norm, showing both
   axes contributing.
All five are generated as PNGs in `nvidia_ingestion/figures/`.
