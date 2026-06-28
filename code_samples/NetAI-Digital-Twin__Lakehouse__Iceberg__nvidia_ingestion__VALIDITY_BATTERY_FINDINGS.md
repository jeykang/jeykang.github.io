# Validity Battery on Existing Gold Sub-scores — Findings (2026-06-23)

Ran the same battery that refuted `mode_spread` against the dimensions already in
production (`nvidia_ingestion/validity_battery.py`, over all 305,724 scored clips
in `iceberg.nvidia_gold.clip_scores`, vs the 1,740 human-flagged hard clips in
`ood_reasoning`; 1,737 overlap). **It turned up a significant problem.**

## External-label OOD AUC (per sub-score)
| dimension | AUC | n_ood | verdict |
|---|---|---|---|
| time_of_day | 0.477 | 1737 | weak / slightly anti |
| season_geography | 0.507 | 1737 | no signal |
| **sensor_coverage** | **0.432** | 1737 | **ANTI-aligned** |
| ego_dynamics | 0.498 | 1737 | no signal |
| perception | 0.564 | 29 | modest (low power, n=29) |
| **conflict** | **0.651** | 200 | **GOOD** (the just-shipped signal) |
| **difficulty_score (composite)** | **0.450** | 1737 | **ANTI-aligned** |

## Convergent / discriminant (Spearman)
- composite ρ: **time_of_day 0.797** (dominates), sensor_coverage 0.598,
  perception 0.461, conflict 0.397, ego_dynamics 0.129, season_geography −0.055.
- `season_geography ~ ego_dynamics = −0.844` — suspiciously strong; likely a
  degeneracy (one near-constant/bimodal) worth a closer look.

## What this means
1. **The production composite does NOT surface the human-hard clips — it is
   slightly anti-aligned (0.450).** It is dominated by `time_of_day` (ρ=0.797):
   Gold is effectively selecting *dark* clips, not *hard-to-drive* clips.
2. **The four metadata dimensions individually carry ~no signal vs human
   difficulty** (AUC 0.43–0.51); `sensor_coverage` is actively anti-aligned (the
   human-hard clips tend to have *good* sensor coverage — they're hard for scene
   reasons, not sensor reasons).
3. **Only `conflict` (0.651) and weakly `perception` (0.564, n=29) track human
   difficulty** — i.e. the agent/scene-interaction signals, not the
   environmental/operational metadata.
4. This **retroactively explains the ego-kinematics planning failures**:
   `ego_dynamics` itself (0.498) doesn't track difficulty, so signals that
   correlated with it (rung-0 CV, open-loop L2) couldn't either.
5. **Coverage gap**: `conflict` exists only for the 31,812 lidar-covered clips
   and `perception` for 3,338 — so for ~274k clips the composite is *metadata
   only*, i.e. anti-aligned.

## Important caveat (what's being measured)
`ood_reasoning` labels driving-**event/agent** hardness (work zones, pedestrian
density, cut-ins…). The metadata dims measure **operational-domain** hardness
(night, winter, sparse sensors) — a *legitimate but different* axis. So "metadata
is anti-aligned" means "it measures a different thing than the human hard-event
labels," not "it is meaningless." The problem is one of **intent + weighting**:
if Gold's purpose is "hardest-to-drive edge cases" (the project's stated goal,
and the whole point of the driving-difficulty effort), then event/agent hardness
is the target and the current metadata-dominated weighting is misaligned.

## Recommended modifications (for decision)
1. **Re-weight** the composite to center the validated signals — `conflict`
   primary, `perception` secondary — and demote the metadata dims to a small
   secondary contribution (or split Gold into an "event-hard" tier and a separate
   "operational-domain" tier).
2. **Close the coverage gap** so the good signal applies broadly: either restrict
   the event-hard tier to lidar-covered clips, or run the BEVFusion *camera*
   detector path for non-lidar clips (label-free, ~0.63).
3. **Drop/repair `sensor_coverage`** (anti-aligned) and investigate the
   `season_geography ~ ego_dynamics` −0.844 degeneracy.
4. **Re-validate** the new composite against `ood_reasoning` (target AUC > 0.6).

## Investigation results (2026-06-23, `battery_investigate.py`)

**(1) Lidar-covered subset (31,737 clips, 200 ood) — the achievable ceiling.**
| signal | AUC |
|---|---|
| composite (current weights) | 0.547 |
| conflict alone | 0.651 |
| perception (n=29) | 0.564 |
| time_of_day | 0.511 |
| ego_dynamics | 0.500 |
| season_geography | 0.450 |
| sensor_coverage | 0.404 |
| **WHATIF (conflict 0.45 / perception 0.25 / metadata small)** | **0.648** |

→ On covered clips the current composite is only 0.547 (the 274k metadata-only
clips drag the full-set figure to 0.450). A conflict-centered re-weighting reaches
0.648 — **essentially conflict-alone (0.651); the metadata dims add nothing
positive (they slightly dilute).**

**(2) `sensor_coverage` is miscalibrated, not just "a different axis".**
- ood mean 0.695 < non mean 0.757 → human-hard clips have *better* sensor
  coverage (they're full-rig clips with events). Anti-alignment confirmed.
- Histogram: values {0.474: 22.7k, 0.526: 87k, 0.579: 50.7k, **1.0: 145.3k**} —
  **47% of all clips are scored maximally sensor-deprived (1.0).** Implausible;
  this is the same naive-sensor-expectation failure mode that the Silver
  `missing_sensors` rewrite already fixed. The dimension should be dropped or
  recalibrated against `feature_presence` (expected-sensor truth).

**(3) `season_geography` and `ego_dynamics` are near-constant (degenerate).**
- season_geography: 5 values, **0.2 covers 281k / 305k clips (92%)**.
- ego_dynamics: **0.5 (neutral default) covers 273k clips (89%)** — i.e. the ego
  aggregate is only actually computed for ~11% of clips; the rest fall to the
  default. And even on the covered subset its AUC is 0.500 (pure chance) — so
  ego-kinematics doesn't track event-hardness even when present (consistent with
  the planning-signal failures).
- The ρ=−0.844 is an **artifact**: where season≠0.2 (the rare 8%), mean ego≈0.17;
  where season=0.2 (the 92%), mean ego≈0.49. Two mostly-constant columns that
  deviate together on a small subset → spurious strong rank correlation. Neither
  carries real discriminative information.

### Net conclusion
Only `time_of_day` has real variance among the metadata dims, and it's a weak
*environmental* signal (~0.48–0.51), not event-hardness. `season_geography` and
`ego_dynamics` are effectively dead (near-constant); `sensor_coverage` is
miscalibrated and anti-aligned. The validated event/agent signal (`conflict`,
≈0.65) is the only thing that works — and a conflict-centered composite matches
conflict-alone. **The real blocker is coverage**: conflict reaches only the
31,737 lidar clips, so a globally-valid Gold needs conflict-like coverage for the
rest (BEVFusion camera-detector path) or an explicit lidar-covered event tier.

### Recommended concrete change
- **Drop** `season_geography` + `ego_dynamics` (degenerate) and **`sensor_coverage`**
  (miscalibrated) from the difficulty composite.
- **Re-weight**: conflict primary, perception secondary, `time_of_day` small
  (keep a light environmental nudge). Re-validate (covered-subset target ~0.65).
- **Close coverage**: run the label-free BEVFusion *camera* conflict path for the
  ~274k non-lidar clips, OR scope the difficulty tier to lidar-covered clips.

## Resolution — APPLIED 2026-06-23
Re-weighted + scoped the difficulty composite (`edge_case_scorer.py`):
- **Dropped** `season_geography`, `ego_dynamics` (degenerate) and `sensor_coverage`
  (miscalibrated) from the blend — weight 0; still computed into `detail` for
  diagnostics. New weights: **conflict 0.60, perception 0.25, time_of_day 0.15**.
- **Scoped** the difficulty tier to sensor-covered clips via a new
  `sensor_covered` score column (has conflict/perception = is in the on-disk 10TB
  sample). Threshold, selection, and stats all filter on it; catalog-only clips
  are excluded from Gold.

**Re-validated**: composite OOD AUC **0.450 → 0.655** on the covered tier (now
slightly above conflict-alone 0.651; metadata no longer drags it). Gold =
**3,174 clips** (top 10% of 31,737 sensor-covered), score std 0.087 → 0.223
(real discrimination). The production difficulty score now tracks human-judged
difficulty instead of opposing it.

## Perceptual axis — is darkness a real, under-served difficulty? (2026-06-23)
`perceptual_axis_analysis.py`. Goal reframed as edge-case mining (keep a clip if
hard on ANY axis), so this matters: are dark clips being wrongly stripped?

**(A) Every OOD cluster is behavioral, and they happen in daylight.** Per-cluster
AUC (positives = cluster; n_on_disk small, so indicative):
| cluster | n | time_of_day | conflict |
|---|---|---|---|
| PEDESTRIAN_DENSITY | 52 | 0.355 | **0.866** |
| SPECIAL_VEHICLE | 34 | 0.320 | 0.654 |
| ROAD_DEBRIS | 1 | — | 0.646 |
| CYCLISTS | 9 | 0.369 | 0.621 |
| EMERGENCY_INCIDENT | 3 | — | 0.588 |
| WORK_ZONES | 88 | 0.366 | 0.561 |
| ANIMALS | 4 | — | 0.541 |
| COMPLEX_INTERSECTION | 5 | — | 0.499 |
| OTHER_LONGTAIL | 4 | — | 0.212 |

→ `conflict` owns the agent clusters (pedestrian 0.866!). `time_of_day` is <0.5
for *every* cluster — the human behavioral-hard clips are predominantly *daytime*.
So ood_reasoning cannot validate darkness, and makes time_of_day look anti.

**(B) Darkness measurably degrades perception in our data** (BEVFusion cohort,
n=3,334):
- spearman(time_of_day, mean_max_conf) = **−0.131**; spearman(time_of_day,
  mean_n_detections) = **−0.174**.
- DAY: conf 0.505, det 11.53. DARK: conf 0.456, det 8.72 → **−10% confidence,
  −24% detections in the dark.** Real perceptual-difficulty axis.
- **But the current scoring under-serves dark clips on every axis**:
  spearman(time_of_day, conflict) = −0.142 (dark = fewer detected agents →
  lower conflict), and spearman(time_of_day, perception_score) = −0.096 (the
  damper reads sparse-detection dark scenes as *easier*, 0.559→0.544 — a perverse
  inversion). And we just down-weighted time_of_day.

### Conclusion
The re-weighting made the score **better for behavioral difficulty** (validated)
but created a **blind spot for perceptual difficulty**: a dark clip where the
perception stack is genuinely struggling is currently rated *easy* on all three
axes and would be **stripped out** — the opposite of the edge-case-mining goal.
Fix: treat difficulty as a **union (max / noisy-OR) of validated axes** —
behavioral (`conflict`) and perceptual (`time_of_day`/adverse-condition,
face-valid + degradation-confirmed) — so a clip survives if hard on *either*.
Weighted-averaging dilutes single-axis-hard clips; union does not. Also repair
the perception damper's perverse dark-scene inversion.

## Final design — noisy-OR union of validated axes (APPLIED 2026-06-23)
The weighted re-weight above was **superseded** by a union, matching the
edge-case-mining goal (keep a clip if hard on EITHER axis; strip only trivial):

`difficulty = 1 − (1 − behavioral) · (1 − perceptual)` where
- **behavioral** = `conflict` (rank-normalized; OOD AUC 0.651, pedestrian 0.866),
- **perceptual** = `max(darkness, 1 − mean_max_conf)`, **rank-normalized over the
  covered population** so it shares conflict's uniform [0,1] scale.

The rank-normalization was essential: with raw darkness (=1.0 for night), the
noisy-OR turned night into a *veto* (Gold 89% dark, validated daytime cases
crowded out). After rank-norm the axes are balanced:

| Gold composition (3,174 clips) | raw darkness | rank-normalized |
|---|---|---|
| dark (time_of_day ≥ 0.7) | 89% | 78% |
| high-conflict (≥ 0.7) | 38% | 70% |
| perceptual-rescued (low conflict) | 70% | 46% |

Validation: conflict (behavioral) AUC 0.651 holds; perceptual axis 0.506 vs OOD
(expected — OOD is daytime-behavioral); composite 0.616 vs OOD (lower than
conflict-alone *by design* — it also keeps perceptually-hard clips OOD doesn't
label); spearman(darkness, composite) **−0.142 → +0.610** (the dark-clip
inversion is fixed). Implemented in `compute_scene_score` (two-pass rank-norm in
`_score_metadata_bulk`); `union_validate.py` is the standing check. Balance is
tunable (axis exponents / weighted noisy-OR) if behavioral should outrank
perceptual.

Regenerate the OOD id list with:
`python3 -c "import pyarrow.parquet as pq; open('nvidia_ingestion/_ood_clips.txt','w').write('\n'.join(pq.read_table('<ood_reasoning.parquet>',columns=['clip_id']).column('clip_id').to_pylist()))"`
