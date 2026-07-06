# Medallion Lakehouse — Progress Report (June 2026)

**Scope**: work on the NVIDIA PhysicalAI medallion pipeline (Bronze/Silver/Gold
curation) — Silver quality hardening, BEVFusion perception scoring, catalog
infrastructure, and an extended effort to build a *validated* driving-difficulty
signal for Gold. Living detail lives in `MEDALLION_PROGRESS.md` (§12 perception,
§13 driving-difficulty); standalone analyses in `planning/diffusiondrive/VALIDITY_REPORT.md`
and `PERCEPTION_DEMOTION_ANALYSIS.md`. This report synthesizes the arc, results,
and lessons.

## Executive summary

- **Silver `missing_sensors` hardened** to be driven by `feature_presence`
  (expected-sensor truth) instead of hardcoded floors → ~99.86% retention; the
  earlier 190 FAILs were ~all non-genuine.
- **BEVFusion perception made operational** (4 stacked contract bugs fixed),
  cascade-scored 3,338 clips, wired into Gold. A loader bug (`pyarrow` absent in
  the spark-submit driver) had silently no-op'd perception historically — fixed
  (read via Spark). Perception acts as a *damper* (§10/§12).
- **Polaris catalog made persistent** (Postgres metastore) — it was in-memory and
  lost all registrations on restart; recovered via `register_table` (zero rebuild).
- **Driving-difficulty: the central effort.** Built a detachable module ladder,
  and — critically — a **validity battery** that *refuted* the learned-planner
  signals and *confirmed* a modest agent-conflict signal. Net: one validated
  driving sub-score (agent-conflict, OOD AUC ~0.65) added to Gold; several
  intuitively-appealing signals shown to be invalid.

## 1. Silver quality hardening
`check_missing_sensors` now consults `aux_sensor_presence` (v26.03
`feature_presence.parquet`): per-clip expected radar count (sum of 19 `radar_*`
flags) and `lidar_top_360fov` flag, with the universe restricted to
`Clip ∩ Camera ∩ feature_presence` (drops version-skew orphans) and on-disk
lidar cross-checked (FAIL→WARN for registration gaps). Verified against the real
parquet (radar present-counts {0,8,9,10}; old `MIN_RADAR_SENSORS=10` floor
falsely failed ~93%). Result: 428 excluded of 306,152 (~99.86% retention).

## 2. Perception (BEVFusion)
Got one-clip multimodal inference working (fixes: 5D image contract via
`inputs['img']` list; image size 256×704; `FORCE_CUDA=1` for the voxel op;
`box_type_3d` metainfo). Refactored into `bevfusion/bevfusion_infer.py`, shared
by smoke test + batch runner. Sampling-adequacy validated → **N=20** frames/clip.
Cascaded the top-30% metadata cohort = **3,338 clips** across both GPUs.
Wired into Gold; **found + fixed** the `_load_perception_scores` pyarrow bug
(perception had never actually applied — read via `spark.read` now). Effect: a
damper that pulls metadata-inflated-but-visually-empty clips out of the cohort
(228 demoted; `PERCEPTION_DEMOTION_ANALYSIS.md`).

## 3. Infrastructure
- **Catalog loss + recovery**: Polaris used an in-memory metastore (no volume) →
  lost all table registrations whenever it stopped (data safe in minio). Bounded
  recovery (re-register `clip_index`/`data_collection`/`aux_*` + rebuild Clip;
  `register_table` for the 5 canonical tables — zero rebuild).
- **Persistence fix**: added a `postgres` service + `polaris-bootstrap` and
  switched Polaris to `relational-jdbc`. Verified: catalog survives restart.
  Footprint ~47 MB. (Memory: `polaris-catalog-ephemeral`.)

## 4. Driving-difficulty signal — the central arc

**Goal**: score how *hard a scene is to drive*, beyond perception jitter.
Designed as a fully detachable module (§13): file-drop parquet → optional Gold
loader + weight renormalization; removal = a few deletions.

| Rung / step | What | Result |
|---|---|---|
| Gate | CV open-loop L2 vs `ego_dynamics`, Spearman | 0.69 — "not pure ego-kinematics", GO (later shown insufficient) |
| Rung-0 | CV open-loop planner module | shipped, then found ego-kinematics-dominated |
| Rung-1a | **SparseDrive** (6-fisheye-cam E2E planner) | integrated; **failed to transfer** (degenerate signal); shelved with evidence |
| (search) | survey for transfer-robust planner | chose **DiffusionDrive** (lidar-BEV + front-cam) |
| Rung-1b | **DiffusionDrive** full run (31,812 clips), `mode_spread` | bounded gate passed; wired into Gold |
| **Validity battery** | negative control + OOD labels + convergent + reproducibility | **REFUTED `mode_spread`** |
| Remediation | detach invalid planning signals | Gold back to metadata + perception |
| Real direction | **agent-conflict** (BEVFusion boxes) | **valid** (neg-control 0.10→0.00; OOD AUC 0.605) |
| Strengthen | class-weight + TTC + multi-frame | multi-frame → 0.633; class/TTC don't help zero-shot |
| obstacle.offline | GT boxes (16 GB) | AUC ~0.65 — **ceiling is the construct, not boxes** |
| **Ship** | agent-conflict sub-score (obstacle.offline, GPU-free) | added to Gold (weight 0.20) |

### The critical finding (validity)
The **bounded gate (discriminative + additive) is necessary but NOT sufficient.**
It passed `mode_spread`, which the validity battery then refuted:
- **Negative control** (decisive): blanking the entire lidar BEV *and* camera
  left the per-clip ranking ~unchanged (Spearman 0.967/0.973) → not scene-driven.
- **External OOD labels**: AUC 0.373 (hard clips scored *lower*).
- mode_spread measured trajectory openness (ego-state-driven), the opposite of
  difficulty. rung-0 CV / open-loop L2 are ego-kinematics by construction.

### The valid signal (agent-conflict)
"How many agents the ego must contend with" (forward-zone, inverse-distance):
- **Passes the negative control** (load 0.10 → 0.003 blanked) — genuinely
  scene-driven.
- **OOD-aligned** (AUC ~0.65), the right direction.
- **Ceiling ~0.65 is intrinsic to the construct**, not box quality: GT boxes
  (obstacle.offline) ≈ zero-shot BEVFusion (0.65 vs 0.63); no metric variant
  (VRU-weight, TTC, any-direction) broke through. Agent presence is *one facet*;
  OOD-hard also spans work-zones/weather/layout/ego-maneuvers, and OOD is a noisy
  proxy.

## 5. Validity methodology (the reusable contribution)
Any difficulty signal added to Gold should pass a standing **battery**:
1. **Negative control** — blank the scene inputs; the score must change. (Always
   available, dataset-agnostic. The single most decisive test.)
2. **External label** — does it rank a held-out hard-set higher? (Needs a
   per-dataset anchor; OOD here.)
3. **Convergent/discriminant** — correlates with independent proxies, not
   reducible to one trivial factor (e.g. ego speed).
4. **Reproducibility** — deterministic + stable across frames.

This caught an invalid-but-gate-passing signal and confirmed a valid one — the
core methodological result of this effort.

## 6. Current Gold state
Gold difficulty blends: **metadata** (time-of-day, season-geography,
sensor-coverage, ego-dynamics) + **perception** (BEVFusion damper, 3,338 clips) +
**agent-conflict** (obstacle.offline, all 31,812 clips, weight 0.20). The invalid
planning signals are detached.

**Re-architected after the battery (2026-06-23, see `VALIDITY_BATTERY_FINDINGS.md`).**
The battery showed the old composite was *anti-aligned* with human-hard labels
(AUC 0.450) — dominated by `time_of_day`, with `season_geography`/`ego_dynamics`
degenerate and `sensor_coverage` miscalibrated. After investigation + the
explicit goal (edge-case mining: keep a clip if hard on *any* axis), the scoring
became a **noisy-OR union of two validated axes** scoped to the on-disk 10TB
sample (~31,737 sensor-covered clips; ~274k catalog-only excluded):
`difficulty = 1 − (1−behavioral)(1−perceptual)`, with **behavioral = conflict**
(OOD AUC 0.651; pedestrian 0.866) and **perceptual = rank-normalized
max(darkness, 1−detection_confidence)** — darkness being a real perceptual axis
(empirically −10% confidence / −24% detections in the dark) that the
behavioral-only OOD labels are blind to. Rank-normalizing the perceptual axis was
essential (raw darkness=1.0 vetoed the union → 89% dark Gold; after, 78% dark /
70% high-conflict, balanced). **Gold = 3,176 clips**; spearman(darkness,
composite) went **−0.142 → +0.610** (perceptually-hard dark clips are now kept,
not stripped — the inversion is fixed).

## 7. Lessons & recommendations
- **Validate, don't assume.** Three intuitive signals failed; the battery is the
  guard. Make it a standing gate for every Gold sub-score — including the
  existing metadata/perception ones, which haven't been battery-tested.
- **Difficulty is multi-facet.** No single signal exceeded ~0.65 vs OOD; treat
  Gold difficulty as a composite of *validated* facets.
- **Label-free generalizes.** For agent-conflict, a zero-shot detector ≈ the
  dataset's own labels — so other datasets ingested later won't need their own
  obstacle labels for this signal (only positions matter). Validation portability
  is the real gap: the negative control ports everywhere; the external-label test
  needs a per-dataset difficulty anchor.
- **Detachable modules paid off.** A failed planner (SparseDrive, DiffusionDrive)
  left zero coupling; swapping signals needed no Gold rework.

## Appendix — key commits (this effort)
Silver hardening `181db47` · perception operational `46cab81` · validation harness
`7bae0d0` · N=20 verdict `9594a22` · §13 roadmap `96925b5` · perception
wire-up/loader-fix `9bbac39` · views recreated + demotion writeup `5bbce5d` ·
Polaris persistence `d0969a3` · gate `5fae8eb` · rung-0 module `abf1113` ·
SparseDrive `0f072d4`/`4002a3d`/`8ce253b` · DiffusionDrive `38e116e`/`ec42bbf`/`ce046d2`
· validity refutation `35a6708` · detach + agent-conflict `c543f36`/`b9d850f` ·
obstacle.offline ceiling `2fb61a2` · agent-conflict sub-score (this change).
