# Medallion Progress — June 2026 (difficulty metric + validation)

*Driving-difficulty roadmap + gate, agent-conflict signal, validity battery, and
the noisy-OR union re-architecture. See the [progress index](../MEDALLION_PROGRESS.md);
synthesis in [../PROGRESS_REPORT_2026-06.md](../PROGRESS_REPORT_2026-06.md).*

---
## 13. Difficulty-metric roadmap: driving-based difficulty (planned, NOT yet implemented)

**End goal (directive, 2026-06-19)**: the difficulty metric should ultimately
judge scenes by *how hard they are to actually drive* — running an autonomous
driving policy over the scene and measuring its struggle — rather than the
current proxy of perception self-consistency (BEVFusion detection jitter, §10/§12).
Closed-loop is out of scope: it needs a reactive simulator that re-renders
sensors for off-trajectory ego poses (NuRec-style neural reconstruction — in
this dataset's DNA, but hours/clip → infeasible at 33K scale). So the target is
**open-loop / semi-open-loop**.

This is a *later* work item. It is documented here so the design is settled
before implementation, and — per directive — **it must be built as a fully
detachable module**: if it never gets off the ground, deleting it must leave the
rest of the pipeline untouched.

### Modularity contract (the hard requirement)

Mirror the existing BEVFusion perception scorer's loose coupling exactly:

- **Self-contained directory** `planning/` (sibling to `bevfusion/`): own
  Dockerfile, own pinned deps, own `planning-runner` compose service with
  `profiles: [manual]` so it never auto-starts. Nothing in the core Spark
  pipeline (`canonical_bronze.py`, `quality_checks.py`, `pipeline.py`) imports it.
- **File-drop interface only.** The runner writes per-clip parquet to
  `<NFS>/.planning/planning_shard_NN.parquet` with schema `{clip_id,
  planning_score ∈ [0,1], + detail columns}` — exactly analogous to the
  perception scorer's `<NFS>/.perception/`. No new canonical Bronze columns, no
  schema migration, no catalog dependency.
- **One optional hook in Gold.** `edge_case_scorer` gains a
  `_load_planning_scores()` mirroring `_load_perception_scores()`, and
  `compute_scene_score` treats a missing `planning_score` as a dropped
  dimension and renormalizes the remaining weights (the renormalization
  machinery from §10 already does this). If `.planning/` is absent, Gold scores
  exactly as it does today.
- **Reuse, don't extend.** Inputs come from existing aux tables (`aux_egomotion`
  for GT ego trajectory + ego state, calibration) and the boxes the BEVFusion
  runner already emits — no new upstream data contracts.
- **Removal cost = three deletions**: the `planning/` dir, the compose service
  block, and the `_load_planning_scores` hook + its weight entry in
  `_SCENE_WEIGHTS`. Nothing else references it.

### Method decision: open-loop planner + NAVSIM-style semi-open-loop scoring

**Caveat that drives the whole design**: naive open-loop L2 (planned vs GT
trajectory) is known to be gameable by ego-kinematic extrapolation
("ego-status-is-all-you-need" / AD-MLP / BEV-Planner). Our cheap `ego_dynamics`
sub-score already captures that, so an L2-based planner difficulty risks
recomputing it for days of GPU. The metric must reward signals that *require
scene understanding*.

**NAVSIM**: evaluated and rejected as a *direct* dependency. NAVSIM's PDM Score
(bicycle-model unroll of the planned trajectory + collision / drivable-area /
TTC / progress / comfort scoring, non-reactive log-replayed agents, no sensor
re-render) is exactly the right *methodology* — it fixes the L2-gaming problem.
But NAVSIM-the-software is map/annotation-centric and assumes nuPlan/OpenScene
format: it needs an HD map (drivable area / lanes — PhysicalAI has **none**),
tracked agents with futures, and traffic-light status. Running it would require
converting PhysicalAI into pseudo-nuPlan with a *synthesized* map, and the
map-dependent metrics (the ones that make PDMS valuable) would score against
fabricated geometry. **Decision: do NOT run the NAVSIM devkit; reimplement its
map-free PDMS subset** (no-collision / TTC / ego-progress / comfort), which
needs only {planned trajectory, ego kinematics, agent boxes — already produced
by the BEVFusion runner}.

**Difficulty = inverse of planner success**, not error-vs-GT: a scene is hard
when a competent planner gets low PDMS (collides / can't progress safely). Best
signal is the **PDMS gap between a trivial constant-velocity/yaw-rate planner
and a learned planner** (large gap or uniformly low PDMS = genuinely hard); the
trivial planner fails the collision/progress checks precisely where the scene
is non-trivial, which is what breaks the ego-kinematics confound.

### Implementation ladder (each rung independently shippable)

0. **Constant-velocity open-loop L2 — DONE (2026-06-22), the detachable
   `planning/` module is live.** `planning/runner.py` scores every on-disk clip
   (32,651) by the L2 @3s of a trivial CV planner vs the GT ego trajectory
   (`labels/egomotion`); CPU + NFS only. Writes `<NFS>/.planning/*.parquet`
   (`clip_id, planning_score, cv_l2_3s_m, …`). Wired into Gold via
   `_load_planning_scores` + a renormalized `planning` dimension
   (`_SCENE_WEIGHTS["planning"]=0.15`); end-to-end run loaded both perception
   (3,338) and planning (32,651) and rebuilt Gold (34,427 clips, score range
   widened to [0.158, 0.839]). Fully detachable (planning/README.md: removal =
   3 deletions; score flows through `difficulty_score`+`detail`, no schema
   change). This is the modular plumbing + a gate-proven signal, with zero GPU.
1. **Learned-planner trajectory + uncertainty** (IN PROGRESS — env milestone
   done 2026-06-22). Planner chosen: **SparseDrive** (efficient E2E, multimodal
   planning output for the uncertainty signal; checkpoint on GitHub releases).
   Container `planning/sparsedrive/` (`netai/sparsedrive-runner`, separate older
   stack: py3.9/torch1.13+cu116/mmcv-full 1.7.1/mmdet 2.28.2 + flash-attn +
   custom deformable-agg ops) **builds and the model loads + checkpoint applies
   on GPU** (86.1M params). The nuScenes-coupling blocker (config wants
   `data/kmeans/*.npy` anchors generated from nuScenes) was bypassed by dumping
   those anchors straight from the checkpoint buffers (`extract_anchors.py`,
   baked into the image) — no nuScenes needed.

   *Inference adapter — WORKS (2026-06-22), but signal not yet usable.*
   `planning/sparsedrive/test_one_clip.py` builds the model's input dict directly
   (img, timestamp, projection_mat, image_wh, 10-dim ego_status from
   aux_egomotion, 3-way nav cmd from GT future, img_metas[T_global]), bypassing
   the nuScenes `.pkl` pipeline. One-clip run produces the full output on
   PhysicalAI: detections (300 boxes), agent forecasts (`trajs_3d` 300×6×12×2),
   and the ego plan (`planning_score` 3×6, `planning` 3×6×6×2, `final_planning`
   6×2). **Must run on the A10 (Ampere)** — flash-attn rejects the RTX 6000
   (Turing). Anchors dumped from checkpoint; calibration is a placeholder
   nuScenes-style ring.

   **Open blocker (signal quality)**: under placeholder calibration the planning
   signal is degenerate — mode entropy ≈ 1.000 (uniform) on every clip tested,
   and `final_planning` undershoots (~30 m over the horizon vs ~108 m implied by
   the ego speed). The planner runs but isn't grounded on PhysicalAI. Likely
   needs **real PhysicalAI calibration** — non-trivial because PhysicalAI
   cameras are **fisheye/polynomial** (`camera_intrinsics` has width/height/cx/cy
   /poly_coefs) while SparseDrive assumes pinhole `projection_mat`; a fisheye→
   pinhole approximation is required, and even then domain shift (no finetuning
   data) may limit transfer.

   **Bounded real-calib test — DONE (2026-06-22): NEGATIVE → SparseDrive
   shelved.** Built pinhole-approx projection from real PhysicalAI calibration
   (per-clip extrinsics + fisheye `fw_poly_1` focal, central ~90° of the 120°
   fisheye; `planning/sparsedrive/calib_test.py`) and tested 15 clips, placeholder
   vs real calib. Real calibration did **not** restore the signal: planning-mode
   entropy stays flat (real-calib mean 0.999, **stdev 0.0009** — same as
   placeholder), and `final_planning` still undershoots (endpoints 6–29 m, mean
   19 m, vs ~100 m expected at highway speed). Both the mode-confidence and the
   trajectory heads fail to transfer to PhysicalAI — domain shift
   (nuScenes→fisheye, no finetuning data) is the bottleneck, not calibration.
   **Verdict: keep rung-0 (the gate-proven CV signal, ρ=0.69, discriminative,
   full-coverage, already in Gold) as the driving-difficulty score. SparseDrive
   stays a documented, detachable experiment** — the container builds and
   one-clip inference runs (`planning/sparsedrive/`), but it is NOT wired into
   Gold. Reviving it would need PhysicalAI finetuning (no planning GT) or a
   transfer-robust planner. Removal cost: `rm -rf planning/sparsedrive/` —
   nothing imports it.

1b. **DiffusionDrive (rung-1, attempt 2) — GATE PASSED (2026-06-22).** Chosen
   after a web survey for a transfer-robust planner: NAVSIM/Transfuser lineage,
   input = **lidar BEV raster + stitched forward camera** (no projection
   matrices), so it sidesteps the camera-appearance + fisheye issues that killed
   SparseDrive. Container `planning/diffusiondrive/` (`netai/diffusiondrive-runner`):
   torch 2.0.1, **no mmcv/mmdet/flash-attn/custom ops** → runs on **both GPUs**.
   Dep pins that mattered: `diffusers==0.27.2` + `huggingface_hub==0.23.4`
   (newer diffusers needs torch.xpu / drops cached_download); numpy re-pinned
   1.23.4. Checkpoint: `diffusiondrive_navsim_88p1_PDMS` (HF) — the lidar-BEV
   NAVSIM variant, NOT the 6-cam `nusc` build. 20-mode trajectory anchor dumped
   from the checkpoint (`extract` inline).

   Adapter `test_one_clip.py` builds the 3 features from PhysicalAI directly
   (forward-cam stitch 1024×256; lidar BEV 256×256 from Draco points;
   status = cmd4+vel2+accel2) — **no calibration needed**. One-clip inference
   runs and returns a sane plan + detections. The 20-mode distribution
   (poses_reg/poses_cls) is captured by hooking `DiffMotionPlanningRefinementModule`.

   **Bounded gate (`gate_test.py`, 25 clips): PASS.** mode_spread mean 9.53,
   **stdev 2.45**, range [6.8,13.9] (discriminative — vs SparseDrive's flat);
   final-plan endpoint mean 31.3 m, stdev 19.3, range [8.1,59.3] (well-grounded,
   variable — vs SparseDrive's undershoot); **Spearman(mode_spread,
   ego_dynamics) = −0.513** → ~75% independent of the rung-0 kinematic signal,
   so the learned planner adds genuine signal. (Negative sign: the planner
   collapses modes under decisive ego motion, spreads under scene ambiguity.)

   **Status: PRODUCTION rung-1 signal — full run + wire-in DONE (2026-06-22).**
   Ran DiffusionDrive over all 31,812 lidar-covered clips (`runner.py`, sharded
   one container per GPU, ~16 h, ~0.27 clips/s/shard, resumable). Signal =
   `planning_score = clip((mode_spread − 5)/10, 0, 1)` (mean 0.484, stdev 0.229,
   full range [0.07,1.00]). `finalize.sh` installed the shards into
   `<NFS>/.planning/`; the loader merge makes DiffusionDrive primary with the
   rung-0 CV score as the per-clip fallback (~839 clips DD couldn't score). Gold
   re-scored: 34,700 Gold clips; planning loaded for 32,651 (DD ∪ rung-0).

   Comparison rung-0-plan → DiffusionDrive-plan (305,724 clips): Spearman 0.981,
   **top-10% Jaccard 0.958 (655 swapped of 30,572)**, cohort top-10% 1,047 →
   1,320 (**+273 net promoted**), score delta mean **+0.027** (range
   [−0.14,+0.15]). So unlike perception (a damper), DiffusionDrive planning is a
   **net promoter** — it raises difficulty for high-`mode_spread` (ambiguous)
   scenes, and because `mode_spread` anti-correlates with `ego_dynamics` it
   elevates calm-but-ambiguous scenes the metadata/ego score under-rated: a
   complementary difficulty axis. Gold difficulty now blends metadata +
   perception (BEVFusion) + planning (DiffusionDrive). Detachable as ever:
   `rm -rf planning/diffusiondrive/`.

   **VALIDITY FAILURE — `mode_spread` is NOT a valid difficulty signal
   (2026-06-23).** A validity battery (`planning/diffusiondrive/validate.py` +
   the `ood_reasoning` external labels; full report:
   [`planning/diffusiondrive/VALIDITY_REPORT.md`](../planning/diffusiondrive/VALIDITY_REPORT.md))
   refuted it. Decisive **negative control**: blanking the entire lidar BEV
   *and* the camera leaves the per-clip ranking ~unchanged (Spearman 0.967 /
   0.973) → the score is driven by ego status + model priors, NOT the scene.
   It also anti-aligns with human hard-event labels (AUC 0.373; pedestrian-
   density lowest), has ~zero correlation with occupancy (−0.045) / perception
   (0.068), and is noisy (nondeterministic Δ0.38 + frame-unstable). The bounded
   gate (discriminative + additive) was necessary but **insufficient**.
   **Action: detach the DiffusionDrive (and likely the rung-0 CV) planning
   signal from Gold** — both are ego-kinematics-dominated, not scene-driven. A
   valid planning-difficulty signal must be scene-content-based (rung-2 PDMS:
   collision/TTC vs detected agents), future work. Perception + metadata
   sub-scores were never negative-control-tested either and warrant the same
   scrutiny.

   **Remediation + the real-scorer direction (2026-06-23).** Detached BOTH
   invalid planning signals (DiffusionDrive + rung-0 CV) from `.planning/` (moved
   to `.planning_detached_invalid/`) and re-scored Gold → back to metadata +
   perception (35,376 Gold clips, "No planning scores found"). Then prototyped a
   **valid** driving-difficulty signal: **agent-conflict** =
   score-weighted inverse-distance sum of BEVFusion-detected agents in the
   forward zone (x∈[0,40] m, |y|<8 m), ego/lidar frame
   (`bevfusion/agent_conflict_test.py`). Validated on 255 clips (134 OOD / 121
   non): **negative control passes** (real load 0.104 → blank-scene 0.003 ⇒
   genuinely scene-driven) and it **aligns with OOD** (AUC 0.605, OOD mean 0.129
   > non 0.093) — the right direction, opposite mode_spread's 0.373. BEVFusion
   zero-shot detections are usable (median 7 agents/clip, 8% zero). Strength is
   **modest** (AUC ~2 SE > 0.5): improvable via TTC/velocity + agent-class
   weighting (pedestrians) + multi-frame aggregation, and/or trustworthy boxes
   from `obstacle.offline`. This is the first signal to pass the validity battery
   — the basis for a real scene-grounded driving-difficulty score.

   **Strengthening attempt (`bevfusion/agent_conflict_test2.py`, 452 clips,
   202 OOD): zero-shot ceiling ~0.63.** Multi-frame aggregation (0.3/0.5/0.7)
   lifted the simple position+confidence metric to **AUC 0.633** (+0.028).
   But adding **class-weighting (VRU) + TTC** *lowered* it (AUC 0.607) — an
   informative negative: BEVFusion's zero-shot **class and velocity** outputs
   don't transfer to PhysicalAI, so weighting by them adds noise; only box
   **positions + confidence** are reliable zero-shot. Negative control still
   strongly passes (rich real 0.679 → blank 0.011). Conclusion: the
   agent-conflict construct is valid but **plateaus at ~0.63 on zero-shot
   boxes**. The levers that should help (VRU-weighting, TTC) need reliable
   class/velocity/tracks → **`obstacle.offline`** (the dataset's own labels) is
   the justified next investment to push past the ceiling; otherwise ship the
   simple+multi-frame metric (0.633, validated) as the production driving signal.

   **obstacle.offline result — the ceiling is the CONSTRUCT, not box quality
   (2026-06-23).** Downloaded the dataset's own boxes for our 340 chunks (16 GB,
   0 fail; per-track 3D boxes in rig frame with reliable `label_class` incl.
   person/rider/stroller + `track_id` for velocity). Built a GPU-free conflict
   scorer (`planning/obstacle_conflict_test.py`; note: obstacle.offline is
   per-track time series, NOT frame-synced — reconstruct a scene at time T by
   each track's nearest detection within 0.1 s). Validated vs OOD (451 clips):
   AUC by metric — simple-forward-proximity **0.647**, rich VRU+TTC 0.633,
   any-direction count 0.646, VRU-near 0.612, nearest-VRU 0.621. **Every variant
   sits at ~0.61–0.65**, and GT boxes barely beat zero-shot BEVFusion (0.633).
   OOD vs non means separate 2–4× (e.g. VRU-near 0.527 vs 0.137) but rank-AUC
   stays ~0.65 → agent-conflict is a *real but modest* facet of difficulty; the
   ~0.65 alignment ceiling is intrinsic to the construct (OOD-hard also includes
   work-zones/weather/layout/ego-maneuver that agent density can't rank) and to
   the noisy OOD proxy — not to detection quality.
   **Implications**: (a) the expensive obstacle.offline download was NOT needed
   for this signal — the cheap label-free BEVFusion path was already at the
   ceiling (a useful generalization result: a detector suffices, no per-dataset
   labels required); (b) a single agent-conflict signal caps ~0.65 — going higher
   needs a *composite* difficulty (agent-conflict + other validated facets), not
   better boxes. Recommended: ship agent-conflict (cheap BEVFusion boxes) as one
   validated, modest sub-score; treat difficulty as a multi-facet composite.

   **SHIPPED 2026-06-23.** agent-conflict added as a production Gold sub-score
   (`planning/conflict_runner.py`, GPU-free, obstacle.offline source → simple
   forward-zone load, rank-normalized over 31,812 clips → `.conflict/`;
   `edge_case_scorer._load_conflict_scores` + `conflict` dim, weight 0.20). Gold
   re-scored = **35,055 clips** (vs 35,376 metadata+perception-only; ~321 shifted
   as the new agent axis reshuffles the cohort). Detachable per the usual contract
   (delete runner + loader hook + weight). Full synthesis:
   `PROGRESS_REPORT_2026-06.md`.
   **→ SUPERSEDED by §14 (2026-06-23):** the whole composite was then re-validated
   and re-architected (the metadata-weighted blend was anti-aligned with human-hard
   labels); Gold is now a noisy-OR union of validated axes = **3,176 clips**.
2. **+ map-free PDMS** (the NAVSIM-style win) — bicycle unroll + collision/TTC/
   progress/comfort using BEVFusion boxes. Cheap geometry; adds meaning over (1)
   almost for free once trajectory+boxes exist.
3. **+ map-dependent metrics** — only if a pseudo-map is estimated (online
   map-prediction model or lidar ground-seg). Noisy; likely skip for a relative
   difficulty signal.

### Prerequisites / data gaps

- **No HD map** (canonical `HDMap` empty) — blocks rung 3; rungs 1–2 don't need it.
- **`obstacle.offline`** (~50–100 GB, not downloaded) — gives agent boxes/tracks
  for collision metrics; the BEVFusion runner's boxes are the no-download
  fallback.
- **Compute**: planners are heavier than BEVFusion and the full BEVFusion Gold
  pass is already multi-day, so run planning **only on perception-selected
  survivors** (cascade: metadata → perception → planning on the top cohort).

### Cheap de-risking experiment — DONE (2026-06-22): GATE PASSED

Computed a trivial constant-velocity open-loop planner's L2 @3s (positions from
`labels/egomotion`, finite-diff velocity, frame-agnostic) on 400 random clips
and correlated (Spearman) against the `ego_dynamics` sub-score (replicated
exactly: `0.6·min(1,accel_std/3) + 0.4·min(1,curv_std/0.1)`). No GPU, no catalog.

**Result: ρ = 0.69** (0.688 single-step velocity, 0.699 central-difference
smoothed — robust, so the independent variance is real signal, not
velocity-estimation noise). CV L2@3s median ≈ 3.1 m.

Verdict: **well below the ρ≳0.9 redundancy threshold → GO.** Refutes the prior
worry that open-loop L2 is just ego-kinematics: only ~half its rank variance is
explained by `ego_dynamics` (the rest is real, likely because `ego_dynamics` is
clipped/saturated and CV-L2 integrates sustained path deviation over the
horizon). So even the trivial planner adds independent difficulty signal; a
reactive learned planner should add more. Plain open-loop L2 is therefore a
legitimate component (residualized against `ego_dynamics`), alongside the
uncertainty/collision (PDMS) signals.

---

## 14. Validity battery + difficulty re-architecture (2026-06-23)

After shipping agent-conflict (§13), ran the same validity battery against the
**existing** Gold sub-scores — it refuted the production composite, which was then
re-architected. Full detail: `VALIDITY_BATTERY_FINDINGS.md`; synthesis:
`PROGRESS_REPORT_2026-06.md`; meeting writeup + figures:
`MEETING_FACTSHEET_2026-06.md`, `figures/`.

### Finding: the old composite was anti-aligned with human difficulty
Tested vs `ood_reasoning` (1,740 human-flagged hard clips). Old composite OOD
**AUC = 0.450** (below the 0.5 chance line — it was selecting *night* clips, not
*hard-to-drive* ones; dominated by `time_of_day`, ρ=0.797). The metadata dims are
degenerate/miscalibrated: `season_geography` 92% one value, `ego_dynamics` 89%
default (AUC ~0.50), `sensor_coverage` 47% maxed + anti-aligned (0.432). Only
`conflict` (**0.651**; pedestrian-density **0.866**) and weakly `perception`
(0.564, n=29) track human difficulty. Diagnostics: `battery_investigate.py`.

### Perceptual axis: darkness is real difficulty the old score discarded
All 9 OOD clusters are *behavioral + daytime*, so they can't validate perceptual
hardness. Validated it directly (`perceptual_axis_analysis.py`, n=3,334): dark
clips lose **−10% detection confidence, −24% detections**. Yet the old scoring
rated dark clips *easier* on every axis (conflict −0.142, perception damper −0.096
— a perverse inversion), so perceptually-hard clips were being stripped.

### Re-architecture (final): noisy-OR union of validated axes
Goal is edge-case mining (keep a clip if hard on **any** axis), so the composite
became a **union**, not a weighted average (`compute_scene_score`):
`difficulty = 1 − (1 − behavioral)·(1 − perceptual)`
- **behavioral** = `conflict` (rank-normalized; OOD AUC 0.651)
- **perceptual** = `max(darkness, 1 − mean_max_conf)`, **rank-normalized over the
  covered population** (two-pass in `_score_metadata_bulk`) so it shares conflict's
  scale — without this, night saturated the union (89% dark Gold).
- Dropped `season_geography`/`ego_dynamics`/`sensor_coverage` from the blend
  (kept in `detail` for diagnostics); old `_SCENE_WEIGHTS` deprecated.
- Scoped the tier to the **on-disk 10 TB sample** via a new `sensor_covered`
  column (conflict/perception present); the ~274k catalog-only clips (not in the
  sample) are excluded from Gold.

### Result
- Dark-clip inversion fixed: rank-corr(darkness, composite) **−0.14 → +0.61**.
- Gold = **3,176 clips** (top 10% of 31,737 sensor-covered); score std 0.087 → ~0.21.
- Composition balanced: **78% dark, 70% high-conflict** (large both-hard overlap),
  46% kept purely on the perceptual axis. Behavioral axis still validates (0.651);
  union vs OOD = 0.616 (lower than conflict-alone *by design* — it also keeps
  perceptual-hard clips the daytime OOD labels don't cover).
- Standing validation gate: `validity_battery.py` (+ `union_validate.py`) — reusable
  on any future dataset/signal.

### Repo cleanup (2026-06-23)
Archived early-setup work (nuScenes/KAIST/Nessie tests, benchmarks, paper, old
reports) into `archive/` (git renames, history preserved). Kept the live
`kaist_ingestion/config.py` (shared AD-lakehouse schema config the NVIDIA pipeline
imports) + BI/infra (`superset`, `trino`). Active tree is now just
`nvidia_ingestion`, `planning`, `bevfusion`.

## 15. Alpamayo-1.5 VLM difficulty — SHELVED (2026-06-26)

Tested using a reasoning VLM (`nvidia/Alpamayo-1.5-10B`) to *judge* difficulty
directly (sidesteps the planner-transfer failure). Output-format problem solved via
logit expected-value (deterministic, 100% parseable). But the signal underperforms:
cold-VQA OOD AUC 0.437 → reasoned-VQA 0.565 → native CoC rollout **0.604**, all
below the production `conflict` (0.651); CoC negative-control is only +0.03 (the
model hallucinates a plausible chain even on blanked frames — weakly grounded);
`minADE` planning-error is anti-aligned (0.350). Infeasible on 24 GB too: fits only
at a degraded 1-frame/64-token config, ~66 s/clip → ~610 h for 33k. **Shelved**;
production stays the conflict+darkness union. Revisit on ≥40 GB GPU (full-config
CoC) — reasoning quality was good but config-constrained. Detail + code:
[`../../planning/alpamayo/FINDINGS.md`](../../planning/alpamayo/FINDINGS.md).

---

## 16. NAVSIM/PDMS trajectory-feasibility — tested, not an upgrade (2026-06-26)

Rule-based open-loop sim (rung-2, §13): roll out a small ego trajectory vocabulary
against the `obstacle.offline` agent tracks, difficulty = fraction of maneuvers that
collide/near-miss (`planning/pdms_test.py`). **First open-loop-sim approach that is
actually valid** — scene-grounded, right direction (no transfer/openness confound).
But OOD AUC 0.598 (VRU-weighted 0.603) does **not** beat `conflict` (0.651) and is
highly redundant with it (Spearman +0.74). Conclusion: **agent-interaction difficulty
caps ~0.65 against the OOD labels regardless of method** (static proximity vs full
feasibility sim) — a construct ceiling, not a metric gap. `conflict` stays the
production agent-interaction signal; PDMS kept as a validated-equivalent alternative.

---

## 17. Camera-only consumer → camera-only perceptual axis (2026-06-27/28)

The difficulty consumer's final product is **camera-only** (lidar-assisted today,
camera-only for scale). This is decisive for the perceptual axis: a night camera
transform drops **camera-only** YOLO confidence **−0.43** (agents vanish) but the
lidar-**fused** BEVFusion confidence **≈0** — clean lidar masks the camera
degradation (suppressing lidar just collapses BEVFusion; it is intrinsically fused).
So the old fused `low_conf` axis is *blind* to the difficulty the final product faces.

Built a camera-only perceptual axis — `planning/camera_perception_runner.py` (YOLO
front-cam over 33,767 clips → `.camera_perception/`; 6 GPU shards). vs fused: Spearman
0.739 on real clips, but fused only ever covered 3,338 clips (camera = **10×**
coverage). Empty-scene confound (25% zero-detection clips are *empty*, not hard) fixed
by **agent-gating** (`planning/write_camera_gated.py`: count camera difficulty only
where `obstacle.offline` has agents) → camera-hard 27.6%→**11.1%**, −5,218 false
positives, OOD 0.43→0.58. Re-pointed the `edge_case_scorer` perceptual leg to the gated
camera axis; full A/B re-score: **Gold reshuffle 11.6%** (369/3,173 swapped, Jaccard
0.79) plus 10× perceptual coverage.

## 18. Dual Gold: camera-only + lidar-fused difficulty (2026-06-29)

Rather than camera *replacing* fused, `clip_scores` now emits **both**
`difficulty_camera` (the consumer's endgame) and `difficulty_lidar` (general-purpose
lidar-fused). Each = behavioral noisy-OR its modality's rank-normed perceptual axis;
behavioral shared. `run_gold_score.py --gold-axis camera|lidar` picks which one
materializes the Gold views (default camera); the other stays derivable from
`clip_scores`. Full re-score (top 10% of 31,737): camera Gold **3,174** / lidar Gold
**3,176**, overlap 2,830, **~374 unique to each tier** (Jaccard 0.79) — neither is
redundant. (The dual write needs `spark-submit --driver-memory 12g`.)

## 19. Cosmos-Transfer augmentation — infra + end-to-end validation (2026-06-27/28)

Goal: generate hard variants of easy clips (day→night/rain/fog) with labels preserved
— to *manufacture* the camera-adverse cases the real data lacks (only 2.2% of clips
are camera-hard). Cosmos-Transfer1-7B is **cluster-only** (~80 GB VRAM ≫ local 24 GB;
no hosted API — download-only).

Pipeline stood up: built a Cosmos-Transfer1 SIF **locally** (`cosmos_transfer1.def`,
apptainer fakeroot — the A100 login node is locked down, can't compile/pull images) →
SFTP to `/scratch` (`cosmos_augmentation/cluster.py`) → weights (113 GB + gated 7B;
`cluster_download_weights.sh`) → patched out the gated guardrail
(`patch_transfer_guardrail.py`, sidesteps Meta Llama-Guard). **End-to-end validated**:
depth-controlled day→night on an easy clip (4× A100-40 GB, one node; `cosmos_infer.sbatch`)
→ photorealistic night render, **geometry + agents preserved** (depth control →
`obstacle.offline` labels transfer), **harder for camera-only perception** (YOLO −0.22
conf, −1.67 detections). ~7 min/clip.

## 20. Augmentation productionization: recipe + safety gate (2026-06-28/29)

- **Recipe** (`cosmos_refine.sbatch` control×condition matrix): **depth control ≫ edge**
  (edge retains daytime → weakest); night kills confidence, fog/rain make agents vanish.
  Use depth + a mix of night/rain/fog.
- **Label-validity bug caught by the first batch**: content-mentioning prompts made
  Cosmos **hallucinate agents** on sparse scenes (empty road → invented taillights) →
  invalid labels. Fixed with **condition-only prompts** (lighting/weather only, never
  vehicles): empty-clip added-detections **+0.8→+0.10**.
- **Safety features** (`cosmos_augmentation/safety.py`): `find_agent_window` (augment
  the 121-frame window that actually has agents — the naive first-121 trim is often
  empty) + `hallucination_gate` (reject aug clips that *gain* detections). Full safe
  pipeline (`select_easy_clips` → `stage_batch` → `cosmos_batch.sbatch` →
  `apply_hallucination_gate`) on a 9-clip batch: **KEEP 7/9** (label-valid *and* harder,
  ~−1.4 detections/clip), gate auto-filtered 1 hallucination + 1 no-op (~78% keep-rate).
  Production-ready; scale by raising `N`. Full detail: `cosmos_augmentation/FINDINGS.md`.
  Cluster use: **one node at a time** (pod09; pod17 reserved for another project).

---

## Figures (2026-06)

Generated from this month's measurements via [`make_figures.py`](../make_figures.py)
(data in [`../MEETING_FACTSHEET_2026-06.md`](../MEETING_FACTSHEET_2026-06.md)).

### Fig 1 — Validity battery: which signals track human-judged hard clips
Old composite **0.450** (below chance) vs conflict/new **0.65**; metadata heuristics cluster at/below the 0.5 line.

![Validity battery per-signal OOD AUC](../figures/fig1_validity_auc.png)

### Fig 2 — Agent-conflict validates on human hard-event categories
Pedestrian-density **0.866**; bars faded where n<20.

![Per-event-cluster conflict AUC](../figures/fig2_conflict_by_cluster.png)

### Fig 3 — Low light measurably degrades perception (n=3,334)
Dark clips: **−10%** detection confidence, **−24%** detections.

![Darkness degrades perception](../figures/fig3_darkness_degrades_perception.png)

### Fig 4 — Dark clips: from discarded to kept
Rank-correlation of the score with darkness flips **−0.14 → +0.61** after the union.

![Inversion fix](../figures/fig4_inversion_fix.png)

### Fig 5 — Both axes contribute after balancing (Gold = 3,176 clips)
Effect of rank-normalizing the perceptual axis.

![Gold composition](../figures/fig5_gold_composition.png)
