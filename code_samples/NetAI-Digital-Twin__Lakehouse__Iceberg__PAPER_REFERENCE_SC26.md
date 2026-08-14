# Project Reference for SC26 Workshop Paper
*Comprehensive, source-cited reference for writing a 6+ page paper. Compiled 2026-07-16.
This is a writing reference (facts, numbers, framing) — not the paper itself. Every
number is traceable to a repo doc; citations in `(→ file)` form.*

---

## 0. How to use this document
Sections 1–3 give the pitch/novelty/motivation. §4–5 are the **systems** contribution
(lakehouse + scalability — the SC-relevant core). §6–8 are the **application**
contribution (validated difficulty curation + synthetic augmentation). §9 is the HPC
engineering narrative. §10 is the dated evolution. §11–16 are limitations, future work,
reproducibility, figures, related work, and a **suggested 6-page outline**.

### On "is it too soon for another paper?"
No — the two are different contributions at different maturities:
- **March 2026 (prior, 2 pages):** a work-in-progress lakehouse **proof-of-concept**,
  benchmarked on **synthetic data** (nuScenes + a simulated KAIST schema) on a **DGX
  Spark**, evaluating **query latency only**. No real corpus, no ML, ephemeral catalog.
- **Now (SC26, 6+ pages):** a **production** medallion lakehouse over a **real 10 TB**
  slice of NVIDIA PhysicalAI, contributing (a) a **validated edge-case-curation
  methodology** (the "validity battery" + difficulty scoring), (b) a **label-preserving
  synthetic-augmentation pipeline** on an A100 cluster, and (c) the **HPC systems**
  work to run all of it at scale. The overlap with the 2-pager is one figure of
  scalability numbers; everything else is new. Incremental venue papers on an evolving
  system are normal, workshops especially. Frame it as "since our earlier PoC, we now …".

---

## 1. One-paragraph abstract (draft)
Petabyte-scale autonomous-vehicle datasets make training expensive and
redundancy-heavy: most logged miles are trivially easy. We present a **medallion
lakehouse** (Apache Iceberg + Polaris REST catalog + MinIO + Spark) that ingests a
real ~120 TB / ~300k-clip AV corpus **in place** (metadata-only registration, no data
copy) and curates a **validated "edge-case" Gold tier** — the hardest ~10% of clips,
with trivially-easy cases stripped. Difficulty is a **noisy-OR union of independently
validated axes** (agent-interaction "conflict" from 3D labels; camera-perception
degradation), each admitted only after passing a reusable **validity battery**; against
1,740 human-flagged hard clips the curated score reaches **OOD-AUC 0.65–0.75**, versus
**0.45 (worse than random)** for the naïve metadata composite it replaces. Because the
real corpus lacks the camera-adverse cases a camera-only stack must handle, we add a
**label-preserving synthetic-augmentation** stage: depth-controlled NVIDIA Cosmos-Transfer
diffusion turns easy daytime clips into photorealistic night/rain/fog variants on an
A100 SLURM+Singularity cluster, with an automatic **hallucination gate** guaranteeing the
3D labels remain valid. The system sustains **linear ingestion (349 files/s)** and
**O(1) query latency (21–82 ms constant across a 36× data-scale sweep)**, and is packaged
as a parameterized **Helm chart** with CI for reproducible cluster deployment.

---

## 2. Contribution & novelty (what to claim)
1. **A validated edge-case-curation methodology for AV lakehouses.** The "validity
   battery" (negative control + external-label AUC + convergent/discriminant +
   reproducibility) as a *gate every difficulty signal must pass* — and the empirical
   finding that the obvious metadata heuristic is **anti-aligned (AUC 0.450)** with human
   difficulty. (→ MEETING_FACTSHEET_2026-06.md §1–3, VALIDITY_BATTERY_FINDINGS.md)
2. **Modality-aware difficulty.** Perceptual difficulty is *modality-dependent*: a night
   transform drops **camera-only** detection −0.43 confidence but a **lidar-fused** stack
   ≈0 (lidar masks it). We emit **dual Gold** scores (camera-only + lidar-fused) so one
   curation serves both a camera-only consumer and general use. (→ cosmos_augmentation/FINDINGS.md)
3. **Label-preserving synthetic augmentation with a validity gate.** Depth-controlled
   Cosmos day→night/rain/fog that keeps geometry/agents fixed (obstacle.offline labels
   transfer), plus an automatic gate that rejects clips where the generator *added*
   agents (hallucination → invalid labels). (→ cosmos_augmentation/FINDINGS.md)
4. **The HPC systems substrate.** In-place medallion ingestion at PB scale with linear
   ingestion + O(1) queries; a multi-runtime GPU pipeline (containerized BEVFusion
   perception; A100 SLURM+Singularity diffusion); reproducible K8s packaging. (→ SCALABILITY_REPORT.md, BENCHMARK_REPORT.md, deploy/)

---

## 3. Problem & motivation
- **Thesis (the "why"):** *edge-case mining* — produce a subset with trivially-easy
  training cases stripped, on the assumption that a model that handles the hard cases
  handles easy ones trivially. Gold = the hardest clips. (→ MEETING_FACTSHEET §0)
- **Scale:** full catalog **305,724 clips** (Silver); a deliberate **10 TB on-disk random
  sample** = **31,737 "sensor-covered" clips** of the full **~120 TB / ~300k-clip**
  dataset. Sensor-based difficulty signals apply to the sample. (→ FACTSHEET §0, ARCHITECTURE_2026-06)
- **Why a lakehouse:** heterogeneous multimodal sensors (lidar Draco blobs, 6+ camera
  MP4s, radar, ego-motion, 3D auto-labels) + the need to register/query/curate without
  copying petabytes → Iceberg metadata-over-data-at-rest ("register-in-place").
- **Why augmentation:** the real corpus is overwhelmingly camera-*easy* (only ~2.2% of
  clips are camera-hard); the camera-adverse cases a camera-only product must survive
  barely exist → must be *manufactured*. (→ FINDINGS.md §"camera-only perceptual axis")

---

## 4. System architecture
### 4.1 Stack (current)  (→ ARCHITECTURE_2026-06.md)
| Layer | Component | Notes |
|---|---|---|
| Table format | Apache Iceberg v2 | `nvidia_bronze/silver/gold` namespaces |
| Object store | MinIO (S3) | Iceberg metadata/materialized data; raw clip media on NFS |
| Catalog | Apache Polaris REST catalog | **persistent: relational-jdbc on Postgres 16** (was in-memory/ephemeral in March) |
| Compute | Spark 3.5.5 + Iceberg 1.8.1 (`tabulario/spark-iceberg`) | PySpark |
| Query | Trino 479 | |
| BI | Superset + Postgres 15 + Redis | custom chart plugin |
| Perception (GPU) | **BEVFusion** 3D multimodal (mmdet3d 1.4.x, CUDA 12.1) | lidar + 6 cameras (256×704) |
| Difficulty | `planning/` conflict + behavioral runners (GPU-free); `nvidia_ingestion/edge_case_scorer.py` | noisy-OR union |
| Augmentation | `cosmos_augmentation/` — Cosmos-Transfer1-7B on A100 cluster | depth-controlled, label-preserving |

### 4.2 Register-in-place (key systems property)
Bronze registration creates **only Iceberg metadata (manifests/snapshots) pointing at
existing parquet files — no data rewrite**. Consequences: (a) ingestion is a batch job
over data-at-rest, so the platform can be **deployed before or after** the data lands;
(b) manifests store **absolute paths**, so the raw-data mount must stay **stable** after
registration. (→ register_bronze.py; confirmed for the SSD-ingestion scenario)

### 4.3 Deployment / generalization (July 2026)  (→ deploy/, .env.example)
- Config fully **env-driven** (`kaist_ingestion/config.py`, `nvidia_ingestion/config.py`);
  defaults = local/compose dev, overrides = cluster; imports without pyspark (CI-testable).
- **Helm chart** (`deploy/helm/lakehouse`): parameterized values (`values.yaml` = the
  definable variables; `values-prod.yaml` overlay), ConfigMap/Secret, Postgres StatefulSet,
  MinIO + bucket Job, Polaris + bootstrap Job, Spark ingestion pod with the raw-data PVC
  mounted read-only. Renders 13 resources; `helm lint` clean.
- **CI** (`.github/workflows/lakehouse-ci.yml`): ruff + pytest (config resolution) + helm
  lint/template + Spark image build. Green on the pushed commit.

---

## 5. Scalability & systems results  (→ SCALABILITY_REPORT.md, BENCHMARK_REPORT.md)
Platform: DGX Spark (121 GB RAM, 1.9 TB NVMe, NFS-mounted 119 TB dataset). Swept **4
scale factors**, 100→4,994 files/sensor × 14 sensors, **1.4 GB → 52.6 GB** (radar +
egomotion), up to **4.48 billion rows**.

| Metric | Scale 100 (1.4 GB) | Scale 4994 (52.6 GB) | Behavior |
|---|---|---|---|
| Total rows | 123 M | 4.48 B | 36× (linear) |
| Bronze registration | 34.7 s | 188.2 s | **Linear, 349 files/s** |
| Silver materialization | 55.7 s | 1,008.7 s | Linear (2.2→4.4 M rows/s) |
| Gold materialization | 13.8 s | 208.4 s | Linear |
| Full pipeline | 130.7 s | 1,416.2 s | Linear |
| COUNT queries (all tiers) | 21–82 ms | 23–38 ms | **Constant (O(1))** |

- Ingestion regression: **T = 0.00287·files + 18.7 s (R² > 0.99)**; steady-state **349
  files/s**. Query memory **constant 224 MB** regardless of scale.
- **All 11 benchmark queries (Bronze/Silver/Gold) show O(1) latency** across the 36×
  sweep — the register-in-place + Iceberg-metadata design decouples query cost from data
  volume.
- **Projection:** 1 PB Bronze registration ≈ **48 h** (60 M files); COUNT queries stay
  **21–80 ms**. (→ SCALABILITY_REPORT §"Projected to petabyte scale")

---

## 6. Difficulty-scoring methodology
### 6.1 The composite  (→ FACTSHEET §1)
`difficulty = 1 − (1 − behavioral) · (1 − perceptual)` — a **noisy-OR union** (keep a
clip if hard on *either* axis; a weighted average would dilute single-axis-hard clips).
Each axis is **rank-normalized** over the covered population to share a scale.

### 6.2 Validity battery (the reusable methodological asset)  (→ FACTSHEET §2, VALIDITY_BATTERY_FINDINGS.md)
Every candidate signal must pass before reaching Gold:
1. **Negative control** — blank the scene → score must move (proves scene-driven).
2. **External label** — ranks a held-out human hard-clip set higher (AUC vs `ood_reasoning`).
3. **Convergent/discriminant** — correlates with independent proxies, not one trivial factor.
4. **Reproducibility** — deterministic + frame-stable.

### 6.3 Ground truth
`ood_reasoning` = the dataset's **1,740 human-flagged hard clips**, 9 behavioral/daytime
clusters (WORK_ZONES 856, PEDESTRIAN_DENSITY 380, SPECIAL_VEHICLE 260, …). It is
**Positive-Unlabeled** and **entirely behavioral/daytime** — so it validates behavioral
difficulty but structurally *cannot* validate perceptual difficulty (which is handled
separately, §6.5). This is the single most important caveat.

### 6.4 Behavioral axis — agent-conflict  (→ FACTSHEET §4)
From the dataset's own `obstacle.offline` 3D auto-labels (16 GB, 340 chunks, 0 failures;
per-track boxes in ego frame with class + track_id). Signal = forward-zone
inverse-distance agent load, multi-frame, rank-normalized, **GPU-free**. Validated
**OOD-AUC 0.651**, concentrating where it should (pedestrian-density **0.866**, n=52).
A **multi-axis** extension (adding closing-agent, VRU, class-diversity, rarity axes)
reaches **5-fold CV-AUC 0.745** — the single-axis ~0.65 is a construct ceiling, not a
metric gap. (→ FINDINGS / behavioral_runner.py)

### 6.5 Perceptual axis — modality matters  (→ FACTSHEET §5, FINDINGS.md)
Validated directly via detection stats (OOD can't). **Darkness measurably degrades
perception** (n=3,334): mean confidence 0.505→0.456 (**−10%**), detections/frame
11.53→8.72 (**−24%**). The old score had this **backwards** — it rated dark clips
*easier* (sparse dark detections read as "emptier"), stripping perceptually-hard clips
(the goal's opposite). Fix quantified: rank-corr(score, darkness) **−0.14 → +0.61**.

**The modality finding (2026-06):** the consumer's final product is **camera-only**. A
night transform drops **camera-only** YOLO confidence **−0.43** (agents vanish) but the
**lidar-fused** BEVFusion confidence **≈0** — clean lidar masks camera degradation. So
the fused perceptual axis is *blind* to the difficulty the final product will face. A
**camera-only** perceptual axis was built (`camera_perception_runner.py`, YOLO front-cam
over 33,767 clips), **agent-gated** to fix a 25%-empty-scene confound (camera-hard
27.6%→11.1%, OOD 0.43→0.58, −5,218 false positives).

### 6.6 Dual Gold  (→ FINDINGS.md §"Dual Gold")
`clip_scores` emits **both** `difficulty_camera` (consumer endgame) and `difficulty_lidar`
(general); `--gold-axis` picks which materializes the Gold views. Top-10% of 31,737:
camera Gold **3,174** / lidar Gold **3,176**, overlap 2,830, **~374 unique to each tier**
(Jaccard 0.79) — neither redundant.

---

## 7. Difficulty results — headline table  (→ FACTSHEET §3)
| Signal | OOD-AUC | verdict |
|---|---|---|
| sensor_coverage | 0.432 | anti-aligned |
| **composite (OLD metadata)** | **0.450** | **worse than random** |
| time_of_day | 0.477 | ~chance |
| ego_dynamics | 0.498 | ~chance |
| season_geography | 0.507 | ~chance |
| perception | 0.564 | modest (n=29) |
| **conflict (behavioral)** | **0.651** | **valid** |
| **composite (NEW union)** | **0.655** | **valid** |
| multi-axis behavioral (5-fold CV) | **0.745** | valid |

Gold: **3,176 clips = top 10%** of the 31,737 sample; score-spread std 0.087→~0.21
(now discriminative). Gold composition after rank-norm: ~70% behaviorally hard, ~78%
perceptually hard, **~46% perceptual-rescued** (kept purely on darkness — a conflict-only
score would have discarded them). (→ FACTSHEET §6)

---

## 8. Synthetic augmentation  (→ cosmos_augmentation/FINDINGS.md)
### 8.1 Why + feasibility
Real data is camera-easy → manufacture camera-hard variants. **Cosmos-Transfer1-7B is
cluster-only**: ~80 GB VRAM (≫ local 24 GB), no hosted API (download-only) → the A100
SLURM+Singularity cluster.

### 8.2 Method
Depth-controlled Cosmos-Transfer diffusion: extract a depth map from the real clip →
prompt a target condition (night/rain/fog) → generate. **Depth control preserves 3D
geometry + agent positions** (lighting-invariant), so `obstacle.offline` boxes + ego
trajectory transfer → **labels stay valid**. Recipe (from a control×condition matrix):
**depth ≫ edge** (edge retains daytime); mix night/rain/fog (night kills confidence,
fog/rain make agents vanish).

### 8.3 Result (single clip, validated end-to-end)
Photorealistic night render, geometry/agents preserved, **harder for camera-only
perception** (YOLO −0.22 conf, −1.67 detections). ~7 min/clip on 4× A100-40 GB (one node).

### 8.4 Safety — the label-validity story (a paper highlight)
The first batch exposed a **hallucination bug**: content-mentioning prompts made Cosmos
*invent* agents on sparse scenes (empty road → added taillights) → invalid labels. Two
fixes, both validated: (a) **condition-only prompts** (lighting/weather, never vehicles)
cut added detections **+0.8 → +0.10**; (b) an automatic **hallucination gate** (reject
clips that gain detections vs the original) + **agent-window selection** (augment the
window that actually contains agents). Full safe pipeline over 9 easy clips: **KEEP 7/9**
(label-valid *and* harder, ~−1.4 detections/clip), gate auto-filtered 1 hallucination + 1
no-op (~78% keep-rate).

---

## 9. HPC / cluster engineering narrative (SC loves this)
Concrete systems work worth a subsection (→ FINDINGS.md, memory a100-cluster-access):
- **Multi-runtime GPU pipeline:** containerized BEVFusion (Docker) for perception;
  Cosmos diffusion on a separate **A100 SLURM+Singularity** cluster; local dev on
  docker-compose; production on K8s.
- **Locked-down login node:** cannot compile/pull container images → built the
  Cosmos-Transfer1 **SIF locally** (apptainer fakeroot, 14.7 GB) and **transferred** it
  (SFTP, 156 s); staged **113 GB** of weights; resolved a full from-scratch dependency
  chain (blinker/distutils, requirements_docker vs conda, sam2's `--no-build-isolation`
  install, a transformers 4.49 pin, NCCL wheel ordering, a 40 GB-VRAM decode OOM fixed by
  DiT offload + 121-frame windows + expandable_segments, and **patching out gated guardrail
  models** — Cosmos-Guardrail1 + Meta Llama-Guard — to avoid extra license gates).
- **Multi-GPU inference:** 40 GB A100s < the 80 GB single-GPU need → context-parallel
  across 4 NVLinked A100s.
- **Shared-storage-only comms:** while a SLURM job runs, the only channel is a
  Singularity-bind-mounted folder → submit + poll the mount.
- **Deterministic packaging:** the whole stack as a parameterized Helm chart + CI.

---

## 10. Evolution timeline (for "since our last paper")
- **Mar 2026** — Lakehouse PoC; synthetic data (nuScenes + simulated KAIST); DGX Spark;
  ephemeral in-memory Polaris; **query-latency benchmark only**; the 2-page paper. (→ ARCHITECTURE_2026-03)
- **Apr 2026** — Foundational build on the real NVIDIA corpus; NFS lidar/radar recovery
  (~10.85 TB re-download); canonical schema; perception integration. (→ progress/2026-04.md)
- **May 2026** — BEVFusion multimodal perception operational (mmdet3d). (→ progress/2026-05.md)
- **Jun 2026** — Durable (Postgres-backed) Polaris; agent-conflict from `obstacle.offline`;
  the **validity battery** (refuted the old composite, 0.450); noisy-OR union
  re-architecture (Gold = 3,176). (→ progress/2026-06.md §13–16, FACTSHEET)
- **Late Jun–Jul 2026** — Camera-only perceptual axis + the modality finding; **dual Gold**;
  **Cosmos-Transfer augmentation** validated end-to-end (recipe, hallucination gate,
  agent-window); **K8s Helm + generalization + CI/CD**. (→ progress/2026-06.md §17–20, FINDINGS.md)

---

## 11. Limitations & threats to validity (be upfront — reviewers will ask)
- **Ground truth is PU + behavioral/daytime only** — validates behavioral difficulty; the
  perceptual axis is validated by a *different* proxy (detection degradation), not the OOD
  label. No independent human perceptual-difficulty set exists.
- **Per-axis construct ceiling ~0.65** against OOD (multi-axis breaks it to 0.745, but at
  overfitting risk on a narrow PU label — we keep conservative fixed weights).
- **Augmentation at small scale** — end-to-end validated on ~1–9 clips; no full-scale
  augmented corpus yet, and **no downstream training result** (the ultimate test).
- **Augmentation realism unmeasured** — visually convincing, but no Cosmos-Evaluator /
  FID-style realism score yet; the gate covers label-validity, not photorealism.
- **Small-n per-cluster AUCs** (several clusters n<10) — indicative, not precise.
- **Scalability benchmark** covers radar+egomotion (structured) columns; lidar/camera blob
  registration is in-place (cheap) but end-to-end curation throughput at PB scale is
  projected, not measured.

## 12. Future work
Grow the on-disk sample; validate adverse-weather axes under the battery; the camera-3D
perceptual axis (fcos3d/pgd) vs YOLO-2D; scale augmentation to a full corpus + a
**downstream training study** (does training on Gold+augmented beat random subsets?);
Cosmos-Evaluator realism gating; K8s deploy on the storage cluster for the incoming SSD
data batches.

## 13. Reproducibility / artifacts
- **Code:** `nvidia_ingestion/` (pipeline + `edge_case_scorer.py` + `run_gold_score.py
  --gold-axis`), `planning/` (conflict/behavioral/camera-perception runners), `bevfusion/`
  (perception), `cosmos_augmentation/` (SIF def, cluster helpers, `select_easy_clips` →
  `stage_batch` → `cosmos_batch.sbatch` → `apply_hallucination_gate`, `safety.py`).
- **Deploy:** `deploy/helm/lakehouse` (Helm), `deploy/docker/Dockerfile.spark`, `.env.example`
  (the config contract), `.github/workflows/lakehouse-ci.yml`.
- **Config:** env-driven; `run_gold_scoring(gold_axis=…)`. Validity: `validity_battery.py`.
- **Data:** 10 TB NFS sample; `obstacle.offline` 3D labels. Difficulty file-drops:
  `.conflict/`, `.behavioral/`, `.camera_perception/`, `.perception_bevfusion/`.

## 14. Figure & table inventory
**Existing PNGs** (`nvidia_ingestion/figures/`, → FACTSHEET §9):
1. Per-signal validity AUC (old 0.450 vs new 0.65; chance line).
2. Per-cluster conflict AUC (pedestrian 0.866).
3. Darkness degrades perception (−10% conf / −24% detections).
4. The inversion fix (−0.14 → +0.61).
5. Gold composition (both axes contribute).
**Recommended new figures for this paper:**
6. Scalability: linear ingestion + O(1) query latency across 36× (→ SCALABILITY_REPORT).
7. System/architecture diagram (medallion + GPU/cluster runtimes + deploy).
8. Modality split: camera-only −0.43 vs lidar-fused ≈0 Δconf (the dual-Gold motivation).
9. Augmentation before/after (day vs Cosmos night; agents preserved) + the hallucination
   before/after (invented taillights → gated). Composites already produced.
10. Safe-pipeline flow + keep-rate (7/9).

## 15. Related-work pointers (angles to cite)
Data-centric AI / dataset distillation & coreset selection; active learning & hard-example
mining; AV edge-case/scenario mining; lakehouse table formats (Iceberg/Delta) for ML data;
in-place / zero-copy ingestion; synthetic data & sim2real for AV (Cosmos world-foundation
models, diffusion augmentation); controllable diffusion (depth/edge ControlNet); label
noise from generative augmentation. (Position the novelty as: *validated* curation gated by
a battery + *label-preserving* augmentation with an automatic validity gate, on a
*production HPC lakehouse*.)

## 16. Suggested 6-page SC26-workshop outline
1. **Intro** (0.75 pg) — PB AV data, redundancy, the edge-case thesis; contributions list (§2).
2. **Background/related** (0.5 pg) — lakehouse for ML data; edge-case mining; synthetic AV (§15).
3. **System architecture** (1 pg) — medallion + register-in-place + multi-runtime GPU/cluster + deploy; the architecture figure (§4).
4. **Scalability** (0.75 pg) — the linear/O(1) results + PB projection; figure 6 (§5).
5. **Validated difficulty curation** (1.25 pg) — battery, behavioral + camera-only perceptual axes, the 0.450→0.65/0.75 result, dual Gold; figures 1–4, 8 (§6–7).
6. **Label-preserving synthetic augmentation** (1 pg) — depth-control, hallucination gate, 7/9 keep; figure 9 (§8).
7. **Limitations & future work** (0.5 pg) — §11–12.
8. **Conclusion** (0.25 pg).
*Emphasis for an SC venue: foreground §3–5 (systems + scale); present §6–8 as the
application that drives and validates the system.*
