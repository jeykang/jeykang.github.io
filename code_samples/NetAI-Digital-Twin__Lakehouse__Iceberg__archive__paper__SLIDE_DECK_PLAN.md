# Slide Deck Plan: Data Lakehouse for Autonomous Driving Data Management

> **Presentation context:** Government project (KAIST/MOTIE) quarterly development check-in
> **Audience:** Other participants in the government-funded project — AI/ML researchers and engineers with deep domain expertise, but not data-infrastructure specialists. They will ask hard questions.
> **Prior context the audience already has:** The project plan states we are building a data lakehouse using open-source technologies. At the previous meeting (~1 month ago) we presented a high-level structural diagram of the lakehouse architecture. That diagram will be placed before Slide 1 as a quick refresher. The audience knows WHAT we're building; this meeting is about HOW we built it and whether the implementation is on track.
> **Tone:** Development progress report with evidence. Not introducing something new — showing that we executed on the approved plan, explaining the implementation-level decisions we made along the way, and validating those decisions with benchmarks.
> **Deployment context:** The Docker Compose + MinIO stack is for local development and testing only. The production deployment target is a dedicated Ceph storage cluster, orchestrated via Kubernetes. All implementation decisions are made with this production target in mind.
> **Format:** 4 poster-density slides — each one a self-contained reference sheet
> **Estimated duration:** 15–20 minutes + Q&A
> **Date:** March 4, 2026
>
> ### Narrative arc (framed as progress against plan)
> 1. **Implementation depth:** Starting from the approved architecture, here are the specific implementation decisions we made during development — schema design, pipeline structure, partitioning strategy — and why
> 2. **Precision about scope:** Exactly what the pipeline transforms (and what it deliberately does not), anticipating the "it's just metadata reorganization" challenge
> 3. **Evidence:** Benchmarks that map to real ML data-loading patterns, not arbitrary SQL, with explicit methodology and limitations
> 4. **Status & open questions:** What works, what doesn't yet, and what needs decisions from this group

---

## Generated Figures

All figures pre-generated at 200 DPI in `paper/figures/`.

| # | Filename | Content |
|---|----------|---------|
| F1 | `data_flow.png` | Data flow — new AD dataset ingestion through the lakehouse |
| F2 | `data_model.png` | KAIST 3-level hierarchy diagram |
| F3 | `medallion_pipeline.png` | Bronze → Silver → Gold pipeline flow |
| F4 | `workload_benchmark.png` | Grouped bar chart — Gold vs Silver, 3 workloads |
| F5 | `scalability.png` | Line chart — latency vs scale factor (1×–50×) |
| F6 | `supplementary_benchmarks.png` | 3-panel: partition pruning, temporal replay, column metrics |
| F7 | `gold_tables.png` | Gold table design cards (3 tables) |
| F8 | `validation_summary.png` | Data quality validation table (20 checks) |

---

## Slide 1 — Implementation Progress: From Approved Architecture to Working Pipeline

> *"You know what we're building — here's how we built it, the implementation decisions we made, and why."*

**Layout:** Top banner (context refresher). Left half: data flow + data model figures. Right half: Implementation Decisions table (the core of this slide — decisions made DURING development, not the top-level tech choices already presented). Bottom strip: dev/test environment note.

**Note:** The high-level architecture diagram from last month's meeting is placed as an unnumbered refresher slide immediately before Slide 1.

### Top Banner — Where We Are vs. Where We Were

> **Last meeting (refresher):** We presented the approved architecture — a data lakehouse built on Apache Iceberg, Spark, and Polaris, with Ceph object storage and Kubernetes deployment as the production target.
>
> **Since then:** We implemented the full three-layer pipeline (Bronze → Silver → Gold), designed the KAIST 3-level schema (14 entity types), built automated validation (20 checks), ran benchmark experiments, and validated the pipeline on both simulated KAIST data and public nuScenes data. Everything runs in a local Docker dev/test environment; production deployment to Ceph + K8s is the next phase.

### Left Half — System Data Flow & Data Model

**Figure (upper-left):** `paper/figures/data_flow.png` (compact)

Shows the end-to-end journey: Vehicle JSON (14 files) → Bronze (schema enforcement) → Silver (physical optimization) → Gold (pre-joined per ML task) → Validation (20 checks) → Consumption (ML training / SQL / dashboards). Infrastructure components (Spark, Iceberg, Polaris, MinIO, Trino, Superset) shown as annotations on the flow, not the focus.

**Figure (lower-left):** `paper/figures/data_model.png` (compact)

KAIST 3-level hierarchy: Session → Clip → Frame → Sensors/Annotations. 14 entity types, 4 named geometric structs (SE3, Quaternion, Box3D, Translation3D).

### Right Half — Implementation Decisions Made During Development

The top-level technology choices (Iceberg, Spark, Polaris, Ceph) were already presented and approved. This table covers the implementation-level decisions we made WHILE building the pipeline — the kinds of choices that only surface once you start writing code against real AD data structures.

| Decision | What We Chose | Alternative We Considered | Why This Choice |
|----------|--------------|--------------------------|-----------------|
| **Schema hierarchy** | KAIST 3-level (Session → Clip → Frame) | Adopt nuScenes 2-level (Scene → Sample) directly | nuScenes has no "Session" concept — multi-drive campaigns are separate scenes. KAIST's multi-day collection campaigns need a Session level that groups Clips. We validated by also implementing nuScenes ingestion — the 3-level schema is a strict superset. |
| **Pipeline layering** | 3 layers (Bronze → Silver → Gold) with distinct responsibilities | Single-stage ingest (raw → final), or direct-to-Gold per-task export (HDF5/TFRecord) | Bronze preserves raw data immutably (audit trail, reprocessability). Silver optimizes physical layout (benefits ALL downstream queries, not just one task). Gold specializes per ML workload. Single-stage ETL loses the raw audit trail. HDF5/TFRecord are monolithic — updating one annotation means regenerating the entire file; no schema evolution, no versioning, no multi-engine access. |
| **Gold table granularity** | One table per ML workload (3 tables) | One giant denormalized table containing everything | Object detection needs one camera at a time; SLAM needs only LiDAR + ego pose. A single denormalized table would include ALL sensors and ALL annotations in every row — enormous, wasteful, and slow to scan. Workload-specific tables include exactly the columns each task needs, partitioned by that task's dominant access pattern. |
| **Partition strategy** | Different partition key per table based on access pattern | Uniform partitioning (e.g., always by `clip_id`) | Camera data is accessed by camera name (6 cameras → 83% pruning). LiDAR/temporal data is accessed by clip (driving sequence). Using the wrong partition key means the engine can't skip irrelevant files. We tuned partition keys per table based on how each ML workload actually queries the data. |
| **Schema enforcement** | Strict PySpark `StructType` at Bronze ingest — wrong type = hard failure | Permissive ingest (accept everything, validate later) | Permissive ingest lets corrupt data silently enter the pipeline. A float where a long is expected, or a missing field, gets caught immediately at Bronze rather than causing mysterious NaN values during training weeks later. |
| **Validation as pipeline gate** | 20 automated checks (PK, FK, quaternion norms, timestamps, row counts) — pipeline halts on failure | Manual spot-checking, or validate-after-the-fact | Non-unit quaternions silently produce wrong rotation matrices. Broken FK references silently produce NULL calibration in Gold joins. These errors don't throw exceptions — they produce wrong training data. Automated validation catches them before data reaches Gold. |

### Bottom Strip — Development & Deployment Context

```
Local dev/test: Docker Compose (5 services) · MinIO as S3 stand-in · single-node · ~60 s cold start
Production target: Ceph storage cluster · Kubernetes orchestration · multi-node Spark
All code uses S3 API — switching from MinIO (dev) to Ceph (prod) is a configuration change, not a code change.
```

**Speaker notes:** "You've seen the high-level architecture before — this slide goes one level deeper into the decisions we made during implementation. The technology stack was already approved; what I want to show here is the reasoning behind the finer-grained choices: why three pipeline layers instead of one, why one Gold table per ML workload instead of one big denormalized table, why we enforce types strictly at ingest rather than being permissive. Each of these came from concrete problems we encountered while working with the AD data structures. I'm happy to go deeper on any of these during Q&A."

**Anticipated Q&A for this slide:**

> **Q: "Why not just export to HDF5 or TFRecord directly? That's what we use for training."**
> A: We could add an HDF5/TFRecord export stage downstream — and may well do so. They're great for batched I/O during training. But they're monolithic: when the annotation team fixes a labeling error in 50 frames, with HDF5 you regenerate the entire file. With Iceberg, you update 50 rows. The lakehouse manages the living, evolving dataset; HDF5/TFRecord are a snapshot format for the training loop. Complementary, not competing.

> **Q: "Why three pipeline layers? Isn't that overengineering for our use case?"**
> A: Bronze gives us an immutable audit trail — if anything goes wrong downstream, we can always reprocess from raw data without going back to the collection team. Silver's physical optimizations (partitioning, sorting, column statistics) benefit every query, not just a specific ML task. Gold materializes task-specific joins. If we collapsed to one layer, we'd lose either the raw audit trail or the ability to optimize for multiple access patterns independently.

> **Q: "How does this deploy to our Ceph cluster? Is the Docker setup the final form?"**
> A: No — Docker + MinIO is strictly for local development and testing. All our code uses the S3 API, which Ceph supports natively via its S3-compatible gateway. Production deployment will use Kubernetes: Spark on K8s, Polaris as a service, Ceph as the storage backend. The switch is configuration-only — connection endpoint and credentials change, but no pipeline code changes. We've designed for this from the start.

---

## Slide 2 — What the Pipeline Actually Does (and What It Deliberately Does Not)

> *"The Gold table doesn't transform pixels — it transforms the access pattern. Here's precisely what that means."*

**Layout:** Top band: scope precision statement (the most important paragraph on this slide). Upper-middle: pipeline detail with concrete examples. Lower-middle: Gold table deep-dive showing what one row actually contains. Bottom: validation framework.

### Top Band — Scope Precision (Anticipating the "Metadata-Only" Critique)

This section exists specifically to preempt: *"The processing between tiers seems like it only operates on metadata — the Gold tier data is functionally raw data, right?"*

> **What the lakehouse transforms — and what it does not:**
>
> Each tier performs a REAL transformation, but the nature of the transformation is structural, not perceptual:
>
> | What changes between tiers | Concrete example |
> |---------------------------|-----------------|
> | **Bronze → Silver: Physical layout** | The `camera` table goes from one unsorted file → partitioned by `camera_name` + `clip_id`, sorted by `sensor_timestamp`, with per-file min/max statistics on every column. A query for "front camera, clip X" now reads **1.6% of files** instead of 100%. |
> | **Silver → Gold: Cross-table join materialization** | A single row in `camera_annotations` contains the camera filepath, the 3D bounding box coordinates, the category label, the camera intrinsic matrix, the extrinsic (sensor-to-vehicle) SE3 transform, the ego-vehicle pose, and the HD map reference — **pulled from 6 different source tables and pre-joined.** |
> | **What does NOT change** | The image pixels are not resized, cropped, normalized, or augmented. The point cloud is not voxelized. The pipeline does not perform feature extraction. |
>
> **Why this matters anyway:** To construct a single training sample for BEVFormer (3D object detection), the `__getitem__` call needs data from **6 tables**: camera path, frame index, annotations with 3D boxes, calibration intrinsics + extrinsics, ego pose, and map metadata. Without the Gold table, this is a 6-table graph traversal — per sample, per epoch, every training run. The Gold table materializes this graph traversal once, at ingest time. The training loop reads one flat table.
>
> **The dividing line:** The lakehouse handles everything up to "give me a complete, validated, correctly-assembled training sample as a single row." What happens AFTER that row is read — image decoding, geometric transforms, augmentation, tensorization — is the ML pipeline's responsibility and is deliberately out of scope.

### Upper-Middle Band — Pipeline Walkthrough with Concrete Data

**Figure:** `paper/figures/medallion_pipeline.png` (compact, spanning width)

| Layer | Transformation | Concrete Example | Implementation |
|-------|---------------|-----------------|----------------|
| **Bronze** | Schema enforcement. Each JSON file → 1 Iceberg table with strict PySpark `StructType` typing. Wrong type or missing field = hard failure. Raw data preserved immutably. | `camera.json`: 23,150 records with `frame_id` (string), `sensor_timestamp` (long), `camera_name` (string), `filename` (string). Schema rejects a record where `sensor_timestamp` is a float. | `ingest_bronze.py` — 14 tables, zero hardcoded column names. Schema defined once in `schemas.py`. |
| **Silver** | Physical reorganization: partition by dominant access pattern, sort within partitions by timestamp, write Iceberg column-level min/max statistics. Data values unchanged; file layout transformed. | `camera` table: partitioned by `(camera_name, clip_id)` → 6 cameras × N clips = many small files. Each file's metadata records its min/max `sensor_timestamp`. Query for `CAM_FRONT, clip X` skips all other files without opening them. | `SilverTransformer` with `PARTITION_CONFIG` (11 tables), `SORT_CONFIG` (8), `METRICS_CONFIG` (8). |
| **Gold** | Cross-table join materialization. Multiple Silver tables joined into one flat table per ML workload. Repartitioned by workload-specific access key, sorted for sequential access. | `camera_annotations`: camera ⋈ frame ⋈ clip ⋈ calibration ⋈ dynamic_object ⋈ hdmap. One row = one (camera, frame) pair with ALL associated metadata pre-joined. Partitioned by `camera_name`, sorted by `(clip_id, frame_idx)`. | `GoldTableBuilder` — 3 build methods, one per workload. Join logic mirrors what a DataLoader would do at runtime. |

### Lower-Middle Band — What a Gold Table Row Actually Contains

This is the most important detail for the "is this actually useful for training?" question. Show a concrete row from `camera_annotations`:

| Column | Type | Example Value | Used in Training For |
|--------|------|---------------|---------------------|
| `frame_id` | string | `frame_f77054fdc78e40808f7005a8` | Sample identification / dedup |
| `clip_id` | string | `clip_e3901eab65214e1b894f6b3b` | Sequence grouping (temporal models) |
| `session_id` | string | `sess_abc123` | Campaign-level filtering |
| `frame_idx` | int | `42` | Frame ordering within clip |
| `sensor_timestamp` | long | `1532402927612460` (μs epoch) | Temporal synchronization, sequence models |
| `camera_name` | string | `CAM_FRONT` | Multi-camera selection |
| `filename` | string | `samples/CAM_FRONT/...jpg` | **Image loading** (path → `cv2.imread()` or `PIL.Image.open()`) |
| `extrinsics` | struct{translation, rotation} | SE3: `{t: [1.7, 0.0, 1.5], r: [0.507, -0.498, ...]}` | **3D→2D projection** in detection loss (`P = K @ T_cam_ego @ T_ego_world`) |
| `camera_intrinsics` | array\<double\> (3×3) | `[1266.4, 0, 816.3, ...]` | **Camera projection matrix K** — directly in the forward pass |
| `annotations` | array\<struct{boxes_3d, category}\> | `[{boxes_3d: [373.2, 1130.4, 0.8, 0.67, 0.62, 1.64, -0.37], category: "human.pedestrian.adult"}, ...]` | **Ground truth labels** — the boxes go into the detection loss, the category into the classification head |
| `city` | string | `singapore-onenorth` | Map-conditioned models |
| `date` | string | `2018-07-24` | Temporal / weather stratification |

**Key point for the audience:** 7 of these 10 columns feed directly into the model's forward pass or loss computation. Only `filename` is a pointer to external data (the image bytes) — and that's true of every dataset management system, because storing 1280×960×3 JPEG bytes in a table row would be pathological. The Gold table contains **training-relevant data, not just metadata.**

### Bottom Band — Automated Validation (20 Checks)

**Figure:** `paper/figures/validation_summary.png` (compact)

| Category | Checks | What We Catch | Why It Matters |
|----------|--------|---------------|----------------|
| **PK uniqueness** (6) | Every record ID unique per table | Duplicate frames → biased training distribution, inflated loss |
| **FK integrity** (4) | Cross-table references resolve | Broken FK → DataLoader gets NULL calibration for some frames → silent NaN in projection |
| **Quaternion norms** (2) | $\|q\| \approx 1.0 \pm \epsilon$ for all rotations | Non-unit quaternion → corrupted rotation matrix → wrong 3D box projection. **Silently wrong** — no error, just bad geometry |
| **Timestamp validity** (4) | All `sensor_timestamp` ≥ 0 | Negative timestamps → data corruption indicator. Breaks time-range filters and temporal ordering |
| **Gold row-count** (4) | Gold rows = expected from source join | Mismatch → join silently dropped or duplicated rows |

Pipeline halts on any CRITICAL failure — bad data never reaches Gold.

**Speaker notes:** "This slide answers the most important question you might have: 'what does the pipeline actually DO to the data?' The honest answer is: it doesn't touch the pixels or point clouds. What it does is solve the data ASSEMBLY problem. To build one training sample for BEVFormer, you need data from 6 different tables — the camera path, the frame metadata, the 3D annotations, the calibration matrices, the ego pose, and the map. Without Gold, your DataLoader does this 6-table lookup per sample, per epoch. With Gold, it reads one pre-joined row. The bottom section shows our 20 automated quality checks — the quaternion validation is particularly important because a non-unit quaternion doesn't throw an error, it just produces slightly wrong rotation matrices, which means slightly wrong 3D bounding box projections, which means silently corrupted training data."

**Anticipated Q&A for this slide:**

> **Q: "The Gold tier is still raw data with better indexing — how is that 'ML-ready'?"**
> A: Look at the column breakdown: 7 of 10 columns in `camera_annotations` feed directly into the model forward pass or loss. The 3D bounding boxes ARE the ground truth; the intrinsics/extrinsics ARE the projection matrices; the ego pose IS the coordinate transform. The only thing that's "just a path" is the image filename, and that's by design — image bytes belong in the DataLoader, not in a table. What the Gold table eliminates is the 6-table LOOKUP that assembles these values. That lookup is the difference between `row = table[idx]` and `row = join(camera[idx], frame[camera.frame_id], calibration[camera.clip_id, camera.name], annotations[camera.frame_id], ego[camera.frame_id], map[camera.clip_id])`.

> **Q: "Why not pre-compute features (e.g., BEV projections, voxelized point clouds) in the Gold layer?"**
> A: Because feature computation is model-specific and changes with every architecture. BEVFormer uses deformable attention on multi-view images; CenterPoint voxelizes point clouds; BEVFusion does both. If we baked BEV features into Gold, we'd need to rebuild the table every time someone tries a new model. The lakehouse provides the invariant: "correctly assembled, validated raw observations." Feature engineering varies per model and belongs in the training pipeline.

> **Q: "What about data augmentation? Doesn't training need augmented data?"**
> A: Augmentation (random flip, scale, rotation, color jitter, mixup, etc.) is stochastic and applied on-the-fly per epoch — it can't be pre-materialized. The Gold table provides the deterministic inputs that augmentation operates ON. This is the same division of labor as nuScenes SDK (provides raw samples) + mmdet3d (applies augmentations at training time).

---

## Slide 3 — Benchmark Evidence

> *"We measured data assembly latency — the per-epoch I/O cost that scales with dataset size. Here's why, how, and what we found."*

**Layout:** Top band: why we benchmark this + methodology. Upper-middle: Experiment 1 (three workloads). Lower-middle: Experiment 2 (scalability). Bottom: supplementary Iceberg feature experiments.

### Top Band — Why Data Assembly Latency, Not End-to-End Training Time

Before showing numbers, address the question: *"How do you know these queries map to real training workloads?"*

> **What we measured:** The time to assemble a complete set of training samples from the data store — the operation that a `DataLoader.__getitem__` or `Dataset.__getitem__` performs once per batch.
>
> **Why this specific metric:**
> 1. **Every ML model needs this step.** Whether you're training BEVFormer, CenterPoint, or UniAD, the first thing the DataLoader does is fetch image paths, annotations, calibration matrices, and ego poses. The model architecture changes; the data assembly step is invariant.
> 2. **It's the step that scales with dataset size.** GPU forward/backward passes take the same time regardless of whether you have 10K or 10M samples. Data assembly scales linearly unless you optimize the storage layer.
> 3. **It's what the lakehouse is designed to optimize.** We are not claiming to make models train faster. We are claiming to make data loading faster and more reliable, which reduces per-epoch wall time.
>
> **What we did NOT measure:** End-to-end training time (would require GPU allocation + model code integration — planned for next quarter, see Slide 4). The benchmarks show the data I/O improvement; the training-time improvement will be smaller because it also includes GPU compute.

> **How each benchmark query maps to a real training operation:**
>
> | Benchmark Query | Training Equivalent | What It Replaces |
> |----------------|--------------------|-----------------| 
> | `SELECT * FROM camera_annotations WHERE camera_name = 'CAM_FRONT'` | `nuScenes.get_sample_data(cam_token)` + `nuScenes.get_sample_data_path()` + `nuScenes.get_boxes()` + calibration lookup | 6-table graph traversal per sample in nuScenes SDK, done once per sample per epoch |
> | `SELECT * FROM lidar_with_ego WHERE clip_id = ?` | Loading a KITTI sequence: lidar bin file + oxts pose + calib file for each frame | 3-file-per-frame access pattern across a driving sequence |
> | `SELECT * FROM sensor_fusion_frame WHERE clip_id = ?` | `nuScenes.get_sample(sample_token)` + iterating over all sample_data entries + annotations | The full multi-modal sample assembly that frameworks like BEVFusion do per batch |

### Methodology (Applies to All Experiments)

| Aspect | Detail | Why |
|--------|--------|-----|
| **Fair comparison** | Silver query replicates the EXACT same join graph as the corresponding Gold table build (verified line-by-line against `build_gold.py`) | Isolates pre-join benefit; doesn't conflate with different query logic |
| **Same filter predicate** | Identical `WHERE` clause for Gold and Silver | Isolates join cost, not filter selectivity |
| **JVM warmup** | 3 throwaway Spark SQL queries before ANY timing | Eliminates JVM class-loading and JIT compilation artifacts |
| **Timing protocol** | 2 untimed warmup runs → 5 timed runs → **median** reported | Warmup eliminates first-query caching effects; median resists outliers |
| **Metric** | `df.count()` action — forces full scan + join execution | Measures engine time without Python serialization. Acknowledged limitation: a real DataLoader would do batched random access, not a full-table count. The speedup from pre-joining is still captured. |
| **Filter values** | Sampled from live data (`camera_name` drawn from actual table) | Not hardcoded — adapts to actual data distribution |
| **Environment** | Single-node Docker dev setup: Spark 3.5.5, Iceberg 1.8.1, Polaris, MinIO (Ceph stand-in) | Reproducible on any machine with Docker. Production deployment on Ceph + K8s (multi-node) would show larger speedups due to parallelism. |

### Experiment 1 — Three AD Workloads: Gold vs. Silver (KAIST Dataset)

**Dataset:** KAIST-simulated (14 tables, 140K camera annotations, 3,935 frames)

**Figure:** `paper/figures/workload_benchmark.png`

| Workload | ML Task | Gold (ms) | Silver JOIN (ms) | Speedup | Join Eliminated |
|----------|---------|-----------|-----------------|---------|-----------------|
| **Object Detection** | BEVFormer, DETR3D, CenterPoint | **79** | 255 | **3.2×** | camera ⋈ frame ⋈ clip ⋈ calibration ⋈ dynamic_object ⋈ hdmap (6 tables) |
| **SLAM / Localization** | ORB-SLAM, LIO-SAM, KISS-ICP | **64** | 138 | **2.2×** | lidar ⋈ ego_motion ⋈ calibration (3 tables) |
| **Sensor Fusion** | TransFusion, BEVFusion, UniAD | **49** | 99 | **2.0×** | frame ⋈ camera ⋈ lidar ⋈ radar ⋈ dynamic_object (5 tables + 3 aggregations) |

**Reading this table:** The "Silver JOIN" column is not a strawman — it is the exact query a training pipeline would need to execute at runtime if there were no Gold table. The speedup is 2–3× on simulated data. On production-scale data with more files, partition pruning and column statistics contribute more, so speedups grow (see Experiment 2).

### Experiment 2 — Scalability: Latency vs. Data Scale (nuScenes, 1×–50×)

**Dataset:** nuScenes v1.0-mini (public), synthetically replicated 1× to 50× (18 scale points). Observation tables scaled; reference tables at 1× (realistic growth pattern).

**Task:** "Load front-camera images with adult-pedestrian 3D annotations" — the exact data assembly step before feeding a batch to BEVFormer.

**Python baseline:** Conventional `for`-loop over JSON dictionaries (nuScenes tutorial style — `for sample in nusc.sample: ...`).

**Figure:** `paper/figures/scalability.png`

| Scale | Rows | Python (ms) | Gold (ms) | Speedup vs. Python | Interpretation |
|------:|-----:|------------:|----------:|-------------------:|----------------|
| 1× | 27K | 15 | 42 | 0.4× (Python wins) | Spark overhead > data assembly savings at tiny scale |
| 3× | 82K | 48 | 50 | 1.0× (crossover) | Break-even point — Spark overhead ≈ join savings |
| 10× | 275K | 153 | 60 | **2.6×** | Partition pruning + pre-join start dominating |
| 25× | 687K | 373 | 73 | **5.1×** | Gap widens — Python is O(n), Gold is O(sub-linear via file-skipping) |
| 50× | 1.37M | 733 | 87 | **8.4×** | Gold stays sub-100ms; Python nears 1 second per query |

**Key takeaway for the audience:** At small scale (< 3×), the conventional Python approach is fine and the lakehouse overhead isn't justified. The bet is that datasets grow — and they always do. At 50× (~1.4M rows, still modest by production standards), the Gold table is 8.4× faster. At real KAIST production volumes (millions of frames across months of collection), the gap will be larger. **The lakehouse is infrastructure investment that pays off at scale.**

### Supplementary — Iceberg Feature Experiments

**Figure:** `paper/figures/supplementary_benchmarks.png`

| Experiment | What It Tests | Result | ML Implication |
|------------|--------------|--------|----------------|
| **Partition Pruning** | I/O elimination via partition-aligned filters | `camera_name` filter → **83% files skipped**. Add `clip_id` → **98.4% skipped** | "I only need front-camera data for clip X" → engine reads 1.6% of files |
| **Temporal Replay** | Pre-sorted data vs. explicit `ORDER BY` at query time | **1.8× faster** with pre-sorted Silver/Gold | Video transformers get frames in temporal order without explicit sort |
| **Column-Level Metrics** | Per-file min/max statistics for range predicates | **4.8× speedup** for narrow time-range query | "Give me a 5-second window" → engine checks each file's timestamp range, skips files outside the window |
| **Time Travel (Snapshot)** | Reproduce exact dataset after new data arrives | Write 23,150 rows → record snapshot → append 23,150 more → read at snapshot → returns exactly **23,150** ✓ | Pin dataset version before training → reproduce exact data months later. Critical for paper submissions and regulatory compliance. |

**Speaker notes:** "I want to be upfront about what we measured and what we didn't. We measured data assembly — the I/O step, not model training. The reason: data assembly is the step that scales with dataset size and is common to every model. We did NOT run BEVFormer end-to-end — that requires GPU allocation and model integration, which is on the roadmap. The benchmark queries aren't arbitrary SQL — each one maps to a specific training operation. Object detection's `SELECT * FROM camera_annotations WHERE camera_name = 'CAM_FRONT'` is exactly what `nuScenes.get_sample_data()` does internally, just pre-materialized. The scalability chart is the most important result: at 50× scale, Gold stays under 100ms while Python scripts hit 733ms. That's the difference between data loading being invisible in your epoch time and data loading being a bottleneck."

**Anticipated Q&A for this slide:**

> **Q: "Your benchmark uses `df.count()` — a full table scan. A real DataLoader does random access by index. How do these results transfer?"**
> A: Fair point. `df.count()` measures total scan + join time, which is the worst case. A real DataLoader would do batched reads (e.g., 32 samples at a time), which benefits even more from partition pruning — if all 32 samples are from `CAM_FRONT`, only the `CAM_FRONT` partition is read. The full-scan speedup (2–3×) is a conservative lower bound on the per-batch speedup, because batched reads with partition-aligned filters skip even more data. We plan to validate this with a real PyTorch DataLoader integration next quarter.

> **Q: "At small scale, Python is faster. We're not at 50× scale yet. Why invest in this now?"**
> A: Two reasons. First, the crossover is at 3× — quite small. The KAIST E2E dataset will be far larger than 3× nuScenes-mini. Second, the lakehouse provides benefits beyond raw speed: schema enforcement catches corrupt data at ingest, validation catches broken references and non-unit quaternions, and time travel lets you pin dataset versions. These are worth having even at 1× scale.

> **Q: "How are you sure this query pattern matches what we'll actually use in training?"**
> A: We derived the queries from the actual nuScenes SDK access patterns (the most-used AD dataset API). `camera_annotations` maps to `NuScenes.get_sample_data()` + `get_boxes()` + calibration lookup. `lidar_with_ego` maps to loading a KITTI-style sequence. `sensor_fusion_frame` maps to assembling a full multi-modal sample. If your training code uses a different access pattern, we can build a different Gold table — the pipeline supports it.

---

## Slide 4 — Status, Honest Limitations & Next Steps

> *"What works today, what we know doesn't work yet, and what needs decisions from this group."*

**Layout:** Top band: status matrix. Upper-middle: known limitations (proactive honesty). Lower-middle: cross-team questions. Bottom: next steps.

### Top Band — What's Operational Today

| Deliverable | Status | Evidence |
|-------------|--------|---------|
| Full medallion pipeline (Bronze → Silver → Gold → Validate) | ✅ Operational | ~24 s end-to-end on simulated data, all 20 checks pass |
| KAIST 3-level schema (14 entity types, 4 geometric structs) | ✅ Implemented | Validated on both KAIST-simulated and nuScenes-mini datasets |
| 5-service dev/test environment (Docker Compose) | ✅ Deployed | Spark, Polaris, MinIO (Ceph stand-in), Trino, Superset. Single `docker compose up`. |
| Benchmark suite (5 experiments, reproducible) | ✅ Complete | All results in `benchmark_results.json` + `scalability_results.json` |
| nuScenes cross-validation | ✅ Complete | Pipeline validated on public nuScenes v1.0-mini (7 tables). Confirms the pipeline generalizes beyond the KAIST schema. |

### Upper-Middle Band — Known Limitations (Proactive)

We know these limitations exist. Listing them explicitly because they will come up in questions.

| Limitation | Severity | Our Assessment | Mitigation Plan |
|-----------|----------|----------------|-----------------|
| **Benchmarks measure data assembly, not E2E training time** | Medium | Data assembly is the I/O-bound step that scales with dataset size. But we haven't quantified the actual training-time impact. The improvement will be smaller than the data assembly speedup because GPU compute time is unchanged. | End-to-end training benchmark with BEVFormer planned for next quarter (needs GPU allocation). |
| **Tested on simulated data, not real KAIST data** | High | Simulated data has realistic schema and scale (~140K annotations) but synthetic values. Real data may have edge cases (missing fields, corrupt timestamps, unexpected formats) that the pipeline hasn't encountered. | Blocked on real data delivery from collection team. Pipeline is designed for this — Bronze schema enforcement will catch type mismatches; validation catches integrity violations. |
| **Single-node dev environment only** | Medium | Docker Compose runs on one machine. Production KAIST data at full scale will use multi-node Spark on K8s with Ceph storage. | Architecture is designed for this — Spark, Polaris, and Ceph are all distributed systems. Production deployment is the next phase after pipeline design is finalized. |
| **`df.count()` benchmark, not batched DataLoader** | Low | `df.count()` is a conservative metric (full scan). Batched DataLoader reads with partition-aligned filters would show equal or better speedups. | PyTorch DataLoader integration planned. |
| **3 of 14 schemas are placeholders** (occupancy, motion, session_ego_motion) | Medium | These tables exist in Bronze but are skipped in Silver/Gold because the actual data format isn't finalized. | Waiting on field definitions from collection/annotation teams. Pipeline supports adding new Silver/Gold tables without affecting existing ones. |
| **Python faster at small scale** (< SF 3) | Low | Expected — Spark has JVM startup and query planning overhead. The crossover at 3× is reasonable, and production datasets will be much larger. | Not a problem — it's a characteristic of the architecture, not a bug. |

### Lower-Middle Band — Cross-Team Questions

These are decisions that affect other teams and need joint resolution.

| Question | Why It Matters | Options | Who Decides |
|----------|---------------|---------|-------------|
| **Schema change workflow** | Currently, a schema change requires updating ~235 hardcoded column references across 5 pipeline files. Unsustainable if schema evolves frequently. | **(a)** Schema team sends spec (DBML/Avro), we implement. **(b)** Invest in auto-generation tooling. **(c)** Co-develop in shared codebase. | Schema team + us |
| **End-to-end training validation** | We can measure data I/O improvement, but validating actual epoch-time impact requires running a model. This would significantly strengthen the evidence. | **(a)** We run it (need GPU allocation + model code). **(b)** An ML team runs it using our Gold tables as input. **(c)** Defer. | ML teams + us |
| **Placeholder schemas** (occupancy, motion, session_ego_motion) | 3 of 14 tables have placeholder field definitions. Silver layer skips them. | Need actual field definitions from collection team. | Data collection team |
| **Production deployment timeline** | Pipeline design is being finalized in the dev environment (Docker + MinIO). Production deployment to the Ceph cluster via K8s is the next phase. | When is the design considered complete enough to deploy? What validation should we run on Ceph before going live? | Infrastructure team + us |
| **Pre-Bronze retention policy** | Bronze preserves raw JSON immutably (audit trail). At production scale on Ceph, this doubles storage usage. | **(a)** Keep indefinitely (safest). **(b)** Delete after validated ingest (saves space). **(c)** Archive to cold tier after N days. | Project-wide policy |

### Bottom Band — Planned Next Steps

| Next Step | Dependency | Timeline Estimate |
|-----------|-----------|------------------|
| End-to-end training benchmark (BEVFormer epoch time) | GPU allocation + model code | Next quarter, pending resource decision |
| Real KAIST data validation | Real data delivery from collection team | As soon as data is available |
| PyTorch DataLoader integration | None (can proceed now) | Next quarter |
| Schema auto-generation tooling (DBML → PySpark StructType) | Decision on schema workflow | If option (b) is chosen |
| Production deployment to Ceph + K8s | Pipeline design finalization | After real data validation + design sign-off |
| Streaming ingestion (Kafka → Iceberg) | Architecture supports it; implementation not started | Q3–Q4 2026 |

**Speaker notes:** "I want to close with honesty about what we DON'T know yet. The biggest gap is that we haven't run a real training loop against Gold tables — we've measured data assembly but not actual epoch time. We believe the improvement is significant because data assembly is the I/O bottleneck, but we need GPU access to prove it. If one of the ML teams is interested in running BEVFormer or CenterPoint against our Gold tables, we can set up the data access in an afternoon — the tables are queryable via SQL right now. The other big gap is real data: everything so far runs on simulated data with realistic schema but synthetic values. We're ready for real data as soon as it's available, and the validation framework is specifically designed to catch the edge cases that real data will inevitably have. One note on deployment: everything you've seen today runs in our local Docker test environment with MinIO standing in for Ceph. The pipeline code uses standard S3 API — once we're satisfied with the design, deploying to the Ceph cluster via Kubernetes is a configuration change, not a code change."

**Anticipated Q&A for this slide:**

> **Q: "You're asking us to trust benchmarks on simulated data. When will you have real numbers?"**
> A: As soon as the collection team delivers real KAIST data. The pipeline is ready — the simulated data has the exact same schema and realistic cardinalities. The main risk with real data is format surprises (unexpected NULLs, inconsistent units), and that's exactly what the Bronze schema enforcement and 20-check validation are designed to catch.

> **Q: "What happens if the schema changes significantly?"**
> A: Today, painfully — ~235 column references to update across 5 files. That's why schema workflow is our top cross-team question. Our recommendation is auto-generation: the schema team defines tables in a format like DBML, and we generate PySpark schemas + Silver/Gold logic from it. This reduces schema change from "2 days of careful editing" to "update one spec file, regenerate."

> **Q: "Can our training code use this today, or is it vaporware?"**
> A: The pipeline works today in the dev environment. `docker compose up`, run the pipeline, and the Gold tables are queryable from Trino (`SELECT * FROM kaist_gold.camera_annotations WHERE camera_name = 'CAM_FRONT'`). You can also read them directly from PySpark. What's NOT built yet is a PyTorch `Dataset` wrapper that reads from Iceberg natively — currently you'd read via Spark and convert to pandas/numpy, or export to Parquet. The native DataLoader integration is on the roadmap. Once we deploy to Ceph + K8s, the same queries work against the production cluster.

> **Q: "When does this move from Docker to the real cluster?"**
> A: Once the pipeline design is finalized — meaning the schema, partitioning strategy, Gold table definitions, and validation rules are stable. The code already uses S3 API throughout, so deploying to Ceph is a configuration change (endpoint + credentials). The Kubernetes manifests for Spark and Polaris are standard; the main work is integration testing on the actual cluster.

---

## Design & Formatting Guidelines

**General:**
- **Poster-density format.** Each slide is a self-contained reference sheet, not a traditional bullet-point presentation slide. Design each as a dense quadrant/band layout that can be read as a standalone document.
- Use a clean, modern slide template (dark header bar, white body)
- Consistent color coding: **Gold = amber/orange (#E8A838)**, **Silver = steel blue (#6C8EBF)**, **Bronze = warm gray**, **Python baseline = muted green**
- All charts use the pre-generated figures from `paper/figures/`
- Slide numbers in footer; project name and date in footer

**Typography (adjusted for density):**
- Slide titles: 24–28 pt, bold
- Section sub-headers within a slide: 18–20 pt, bold
- Body text: 14–16 pt
- Table text: 11–13 pt (dense tables are expected and acceptable)
- Speaker notes: not displayed, for presenter reference

**Figure placement:**
- Figures are embedded inline within their section band — smaller than in a traditional deck
- Multiple figures per slide is normal (Slide 1 has 2, Slide 2 has 1, Slide 3 has 3)
- No figure borders; figures have built-in white backgrounds

**Slide dimensions:**
- Consider 16:9 widescreen for maximum horizontal space
- If content overflows, widen tables rather than cutting content — information density is the design goal

**Transitions:**
- None. These slides are reference-grade — no animation needed.
