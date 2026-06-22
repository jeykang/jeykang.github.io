# Data Lakehouse Codebase Report

> **Date:** March 11, 2026
> **Scope:** Full technical inventory of the NetAI Digital Twin Iceberg Lakehouse
> **Audience:** Infrastructure/hardware team

---

## 1. Executive Summary

This repository implements an **Apache Iceberg data lakehouse** for multi-modal autonomous driving datasets. The system ingests raw sensor data (camera, LiDAR, radar, annotations, calibration, ego-pose) through a **3-layer medallion architecture** (Bronze → Silver → Gold), producing ML-ready pre-joined tables optimized for specific training workloads.

**Bottom-line performance numbers:**
- **2–3.2× query speedup** (Gold vs. Silver JOINs) on three representative AD workloads
- **Constant ~33–42 ms Gold latency** at 1000× scale factor (23.4M rows), versus 4.7 s for Python baseline and 1.2 s for Silver JOINs
- **98.4% file pruning** with combined partition + clip filters on camera tables
- **20 automated validation checks** gate every pipeline run (PK/FK integrity, quaternion norms, timestamp sanity, row-count assertions)

**Current state:** Fully operational in Docker Compose (local dev/test). Production target is **Ceph object storage + Kubernetes**.

---

## 2. Infrastructure Stack

### 2.1 Docker Compose Services (Development)

| Service | Image | Purpose | Ports |
|---------|-------|---------|-------|
| **Polaris** | `apache/polaris:latest` | Iceberg REST API Catalog (metadata) | 8181 (API), 8182 (health) |
| **MinIO** | `quay.io/minio/minio:2025-09-07T16-13-09Z` | S3-compatible object storage (dev stand-in for Ceph) | 9000 (API), 9001 (console) |
| **Spark** | `tabulario/spark-iceberg:3.5.5_1.8.1` | Distributed ETL & data processing | 4040–4045 (Web UI) |
| **Trino** | `trinodb/trino:479` | Interactive SQL query engine | 8080 |
| **Superset** | Custom build (`./superset`) | BI dashboards | 8088 |
| **PostgreSQL 15** | `postgres:15` | Superset metadata store | 5432 (internal) |
| **Redis 7** | `redis:7` | Superset cache | 6379 (internal) |
| **setup_bucket** | `quay.io/minio/mc:latest` | Auto-creates `spark1` bucket on startup | — |
| **polaris-setup** | `alpine/curl` | Bootstraps Polaris catalog via REST API | — |

All services connected via `lakehouse_net` Docker bridge network. Internal DNS enables service-to-service comms (e.g., `http://polaris:8181`, `http://minio:9000`).

**Startup:** `./start.sh` → creates directories, sets permissions, runs `docker compose up -d`. Cold start ~60 s, warm ~30 s.

### 2.2 Software Versions

| Component | Version |
|-----------|---------|
| Apache Spark | 3.5.5 |
| Scala | 2.12.18 |
| Apache Iceberg | 1.8.1 (bundled JAR: `iceberg-spark-runtime-3.5_2.12-1.8.1.jar`) |
| Polaris (Iceberg REST Catalog) | latest |
| MinIO | 2025-09-07T16-13-09Z |
| Trino | 479 |
| Apache Superset | latest (custom Dockerfile) |
| PostgreSQL | 15 |
| Redis | 7 |

### 2.3 Storage Architecture

```
┌────────────────────────────────────────────────────┐
│              Polaris REST Catalog                   │
│    (Namespaces, table metadata, snapshots)          │
└──────────────────┬─────────────────────────────────┘
                   │
         Iceberg Table Format (v2)
         (Manifest files, data files)
                   │
┌──────────────────┴─────────────────────────────────┐
│         S3-Compatible Object Storage               │
│   Dev:  MinIO (localhost:9000)                     │
│   Prod: Ceph RGW (S3 gateway)                     │
│                                                    │
│   Bucket: spark1                                   │
│   Layout: s3://spark1/iceberg/{namespace}/{table}/ │
└────────────────────────────────────────────────────┘
```

### 2.4 Catalog & Credentials Configuration

**Polaris catalog:**
- URI: `http://polaris:8181/api/catalog`
- Catalog name: `lakehouse_catalog`
- Auth: OAuth2 client credentials (`root:s3cr3t`, realm `POLARIS`, scope `PRINCIPAL_ROLE:ALL`)

**S3 (MinIO):**
- Endpoint: `http://minio:9000`
- Credentials: `minioadmin:minioadmin`
- Region: `us-east-1`
- Path-style access: enabled (required for MinIO & Ceph RGW)

**Trino connector** (`trino/etc/catalog/iceberg.properties`):
- `connector.name=iceberg`, type `rest`
- Connects to Polaris at `http://polaris:8181/api/catalog`
- S3 via native filesystem (`fs.native-s3.enabled=true`)
- Same MinIO/Ceph credentials

### 2.5 Production Deployment Target

```
┌──────────────────────────────────────────┐
│          Kubernetes Cluster              │
│  ┌────────────────────────────────────┐  │
│  │ Spark on K8s (Operator)           │  │
│  │  - Driver pod                     │  │
│  │  - Executor pods (auto-scaled)    │  │
│  ├────────────────────────────────────┤  │
│  │ Polaris Service                   │  │
│  ├────────────────────────────────────┤  │
│  │ Trino Service                     │  │
│  ├────────────────────────────────────┤  │
│  │ Superset Deployment               │  │
│  └────────────────────────────────────┘  │
└──────────────────┬───────────────────────┘
                   │ S3 API
           ┌───────┴───────┐
           │   Ceph RGW    │
           │  (S3 Gateway) │
           └───────┬───────┘
           ┌───────┴───────┐
           │ RADOS Storage │
           │   Cluster     │
           └───────────────┘
```

**What changes:** S3 endpoint URL + credentials (config-only, no code changes). Kubernetes secrets replace `.env` file. HPA for Spark executors.

**What stays the same:** All pipeline code, Spark SQL queries, Iceberg table format, Polaris catalog API.

---

## 3. Data Model: KAIST 3-Level Hierarchy

### 3.1 Entity Hierarchy

```
Session (1 per recording campaign)
  └── Clip (N driving sequences)
       ├── Frame (N synchronized sensor snapshots)
       │    ├── Camera (N cameras per frame: CAM_FRONT, CAM_BACK, etc.)
       │    ├── Lidar (1 per frame)
       │    ├── Radar (N radars per frame)
       │    ├── DynamicObject (N 3D bbox annotations per frame)
       │    ├── Occupancy (1 occupancy grid per frame)
       │    ├── Motion (1 motion vector per frame)
       │    └── EgoMotion (1 ego vehicle pose per frame)
       ├── Calibration (M per sensor type)
       └── HDMap (geographic metadata)
```

### 3.2 14 Source Tables

| Table | Purpose | Key Columns | Row Count (simulated) |
|-------|---------|-------------|----------------------|
| `session` | Recording campaign | session_id, clip_id_list | 1 |
| `clip` | Driving sequence | clip_id, session_id, frame_id_list | 14 |
| `frame` | Synchronized snapshot | frame_id, clip_id, frame_idx | 3,935 |
| `camera` | Camera observations | frame_id, camera_name, filename | 140,080 |
| `lidar` | LiDAR point clouds | frame_id, filename, sensor_timestamp | 3,935 |
| `radar` | Radar detections | frame_id, radar_name, filename | 19,675 |
| `calibration` | Sensor extrinsics/intrinsics | clip_id, sensor_name | 70 |
| `category` | Object type labels | category | 23 |
| `dynamic_object` | 3D bounding boxes | frame_id, boxes_3d, category | 23,150 |
| `occupancy` | Occupancy grids | frame_id, occupancy_data | 3,935 |
| `motion` | Motion vectors | frame_id, motion_data | 3,935 |
| `ego_motion` | Ego vehicle pose/frame | frame_id, translation, rotation | 3,935 |
| `session_ego_motion` | Session-level ego summary | session_id, translation, rotation | 1 |
| `hdmap` | Geographic/map context | clip_id, city, site | 14 |

### 3.3 Geometric Types (PySpark StructType)

Defined in `kaist_ingestion/schemas.py`:

| Type | Fields | Used By |
|------|--------|---------|
| **SE3Type** | translation `[x,y,z]`, rotation `[qw,qx,qy,qz]` | calibration, ego_motion |
| **QuaternionType** | qw, qx, qy, qz (all DoubleType) | ego_motion, session_ego_motion |
| **Box3DType** | center_x/y/z, length, width, height, yaw | dynamic_object |
| **Translation3DType** | x, y, z | ego_motion |

---

## 4. Medallion Pipeline Architecture

### 4.1 Overview

```
user_data/kaist-simulated/  (14 JSON files)
        │
        ▼
┌─ BRONZE ─────────────────────────────────────────┐
│  ingest_bronze.py                                │
│  Strict PySpark StructType schema enforcement    │
│  14 Iceberg tables (1:1 from source)             │
│  Immutable raw data preservation                 │
│  → s3://spark1/iceberg/kaist_bronze/{table}/     │
└──────────────────────────────────────────────────┘
        │
        ▼
┌─ SILVER ─────────────────────────────────────────┐
│  transform_silver.py                             │
│  Partitioning by access patterns                 │
│  Sort orders for temporal locality               │
│  Column-level min/max statistics                 │
│  11 tables (occupancy, motion, session_ego skip) │
│  → s3://spark1/iceberg/kaist_silver/{table}/     │
└──────────────────────────────────────────────────┘
        │
        ▼
┌─ GOLD ───────────────────────────────────────────┐
│  build_gold.py                                   │
│  Pre-joined denormalized tables per ML workload  │
│  3 tables: camera_annotations,                   │
│            lidar_with_ego,                       │
│            sensor_fusion_frame                   │
│  Zero-join queries at consumption time           │
│  → s3://spark1/iceberg/kaist_gold/{table}/       │
└──────────────────────────────────────────────────┘
        │
        ▼
┌─ VALIDATION ─────────────────────────────────────┐
│  validators.py                                   │
│  20 automated checks (PK, FK, quaternion norms,  │
│  timestamps, row counts)                         │
│  Pipeline halts on any CRITICAL failure           │
└──────────────────────────────────────────────────┘
```

**Entry point:** `kaist_runner.py` (CLI orchestrator)
```bash
python kaist_runner.py all          # Full pipeline
python kaist_runner.py bronze       # Bronze only
python kaist_runner.py silver       # Silver only
python kaist_runner.py gold         # Gold only
python kaist_runner.py validate     # Validation only
```

### 4.2 Bronze Layer — Raw Ingestion (`ingest_bronze.py`, 305 lines)

- `BronzeIngester` class: loads each JSON file, enforces strict PySpark `StructType` schema, writes as Iceberg format-version=2
- 14 table mappings (e.g., `camera` ← `camera.json`)
- Type mismatches cause immediate hard failure — no permissive mode
- Returns per-table row counts for auditing

### 4.3 Silver Layer — Physical Optimization (`transform_silver.py`, 436 lines)

`SilverTransformer` class with per-table optimization configs:

| Table | Partition Key(s) | Sort Order | Metrics Columns |
|-------|-----------------|------------|-----------------|
| `camera` | `camera_name, clip_id` | `clip_id, frame_id, sensor_timestamp` | `sensor_timestamp, clip_id, frame_id, camera_name` |
| `lidar` | `clip_id` | `clip_id, frame_id, sensor_timestamp` | `sensor_timestamp, clip_id, frame_id` |
| `radar` | `radar_name, clip_id` | `clip_id, frame_id, sensor_timestamp` | `sensor_timestamp, clip_id, frame_id, radar_name` |
| `frame` | `clip_id` | `clip_id, frame_idx` | `clip_id, frame_idx` |
| `clip` | `session_id` | — | `session_id, clip_id` |
| `dynamic_object` | `clip_id` | `clip_id, frame_id` | `clip_id, frame_id` |
| `calibration` | `clip_id, sensor_name` | — | `clip_id, sensor_name` |
| `ego_motion` | `clip_id` | `clip_id, frame_id` | `clip_id, frame_id` |
| `session` | — | — | — |
| `category` | — | — | — |
| `hdmap` | — | — | — |

**Iceberg-native optimizations applied:**
- `WRITE ORDERED BY` for sort orders (maintained on all future writes)
- `write.metadata.metrics.column.{col} = "full"` for min/max statistics
- `write.distribution-mode = hash` (partition-aligned file distribution)
- Target file size: 128 MB
- Snapshot retention: 10 min snapshots, 7-day max age

### 4.4 Gold Layer — ML-Ready Tables (`build_gold.py`, 416 lines)

`GoldTableBuilder` class with 3 specialized build methods:

#### `camera_annotations` — Object Detection Training
- **Joins:** camera ⋈ frame ⋈ clip ⋈ calibration ⋈ dynamic_object ⋈ hdmap (6 tables)
- **Partition:** `camera_name` (6-way sensor filtering → 83% pruning)
- **Sort:** `clip_id, frame_idx` (temporal locality)
- **Output columns:** frame_id, clip_id, session_id, frame_idx, sensor_timestamp, camera_name, filename, extrinsics, camera_intrinsics, annotations (array of `{boxes_3d, category}`), city, site, date

#### `lidar_with_ego` — SLAM/Localization
- **Joins:** lidar ⋈ ego_motion ⋈ calibration (3 tables)
- **Partition:** `clip_id` (bucketed)
- **Sort:** `sensor_timestamp` (temporal sequencing)
- **Output columns:** frame_id, clip_id, filename, sensor_timestamp, ego_translation, ego_rotation, extrinsics

#### `sensor_fusion_frame` — Multi-Modal Perception
- **Joins:** frame ⋈ camera ⋈ lidar ⋈ radar ⋈ dynamic_object (5 tables + 3 aggregations)
- **Partition:** `clip_id` (bucketed)
- **Output columns:** frame_id, clip_id, frame_idx, cameras (array), lidar_filename, lidar_timestamp, radars (array), annotations (array)

### 4.5 Validation Framework (`validators.py`, 365+ lines)

20 automated checks, split by severity:

| Category | Count | Severity | Examples |
|----------|-------|----------|---------|
| PK uniqueness | 6 | CRITICAL | frame_id unique in frame, camera PK on (frame_id, camera_name) |
| FK integrity | 4 | CRITICAL | clip.session_id → session, camera.frame_id → frame |
| Quaternion norms | 2 | WARNING | $\|q\| = 1.0 \pm 0.01$ in ego_motion, session_ego_motion |
| Timestamp validity | 4 | WARNING | All sensor_timestamp ≥ 0 |
| Row count assertions | 4 | CRITICAL | Gold row counts match expected from source joins |

Pipeline halts on any CRITICAL failure — corrupted data never reaches Gold.

### 4.6 Configuration (`config.py`, 250 lines)

**Key configuration classes:**

| Class | Purpose | Key Parameters |
|-------|---------|---------------|
| `StorageConfig` | S3 connection | endpoint, access_key, secret_key, bucket (`spark1`), path_style_access |
| `CatalogConfig` | Polaris REST | uri, warehouse (`lakehouse_catalog`), credential, OAuth2 scope/server |
| `KAISTConfig` | Dataset params | source_path, namespace_bronze/silver/gold, target_file_size (128 MB), shuffle_partitions (200), snapshot retention |
| `PipelineConfig` | Composite | Combines above; `build_spark_session()`, `create_namespaces()`, `apply_ad_table_optimizations()` |

### 4.7 Data Simulator (`simulate_from_nuscenes.py`)

Generates synthetic KAIST-format data from nuScenes-mini when real KAIST data is unavailable:
- nuScenes `scene` → KAIST `Session`
- nuScenes `sample` → KAIST `Frame`
- nuScenes `sample_data` → KAIST `Camera/Lidar/Radar`
- nuScenes `sample_annotation` → KAIST `DynamicObject`
- nuScenes `calibrated_sensor` → KAIST `Calibration`

---

## 5. Benchmark Results

### 5.1 Three-Workload Benchmark (`benchmarks/ad_workload_benchmark.py`, 1001 lines)

Measures query latency for three realistic AD ML workloads. Methodology: 3 JVM warmup queries, then 2 untimed + 5 timed runs, median reported.

| Workload | Gold (ms) | Silver JOIN (ms) | Speedup | Rows |
|----------|-----------|-----------------|---------|------|
| Object Detection | **79** | 255 | **3.2×** | 23,150 |
| SLAM/Localization | **64** | 138 | **2.2×** | 389 |
| Multi-Modal Fusion | **49** | 99 | **2.0×** | 389 |

### 5.2 KAIST Scalability Benchmark (`benchmarks/kaist_scalability_benchmark.py`, SF 1–1000×)

Fact tables replicated synthetically; reference tables at 1×. Query: "Assemble front-camera images + annotations."

| Scale Factor | Rows | Python (ms) | Silver JOIN (ms) | Gold (ms) | Gold vs Python |
|:--:|:--:|:--:|:--:|:--:|:--:|
| 1× | 2,342 | 1.8 | 277 | 42 | 0.04× (Python wins) |
| 100× | 234,200 | 301 | 341 | 41 | **7.3×** |
| 500× | 1,171,000 | 2,041 | 700 | 31 | **65.9×** |
| 1000× | 2,342,000 | 4,661 | 1,203 | 33 | **139.6×** |

**Key finding:** Gold latency is effectively constant (~30–42 ms) across all scale factors. Iceberg partition pruning restricts scans to a single partition regardless of total data volume. Silver JOINs grow sub-linearly (Spark optimizer). Python grows linearly.

### 5.3 Supplementary Experiments

| Experiment | Result | Significance |
|-----------|--------|-------------|
| Partition Pruning | camera_name + clip_id filter → **98.4% data skipped** | Single-partition reads regardless of table size |
| Temporal Replay | Pre-sorted read 1.8× faster than runtime ORDER BY | Sorted Iceberg files eliminate query-time sort |
| Column Metrics | 4.8× speedup for narrow time-range filter | Per-file min/max metadata enables range predicate pushdown |
| Time Travel | Snapshot pinning preserves exact row counts across writes | Reproducible training datasets via VERSION AS OF |

---

## 6. nuScenes Cross-Validation Experiment

Separate validation track using the public **nuScenes v1.0-mini** dataset to confirm the pipeline generalizes.

### 6.1 Ingestion (`python-scripts/ingest_nuscenes_mini.py`)

13 core nuScenes tables ingested into `iceberg.nuscenes` namespace.

### 6.2 Scalability Benchmark (`nuscenes_experiment/scalability_benchmark.py`, SF 1–50×)

| Scale | Python (ms) | Gold (ms) | Speedup |
|------:|------------:|----------:|--------:|
| 1× | 15 | 42 | 0.4× |
| 3× | 48 | 50 | 1.0× (crossover) |
| 10× | 153 | 60 | **2.6×** |
| 50× | 733 | 87 | **8.4×** |

### 6.3 Experiment Artifacts

- `01_python_code_no-def/` — Standalone Python scripts (baseline, silver, gold)
- `02_ipynb_exp_no-def/` — Simplified Jupyter notebooks
- `03_ipynb_exp_yes-def/` — Full notebooks with definitions + analysis
- `superset_chart_recipes.md` — 30+ SQL templates for Superset dashboards

---

## 7. Query Engine: Trino

**Connector:** Iceberg REST catalog via Polaris **(trino/etc/catalog/iceberg.properties)**

```sql
-- Available namespaces
USE iceberg.kaist_gold;
SELECT * FROM camera_annotations WHERE camera_name = 'CAM_FRONT';

USE iceberg.kaist_silver;
SELECT * FROM camera WHERE clip_id = 'clip_xxx' AND camera_name = 'CAM_BACK';

USE iceberg.nuscenes;
SELECT COUNT(*) FROM sample_annotation;
```

---

## 8. BI Layer: Superset

- Custom Dockerfile in `superset/` (base: `apache/superset:latest`)
- PostgreSQL 15 metadata backend, Redis 7 cache
- Port 8088; auto-initialized with admin user
- Connects to Trino at port 8080 for querying all Iceberg tables
- Custom visualization plugin: `superset-plugin-chart-databahn-pipelines/` (TypeScript + React, for pipeline monitoring charts)

---

## 9. Data Inventory (MinIO `spark1` Bucket)

```
spark1/
├── kaist_bronze/       14 Iceberg tables (raw)
├── kaist_silver/       11 Iceberg tables (optimized)
├── kaist_gold/
│   ├── camera_annotations/                 (partitioned by camera_name)
│   ├── camera_annotations_unpartitioned/   (benchmark comparison)
│   ├── lidar_with_ego/                     (partitioned by clip_id)
│   ├── sensor_fusion_frame/                (partitioned by clip_id)
│   └── time_travel_demo/                   (multi-snapshot)
├── nuscenes/           13 nuScenes-mini tables
├── nusc_exp/           Scalability experiment tables
├── nusc_scalability/   Scaled replica tables (SF 1–50)
└── kaist_scalability/  Scaled replica tables (SF 1–1000)
```

Source data mounted at `/user_data`:
- `kaist-simulated/` — 14 JSON files (simulated from nuScenes-mini)
- `nuscenes-mini/v1.0-mini/` — 13 nuScenes JSON files

---

## 10. Codebase File Inventory

### Pipeline Code (`kaist_ingestion/`)

| File | Lines | Purpose |
|------|-------|---------|
| `config.py` | ~250 | Storage, catalog, dataset config; Spark session builder |
| `schemas.py` | ~200 | PySpark StructType definitions for all 14 tables + geometric types |
| `ingest_bronze.py` | ~305 | BronzeIngester: JSON → Iceberg with strict schema enforcement |
| `transform_silver.py` | ~436 | SilverTransformer: partitioning, sorting, column metrics |
| `build_gold.py` | ~416 | GoldTableBuilder: 3 pre-joined ML-ready tables |
| `validators.py` | ~365 | 20 automated data quality checks |
| `kaist_runner.py` | ~150 | CLI orchestrator (run layers individually or all) |
| `simulate_from_nuscenes.py` | ~200 | Generate synthetic KAIST data from nuScenes-mini |

### Benchmarks (`benchmarks/`)

| File | Purpose |
|------|---------|
| `ad_workload_benchmark.py` (~1001 lines) | 3-workload + 5 supplementary experiments |
| `benchmark_results.json` | Three-workload + supplementary results |
| `kaist_scalability_benchmark.py` (~355 lines) | SF 1–1000× scaling test |
| `kaist_scalability_results.json` | 33 data points (11 per strategy) |

### nuScenes Experiment (`nuscenes_experiment/`)

| File | Purpose |
|------|---------|
| `scalability_benchmark.py` | SF 1–50× scaling on public nuScenes data |
| `scalability_results.json` | Results |
| `superset_chart_recipes.md` | 30+ SQL templates for Superset dashboards |
| `01_python_code_no-def/` | Standalone Python scripts (3 strategies) |
| `02_ipynb_exp_no-def/` | Simplified Jupyter notebooks |
| `03_ipynb_exp_yes-def/` | Full notebooks with definitions |

### Paper (`paper/`)

| File | Purpose |
|------|---------|
| `generate_figures.py` | Generates 8 publication figures at 200 DPI |
| `PAPER_OUTLINE.md` | 2-page IEEE two-column paper outline |
| `SLIDE_DECK_PLAN.md` | 4-slide presentation plan (poster-density format) |
| `figures/` | Pre-generated PNG figures |

### Infrastructure

| File | Purpose |
|------|---------|
| `docker-compose.yml` | 9 services (Polaris, MinIO, Spark, Trino, Superset, PostgreSQL, Redis, setup-bucket, polaris-setup) |
| `start.sh` | Directory creation + `docker compose up -d` |
| `example.env` | Template for credentials and configuration |
| `trino/etc/` | Trino config: node properties, JVM config, Iceberg catalog connector |
| `superset/` | Dockerfile, superset_config.py, requirements |
| `superset-plugin-chart-databahn-pipelines/` | Custom Superset visualization plugin (TypeScript/React) |

---

## 11. Known Limitations

| Limitation | Severity | Status |
|-----------|----------|--------|
| Benchmarks on simulated data, not real KAIST | High | Blocked on real data delivery |
| Single-node only (no distributed Spark tests) | Medium | By design for dev; production on K8s |
| `df.count()` benchmark (full scan, not batched DataLoader) | Low | Conservative lower bound; real DataLoader benefits more from pruning |
| 3 placeholder schemas (occupancy, motion, session_ego_motion) | Medium | Waiting on field definitions from collection team |
| No streaming ingestion | Medium | Batch-only; Kafka→Iceberg planned for Q3–Q4 2026 |
| No native PyTorch DataLoader integration | Medium | Gold tables readable via Spark→pandas; native wrapper planned |
| Python faster at small scale (< SF 3) | Low | Expected Spark overhead; crossover at 3× |
| ~235 hardcoded column references across pipeline files | Medium | Schema auto-generation tooling under consideration |

---

## 12. Production Migration Path (Docker → Ceph + K8s)

| Aspect | Dev (Current) | Prod (Target) |
|--------|--------------|---------------|
| Object storage | MinIO (single-node) | Ceph RGW (distributed RADOS) |
| Compute | Single Spark container | Spark on K8s (Operator, HPA for executors) |
| Catalog | Polaris (single container) | Polaris (K8s service) |
| Query engine | Trino (single container) | Trino (K8s deployment, worker scaling) |
| Credentials | `.env` file | Kubernetes Secrets |
| Config changes | `s3.endpoint` + credentials | Same S3 API surface |
| Code changes | **None** | All pipeline code unchanged |

**Key architectural decision:** All code uses S3 API exclusively. MinIO↔Ceph swap is a configuration change. Validated by running identical queries against both storage backends.
