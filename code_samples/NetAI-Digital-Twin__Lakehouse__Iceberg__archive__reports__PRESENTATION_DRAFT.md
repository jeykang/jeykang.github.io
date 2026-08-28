# Presentation Draft: Data Lakehouse — Infrastructure Team Update

> **Date:** March 11, 2026
> **Audience:** Infrastructure / hardware team (internal)
> **Format:** 8 slides, technical depth, 16:9 widescreen
> **Tone:** Development progress report with evidence — what's built, how it works, what it needs from infra

---

## Slide 1: Title & Context

**Title:** Data Lakehouse for Autonomous Driving — Development Status Update

**Subtitle:** NetAI Digital Twin Project · March 2026

**Key message (speaker notes):**
> This is a progress update on the data lakehouse component. We'll cover: what's running, how it's architected, the performance we're seeing, and what we need from infrastructure to move to production.

---

## Slide 2: System Architecture Overview

**Title:** Architecture: What's Running Today

**Visual:** Architecture diagram showing all Docker Compose services and their connections

**Content:**

### Development Stack (Docker Compose — 9 containers)

```
                    ┌────────────────────────────┐
                    │     Consumers              │
                    │  Trino (SQL) · Superset (BI)│
                    │  Spark (ML pipelines)       │
                    └──────────┬─────────────────┘
                               │
                    ┌──────────┴─────────────────┐
                    │     Polaris REST Catalog    │
                    │  (Iceberg metadata + ACL)   │
                    │  Port 8181                  │
                    └──────────┬─────────────────┘
                               │
                    ┌──────────┴─────────────────┐
                    │  Apache Iceberg (v2 format) │
                    │  Manifest files + data files│
                    └──────────┬─────────────────┘
                               │ S3 API
                    ┌──────────┴─────────────────┐
                    │   MinIO (dev) → Ceph (prod) │
                    │   Bucket: spark1            │
                    │   Port 9000 (API) / 9001    │
                    └────────────────────────────┘
```

### Service Inventory

| Service | Image | Role | Port |
|---------|-------|------|------|
| Polaris | `apache/polaris:latest` | Iceberg REST catalog (metadata, namespaces, table versioning) | 8181 |
| MinIO | `minio:2025-09-07` | S3-compatible object storage (Ceph stand-in) | 9000/9001 |
| Spark | `spark-iceberg:3.5.5_1.8.1` | ETL engine — runs Bronze/Silver/Gold pipeline | 4040 |
| Trino | `trino:479` | Interactive SQL over Iceberg tables | 8080 |
| Superset | Custom build | BI dashboards connected to Trino | 8088 |
| PostgreSQL 15 | `postgres:15` | Superset metadata DB | 5432 (int) |
| Redis 7 | `redis:7` | Superset cache | 6379 (int) |

**Footer note:** `docker compose up -d` — cold start ~60s. All services on `lakehouse_net` bridge network.

**Speaker notes:**
> Everything you see here runs in Docker Compose on a single machine. MinIO is our local stand-in for the Ceph cluster — same S3 API, same path-style access. The important point for infra: this entire stack talks S3. When we point it at Ceph RGW instead of MinIO, nothing changes except the endpoint URL and credentials.

---

## Slide 3: Software Stack & Versions

**Title:** Technology Stack — Versions & Dependencies

| Layer | Component | Version | Notes |
|-------|-----------|---------|-------|
| **Table Format** | Apache Iceberg | 1.8.1 | v2 spec; schema evolution, partition evolution, time travel |
| **Compute** | Apache Spark | 3.5.5 | PySpark; Iceberg runtime JAR (`iceberg-spark-runtime-3.5_2.12-1.8.1.jar`) |
| **Catalog** | Polaris | latest | REST API; OAuth2 auth; multi-tenant; manages table metadata |
| **Object Storage** | MinIO (dev) / Ceph RGW (prod) | 2025-09-07 | S3 API; path-style access; bucket `spark1` |
| **Query Engine** | Trino | 479 | ANSI SQL; Iceberg REST connector to Polaris |
| **BI** | Apache Superset | latest | PostgreSQL 15 backend; Redis 7 cache; custom plugin |
| **Language** | Scala 2.12.18 / Python 3.x | — | Spark runtime + PySpark pipeline code |

### Storage Configuration

```
S3 Endpoint:       http://minio:9000  (dev)  →  Ceph RGW endpoint (prod)
Bucket:            spark1
Path Style Access: enabled (required for MinIO + Ceph RGW)
Region:            us-east-1
Auth:              AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY
                   (dev: minioadmin/minioadmin → prod: K8s Secrets)
```

### Catalog Configuration

```
Polaris URI:       http://polaris:8181/api/catalog
Catalog Name:      lakehouse_catalog
Auth:              OAuth2 client credentials (root:s3cr3t)
Scope:             PRINCIPAL_ROLE:ALL
```

**Speaker notes:**
> Key infra dependencies: S3-compatible storage (currently MinIO, target Ceph), a REST catalog service (Polaris), and a Spark runtime. All open-source, no vendor lock-in. Iceberg v2 format is the critical piece — it's what gives us schema evolution, partition pruning, and time travel without custom code.

---

## Slide 4: Data Pipeline — Medallion Architecture

**Title:** 3-Layer Medallion Pipeline: Bronze → Silver → Gold

**Visual:** Flow diagram with data volume annotations

```
Source: 14 JSON files (user_data/kaist-simulated/)
  │
  ▼
┌─ BRONZE (Raw) ───────────────────────────────────────────┐
│  • Strict PySpark StructType schema enforcement           │
│  • 14 Iceberg tables, 1:1 from source files              │
│  • Wrong type/missing field = hard failure (no permissive)│
│  • Immutable — raw data preserved for audit/reprocessing  │
│  • Volume: ~199K total rows across 14 tables              │
│  → s3://spark1/iceberg/kaist_bronze/                      │
└──────────────────────────────────────────────────────────┘
  │
  ▼
┌─ SILVER (Optimized) ─────────────────────────────────────┐
│  • Partition by dominant access pattern per table          │
│    (camera → camera_name+clip_id, lidar → clip_id, etc.) │
│  • Sort within partitions (temporal locality)             │
│  • Column-level min/max statistics (per-file metadata)    │
│  • 11 tables (3 placeholder schemas skipped)              │
│  • Target file size: 128 MB, write mode: hash             │
│  → s3://spark1/iceberg/kaist_silver/                      │
└──────────────────────────────────────────────────────────┘
  │
  ▼
┌─ GOLD (ML-Ready) ────────────────────────────────────────┐
│  • Pre-joined denormalized tables, one per ML workload    │
│  • 3 tables:                                              │
│    - camera_annotations (6-table join → 1 table)          │
│    - lidar_with_ego (3-table join → 1 table)              │
│    - sensor_fusion_frame (5 tables + 3 aggs → 1 table)   │
│  • Zero JOINs at query time                               │
│  → s3://spark1/iceberg/kaist_gold/                        │
└──────────────────────────────────────────────────────────┘
  │
  ▼
┌─ VALIDATION (20 checks) ─────────────────────────────────┐
│  • PK uniqueness (6), FK integrity (4)                    │
│  • Quaternion norms ‖q‖ = 1.0 ± 0.01 (2)                │
│  • Non-negative timestamps (4), row count assertions (4)  │
│  • Pipeline HALTS on any CRITICAL failure                 │
└──────────────────────────────────────────────────────────┘
```

**CLI:**
```bash
python kaist_runner.py all        # Full pipeline: ~24 sec on simulated data
python kaist_runner.py bronze     # Individual layers
python kaist_runner.py validate   # Validation only
```

**Speaker notes:**
> Three layers, each with a distinct purpose. Bronze = raw data preservation and schema enforcement. Silver = physical storage optimization (partitioning, sorting, file-level statistics) — benefits every downstream query regardless of workload. Gold = pre-joined tables per ML task — eliminates runtime JOINs entirely. The validation gate is critical: non-unit quaternions, broken foreign keys, and other data quality issues are caught _before_ data reaches Gold. These errors are silent — they don't throw exceptions, they just produce wrong training data.

---

## Slide 5: Data Model & Storage Layout

**Title:** Data Model: 14 Entity Types, 3 Namespaces

### KAIST 3-Level Hierarchy

```
Session (1) ── recording campaign
  └── Clip (14) ── driving sequence
       ├── Frame (3,935) ── synchronized sensor snapshot
       │    ├── Camera (140,080) ── 6 cameras × frames
       │    ├── Lidar (3,935) ── 1 per frame
       │    ├── Radar (19,675) ── 5 radars × frames
       │    ├── DynamicObject (23,150) ── 3D bboxes
       │    ├── EgoMotion (3,935) ── vehicle pose
       │    ├── Occupancy (3,935) ── grids [placeholder]
       │    └── Motion (3,935) ── vectors [placeholder]
       ├── Calibration (70) ── extrinsics/intrinsics
       └── HDMap (14) ── geographic metadata
```

### S3 Storage Layout (MinIO bucket `spark1`)

```
spark1/
├── kaist_bronze/           14 Iceberg tables (raw, immutable)
│   ├── camera/             140,080 rows
│   ├── lidar/              3,935 rows
│   ├── dynamic_object/     23,150 rows
│   └── ... (11 more)
├── kaist_silver/           11 Iceberg tables (partitioned + sorted)
│   ├── camera/             partitioned by (camera_name, clip_id)
│   ├── lidar/              partitioned by clip_id
│   └── ...
├── kaist_gold/             3 ML-ready tables
│   ├── camera_annotations/         partitioned by camera_name
│   ├── lidar_with_ego/             partitioned by clip_id
│   └── sensor_fusion_frame/        partitioned by clip_id
├── nuscenes/               13 nuScenes-mini tables (cross-validation)
├── nusc_scalability/       Scaled replicas (SF 1–50)
└── kaist_scalability/      Scaled replicas (SF 1–1000)
```

### Geometric Types (in schemas.py)

| Type | Fields | Used By |
|------|--------|---------|
| SE3 | translation `[x,y,z]` + rotation `[qw,qx,qy,qz]` | Calibration, EgoMotion |
| Quaternion | qw, qx, qy, qz | EgoMotion |
| Box3D | center_x/y/z, length, width, height, yaw | DynamicObject |

**Speaker notes:**
> The data model follows the KAIST E2E dataset schema: Session → Clip → Frame, with sensor observations and annotations hanging off frames. Currently running on simulated data (~200K rows total) — real KAIST data ingestion will follow the same pipeline with zero code changes. The bucket layout shows Bronze/Silver/Gold as separate Iceberg namespaces in the same S3 bucket. For infra planning: at production scale, the camera table alone will be the largest — it's the number of cameras times the number of frames.

---

## Slide 6: Benchmark Results

**Title:** Performance: Gold vs Silver vs Python Baseline

### Experiment 1 — Three AD Workloads (KAIST simulated data)

| Workload | ML Task | Gold (ms) | Silver JOIN (ms) | Speedup |
|----------|---------|-----------|-----------------|---------|
| **Object Detection** | BEVFormer, DETR3D | **79** | 255 | **3.2×** |
| **SLAM / Localization** | ORB-SLAM, LIO-SAM | **64** | 138 | **2.2×** |
| **Sensor Fusion** | BEVFusion, UniAD | **49** | 99 | **2.0×** |

*Methodology: 3 JVM warmup queries, 2 untimed + 5 timed runs, median reported. Same WHERE clause for Gold and Silver.*

### Experiment 2 — Scalability (KAIST SF 1–1000×)

| Scale Factor | Total Rows | Python (ms) | Silver JOIN (ms) | Gold (ms) | Gold vs Python |
|:--:|:--:|:--:|:--:|:--:|:--:|
| 1× | 2,342 | 1.8 | 277 | 42 | 0.04× |
| 100× | 234,200 | 301 | 341 | 41 | **7.3×** |
| 500× | 1,171,000 | 2,041 | 700 | 31 | **65.9×** |
| **1000×** | **2,342,000** | **4,661** | **1,203** | **33** | **139.6×** |

### Key Insight: Gold Latency Is Constant

```
Gold @ SF 1:    42 ms
Gold @ SF 1000: 33 ms   ← effectively unchanged
                         (Iceberg partition pruning → reads 1 partition regardless of total size)

Silver @ SF 1000: 1,203 ms (36× slower than Gold)
Python @ SF 1000: 4,661 ms (140× slower than Gold)
```

### Supplementary: Iceberg Feature Impact

| Feature | Test | Result |
|---------|------|--------|
| Partition Pruning | `camera_name` + `clip_id` filter | **98.4% files skipped** |
| Temporal Sort | Pre-sorted scan vs runtime ORDER BY | **1.8× faster** |
| Column Metrics | Narrow time-range filter (5% window) | **4.8× speedup** via min/max metadata |
| Time Travel | Snapshot pinning across writes | Exact row-count reproducibility confirmed |

**Speaker notes:**
> The headline number: Gold table latency stays at ~33 ms regardless of whether the dataset has 2K or 2.3M rows. This is Iceberg partition pruning at work — the engine reads exactly one partition's data files and ignores everything else. For the infra team, this means: storage can grow linearly with data collection, but query latency stays constant for targeted ML workloads. The 140× speedup over Python at 1000× scale is the most dramatic result, but the 2–3× over Silver JOINs at current scale is the more practically relevant one — it's the difference between pre-joining tables at ingest time vs. joining them at query time.

---

## Slide 7: Production Deployment — What We Need from Infrastructure

**Title:** Moving to Production: Docker → Ceph + Kubernetes

### Current vs. Target

| Aspect | Dev (Current) | Prod (Target) | Change Required |
|--------|--------------|---------------|-----------------|
| **Object Storage** | MinIO (1 container) | Ceph RGW (RADOS cluster) | Config: S3 endpoint URL |
| **Compute** | Spark (1 container) | Spark on K8s (Operator + HPA) | K8s manifests + Spark Operator |
| **Catalog** | Polaris (1 container) | Polaris (K8s Service) | K8s deployment YAML |
| **Query Engine** | Trino (1 container) | Trino (K8s + worker scaling) | K8s deployment |
| **BI** | Superset (1 container) | Superset (K8s + PG + Redis) | K8s deployment |
| **Credentials** | `.env` file | Kubernetes Secrets | Secret management |
| **Pipeline Code** | — | — | **No changes** |
| **SQL Queries** | — | — | **No changes** |

### What We Need from the Infrastructure Team

| # | Request | Why | Priority |
|---|---------|-----|----------|
| 1 | **Ceph RGW S3 endpoint + credentials** | All storage I/O goes through S3 API. We need: endpoint URL, access key, secret key, region, bucket name | Blocking |
| 2 | **Kubernetes namespace + resource quotas** | Pipeline services: Polaris (256 MB–1 GB), Spark driver (2 GB), Spark executors (2–8 GB × N), Trino (2 GB coordinator + workers) | Blocking |
| 3 | **Spark Kubernetes Operator** | Or equivalent mechanism for submitting Spark jobs to K8s. Spark on K8s uses `spark-submit --master k8s://...` | High |
| 4 | **Network policy: pods → Ceph RGW** | Spark, Polaris, and Trino all need S3 access to Ceph | Blocking |
| 5 | **PostgreSQL instance** (or shared) | Superset metadata DB (small, ~100 MB). Can share existing PG or deploy dedicated | Medium |
| 6 | **Persistent volumes** | Polaris metadata (SQLite or PG-backed), Superset DB | Medium |

### Storage Sizing Estimate (Production)

| Data | Current (simulated) | Expected Production | Notes |
|------|--------------------|--------------------|-------|
| Bronze (raw) | ~50 MB | ~50–500 GB | 14 tables × months of collection |
| Silver (optimized) | ~40 MB | ~40–400 GB | Same data, reorganized (Parquet compression) |
| Gold (ML-ready) | ~20 MB | ~20–200 GB | Subset of Silver (pre-joined) |
| **Total** | **~110 MB** | **~100 GB – 1 TB** | Depends on collection campaign length |

*Note: Iceberg uses Parquet columnar format → ~3–10× compression vs. raw JSON.*

**Speaker notes:**
> This is the most relevant slide for the infrastructure team. The pipeline code is storage-agnostic — everything talks S3 API. The migration from MinIO to Ceph is a config change: swap the endpoint URL and credentials. What we need from infra: a Ceph RGW endpoint, a K8s namespace with resource quotas, and the Spark Kubernetes Operator for job submission. Storage-wise, expect 100 GB to 1 TB at full production scale — Iceberg's Parquet format compresses well. No special hardware requirements beyond what Ceph and K8s already provide.

---

## Slide 8: Status Summary & Next Steps

**Title:** Current Status & Roadmap

### What's Operational

| Deliverable | Status |
|-------------|--------|
| Full medallion pipeline (Bronze → Silver → Gold → Validate) | ✅ Complete — ~24s end-to-end |
| KAIST 3-level schema (14 entity types, 4 geometric structs) | ✅ Implemented — validated on KAIST + nuScenes |
| 9-service Docker Compose dev environment | ✅ Running — single `docker compose up` |
| Benchmark suite (8 experiments, reproducible) | ✅ Complete — results in JSON |
| nuScenes cross-validation (public dataset) | ✅ Complete — confirms pipeline generalizes |
| Trino SQL access to all Iceberg tables | ✅ Working — `SELECT * FROM kaist_gold.camera_annotations` |
| Superset BI dashboards | ✅ Connected to Trino |

### Known Limitations

| Limitation | Impact | Plan |
|-----------|--------|------|
| Running on simulated data | Cannot validate against real-world edge cases | Blocked on real KAIST data delivery |
| Single-node benchmarks only | Production perf unvalidated | Deploy to K8s + Ceph first |
| No PyTorch DataLoader integration | Gold tables accessible via Spark→pandas, not native Dataset | Planned after production deployment |
| 3 placeholder schemas (occupancy, motion, session_ego) | Missing from Silver/Gold | Waiting on schema definitions |
| No streaming ingestion | Batch-only | Kafka → Iceberg planned Q3–Q4 2026 |

### Next Steps

| Step | Dependency | Order |
|------|-----------|-------|
| Obtain Ceph RGW endpoint + K8s namespace | **Infrastructure team** | 1 |
| Deploy Polaris + Spark Operator to K8s | K8s access | 2 |
| Integration test on Ceph (identical pipeline) | Ceph endpoint | 3 |
| Ingest real KAIST data through pipeline | Data delivery from collection team | 4 |
| Distributed Spark benchmark (multi-executor) | K8s Spark Operator | 5 |
| PyTorch DataLoader integration | None | 6 (parallel) |
| Streaming ingestion (Kafka → Iceberg) | Architecture decision | Q3–Q4 2026 |

**Speaker notes:**
> Summary: the pipeline is fully implemented and validated in our dev environment. The main blocker for production is infrastructure access — Ceph endpoint and K8s namespace. Once we have those, deployment is configuration-only: same code, same queries, just pointed at a different storage backend. The second blocker is real data — everything runs on simulated data right now, and while the schema and scale are realistic, we can't validate against real-world edge cases until we have actual KAIST collection data. Happy to take questions.

---

## Appendix: Quick Reference

### How to Start the Dev Environment

```bash
cp example.env .env
chmod +x start.sh
./start.sh
```

### Service URLs (Dev)

| Service | URL |
|---------|-----|
| MinIO Console | http://localhost:9001 (minioadmin/minioadmin) |
| Polaris API | http://localhost:8181 |
| Spark Web UI | http://localhost:4040 |
| Trino | http://localhost:8080 |
| Superset | http://localhost:8088 |

### Run the Pipeline

```bash
# Inside Spark container
docker exec -it spark-iceberg bash
python /home/iceberg/kaist_ingestion/kaist_runner.py all
```

### Query Gold Tables (Trino)

```sql
-- Object detection training data
SELECT * FROM iceberg.kaist_gold.camera_annotations
WHERE camera_name = 'CAM_FRONT' LIMIT 10;

-- SLAM/localization data
SELECT * FROM iceberg.kaist_gold.lidar_with_ego
WHERE clip_id = 'clip_xxx';

-- Multi-modal fusion data
SELECT * FROM iceberg.kaist_gold.sensor_fusion_frame
WHERE clip_id = 'clip_xxx';
```
