# Data Lakehouse — Biweekly Progress Report

**Date:** April 2026
**Project:** NetAI Digital Twin — Data Lakehouse
**Dataset:** Nvidia PhysicalAI Autonomous Vehicles (~119 TB, 310,895 clips, 14 sensor types)
**Platform (development):** NVIDIA DGX Spark (ARM64/Blackwell), 121 GB RAM, NFS-mounted dataset
**Stack:** Apache Iceberg 1.8.1, Polaris REST Catalog, MinIO (dev) / Ceph (prod), Trino 479, Superset

---

## Part 1: Pipeline Validated at Scale

### 1.1 Medallion Architecture

A full Bronze → Silver → Gold medallion pipeline was implemented and benchmarked for the Nvidia PhysicalAI AV dataset. The architecture adheres to the following tier guarantees:

- **Bronze:** Functionally raw data, but schema-enforced against typed PySpark `StructType` definitions. 41 Iceberg tables covering clip metadata (3), calibration (3), egomotion (1), lidar (1), radar (19 sensors), and camera metadata (14).
- **Silver:** Bronze data with basic Iceberg optimization features applied — partition pruning, sort orders for temporal locality, and column-level min/max statistics. Implemented as zero-copy SQL views over Bronze (0 bytes additional storage).
- **Gold:** Pre-joined, per-workload tables for ML dataloaders. Three tables targeting specific training workloads:
  - `lidar_with_ego` — LiDAR point-cloud + ego-motion trajectory (SLAM/localization)
  - `sensor_fusion_clip` — Per-clip denormalized metadata (multi-modal perception)
  - `radar_ego_fusion` — 19 radar sensors + ego-motion + extrinsics (radar-centric tracking)

### 1.2 Per-Table Benchmark (2-Clip Baseline)

Full pipeline profiled with per-table granularity across 85 tables (41 Bronze + 41 Silver + 3 Gold):

| Phase | Tables | Wall (s) | Wall % | CPU User (s) | Rows |
|-------|-------:|--------:|------:|-----------:|-----:|
| Bronze | 41 | 143.8 | 85.2% | 124.1 | 4,205,450 |
| Silver | 41 | 18.9 | 11.2% | 0.1 | 4,205,450 |
| Gold | 3 | 6.2 | 3.7% | 0.0 | 3,541,979 |
| **Total** | **85** | **168.8** | **100%** | **124.3** | **11,952,879** |

Peak RSS: 1,770.6 MB. Bronze dominates cost (>85% wall time) because it reads from NFS-mounted zip archives, decompresses Parquet files, concatenates Arrow tables, and writes to Iceberg.

**Cost distribution within Bronze:**
- Radar (19 sensors): 78% of all rows, ~115s CPU. Most expensive single table: `radar_front_center_imaging_lrr_1` (571K rows, 18.3s).
- Lidar: Dominates I/O (21.0 GB input) but only 400 rows in blob mode (2.9s wall).
- Camera metadata: 14 tables, fast (<0.4s each) despite multi-GB zip reads — only extracts small JSON files.

**Silver overhead:** 0.3–0.9s per table (catalog operations only). Zero CPU, zero memory, zero storage.

**Gold overhead:** 6.2s total for 3 tables. `radar_ego_fusion` produces 3.2M rows (19-way union + ego join) in <3s.

### 1.3 Data Volumes Per Clip

| Table group | Tables | Rows/clip | Bytes/clip (compressed) |
|------------|-------:|----------:|------------------------:|
| Clip metadata | 3 | ~12,300 | <2 MB |
| Calibration | 3 | ~33 | <1 MB |
| Egomotion | 1 | ~2,700 | ~1.2 MB |
| Lidar (blob) | 1 | ~200 | ~173 MB |
| Radar (19 sensors) | 19 | ~1,500,000 | ~330 MB |
| Camera metadata | 14 | ~600,000 | ~50 MB |
| **Total** | **41** | **~2,115,000** | **~555 MB** |

### 1.4 Scalability Benchmark (7-Level Sweep, 2–75 Clips)

| Level | Clips | Total Rows (B+S+G) | Peak RSS (MB) | Wall (s) | MB/clip | s/clip |
|------:|------:|--------------------:|---------------:|---------:|--------:|-------:|
| 0 | 2 | 11,952,879 | 1,857 | 168.0 | 928.5 | 84.0 |
| 1 | 5 | 28,853,704 | 3,178 | 347.2 | 635.6 | 69.5 |
| 2 | 10 | 54,324,602 | 4,560 | 628.8 | 456.0 | 62.9 |
| 3 | 25 | 133,962,002 | 11,131 | 1,529.8 | 445.2 | 61.2 |
| 4 | 50 | 261,406,994 | 22,964 | 2,918.0 | 459.3 | 58.4 |
| 5 | 60 | 310,604,218 | 27,353 | 3,545.0 | 455.9 | 59.1 |
| 6 | 75 | 339,648,381 | 31,141 | 3,925.9 | 415.2 | 52.3 |

**Linear regression models (all R² > 0.99):**

| Metric | Per-clip cost | Fixed overhead | R² |
|--------|-------------:|---------------:|---:|
| Peak memory (RSS) | 421 MB/clip | + 944 MB | 0.995 |
| Wall time | 53.9 s/clip | + 117 s | 0.993 |
| Row count | 4.74 M/clip | + 9.3 M | 0.989 |

**Phase breakdown at scale (Bronze dominates):**

| Level | Clips | Bronze (s) | Silver (s) | Gold (s) | Bronze % |
|------:|------:|-----------:|-----------:|---------:|---------:|
| 0 | 2 | 143.2 | 19.1 | 5.7 | 85.2% |
| 3 | 25 | 1,469.0 | 35.5 | 25.3 | 96.0% |
| 6 | 75 | 3,784.8 | 70.2 | 70.9 | 96.4% |

**GC optimization:** Reduced 75-clip peak RSS from 41.4 GB (OOM crash) to 31.1 GB (success) — a 25% memory reduction via `gc.collect()` between table groups.

### 1.5 Blob vs Decoded Lidar

| Metric | Blob | Decoded | Ratio |
|--------|-----:|--------:|------:|
| Wall time (1 clip) | 4.2 s | 32.4 s | 7.7× |
| Peak RSS | 1,253 MB | 7,363 MB | 5.9× |
| CPU (user+sys) | 4.6 s | 128 s | 27.7× |
| JVM heap required | 4 GB | 16 GB | 4.0× |

Decoded expansion is only viable for single-clip point analytics. Blob mode is mandatory for pipeline-scale ingestion.

---

## Part 2: Storage Overhead Reduction

### 2.1 Problem: Naive Materialization Replicates Data

A traditional all-materialized medallion architecture replicates data at each tier:
- Silver rewrites all Bronze data with optimization columns → **+100% storage**
- Gold materializes pre-joined tables → **+~30% storage** (dominated by `radar_ego_fusion` at ~16 GB for 52.6 GB source)
- **Total: ~2.3× source data**

At 169 TB source, this would require **~388 TB** of storage.

### 2.2 Three Strategies Benchmarked

| Strategy | Silver | Gold | Total multiplier | Projected (169 TB source) |
|----------|--------|------|----------------:|------------------------:|
| All materialized | +100% (full rewrite) | +30% (joined tables) | **2.3×** | 388 TB |
| Zero-copy Silver + materialized Gold | +0% (SQL views) | +30% (joined tables) | **1.3×** | 220 TB |
| Zero-copy Silver + view Gold | +0% (SQL views) | +0% (views + cache) | **1.0×** | 169 TB |

All three strategies produce **identical query interfaces** — the tier guarantees (schema enforcement, Iceberg optimizations, pre-joined ML-ready tables) are preserved regardless of materialization strategy.

### 2.3 Implementation Details

**Zero-copy Silver** (`--silver-mode view`): `CREATE OR REPLACE VIEW` over Bronze tables. Iceberg partition pruning, sort orders, and column statistics are applied via view definitions — no data movement. Measured overhead: 0.3–0.9s per table (catalog operations only), 0 bytes storage, 0 CPU.

**Gold view mode** (`--gold-mode view`): `CREATE OR REPLACE VIEW` for `lidar_with_ego` and `radar_ego_fusion`. Hot-path queries use `CACHE TABLE` for in-memory, ephemeral acceleration (no persistent storage). Exception: `sensor_fusion_clip` always materializes because it uses `monotonically_increasing_id()` for positional joins (non-deterministic, cannot be a stable view) — but this table is trivial in size (<1 MB).

**Hybrid benchmark results** (views + `CACHE TABLE`): At all scales tested (1.4–52.6 GB), Gold view queries with caching achieve **21–80 ms latency** — identical to materialized Gold. The cache is ephemeral (in-memory only, lost on restart) and adds zero persistent storage.

**Mode switching:** Both Silver and Gold modes are runtime flags (`--silver-mode`, `--gold-mode`) or environment variables (`NVIDIA_SILVER_MODE`, `NVIDIA_GOLD_MODE`). No code changes required to switch between strategies.

### 2.4 Scalability Report: Storage Overhead at 52.6 GB

From the 4-point scalability sweep (materialized Silver + Gold):

| Tier | Storage model | Data at largest scale (52.6 GB source) |
|------|--------------|---------------------------------------|
| Bronze | Zero-copy (`add_files()`) | 52.6 GB (original files, zero overhead) |
| Silver | Materialized CTAS | ~52.6 GB (full rewrite + clip_id column) |
| Gold | Materialized CTAS | ~16 GB (radar_ego_fusion dominates) |
| **Total** | | **~121 GB (~2.3× source)** |

With zero-copy Silver: 52.6 + 0 + 16 = **~69 GB (~1.3× source)**.
With zero-copy Silver + view Gold: 52.6 + 0 + 0 = **~52.6 GB (~1.0× source)**.

---

## Part 3: Query Performance

### 3.1 Gold vs Silver+Join Latency (2-Clip Dataset)

| Query | Gold (ms) | Silver+Join (ms) | Speedup | Rows |
|-------|----------:|------------------:|--------:|-----:|
| radar_ego_fusion (19-sensor union + ego join) | 75 | 300 | **4.0×** | 3,230,684 |
| sensor_fusion_clip (4-table metadata join) | 62 | 250 | **4.1×** | 310,895 |
| lidar_with_ego (lidar + egomotion) | 113 | 260 | **2.3×** | 400 |

Gold reads 5.7× more rows than a single Silver table yet returns in less time (37 ms vs 44 ms), because a single Iceberg table with 19 partitions has fewer catalog operations than 19 separate Silver tables.

### 3.2 Query Latency Across 36× Scale Increase

All queries exhibit **O(1) constant latency** from 123M to 4.48B rows:

| Query | Tier | At 123M rows | At 4.48B rows | Ratio |
|-------|------|------------:|-------------:|------:|
| COUNT (radar) | Bronze | 82 ms | 38 ms | 0.46× |
| COUNT (radar) | Silver | 67 ms | 24 ms | 0.36× |
| COUNT (radar fusion) | Gold | 67 ms | 23 ms | 0.35× |
| Aggregation (clip_id grouping) | Silver | 214 ms | 295 ms | 1.38× |
| Sample query | Gold | 89 ms | 80 ms | 0.90× |

Memory during queries: **constant 224 MB** at all scales.

### 3.3 Time Travel

| Query | Median (ms) | Rows | Overhead vs current |
|-------|------------:|-----:|-------------------:|
| Current snapshot | 41 | 3,230,684 | — |
| Historical snapshot (VERSION AS OF) | 37 | 3,230,684 | **−4 ms (zero)** |

Snapshot-based dataset pinning for reproducible training has zero query overhead.

### 3.4 Projected Latency at Petabyte Scale

| Query pattern | 2 clips (measured) | 10K clips (est.) | 1M clips (est.) |
|--------------|-------------------:|-----------------:|----------------:|
| Gold count (all) | 37 ms | 0.1–0.5 s | 1–5 s |
| Gold single partition (clip_id) | 34 ms | 40–100 ms | 50–500 ms |
| **Single clip_id lookup** | **47 ms** | **50–100 ms** | **50–100 ms (O(1))** |
| Silver 19-way union + join | 300 ms | 2–5 s | 10–60 s |
| Time travel (VERSION AS OF) | 37 ms | 40–100 ms | 50–500 ms |

The primary ML access pattern — fetching one clip's data for a training dataloader — is a `clip_id` partition lookup and remains **O(1) at any scale**.

---

## Part 4: Hardware Projections

### 4.1 Ingestion Phase (Burst Workload)

Ingestion uses chunk-based processing: each chunk runs in an independent JVM/container, ingests a fixed number of clips, writes to Iceberg, and exits. Iceberg ACID commits ensure consistency across concurrent writes.

**Chunk sizing:**

| Node RAM | Max clips/chunk | Wall time/chunk |
|---------:|----------------:|----------------:|
| 64 GB | 130 | ~2.0 hr |
| 128 GB | 264 | ~4.0 hr |
| 256 GB | 544 | ~8.2 hr |

**Cluster sizing for target datasets:**

| Dataset | Size | Nodes × RAM | Estimated elapsed |
|---------|-----:|:------------|------------------:|
| Nvidia PhysicalAI full | 169 TB | 10 × 128 GB | ~20 days |
| | | 50 × 128 GB | ~4 days |
| 1 PB | 1 PB | 50 × 128 GB | ~23 days |
| | | 100 × 256 GB | ~6 days |
| 5 PB | 5 PB | 100 × 256 GB | ~30 days |
| | | 500 × 128 GB | ~6 days |

### 4.2 Query Phase (Persistent, Lightweight)

Query-time memory is constant at 224 MB regardless of data volume. The query cluster is a fraction of the ingestion cluster's cost.

| Dataset size | Query cluster | Memory/worker | Notes |
|-------------|:-------------|:-------------|:------|
| 169 TB | 3–5 Trino workers | 32 GB | Clip lookups: O(1), 50–100 ms |
| 1 PB | 5–10 Trino workers | 32 GB | Full scans: 1–5 s; clip lookups: O(1) |
| 5 PB | 10–20 Trino workers | 32 GB | Scale with concurrent users, not data |

### 4.3 Storage

| Strategy | 169 TB source | 1 PB source | 5 PB source |
|----------|:-------------|:-----------|:-----------|
| View Silver + materialized Gold (1.3×) | 220 TB | 1.3 PB | 6.5 PB |
| View Silver + view Gold (1.0×) | 169 TB | 1.0 PB | 5.0 PB |

Add 10–20% headroom for Iceberg snapshots (time travel) and compaction staging.

---

## Part 5: Cosmos Synthetic Data Augmentation (In Progress)

### 5.1 Architecture

A modular pipeline that reads from Gold tables, calls the Nvidia Cosmos world foundation model API, and writes synthetic video + metadata to a dedicated `nvidia_cosmos` Iceberg namespace.

**Three-phase pipeline:**

1. **Extract:** Query `sensor_fusion_clip` Gold table for eligible clips. Extracts clip metadata (clip_id, split, country, hour_of_day, platform).
2. **Generate:** Call Cosmos API with variation-specific prompts. Each clip × variation produces one MP4 video.
3. **Ingest:** Upload MP4 to S3 object storage. Write metadata + lineage rows to Iceberg tables.

**Output tables:**
- `nvidia_cosmos.generated_scenes` — Partitioned by variation. Schema: clip_id, variation, prompt, model, seed, video_s3_uri, generation_time_s, source_split, created_at.
- `nvidia_cosmos.generation_lineage` — Source → synthetic traceability mapping.

### 5.2 Design Decisions

- **Separate Iceberg namespace** (`nvidia_cosmos`): Entire module can be dropped without affecting the base lakehouse. Never writes to nvidia_bronze/silver/gold.
- **Dual backend**: NVIDIA API Catalog (build.nvidia.com, default — cloud, no GPU required) or self-hosted NIM container (for on-premise GPU inference). Switchable via `--backend` flag.
- **Config-driven variations**: 6 default weather/lighting conditions (fog, rain, night, snow, golden hour, overcast). New variations added by editing a dict constant — no code changes.
- **Prompt templates as data**: Each variation maps to a prompt suffix appended to a base driving-scene prompt. Extensible without touching pipeline logic.

### 5.3 End-to-End Test Results (Mock Server)

| Metric | Result |
|--------|-------:|
| Clips processed | 3 |
| Variations per clip | 3 |
| Videos generated | 9 |
| Metadata rows written (Iceberg) | 9 |
| Lineage rows written (Iceberg) | 9 |
| Queryable via Trino | Yes |

### 5.4 Status

- [x] Pipeline code complete (5 modules: config, extract, generate, ingest_results, cosmos_runner)
- [x] End-to-end tested with mock Cosmos server
- [x] Dual backend support (cloud API + NIM container)
- [x] Iceberg metadata + lineage tables written and queryable
- [ ] Validate with live NVIDIA API Catalog endpoint
- [ ] Run at scale on full Gold dataset

### 5.5 Production Projection

311K clips × 6 variations = **~1.87M generated videos**. Generation is embarrassingly parallel — each clip is independent. Throughput scales linearly with API concurrency or NIM replica count.

---

## Data Sources

This report consolidates findings from:

- `nvidia_ingestion/BENCHMARK_REPORT.md` — 7-level scalability sweep (2–75 clips), per-table profiling, query latency, blob vs decoded lidar
- `nvidia_ingestion/SCALABILITY_REPORT.md` — 4-point scalability sweep (1.4–52.6 GB), zero-copy Bronze + materialized Silver/Gold, petabyte projections
- `nvidia_ingestion/hybrid_scalability_report.json` — Hybrid benchmark (views + CACHE TABLE), zero-overhead Gold views
- `cosmos_augmentation/` — Pipeline source code and mock server test results
