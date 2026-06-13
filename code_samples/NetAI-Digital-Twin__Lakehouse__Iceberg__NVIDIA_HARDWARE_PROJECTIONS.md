# Nvidia PhysicalAI Lakehouse — Hardware Projections for Petabyte-Scale Operation

**Date:** April 2026
**Basis:** Measured scaling laws from 7-level benchmark (2–75 clips) and 4-point scalability sweep (1.4–52.6 GB)

---

## 1. Measured Scaling Laws

All projections derive from empirically validated linear models (R² > 0.99):

```
Ingestion RSS (MB)  = 421.0  × clips + 944       (per JVM invocation)
Ingestion wall (s)  = 53.9   × clips + 117        (per JVM invocation)
Row count           = 4.738M × clips + 9.3M       (Bronze layer)
Storage per clip    ≈ 555 MB compressed (Iceberg/Parquet)
```

Query-time memory is **constant at 224 MB** regardless of data volume (proven across 36× scale increase from 123M to 4.48B rows).

---

## 2. Ingestion Phase — Resource Requirements

### 2.1 Key Insight: Memory Is the Only Hard Constraint

| Resource | Scaling behavior | Mitigation |
|----------|-----------------|------------|
| **Memory (RSS)** | **Linear, accumulates per JVM** | **Chunk-based ingestion: restart JVM between chunks** |
| Wall time | Linear, parallelizable | Add nodes |
| CPU | Linear, parallelizable | Add cores/nodes |
| Storage | Linear, distributable | Add object storage capacity |
| Network I/O | Linear, pipelineable | Parallel reads/writes |

### 2.2 Chunk-Based Ingestion Model

Since RSS grows linearly within a single JVM, ingestion is partitioned into fixed-size chunks. Each chunk runs in an independent JVM (or container), ingests N clips, writes to Iceberg via ACID commits, and exits — freeing all memory. Iceberg's optimistic concurrency ensures consistency across concurrent chunk writes with no merge step.

**Chunk sizing by node memory:**

| Node RAM | JVM Heap | Usable RSS | Max clips/chunk | Wall time/chunk | Storage written/chunk |
|---------:|---------:|-----------:|----------------:|----------------:|----------------------:|
| 32 GB | 24 GB | 28 GB | 64 | ~58 min | ~36 GB |
| 64 GB | 48 GB | 56 GB | 130 | ~2.0 hr | ~72 GB |
| 128 GB | 96 GB | 112 GB | 264 | ~4.0 hr | ~147 GB |
| 256 GB | 200 GB | 230 GB | 544 | ~8.2 hr | ~302 GB |
| 512 GB | 400 GB | 460 GB | 1,090 | ~16.5 hr | ~605 GB |

Formula: `Max clips/chunk = (Usable RSS − 944 MB) ÷ 421 MB/clip`

### 2.3 Cluster Sizing for Target Datasets

#### Scenario A: Nvidia PhysicalAI Full Dataset (169 TB)

- 311,600 clips × 555 MB = **169 TB** Iceberg storage
- 311,600 clips × 4.74M = **~1.48 trillion rows**

| Cluster | Node RAM | Nodes | Chunks total | Wall time (sequential/node) | **Estimated elapsed** |
|---------|---------|------:|-------------:|----------------------------:|----------------------:|
| Minimal | 64 GB | 4 | 2,397 | ~600 hr/node | **~6.3 months** |
| Medium | 128 GB | 10 | 1,180 | ~472 hr/node | **~20 days** |
| Large | 128 GB | 50 | 1,180 | ~94 hr/node | **~4 days** |
| Fast | 256 GB | 20 | 573 | ~47 hr/node | **~2 days** |

#### Scenario B: 1 PB Dataset (~1.8M clips)

| Cluster | Node RAM | Nodes | **Estimated elapsed** |
|---------|---------|------:|----------------------:|
| Medium | 128 GB | 10 | ~113 days |
| Large | 128 GB | 50 | ~23 days |
| Fast | 256 GB | 100 | ~6 days |

#### Scenario C: 5 PB Dataset (~9M clips)

| Cluster | Node RAM | Nodes | **Estimated elapsed** |
|---------|---------|------:|----------------------:|
| Large | 128 GB | 100 | ~65 days |
| Fast | 256 GB | 100 | ~30 days |
| HPC | 256 GB | 500 | ~6 days |

### 2.4 Ingestion Hardware Summary

| Component | Recommendation | Rationale |
|-----------|---------------|-----------|
| Compute nodes | 128–256 GB RAM, 8+ cores | RSS = 421 MB/clip; 128 GB handles 264 clips/chunk |
| Storage backend | Ceph RGW / S3 | Iceberg S3FileIO; decoupled from compute |
| Orchestration | Kubernetes Jobs / Airflow | Each chunk = one container; retry-safe via Iceberg ACID |
| Network | 10+ Gbps node-to-storage | Bronze is I/O-bound on source reads |
| Lidar mode | Blob only | 7.7× faster, 5.9× less memory than decoded |
| GC tuning | `gc.collect()` between table groups | 25% RSS reduction at scale |

---

## 3. Query Phase — Resource Requirements

### 3.1 Key Insight: Query Resources Are Constant

Query-time memory (RSS) was measured at **224 MB and constant** across a 36× data scale increase (1.4 GB → 52.6 GB, 123M → 4.48B rows). This is because Iceberg queries work on metadata (manifests, partition statistics) to identify relevant data files, then stream through matching files without loading the entire dataset.

### 3.2 Measured Query Latency (Constant Across Scale)

| Query | Tier | At 123M rows | At 4.48B rows | Scaling |
|-------|------|------------:|-------------:|---------|
| COUNT (all) | Bronze | 82 ms | 38 ms | Constant |
| COUNT (all) | Silver | 67 ms | 24 ms | Constant |
| COUNT (all) | Gold | 67 ms | 23 ms | Constant |
| Aggregation (GROUP BY clip) | Silver | 214 ms | 295 ms | Constant |
| Gold fusion sample | Gold | 89 ms | 80 ms | Constant |
| Gold summary | Gold | 60 ms | 21 ms | Constant |

Queries are actually **faster** at larger scale (JVM warmup / metadata caching effects).

### 3.3 Projected Query Latency at Petabyte Scale

| Query pattern | 2 clips (measured) | 10K clips (est.) | 1M clips (est.) |
|--------------|-------------------:|-----------------:|----------------:|
| Gold count (all) | 37 ms | 0.1–0.5 s | 1–5 s |
| Gold single partition (clip_id) | 34 ms | 40–100 ms | 50–500 ms |
| Single clip_id lookup | 47 ms | 50–100 ms | **50–100 ms (O(1))** |
| Silver 19-way union + join | 300 ms | 2–5 s | 10–60 s |
| Aggregation (3.2M rows) | 181 ms | 1–3 s | 5–30 s |
| Time travel (VERSION AS OF) | 37 ms | 40–100 ms | 50–500 ms |

**Critical insight for ML workloads:** The primary access pattern — fetching one clip's data for a training dataloader — is a `clip_id` partition lookup, which remains **O(1)** at any scale. Iceberg's partition pruning reduces a full-table scan to a single file read regardless of total dataset size.

### 3.4 Query-Time Hardware Summary

| Component | Recommendation | Rationale |
|-----------|---------------|-----------|
| Query engine | Trino (distributed) or DuckDB (single-node) | Trino for large scans; DuckDB for clip lookups |
| Memory per worker | 8–32 GB | Queries are streaming; 224 MB measured at 4.48B rows |
| Workers | 3–10 for interactive; scale with concurrent users | Partition pruning keeps per-query cost low |
| Storage | Same Ceph/S3 as ingestion | Iceberg decouples compute from storage |
| Caching | Alluxio or local SSD | Amortize S3 latency for repeated training queries |
| Gold tables | **Required** at >10K clips | 10–100× faster than Silver joins at scale |

### 3.5 Query Infrastructure Cost Comparison

| Dataset size | Ingestion cluster (temporary) | Query cluster (persistent) |
|-------------|------------------------------|---------------------------|
| 169 TB (full NV) | 10–50 × 128 GB (4–20 days) | 3–5 × 32 GB Trino workers |
| 1 PB | 50–100 × 128–256 GB (6–23 days) | 5–10 × 32 GB Trino workers |
| 5 PB | 100–500 × 128–256 GB (6–30 days) | 10–20 × 32 GB Trino workers |

Ingestion is a burst workload (spin up, ingest, tear down). Query infrastructure is lightweight and persistent. At all scales, the query cluster is a fraction of the ingestion cluster's cost.

---

## 4. Storage Requirements

### 4.1 By Overhead Strategy

| Strategy | Overhead multiplier | 169 TB source | 1 PB source | 5 PB source |
|----------|-------------------:|---------------:|------------:|------------:|
| All materialized (Silver + Gold) | 2.3× | 388 TB | 2.3 PB | 11.5 PB |
| View Silver + materialized Gold | 1.3× | 220 TB | 1.3 PB | 6.5 PB |
| View Silver + view Gold | 1.0× | 169 TB | 1.0 PB | 5.0 PB |

### 4.2 Object Storage Sizing

| Dataset | Recommended capacity | Rationale |
|---------|--------------------:|-----------|
| 169 TB (full NV) | 250–400 TB | 1.3× data + snapshots + staging |
| 1 PB | 1.5–2.5 PB | With snapshot retention for time travel |
| 5 PB | 6–12 PB | Multiple snapshot versions + compaction headroom |

Iceberg snapshot retention (default: 10 snapshots, 7 days) adds ~10–20% overhead for time travel capability.

---

## 5. Summary: Minimum Viable Clusters

### For 169 TB Nvidia PhysicalAI Full Dataset

| Phase | Nodes | RAM/node | Duration | Persistent? |
|-------|------:|--------:|---------:|:-----------:|
| **Ingestion** | 10 | 128 GB | ~20 days | No (burst) |
| **Query** | 3 | 32 GB | — | Yes |
| **Storage** | — | — | — | 250–400 TB Ceph/S3 |

### For 1 PB

| Phase | Nodes | RAM/node | Duration | Persistent? |
|-------|------:|--------:|---------:|:-----------:|
| **Ingestion** | 50 | 128 GB | ~23 days | No (burst) |
| **Query** | 5 | 32 GB | — | Yes |
| **Storage** | — | — | — | 1.5–2.5 PB Ceph/S3 |

### For 5 PB

| Phase | Nodes | RAM/node | Duration | Persistent? |
|-------|------:|--------:|---------:|:-----------:|
| **Ingestion** | 100 | 256 GB | ~30 days | No (burst) |
| **Query** | 10 | 32 GB | — | Yes |
| **Storage** | — | — | — | 6–12 PB Ceph/S3 |
