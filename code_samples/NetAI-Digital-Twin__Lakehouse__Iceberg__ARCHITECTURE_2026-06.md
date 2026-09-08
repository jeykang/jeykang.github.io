# System Architecture — Current (June 2026)

*Snapshot of the lakehouse software stack as of `HEAD` (June 2026). Companion doc:
`ARCHITECTURE_2026-03.md` (late-March baseline).*

## Purpose at this stage
Production **medallion curation of the real NVIDIA PhysicalAI** autonomous-vehicle
dataset (a deliberate **10 TB on-disk sample** of the ~120 TB / ~300k-clip
dataset), whose headline output is a **validated edge-case "difficulty" Gold tier**
— a subset with the trivially-easy clips stripped out for training.

## Software stack (layers)

| Layer | Component | Notes |
|---|---|---|
| **Table format** | Apache Iceberg v2 | medallion `nvidia_bronze` / `nvidia_silver` / `nvidia_gold` |
| **Object store** | MinIO (`quay.io/minio/minio`), bucket `spark1` | clip media (lidar/camera/labels) on **NFS** |
| **Catalog** | **Apache Polaris** REST catalog (`apache/polaris:latest`) — **persistent: `relational-jdbc` backed by a dedicated Postgres 16 (`polaris_postgres`)** | bootstrapped by `apache/polaris-admin-tool` (`polaris_bootstrap`) + `polaris-setup`. Survives restarts (was ephemeral in March) |
| **Compute** | Spark 3.5.5 + Iceberg 1.8.1 (`tabulario/spark-iceberg:3.5.5_1.8.1`, PySpark) | unchanged core |
| **Query engine** | Trino 479 | unchanged |
| **BI** | Superset + Postgres 15 + Redis 7 + `superset-init` | unchanged |
| **Perception (GPU)** | **BEVFusion** 3D multimodal detection — `bevfusion/` on `pytorch/pytorch:2.1.2-cuda12.1-cudnn8-devel` + OpenMMLab `mmdet3d` 1.4.x (CUDA 12.1, GPU arch sm_75/sm_86) | runs on lidar + 6 cameras; outputs detection stats used in scoring |
| **Difficulty signals** | `planning/` — `conflict_runner.py` (GPU-free agent-conflict from `obstacle.offline` 3D labels); shelved planner experiments in `diffusiondrive/`, `sparsedrive/` | feeds the Gold difficulty composite |
| **Optional / aux** | `cosmos_augmentation/` (NVIDIA Cosmos synthetic-scene generation via NIM `nvcr.io/nim/.../cosmos-transfer2`, **commented-out** in compose; needs GPU + NGC key); `demo_wall/` (static HTML/JS results-visualization wall) | |

## Pipelines & data modules
- **`nvidia_ingestion/`** — the production medallion pipeline (Bronze → Silver →
  Gold) for NVIDIA PhysicalAI, including:
  - Silver quality checks (sensor presence via `feature_presence`).
  - `edge_case_scorer.py` — the Gold difficulty scorer. **Difficulty = noisy-OR
    union of two validated axes**: behavioral (`conflict`) + perceptual
    (darkness / low detection confidence), rank-normalized, scoped to the on-disk
    sample. Gated by a **validity battery** (`validity_battery.py`).
- **`bevfusion/`** — perception runner + scorer (containerized, GPU).
- **`planning/`** — driving-difficulty signal runners (agent-conflict; shelved
  learned planners).
- **`obstacle.offline`** — the dataset's own 3D auto-labels (16 GB) → populates the
  canonical `DynamicObject` table and feeds agent-conflict scoring.
- **Schema base**: NVIDIA conforms to the KAIST-derived AD schema; the live
  `kaist_ingestion/config.py` (`KAISTConfig`) is retained as the **shared schema
  config** the NVIDIA pipeline imports; design in `kaist_schema_v2.dbml`.
- **`archive/`** — early-setup work (nuScenes/KAIST/Nessie tests, benchmarks,
  paper) moved out of the active tree.

## Changes since March (delta)
- **Core lakehouse unchanged** — same Iceberg + Polaris + MinIO + Spark + Trino +
  Superset versions; the platform was already in place.
- **Catalog: ephemeral → durable** — Polaris now Postgres-backed (no catalog loss
  on restart).
- **Dataset: synthetic/benchmark → real production** — nuScenes + simulated KAIST
  replaced by the NVIDIA PhysicalAI dataset with a dedicated pipeline.
- **New GPU ML layer** — BEVFusion perception + the difficulty-scoring stack
  (agent-conflict from `obstacle.offline`, perceptual axis, validity battery);
  none of this existed in March.
- **KAIST demoted** from "the pipeline" to "the shared schema config"; old
  experiments archived.
