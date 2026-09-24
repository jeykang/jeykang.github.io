# System Architecture — Late March 2026

*Snapshot of the lakehouse software stack as of commit `c468ba7` (2026-03-05, the
last state before April). Companion doc: `ARCHITECTURE_2026-06.md` (current).*

## Purpose at this stage
A lakehouse **proof-of-concept**, benchmarked on autonomous-driving datasets:
**nuScenes** and a **simulated KAIST** dataset (nuScenes data conformed to the
first version of the KAIST schema). The focus was demonstrating the medallion
architecture, schema design, and scalability — not curating a production dataset.

## Software stack (layers)

| Layer | Component | Notes |
|---|---|---|
| **Table format** | Apache Iceberg v2 | medallion Bronze/Silver/Gold |
| **Object store** | MinIO (`quay.io/minio/minio`) | bucket `spark1`; `mc` (`setup_bucket`) creates it |
| **Catalog** | **Apache Polaris** REST catalog (`apache/polaris:latest`) | **in-memory metastore — ephemeral; the catalog was lost on every restart.** Bootstrapped by a `polaris-setup` (`alpine/curl`) job hitting the Polaris REST API. Default catalog name `lakehouse_catalog` |
| **Compute** | Spark 3.5.5 + Iceberg 1.8.1 (`tabulario/spark-iceberg:3.5.5_1.8.1`, PySpark) | image also bundles Nessie + S3 JARs (Nessie not used as the catalog) |
| **Query engine** | Trino 479 (`trinodb/trino:479`) | |
| **BI** | Superset (built from `./superset`) + Postgres 15 (Superset metadata DB) + Redis 7 + `superset-init` | custom viz plugin `superset-plugin-chart-databahn-pipelines` |
| **ML / GPU** | **none** | no perception or model inference; CPU-only |

*(Postgres 15 was present only as Superset's metadata DB — the Polaris catalog had
no database backing.)*

## Pipelines & data modules
- **`kaist_ingestion/`** — KAIST E2E medallion pipeline (Bronze → Silver → Gold)
  over the **simulated dataset** (`simulate_from_nuscenes.py`, `schemas.py`,
  `validators.py`, `ingest_bronze`/`transform_silver`/`build_gold`,
  `kaist_runner.py`).
- **`nuscenes_experiment/`** — nuScenes lakehouse-vs-baseline scalability
  experiments (Python scripts + notebooks, results, charts).
- **`benchmarks/`** — KAIST scalability + AD-workload benchmarks.
- **`python-scripts/`** — nuScenes-mini ingest, Spark/Iceberg/Nessie test.
- **`superset-plugin-chart-databahn-pipelines/`** — custom Superset chart plugin.
- **`paper/`** — research write-up + figures.
- **Schema**: `kaist_schema` (first KAIST schema design).

## Defining characteristics
- Datasets are **synthetic/benchmark** (nuScenes + simulated KAIST), not a real
  production corpus.
- Catalog is **non-persistent** (in-memory Polaris).
- **No ML/GPU layer** — purely storage + query + BI over Iceberg.
- Emphasis on **schema + scalability benchmarking** (paper-oriented).
