# archive/ — early lakehouse setup work (no longer active)

Superseded experiments and scaffolding from when the Iceberg lakehouse was first
set up. Kept for reference; **not part of the active pipeline** (all current work
is the NVIDIA PhysicalAI dataset under `nvidia_ingestion/`, `planning/`,
`bevfusion/`). Moved here 2026-06-23.

| Item | What it was |
|---|---|
| `nuscenes_experiment/` | nuScenes lakehouse-vs-baseline benchmark notebooks/scripts |
| `kaist_ingestion/` | KAIST E2E ingestion test pipeline — **simulated a dataset by conforming nuScenes to the first KAIST schema** (bronze/silver/gold runners, schemas, validators, `simulate_from_nuscenes.py`). NOTE: the live `kaist_ingestion.config` (shared AD-lakehouse schema config the NVIDIA pipeline imports) stays at the top-level `kaist_ingestion/` — only the old test code is archived here. |
| `benchmarks/` | KAIST scalability + AD-workload benchmarks |
| `auto_label_pipeline/` | Earlier 2D/3D auto-labeling pipeline (superseded by `bevfusion/` + the dataset's `obstacle.offline` labels) |
| `python-scripts/` | `ingest_nuscenes_mini.py`, `spark-iceberg-nessie_test.py` (Nessie predates the Polaris catalog) |
| `paper/` | Early paper draft |
| `reports/` | Older write-ups (CODEBASE_REPORT, PROGRESS_REPORT_2026-04-06 + html/tex, PRESENTATION_DRAFT, SCHEMA_REVIEW_RESPONSE) — superseded by `nvidia_ingestion/PROGRESS_REPORT_2026-06.md` |
| `kaist_schema_v1` | First KAIST schema (the live version is `kaist_schema_v2.dbml` at repo root) |

`docker-compose.yml` still mounts `archive/{python-scripts,nuscenes_experiment,benchmarks}`
into the Spark container at their original `/opt/spark/...` paths, so the archived
code runs unchanged if ever needed. Archived scripts that imported sibling
`kaist_ingestion` test modules may need path fixes to run (those modules live here now).
