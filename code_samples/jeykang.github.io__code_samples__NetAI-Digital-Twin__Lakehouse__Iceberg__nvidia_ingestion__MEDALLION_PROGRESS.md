# Medallion Lakehouse — Progress Index

NVIDIA PhysicalAI medallion pipeline (Apache Iceberg + Polaris + MinIO + Spark).
**Owner**: jeykang · **Last updated**: 2026-06-26

Detailed progress logs are split by month under [`progress/`](progress/) (this file
was getting unwieldy as one scroll). Companion reference docs:
- **Architecture**: [`../ARCHITECTURE_2026-06.md`](../ARCHITECTURE_2026-06.md) (current) · [`../ARCHITECTURE_2026-03.md`](../ARCHITECTURE_2026-03.md) (baseline)
- **Current-state synthesis**: [`PROGRESS_REPORT_2026-06.md`](PROGRESS_REPORT_2026-06.md) · meeting facts: [`MEETING_FACTSHEET_2026-06.md`](MEETING_FACTSHEET_2026-06.md)
- **Validation methodology + findings**: [`VALIDITY_BATTERY_FINDINGS.md`](VALIDITY_BATTERY_FINDINGS.md)

## Current status (2026-06)
Medallion tiers are **curation tiers**: **Bronze** (full dataset registered as-is,
zero-copy via `add_files()`) → **Silver** (quality-filtered views, exclude FAIL
clips) → **Gold** (hardest edge-case clips). Built over a **~13.5 TB on-disk
sample** (~32,986 canonical clips) of the NVIDIA PhysicalAI dataset.

Gold difficulty is a **validated noisy-OR union** of a behavioral axis
(agent-conflict from `obstacle.offline`) and a perceptual axis (darkness / low
detection confidence), scoped to the sensor-covered sample → **Gold = 3,176 clips**.
The earlier metadata-weighted composite was found anti-aligned with human-hard
labels (OOD AUC 0.450) and re-architected. Full detail: [`progress/2026-06.md`](progress/2026-06.md).

## Iceberg namespaces (catalog `iceberg`)
| Namespace | Content |
|---|---|
| `nvidia_bronze` | raw sensor data registered via `add_files()` |
| `nvidia_silver` | quality-filtered views (exclude FAIL clips from `quality_report`) |
| `nvidia_gold` | edge-case views (top-N% hardest) + `clip_scores` |

## Monthly progress log
- **[2026-04 — Foundational build](progress/2026-04.md)** — Bronze/Silver/Gold
  setup, NFS lidar/radar recovery (~10.85 TB re-download), canonical schema
  migration, perception integration v3–v5, known issues, benchmarks.
- **[2026-05 — Perception scoring](progress/2026-05.md)** — BEVFusion multimodal
  perception made operational (mmdet3d); sampling-adequacy verdict (N=20).
- **[2026-06 — Difficulty metric + validation](progress/2026-06.md)** —
  driving-difficulty roadmap + gate, agent-conflict from `obstacle.offline`,
  validity battery (refuted the old composite), noisy-OR union re-architecture,
  repo cleanup, **+ figures**.

## How to run (quick reference)
- **Pipeline / tiers**: see [`progress/2026-04.md`](progress/2026-04.md) §2–§3
  (`register_bronze.py`, `quality_checks.py`, `pipeline.py`).
- **Gold difficulty scoring**: `run_gold_scoring` in `edge_case_scorer.py`
  (metadata backend) after `planning/conflict_runner.py` writes `.conflict/` and
  the BEVFusion runner writes `.perception/`.
- **Validation**: `validity_battery.py` / `union_validate.py`.
