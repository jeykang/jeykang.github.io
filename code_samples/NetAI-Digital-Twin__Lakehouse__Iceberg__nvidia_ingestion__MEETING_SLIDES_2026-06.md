# Progress Since May 28 — NVIDIA PhysicalAI Lakehouse
*Meeting deck source · 2026-06-23 · medallion pipeline (Iceberg + Polaris + MinIO + Spark)*

Headline: **the Gold "edge-case" tier now produces a rigorously-validated subset of
the hardest-to-drive clips** — replacing a metadata heuristic that, on inspection,
did not track real difficulty.

---

## Slide 1 — Headline outcomes

- **Validated difficulty scoring for edge-case mining.** Gold now keeps the
  hardest clips (strip the easy) on two *validated* axes, not a heuristic:
  **behavioral** (agent interaction) + **perceptual** (low-light/degradation).
- **Perception operational.** BEVFusion 3D multimodal detection runs end-to-end
  on the dataset (lidar + 6 cameras); integrated into Gold scoring.
- **Agent-interaction difficulty from the dataset's own 3D labels.** Validated
  against human-flagged hard clips: **AUC 0.65 overall, 0.87 on pedestrian-density**.
- **Infrastructure hardened.** Persistent catalog (Postgres-backed Polaris,
  survives restarts); Silver sensor-quality check corrected → **99.86% retention**.
- **Repo cleaned up** — early-setup experiments archived; active tree is just the
  NVIDIA pipeline.

*Speaker note: the month's theme — we turned "difficulty scoring" from a plausible
heuristic into a measured, validated signal.*

---

## Slide 2 — The core result: difficulty scoring is now valid

**Problem found:** the prior Gold difficulty score was **anti-aligned** with
human-judged hard clips (AUC **0.450** — it was effectively selecting *night*
clips, not *hard-to-drive* ones). Established a **validation battery** (negative
control + external human "hard-clip" labels) as a standing gate; several candidate
signals were tested and only the ones that passed were kept.

**Fix — a two-axis union built for edge-case mining** (keep a clip if hard on
*either* axis):

| Axis | Signal | Validation |
|---|---|---|
| **Behavioral** | agent-interaction conflict (dataset 3D labels) | OOD AUC **0.651**; pedestrian-density **0.866** |
| **Perceptual** | darkness / detection degradation | measured: **−24% detections, −10% confidence** in the dark |

- Composite fixed: dark, perceptually-hard clips went from being **discarded** to
  correctly **kept** (rank-corr with darkness −0.14 → **+0.61**).
- Output: **Gold = ~3,200 hardest clips** (top 10% of the on-disk 10 TB sample,
  ~31.7 k clips), each defensibly "hard" on a measured axis.

*Speaker note: the headline number — went from 0.45 (worse than random) to a
validated signal, and proved darkness is real difficulty we were previously
throwing away.*

---

## Slide 3 — Supporting work + next steps

**Also delivered**
- BEVFusion perception scoring cohort + Gold integration (fixed a loader bug that
  had silently disabled perception).
- Persistent Polaris catalog (Postgres) — no more catalog loss on restart.
- Silver `missing_sensors` rewritten against expected-sensor truth → 99.86%
  retention (prior version falsely failed valid clips).
- Reproducible validation tooling (`validity_battery.py`) — reusable on any future
  dataset/signal.

**Methodology takeaway** (reusable): every difficulty signal must pass the battery
(scene-driven + tracks an external hard-clip label) before it reaches Gold.

**Next steps**
- Grow the on-disk sample to extend coverage of the validated signals.
- Add/validate further hardness axes (e.g. adverse weather) under the same battery.
- Tune the behavioral/perceptual balance against downstream training outcomes.

*Speaker note: the validation battery is the durable asset — it generalizes to the
other datasets the lakehouse will ingest.*
