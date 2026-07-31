# planning/ — driving model-based scene difficulty (detachable module)

Scores each clip by *how hard it was to drive*, as an optional Gold sub-score.
Designed to be **fully detachable** (MEDALLION_PROGRESS.md §13): if this never
matures, delete it with no effect on the rest of the pipeline.

## Status: rung 1 (constant-velocity open-loop planner)

`runner.py` computes a per-clip open-loop difficulty = the L2 error (@3 s) of a
trivial **constant-velocity planner** vs the ground-truth ego trajectory, from
`labels/egomotion`. CPU + NFS only — **no GPU, no catalog, no learned model**.

Gate-proven (§13, 2026-06-22): Spearman 0.69 vs `ego_dynamics` — correlated but
NOT redundant (~half its variance is independent), so it adds real signal.

Later rungs (separate work) keep the same output contract:
- rung 2: map-free PDMS (collision / time-to-collision / progress) using a real
  E2E planner's trajectory + the BEVFusion boxes;
- rung 3: map-dependent PDMS (needs an estimated drivable-area map).

## Interface (the only coupling)

- **Output**: `<NFS>/.planning/planning_shard_NN.parquet` with columns
  `clip_id, planning_score ∈ [0,1], cv_l2_3s_m, n_points, scored_at`.
- **Consumer hook**: `edge_case_scorer._load_planning_scores()` reads that dir
  via Spark; `compute_scene_score(..., planning_score=...)` adds a `planning`
  dimension and renormalizes the active weights (drops it when absent).
- **Weight**: `_SCENE_WEIGHTS["planning"]` (0.15, conservative — overlaps
  `ego_dynamics`; tunable).

Nothing in the core Spark pipeline (`canonical_bronze`, `quality_checks`,
`pipeline`) imports this module.

## Run

```bash
# host python3 (has pyarrow; NFS mounted at ./netai-e2e)
python3 planning/runner.py            # scores all on-disk egomotion clips
# options: --horizon 3.0 --scale 10.0 --workers 32 --max-clips N
```

Then re-run Gold scoring (`edge_case_scorer --backend metadata`) — it will log
"Loaded planning scores for N clips" and blend the dimension in.

## Remove (cost = 3 deletions)

1. delete `planning/`,
2. delete `_load_planning_scores` + its call in `edge_case_scorer._score_metadata_bulk`,
3. delete `"planning"` from `_SCENE_WEIGHTS` (and the `planning_score` arg/dim in
   `compute_scene_score`).

Optionally `rm -rf <NFS>/.planning/`. No schema migration is involved — the
score flows through `difficulty_score` + the `detail` JSON only.
