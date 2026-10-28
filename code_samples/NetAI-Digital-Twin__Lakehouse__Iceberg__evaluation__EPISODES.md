# Scenario × episode — the space the proving ground validates in (2026-09-21, updated 2026-10-06)

Every result the proving ground produces is a driving policy scored on an **episode**, and every
episode belongs to a **scenario class**. Both used to be implicit (decision windows chosen in
memory by the evaluator, interaction windows in a Cosmos manifest, closed-loop capability decided
by the NuRec catalog, twins in pipeline output folders). They are now rows in Iceberg, and the
closed-loop results are keyed to them.

![scenario x episode space](figures/fig_scenario_space.png)

## Definitions

- **Episode** (≈ ISO 34502 "concrete scenario", a test case): one scored time window of one clip,
  in one serving mode. Four kinds: a *decision window* (the open-loop evaluator's decision point,
  history + horizon around it), an *interaction window* (the 121 frames a Cosmos variant was
  rendered from), a *NuRec scene* (NVIDIA's reconstruction of the whole clip), and a *twin scene*
  (our own reconstruction of the whole clip, NuRec or HUGS).
- **Scenario class** (≈ ISO "logical scenario"): recording condition × augmentation × serving mode.
  The condition comes from the dataset's `hour_of_day` in bands (night 21–05 h, dawn/dusk 06–07 and
  18–20 h, day 08–17 h; `episodes.condition_of`).
- **Serving mode** (진열, `TERMINOLOGY.md`): the form a validator consumes an episode in.

| serving mode | validator | episodes come from |
|---|---|---|
| `openloop-mfpdms` | open-loop evaluator (MF-PDMS) over recorded logs | `harness.decision_times()` |
| `augmented-openloop` | the same, over a Cosmos variant | `cosmos_augmentation/batch_manifest.json` |
| `closedloop-nurec` | AlpaSim over NVIDIA's NuRec scene of the clip | AlpaSim's `sim_scenes.csv` |
| `closedloop-nurec-ours` | AlpaSim over our NuRec twin of the clip (NVIDIA's map layers borrowed) | `nurec/twin_pipeline.sh` outputs |
| `closedloop-hugsim` | HUGSIM over our HUGS twin of the clip (no map) | `hugsim/pai/twin_hugsim.sh` outputs |

## Tables

| table | rows (2026-10-06) | key |
|---|---|---|
| `nvidia_gold.episode` | 189,638: 187,212 decision windows (31,202 clips), 2,364 NuRec scenes, 50 interaction windows, 12 twin scenes | `episode_id` |
| `nvidia_gold.scenario` | 20 scenario classes, with episode, clip and kind counts and mean actors at the decision time | `scenario_id` = `condition\|augmentation\|serving_mode` |
| `eval.rollout` | 182 closed-loop rollouts (AlpaSim 149, HUGSIM 33), every one matched to its episode | `rollout_key`; joins on `episode_id` |

Episode ids: `<clip>:decision:<t0_us>:<a\|s>`, `<clip>:aug:<window_start>:<cond>`,
`<clip>:nurec:<scene uuid>`, `<clip>:twin:<serving mode>:<twin name>`.

**Actors at the decision time, by condition** (open-loop decision windows, `mean_actors_at_t0`):
day 37.6, dawn/dusk 32.3, night 25.6. Night scenes carry about a third fewer traffic participants,
which any day/night comparison of difficulty has to account for.

**What the figure shows.** Open-loop covers the recorded data densely (187,212 decision windows).
Closed-loop is where the budget binds: of 2,364 NuRec scenes, 89 have been rolled out (3.8 %), and
our own twins exist for nine clips (NuRec) and one clip (HUGS, eight more in progress). Augmented
episodes are 50, all open-loop. Choosing which of the 2,364 scenes earn a rollout is the job of
rollout triage (`SKIP.md`).

## eval.rollout — closed-loop results keyed by episode

One row per rollout (`rollouts.py`): simulator, run, clip, `episode_id`, serving mode, twin,
driving policy, rollout id or seed, how the run was configured (route generator, image format,
camera; read from the run's own files), each simulator's own metrics, and two harmonised columns:

- `outcome`: AlpaSim `collision | off-road | clean` within the span AlpaSim scores; HUGSIM how the
  episode ended, `collision | off-route | complete | step-cap`.
- `failed`: AlpaSim `offroad_or_collision`; HUGSIM `collision`.

They are not interchangeable: a roadside barrier is off-road in AlpaSim (scenery has no collision
geometry) and a background collision in HUGSIM (`HUGSIM.md`, caveats). Compare where and when a
policy fails, not metric names.

## Caveats

- The condition is the dataset's label, not the pictures: `ba91fe2c` is tagged 02:00 and was filmed
  in daylight (`nurec/README.md`, fidelity table note). 718 NuRec scenes have no `hour_of_day` in the
  on-disk metadata (`unknown`).
- `nvidia_gold.episode` includes the decision windows of the 143-clip NuRec slice
  (`episodes_nurec_slice.parquet`) beside the on-disk pass; de-duplication is on `episode_id`.
- Twin rows are found on disk: rerun the twins step after a twin pipeline finishes.

## Rebuild

```bash
# 1. episodes (host); the full on-disk pass takes hours, the twins pass seconds
.skip_venv/bin/python episodes.py --out ../user_data/episodes_ondisk.parquet --scenarios-out ../user_data/scenarios_ondisk.parquet
.skip_venv/bin/python episodes.py --no-decision --no-aug --no-nurec --out ../user_data/episodes_twins.parquet --scenarios-out ../user_data/scenarios_twins.parquet
# 2. land them (spark-iceberg container; reads /user_data only)
docker exec -w /opt/spark spark-iceberg /opt/spark/bin/spark-submit nvidia_ingestion/build_episode_tables.py
# 3. closed-loop rollouts, keyed by episode
.skip_venv/bin/python rollouts.py
docker exec -w /opt/spark spark-iceberg /opt/spark/bin/spark-submit nvidia_ingestion/build_rollout_table.py
# 4. the figure
.skip_venv/bin/python fig_scenario_space.py
```

Files written before 2026-10-06 carry the old column names `validator_mode` and `n_agents`; the
table builder renames them on read.
