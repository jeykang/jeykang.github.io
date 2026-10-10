# evaluation — Tier 1 driving-policy evaluator over the lakehouse

Terms follow [TERMINOLOGY.md](../TERMINOLOGY.md) (evaluator, driving policy, actors, policy bridge, rollout triage, serving mode).

Open-loop / pseudo-simulation scoring of a driving policy against curated slices of
the lakehouse. Runs on every clip we hold, needs no map, no sensor data, no
simulator, and no NVIDIA component. See [FEASIBILITY.md](FEASIBILITY.md) for why
this tier exists and what Tier 2 (closed-loop via AlpaSim) would add.

## What a consumer has to do

Implement one method:

```python
class MyPlanner:
    name = "my-planner"
    def plan(self, obs):                 # obs: history only, never the future
        return [(x, y), ...]             # obs.n_steps points, ego frame at t0,
                                         # +x forward, +y left, at t0+(k+1)*dt
```

```bash
python run_eval.py --policy mypkg.planner:MyPlanner --workers 8
```

No dataset knowledge, no map handling, no I/O. That is the whole contract.

A **vision** model additionally reads `obs.sensors`, a history-bounded reader that
refuses any request past the decision time or outside recorded sensor coverage, and
sets `needs_sensors = True` so decision points are restricted to frames that exist.
`policy_alpamayo.py` is a worked example: it builds Alpamayo's 4-camera / 16-step
inputs from the Observation alone, and was verified bit-identical to the model's own
dataset loader (ego history to 0.0019 m, decoded pixels to 0).

## What it reports

**MF-PDMS** — map-free PDMS, the five EPDMS sub-metrics computable without an HD map:

| term | weight | meaning |
|---|---|---|
| NC | multiplier | no at-fault collision over the horizon |
| TTC | 5 | no collision under a 1 s constant-velocity projection |
| EP | 5 | path length achieved vs the human's, same horizon |
| HC | 2 | comfort relative to how this clip was actually driven |
| EC | 2 | comfort against absolute (nuPlan-derived) bounds |

`MF-PDMS = NC * (5*TTC + 5*EP + 2*HC + 2*EC) / 14`

**It is not EPDMS and must never be reported as one.** EPDMS's other four terms
(DAC, DDC, TLC, LK) need map layers this dataset has no equivalent of, and since
DAC/DDC/TLC are multipliers, dropping them makes EPDMS *undefined* rather than
merely degraded. MF-PDMS scores are comparable to each other, never to a published
EPDMS number.

## Calibration — read this before trusting a score

Reference policies exist so a number is interpretable. All rows below are the **same
60 clips at the same decision points** (`--require-sensors`, which every policy in a
comparison must use once one of them consumes sensors):

| policy | MF-PDMS | NC | TTC | EP | HC | EC | clips w/ collision |
|---|---|---|---|---|---|---|---|
| `replay_human` *(oracle)* | 0.949 | 0.994 | 0.871 | 1.000 | 0.997 | 0.997 | 1.8% |
| **`alpamayo_1_5`** *(real VLA model)* | **0.884** | 0.988 | 0.825 | 0.877 | 0.993 | 0.991 | **3.5%** |
| `constant_velocity` | 0.853 | 0.942 | 0.801 | 0.880 | 1.000 | 1.000 | 15.8% |
| `stationary` | 0.421 | 0.690 | 0.591 | 0.035 | 1.000 | 1.000 | 38.6% |

The ordering is the one a working benchmark should produce: oracle > real model >
naive > degenerate. Alpamayo-1.5-10B lands between ground truth and constant
velocity, and the interesting detail is *where*: its collision rate is 3.5% against
constant velocity's 15.8% — near-oracle safety — while its progress (EP 0.877) is
essentially the same as the naive baseline's 0.880. It drives carefully, slightly
conservatively. That is a believable portrait of a real planner, and it is the
evidence that the evaluator measures something.

Note also that the MF-PDMS aggregate compresses this: 0.884 vs 0.853 is a 0.031 gap,
while the safety term underneath differs by 4.5x. When comparing policies, read NC
and TTC, not just the aggregate.

### Curation sharpens the suite

Same policies on the top 60 clips by `conflict_score`:

| policy | MF-PDMS | NC | TTC | clips w/ collision |
|---|---|---|---|---|
| `replay_human` *(oracle)* | 0.829 | 0.883 | 0.733 | 21.7% |
| `constant_velocity` | 0.656 | 0.739 | 0.572 | 50.0% |
| `stationary` | 0.337 | 0.472 | 0.378 | 71.7% |

| | random slice | curated slice | change |
|---|---|---|---|
| oracle − constant_velocity gap | 0.096 | **0.173** | **1.8x** |
| constant_velocity collision rate | 15.8% | **50.0%** | **3.2x** |

Selecting on a traffic-interaction axis makes the benchmark measurably better at
separating a trivial baseline from ground truth. That is the lakehouse's value
proposition for evaluation, measured rather than asserted.

### The bug that a real model exposed

An earlier version of this README reported the random-slice oracle−baseline gap as
**0.010** and concluded that open-loop metrics "barely separate a trivial baseline
from ground truth". **That was my bug, not a property of the metric.**

`decision_times()` derived its window from the **ego** span. On this dataset
egomotion runs to ~140 s while obstacle labels and video stop at ~20 s, so **89% of
decision points landed outside the annotated window** — scoring collisions against
scenes containing no annotated actors, where every safety metric is trivially
perfect. Fixing it to intersect ego ∩ actor ∩ (optionally) sensor coverage moved the
oracle−baseline gap from 0.010 to 0.096 and stationary's collision rate from 4.7% to
38.6%.

Two things worth carrying forward. First, the track-only baselines could never have
revealed this: they ran happily on empty scenes and produced plausible numbers.
It surfaced only when a **sensor-consuming** policy asked for camera frames 22 s past
the end of the video. Plugging in a real model is not just a demo — it exercises
constraints that synthetic baselines cannot. Second, `NvidiaSensorReader.frames()`
originally *clamped* out-of-range requests to the nearest frame, silently pairing
stale pixels with fresh ego history; it now raises. Silent nearest-neighbour lookup
across a coverage gap is exactly how confidently-wrong numbers get produced.

### Known limitation: the collision false-positive floor

The oracle replays recorded human driving, so its collision rate is a pure false
positive rate: **1.8% on random clips, 21.7% on the curated slice**. Humans do not
crash a fifth of the time in dense traffic — on crowded scenes the OBB check
over-triggers from track jitter and box-size noise, and `metrics.no_collision` uses a
deliberately simplified at-fault rule (it only excludes strikes from behind, since
right-of-way needs a map). Read absolute NC against that floor, and prefer
policy-minus-oracle deltas on the same slice over absolute values.

## Usage

```bash
# calibrate (add --require-sensors to match a sensor model's decision points)
python run_eval.py --policy replay_human --limit 60 --workers 8 --require-sensors
python run_eval.py --policy stationary   --limit 60 --workers 8 --require-sensors

# a real driving model (needs the vendored Alpamayo venv; ~22 GB VRAM, workers=1)
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=. \
  ../planning/alpamayo/alpamayo1.5/a1_5_venv/bin/python run_eval.py \
  --policy policy_alpamayo:AlpamayoPolicy --limit 60 --workers 1 --require-sensors

# a curated slice, ranked by any score column in the lakehouse
python run_eval.py --policy constant_velocity --workers 8 \
  --clips-from-parquet <NFS>/.conflict/conflict_shard_00_of_01.parquet \
  --rank-col conflict_score --top-frac 0.1

# land results in Iceberg beside the curation scores
python publish.py .results_constant_velocity.parquet --run-id nightly-2026-08-11
```

Cost: ~0.5 s/clip for track-only policies at `--workers 8` (loading is NFS-bound;
scoring itself is 0.016 s/clip, so workers are near-linear). Alpamayo-1.5-10B is
**26.4 s/clip** at `--workers 1` — three decision points, each a VLM rollout plus a
diffusion action expert, on one A10 at 23.2 GB peak. Budget ~7 h for a 1,000-clip
model evaluation, or shard it.

## Adding a dataset

The pipeline is multi-dataset by construction: metrics and evaluator see only
`scenario.Scenario`, and nothing downstream imports a dataset module. Implement
`scenario.DatasetAdapter` — `list_clips()` and `load(clip_id) -> Scenario` — and
register it in `adapters.ADAPTERS`. Required source data is only **actor tracks and
ego poses**, which every AV dataset has.

The one thing an adapter must get right is the frame convention: ego poses and
actor boxes in a single per-clip world frame, metres, yaw CCW from +x. Datasets that
store actors in a per-timestamp rig frame (NVIDIA PhysicalAI does) must lift each
box using the ego pose at *that box's* reference timestamp — see
`NvidiaAdapter._agents`.

## Serving, budgeting and the episode space (added 2026-09-21)

Three tools sit beside the evaluator; none imports it or AlpaSim, they consume outputs.

**`materialize.py MODE`** — serve a Gold selection in a validator's compatibility
mode (호환 모드로 진열). `openloop` writes a clip list for `run_eval.py`; `nurec`
builds a `LOCAL_USDZ_DIR` directory of NuRec scenes (cached ones hardlinked, missing
ones fetched with `--download`, clips with no artifact listed as reconstruction
candidates); `ncore` stages each clip in NVIDIA's `pai-clip-dl` layout and runs
NVIDIA's own PAI→NCore converter (`ncore/README.md`). Every mode writes a
`manifest.json` naming what was served and what was skipped.

**`skip.py`** — rollout triage, the validation-budget tool. `features` joins the curation axes,
the open-loop reference ladder and closed-loop per-clip outcomes into one table;
`fit` evaluates a leave-one-out screen (predicted closed-loop failure) against the
baselines as recall-vs-budget; `select` ranks every featured clip and emits the
rollout list for a budget — the dial. Needs `.skip_venv` (numpy, pandas,
scikit-learn). Results in `SKIP.md`.

**`episodes.py`** — makes the scenario × episode space explicit: one row per
decision window (`harness.decision_times`), Cosmos augmentation window
(`cosmos_augmentation/batch_manifest.json`) and NuRec scene, each tagged with a
`scenario_id` = recording condition × augmentation × serving mode. Writes
`user_data/episodes_*.parquet`; `nvidia_ingestion/build_episode_tables.py` lands them
as `nvidia_gold.episode` and `nvidia_gold.scenario`. Slice roots need
`--clips-file`, because the adapter's clip list is a cache of the on-disk dataset.

The curation-axis runners (`planning/conflict_runner.py`, `behavioral_runner.py`,
`camera_perception_runner.py`) take `NFS_ROOT=<dir>` to score a slice such as
`.av_slice_nurec` instead of the NFS subset; the NuRec slice is scored that way
because its clips are not in the on-disk subset.

## Files

| file | role |
|---|---|
| `scenario.py` | `Scenario` / `Observation` types and the `DatasetAdapter` contract |
| `adapters.py` | `NvidiaAdapter` (rig→world lift, footprint, tracks) |
| `metrics.py` | MF-PDMS sub-metrics, OBB collision, comfort |
| `evaluator.py` (`harness.py` is a re-export shim) | decision times, no-future `Observation` construction, scoring; per-decision rows with `run_eval.py --per-decision` |
| `policies.py` | `Policy` contract + oracle and naive baselines |
| `policy_alpamayo.py` | Alpamayo-1.5-10B (real VLA driving model) through the same contract |
| `run_eval.py` | CLI |
| `publish.py` | optional Iceberg write (`eval.policy_runs`) |
| `BENCHMARKS.md` | recorded scores, cost and resources per run; model-availability notes |
| `materialize.py` | serve a Gold selection in a serving mode (openloop / nurec / ncore) |
| `skip.py` | validation-budget screen: features / fit / select |
| `episodes.py` | scenario × episode rows for Iceberg (`nvidia_ingestion/build_episode_tables.py`) |
| `alpasim/` | closed-loop layer: policy bridge (AlpaSim plugin, `driver=harness`), `run_scene.sh` (catalog or `LOCAL_USDZ_DIR`), `per_clip.py` |
| `ncore/` | NCore v4 serving mode: staging script + vendored NVIDIA converter |

`run_eval.py` deliberately has no Spark dependency — the evaluation pipeline should
be usable by people who do not run this lakehouse. `publish.py` is the opt-in step
for people who do.
