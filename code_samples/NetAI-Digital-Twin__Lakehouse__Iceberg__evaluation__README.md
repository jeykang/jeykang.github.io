# evaluation — Tier 1 driving-policy evaluation over the lakehouse

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

Three reference policies exist so a number is interpretable. Measured on 149 random
clips:

| policy | MF-PDMS | NC | TTC | EP | HC | EC |
|---|---|---|---|---|---|---|
| `replay_human` *(oracle)* | 0.980 | 1.000 | 0.996 | 1.000 | 0.937 | 0.937 |
| `constant_velocity` | 0.970 | 0.996 | 0.984 | 0.940 | 1.000 | 1.000 |
| `stationary` | 0.622 | 0.970 | 0.959 | 0.007 | 1.000 | 1.000 |

The oracle replays recorded ground truth, so it must sit at the top of the scale —
it is a self-test, not a baseline. It has already earned its keep: the first
implementation scored the *human's own driving* at EC=0.62, which turned out to be
finite-difference noise (three derivatives of sampled position amplify jitter by
1/dt^3). Comfort now uses least-squares cubic fits instead.

Note how close `constant_velocity` sits to the oracle on random clips. That is not a
bug — it is the well-known weakness of open-loop AV metrics, where 4 s of ordinary
driving is nearly straight and ego extrapolation alone scores well. It is also the
argument for curated eval suites, which is what the lakehouse is for.

## Usage

```bash
# calibrate
python run_eval.py --policy replay_human --limit 200 --workers 8
python run_eval.py --policy stationary   --limit 200 --workers 8

# a curated slice, ranked by any score column in the lakehouse
python run_eval.py --policy constant_velocity --workers 8 \
  --clips-from-parquet <NFS>/.conflict/conflict_shard_00_of_01.parquet \
  --rank-col conflict_score --top-frac 0.1

# land results in Iceberg beside the curation scores
python publish.py .results_constant_velocity.parquet --run-id nightly-2026-08-11
```

Cost: ~0.27 s/clip at `--workers 8`. Loading is NFS-bound (~0.94 s/clip serial);
scoring itself is 0.016 s/clip, so workers are near-linear speedup. A 3,174-clip
Gold tier is ~15 minutes.

## Adding a dataset

The pipeline is multi-dataset by construction: metrics and harness see only
`scenario.Scenario`, and nothing downstream imports a dataset module. Implement
`scenario.DatasetAdapter` — `list_clips()` and `load(clip_id) -> Scenario` — and
register it in `adapters.ADAPTERS`. Required source data is only **agent tracks and
ego poses**, which every AV dataset has.

The one thing an adapter must get right is the frame convention: ego poses and
agent boxes in a single per-clip world frame, metres, yaw CCW from +x. Datasets that
store agents in a per-timestamp rig frame (NVIDIA PhysicalAI does) must lift each
box using the ego pose at *that box's* reference timestamp — see
`NvidiaAdapter._agents`.

## Files

| file | role |
|---|---|
| `scenario.py` | `Scenario` / `Observation` types and the `DatasetAdapter` contract |
| `adapters.py` | `NvidiaAdapter` (rig→world lift, footprint, tracks) |
| `metrics.py` | MF-PDMS sub-metrics, OBB collision, comfort |
| `harness.py` | decision times, no-future `Observation` construction, scoring |
| `policies.py` | `Policy` contract + oracle and naive baselines |
| `run_eval.py` | CLI |
| `publish.py` | optional Iceberg write (`eval.policy_runs`) |

`run_eval.py` deliberately has no Spark dependency — the evaluation pipeline should
be usable by people who do not run this lakehouse. `publish.py` is the opt-in step
for people who do.
