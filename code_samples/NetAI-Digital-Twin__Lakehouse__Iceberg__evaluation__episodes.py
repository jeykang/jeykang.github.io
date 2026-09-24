#!/usr/bin/env python3
"""episodes.py — make the scenario x episode space an explicit object.

An *episode* is a time window of one clip that a validator actually scores; a
*scenario* is the class the episode belongs to (recording condition x augmentation
x validator mode). Both have been implicit: the evaluator (harness.py) picks decision windows in
memory, the Cosmos batch manifest records agent windows, and the NuRec catalog
decides which clips can run closed-loop. This pre-pass enumerates all three as
rows so they can be landed in Iceberg (nvidia_ingestion/build_episode_tables.py)
and joined to evaluation results and curation scores.

Sources (each row says which):
  decision_window   harness.decision_times() over a clip set, one row per decision
                    point, window = [t0 - HISTORY_S, t0 + HORIZON_S]
  aug_window        cosmos_augmentation/batch_manifest.json, the 121-frame agent
                    window each Cosmos variant was rendered from, one row per variant
  nurec_scene       AlpaSim's sim_scenes.csv: every clip with a NuRec reconstruction
                    (closed-loop capable), whole-scene window unknown until opened

Columns follow the canonical Episode table (episode_id, from_clip_id, to_clip_id,
frame_id_list) and add what the validators need. Runs on the host, not in Spark:
it needs the evaluator's adapters. Output: two parquet files under user_data/, which
the spark-iceberg container sees at /user_data.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from multiprocessing import Pool

import pyarrow as pa
import pyarrow.parquet as pq

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from adapters import NvidiaAdapter, NVIDIA_ROOT  # noqa: E402
from harness import decision_times, HORIZON_S, HISTORY_S, DECISION_FRACS  # noqa: E402

ROOT_REPO = os.path.dirname(HERE)
META = "/mnt/netai-e2e/nvidia-physicalai-av-subset/metadata/data_collection.parquet"
MANIFEST = os.path.join(ROOT_REPO, "cosmos_augmentation", "batch_manifest.json")
SIM_SCENES = os.path.join(HERE, "alpasim", "repo", "data", "scenes", "sim_scenes.csv")
AUG_FRAMES = 121            # Cosmos-Transfer1 window length (stage_batch.py)
WIN_US = 100_000            # agent presence tolerance at t0, same as the runners

# Recording condition from hour_of_day, the same bands edge_case_scorer's
# time-of-day axis uses (see _time_of_day_score there): night is the hard end.
def condition_of(hour) -> str:
    if hour is None:
        return "unknown"
    h = int(hour)
    if h <= 5 or h >= 21:          # _HOUR_DIFFICULTY 1.0 (0-5) and 0.8 (21-23)
        return "night"
    if h in (6, 7, 18, 19, 20):    # 0.7
        return "dawn_dusk"
    return "day"                   # 8-17


VALIDATOR_MODE = {"decision_window": "openloop-mfpdms", "aug_window": "augmented-openloop",
                  "nurec_scene": "closedloop-nurec"}


def scenario_id(condition: str, augmentation: str, mode: str) -> str:
    return f"{condition}|{augmentation}|{mode}"


# ----------------------------------------------------------------------------- decision windows
_ADAPTER = None


def _init(root):
    global _ADAPTER
    _ADAPTER = NvidiaAdapter(root)


def _decision_rows(clip_id: str):
    try:
        sc = _ADAPTER.load(clip_id)
    except Exception as e:                       # a bad clip must not kill the pass
        return [], f"{clip_id[:8]}: {str(e)[:60]}"
    if sc is None:
        return [], None
    rows = []
    for req in (False, True):
        for k, t0 in enumerate(decision_times(sc, require_sensors=req)):
            # agents present at t0: nearest sample of each track within WIN_US
            n_ag = 0
            for tr in sc.agents.values():
                if any(abs(b.t_us - t0) <= WIN_US for b in tr):
                    n_ag += 1
            rows.append({
                "episode_id": f"{clip_id}:decision:{t0}:{'s' if req else 'a'}",
                "from_clip_id": clip_id, "to_clip_id": clip_id, "frame_id_list": None,
                "clip_id": clip_id, "kind": "decision_window", "source": "harness.decision_times",
                "t0_us": int(t0), "t_start_us": int(t0 - HISTORY_S * 1e6),
                "t_end_us": int(t0 + HORIZON_S * 1e6),
                "horizon_s": HORIZON_S, "history_s": HISTORY_S, "decision_frac": DECISION_FRACS[k],
                "require_sensors": req, "n_agents": n_ag,
                "augmentation": "none", "validator_mode": VALIDATOR_MODE["decision_window"],
            })
    return rows, None


def decision_windows(root: str, clips, workers: int):
    rows, errs = [], 0
    t0 = time.time()
    with Pool(workers, initializer=_init, initargs=(root,)) as pool:
        for i, (r, err) in enumerate(pool.imap_unordered(_decision_rows, clips, chunksize=8)):
            rows.extend(r)
            errs += err is not None
            if (i + 1) % 500 == 0:
                print(f"[episodes] {i+1}/{len(clips)} clips, {len(rows)} windows, "
                      f"{(i+1)/(time.time()-t0):.1f} clips/s", flush=True)
    print(f"[episodes] decision windows: {len(rows)} rows from {len(clips)} clips ({errs} unreadable)")
    return rows


# ----------------------------------------------------------------------------- augmentation windows
def aug_windows():
    if not os.path.exists(MANIFEST):
        return []
    rows = []
    for e in json.load(open(MANIFEST)):
        fps = float(e.get("fps") or 30.0)
        t_start = int(e["window_start"] / fps * 1e6)
        t_end = int((e["window_start"] + AUG_FRAMES) / fps * 1e6)
        rows.append({
            "episode_id": f"{e['clip']}:aug:{e['window_start']}:{e['cond']}",
            "from_clip_id": e["clip"], "to_clip_id": e["clip"], "frame_id_list": None,
            "clip_id": e["clip"], "kind": "aug_window", "source": "cosmos_augmentation/batch_manifest.json",
            "t0_us": t_start, "t_start_us": t_start, "t_end_us": t_end,
            "horizon_s": AUG_FRAMES / fps, "history_s": 0.0, "decision_frac": None,
            "require_sensors": True, "n_agents": int(e.get("window_agents") or 0),
            "augmentation": f"cosmos-{e['cond']}", "validator_mode": VALIDATOR_MODE["aug_window"],
        })
    print(f"[episodes] augmentation windows: {len(rows)} rows")
    return rows


# ----------------------------------------------------------------------------- nurec scenes
def nurec_scenes():
    if not os.path.exists(SIM_SCENES):
        return []
    newest = {}
    for r in csv.DictReader(open(SIM_SCENES)):
        sid = r["scene_id"]
        if sid not in newest or r["last_modified"] > newest[sid]["last_modified"]:
            newest[sid] = r
    rows = []
    for sid, r in newest.items():
        clip = sid.replace("clipgt-", "")
        rows.append({
            "episode_id": f"{clip}:nurec:{r['uuid']}",
            "from_clip_id": clip, "to_clip_id": clip, "frame_id_list": None,
            "clip_id": clip, "kind": "nurec_scene", "source": f"alpasim sim_scenes.csv {r['hf_revision']}",
            "t0_us": None, "t_start_us": None, "t_end_us": None,
            "horizon_s": None, "history_s": None, "decision_frac": None,
            "require_sensors": True, "n_agents": None,
            "augmentation": "none", "validator_mode": VALIDATOR_MODE["nurec_scene"],
        })
    print(f"[episodes] nurec scenes: {len(rows)} rows")
    return rows


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=NVIDIA_ROOT, help="dataset root for decision windows (AV_ROOT)")
    ap.add_argument("--clips-file", help="clip ids, one per line (default: every clip under --root)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--no-decision", action="store_true")
    ap.add_argument("--no-aug", action="store_true")
    ap.add_argument("--no-nurec", action="store_true")
    ap.add_argument("--out", default=os.path.join(ROOT_REPO, "user_data", "episodes.parquet"))
    ap.add_argument("--scenarios-out", default=os.path.join(ROOT_REPO, "user_data", "scenarios.parquet"))
    a = ap.parse_args()

    rows = []
    if not a.no_decision:
        ad = NvidiaAdapter(a.root)
        # list_clips() serves a cached list of the on-disk dataset regardless of root,
        # so a slice root must come with its own clip list
        if a.clips_file:
            clips = [l.strip() for l in open(a.clips_file) if l.strip()]
        elif os.path.abspath(a.root) != os.path.abspath(NVIDIA_ROOT):
            sys.exit(f"--clips-file is required when --root is not the dataset root ({NVIDIA_ROOT})")
        else:
            clips = list(ad.list_clips())
        if a.limit:
            clips = clips[:a.limit]
        print(f"[episodes] root={a.root} clips={len(clips)} workers={a.workers}")
        rows += decision_windows(a.root, clips, a.workers)
    if not a.no_aug:
        rows += aug_windows()
    if not a.no_nurec:
        rows += nurec_scenes()
    if not rows:
        sys.exit("nothing to write")

    # recording condition from the dataset's own metadata (hour_of_day)
    cond = {}
    if os.path.exists(META):
        need = {r["clip_id"] for r in rows}
        t = pq.read_table(META, columns=["clip_id", "hour_of_day"]).to_pydict()
        cond = {c: condition_of(h) for c, h in zip(t["clip_id"], t["hour_of_day"]) if c in need}
    for r in rows:
        r["condition"] = cond.get(r["clip_id"], "unknown")
        r["scenario_id"] = scenario_id(r["condition"], r["augmentation"], r["validator_mode"])
        r["created_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    # explicit schema: an all-null column (frame_id_list, or t0_us in a nurec-only
    # run) must not be inferred as a null type, or the files will not union in Spark
    schema = pa.schema([
        ("episode_id", pa.string()), ("from_clip_id", pa.string()), ("to_clip_id", pa.string()),
        ("frame_id_list", pa.string()), ("clip_id", pa.string()), ("kind", pa.string()),
        ("source", pa.string()), ("t0_us", pa.int64()), ("t_start_us", pa.int64()),
        ("t_end_us", pa.int64()), ("horizon_s", pa.float64()), ("history_s", pa.float64()),
        ("decision_frac", pa.float64()), ("require_sensors", pa.bool_()), ("n_agents", pa.int32()),
        ("augmentation", pa.string()), ("validator_mode", pa.string()), ("condition", pa.string()),
        ("scenario_id", pa.string()), ("created_at", pa.string()),
    ])
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), a.out)
    seen = {}
    for r in rows:
        s = seen.setdefault(r["scenario_id"], {"scenario_id": r["scenario_id"], "condition": r["condition"],
                                               "augmentation": r["augmentation"],
                                               "validator_mode": r["validator_mode"], "n_episodes": 0,
                                               "n_clips": set()})
        s["n_episodes"] += 1; s["n_clips"].add(r["clip_id"])
    scen = [{**s, "n_clips": len(s["n_clips"])} for s in seen.values()]
    pq.write_table(pa.Table.from_pylist(scen), a.scenarios_out)
    print(f"[episodes] wrote {a.out} ({len(rows)} episodes) and {a.scenarios_out} ({len(scen)} scenarios)")
    for s in sorted(scen, key=lambda s: -s["n_episodes"]):
        print(f"  {s['scenario_id']:48s} episodes={s['n_episodes']:6d} clips={s['n_clips']:6d}")


if __name__ == "__main__":
    main()
