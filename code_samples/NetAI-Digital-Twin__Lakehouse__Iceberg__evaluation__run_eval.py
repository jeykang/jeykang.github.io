#!/usr/bin/env python3
"""CLI: evaluate a driving policy over a slice of the lakehouse.

    # calibrate the harness (oracle must be ~1.0, stationary must be safe-and-useless)
    python run_eval.py --policy replay_human --limit 200
    python run_eval.py --policy stationary   --limit 200

    # a consumer's own model
    python run_eval.py --policy mypkg.planner:MyPlanner --clips-file gold.txt

    # a curated slice straight out of the lakehouse
    python run_eval.py --policy constant_velocity \
        --clips-from-parquet <NFS>/.conflict/conflict_shard_00_of_01.parquet \
        --rank-col conflict_score --top-frac 0.1

Writes a parquet of per-clip rows; `publish.py` lands it in Iceberg.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import random
import statistics as st
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import adapters                     # noqa: E402
import harness                      # noqa: E402
import policies as P                # noqa: E402


def load_policy(spec: str, adapter):
    if spec in P.BUILTIN:
        return P.BUILTIN[spec]()
    if ":" not in spec:
        raise SystemExit(f"unknown policy '{spec}'; builtins: "
                         f"{sorted(P.BUILTIN) + ['replay_human']}, or use module:Class")
    mod, cls = spec.split(":", 1)
    return getattr(importlib.import_module(mod), cls)()


def select_clips(a, adapter):
    if a.clips_file:
        ids = [l.strip().split(",")[0] for l in open(a.clips_file) if l.strip()]
    elif a.clips_from_parquet:
        import pyarrow.parquet as pq
        cols = ["clip_id"] + ([a.rank_col] if a.rank_col else [])
        d = pq.read_table(a.clips_from_parquet, columns=cols).to_pydict()
        ids = list(d["clip_id"])
        if a.rank_col:
            ids = [c for _, c in sorted(zip(d[a.rank_col], ids), reverse=True)]
            if a.top_frac:
                ids = ids[:max(1, int(len(ids) * a.top_frac))]
    else:
        ids = list(adapter.list_clips())
    if a.shuffle:
        random.Random(a.seed).shuffle(ids)
    return ids[:a.limit] if a.limit else ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="nvidia")
    ap.add_argument("--policy", default="constant_velocity")
    ap.add_argument("--clips-file")
    ap.add_argument("--clips-from-parquet")
    ap.add_argument("--rank-col", help="column to rank by (e.g. a difficulty score)")
    ap.add_argument("--top-frac", type=float, help="keep this top fraction after ranking")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shuffle", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--workers", type=int, default=1,
                    help="parallel clip loading; scoring is I/O-bound")
    a = ap.parse_args()

    adapter = adapters.get_adapter(a.dataset)
    policy = load_policy(a.policy, adapter)
    clips = select_clips(a, adapter)
    print(f"[eval] dataset={adapter.name} policy={policy.name} clips={len(clips)} "
          f"horizon={harness.HORIZON_S}s dt={harness.DT_S}s", flush=True)

    t0 = time.time()
    rows = harness.evaluate(policy, adapter, clips, workers=a.workers)
    el = time.time() - t0
    if not rows:
        raise SystemExit("[eval] no clips scored — check the adapter root / clip ids")

    print(f"\n===== MF-PDMS  policy={policy.name}  n={len(rows)} clips "
          f"({el/len(rows):.3f}s/clip) =====")
    if rows[0]["is_oracle"]:
        print("  *** ORACLE — replays ground truth; a calibration reference, not a baseline")
    for k in ("mf_pdms", "nc", "ttc", "ep", "hc", "ec"):
        v = [r[k] for r in rows]
        print(f"  {k:<8} mean={st.mean(v):.3f}  median={st.median(v):.3f}  "
              f"min={min(v):.3f}  max={max(v):.3f}")
    coll = sum(1 for r in rows if r["nc"] < 1.0)
    print(f"  clips with an at-fault collision at some decision point: "
          f"{coll}/{len(rows)} ({coll/len(rows):.1%})")
    print("  NOTE: MF-PDMS omits DAC/DDC/TLC/LK (no map). Not comparable to EPDMS.")

    out = a.out or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                f".results_{policy.name}.parquet")
    try:
        import pyarrow as pa, pyarrow.parquet as pq
        pq.write_table(pa.Table.from_pylist(rows), out)
    except Exception:
        out = out.replace(".parquet", ".json")
        json.dump(rows, open(out, "w"))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
