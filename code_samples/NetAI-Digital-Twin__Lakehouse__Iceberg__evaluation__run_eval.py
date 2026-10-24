#!/usr/bin/env python3
"""CLI: evaluate a driving policy over a slice of the lakehouse.

    # calibrate the evaluator (oracle must be ~1.0, stationary must be safe-and-useless)
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
import evaluator                    # noqa: E402
import policies as P                # noqa: E402


def load_policy(spec: str, adapter):
    if spec in P.BUILTIN:
        return P.BUILTIN[spec]()
    if ":" not in spec:
        raise SystemExit(f"unknown policy '{spec}'; builtins: "
                         f"{sorted(P.BUILTIN)}, or use module:Class")
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


def _model_stats(policy) -> dict:
    """Size the thing under test: parameters, dtype, sharding, checkpoint bytes.

    Wall-clock alone is not comparable across models — a 34B sharded over 4 GPUs and
    a 10B on 1 GPU cost very differently per clip.
    """
    out = {}
    # Policies name their module differently (Alpamayo: .model, VaVAM: .vam,
    # DiffusionDrive: .agent). Looking only for `.model` reported VaVAM as CPU-only
    # while it was using 4 GB of VRAM.
    m = next((getattr(policy, a) for a in ("model", "vam", "agent", "net")
              if getattr(policy, a, None) is not None), None)
    if m is None:
        return out
    try:
        params = sum(p.numel() for p in m.parameters())
        out.update(params=params, params_b=round(params / 1e9, 2),
                   dtype=str(next(m.parameters()).dtype))
        devs = sorted({str(p.device) for p in m.parameters()})
        out.update(param_devices=devs, sharded_over=len(devs))
    except Exception:
        pass
    mid = getattr(policy, "default_model", None)
    if mid:
        out["model_id"] = mid
        try:
            from huggingface_hub import snapshot_download
            d = snapshot_download(mid, local_files_only=True)
            out["checkpoint_gb"] = round(_dir_bytes(d) / 1e9, 2)
        except Exception:
            pass
    return out


def _dir_bytes(path) -> int:
    tot = 0
    for r, _, fs in os.walk(path):
        for f in fs:
            fp = os.path.join(r, f)
            try:
                tot += os.path.getsize(fp)
            except OSError:
                pass
    return tot


def _resources(device_hint=None) -> dict:
    """Best-effort host/GPU facts. Recorded per run so a benchmark row is reproducible
    and so cost can be compared across policies and hardware later."""
    import platform
    out = {"host": platform.node(), "python": platform.python_version(),
           "cpu_count": os.cpu_count()}
    try:
        import torch
        out["torch"] = torch.__version__
        gpus = []
        for i in range(torch.cuda.device_count()):
            pr = torch.cuda.get_device_properties(i)
            gpus.append({"index": i, "name": pr.name,
                         "capability": f"sm_{pr.major}{pr.minor}",
                         "total_gb": round(pr.total_memory / 1e9, 2)})
        out["gpus"] = gpus
        if torch.cuda.is_available():
            out["peak_vram_gb"] = round(
                max(torch.cuda.max_memory_allocated(i) for i in
                    range(torch.cuda.device_count())) / 1e9, 2)
            out["peak_vram_reserved_gb"] = round(
                max(torch.cuda.max_memory_reserved(i) for i in
                    range(torch.cuda.device_count())) / 1e9, 2)
    except Exception:
        out["torch"] = None
    return out


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
    ap.add_argument("--per-decision", default=None,
                    help="also write one row per scored decision point (episode_id = nvidia_gold.episode's)")
    ap.add_argument("--workers", type=int, default=1,
                    help="parallel clip loading; scoring is I/O-bound")
    ap.add_argument("--require-sensors", action="store_true",
                    help="restrict decision times to sensor coverage. Set this for "
                         "EVERY policy in a comparison that includes a sensor-based "
                         "model, or policies get different decision points and the "
                         "numbers are not comparable.")
    a = ap.parse_args()

    adapter = adapters.get_adapter(a.dataset)
    policy = load_policy(a.policy, adapter)
    if a.require_sensors:
        policy.needs_sensors = True
    clips = select_clips(a, adapter)
    print(f"[eval] dataset={adapter.name} policy={policy.name} clips={len(clips)} "
          f"horizon={evaluator.HORIZON_S}s dt={evaluator.DT_S}s", flush=True)

    t0 = time.time()
    rows = evaluator.evaluate(policy, adapter, clips, workers=a.workers)
    el = time.time() - t0
    if not rows:
        raise SystemExit("[eval] no clips scored — check the adapter root / clip ids")
    decisions = [dict(d, clip_id=r["clip_id"], policy=r["policy"]) for r in rows for d in r.pop("_decisions", [])]

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

    # ── resource + throughput record ────────────────────────────────────────
    walls = sorted(r["wall_s"] for r in rows if "wall_s" in r)
    n_dec = sum(r["n_decisions"] for r in rows)
    res = _resources()
    mstats = _model_stats(policy)
    plan_tot = sum(r.get("plan_s", 0.0) for r in rows)
    load_tot = sum(r.get("scenario_load_s", 0.0) for r in rows)
    # GPUs actually used by THIS policy. A track-only baseline touches none, so
    # counting the machine's cards would invent a cost it never paid; and with
    # workers>1 the summed per-clip times exceed wall-clock, so fractions must be
    # taken against worker-time, not wall-time.
    if mstats:
        ngpu = (mstats.get("sharded_over")
                or int(os.environ.get("SLURM_GPUS_ON_NODE") or 0) or 1)
    else:
        ngpu = 0
    gpu_s_clip = ((el / max(1, len(rows))) * ngpu) if ngpu else 0.0
    worker_time = el * max(1, a.workers)
    meta = {
        "policy": policy.name, "dataset": adapter.name,
        "clips_requested": len(clips), "clips_scored": len(rows),
        "decisions_scored": n_dec,
        "workers": a.workers, "require_sensors": bool(a.require_sensors),
        "horizon_s": evaluator.HORIZON_S, "dt_s": evaluator.DT_S,
        "decision_fracs": list(evaluator.DECISION_FRACS),
        "wall_total_s": round(el, 1),
        "s_per_clip": round(el / max(1, len(rows)), 3),
        "s_per_decision": round(el / max(1, n_dec), 3),
        "clips_per_hour": round(3600 * len(rows) / max(1e-9, el), 1),
        "s_per_clip_p50": round(walls[len(walls) // 2], 3) if walls else None,
        "s_per_clip_p95": round(walls[int(len(walls) * 0.95)], 3) if walls else None,
        "s_per_clip_max": round(walls[-1], 3) if walls else None,
        "resources": res,
        "model": mstats,
        # where the time actually goes
        "plan_total_s": round(plan_tot, 1),
        "scenario_load_total_s": round(load_tot, 1),
        "plan_frac": round(plan_tot / max(1e-9, worker_time), 3),
        "scenario_load_frac": round(load_tot / max(1e-9, worker_time), 3),
        "worker_time_s": round(worker_time, 1),
        # normalised cost — the only fair comparison when GPU counts differ
        "gpus_used": ngpu,
        "gpu_seconds_per_clip": round(gpu_s_clip, 2),
        "gpu_hours_per_1k_clips": round(gpu_s_clip * 1000 / 3600, 2),
        "projected_hours_33k": round(33000 * (el / max(1, len(rows))) / 3600, 1),
        "av_root_gb": (round(_dir_bytes(os.environ["AV_ROOT"]) / 1e9, 2)
                       if os.environ.get("AV_ROOT") else None),
        "slurm": {k: os.environ.get(k) for k in
                  ("SLURM_JOB_ID", "SLURM_JOB_NODELIST", "SLURM_GPUS_ON_NODE")
                  if os.environ.get(k)},
    }
    print(f"\n----- run metrics -----")
    print(f"  wall total       {meta['wall_total_s']}s for {len(rows)} clips / {n_dec} decisions")
    print(f"  throughput       {meta['s_per_clip']}s/clip  {meta['s_per_decision']}s/decision"
          f"  ({meta['clips_per_hour']} clips/h)")
    if walls:
        print(f"  per-clip p50/p95/max  {meta['s_per_clip_p50']} / "
              f"{meta['s_per_clip_p95']} / {meta['s_per_clip_max']} s")
    if mstats:
        print(f"  model            {mstats.get('params_b','?')}B params {mstats.get('dtype','?')}"
              f"  checkpoint {mstats.get('checkpoint_gb','?')} GB"
              f"  sharded over {mstats.get('sharded_over','?')} device(s)")
    print(f"  time split       policy {meta['plan_frac']:.0%} / data load "
          f"{meta['scenario_load_frac']:.0%} / other "
          f"{max(0.0, 1 - meta['plan_frac'] - meta['scenario_load_frac']):.0%}")
    if ngpu:
        print(f"  normalised cost  {meta['gpu_seconds_per_clip']} GPU-s/clip on {ngpu} GPU(s)"
              f"  = {meta['gpu_hours_per_1k_clips']} GPU-h / 1k clips"
              f"  (33k ~= {meta['projected_hours_33k']}h wall)")
    else:
        print(f"  normalised cost  CPU-only policy (no GPU); "
              f"33k ~= {meta['projected_hours_33k']}h wall at {a.workers} worker(s)")
    if meta.get("av_root_gb"):
        print(f"  data footprint   {meta['av_root_gb']} GB (AV_ROOT)")
    if res.get("gpus"):
        g = ", ".join(f"{x['name']} ({x['total_gb']}GB {x['capability']})" for x in res["gpus"])
        print(f"  gpus             {g}")
    if res.get("peak_vram_gb"):
        print(f"  peak VRAM        {res['peak_vram_gb']} GB allocated / "
              f"{res.get('peak_vram_reserved_gb')} GB reserved")
    print(f"  workers          {a.workers}   require_sensors={bool(a.require_sensors)}")

    out = a.out or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                f".results_{policy.name}.parquet")
    try:
        import pyarrow as pa, pyarrow.parquet as pq
        pq.write_table(pa.Table.from_pylist(rows), out)
    except Exception:
        out = out.replace(".parquet", ".json")
        json.dump(rows, open(out, "w"))
    mp = out.replace(".parquet", ".runmeta.json").replace(".json", ".runmeta.json") \
        if not out.endswith(".runmeta.json") else out
    mp = os.path.splitext(out)[0] + ".runmeta.json"
    json.dump(meta, open(mp, "w"), indent=1)
    print(f"\nwrote {out}\nwrote {mp}")
    if a.per_decision:
        import pyarrow as pa, pyarrow.parquet as pq
        pq.write_table(pa.Table.from_pylist(decisions), a.per_decision)
        print(f"wrote {a.per_decision} ({len(decisions)} decisions)")


if __name__ == "__main__":
    main()
