#!/usr/bin/env python3
"""compare_simulators.py — the same policy on the same clip in three closed-loop set-ups.

    .skip_venv/bin/python rollouts.py && .skip_venv/bin/python compare_simulators.py [--policy vavam]

Reads ../user_data/rollouts.parquet (rollouts.py; eval.rollout in Iceberg) and, for every clip
that has a HUGS twin, puts side by side:
  NVIDIA    AlpaSim over NVIDIA's NuRec scene, recorded-waypoint routes (the reference; the
            original run plus the repeat and PNG-frame runs where they exist)
  NuRec     AlpaSim over our NuRec twin (NVIDIA's map layers borrowed), recorded routes
  HUGS      HUGSIM over our LiDAR-seeded HUGS twin, recorded traffic in its own appearance
            (scenario rec_02), 100-degree pinhole front view, one row per seed
Each cell: failures / rollouts and the distance driven when the episode ended (median, range).
"failed" is AlpaSim's offroad_or_collision and HUGSIM's collision (rollouts.py); a barrier is
off-road in one and a collision in the other, so compare whether and where a policy fails, not
the class name. Agreement = the majority verdict matches the reference's.
"""
import argparse
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))


def cell(g: pd.DataFrame) -> dict:
    if g.empty:
        return {"n": 0, "fail": None, "txt": "—"}
    d = g["dist_m"].astype(float)
    k = int(g["failed"].sum())
    rng = f"{d.min():.1f}–{d.max():.1f}" if len(d) > 1 and d.max() - d.min() >= 0.05 else f"{d.median():.1f}"
    return {"n": len(g), "fail": k / len(g) > 0.5, "k": k, "med": float(d.median()),
            "txt": f"{k}/{len(g)} failed, {rng} m"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rollouts", default=os.path.join(HERE, "..", "user_data", "rollouts.parquet"))
    ap.add_argument("--policy", default="vavam")
    ap.add_argument("--out", default=os.path.join(HERE, ".compare_simulators.csv"))
    ap.add_argument("--protocol", choices=["original", "warmup"], default="warmup",
                    help="warmup (2026-10-07): HUGSIM episodes with AlpaSim's 3.0 s recorded warm-up, and AlpaSim "
                         "reference-policy runs from the fixed policy bridge; original: the first runs")
    a = ap.parse_args()
    r = pd.read_parquet(a.rollouts)
    r = r[r["policy"] == a.policy]
    wu = 3.0 if a.protocol == "warmup" else 0.0
    hugs = r[(r.simulator == "hugsim") & r.twin.str.endswith("_lidar", na=False) & (r.scenario_variant == "rec_02")
             & ((r.camera == "front-100-pinhole") if a.policy == "vavam" else True) & (r.warmup_s == wu)]
    if a.policy != "vavam":
        keep = "fixed" if a.protocol == "warmup" else "zero-speed bug"
        r = r[(r.simulator != "alpasim") | (r.bridge == keep)]
    rows = []
    for clip in sorted(hugs.clip_id.unique()):
        al = r[(r.simulator == "alpasim") & (r.clip_id == clip)]
        routed = (al.route_generator == "RECORDED") if a.policy != "constant_velocity" else True   # cv ignores routes
        nv = al[(al.serving_mode == "closedloop-nurec") & routed]
        ours = al[(al.serving_mode == "closedloop-nurec-ours") & (al.map_layers == "borrowed") & routed
                  & al.twin.isin([f"{clip[:8]}_a10_prod", "ac73935a_a10_prod_v3"])]
        hg = hugs[hugs.clip_id == clip]
        c_nv, c_ours, c_hg = cell(nv), cell(ours), cell(hg)
        rows.append({"clip": clip[:8], "nvidia": c_nv["txt"], "ours_nurec": c_ours["txt"], "ours_hugs": c_hg["txt"],
                     "nurec_agrees": None if c_nv["fail"] is None or c_ours["fail"] is None else c_nv["fail"] == c_ours["fail"],
                     "hugs_agrees": None if c_nv["fail"] is None or c_hg["fail"] is None else c_nv["fail"] == c_hg["fail"],
                     "nvidia_med_m": c_nv.get("med"), "nurec_med_m": c_ours.get("med"), "hugs_med_m": c_hg.get("med")})
    df = pd.DataFrame(rows)
    df.to_csv(a.out, index=False)
    print(f"| clip | AlpaSim, NVIDIA's scene | AlpaSim, our NuRec twin | HUGSIM, our HUGS twin | NuRec agrees | HUGS agrees |")
    print("|---|---|---|---|---|---|")
    for x in rows:
        yn = lambda v: "—" if v is None else ("yes" if v else "no")
        print(f"| {x['clip']} | {x['nvidia']} | {x['ours_nurec']} | {x['ours_hugs']} | {yn(x['nurec_agrees'])} | {yn(x['hugs_agrees'])} |")
    for col in ("nurec_agrees", "hugs_agrees"):
        v = df[col].dropna()
        print(f"{col}: {int(v.sum())}/{len(v)}")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
