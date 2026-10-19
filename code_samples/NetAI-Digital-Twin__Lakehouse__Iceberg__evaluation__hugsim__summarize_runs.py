"""summarize_runs.py — one row per HUGSIM closed-loop episode.

  docker run --rm -v $H:$H -w $H/repo hugsim-dev:cu118 pixi run python $H/summarize_runs.py runs/pai_vavam_s*_ltf/*
  ... summarize_runs.py --json out.json 'runs/pai*_ltf/*'   # also write the rows as JSON (evaluation/rollouts.py)

For each episode dir (closed_loop.py output): steps, simulated seconds, distance driven, the
largest gap to the recorded front-camera path (HUGSIM ends an episode at > 10 m), how it ended,
the HD-Score terms from eval.json, and when the ego itself first left the drivable area.

HUGSIM's DAC term scores the *planned* trajectory, and leaving the road never ends an episode;
the scorer's per-frame check on the ego's own footprint is commented out
(`sim/utils/score_calculator.py`, `_single_frame_drivable_area_compliance`). It is reproduced
here post hoc with the same rule (2 x 2 grid over the ego box, off-road when fewer than 30 % of
the cells hold a road-class ground point of the episode's `ground.ply`), so an ego that leaves
the road counts as a failure, as AlpaSim's map-based off-road check does.
"""
import glob
import json
import os
import pickle
import sys

import numpy as np

H = os.path.dirname(os.path.abspath(__file__))
args = sys.argv[1:]
json_out = None
if "--json" in args:
    i = args.index("--json")
    json_out = args[i + 1]
    del args[i:i + 2]
dirs = [d for a in args for d in sorted(glob.glob(a if os.path.isabs(a) else os.path.join(H, a)))]
rows = []
print(f"{'episode':52s} {'steps':>5s} {'sim s':>5s} {'dist m':>6s} {'max gap':>7s} {'end':10s}"
      f" {'NC':>5s} {'DAC':>5s} {'TTC':>5s} {'C':>4s} {'Rc':>5s} {'HD':>6s}")
for d in dirs:
    if not os.path.exists(os.path.join(d, "eval.json")):
        print(f"{os.path.relpath(d, H):52s} (no eval.json)")
        continue
    ev = json.load(open(os.path.join(d, "eval.json")))
    infos = pickle.load(open(os.path.join(d, "infos.pkl"), "rb"))
    # the scene's recorded front-camera path, as the simulator loaded it (model_base/<scene>)
    scene = os.path.basename(d).rsplit("_", 2)[0]
    gp = None
    for root in ("data/scenes/pai", "data/scenes/nuscenes"):
        p = os.path.join(H, root, scene, "ground_param.pkl")
        if os.path.exists(p):
            gp = pickle.load(open(p, "rb"))[0]
            break
    # infos.pkl stops one step short (the terminating step's info is not appended); data.pkl
    # has every step, with ego_box in HUGSIM's IMU frame (x = world z, y = -world x)
    frames = pickle.load(open(os.path.join(d, "data.pkl"), "rb"))[0]["frames"]
    xz = np.array([[infos[0]["ego_pos"][0], infos[0]["ego_pos"][2]]]
                  + [[-f["ego_box"][1], f["ego_box"][0]] for f in frames])
    dist = float(np.sum(np.linalg.norm(np.diff(xz, axis=0), axis=1)))
    gap = float("nan")
    if gp is not None:
        ref = np.asarray(gp)[:, [0, 2], 3]
        gap = float(max(np.min(np.linalg.norm(ref - p, axis=1)) for p in xz))
    # the ego's own drivable-area check, per frame (the scorer's commented-out single-frame rule)
    off_i = None
    gp_path = os.path.join(d, "ground.ply")
    if os.path.exists(gp_path):
        raw = open(gp_path, "rb").read()
        head_end = raw.index(b"end_header\n") + len(b"end_header\n")
        nv = int([l for l in raw[:head_end].decode().splitlines() if l.startswith("element vertex")][0].split()[-1])
        g = np.frombuffer(raw[head_end:head_end + nv * 24], dtype="<f8").reshape(nv, 3)
        ground_xy = np.stack([g[:, 2], -g[:, 0]], axis=1)          # camera -> IMU coordinates, as the scorer does
        for i, f in enumerate(frames):
            x, y, _, w, l, _, yaw = f["ego_box"]
            c, s_ = np.cos(yaw), np.sin(yaw)
            near = ground_xy[np.abs(ground_xy[:, 0] - x) + np.abs(ground_xy[:, 1] - y) < l + w]
            loc = (near - np.array([x, y])) @ np.array([[c, -s_], [s_, c]])   # inv(R) @ v for each row
            cells = 0
            for xi in range(2):
                for yi in range(2):
                    mx0, mx1 = -l / 2 + xi * l / 2, -l / 2 + (xi + 1) * l / 2
                    my0, my1 = -w / 2 + yi * w / 2, -w / 2 + (yi + 1) * w / 2
                    if np.any((mx0 < loc[:, 0]) & (loc[:, 0] < mx1) & (my0 < loc[:, 1]) & (loc[:, 1] < my1)):
                        cells += 1
            if cells / 4 < 0.3:
                off_i = i
                break
    seg = np.linalg.norm(np.diff(xz, axis=0), axis=1)
    off_t = float(frames[off_i]["time_stamp"]) if off_i is not None else None
    off_dist = float(np.sum(seg[:off_i + 1])) if off_i is not None else None
    last = frames[-1]
    if last["collision"]:
        end = "collision"
    elif last["rc"] >= 1:
        end = "complete"
    elif gap > 10:
        end = "off path"
    else:
        end = "step cap"
    print(f"{os.path.relpath(d, H):52s} {len(frames):5d} {last['time_stamp']:5.2f} {dist:6.1f} {gap:7.1f} {end:10s}"
          f" {ev['nc']:5.2f} {ev['dac']:5.2f} {ev['ttc']:5.2f} {ev['c']:4.1f} {ev['rc']:5.2f} {ev['hdscore']:6.3f}")
    if off_t is not None:
        print(f"{'':52s} ego off the drivable area from {off_t:.2f} s, after {off_dist:.1f} m")
    rows.append({"episode_dir": os.path.relpath(d, H), "steps": len(frames), "sim_s": float(last["time_stamp"]),
                 "dist_m": dist, "max_gap_m": gap, "end": end, "offroad_t": off_t, "offroad_dist_m": off_dist,
                 **{k: float(ev[k]) for k in ("nc", "dac", "ttc", "c", "rc", "hdscore") if k in ev}})
if json_out:
    json.dump(rows, open(json_out, "w"), indent=1)
    print(f"wrote {len(rows)} rows to {json_out}")
