"""cv_client.py — constant-velocity policy for HUGSIM, speaking its named-pipe protocol.

HUGSIM's closed_loop.py writes pickle((obs, info)) to <output>/obs_pipe and reads a plan
from <output>/plan_pipe: an (N, 2) array of waypoints in its ego frame (x right, y forward),
0.5 s apart; 'Done' ends the episode. This client keeps the current speed straight ahead —
the same reference policy as `constant_velocity` in our AlpaSim runs, so the two simulators
can be compared on it, and a template for bridging our other evaluator policies.
"""
import argparse
import os
import pickle
import sys

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("output")
# AlpaSim forces the recorded trajectory for its first 3.0 s; --warmup 3.0 sends the recorded path
# (clients/warmup.py) until then. --hold-speed keeps the recorded speed at handover for the rest of
# the episode, as AlpaSim's constant_velocity does once its bridge reads the speed correctly;
# without it the speed is re-read from the simulator every step and drifts with the controller.
ap.add_argument("--warmup", type=float, default=0.0)
ap.add_argument("--hold-speed", action="store_true")
args = ap.parse_args()
out = args.output
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from warmup import RecordedPath  # noqa: E402
recorded = RecordedPath(out) if (args.warmup > 0 or args.hold_speed) else None
v_hold = recorded.speed(args.warmup) if recorded is not None else None
obs_pipe, plan_pipe = os.path.join(out, "obs_pipe"), os.path.join(out, "plan_pipe")
for p in (obs_pipe, plan_pipe):
    if not os.path.exists(p):
        os.mkfifo(p)
print(f"cv client ready (warmup {args.warmup} s, hold speed {v_hold})", flush=True)
n = 0
while True:
    with open(obs_pipe, "rb") as f:
        data = pickle.loads(f.read())
    if isinstance(data, str) and data == "Done":
        break
    obs, info = data
    if recorded is not None and float(info["timestamp"]) < args.warmup - 1e-6:
        plan = recorded.plan(info, 8)
    else:
        v = v_hold if args.hold_speed else float(info.get("ego_velo", 1.0))
        plan = np.array([[0.0, v * 0.5 * (k + 1)] for k in range(8)], dtype=np.float64)
    with open(plan_pipe, "wb") as f:
        f.write(pickle.dumps(plan))
    n += 1
print(f"cv client done after {n} steps", flush=True)
