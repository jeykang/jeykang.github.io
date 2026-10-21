"""vavam_client.py — VaVAM as a HUGSIM client, through AlpaSim's own VAM wrapper.

Runs inside alpasim-harness-base (its venv has `vam` and `alpasim_driver`), so the model,
checkpoints, preprocessing (resize/center-crop to 900x1600 + NeuroNCAPTransform), command
encoding and fp16 autocast are exactly what our AlpaSim VaVAM rollouts used. Only the
transport differs: HUGSIM's named pipes instead of AlpaSim's gRPC.

  python vavam_client.py <hugsim output dir> [--camera CAM_FRONT] [--context 1] [--seed N]

Per step: read pickle((obs, info)) from <out>/obs_pipe, take obs['rgb'][camera] (HWC uint8),
map info['command'] (HUGSIM: 0 right, 1 left, 2 forward) to DriveCommand, predict, and write
the plan to <out>/plan_pipe in HUGSIM's frame (x right, y forward; 0.5 s spacing = VaVAM's
2 Hz output). 'Done' ends the episode. A per-step log goes to <out>/vavam_steps.jsonl.
"""
import argparse
import json
import os
import pickle
import time
from collections import deque

import numpy as np
import torch
from alpasim_driver.models.base import CameraFrame, DriveCommand, PredictionInput
from alpasim_driver.models.vam_model import VAMModel

HUGSIM_TO_DRIVE = {0: DriveCommand.RIGHT, 1: DriveCommand.LEFT, 2: DriveCommand.STRAIGHT}

ap = argparse.ArgumentParser()
ap.add_argument("output")
ap.add_argument("--ckpt", default="/mnt/drivers/vavam/VAM_width_1024_pretrained_139k.pt")
ap.add_argument("--tokenizer", default="/mnt/drivers/vavam/VQ_ds16_16384_llamagen_encoder.jit")
ap.add_argument("--camera", default="CAM_FRONT")
# 1 = what our AlpaSim driver config used (inference.context_length); VaVAM was trained on
# 2 Hz frames, so with a longer context every other 4 Hz HUGSIM frame is kept
ap.add_argument("--context", type=int, default=1)
# AlpaSim does not seed VAM (its flow-matching sampler draws fresh noise per call); a seed
# here makes a HUGSIM episode reproducible, and several seeds give the policy's own spread
ap.add_argument("--seed", type=int, default=None)
# AlpaSim forces the recorded trajectory for its first 3.0 s (force_gt_duration_us); with
# --warmup 3.0 this client sends the recorded path (clients/warmup.py) until then, so the policy
# takes over in the same state as in AlpaSim. 0 = the original protocol (control from t = 0).
ap.add_argument("--warmup", type=float, default=0.0)
args = ap.parse_args()
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from warmup import RecordedPath  # noqa: E402
recorded = RecordedPath(args.output) if args.warmup > 0 else None

obs_pipe, plan_pipe = (os.path.join(args.output, p) for p in ("obs_pipe", "plan_pipe"))
for p in (obs_pipe, plan_pipe):
    if not os.path.exists(p):
        os.mkfifo(p)

t0 = time.time()
model = VAMModel(args.ckpt, args.tokenizer, torch.device("cuda"), [args.camera],
                 context_length=args.context)
print(f"vavam client ready ({time.time() - t0:.0f} s to load, {torch.cuda.get_device_name(0)})",
      flush=True)

frames = deque(maxlen=2 * args.context - 1)  # 4 Hz history; [::2] gives 2 Hz frames
log_path = os.path.join(args.output, "vavam_steps.jsonl")
log = open(log_path, "w")
# the image's venv interpreter lives under /root, so this runs as root: hand the file back
st = os.stat(args.output)
os.chown(log_path, st.st_uid, st.st_gid)
n = 0
while True:
    with open(obs_pipe, "rb") as f:
        data = pickle.loads(f.read())
    if isinstance(data, str) and data == "Done":
        break
    obs, info = data
    ts = int(round(float(info["timestamp"]) * 1e6))
    frames.append(CameraFrame(ts, obs["rgb"][args.camera]))
    ctx = list(frames)[::-1][::2][::-1]
    ctx = [ctx[0]] * (args.context - len(ctx)) + ctx  # pad the first steps with the oldest frame
    command = HUGSIM_TO_DRIVE.get(int(info["command"]), DriveCommand.STRAIGHT)
    inp = PredictionInput(camera_images={args.camera: ctx}, command=command,
                          speed=float(info["ego_velo"]), acceleration=float(info["accelerate"] or 0.0),
                          ego_pose_history=[], inference_seed=n, previous_plan=None, route=None)
    t1 = time.time()
    warm = recorded is not None and float(info["timestamp"]) < args.warmup - 1e-6
    if warm:
        plan = recorded.plan(info, 6)                       # the recorded drive, as AlpaSim forces it
    else:
        if args.seed is not None:
            torch.manual_seed(args.seed * 100003 + n)
        pred = model.predict(inp)
        xy = pred.candidate_positions[pred.selected_index][:, :2]  # rig frame: x forward, y left
        plan = np.stack([-xy[:, 1], xy[:, 0]], axis=1).astype(np.float64)  # HUGSIM: x right, y forward
    with open(plan_pipe, "wb") as f:
        f.write(pickle.dumps(plan))
    log.write(json.dumps({"step": n, "t": info["timestamp"], "warmup": bool(warm), "command": int(info["command"]),
                          "v": float(info["ego_velo"]), "ms": round(1e3 * (time.time() - t1)),
                          "plan": np.round(plan, 2).tolist()}) + "\n")
    log.flush()
    n += 1
print(f"vavam client done after {n} steps", flush=True)
