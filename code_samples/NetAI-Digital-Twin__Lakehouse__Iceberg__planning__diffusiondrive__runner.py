#!/usr/bin/env python3
"""Rung-1 driving-difficulty scorer: DiffusionDrive trajectory-mode spread.

For each lidar-covered clip, runs DiffusionDrive (NAVSIM/Transfuser) once and
emits planning_score from the spatial diversity of its 20 candidate trajectories
(mode_spread) — a transfer-robust scene-ambiguity signal (gate: discriminative,
~75% independent of ego_dynamics). Writes per-shard parquet matching the rung-0
contract so it drops into edge_case_scorer._load_planning_scores.

Output (staging): <out>/planning_shard_NN_of_MM.parquet
  clip_id, planning_score in [0,1], mode_spread, mode_entropy, plan_endpoint_m, scored_at

Sharded across GPUs (no flash-attn => any GPU). Resumable (--resume).
"""
import argparse, glob, hashlib, math, os, sys, time
import numpy as np, torch
import pyarrow as pa, pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_one_clip import CKPT, camera_feature, lidar_feature, status_feature

LIDAR_GLOB = "/mnt/netai-e2e/nvidia-physicalai-av-subset/lidar/lidar_top_360fov/*/*.parquet"


def build_agent():
    sd = torch.load(CKPT, map_location="cpu")["state_dict"]
    sd = {k.replace("agent.", ""): v for k, v in sd.items()}
    np.save("/tmp/plan_anchor.npy", sd["_transfuser_model._trajectory_head.plan_anchor"].cpu().numpy())
    from navsim.agents.diffusiondrive.transfuser_config import TransfuserConfig
    from navsim.agents.diffusiondrive.transfuser_agent import TransfuserAgent
    cfg = TransfuserConfig(); cfg.plan_anchor_path = "/tmp/plan_anchor.npy"
    agent = TransfuserAgent(cfg, lr=1e-4, checkpoint_path=CKPT); agent.eval().cuda()
    cap = {}
    for mod in agent.modules():
        if type(mod).__name__ == "DiffMotionPlanningRefinementModule":
            mod.register_forward_hook(lambda m, i, o: cap.update(reg=o[0].detach(), cls=o[1].detach()))
    return agent, cap


def score_clip(agent, cap, root, clip, scale):
    feats = {
        "camera_feature": camera_feature(root, clip)[None].cuda(),
        "lidar_feature": lidar_feature(root, clip)[None].cuda(),
        "status_feature": status_feature(root, clip)[None].cuda(),
    }
    with torch.no_grad():
        out = agent.forward(feats)
    reg = cap["reg"][0].float().cpu().numpy()            # (20,8,3)
    cls = cap["cls"][0].float().cpu()
    p = torch.softmax(cls, dim=-1).numpy()
    ent = float(-(p * np.log(p + 1e-9)).sum() / np.log(len(p)))
    ends = reg[:, -1, :2]
    spread = float(np.sqrt(((ends - ends.mean(0)) ** 2).sum(1).mean()))
    traj = out["trajectory"][0].float().cpu().numpy()
    endpt = float(np.hypot(traj[-1, 0], traj[-1, 1]))
    score = float(min(1.0, max(0.0, (spread - 5.0) / 10.0)))   # [5,15]m -> [0,1]
    return score, spread, ent, endpt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--nfs-root", default="/mnt/netai-e2e/nvidia-physicalai-av-subset")
    ap.add_argument("--shard-id", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--scale", type=float, default=10.0)
    ap.add_argument("--max-clips", type=int, default=0)
    ap.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    args = ap.parse_args()

    clips = sorted(os.path.basename(p).split(".", 1)[0] for p in glob.glob(LIDAR_GLOB))
    clips = [c for c in clips
             if int(hashlib.md5(c.encode()).hexdigest()[:4], 16) % args.n_shards == args.shard_id]
    if args.max_clips:
        clips = clips[:args.max_clips]
    print(f"[dd] shard {args.shard_id}/{args.n_shards}: {len(clips):,} clips", flush=True)

    os.makedirs(args.output_dir, exist_ok=True)
    out = os.path.join(args.output_dir, f"planning_shard_{args.shard_id:02d}_of_{args.n_shards:02d}.parquet")
    rows, done = [], set()
    if args.resume and os.path.exists(out):
        rows = pq.read_table(out).to_pylist(); done = {r["clip_id"] for r in rows}
        print(f"[dd] resume: {len(done):,} already scored", flush=True)

    agent, cap = build_agent()
    print("[dd] model loaded", flush=True)
    t0 = time.time(); n = 0
    for i, c in enumerate(clips):
        if c in done:
            continue
        try:
            score, spread, ent, endpt = score_clip(agent, cap, args.nfs_root, c, args.scale)
        except SystemExit:
            continue                                     # missing a front cam -> rung-0 fallback covers it
        except Exception as e:
            print(f"  [WARN] {c[:8]}: {e}", flush=True); continue
        rows.append({"clip_id": c, "planning_score": score, "mode_spread": spread,
                     "mode_entropy": ent, "plan_endpoint_m": endpt,
                     "scored_at": time.strftime("%Y-%m-%dT%H:%M:%S")})
        n += 1
        if n % 50 == 0:
            pq.write_table(pa.Table.from_pylist(rows), out)
            print(f"[dd] {i+1}/{len(clips)} scored={n} ({n/(time.time()-t0):.2f}/s)", flush=True)
    pq.write_table(pa.Table.from_pylist(rows), out)
    print(f"[dd] DONE shard {args.shard_id}: wrote {len(rows)} rows to {out}", flush=True)


if __name__ == "__main__":
    main()
