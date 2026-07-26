"""Smoke test: run DiffusionDrive (NAVSIM/Transfuser) on ONE PhysicalAI clip.

Builds the 3 features the model consumes directly — stitched forward camera
(1024x256), lidar BEV histogram (1x256x256 over [-32,32]m), 8-dim status
(cmd4 + vel2 + accel2) — bypassing the NAVSIM Scene/data pipeline. Lidar BEV is
the transfer-robust input (geometry, not appearance) that motivated this planner.

Goal: non-empty multimodal plan (20 trajectory modes + mode scores). Calibration
isn't needed (Transfuser uses raw stitched image + BEV raster, no projection
matrices) — a key reason this should transfer better than SparseDrive.
"""
import argparse, glob, math, os
import cv2, numpy as np, torch

CKPT = "ckpt/diffusiondrive_navsim_88p1.pth"
# nuScenes/NAVSIM forward triplet -> PhysicalAI 120-fov forward cameras
CAM_L, CAM_F, CAM_R = ("camera_cross_left_120fov", "camera_front_wide_120fov",
                       "camera_cross_right_120fov")
LMINX = LMINY = -32.0
LMAXX = LMAXY = 32.0
SPLIT_H = 0.2
MAXH = 100.0
HISTMAX = 5.0
BEV = 256


def load_native(mp4, idx_frac=0.5):
    cap = cv2.VideoCapture(mp4)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(n * idx_frac))
    ok, f = cap.read(); cap.release()
    if not ok:
        raise RuntimeError(f"read fail {mp4}")
    return cv2.cvtColor(f, cv2.COLOR_BGR2RGB)            # native HxWx3 RGB


def camera_feature(root, clip):
    def g(sensor):
        m = glob.glob(f"{root}/camera/{sensor}/*/{clip}.{sensor}.mp4")
        if not m:
            raise SystemExit(f"missing {sensor}")
        return load_native(m[0])
    l, f, r = g(CAM_L), g(CAM_F), g(CAM_R)
    # NAVSIM crops for 1920x1080: L/R [28:-28, 416:-416], F [28:-28]
    l = l[28:-28, 416:-416]; f = f[28:-28]; r = r[28:-28, 416:-416]
    stitched = np.concatenate([l, f, r], axis=1)         # (1024, 4096, 3)
    stitched = cv2.resize(stitched, (1024, 256))
    return torch.from_numpy(stitched.transpose(2, 0, 1)).float() / 255.0  # (3,256,1024)


def lidar_feature(root, clip, idx_frac=0.5):
    import DracoPy, pyarrow.parquet as pq
    m = glob.glob(f"{root}/lidar/lidar_top_360fov/*/{clip}.lidar_top_360fov.parquet")
    if not m:
        raise SystemExit("missing lidar")
    blobs = pq.read_table(m[0], columns=["draco_encoded_pointcloud"]).column(0).to_pylist()
    pts = np.asarray(DracoPy.decode(blobs[int(len(blobs) * idx_frac)]).points, np.float32)
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    m_above = (z > SPLIT_H) & (z < MAXH)
    hist, _, _ = np.histogram2d(
        x[m_above], y[m_above], bins=BEV,
        range=[[LMINX, LMAXX], [LMINY, LMAXY]])
    hist = np.clip(hist, 0, HISTMAX) / HISTMAX
    return torch.from_numpy(hist[None].astype(np.float32))           # (1,256,256)


def status_feature(root, clip, idx_frac=0.5):
    import pyarrow.parquet as pq
    m = glob.glob(f"{root}/labels/egomotion/*/{clip}.egomotion.parquet")
    d = pq.read_table(m[0], columns=["timestamp", "x", "y", "vx", "vy", "ax", "ay"]).to_pydict()
    o = sorted(range(len(d["timestamp"])), key=lambda k: d["timestamp"][k])
    i = int(len(o) * idx_frac); k = o[i]
    cmd = [0.0, 1.0, 0.0, 0.0]                            # default: straight (NAVSIM 4-dim)
    return torch.tensor(cmd + [d["vx"][k], d["vy"][k], d["ax"][k], d["ay"][k]],
                        dtype=torch.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip-id", required=True)
    ap.add_argument("--nfs-root", default="/mnt/netai-e2e/nvidia-physicalai-av-subset")
    args = ap.parse_args()

    # Extract the 20-mode trajectory anchor from the checkpoint (the configured
    # path is the author's machine; load_state_dict overwrites it anyway).
    sd = torch.load(CKPT, map_location="cpu")["state_dict"]
    sd = {k.replace("agent.", ""): v for k, v in sd.items()}
    anchor = sd["_transfuser_model._trajectory_head.plan_anchor"].cpu().numpy()
    np.save("/tmp/plan_anchor.npy", anchor)
    print(f">>> anchor {anchor.shape} -> /tmp/plan_anchor.npy", flush=True)

    from navsim.agents.diffusiondrive.transfuser_config import TransfuserConfig
    from navsim.agents.diffusiondrive.transfuser_agent import TransfuserAgent
    cfg = TransfuserConfig()
    cfg.plan_anchor_path = "/tmp/plan_anchor.npy"
    print(">>> loading agent ...", flush=True)
    agent = TransfuserAgent(cfg, lr=1e-4, checkpoint_path=CKPT)
    agent.eval().cuda()

    root = args.nfs_root
    feats = {
        "camera_feature": camera_feature(root, args.clip_id)[None].cuda(),
        "lidar_feature": lidar_feature(root, args.clip_id)[None].cuda(),
        "status_feature": status_feature(root, args.clip_id)[None].cuda(),
    }
    # Hook the multimodal refinement head to capture the 20-mode distribution
    # (poses_reg [b,20,8,3], poses_cls [b,20]) — not exposed in the output dict.
    cap = {}
    for mod in agent.modules():
        if type(mod).__name__ == "DiffMotionPlanningRefinementModule":
            mod.register_forward_hook(
                lambda m, i, o: cap.update(reg=o[0].detach(), cls=o[1].detach()))

    print(">>> running inference ...", flush=True)
    with torch.no_grad():
        out = agent.forward(feats)

    traj = out["trajectory"][0].float().cpu().numpy()    # (8,3) final plan
    endpt = float(np.hypot(traj[-1, 0], traj[-1, 1]))
    line = f"  final_plan endpoint={endpt:.1f}m"
    if "cls" in cap:
        cls = cap["cls"][0].float().cpu()                # (20,)
        reg = cap["reg"][0].float().cpu().numpy()        # (20,8,3)
        p = torch.softmax(cls, dim=-1).numpy()
        ent = float(-(p * np.log(p + 1e-9)).sum() / np.log(len(p)))
        ends = reg[:, -1, :2]                             # 20 mode endpoints
        spread = float(np.sqrt(((ends - ends.mean(0)) ** 2).sum(1).mean()))
        line += f"  mode_entropy={ent:.3f}  mode_spread={spread:.2f}m  n_modes={len(p)}"
    print(line, flush=True)
    print(">>> DIFFUSIONDRIVE ONE-CLIP OK")


if __name__ == "__main__":
    main()
