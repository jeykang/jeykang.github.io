"""Smoke test: run SparseDrive open-loop on ONE PhysicalAI clip frame.

Bypasses the nuScenes .pkl data pipeline — builds the input dict the model's
simple_test() consumes directly (img, timestamp, projection_mat, image_wh,
ego_status, gt_ego_fut_cmd, img_metas[T_global]).

Goal: get a non-empty planning output (ego future trajectory + mode scores) for
one (6-cam + ego_status) input. Calibration is placeholder (nuScenes-style
ring); we accept domain shift — only a relative difficulty signal is needed.
"""
import argparse, glob, math, os
import cv2, numpy as np, torch
from mmcv import Config
from mmcv.runner import load_checkpoint

CFG = "projects/configs/sparsedrive_small_stage2.py"
CKPT = "ckpt/sparsedrive_stage2.pth"
H, W = 256, 704
MEAN = np.array([123.675, 116.28, 103.53], np.float32)
STD = np.array([58.395, 57.12, 57.375], np.float32)
# nuScenes cam order -> PhysicalAI sensors (rear_left substitutes rear-center)
CAM_ORDER = [
    ("CAM_FRONT", "camera_front_wide_120fov", 0.0),
    ("CAM_FRONT_RIGHT", "camera_cross_right_120fov", -55.0),
    ("CAM_FRONT_LEFT", "camera_cross_left_120fov", 55.0),
    ("CAM_BACK", "camera_rear_left_70fov", 180.0),
    ("CAM_BACK_LEFT", "camera_rear_left_70fov", 110.0),
    ("CAM_BACK_RIGHT", "camera_rear_right_70fov", -110.0),
]


def load_frame(mp4, idx_frac=0.5):
    cap = cv2.VideoCapture(mp4)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(n * idx_frac))
    ok, f = cap.read(); cap.release()
    if not ok:
        raise RuntimeError(f"read fail {mp4}")
    f = cv2.resize(f, (W, H)).astype(np.float32)          # BGR
    f = f[:, :, ::-1]                                       # to_rgb=True
    f = (f - MEAN) / STD
    return f.transpose(2, 0, 1)                             # CHW


def placeholder_projection():
    """6 plausible ring lidar2img (intrinsic @ lidar2cam). Geometry approximate."""
    K = np.eye(4, dtype=np.float32); K[0, 0] = K[1, 1] = 512.0; K[0, 2] = W / 2; K[1, 2] = H / 2
    mats = []
    for _name, _sensor, yaw in CAM_ORDER:
        th = math.radians(yaw)
        # ego(x fwd, y left, z up) -> cam(x right, y down, z fwd)
        Rz = np.array([[math.cos(th), math.sin(th), 0],
                       [-math.sin(th), math.cos(th), 0],
                       [0, 0, 1]], np.float32)
        ego2camaxes = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]], np.float32)
        l2c = np.eye(4, dtype=np.float32); l2c[:3, :3] = ego2camaxes @ Rz
        mats.append(K @ l2c)
    return np.stack(mats).astype(np.float32)


def build_ego_status(ego_parquet, idx_frac=0.5):
    import pyarrow.parquet as pq
    d = pq.read_table(ego_parquet,
                      columns=["timestamp", "x", "y", "vx", "vy", "vz",
                               "ax", "ay", "az", "curvature"]).to_pydict()
    o = sorted(range(len(d["timestamp"])), key=lambda k: d["timestamp"][k])
    g = lambda k: [d[k][i] for i in o]
    ts, xs, ys = g("timestamp"), g("x"), g("y")
    vx, vy, vz = g("vx"), g("vy"), g("vz")
    ax, ay, az = g("ax"), g("ay"), g("az")
    curv = g("curvature")
    i = int(len(ts) * idx_frac)
    speed = math.hypot(vx[i], vy[i])
    yaw_rate = speed * curv[i]                               # wz ~ v * curvature
    ego_status = np.array([ax[i], ay[i], az[i],              # accel
                           0.0, 0.0, yaw_rate,               # rotation_rate
                           vx[i], vy[i], vz[i],              # velocity
                           0.0], np.float32)                 # steering (unknown)
    # nav command from future lateral displacement (ego frame, ~3s ahead)
    tgt = ts[i] + 3_000_000
    j = i
    while j < len(ts) - 1 and ts[j] < tgt:
        j += 1
    head = math.atan2(ys[min(i + 5, len(ts) - 1)] - ys[i], xs[min(i + 5, len(ts) - 1)] - xs[i])
    dx, dy = xs[j] - xs[i], ys[j] - ys[i]
    lateral = -math.sin(head) * dx + math.cos(head) * dy
    cmd = [1, 0, 0] if lateral > 3 else ([0, 1, 0] if lateral < -3 else [0, 0, 1])
    return ego_status, np.array(cmd, np.float32), ts[i] / 1e6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip-id", required=True)
    ap.add_argument("--nfs-root", default="/mnt/netai-e2e/nvidia-physicalai-av-subset")
    args = ap.parse_args()

    print(">>> loading model ...", flush=True)
    cfg = Config.fromfile(CFG)
    import importlib
    importlib.import_module(cfg.get("plugin_dir", "projects/mmdet3d_plugin/").rstrip("/").replace("/", "."))
    from mmdet.models import build_detector
    model = build_detector(cfg.model, test_cfg=cfg.get("test_cfg"))
    load_checkpoint(model, CKPT, map_location="cpu")
    model.cuda().eval()

    root = args.nfs_root
    imgs = []
    for _name, sensor, _yaw in CAM_ORDER:
        m = glob.glob(f"{root}/camera/{sensor}/*/{args.clip_id}.{sensor}.mp4")
        if not m:
            print(f"  MISSING {sensor}"); raise SystemExit(2)
        imgs.append(load_frame(m[0]))
    egop = glob.glob(f"{root}/labels/egomotion/*/{args.clip_id}.egomotion.parquet")
    if not egop:
        print("  MISSING egomotion"); raise SystemExit(2)
    ego_status, cmd, t = build_ego_status(egop[0])
    print(f"  ego_status={np.round(ego_status,3)} cmd={cmd}", flush=True)

    dev = "cuda:0"
    img = torch.from_numpy(np.stack(imgs)[None]).float().to(dev)        # (1,6,3,H,W)
    data = dict(
        timestamp=torch.tensor([t], dtype=torch.float64).to(dev),
        projection_mat=torch.from_numpy(placeholder_projection()[None]).to(dev),
        image_wh=torch.tensor([[[W, H]] * 6], dtype=torch.float32).to(dev),
        ego_status=torch.from_numpy(ego_status[None]).to(dev),
        gt_ego_fut_cmd=torch.from_numpy(cmd[None]).to(dev),
        img_metas=[{"T_global": np.eye(4, dtype=np.float32),
                    "T_global_inv": np.eye(4, dtype=np.float32),
                    "timestamp": t}],
        T_global_inv=torch.from_numpy(np.eye(4, dtype=np.float32)[None]).to(dev),
    )
    print(">>> running inference ...", flush=True)
    with torch.no_grad():
        out = model(img, **data)
    r = out[0]["img_bbox"]
    print(">>> SMOKE TEST result keys:", list(r.keys()), flush=True)

    # Difficulty signal: planning-mode uncertainty for the issued command.
    # planning_score is (n_cmd=3, n_mode=6); softmax over modes -> entropy
    # (normalized by log(6)). High entropy = planner can't commit = hard scene.
    ps = r["planning_score"].float().cpu()
    cmd_idx = int(cmd.argmax())
    probs = torch.softmax(ps[cmd_idx], dim=-1).numpy()
    ent = float(-(probs * np.log(probs + 1e-9)).sum() / np.log(len(probs)))
    fp = r["final_planning"].float().cpu().numpy()   # (6,2) chosen ego future
    print(f"  cmd={cmd_idx} mode_probs={np.round(probs,3)}")
    print(f"  >>> planning_uncertainty(norm entropy) = {ent:.3f}")
    print(f"  final_planning endpoint (m) = {np.round(fp[-1],2)}  n_det={len(r['scores_3d'])}")
    print(">>> SPARSEDRIVE ONE-CLIP OK")


if __name__ == "__main__":
    main()
