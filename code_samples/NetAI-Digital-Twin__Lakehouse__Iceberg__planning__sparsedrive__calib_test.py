"""Bounded test: does REAL PhysicalAI calibration restore a discriminative
SparseDrive planning signal (vs the placeholder nuScenes ring)?

For N clips, run the planner with (a) placeholder projection and (b) a
pinhole-approx projection built from PhysicalAI extrinsics + fisheye intrinsics
(effective focal = fw_poly_1, captures the central ~90deg of the 120deg fisheye),
and compare planning-mode entropy spread + plan-endpoint magnitude. If real
calib gives varying entropy and sane plan magnitudes, grounding works.
"""
import glob, math, os, re, sys
import numpy as np, torch
import pyarrow.parquet as pq
from pyquaternion import Quaternion
from mmcv import Config
from mmcv.runner import load_checkpoint

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_one_clip import (CFG, CKPT, H, W, CAM_ORDER, load_frame,
                           build_ego_status, placeholder_projection)


def _chunk_of(path):
    m = re.search(r"chunk_(\d+)", path)
    return m.group(1) if m else None


def real_projection(clip_id, root):
    """(6,4,4) lidar(ego)2img from PhysicalAI calib; None if any cam missing."""
    mats = []
    for _name, sensor, _yaw in CAM_ORDER:
        cam_glob = glob.glob(f"{root}/camera/{sensor}/*/{clip_id}.{sensor}.mp4")
        if not cam_glob:
            return None
        ch = _chunk_of(cam_glob[0])
        ext_f = glob.glob(f"{root}/calibration/sensor_extrinsics/*chunk_{ch}*.parquet")
        int_f = glob.glob(f"{root}/calibration/camera_intrinsics/*chunk_{ch}*.parquet")
        if not ext_f or not int_f:
            return None
        e = pq.read_table(ext_f[0]).to_pylist()
        i = pq.read_table(int_f[0]).to_pylist()
        er = next((r for r in e if r["clip_id"] == clip_id and r["sensor_name"] == sensor), None)
        ir = next((r for r in i if r["clip_id"] == clip_id and r["camera_name"] == sensor), None)
        if er is None or ir is None:
            return None
        # cam2ego from quat (w,x,y,z) + translation
        R = Quaternion(float(er["qw"]), float(er["qx"]), float(er["qy"]),
                       float(er["qz"])).rotation_matrix
        cam2ego = np.eye(4); cam2ego[:3, :3] = R
        cam2ego[:3, 3] = [float(er["x"]), float(er["y"]), float(er["z"])]
        ego2cam = np.linalg.inv(cam2ego)
        # pinhole-approx K from fisheye: f=fw_poly_1 at native, scaled to (W,H)
        nw, nh = float(ir["width"]), float(ir["height"])
        f = float(ir["fw_poly_1"])
        sx, sy = W / nw, H / nh
        K = np.eye(4)
        K[0, 0] = f * sx; K[1, 1] = f * sy
        K[0, 2] = float(ir["cx"]) * sx; K[1, 2] = float(ir["cy"]) * sy
        mats.append((K @ ego2cam).astype(np.float32))
    return np.stack(mats)


def run(model, clip_id, root, proj):
    imgs = []
    for _n, sensor, _y in CAM_ORDER:
        m = glob.glob(f"{root}/camera/{sensor}/*/{clip_id}.{sensor}.mp4")
        imgs.append(load_frame(m[0]))
    egop = glob.glob(f"{root}/labels/egomotion/*/{clip_id}.egomotion.parquet")
    ego_status, cmd, t = build_ego_status(egop[0])
    dev = "cuda:0"
    img = torch.from_numpy(np.stack(imgs)[None]).float().to(dev)
    data = dict(
        timestamp=torch.tensor([t], dtype=torch.float64).to(dev),
        projection_mat=torch.from_numpy(proj[None]).float().to(dev),
        image_wh=torch.tensor([[[W, H]] * 6], dtype=torch.float32).to(dev),
        ego_status=torch.from_numpy(ego_status[None]).to(dev),
        gt_ego_fut_cmd=torch.from_numpy(cmd[None]).to(dev),
        img_metas=[{"T_global": np.eye(4, dtype=np.float32),
                    "T_global_inv": np.eye(4, dtype=np.float32), "timestamp": t}],
        T_global_inv=torch.from_numpy(np.eye(4, dtype=np.float32)[None]).to(dev),
    )
    with torch.no_grad():
        r = model(img, **data)[0]["img_bbox"]
    ps = r["planning_score"].float().cpu()
    p = torch.softmax(ps[int(cmd.argmax())], dim=-1).numpy()
    ent = float(-(p * np.log(p + 1e-9)).sum() / np.log(len(p)))
    endpt = r["final_planning"].float().cpu().numpy()[-1]
    return ent, float(np.hypot(*endpt))


def main():
    root = "/mnt/netai-e2e/nvidia-physicalai-av-subset"
    clip_ids = [c.strip() for c in open("/work/clips.txt") if c.strip()]
    cfg = Config.fromfile(CFG)
    import importlib
    importlib.import_module(cfg.get("plugin_dir", "projects/mmdet3d_plugin/").rstrip("/").replace("/", "."))
    from mmdet.models import build_detector
    model = build_detector(cfg.model, test_cfg=cfg.get("test_cfg"))
    load_checkpoint(model, CKPT, map_location="cpu"); model.cuda().eval()

    ph_ent, re_ent, re_mag = [], [], []
    for c in clip_ids:
        proj_r = real_projection(c, root)
        if proj_r is None:
            print(f"{c[:8]} calib MISSING"); continue
        e_ph, _ = run(model, c, root, placeholder_projection())
        e_re, m_re = run(model, c, root, proj_r)
        ph_ent.append(e_ph); re_ent.append(e_re); re_mag.append(m_re)
        print(f"{c[:8]}  entropy placeholder={e_ph:.3f} real={e_re:.3f}  real_plan_endpoint={m_re:.1f}m", flush=True)

    import statistics as st
    if re_ent:
        print(f"\nplaceholder entropy: mean={st.mean(ph_ent):.3f} stdev={st.pstdev(ph_ent):.4f}")
        print(f"real-calib  entropy: mean={st.mean(re_ent):.3f} stdev={st.pstdev(re_ent):.4f}")
        print(f"real-calib  plan endpoint(m): min={min(re_mag):.1f} max={max(re_mag):.1f} mean={st.mean(re_mag):.1f}")
        print(">>> CALIB TEST DONE")


if __name__ == "__main__":
    main()
