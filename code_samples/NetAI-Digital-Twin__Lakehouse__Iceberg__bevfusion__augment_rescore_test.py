"""Cheap-augmentation re-score probe: does a camera condition-transform actually
register in the difficulty scorer's perceptual axis (BEVFusion detection confidence)?

For a few clips: run BEVFusion on (a) original camera frames and (b) the same frames
after a classical degradation (night/rain/fog), with LIDAR UNCHANGED — exactly what
a Cosmos-Transfer (camera appearance) augmentation would do. If confidence/detections
drop, perceptual augmentation registers as harder; if clean lidar masks it, that's a
key finding for the augmentation design. Runs in the netai/bevfusion-runner container.
"""
import sys, glob, math
sys.path.insert(0, "/workspace")
import numpy as np, cv2, DracoPy, torch, pyarrow.parquet as pq
import projects.BEVFusion.bevfusion  # noqa: register
from mmdet3d.apis import init_model
from bevfusion_infer import build_data, CAM_ORDER
import transforms

CFG = "/workspace/mmdetection3d/projects/BEVFusion/configs/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py"
CKPT = "/workspace/checkpoints/bevfusion_lidar-cam_spconv2.pth"
ROOT = "/mnt/netai-e2e/nvidia-physicalai-av-subset"
DEV = "cuda:0"
COND = sys.argv[1] if len(sys.argv) > 1 else "night"
K = int(sys.argv[2]) if len(sys.argv) > 2 else 6
CAMERA_ONLY = len(sys.argv) > 3 and sys.argv[3] == "1"   # suppress lidar -> camera-reliant proxy


def _scatter_cloud():
    """Object-free scattered ground points: model gets no useful lidar -> must use
    the camera (rough camera-only proxy for the camera-only downstream target)."""
    rs = np.random.RandomState(0); p = np.zeros((4000, 5), np.float32)
    p[:, 0] = rs.uniform(-50, 50, 4000); p[:, 1] = rs.uniform(-50, 50, 4000)
    p[:, 2] = rs.uniform(-3.0, -1.5, 4000)   # near ground, no objects
    return p


def cam_frac(mp4, frac=0.5):
    cap = cv2.VideoCapture(mp4); n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(n * frac)); ok, f = cap.read(); cap.release()
    if not ok:
        return np.zeros((256, 704, 3), np.uint8)
    return cv2.cvtColor(cv2.resize(f, (704, 256)), cv2.COLOR_BGR2RGB)


def lidar_frac(path, frac=0.5):
    blobs = pq.read_table(path, columns=["draco_encoded_pointcloud"]).column(0).to_pylist()
    pts = np.asarray(DracoPy.decode(blobs[int(len(blobs) * frac)]).points, np.float32)
    if pts.shape[1] < 5:
        pad = np.zeros((len(pts), 5 - pts.shape[1]), np.float32)
        pts = np.concatenate([pts[:, :3], pad], 1); pts[:, 3] = 1.0
    return pts


def boxes(model, pts, imgs):
    with torch.no_grad():
        out = model.test_step(build_data(pts, imgs, DEV))
    pi = out[0].pred_instances_3d
    return pi.scores_3d.float().detach().cpu().numpy()


def conf_n(s, thr=0.3):
    s = np.asarray(s)
    return (float(s.max()) if len(s) else 0.0), int((s > thr).sum())


def cam_for(cid, sensor):
    m = glob.glob(f"{ROOT}/camera/{sensor}/*/{cid}.{sensor}.mp4")
    return cam_frac(m[0]) if m else np.zeros((256, 704, 3), np.uint8)


def main():
    print(f"loading BEVFusion; condition={COND} camera_only={CAMERA_ONLY}", flush=True)
    model = init_model(CFG, CKPT, device=DEV); model.eval()
    lid = sorted(glob.glob(f"{ROOT}/lidar/lidar_top_360fov/*/*.lidar_top_360fov.parquet"))
    clips = []
    for p in lid:
        cid = p.split("/")[-1].split(".")[0]
        if glob.glob(f"{ROOT}/camera/camera_front_wide_120fov/*/{cid}.camera_front_wide_120fov.mp4"):
            clips.append((cid, p))
        if len(clips) >= K:
            break
    print(f"{len(clips)} clips\n", flush=True)
    dc, dn = [], []
    for cid, lp in clips:
        try:
            pts = _scatter_cloud() if CAMERA_ONLY else lidar_frac(lp)
            imgs = [cam_for(cid, s) for s in CAM_ORDER]
            co, no = conf_n(boxes(model, pts, imgs))
            cd, nd = conf_n(boxes(model, pts, transforms.apply(imgs, COND)))
            dc.append(cd - co); dn.append(nd - no)
            print(f"[{cid[:8]}] orig maxconf={co:.3f} ndet={no:2d} | {COND} maxconf={cd:.3f} ndet={nd:2d}"
                  f" | Δconf={cd-co:+.3f} Δndet={nd-no:+d}", flush=True)
        except Exception as e:
            print(f"[{cid[:8]}] WARN {str(e)[:70]}", flush=True)
    if dc:
        print(f"\nmean Δconf = {np.mean(dc):+.3f}   mean Δndet = {np.mean(dn):+.2f}", flush=True)
        print(f"(negative = camera degradation DID lower BEVFusion confidence -> perceptual aug registers)", flush=True)
    print(">>> AUG RESCORE DONE", flush=True)


if __name__ == "__main__":
    main()
