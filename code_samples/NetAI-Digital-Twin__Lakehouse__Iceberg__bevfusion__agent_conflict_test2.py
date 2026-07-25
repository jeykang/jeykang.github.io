"""Strengthened agent-conflict signal + re-validation.

Adds to the v1 forward-zone metric:
  - class weighting: VRU (ped/bike/motorcycle) >> vehicles; cones/barriers
    up-weighted (work-zone indicator),
  - TTC: time-to-collision from box velocity + ego speed (moving threat),
  - multi-frame: aggregate over frames 0.3/0.5/0.7 to cut detection noise.

Reports OOD AUC for the simple (v1) and rich (v2) metrics on the SAME boxes
(to isolate the metric gain from multi-frame), plus the negative control.
clips.txt: "clip_id,is_ood".
"""
import glob, math, os, sys, statistics as st
import cv2, numpy as np, torch
import DracoPy, pyarrow.parquet as pq

sys.path.insert(0, "/workspace")
import projects.BEVFusion.bevfusion  # noqa
from mmdet3d.apis import init_model
from bevfusion_infer import build_data, CAM_ORDER

CFG = "/workspace/mmdetection3d/projects/BEVFusion/configs/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py"
CKPT = "/workspace/checkpoints/bevfusion_lidar-cam_spconv2.pth"
ROOT = "/mnt/netai-e2e/nvidia-physicalai-av-subset"
DEV = "cuda:0"
FRACS = [0.3, 0.5, 0.7]
# nuScenes idx: car truck constr bus trailer barrier motorcycle bicycle pedestrian cone
WCLASS = [1.0, 1.0, 1.2, 1.0, 1.0, 1.5, 3.0, 3.0, 3.0, 1.5]


def cam_frac(mp4, frac):
    cap = cv2.VideoCapture(mp4); n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(n*frac)); ok, f = cap.read(); cap.release()
    if not ok: return np.zeros((256, 704, 3), np.uint8)
    return cv2.cvtColor(cv2.resize(f, (704, 256)), cv2.COLOR_BGR2RGB)


def lidar_frac(path, frac):
    blobs = pq.read_table(path, columns=["draco_encoded_pointcloud"]).column(0).to_pylist()
    pts = np.asarray(DracoPy.decode(blobs[int(len(blobs)*frac)]).points, np.float32)
    if pts.shape[1] < 5:
        pad = np.zeros((len(pts), 5-pts.shape[1]), np.float32)
        pts = np.concatenate([pts[:, :3], pad], 1); pts[:, 3] = 1.0
    return pts


def ego_speed(clip, frac):
    m = glob.glob(f"{ROOT}/labels/egomotion/*/{clip}.egomotion.parquet")
    if not m: return 5.0
    d = pq.read_table(m[0], columns=["timestamp", "vx", "vy"]).to_pydict()
    o = sorted(range(len(d["timestamp"])), key=lambda k: d["timestamp"][k]); k = o[int(len(o)*frac)]
    return math.hypot(d["vx"][k], d["vy"][k])


def boxes(model, pts, imgs):
    with torch.no_grad():
        out = model.test_step(build_data(pts, imgs, DEV))
    pi = out[0].pred_instances_3d
    return (pi.bboxes_3d.tensor.float().detach().cpu().numpy(),
            pi.scores_3d.float().detach().cpu().numpy(), pi.labels_3d.detach().cpu().numpy())


def loads(b, s, l, vego):
    if len(b) == 0: return 0.0, 0.0
    x, y = b[:, 0], b[:, 1]; vx = b[:, 7] if b.shape[1] > 7 else np.zeros(len(b))
    d = np.hypot(x, y); fwd = (x > 0) & (x < 40) & (np.abs(y) < 8)
    simple = float((s[fwd]/(1+d[fwd])).sum()) if fwd.any() else 0.0
    rich = 0.0
    for i in np.where(fwd)[0]:
        wc = WCLASS[int(l[i])] if 0 <= int(l[i]) < len(WCLASS) else 1.0
        ttc = x[i]/max(0.5, vego - vx[i])
        rich += float(s[i]*wc/(1.0+ttc))
    return simple, rich


def auc(rows, key):
    o = [r[key] for r in rows if r["ood"]]; n = [r[key] for r in rows if not r["ood"]]
    if not o or not n: return float("nan")
    al = sorted(rows, key=lambda r: r[key])
    rsum = sum(i+1 for i, r in enumerate(al) if r["ood"])
    return (rsum - len(o)*(len(o)+1)/2) / (len(o)*len(n))


def main():
    clips = [ln.strip().split(",") for ln in open("/work/clips.txt") if ln.strip()]
    clips = [(c, int(o)) for c, o in clips]
    model = init_model(CFG, CKPT, device=DEV); model.eval()
    print(f"[c2] model loaded; {len(clips)} clips", flush=True)
    blank_pts = np.zeros((50, 5), np.float32); blank_imgs = [np.zeros((256, 704, 3), np.uint8) for _ in CAM_ORDER]

    rows = []
    for n, (c, isood) in enumerate(clips):
        lid = glob.glob(f"{ROOT}/lidar/lidar_top_360fov/*/{c}.lidar_top_360fov.parquet")
        if not lid: continue
        camp = {s: glob.glob(f"{ROOT}/camera/{s}/*/{c}.{s}.mp4") for s in set(CAM_ORDER)}
        if not camp.get("camera_front_wide_120fov"): continue
        sims, richs = [], []
        try:
            for fr in FRACS:
                pts = lidar_frac(lid[0], fr)
                imgs = [cam_frac(camp[s][0], fr) if camp.get(s) else np.zeros((256, 704, 3), np.uint8) for s in CAM_ORDER]
                b, s_, l_ = boxes(model, pts, imgs)
                si, ri = loads(b, s_, l_, ego_speed(c, fr))
                sims.append(si); richs.append(ri)
        except Exception as e:
            print(f"  [WARN] {c[:8]}: {e}", flush=True); continue
        rec = dict(clip=c, ood=isood, simple=st.mean(sims), rich=st.mean(richs))
        if len([r for r in rows if "blank" in r]) < 20:
            bb, bs, bl = boxes(model, blank_pts, blank_imgs)
            rec["blank"] = loads(bb, bs, bl, 5.0)[1]
        rows.append(rec)
        if (n+1) % 25 == 0: print(f"[c2] {n+1}/{len(clips)}", flush=True)

    no = sum(r["ood"] for r in rows)
    print(f"\n===== STRENGTHENED CONFLICT (N={len(rows)}, ood={no}, non={len(rows)-no}) =====")
    print(f"[v1 simple, multi-frame]  AUC={auc(rows,'simple'):.3f}   (single-frame v1 was 0.605)")
    print(f"[v2 rich  , multi-frame]  AUC={auc(rows,'rich'):.3f}   (class-weight + TTC)")
    ro = [r['rich'] for r in rows if r['ood']]; rn = [r['rich'] for r in rows if not r['ood']]
    print(f"  rich load: OOD mean={st.mean(ro):.3f} | non mean={st.mean(rn):.3f}")
    nc = [(r['rich'], r['blank']) for r in rows if 'blank' in r]
    if nc: print(f"[negative control] rich real mean={st.mean([a for a,_ in nc]):.3f}  blank mean={st.mean([b for _,b in nc]):.3f}")
    print(">>> CONFLICT2 DONE")


if __name__ == "__main__":
    main()
