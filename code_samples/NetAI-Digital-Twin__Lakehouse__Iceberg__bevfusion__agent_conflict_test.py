"""Prototype + validate an agent-interaction (conflict) difficulty signal from
BEVFusion boxes.

conflict_load = score-weighted inverse-distance sum of detected agents in the
forward conflict zone (x in [0,40] m, |y| < 8 m, ego/lidar frame). Scene-driven
by construction.

Tests:
  - Negative control: blank lidar+camera -> conflict should collapse (proves the
    signal is driven by detected agents, not priors).
  - External labels: AUC(conflict, OOD hard-event clips) -> should be > 0.5
    (pedestrian-density / proximity = high conflict), the opposite of mode_spread.

clips.txt: lines "clip_id,is_ood".
"""
import glob, math, os, sys, statistics as st
import numpy as np, torch

sys.path.insert(0, "/workspace")
import projects.BEVFusion.bevfusion  # noqa: registers BEVFusion
from mmdet3d.apis import init_model
from bevfusion_infer import build_data, CAM_ORDER
from test_one_clip import load_lidar_points, load_camera_frame, find_files

CFG = "/workspace/mmdetection3d/projects/BEVFusion/configs/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py"
CKPT = "/workspace/checkpoints/bevfusion_lidar-cam_spconv2.pth"
ROOT = "/mnt/netai-e2e/nvidia-physicalai-av-subset"
DEV = "cuda:0"


def get_boxes(model, points, cam_imgs):
    data = build_data(points, cam_imgs, DEV)
    with torch.no_grad():
        out = model.test_step(data)
    pi = out[0].pred_instances_3d
    return pi.bboxes_3d.tensor.float().cpu().numpy(), pi.scores_3d.float().cpu().numpy()


def conflict(boxes, scores):
    if len(boxes) == 0:
        return 0.0, 999.0
    x, y = boxes[:, 0], boxes[:, 1]
    d = np.hypot(x, y)
    fwd = (x > 0) & (x < 40) & (np.abs(y) < 8)
    if not fwd.any():
        return 0.0, 999.0
    load = float((scores[fwd] / (1.0 + d[fwd])).sum())     # score-weighted, inverse-distance
    return load, float(d[fwd].min())


def auc(rows, key):
    o = [r[key] for r in rows if r["ood"]]; n = [r[key] for r in rows if not r["ood"]]
    if not o or not n:
        return float("nan")
    al = sorted(rows, key=lambda r: r[key])
    rsum = sum(i + 1 for i, r in enumerate(al) if r["ood"])
    return (rsum - len(o) * (len(o) + 1) / 2) / (len(o) * len(n))


def main():
    clips = []
    for line in open("/work/clips.txt"):
        c, o = line.strip().split(","); clips.append((c, int(o)))
    model = init_model(CFG, CKPT, device=DEV); model.eval()
    print(f"[conf] model loaded; {len(clips)} clips", flush=True)

    blank_pts = np.zeros((50, 5), np.float32)
    blank_imgs = [np.zeros((256, 704, 3), np.uint8) for _ in CAM_ORDER]

    rows = []
    for n, (c, isood) in enumerate(clips):
        cams, lidar = find_files(ROOT, c)
        if not lidar or not cams.get("camera_front_wide_120fov"):
            continue
        try:
            pts = load_lidar_points(lidar, 100)
            imgs = [load_camera_frame(cams[s]) if cams.get(s) else np.zeros((256, 704, 3), np.uint8)
                    for s in CAM_ORDER]
            b, sc = get_boxes(model, pts, imgs)
        except Exception as e:
            print(f"  [WARN] {c[:8]}: {e}", flush=True); continue
        load, mind = conflict(b, sc)
        rec = dict(clip=c, ood=isood, load=load, mind=mind, ndet=int((sc > 0.1).sum()))
        if len([r for r in rows if "load_blank" in r]) < 20:
            bb, ss = get_boxes(model, blank_pts, blank_imgs)
            rec["load_blank"] = conflict(bb, ss)[0]
        rows.append(rec)
        if (n + 1) % 25 == 0:
            print(f"[conf] {n+1}/{len(clips)}", flush=True)

    ood = [r["load"] for r in rows if r["ood"]]; non = [r["load"] for r in rows if not r["ood"]]
    print(f"\n===== AGENT-CONFLICT VALIDATION (N={len(rows)}, ood={len(ood)}, non={len(non)}) =====")
    print(f"conflict_load: OOD mean={st.mean(ood):.3f} median={st.median(ood):.3f} | non mean={st.mean(non):.3f} median={st.median(non):.3f}")
    print(f">>> AUC(OOD higher conflict) = {auc(rows,'load'):.3f}  (>0.5 = valid direction)")
    print(f">>> AUC(OOD via ndet)        = {auc(rows,'ndet'):.3f}")
    nc = [(r["load"], r["load_blank"]) for r in rows if "load_blank" in r]
    if nc:
        print(f"[negative control] real load mean={st.mean([a for a,_ in nc]):.3f}  blank-scene load mean={st.mean([b for _,b in nc]):.3f}  (blank≈0 => scene-driven)")
    nd = [r["ndet"] for r in rows]
    print(f"[detections] ndet(score>0.1): mean={st.mean(nd):.1f} median={st.median(nd)} zero-frac={sum(1 for x in nd if x==0)/len(nd):.2f}")
    print(">>> CONFLICT TEST DONE")


if __name__ == "__main__":
    main()
