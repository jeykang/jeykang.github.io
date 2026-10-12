"""Validity battery for the DiffusionDrive planning signal (mode_spread).

For a clip sample, runs:
  1. Negative-control ablations: blank-lidar / blank-camera -> is the score
     driven by scene content (necessary for validity)?
  2. Plan-vs-GT: DiffusionDrive's final plan vs the actual human ego future
     (open-loop L2) -> is the planner actually working on PhysicalAI?
  3. Frame stability + determinism: score at idx 0.3/0.5/0.7 and a 0.5 re-run
     -> is it a stable scene property vs single-frame noise?
  4. Convergent: Spearman(mode_spread, {ego_dynamics, bev_occupancy, perception}).

Reports aggregate stats + a verdict per axis. Read-only on data.
"""
import glob, math, os, sys, statistics as st
import cv2, numpy as np, torch
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_one_clip import CKPT, CAM_L, CAM_F, CAM_R, SPLIT_H, MAXH, HISTMAX, BEV, LMINX, LMAXX, LMINY, LMAXY

ROOT = "/mnt/netai-e2e/nvidia-physicalai-av-subset"


def _frame(mp4, idx):
    cap = cv2.VideoCapture(mp4); n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(n*idx)); ok, f = cap.read(); cap.release()
    if not ok: raise RuntimeError("read")
    return cv2.cvtColor(f, cv2.COLOR_BGR2RGB)


def camera_feat(clip, idx):
    def g(s):
        m = glob.glob(f"{ROOT}/camera/{s}/*/{clip}.{s}.mp4")
        if not m: raise SystemExit("nocam")
        return _frame(m[0], idx)
    l, f, r = g(CAM_L), g(CAM_F), g(CAM_R)
    l = l[28:-28, 416:-416]; f = f[28:-28]; r = r[28:-28, 416:-416]
    s = cv2.resize(np.concatenate([l, f, r], axis=1), (1024, 256))
    return torch.from_numpy(s.transpose(2, 0, 1)).float()/255.0


def lidar_feat_and_occ(clip, idx):
    import DracoPy
    m = glob.glob(f"{ROOT}/lidar/lidar_top_360fov/*/{clip}.lidar_top_360fov.parquet")
    if not m: raise SystemExit("nolidar")
    blobs = pq.read_table(m[0], columns=["draco_encoded_pointcloud"]).column(0).to_pylist()
    pts = np.asarray(DracoPy.decode(blobs[int(len(blobs)*idx)]).points, np.float32)
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    a = (z > SPLIT_H) & (z < MAXH)
    hist, _, _ = np.histogram2d(x[a], y[a], bins=BEV, range=[[LMINX, LMAXX], [LMINY, LMAXY]])
    occ = float((hist > 0).mean())
    hist = np.clip(hist, 0, HISTMAX)/HISTMAX
    return torch.from_numpy(hist[None].astype(np.float32)), occ


def status_feat(clip, idx):
    m = glob.glob(f"{ROOT}/labels/egomotion/*/{clip}.egomotion.parquet")
    d = pq.read_table(m[0], columns=["timestamp", "x", "y", "vx", "vy", "ax", "ay"]).to_pydict()
    o = sorted(range(len(d["timestamp"])), key=lambda k: d["timestamp"][k]); k = o[int(len(o)*idx)]
    return torch.tensor([0., 1., 0., 0., d["vx"][k], d["vy"][k], d["ax"][k], d["ay"][k]], dtype=torch.float32)


def ego_dynamics(clip):
    m = glob.glob(f"{ROOT}/labels/egomotion/*/{clip}.egomotion.parquet")
    d = pq.read_table(m[0], columns=["ax", "ay", "az", "curvature"]).to_pydict()
    amag = [math.sqrt(a*a+b*b+c*c) for a, b, c in zip(d["ax"], d["ay"], d["az"])]
    A = st.pstdev(amag) if len(amag) > 1 else 0.0
    C = st.pstdev(d["curvature"]) if len(d["curvature"]) > 1 else 0.0
    return 0.6*min(1.0, A/3.0) + 0.4*min(1.0, C/0.1)


def gt_l2(clip, plan):
    """mean L2 (m) between the 8-pose plan (ego frame) and GT ego future."""
    m = glob.glob(f"{ROOT}/labels/egomotion/*/{clip}.egomotion.parquet")
    d = pq.read_table(m[0], columns=["timestamp", "x", "y"]).to_pydict()
    o = sorted(range(len(d["timestamp"])), key=lambda k: d["timestamp"][k])
    ts = [d["timestamp"][k] for k in o]; xs = [d["x"][k] for k in o]; ys = [d["y"][k] for k in o]
    i = len(ts)//2
    # heading from ~2s of travel
    j2 = i
    while j2 < len(ts)-1 and ts[j2] < ts[i]+2_000_000: j2 += 1
    head = math.atan2(ys[j2]-ys[i], xs[j2]-xs[i])
    ch, sh = math.cos(-head), math.sin(-head)
    errs = []
    for k in range(plan.shape[0]):
        tgt = ts[i] + (k+1)*500_000
        if tgt > ts[-1]: break
        jj = i
        while jj < len(ts)-1 and ts[jj] < tgt: jj += 1
        dx, dy = xs[jj]-xs[i], ys[jj]-ys[i]
        ex = ch*dx - sh*dy; ey = sh*dx + ch*dy          # GT in ego frame
        errs.append(math.hypot(plan[k, 0]-ex, plan[k, 1]-ey))
    return st.mean(errs) if errs else float("nan")


def spearman(a, b):
    pairs = [(x, y) for x, y in zip(a, b) if not (math.isnan(x) or math.isnan(y))]
    if len(pairs) < 3: return float("nan")
    a, b = zip(*pairs)
    def rk(v):
        o = sorted(range(len(v)), key=lambda i: v[i]); r = [0.0]*len(v); i = 0
        while i < len(v):
            j = i
            while j+1 < len(v) and v[o[j+1]] == v[o[i]]: j += 1
            av = (i+j)/2+1
            for k in range(i, j+1): r[o[k]] = av
            i = j+1
        return r
    rx, ry = rk(a), rk(b); n = len(a); mx = sum(rx)/n; my = sum(ry)/n
    cov = sum((rx[i]-mx)*(ry[i]-my) for i in range(n))
    sx = math.sqrt(sum((x-mx)**2 for x in rx)); sy = math.sqrt(sum((y-my)**2 for y in ry))
    return cov/(sx*sy) if sx*sy else float("nan")


def main():
    clips = [c.strip() for c in open("/work/clips.txt") if c.strip()]
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

    def run(cam, lid, sta):
        feats = {"camera_feature": cam[None].cuda(), "lidar_feature": lid[None].cuda(),
                 "status_feature": sta[None].cuda()}
        with torch.no_grad():
            out = agent.forward(feats)
        reg = cap["reg"][0].float().cpu().numpy()
        ends = reg[:, -1, :2]
        spread = float(np.sqrt(((ends-ends.mean(0))**2).sum(1).mean()))
        return spread, out["trajectory"][0].float().cpu().numpy()

    # load perception scores for convergent check
    perc = {}
    for f in glob.glob(f"{ROOT}/.perception/bevfusion_shard_*.parquet"):
        for r in pq.read_table(f, columns=["clip_id", "perception_score"]).to_pylist():
            if r["perception_score"] is not None: perc[r["clip_id"]] = float(r["perception_score"])

    R = {k: [] for k in ["spread", "bl", "bc", "f03", "f07", "rerun", "gtl2", "ego", "occ", "perc"]}
    for n, c in enumerate(clips):
        try:
            cam5 = camera_feat(c, 0.5); lid5, occ = lidar_feat_and_occ(c, 0.5); sta5 = status_feat(c, 0.5)
        except SystemExit:
            continue
        sp, plan = run(cam5, lid5, sta5)
        R["spread"].append(sp); R["occ"].append(occ); R["ego"].append(ego_dynamics(c))
        R["gtl2"].append(gt_l2(c, plan)); R["perc"].append(perc.get(c, float("nan")))
        R["bl"].append(run(cam5, torch.zeros_like(lid5), sta5)[0])     # blank lidar
        R["bc"].append(run(torch.zeros_like(cam5), lid5, sta5)[0])     # blank camera
        try:
            R["f03"].append(run(camera_feat(c, 0.3), lidar_feat_and_occ(c, 0.3)[0], status_feat(c, 0.3))[0])
            R["f07"].append(run(camera_feat(c, 0.7), lidar_feat_and_occ(c, 0.7)[0], status_feat(c, 0.7))[0])
        except SystemExit:
            R["f03"].append(float("nan")); R["f07"].append(float("nan"))
        R["rerun"].append(run(cam5, lid5, sta5)[0] if n < 12 else float("nan"))
        if (n+1) % 25 == 0: print(f"[val] {n+1}/{len(clips)}", flush=True)

    sp = R["spread"]; nN = len(sp)
    print(f"\n===== VALIDITY BATTERY (N={nN}) =====")
    # 1. ablation
    dbl = st.mean(abs(a-b) for a, b in zip(sp, R["bl"]))
    dbc = st.mean(abs(a-b) for a, b in zip(sp, R["bc"]))
    print(f"[ablation] mean|spread|: real={st.mean(sp):.2f}  blank-lidar={st.mean(R['bl']):.2f}  blank-cam={st.mean(R['bc']):.2f}")
    print(f"[ablation] mean abs change vs real: blank-lidar={dbl:.2f}  blank-cam={dbc:.2f}  (large => modality is used)")
    print(f"[ablation] Spearman(real, blank-lidar)={spearman(sp, R['bl']):.3f}  Spearman(real, blank-cam)={spearman(sp, R['bc']):.3f}")
    # 2. determinism
    rr = [(a, b) for a, b in zip(sp, R["rerun"]) if not math.isnan(b)]
    if rr:
        print(f"[determinism] max|real - rerun| over {len(rr)} clips = {max(abs(a-b) for a, b in rr):.4f}")
    # 3. frame stability
    fst = [st.pstdev([a, b, cc]) for a, b, cc in zip(sp, R["f03"], R["f07"]) if not (math.isnan(b) or math.isnan(cc))]
    print(f"[stability] within-clip stdev across frames(0.3/0.5/0.7): mean={st.mean(fst):.2f}  vs between-clip stdev={st.pstdev(sp):.2f}  (ratio {st.mean(fst)/st.pstdev(sp):.2f})")
    # 4. plan-vs-GT
    gl = [v for v in R["gtl2"] if not math.isnan(v)]
    print(f"[plan-vs-GT] open-loop L2 (m): median={st.median(gl):.2f} mean={st.mean(gl):.2f}  (lower => planner tracks human driving)")
    # 5. convergent
    print(f"[convergent] Spearman(spread, ego_dynamics)={spearman(sp, R['ego']):.3f}")
    print(f"[convergent] Spearman(spread, bev_occupancy)={spearman(sp, R['occ']):.3f}")
    print(f"[convergent] Spearman(spread, perception)={spearman(sp, R['perc']):.3f}")
    print(">>> VALIDATE DONE")


if __name__ == "__main__":
    main()
