"""Bounded gate for DiffusionDrive: is the planning signal discriminative AND
additive beyond ego_dynamics (the rung-0 baseline)?

Loads the model once, scores N clips, captures the 20-mode distribution
(mode_spread = spatial diversity of candidate trajectories; entropy; final-plan
endpoint), computes ego_dynamics from egomotion, and reports discrimination
(stdev) + Spearman(signal, ego_dynamics). Low |Spearman| => the planner adds
signal the cheap kinematic score doesn't have.
"""
import glob, math, os, sys
import numpy as np, torch, statistics as st
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_one_clip import (CKPT, camera_feature, lidar_feature, status_feature)


def ego_dynamics(root, clip):
    m = glob.glob(f"{root}/labels/egomotion/*/{clip}.egomotion.parquet")
    d = pq.read_table(m[0], columns=["ax", "ay", "az", "curvature"]).to_pydict()
    amag = [math.sqrt(a*a+b*b+c*c) for a, b, c in zip(d["ax"], d["ay"], d["az"])]
    a = st.pstdev(amag) if len(amag) > 1 else 0.0
    c = st.pstdev(d["curvature"]) if len(d["curvature"]) > 1 else 0.0
    return 0.6*min(1.0, a/3.0) + 0.4*min(1.0, c/0.1)


def spearman(a, b):
    def rk(v):
        o = sorted(range(len(v)), key=lambda i: v[i]); r = [0.0]*len(v); i = 0
        while i < len(v):
            j = i
            while j+1 < len(v) and v[o[j+1]] == v[o[i]]:
                j += 1
            av = (i+j)/2+1
            for k in range(i, j+1):
                r[o[k]] = av
            i = j+1
        return r
    rx, ry = rk(a), rk(b); n = len(a); mx = sum(rx)/n; my = sum(ry)/n
    cov = sum((rx[i]-mx)*(ry[i]-my) for i in range(n))
    sx = math.sqrt(sum((x-mx)**2 for x in rx)); sy = math.sqrt(sum((y-my)**2 for y in ry))
    return cov/(sx*sy) if sx*sy else 0.0


def main():
    root = "/mnt/netai-e2e/nvidia-physicalai-av-subset"
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

    spreads, ents, endpts, egos = [], [], [], []
    for c in clips:
        try:
            feats = {
                "camera_feature": camera_feature(root, c)[None].cuda(),
                "lidar_feature": lidar_feature(root, c)[None].cuda(),
                "status_feature": status_feature(root, c)[None].cuda(),
            }
            with torch.no_grad():
                out = agent.forward(feats)
        except SystemExit:
            continue
        reg = cap["reg"][0].float().cpu().numpy(); cls = cap["cls"][0].float().cpu()
        p = torch.softmax(cls, dim=-1).numpy()
        ent = float(-(p*np.log(p+1e-9)).sum()/np.log(len(p)))
        ends = reg[:, -1, :2]; spread = float(np.sqrt(((ends-ends.mean(0))**2).sum(1).mean()))
        traj = out["trajectory"][0].float().cpu().numpy(); endpt = float(np.hypot(traj[-1, 0], traj[-1, 1]))
        eg = ego_dynamics(root, c)
        spreads.append(spread); ents.append(ent); endpts.append(endpt); egos.append(eg)
        print(f"{c[:8]} spread={spread:.2f} ent={ent:.3f} endpt={endpt:.1f} ego={eg:.3f}", flush=True)

    n = len(spreads)
    print(f"\nN={n}")
    print(f"mode_spread : mean={st.mean(spreads):.2f} stdev={st.pstdev(spreads):.2f} range=[{min(spreads):.1f},{max(spreads):.1f}]")
    print(f"mode_entropy: mean={st.mean(ents):.3f} stdev={st.pstdev(ents):.3f}")
    print(f"endpoint(m) : mean={st.mean(endpts):.1f} stdev={st.pstdev(endpts):.1f} range=[{min(endpts):.1f},{max(endpts):.1f}]")
    print(f"Spearman(mode_spread, ego_dynamics) = {spearman(spreads, egos):.3f}")
    print(f"Spearman(endpoint,    ego_dynamics) = {spearman(endpts, egos):.3f}")
    print(">>> GATE DONE")


if __name__ == "__main__":
    main()
