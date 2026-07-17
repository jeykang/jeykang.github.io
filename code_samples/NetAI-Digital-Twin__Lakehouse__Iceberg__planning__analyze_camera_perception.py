"""Merge camera-only perception shards + quantify divergence from lidar-fused.

The divergence (clips HARD for the camera-only detector but EASY for fused BEVFusion)
is the population the consumer's camera-only final product will struggle with that the
current fused perceptual axis rates easy — the whole reason to re-point the axis.
"""
import glob, os
import numpy as np, pyarrow as pa, pyarrow.parquet as pq

ROOT = "netai-e2e/nvidia-physicalai-av-subset"
CAM = f"{ROOT}/.camera_perception"
FUSED = f"{ROOT}/.perception_bevfusion"


def ranknorm(x):
    x = np.asarray(x, float); o = x.argsort(); r = np.empty(len(x)); r[o] = np.arange(len(x))
    return r / max(1, len(x) - 1)


def spearman(a, b):
    return float(np.corrcoef(ranknorm(a), ranknorm(b))[0, 1])


# merge camera shards
shards = sorted(glob.glob(f"{CAM}/camera_perception_shard_*.parquet"))
tbl = pa.concat_tables([pq.read_table(s) for s in shards])
pq.write_table(tbl, f"{CAM}/camera_perception.parquet")
cam = tbl.to_pydict()
camd = {c: (mc, lc, nd) for c, mc, lc, nd in
        zip(cam["clip_id"], cam["cam_max_conf"], cam["camera_low_conf"], cam["cam_ndet"])}
print(f"merged {len(camd)} camera clips -> camera_perception.parquet")

# fused
ft = pa.concat_tables([pq.read_table(s) for s in glob.glob(f"{FUSED}/bevfusion_shard_*.parquet")]).to_pydict()
fused = {c: m for c, m in zip(ft["clip_id"], ft["mean_max_conf"])}
print(f"fused clips: {len(fused)}")

common = [c for c in camd if c in fused]
cam_lc = np.array([camd[c][1] for c in common])            # camera_low_conf (high=hard)
fused_lc = np.array([1.0 - fused[c] for c in common])      # fused low_conf
print(f"\ncommon clips: {len(common)}")
print(f"camera_low_conf : mean={cam_lc.mean():.3f}  std={cam_lc.std():.3f}")
print(f"fused  low_conf : mean={fused_lc.mean():.3f}  std={fused_lc.std():.3f}")
print(f"Spearman rho(camera, fused) = {spearman(cam_lc, fused_lc):.3f}   (low => measure different difficulty)")

# divergence: hard-for-camera (top quartile) but easy-for-fused (bottom half)
cr = ranknorm(cam_lc); fr = ranknorm(fused_lc)
div = (cr >= 0.75) & (fr <= 0.50)
print(f"\nDIVERGENCE (camera-hard top-25% AND fused-easy bottom-50%): "
      f"{int(div.sum())} clips ({100*div.mean():.1f}%)")
print("  -> camera-only final product struggles here; current fused axis rates them easy")

# OOD AUC (note: OOD is a behavioral label; camera-perceptual is a different axis)
if os.path.exists("/tmp/conf/clips.txt"):
    lab = {c: int(o) for c, o in (l.strip().split(",") for l in open("/tmp/conf/clips.txt") if l.strip())}
    ev = [(camd[c][1], lab[c]) for c in camd if c in lab]
    if ev:
        pos = [v for v, o in ev if o]; neg = [v for v, o in ev if not o]
        al = sorted(ev, key=lambda p: p[0]); rs = sum(i + 1 for i, p in enumerate(al) if p[1])
        auc = (rs - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))
        print(f"\ncamera_low_conf OOD AUC = {auc:.3f} (N={len(ev)}; OOD is behavioral, expect modest)")
print(">>> ANALYSIS DONE")
