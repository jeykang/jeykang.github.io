#!/usr/bin/env python3
"""Camera-ONLY perception difficulty axis (for the camera-only downstream endgame).

The production perceptual axis uses lidar-fused BEVFusion confidence, which is
night/weather-robust (lidar masks camera degradation) -> it under-rates exactly the
clips the consumer's camera-only final product will struggle with. This runner
measures perception difficulty with a CAMERA-ONLY detector (YOLO), mirroring the
existing mean_max_conf definition so it can swap into the difficulty union.

Per clip: sample FRACS front-camera frames, run YOLO over AD-relevant classes, take
the per-frame max confidence, average -> cam_max_conf. camera_low_conf = 1 - that
(high = hard to perceive). Writes <NFS>/.camera_perception/camera_perception.parquet.
v1 = YOLO-2D (front-wide); a camera-3D variant (fcos3d/pgd) is the planned upgrade.
"""
import glob, os, sys, time
import numpy as np, cv2
import pyarrow as pa, pyarrow.parquet as pq
from ultralytics import YOLO

cv2.setNumThreads(2)   # decode-bound + sharded -> limit per-process threads
SHARD = int(os.environ.get("SHARD", "0"))
NSHARDS = int(os.environ.get("NSHARDS", "1"))

_C = "/mnt/netai-e2e/nvidia-physicalai-av-subset"
ROOT = _C if os.path.isdir(_C) else os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "netai-e2e", "nvidia-physicalai-av-subset")
OUT = f"{ROOT}/.camera_perception"
SENSOR = "camera_front_wide_120fov"
FRACS = [float(x) for x in os.environ.get("FRACS", "0.3,0.5,0.7").split(",")]
AD = [0, 1, 2, 3, 5, 7, 9, 11]   # person,bicycle,car,motorcycle,bus,truck,traffic light,stop sign
WEIGHTS = os.environ.get("YOLO_W", "yolo11x.pt")
DEV = int(os.environ.get("DEV", "0"))


def grab(mp4):
    cap = cv2.VideoCapture(mp4); n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); out = []
    for fr in FRACS:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(n * fr)); ok, f = cap.read()
        if ok:
            out.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    cap.release(); return out


def main():
    model = YOLO(WEIGHTS)
    mp4s = sorted(glob.glob(f"{ROOT}/camera/{SENSOR}/*/*.mp4"))[SHARD::NSHARDS]
    print(f"[campercep] shard {SHARD}/{NSHARDS}: {len(mp4s)} clips, detector={WEIGHTS} dev={DEV}", flush=True)
    rows = []; t0 = time.time()
    for k, mp4 in enumerate(mp4s):
        cid = os.path.basename(mp4).split(".")[0]
        try:
            frames = grab(mp4)
            if not frames:
                continue
            res = model.predict(frames, conf=0.05, classes=AD, verbose=False, device=DEV)
            maxc, nd = [], []
            for r in res:
                c = r.boxes.conf.cpu().numpy()
                maxc.append(float(c.max()) if len(c) else 0.0)
                nd.append(int((c > 0.3).sum()))
            cam_max_conf = float(np.mean(maxc))
            rows.append({"clip_id": cid, "cam_max_conf": round(cam_max_conf, 4),
                         "camera_low_conf": round(1.0 - cam_max_conf, 4),
                         "cam_ndet": round(float(np.mean(nd)), 2)})
        except Exception as e:
            print(f"  [WARN] {cid[:8]}: {str(e)[:60]}", flush=True)
        if (k + 1) % 1000 == 0:
            print(f"[campercep] {k+1} clips ({(k+1)/(time.time()-t0):.1f} c/s)", flush=True)

    os.makedirs(OUT, exist_ok=True)
    fn = (f"{OUT}/camera_perception_shard_{SHARD:02d}_of_{NSHARDS:02d}.parquet"
          if NSHARDS > 1 else f"{OUT}/camera_perception.parquet")
    pq.write_table(pa.Table.from_pylist(rows), fn)
    lc = [r["camera_low_conf"] for r in rows]
    print(f"[campercep] shard {SHARD} wrote {len(rows)} clips; camera_low_conf mean={np.mean(lc):.3f}", flush=True)

    if NSHARDS == 1 and os.path.exists("/tmp/conf/clips.txt"):
        lab = {c: int(o) for c, o in (l.strip().split(",") for l in open("/tmp/conf/clips.txt") if l.strip())}
        ev = [(r["camera_low_conf"], lab[r["clip_id"]]) for r in rows if r["clip_id"] in lab]
        if ev:
            pos = [v for v, o in ev if o]; neg = [v for v, o in ev if not o]
            al = sorted(ev, key=lambda p: p[0]); rs = sum(i + 1 for i, p in enumerate(al) if p[1])
            auc = (rs - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))
            print(f"[campercep] self-check OOD AUC = {auc:.3f} (N={len(ev)}; note: OOD is behavioral, "
                  f"camera-perceptual is a different difficulty type)", flush=True)
    print(">>> CAMERA PERCEPTION DONE", flush=True)


if __name__ == "__main__":
    main()
