#!/usr/bin/env python3
"""extract_assets.py — build a real-clip asset library for the wall display.

Samples N random clips from the NFS Nvidia PhysicalAI subset that have BOTH a
front-wide camera mp4 and a LiDAR parquet, then for each clip:
  - extracts one representative camera frame  -> assets/clips/<id>.jpg
  - decodes one representative LiDAR spin     -> assets/clips/<id>.bin (Float32 xyz)
  - pulls real metadata (country / hour / month / season)
  - (optional) runs YOLOv8n for real 2D detections        [--yolo]
  - (optional) pulls the real Gold difficulty score from Trino [--trino]
and writes assets/clips/manifest.json. The wall app (app.js) cycles through this
playlist so every loop streams a different real clip through the pipeline.

Core deps : opencv-python-headless  DracoPy  pyarrow  numpy
Optional  : ultralytics (--yolo)   trino (--trino)

Run inside a venv (see README). Example:
  python extract_assets.py --n 40 --yolo
"""
from __future__ import annotations
import argparse
import json
import os
import random
import struct
import sys
import time
from glob import glob
from pathlib import Path

import numpy as np

NFS_DEFAULT = "./netai-e2e/nvidia-physicalai-av-subset"
FRONT_CAM = "camera_front_wide_120fov"
SEASONS_N = {12: "winter", 1: "winter", 2: "winter", 3: "spring", 4: "spring",
             5: "spring", 6: "summer", 7: "summer", 8: "summer", 9: "autumn",
             10: "autumn", 11: "autumn"}


def log(*a):
    print("[extract]", *a, flush=True)


def find_candidates(nfs: str):
    cam_glob = f"{nfs}/camera/{FRONT_CAM}/*/*.{FRONT_CAM}.mp4"
    cams = {}
    for p in glob(cam_glob):
        cid = os.path.basename(p).split(".")[0]
        cams[cid] = p
    lidar_glob = f"{nfs}/lidar/lidar_top_360fov/*/*.lidar_top_360fov.parquet"
    lids = {}
    for p in glob(lidar_glob):
        cid = os.path.basename(p).split(".")[0]
        lids[cid] = p
    both = [(cid, cams[cid], lids[cid]) for cid in cams if cid in lids]
    return both


def load_metadata(nfs: str):
    """Best-effort per-clip metadata from data_collection.parquet."""
    meta = {}
    candidates = [
        f"{nfs}/metadata/data_collection.parquet",
        f"{nfs}/data_collection.parquet",
    ]
    path = next((c for c in candidates if os.path.exists(c)), None)
    if not path:
        log("no data_collection.parquet found — metadata will be sparse")
        return meta
    try:
        import pyarrow.parquet as pq
        t = pq.read_table(path)
        cols = {c.lower(): c for c in t.column_names}
        log("data_collection columns:", list(cols.keys())[:20])

        def col(*names):
            for n in names:
                if n in cols:
                    return t.column(cols[n]).to_pylist()
            return None

        ids = col("clip_id", "clip_uuid", "id")
        if ids is None:
            log("no clip_id column in data_collection")
            return meta
        country = col("country", "country_name", "region")
        hour = col("hour_of_day", "hour", "local_hour")
        month = col("month", "collection_month")
        for i, cid in enumerate(ids):
            meta[cid] = {
                "country": country[i] if country else None,
                "hour": hour[i] if hour else None,
                "month": month[i] if month else None,
            }
        log(f"loaded metadata for {len(meta):,} clips")
    except Exception as e:
        log("metadata load failed:", e)
    return meta


_FACTOR_LABELS = {
    "time_of_day": "time of day", "season_geography": "season & geography",
    "sensor_coverage": "sensor coverage", "ego_dynamics": "ego dynamics",
    "obstacle_density": "obstacle density", "perception": "perception",
}


def _dominant_factor(detail: str):
    """Parse the sub-score JSON in clip_scores.detail → human factor label."""
    try:
        d = json.loads(detail)
        subs = d.get("sub_scores", d)
        best, bestv = None, -1
        for k, v in subs.items():
            if isinstance(v, (int, float)) and v is not None and v > bestv:
                best, bestv = k, v
        return _FACTOR_LABELS.get(best, "edge case")
    except Exception:
        return "edge case"


def load_trino_scores():
    """Best-effort: pull real Gold difficulty score + dominant factor per clip."""
    scores = {}
    try:
        import trino  # type: ignore
    except Exception:
        log("trino client not installed — skipping real scores (--trino)")
        return scores
    try:
        conn = trino.dbapi.connect(host="localhost", port=8080, user="wall",
                                   catalog="iceberg", schema="nvidia_gold")
        cur = conn.cursor()
        cur.execute("SELECT clip_id, difficulty_score, detail FROM iceberg.nvidia_gold.clip_scores")
        for cid, sc, detail in cur.fetchall():
            scores[cid] = {"score": float(sc), "factor": _dominant_factor(detail)}
        log(f"pulled {len(scores):,} real difficulty scores from Trino")
    except Exception as e:
        log("trino score query failed:", e)
    return scores


def extract_frame(mp4: str, out_jpg: str, size):
    """Grab a representative frame ~40% into the clip, resize, save jpg.

    Returns an RGB ndarray (for optional YOLO) or None on failure.
    """
    import cv2
    cap = cv2.VideoCapture(mp4)
    if not cap.isOpened():
        return None
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    target = int(n * 0.4) if n else 60
    cap.set(cv2.CAP_PROP_POS_FRAMES, target)
    ok, bgr = cap.read()
    cap.release()
    if not ok or bgr is None:
        return None
    bgr = cv2.resize(bgr, size, interpolation=cv2.INTER_AREA)
    cv2.imwrite(out_jpg, bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)  # rgb ndarray for optional yolo


def decode_cloud(parquet: str, out_bin: str, max_points: int):
    """Decode one LiDAR spin (~40% in), subsample, write Float32 xyz."""
    import DracoPy
    import pyarrow.parquet as pq
    t = pq.read_table(parquet, columns=["draco_encoded_pointcloud"])
    blobs = t.column("draco_encoded_pointcloud").to_pylist()
    if not blobs:
        return 0
    idx = int(len(blobs) * 0.4)
    mesh = DracoPy.decode(blobs[idx])
    pts = np.asarray(mesh.points, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] < 3:
        return 0
    pts = pts[:, :3]
    if len(pts) > max_points:
        sel = np.random.choice(len(pts), max_points, replace=False)
        pts = pts[sel]
    pts.astype("<f4").tofile(out_bin)
    return len(pts)


def run_yolo(model, rgb, size):
    """Return list of center-normalized detection dicts, or []."""
    try:
        res = model.predict(rgb, verbose=False, imgsz=640, conf=0.30)[0]
        H, W = rgb.shape[:2]
        names = res.names
        out = []
        for b in res.boxes:
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            out.append({
                "x": round((x1 + x2) / 2 / W, 4), "y": round((y1 + y2) / 2 / H, 4),
                "w": round((x2 - x1) / W, 4), "h": round((y2 - y1) / H, 4),
                "label": names[int(b.cls[0])], "conf": round(float(b.conf[0]), 3),
            })
        return out[:8]
    except Exception as e:
        log("yolo predict failed:", e)
        return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--nfs", default=NFS_DEFAULT)
    ap.add_argument("--out", default="demo_wall/assets/clips")
    ap.add_argument("--frame-w", type=int, default=960)
    ap.add_argument("--frame-h", type=int, default=540)
    ap.add_argument("--cloud-points", type=int, default=12000)
    ap.add_argument("--yolo", action="store_true", help="run YOLOv8n for real detections")
    ap.add_argument("--trino", action="store_true", help="pull real Gold scores from Trino")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    nfs = args.nfs.rstrip("/")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    size = (args.frame_w, args.frame_h)

    log("scanning NFS for candidate clips ...")
    cands = find_candidates(nfs)
    log(f"{len(cands):,} clips have both front-wide camera + LiDAR")
    if not cands:
        log("FATAL: no candidates — check --nfs path")
        sys.exit(1)

    meta = load_metadata(nfs)
    scores = load_trino_scores() if args.trino else {}

    # Prefer clips that ALSO have a real Gold score, so every displayed clip
    # shows a real difficulty score + dominant factor. The on-disk LiDAR set
    # and the scored set only partially overlap (random subset), so fall back
    # to the full candidate pool if the intersection is too small.
    if scores:
        scored = [c for c in cands if c[0] in scores]
        log(f"{len(scored):,} of those also have a real Gold score")
        if len(scored) >= args.n:
            cands = scored
        else:
            log("intersection < --n; keeping full pool (some clips will lack scores)")
    random.shuffle(cands)
    cands = cands[: args.n]

    model = None
    if args.yolo:
        try:
            from ultralytics import YOLO
            model = YOLO("yolov8n.pt")
            log("YOLOv8n loaded for real 2D detections")
        except Exception as e:
            log("ultralytics unavailable, continuing without detections:", e)

    clips = []
    t0 = time.time()
    for i, (cid, mp4, parquet) in enumerate(cands):
        jpg = out / f"{cid}.jpg"
        binf = out / f"{cid}.bin"
        try:
            rgb = extract_frame(mp4, str(jpg), size)
            if rgb is None:
                log(f"  skip {cid[:8]} (no frame)"); continue
            npts = decode_cloud(parquet, str(binf), args.cloud_points)
        except Exception as e:
            log(f"  skip {cid[:8]}: {e}"); continue

        m = meta.get(cid, {})
        month = m.get("month")
        season = SEASONS_N.get(int(month), None) if month not in (None, "") else None
        country = m.get("country")
        hour = m.get("hour")
        where_bits = [str(country) if country else None,
                      season,
                      (f"{int(hour):02d}:00" if hour not in (None, "") else None)]
        where = " · ".join(b for b in where_bits if b)
        dets = run_yolo(model, rgb, size) if model is not None else None
        sc = scores.get(cid)

        clips.append({
            "id": cid,
            "img": f"assets/clips/{cid}.jpg",
            "cloud": f"assets/clips/{cid}.bin" if npts else None,
            "n_points": npts,
            "country": country, "season": season, "hour": hour,
            "where": where or "location withheld",
            "score": round(sc["score"], 3) if sc else None,
            "factor": sc["factor"] if sc else "edge case",
            "detections": dets,
        })
        if (i + 1) % 5 == 0 or i + 1 == len(cands):
            log(f"  [{i+1}/{len(cands)}] {time.time()-t0:.0f}s  last={cid[:8]} pts={npts}")

    manifest = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "source": "NVIDIA PhysicalAI — Autonomous Vehicles (NFS subset)",
        "total_clips": 310895,
        "total_scored": 33719,
        "clips": clips,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    log(f"DONE: {len(clips)} clips -> {out}/manifest.json  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
