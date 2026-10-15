"""Strong agent-conflict signal from obstacle.offline (the dataset's own boxes).

GPU-free. Per clip: read tracked 3D boxes (rig/ego frame), pick decision frames,
compute a forward-zone conflict load with reliable VRU class-weighting + TTC
(track velocity vs ego speed). Validate vs OOD labels (AUC) and compare to the
zero-shot BEVFusion ceiling (0.633).

clips.txt: "clip_id,is_ood".
"""
import glob, io, math, os, statistics as st, zipfile
import pyarrow.parquet as pq

ROOT = "netai-e2e/nvidia-physicalai-av-subset"
OO = f"{ROOT}/labels/obstacle.offline"
FRACS = [0.3, 0.5, 0.7]
VRU = {"person", "rider", "stroller", "animal"}
WZ = {"protruding_object"}


def wclass(c):
    return 4.0 if c in VRU else (1.5 if c in WZ else 1.0)


_zcache = {}
def read_clip(clip):
    lid = glob.glob(f"{ROOT}/lidar/lidar_top_360fov/*/{clip}.lidar_top_360fov.parquet")
    if not lid:
        return None
    ch = next((p.split("chunk_")[1][:4] for p in [lid[0]] if "chunk_" in p), None)
    zp = f"{OO}/obstacle.offline.chunk_{ch}.zip"
    if not os.path.exists(zp):
        return None
    zf = _zcache.get(zp) or _zcache.setdefault(zp, zipfile.ZipFile(zp))
    nm = f"{clip}.obstacle.offline.parquet"
    if nm not in zf.namelist():
        return None
    return pq.read_table(io.BytesIO(zf.read(nm))).to_pydict()


def ego_speed_at(clip, ts):
    m = glob.glob(f"{ROOT}/labels/egomotion/*/{clip}.egomotion.parquet")
    if not m:
        return 5.0
    d = pq.read_table(m[0], columns=["timestamp", "vx", "vy"]).to_pydict()
    j = min(range(len(d["timestamp"])), key=lambda i: abs(d["timestamp"][i] - ts))
    return math.hypot(d["vx"][j], d["vy"][j])


def conflict(clip):
    """Reconstruct the scene at a decision TIME by taking each track's nearest
    detection within a window (obstacle.offline is per-track time series, NOT
    frame-synchronized — rows each carry their own timestamp)."""
    d = read_clip(clip)
    if d is None or not d["timestamp_us"]:
        return None
    n = len(d["timestamp_us"])
    tracks = {}
    for i in range(n):
        tracks.setdefault(d["track_id"][i], []).append(i)
    for t in tracks:
        tracks[t].sort(key=lambda i: d["timestamp_us"][i])
    ts_all = d["timestamp_us"]
    tmin, tmax = min(ts_all), max(ts_all)
    WIN = 100_000  # 0.1 s window to call a track "present" at T
    agg = {}
    for fr in FRACS:
        T = tmin + fr * (tmax - tmin)
        vego = ego_speed_at(clip, T)
        m = dict(simple=0.0, rich=0.0, vru_near=0.0, vru_min=0.0, allnear=0.0)
        vru_dists = []
        for tid, ids in tracks.items():
            j = min(ids, key=lambda i: abs(d["timestamp_us"][i] - T))
            if abs(d["timestamp_us"][j] - T) > WIN:
                continue
            x, y = d["center_x"][j], d["center_y"][j]
            dist = math.hypot(x, y)
            cls = d["label_class"][j]
            # any-direction proximity (peds cross / are lateral)
            if dist < 30:
                m["allnear"] += 1.0
                if cls in VRU:
                    m["vru_near"] += 1.0 / (1.0 + dist); vru_dists.append(dist)
            # forward-zone (v1/v2)
            if 0 < x < 40 and abs(y) < 8:
                m["simple"] += 1.0 / (1.0 + dist)
                pos = ids.index(j); vx = 0.0
                if pos + 1 < len(ids):
                    j2 = ids[pos + 1]
                    dt = max(1e-3, (d["timestamp_us"][j2] - d["timestamp_us"][j]) / 1e6)
                    vx = (d["center_x"][j2] - x) / dt
                m["rich"] += wclass(cls) / (1.0 + x / max(0.5, vego - vx))
        m["vru_min"] = 1.0 / (1.0 + min(vru_dists)) if vru_dists else 0.0
        for k in m: agg.setdefault(k, []).append(m[k])
    return {k: st.mean(v) for k, v in agg.items()}


def auc(rows, key):
    o = [r[key] for r in rows if r["ood"]]; nn = [r[key] for r in rows if not r["ood"]]
    if not o or not nn:
        return float("nan")
    al = sorted(rows, key=lambda r: r[key])
    rsum = sum(i + 1 for i, r in enumerate(al) if r["ood"])
    return (rsum - len(o) * (len(o) + 1) / 2) / (len(o) * len(nn))


def main():
    clips = [ln.strip().split(",") for ln in open("/tmp/conf/clips.txt") if ln.strip()]
    rows = []
    for k, (c, o) in enumerate(clips):
        r = conflict(c)
        if r is None:
            continue
        r["clip"] = c; r["ood"] = int(o); rows.append(r)
        if (k + 1) % 50 == 0:
            print(f"[oo-conf] {k+1}/{len(clips)}", flush=True)
    no = sum(r["ood"] for r in rows)
    print(f"\n===== OBSTACLE.OFFLINE CONFLICT (N={len(rows)}, ood={no}, non={len(rows)-no}) =====")
    for key in ["simple", "rich", "allnear", "vru_near", "vru_min"]:
        o = [r[key] for r in rows if r["ood"]]; nn = [r[key] for r in rows if not r["ood"]]
        print(f"[{key:9s}] AUC={auc(rows,key):.3f}  OOD mean={st.mean(o):.3f} | non mean={st.mean(nn):.3f}")
    print("(zero-shot BEVFusion forward-conflict ceiling was 0.633)")
    print(">>> OO CONFLICT DONE")


if __name__ == "__main__":
    main()
