"""NAVSIM-style trajectory-feasibility difficulty (rung-2, §13) — GATE.

Open-loop, non-reactive simulation over the dataset's own agent tracks
(`obstacle.offline`): roll out a small ego trajectory *vocabulary* against the
recorded agents and measure how constrained the safe set is.

  difficulty = fraction of candidate ego maneuvers that hit a collision/near-miss
               over the horizon (avg over a few decision times).

Empty road -> every maneuver safe -> 0. Boxed-in by agents -> few safe maneuvers
-> high. This is *feasibility* (collision-based), NOT trajectory openness/spread
(the inverse-construct that sank mode_spread/minADE/action-expert-spread). GPU-free.

Validates vs OOD labels (AUC) + correlation with the production `conflict` signal.
clips.txt: "clip_id,is_ood".
"""
import glob, io, math, os, statistics as st, zipfile
import pyarrow.parquet as pq

ROOT = "netai-e2e/nvidia-physicalai-av-subset"
OO = f"{ROOT}/labels/obstacle.offline"
FRACS = [0.3, 0.5, 0.7]        # decision times (fraction of clip)
H, DT = 3.0, 0.5               # horizon 3s, 0.5s steps
WIN = 150_000                  # 0.15s window to read an agent's position at a time
VRU = {"person", "rider", "stroller", "animal"}

# Ego trajectory vocabulary: longitudinal speed multipliers x target lateral offset.
SPEED_MULT = [0.0, 0.5, 1.0, 1.5]          # stop / slow / maintain / accelerate
LAT_OFF = [-3.0, -1.5, 0.0, 1.5, 3.0]      # swerve/lane-change targets (m, rig +y = left)
STEPS = [DT * (i + 1) for i in range(int(H / DT))]   # 0.5..3.0


def read_clip(clip):
    lid = glob.glob(f"{ROOT}/lidar/lidar_top_360fov/*/{clip}.lidar_top_360fov.parquet")
    if not lid:
        return None
    ch = lid[0].split("chunk_")[1][:4]
    zp = f"{OO}/obstacle.offline.chunk_{ch}.zip"
    if not os.path.exists(zp):
        return None
    zf = zipfile.ZipFile(zp)
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


def build_tracks(d):
    """track_id -> sorted list of (t_us, x, y, half_extent, is_vru)."""
    n = len(d["timestamp_us"])
    tr = {}
    for i in range(n):
        he = 0.5 * max(d["size_x"][i], d["size_y"][i])
        tr.setdefault(d["track_id"][i], []).append(
            (d["timestamp_us"][i], d["center_x"][i], d["center_y"][i], he,
             d["label_class"][i] in VRU))
    for t in tr:
        tr[t].sort()
    return tr


def agent_at(samples, t_us):
    """Nearest recorded position within WIN; None if the track isn't present then."""
    best = min(samples, key=lambda s: abs(s[0] - t_us))
    return best if abs(best[0] - t_us) <= WIN else None


def unsafe_fraction(tracks, T_us, vego, vru_weight=False):
    """Fraction of the ego vocabulary that collides/near-misses over the horizon."""
    cands = [(sm, lo) for sm in SPEED_MULT for lo in LAT_OFF]
    unsafe = 0.0
    wsum = 0.0
    for sm, lo in cands:
        hit = False; hit_w = 1.0
        for tk in STEPS:
            ex = sm * vego * tk                 # longitudinal (rig +x fwd)
            ey = lo * (tk / H)                   # lateral ramp to target offset
            t_us = T_us + int(tk * 1e6)
            for samples in tracks.values():
                a = agent_at(samples, t_us)
                if a is None:
                    continue
                _, ax, ay, he, is_vru = a
                clr = 1.5 + he                    # ego half-width + agent half-extent + buffer
                if math.hypot(ex - ax, ey - ay) < clr:
                    hit = True
                    if is_vru:
                        hit_w = 3.0               # VRU collision weighted heavier
                    break
            if hit:
                break
        w = hit_w if vru_weight else 1.0
        wsum += w
        if hit:
            unsafe += w
    return unsafe / wsum if wsum else 0.0


def score(clip):
    d = read_clip(clip)
    if d is None or not d["timestamp_us"]:
        return None
    tracks = build_tracks(d)
    ts = d["timestamp_us"]; tmin, tmax = min(ts), max(ts)
    plain, vruw = [], []
    for fr in FRACS:
        T = tmin + int(fr * (tmax - tmin))
        v = ego_speed_at(clip, T)
        plain.append(unsafe_fraction(tracks, T, v, vru_weight=False))
        vruw.append(unsafe_fraction(tracks, T, v, vru_weight=True))
    return st.mean(plain), st.mean(vruw)


def auc(rows, key):
    o = [r[key] for r in rows if r["ood"]]; nn = [r[key] for r in rows if not r["ood"]]
    if not o or not nn:
        return float("nan")
    al = sorted(rows, key=lambda r: r[key])
    rsum = sum(i + 1 for i, r in enumerate(al) if r["ood"])
    return (rsum - len(o) * (len(o) + 1) / 2) / (len(o) * len(nn))


def rank(xs):
    o = sorted(range(len(xs)), key=lambda i: xs[i]); r = [0.0] * len(xs); i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[o[j + 1]] == xs[o[i]]:
            j += 1
        for k in range(i, j + 1):
            r[o[k]] = (i + j) / 2.0 + 1
        i = j + 1
    return r


def spearman(a, b):
    if len(a) < 8:
        return float("nan")
    ra, rb = rank(a), rank(b); n = len(a); ma = sum(ra) / n; mb = sum(rb) / n
    num = sum((ra[i] - ma) * (rb[i] - mb) for i in range(n))
    da = sum((x - ma) ** 2 for x in ra) ** 0.5; db = sum((x - mb) ** 2 for x in rb) ** 0.5
    return num / (da * db) if da and db else float("nan")


def main():
    conf = {}
    for p in glob.glob(f"{ROOT}/.conflict/*.parquet"):
        dd = pq.read_table(p, columns=["clip_id", "conflict_score"]).to_pydict()
        conf.update(dict(zip(dd["clip_id"], dd["conflict_score"])))
    clips = [ln.strip().split(",") for ln in open("/tmp/conf/clips.txt") if ln.strip()]
    rows = []
    for k, (c, o) in enumerate(clips):
        r = score(c)
        if r is None:
            continue
        rows.append(dict(clip=c, ood=int(o), pdms=r[0], pdms_vru=r[1]))
        if (k + 1) % 50 == 0:
            print(f"[pdms] {k+1}/{len(clips)}", flush=True)
    no = sum(r["ood"] for r in rows)
    print(f"\n===== PDMS FEASIBILITY GATE (N={len(rows)}, ood={no}, non={len(rows)-no}) =====")
    for key in ["pdms", "pdms_vru"]:
        a = auc(rows, key)
        pairs = [(r[key], conf[r["clip"]]) for r in rows if r["clip"] in conf]
        rho = spearman([x for x, _ in pairs], [y for _, y in pairs])
        om = st.mean([r[key] for r in rows if r["ood"]]); nm = st.mean([r[key] for r in rows if not r["ood"]])
        print(f"  {key:9s} OOD AUC={a:.3f}  vs-conflict ρ={rho:+.3f}  (OOD mean={om:.3f} | non {nm:.3f})")
    print("  (refs: conflict 0.651; >0.65 + positive ρ = a real upgrade)")
    print(">>> PDMS GATE DONE")


if __name__ == "__main__":
    main()
