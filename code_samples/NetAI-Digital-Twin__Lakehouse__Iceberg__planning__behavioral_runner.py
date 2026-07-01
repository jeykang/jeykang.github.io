#!/usr/bin/env python3
"""Best-effort multi-axis BEHAVIORAL difficulty scorer (GPU-free).

Generalizes conflict_runner.py: instead of one agent-proximity axis, it computes a
set of agent-interaction axes and combines the ones the input dataset can actually
support. Each axis declares the box fields it needs ({pos, class, track}); axes
whose inputs are absent are silently dropped and the remaining weights renormalize
— so the scorer runs on any dataset's agent labels, using all available axes.

Axes (from the 2026-06-27 battery; combined CV-AUC 0.745 vs single conflict 0.65):
  conflict (pos)     forward-zone inverse-distance load   [validated core]
  multidir (pos)     # quadrants occupied (boxed-in)
  closing  (pos,track) # forward agents approaching
  vru      (pos,class) vulnerable-road-user count near
  cls_div  (class)   agent-class entropy (scene heterogeneity)
  rare     (class)   rare/unusual agent class present
Perception axes (low_conf, det_count) fold in via .perception when present.

Per-axis values are rank-normalized over the scored population, then combined as a
weighted mean over the axes present for each clip. Output: <NFS>/.behavioral/
behavioral_shard.parquet (clip_id, behavioral_score, + per-axis ranks). Detachable;
consumed by edge_case_scorer as the behavioral leg of the difficulty union.

Porting to another dataset = swap `read_boxes()` for that dataset's agent-label
reader (return whatever fields it has); everything else adapts automatically.
"""
import glob, io, math, os, statistics as st, time, zipfile
import pyarrow as pa, pyarrow.parquet as pq

_C = "/mnt/netai-e2e/nvidia-physicalai-av-subset"
ROOT = _C if os.path.isdir(_C) else os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "netai-e2e", "nvidia-physicalai-av-subset")
OO = f"{ROOT}/labels/obstacle.offline"
OUT = f"{ROOT}/.behavioral"
FRACS = [0.3, 0.5, 0.7]; WIN = 100_000
# class -> category mapping (dataset-configurable; unmapped classes are ignored by
# the semantic axes but still counted by geometry axes)
VRU = {"person", "rider", "stroller", "animal"}
RARE = {"animal", "other_vehicle", "train_or_tram_car", "protruding_object", "stroller"}

# axis -> (weight, required field-categories)
AXES = {
    "conflict": (0.30, {"pos"}),
    "multidir": (0.10, {"pos"}),
    "closing":  (0.10, {"pos", "track"}),
    "vru":      (0.20, {"pos", "class"}),
    "cls_div":  (0.20, {"class"}),
    "rare":     (0.10, {"class"}),
}


def read_boxes(clip):
    """NVIDIA obstacle.offline adapter. Returns dict of available fields, or None.
    Other datasets: replace this fn; provide whatever of x/y/cls/track/t you have."""
    lid = glob.glob(f"{ROOT}/lidar/lidar_top_360fov/*/{clip}.lidar_top_360fov.parquet")
    if not lid:
        return None
    ch = lid[0].split("chunk_")[1][:4]
    zp = f"{OO}/obstacle.offline.chunk_{ch}.zip"
    if not os.path.exists(zp):
        return None
    zf = zipfile.ZipFile(zp); nm = f"{clip}.obstacle.offline.parquet"
    if nm not in zf.namelist():
        return None
    d = pq.read_table(io.BytesIO(zf.read(nm))).to_pydict()
    if not d.get("timestamp_us"):
        return None
    return {"x": d["center_x"], "y": d["center_y"], "cls": d.get("label_class"),
            "track": d.get("track_id"), "t": d.get("timestamp_us"),
            "sz": [max(a, b) for a, b in zip(d["size_x"], d["size_y"])]}


def available_fields(b):
    f = set()
    if b.get("x") is not None and b.get("y") is not None:
        f.add("pos")
    if b.get("cls") is not None:
        f.add("class")
    if b.get("track") is not None and b.get("t") is not None:
        f.add("track")
    return f


def _reconstruct(b):
    """Per decision time, the set of agents present (nearest track sample within WIN;
    if no track/time, treat all boxes as one snapshot)."""
    n = len(b["x"])
    if "track" in available_fields(b):
        tr = {}
        for i in range(n):
            tr.setdefault(b["track"][i], []).append(i)
        for t in tr:
            tr[t].sort(key=lambda i: b["t"][i])
        ts = b["t"]; tmin, tmax = min(ts), max(ts)
        for fr in FRACS:
            T = tmin + fr * (tmax - tmin); present = []
            for ids in tr.values():
                j = min(ids, key=lambda i: abs(ts[i] - T))
                if abs(ts[j] - T) <= WIN:
                    nxt = ids[ids.index(j) + 1] if ids.index(j) + 1 < len(ids) else None
                    present.append((j, nxt))
            yield present
    else:
        yield [(i, None) for i in range(n)]   # single snapshot


def compute_axes(b, active):
    """Raw per-axis values for one clip, only for `active` axes."""
    vals = {a: [] for a in active}
    ts = b.get("t")
    for present in _reconstruct(b):
        fwd = near = vru = close = 0; quad = set(); classes = {}
        for j, nxt in present:
            x, y = b["x"][j], b["y"][j]; dist = math.hypot(x, y)
            cls = b["cls"][j] if b.get("cls") else None
            if dist < 30:
                near += 1; quad.add((x > 0, y > 0))
                if cls is not None:
                    classes[cls] = classes.get(cls, 0) + 1
                    if cls in VRU:
                        vru += 1
            if 0 < x < 40 and abs(y) < 8:
                if nxt is not None and "closing" in active:
                    dt = max(1e-3, (ts[nxt] - ts[j]) / 1e6)
                    if (b["x"][nxt] - x) / dt < -0.5:
                        close += 1
        # conflict = forward inverse-distance load
        if "conflict" in active:
            load = sum(1.0 / (1.0 + math.hypot(b["x"][j], b["y"][j]))
                       for j, _ in present if 0 < b["x"][j] < 40 and abs(b["y"][j]) < 8)
            vals["conflict"].append(load)
        if "multidir" in active:
            vals["multidir"].append(len(quad))
        if "closing" in active:
            vals["closing"].append(close)
        if "vru" in active:
            vals["vru"].append(vru)
        if "cls_div" in active and classes:
            tot = sum(classes.values())
            vals["cls_div"].append(-sum((c / tot) * math.log(c / tot) for c in classes.values()))
        elif "cls_div" in active:
            vals["cls_div"].append(0.0)
        if "rare" in active:
            vals["rare"].append(1.0 if any(
                (b["cls"][j] in RARE) for j, _ in present if b.get("cls")) else 0.0)
    return {a: (st.mean(v) if v else 0.0) for a, v in vals.items()}


def _rank_norm(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i]); r = [0.0] * len(xs)
    for rank, i in enumerate(order):
        r[i] = rank / max(1, len(xs) - 1)
    return r


def _box_from_parquet(d):
    return {"x": d["center_x"], "y": d["center_y"], "cls": d.get("label_class"),
            "track": d.get("track_id"), "t": d.get("timestamp_us")}


def main():
    zips = sorted(glob.glob(f"{OO}/obstacle.offline.chunk_*.zip"))
    print(f"[behavioral] {len(zips)} chunks", flush=True)
    # determine active axes from the first readable clip's schema (one read)
    probe = None
    for zp in zips:
        zf = zipfile.ZipFile(zp)
        for nm in zf.namelist():
            if nm.endswith(".parquet"):
                probe = _box_from_parquet(pq.read_table(io.BytesIO(zf.read(nm))).to_pydict()); break
        if probe:
            break
    avail = available_fields(probe)
    active = [a for a, (w, need) in AXES.items() if need <= avail]
    dropped = [a for a in AXES if a not in active]
    print(f"[behavioral] available fields: {sorted(avail)}", flush=True)
    print(f"[behavioral] ACTIVE axes: {active}", flush=True)
    print(f"[behavioral] DROPPED (missing inputs): {dropped}", flush=True)
    if not active:
        print("[behavioral] no axes computable for this dataset — nothing to score."); return

    rows = []; t0 = time.time(); n = 0
    for zp in zips:                              # open each chunk zip ONCE
        zf = zipfile.ZipFile(zp)
        for nm in zf.namelist():
            if not nm.endswith(".parquet"):
                continue
            cid = os.path.basename(nm).split(".", 1)[0]
            try:
                b = _box_from_parquet(pq.read_table(io.BytesIO(zf.read(nm))).to_pydict())
                if not b["x"]:
                    continue
                rows.append({"clip_id": cid, **compute_axes(b, active)})
            except Exception as e:
                print(f"  [WARN] {cid[:8]}: {str(e)[:60]}", flush=True)
            n += 1
            if n % 1000 == 0:
                print(f"[behavioral] {n} clips ({n/(time.time()-t0):.0f} c/s)", flush=True)

    # rank-normalize each active axis over the population
    ranks = {a: _rank_norm([r[a] for r in rows]) for a in active}
    tw = sum(AXES[a][0] for a in active)   # renormalize weights over present axes
    for i, r in enumerate(rows):
        r["behavioral_score"] = sum(AXES[a][0] * ranks[a][i] for a in active) / tw
        for a in active:
            r[a + "_rank"] = round(ranks[a][i], 4)
        r["scored_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        r["active_axes"] = ",".join(active)

    os.makedirs(OUT, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), f"{OUT}/behavioral_shard_00_of_01.parquet")
    bs = [r["behavioral_score"] for r in rows]
    print(f"[behavioral] wrote {len(rows)} clips; score mean={st.mean(bs):.3f} "
          f"axes={active} (weights renormalized over present)", flush=True)

    # optional self-validation vs OOD labels + conflict baseline
    if os.path.exists("/tmp/conf/clips.txt"):
        lab = {c: int(o) for c, o in (l.strip().split(",") for l in open("/tmp/conf/clips.txt") if l.strip())}
        ev = [(r["behavioral_score"], lab[r["clip_id"]]) for r in rows if r["clip_id"] in lab]
        if ev:
            pos = [v for v, o in ev if o]; neg = [v for v, o in ev if not o]
            al = sorted(ev, key=lambda p: p[0]); rs = sum(i + 1 for i, p in enumerate(al) if p[1])
            auc = (rs - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))
            print(f"[behavioral] self-check OOD AUC = {auc:.3f} (N={len(ev)}; conflict baseline ~0.65)", flush=True)
    print(">>> BEHAVIORAL RUNNER DONE", flush=True)


if __name__ == "__main__":
    main()
