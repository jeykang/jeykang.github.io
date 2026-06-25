#!/usr/bin/env python3
"""Agent-conflict difficulty sub-score (production) — GPU-free, obstacle.offline.

Validated facet (VALIDITY_REPORT-style battery: scene-driven; OOD AUC ~0.65,
the best of the conflict variants = simple forward-zone proximity). Computes,
per clip, the forward-zone (x in [0,40] m, |y| < 8 m, rig frame) inverse-distance
agent load, averaged over 3 decision times (obstacle.offline is per-track time
series → reconstruct each time by each track's nearest detection within 0.1 s),
then rank-normalizes to [0,1].

Writes <NFS>/.conflict/conflict_shard_00_of_01.parquet
  (clip_id, conflict_score in [0,1], conflict_load), read by
  edge_case_scorer._load_conflict_scores. Detachable (same contract as
  perception/planning): removal = delete this + the loader hook + the weight.
"""
import glob, io, math, os, statistics as st, time, zipfile
import pyarrow as pa, pyarrow.parquet as pq

_C = "/mnt/netai-e2e/nvidia-physicalai-av-subset"
ROOT = _C if os.path.isdir(_C) else os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "netai-e2e", "nvidia-physicalai-av-subset")
OO = f"{ROOT}/labels/obstacle.offline"
OUT = f"{ROOT}/.conflict"
FRACS = [0.3, 0.5, 0.7]
WIN = 100_000  # 0.1 s


def clip_load(d):
    """simple forward-zone inverse-distance agent load, averaged over decision times."""
    n = len(d["timestamp_us"])
    if n == 0:
        return 0.0
    tracks = {}
    for i in range(n):
        tracks.setdefault(d["track_id"][i], []).append(i)
    ts = d["timestamp_us"]; tmin, tmax = min(ts), max(ts)
    loads = []
    for fr in FRACS:
        T = tmin + fr * (tmax - tmin)
        load = 0.0
        for ids in tracks.values():
            j = min(ids, key=lambda i: abs(ts[i] - T))
            if abs(ts[j] - T) > WIN:
                continue
            x, y = d["center_x"][j], d["center_y"][j]
            if 0 < x < 40 and abs(y) < 8:
                load += 1.0 / (1.0 + math.hypot(x, y))
        loads.append(load)
    return st.mean(loads)


def main():
    zips = sorted(glob.glob(f"{OO}/obstacle.offline.chunk_*.zip"))
    print(f"[conflict] {len(zips)} chunks", flush=True)
    rows = []
    t0 = time.time()
    for k, zp in enumerate(zips):
        try:
            zf = zipfile.ZipFile(zp)
        except Exception as e:
            print(f"  [WARN] {os.path.basename(zp)}: {e}", flush=True); continue
        for nm in zf.namelist():
            if not nm.endswith(".parquet"):
                continue
            clip = os.path.basename(nm).split(".", 1)[0]
            try:
                d = pq.read_table(io.BytesIO(zf.read(nm)),
                                  columns=["timestamp_us", "track_id", "center_x", "center_y"]).to_pydict()
                rows.append({"clip_id": clip, "conflict_load": clip_load(d)})
            except Exception as e:
                print(f"  [WARN] {clip[:8]}: {e}", flush=True)
        if (k + 1) % 25 == 0:
            print(f"[conflict] {k+1}/{len(zips)} chunks, {len(rows)} clips ({(k+1)/(time.time()-t0):.1f} ch/s)", flush=True)

    # rank-normalize load -> conflict_score in [0,1]
    order = sorted(range(len(rows)), key=lambda i: rows[i]["conflict_load"])
    for rank, i in enumerate(order):
        rows[i]["conflict_score"] = rank / max(1, len(rows) - 1)
        rows[i]["scored_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")

    os.makedirs(OUT, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), f"{OUT}/conflict_shard_00_of_01.parquet")
    ld = [r["conflict_load"] for r in rows]
    print(f"[conflict] wrote {len(rows)} clips; load mean={st.mean(ld):.3f} "
          f"max={max(ld):.2f} zero-frac={sum(1 for x in ld if x==0)/len(ld):.2f}", flush=True)
    print(">>> CONFLICT RUNNER DONE", flush=True)


if __name__ == "__main__":
    main()
