#!/usr/bin/env python3
"""Stage an evaluation slice to the cluster.

Builds a MINIMAL mirror of the dataset layout containing only what the evaluation
harness reads, so `AV_ROOT` on the cluster points at it and `adapters.NvidiaAdapter`
works unmodified:

    camera/<cam>/<cam>.chunk_XXXX/<clip>.<cam>.mp4
    camera/<cam>/<cam>.chunk_XXXX/<clip>.<cam>.timestamps.parquet
    labels/egomotion/egomotion.chunk_XXXX/<clip>.egomotion.parquet
    labels/obstacle.offline/obstacle.offline.chunk_XXXX.zip   <- REBUILT, see below
    calibration/vehicle_dimensions/vehicle_dimensions.chunk_XXXX.parquet

The obstacle labels ship as one zip per chunk holding every clip in that chunk, so
copying them wholesale would move far more than the slice needs. Each chunk zip is
rebuilt here containing only the selected clips' parquets — same filename, same
internal names, so nothing downstream changes.

    python stage_slice.py --limit 60 --cameras 4 --upload
    python stage_slice.py --clips-from-parquet <p> --rank-col conflict_score --limit 200

Without --upload it only builds the tarball, which is the safe way to check size first.
"""
from __future__ import annotations

import argparse
import io
import os
import sys
import tarfile
import zipfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import adapters  # noqa: E402

CAM4 = ["camera_cross_left_120fov", "camera_front_wide_120fov",
        "camera_cross_right_120fov", "camera_front_tele_30fov"]
CAM7 = CAM4 + ["camera_rear_left_70fov", "camera_rear_right_70fov",
               "camera_rear_tele_30fov"]


def pick_clips(a, ad):
    if a.clips_file:
        ids = [l.strip().split(",")[0] for l in open(a.clips_file) if l.strip()]
    elif a.clips_from_parquet:
        import pyarrow.parquet as pq
        cols = ["clip_id"] + ([a.rank_col] if a.rank_col else [])
        d = pq.read_table(a.clips_from_parquet, columns=cols).to_pydict()
        ids = list(d["clip_id"])
        if a.rank_col:
            ids = [c for _, c in sorted(zip(d[a.rank_col], ids), reverse=True)]
    else:
        ids = list(ad.list_clips())
    return ids[:a.limit] if a.limit else ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=60)
    ap.add_argument("--cameras", type=int, choices=(4, 7), default=4,
                    help="4 = Alpamayo 1.5/R1; 7 = Alpamayo2-Super's full rig")
    ap.add_argument("--clips-file")
    ap.add_argument("--clips-from-parquet")
    ap.add_argument("--rank-col")
    ap.add_argument("--out", default="/tmp/av_slice.tar")
    ap.add_argument("--upload", action="store_true")
    ap.add_argument("--remote-dir", default=None)
    a = ap.parse_args()

    ad = adapters.get_adapter("nvidia")
    cams = CAM4 if a.cameras == 4 else CAM7
    clips = pick_clips(a, ad)
    print(f"[stage] {len(clips)} clips x {len(cams)} cameras")

    by_chunk: dict[str, list[str]] = {}
    for c in clips:
        ch = ad._chunk(c)
        if ch:
            by_chunk.setdefault(ch, []).append(c)
    print(f"[stage] spans {len(by_chunk)} chunks")

    total = 0
    with tarfile.open(a.out, "w") as tar:
        def add(path, arc):
            nonlocal total
            if os.path.exists(path):
                tar.add(path, arcname=arc)
                total += os.path.getsize(path)

        for ch, cids in sorted(by_chunk.items()):
            for cid in cids:
                for cam in cams:
                    base = f"camera/{cam}/{cam}.chunk_{ch}/{cid}.{cam}"
                    add(f"{ad.root}/{base}.mp4", f"{base}.mp4")
                    add(f"{ad.root}/{base}.timestamps.parquet", f"{base}.timestamps.parquet")
                eg = f"labels/egomotion/egomotion.chunk_{ch}/{cid}.egomotion.parquet"
                add(f"{ad.root}/{eg}", eg)
            vd = f"calibration/vehicle_dimensions/vehicle_dimensions.chunk_{ch}.parquet"
            add(f"{ad.root}/{vd}", vd)

            # rebuild a per-chunk obstacle zip holding only this slice's clips
            src = f"{ad.root}/labels/obstacle.offline/obstacle.offline.chunk_{ch}.zip"
            if os.path.exists(src):
                buf = io.BytesIO()
                kept = 0
                with zipfile.ZipFile(src) as zin, \
                     zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zout:
                    names = set(zin.namelist())
                    for cid in cids:
                        nm = f"{cid}.obstacle.offline.parquet"
                        if nm in names:
                            zout.writestr(nm, zin.read(nm)); kept += 1
                data = buf.getvalue()
                ti = tarfile.TarInfo(
                    f"labels/obstacle.offline/obstacle.offline.chunk_{ch}.zip")
                ti.size = len(data)
                tar.addfile(ti, io.BytesIO(data))
                total += len(data)
                print(f"  chunk {ch}: {len(cids)} clips, {kept} obstacle records", flush=True)

    sz = os.path.getsize(a.out)
    print(f"[stage] {a.out}  {sz/1e9:.2f} GB (raw inputs {total/1e9:.2f} GB)")

    if not a.upload:
        print("[stage] not uploading (--upload to send)")
        return
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "..", "..", "cosmos_augmentation"))
    import cluster
    env = cluster.load_env()
    remote_dir = a.remote_dir or f"{env['CLUSTER_HOME']}/alpamayo_eval"
    remote = f"{remote_dir}/av_slice.tar"
    print(f"[stage] uploading -> {remote}")
    cluster.run(f"mkdir -p {remote_dir}/av_root")
    cluster.put(a.out, remote)
    o, e = cluster.run(f"cd {remote_dir}/av_root && tar xf {remote} && "
                       f"du -sh . && find . -name '*.mp4' | wc -l", timeout=900)
    print(o or e)
    print(">>> STAGED")


if __name__ == "__main__":
    main()
