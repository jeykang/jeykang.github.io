#!/usr/bin/env python3
"""Fetch a slice's raw sensor data from HuggingFace to local disk.

Local counterpart of fetch_slice.sbatch (which was SLURM/Singularity-shaped and is
now unusable — the cluster is gone). Downloads whole per-chunk zips, because that is
how the dataset is packaged, but extracts ONLY the manifest's clips, so the resident
footprint stays small.

    python fetch_slice_local.py --manifest cluster/slice_manifest_nurec.json \
                                --out .av_slice_nurec
"""
from __future__ import annotations

import argparse, io, json, os, shutil, zipfile


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--repo", default="nvidia/PhysicalAI-Autonomous-Vehicles")
    a = ap.parse_args()

    from huggingface_hub import hf_hub_download
    tok = open(os.path.expanduser("~/.cache/huggingface/token")).read().strip()
    man = json.load(open(a.manifest))
    root = os.path.abspath(a.out)
    os.makedirs(root, exist_ok=True)
    cams = man["cameras"]
    print(f"[fetch] {len(man['clips'])} clips / {len(man['chunks'])} chunks / {len(cams)} cameras -> {root}", flush=True)

    def grab(path):
        return hf_hub_download(a.repo, path, repo_type="dataset", token=tok)

    def extract(zpath, dest, keep):
        os.makedirs(dest, exist_ok=True)
        n = 0
        with zipfile.ZipFile(zpath) as z:
            for nm in z.namelist():
                base = os.path.basename(nm)
                if base and any(base.startswith(c) for c in keep):
                    with z.open(nm) as src, open(os.path.join(dest, base), "wb") as dst:
                        shutil.copyfileobj(src, dst)
                    n += 1
        return n

    for ch in man["chunks"]:
        keep = man["clips_by_chunk"][ch]
        for cam in cams:
            p = grab(f"camera/{cam}/{cam}.chunk_{ch}.zip")
            print(f"  chunk {ch} {cam}: {extract(p, f'{root}/camera/{cam}/{cam}.chunk_{ch}', keep)} files", flush=True)
        p = grab(f"labels/egomotion/egomotion.chunk_{ch}.zip")
        print(f"  chunk {ch} egomotion: {extract(p, f'{root}/labels/egomotion/egomotion.chunk_{ch}', keep)} files", flush=True)

        # obstacle labels stay zipped — the adapter opens the chunk zip directly
        src = grab(f"labels/obstacle.offline/obstacle.offline.chunk_{ch}.zip")
        d = f"{root}/labels/obstacle.offline"; os.makedirs(d, exist_ok=True)
        kept = 0
        with zipfile.ZipFile(src) as zin, zipfile.ZipFile(
                f"{d}/obstacle.offline.chunk_{ch}.zip", "w", zipfile.ZIP_DEFLATED) as zout:
            for nm in zin.namelist():
                if any(os.path.basename(nm).startswith(c) for c in keep):
                    zout.writestr(os.path.basename(nm), zin.read(nm)); kept += 1
        print(f"  chunk {ch} obstacle: {kept} records", flush=True)

        try:
            p = grab(f"calibration/vehicle_dimensions/vehicle_dimensions.chunk_{ch}.parquet")
            dd = f"{root}/calibration/vehicle_dimensions"; os.makedirs(dd, exist_ok=True)
            shutil.copyfile(p, f"{dd}/vehicle_dimensions.chunk_{ch}.parquet")
        except Exception as ex:
            print(f"  chunk {ch} vehicle_dimensions unavailable ({str(ex)[:50]})", flush=True)

    with open(os.path.join(os.path.dirname(root), ".av_slice_nurec_clips.txt"), "w") as f:
        f.write("\n".join(sorted(man["clips"])) + "\n")
    print(">>> SLICE READY", flush=True)


if __name__ == "__main__":
    main()
