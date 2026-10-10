#!/usr/bin/env python3
"""stage_pai_clip.py — lay one on-disk clip out the way NVIDIA's PAI->NCore converter expects.

The converter's local mode (`pai-v4 --root-dir`) reads the per-clip directory that
`pai-clip-dl download` produces. Our NFS subset holds the same data chunk-organised
(tools/data_converter/pai/README.md describes both), plus it lacks the four
`.offline` features the converter insists on — offline egomotion and offline camera /
sensor / lidar calibration — which exist on HuggingFace for 298k of the 306k corpus
clips as small per-chunk files. This script:

  1. locates the clip's chunk from its egomotion path,
  2. filters the chunk-level calibration and metadata parquets to the clip
     (`df.loc[[clip_id]]`, exactly what the download tool does),
  3. fetches the chunk's `.offline` calibration parquets and `egomotion.offline`
     zip from HuggingFace (gated main dataset; needs the token) and filters/extracts,
  4. extracts `obstacle.offline` from the NFS chunk zip,
  5. symlinks the large media (camera mp4/timestamps/blurred_boxes, lidar, radar),
  6. writes provenance.json.

Output: <out>/<clip_id>/{calibration,labels,camera,lidar,radar,metadata}/ ready for
  PYTHONPATH=repo .venv/bin/python -m tools.data_converter.pai.converter \
      --root-dir <out> --output-dir <ncore-out> pai-v4 --clip-id <clip_id>
"""
from __future__ import annotations

import argparse
import glob
import io
import json
import os
import sys
import time
import zipfile

import pandas as pd

HF_REPO = "nvidia/PhysicalAI-Autonomous-Vehicles"
CALIB = ["camera_intrinsics", "sensor_extrinsics", "vehicle_dimensions"]
CALIB_OFFLINE = ["camera_intrinsics", "sensor_extrinsics", "lidar_intrinsics"]
META = ["feature_presence", "data_collection"]


def chunk_of(root: str, clip: str) -> str:
    m = glob.glob(f"{root}/labels/egomotion/*/{clip}.egomotion.parquet")
    if not m:
        sys.exit(f"{clip}: no egomotion under {root}")
    return os.path.basename(os.path.dirname(m[0])).split("chunk_")[-1]


def filt(df: pd.DataFrame, clip: str) -> pd.DataFrame:
    """Rows of one clip: MultiIndex level `clip_id`, plain index, or a column."""
    if isinstance(df.index, pd.MultiIndex) and "clip_id" in df.index.names:
        return df.xs(clip, level="clip_id", drop_level=False)
    if df.index.name == "clip_id":
        return df.loc[[clip]] if clip in df.index else df.iloc[0:0]
    if "clip_id" in df.columns:
        return df[df["clip_id"] == clip]
    return df.iloc[0:0]


def link(src: str, dst: str, mode: str):
    if os.path.lexists(dst):
        os.remove(dst)
    if mode == "symlink":
        os.symlink(src, dst)
    elif mode == "hardlink":
        os.link(src, dst)
    else:
        import shutil
        shutil.copy2(src, dst)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=os.environ.get("AV_ROOT", "/mnt/netai-e2e/nvidia-physicalai-av-subset"))
    ap.add_argument("--clip", required=True)
    ap.add_argument("--out", required=True, help="staging root; the clip goes in <out>/<clip>/")
    ap.add_argument("--link", choices=["symlink", "hardlink", "copy"], default="symlink")
    ap.add_argument("--no-offline", action="store_true", help="skip the HuggingFace offline features")
    ap.add_argument("--allow-empty", action="store_true",
                    help="stage anyway when a camera has 0-byte source files (exclude it at conversion)")
    ap.add_argument("--hf-cache", default=None, help="where chunk-level HF files are cached (default <out>/.hf)")
    a = ap.parse_args()

    root, clip = a.root, a.clip
    ch = chunk_of(root, clip)
    d = os.path.join(a.out, clip)
    for sub in ("calibration", "labels", "camera", "lidar", "radar", "metadata"):
        os.makedirs(os.path.join(d, sub), exist_ok=True)
    report = {"clip_id": clip, "chunk": ch, "present": [], "missing": []}

    # 2. calibration + metadata, filtered to the clip
    for name in CALIB:
        p = f"{root}/calibration/{name}/{name}.chunk_{ch}.parquet"
        if os.path.exists(p):
            filt(pd.read_parquet(p), clip).to_parquet(f"{d}/calibration/{name}.parquet")
            report["present"].append(name)
        else:
            report["missing"].append(name)
    for name in META:
        p = f"{root}/metadata/{name}.parquet"
        filt(pd.read_parquet(p), clip).to_parquet(f"{d}/metadata/{name}.parquet")
        report["present"].append(name)

    # 3. offline features from HuggingFace
    if not a.no_offline:
        from huggingface_hub import hf_hub_download
        tok = os.environ.get("HF_TOKEN") or open(os.path.expanduser("~/.cache/huggingface/token")).read().strip()
        cache = a.hf_cache or os.path.join(a.out, ".hf")
        for name in CALIB_OFFLINE:
            rp = f"calibration/{name}.offline/{name}.offline.chunk_{ch}.parquet"
            try:
                p = hf_hub_download(HF_REPO, rp, repo_type="dataset", token=tok, local_dir=cache)
                filt(pd.read_parquet(p), clip).to_parquet(f"{d}/calibration/{name}.offline.parquet")
                report["present"].append(f"{name}.offline")
            except Exception as e:
                report["missing"].append(f"{name}.offline ({str(e)[:60]})")
        rp = f"labels/egomotion.offline/egomotion.offline.chunk_{ch}.zip"
        try:
            p = hf_hub_download(HF_REPO, rp, repo_type="dataset", token=tok, local_dir=cache)
            zf = zipfile.ZipFile(p)
            nm = [n for n in zf.namelist() if n.endswith(f"{clip}.egomotion.offline.parquet")]
            if nm:
                open(f"{d}/labels/{clip}.egomotion.offline.parquet", "wb").write(zf.read(nm[0]))
                report["present"].append("egomotion.offline")
            else:
                report["missing"].append("egomotion.offline (not in chunk zip)")
        except Exception as e:
            report["missing"].append(f"egomotion.offline ({str(e)[:60]})")

    # 4. labels from NFS
    link(glob.glob(f"{root}/labels/egomotion/*/{clip}.egomotion.parquet")[0],
         f"{d}/labels/{clip}.egomotion.parquet", a.link)
    report["present"].append("egomotion")
    zp = f"{root}/labels/obstacle.offline/obstacle.offline.chunk_{ch}.zip"
    if os.path.exists(zp):
        zf = zipfile.ZipFile(zp)
        nm = [n for n in zf.namelist() if n.endswith(f"{clip}.obstacle.offline.parquet")]
        if nm:
            open(f"{d}/labels/{clip}.obstacle.offline.parquet", "wb").write(zf.read(nm[0]))
            report["present"].append("obstacle.offline")
        else:
            report["missing"].append("obstacle.offline (not in chunk zip)")

    # 5. media
    # a 0-byte file is a data-loss artifact (see the April extraction bug), not a
    # sensor absence; the converter crashes on it, so it fails the clip here
    for cam in sorted(os.listdir(f"{root}/camera")):
        base = f"{root}/camera/{cam}/{cam}.chunk_{ch}/{clip}.{cam}"
        got, empty = 0, []
        for suffix in ("mp4", "timestamps.parquet", "blurred_boxes.parquet"):
            src = f"{base}.{suffix}"
            if os.path.exists(src):
                if os.path.getsize(src) == 0:
                    empty.append(suffix); continue
                link(src, f"{d}/camera/{clip}.{cam}.{suffix}", a.link); got += 1
        if empty:
            report["missing"].append(f"{cam} (EMPTY source: {', '.join(empty)})")
        else:
            (report["present"] if got == 3 else report["missing"]).append(f"{cam} ({got}/3 files)")
    # a camera whose copy is empty here is still marked present in the clip's
    # feature_presence row, and the converter decodes every camera that row marks
    # present -> KeyError on the missing timestamps. With --allow-empty, mark such
    # cameras absent in the staged row so the converter builds the store from the
    # cameras we have (the six-camera set NVIDIA's own NCore release uses lacks
    # camera_rear_tele_30fov anyway).
    empty_cams = [m.split(" ")[0] for m in report["missing"] if "EMPTY source" in m]
    if empty_cams and a.allow_empty:
        fp = f"{d}/metadata/feature_presence.parquet"
        if os.path.exists(fp):
            df = pd.read_parquet(fp)
            for cam in empty_cams:
                if cam in df.columns:
                    df[cam] = False
            df.to_parquet(fp)
            report["feature_presence_overrides"] = {cam: False for cam in empty_cams}
    lp = f"{root}/lidar/lidar_top_360fov/lidar_top_360fov.chunk_{ch}/{clip}.lidar_top_360fov.parquet"
    if os.path.exists(lp):
        link(lp, f"{d}/lidar/{clip}.lidar_top_360fov.parquet", a.link); report["present"].append("lidar_top_360fov")
    else:
        report["missing"].append("lidar_top_360fov")
    n_radar = 0
    for rp_ in glob.glob(f"{root}/radar/*/*.chunk_{ch}/{clip}.*.parquet"):
        link(rp_, f"{d}/radar/{os.path.basename(rp_)}", a.link); n_radar += 1
    report["present"].append(f"radar x{n_radar}")

    # 6. provenance — the converter copies repo_id / revision / commit_sha into the
    #    sequence meta (source_repo_id, source_revision, source_commit_sha), so name
    #    them the way pai-clip-dl does; the rest is our own record
    report["staged_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    report["source"] = {"media_and_labels": root, "offline_features": HF_REPO, "layout": "pai-clip-dl"}
    report["repo_id"] = HF_REPO
    report["revision"] = "main"
    if not a.no_offline:
        try:
            from huggingface_hub import HfApi
            report["commit_sha"] = HfApi(token=tok).dataset_info(HF_REPO).sha
        except Exception:
            report["commit_sha"] = None
    json.dump(report, open(f"{d}/metadata/provenance.json", "w"), indent=1)
    print(f"[stage] {clip} (chunk {ch}) -> {d}")
    print("  present:", ", ".join(report["present"]))
    if report["missing"]:
        print("  MISSING:", ", ".join(report["missing"]))
    required = {"egomotion.offline", "camera_intrinsics.offline", "sensor_extrinsics.offline", "lidar_intrinsics.offline"}
    if not a.no_offline and not required <= set(report["present"]):
        sys.exit(2)
    if any("EMPTY source" in m for m in report["missing"]) and not a.allow_empty:
        sys.exit(3)


if __name__ == "__main__":
    main()
