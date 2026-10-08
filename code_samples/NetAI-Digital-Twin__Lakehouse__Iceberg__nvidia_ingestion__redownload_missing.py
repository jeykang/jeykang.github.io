"""Redownload + re-extract lidar/radar zips lost to the extract_remaining.sh bug.

Streams through transfer_manifest.json: for each zip not yet present on NFS as
extracted files, download from HuggingFace to local /tmp staging, extract,
move extracted files to NFS at the manifest's dst path, delete the staged zip.

Safety:
  * Never deletes a zip until `mv` succeeds AND file count matches
  * Skips entries that already have >1 KB files on NFS
  * Honours --max-gb ceiling (raw bytes downloaded this run)
  * Respects --modalities filter so radar-first runs don't touch lidar
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

REPO_ID = "nvidia/PhysicalAI-Autonomous-Vehicles"
MANIFEST = Path("/home/netai/jeykang/NetAI-Digital-Twin/Lakehouse/Iceberg/"
                "netai-e2e/nvidia-physicalai-av-subset/transfer_manifest.json")
SUBSET_ROOT = MANIFEST.parent
STAGING = Path("/tmp/hf_redownload")


def has_real_data(extract_dir: Path) -> bool:
    if not extract_dir.is_dir():
        return False
    try:
        for f in extract_dir.iterdir():
            if f.is_file() and f.stat().st_size > 1024:
                return True
    except OSError:
        return False
    return False


def hf_relpath(entry: dict) -> str:
    """Path inside the HF repo corresponds to zip location on NFS after
    /mnt/netai-e2e/nvidia-physicalai-av-subset/ prefix."""
    dst = Path(entry["dst"])
    parts = dst.parts
    idx = parts.index("nvidia-physicalai-av-subset")
    return "/".join(parts[idx+1:])


def nfs_dst(entry: dict) -> Path:
    """Translate manifest container path /mnt/netai-e2e/... to host SUBSET_ROOT."""
    return SUBSET_ROOT / hf_relpath(entry)


def process_one(entry: dict) -> tuple[str, str, int]:
    """Download → extract → move. Returns (status, zip_path, bytes).

    Most entries are zip files containing parquets; some (calibration) are
    bare parquets that ship as-is. Detected by the manifest entry's `dst`
    suffix: `.zip` → extract; `.parquet` → direct copy.
    """
    rel = hf_relpath(entry)
    dst_zip_on_nfs = nfs_dst(entry)
    is_zip = dst_zip_on_nfs.suffix == ".zip"

    if is_zip:
        extract_dir = dst_zip_on_nfs.with_suffix("")
    else:
        # Bare-file entry: target is the file path itself, "ready" check looks at file
        extract_dir = dst_zip_on_nfs

    if is_zip and has_real_data(extract_dir):
        return ("SKIP", rel, 0)
    if not is_zip and extract_dir.is_file() and extract_dir.stat().st_size > 1024:
        return ("SKIP", rel, 0)

    local_zip = STAGING / "zips" / rel
    local_extract = STAGING / "ex" / extract_dir.name
    local_zip.parent.mkdir(parents=True, exist_ok=True)
    if local_extract.exists():
        shutil.rmtree(local_extract, ignore_errors=True)

    # Download into a throwaway dir, let hf resolve to blob cache, then hardlink/move
    t0 = time.time()
    proc = subprocess.run(
        ["hf", "download", REPO_ID, rel,
         "--repo-type", "dataset",
         "--local-dir", str(STAGING / "dl" / entry["sensor"])],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        return ("DL_FAIL", rel, 0)
    downloaded = STAGING / "dl" / entry["sensor"] / rel
    if not downloaded.exists():
        return ("DL_MISSING", rel, 0)
    dl_bytes = downloaded.stat().st_size

    # Bare-file path: just place the file at the destination, no extraction.
    if not is_zip:
        extract_dir.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.move(str(downloaded), str(extract_dir))
        except OSError as e:
            return (f"MV_FAIL:{e}", rel, dl_bytes)
        if not extract_dir.is_file() or extract_dir.stat().st_size != dl_bytes:
            return (f"VERIFY_FAIL:size mismatch", rel, dl_bytes)
        try:
            downloaded.parent.rmdir()
        except OSError:
            pass
        return ("OK", rel, dl_bytes)

    # Zip path: extract locally
    local_extract.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(downloaded) as z:
            z.extractall(local_extract)
    except Exception as e:
        return (f"UNZIP_FAIL:{e}", rel, dl_bytes)

    nfiles_staged = sum(1 for _ in local_extract.rglob("*") if _.is_file())

    # Move to NFS. Parent dir should exist (0-byte files there now — remove them first).
    extract_dir.parent.mkdir(parents=True, exist_ok=True)
    if extract_dir.exists():
        shutil.rmtree(extract_dir, ignore_errors=True)
    try:
        shutil.move(str(local_extract), str(extract_dir))
    except OSError as e:
        # Staging retained for retry
        return (f"MV_FAIL:{e}", rel, dl_bytes)

    # Verify
    nfiles_moved = sum(1 for _ in extract_dir.rglob("*") if _.is_file())
    if nfiles_moved < nfiles_staged:
        return (f"VERIFY_FAIL:{nfiles_moved}<{nfiles_staged}", rel, dl_bytes)

    # Cleanup downloaded zip
    downloaded.unlink(missing_ok=True)
    # Best-effort: prune empty parents under STAGING/dl
    try:
        downloaded.parent.rmdir()
    except OSError:
        pass
    return ("OK", rel, dl_bytes)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--modalities", nargs="+", default=["radar", "lidar"],
                    choices=["radar", "lidar", "camera", "labels", "calibration"])
    ap.add_argument("--max-gb", type=float, default=0,
                    help="Stop after this many GB downloaded this run (0 = no limit).")
    ap.add_argument("--limit", type=int, default=0,
                    help="Stop after this many entries processed (0 = no limit).")
    ap.add_argument("--workers", type=int, default=3,
                    help="Parallel download+extract workers.")
    ap.add_argument("--sensors", nargs="*", default=[],
                    help="Restrict to these sensor names.")
    args = ap.parse_args()

    STAGING.mkdir(parents=True, exist_ok=True)
    (STAGING / "zips").mkdir(exist_ok=True)
    (STAGING / "ex").mkdir(exist_ok=True)
    (STAGING / "dl").mkdir(exist_ok=True)

    entries = json.loads(MANIFEST.read_text())
    todo = [e for e in entries
            if e["modality"] in args.modalities
            and (not args.sensors or e["sensor"] in args.sensors)]
    # Filter to missing
    todo_missing = []
    for e in todo:
        ext = nfs_dst(e).with_suffix("")
        if not has_real_data(ext):
            todo_missing.append(e)
    # Smallest-first so quick validation wins
    todo_missing.sort(key=lambda e: e["size"])

    if args.max_gb:
        budget = int(args.max_gb * 1e9)
    else:
        budget = 0

    total_bytes = sum(e["size"] for e in todo_missing)
    print(f"[plan] {len(todo_missing):,} zips missing across "
          f"{sorted(set(e['modality'] for e in todo_missing))}"
          f"  ({total_bytes/1e9:.1f} GB raw)", flush=True)
    if budget:
        print(f"[plan] budget: {args.max_gb} GB", flush=True)
    if args.limit:
        todo_missing = todo_missing[:args.limit]
        print(f"[plan] limited to first {len(todo_missing)} entries", flush=True)

    done_ok = done_skip = done_fail = 0
    bytes_downloaded = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(process_one, e): e for e in todo_missing}
        for fut in as_completed(futs):
            e = futs[fut]
            try:
                status, rel, n = fut.result()
            except Exception as err:
                status, rel, n = (f"EXC:{err}", "?", 0)
            if status == "OK":
                done_ok += 1; bytes_downloaded += n
            elif status == "SKIP":
                done_skip += 1
            else:
                done_fail += 1
            elapsed = time.time() - t0
            rate = bytes_downloaded / elapsed if elapsed > 0 else 0
            done = done_ok + done_skip + done_fail
            print(f"[{done}/{len(todo_missing)}] {status:14s} "
                  f"{rel}  (dl={bytes_downloaded/1e9:.1f}GB, "
                  f"{rate/1e6:.1f} MB/s)", flush=True)
            if budget and bytes_downloaded >= budget:
                print(f"[BUDGET HIT] stopping, {bytes_downloaded/1e9:.1f} GB downloaded",
                      flush=True)
                for f in futs:
                    f.cancel()
                break

    print(f"\n[summary] ok={done_ok}  skipped={done_skip}  failed={done_fail}"
          f"  downloaded={bytes_downloaded/1e9:.2f} GB"
          f"  elapsed={(time.time()-t0)/60:.1f} min", flush=True)
    return 0 if done_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
