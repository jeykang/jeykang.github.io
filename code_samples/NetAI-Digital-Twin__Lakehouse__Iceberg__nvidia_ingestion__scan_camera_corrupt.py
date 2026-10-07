#!/usr/bin/env python3
"""Full-census camera MP4 integrity scan.

Complements `quality_checks.check_camera_integrity`, which only samples 100 mp4s
per camera. This walks EVERY `<clip_id>.<sensor>.mp4` under the recovered NFS
camera tree and flags a file as corrupt when it is missing, zero/too-small, or
lacks an `ftyp`/`moov` box in the first 12 bytes -- the same rule the Spark
check uses, applied to the whole dataset.

Output: a CSV of (clip_id, sensor, reason) per bad file plus a deduped
clip-level list, written next to this script under bad_clip_lists/.

Usage:
    python3 scan_camera_corrupt.py [CAMERA_ROOT] [--workers N]
"""
import csv
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

DEFAULT_ROOT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "netai-e2e", "nvidia-physicalai-av-subset", "camera",
)
OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bad_clip_lists")


def check(path):
    """Return reason string if the mp4 is bad, else None."""
    try:
        sz = os.path.getsize(path)
    except OSError as e:
        return f"stat failed: {e}"
    if sz < 12:
        return f"Too small: {sz} bytes"
    try:
        with open(path, "rb") as f:
            head = f.read(12)
    except OSError as e:
        return f"open failed: {e}"
    if len(head) < 12 or not (b"ftyp" in head or b"moov" in head):
        return "invalid mp4 box header"
    return None


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    root = args[0] if args else DEFAULT_ROOT
    workers = 48
    for a in sys.argv[1:]:
        if a.startswith("--workers"):
            workers = int(a.split("=")[1]) if "=" in a else 48
    root = os.path.normpath(root)
    print(f"[scan] root={root} workers={workers}", flush=True)

    t0 = time.perf_counter()
    paths = []
    for dirpath, _dirs, files in os.walk(root):
        for fn in files:
            if fn.endswith(".mp4"):
                paths.append(os.path.join(dirpath, fn))
    print(f"[scan] enumerated {len(paths):,} mp4s in {time.perf_counter()-t0:.0f}s", flush=True)

    bad = []  # (clip_id, sensor, reason)
    done = 0
    t1 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for path, reason in zip(paths, ex.map(check, paths)):
            done += 1
            if reason:
                base = os.path.basename(path)            # <clip_id>.<sensor>.mp4
                clip_id = base.split(".", 1)[0]
                sensor = base.split(".", 1)[1][:-4] if "." in base else ""
                bad.append((clip_id, sensor, reason))
            if done % 20000 == 0:
                print(f"[scan] {done:,}/{len(paths):,} bad={len(bad):,} "
                      f"({done/(time.perf_counter()-t1):.0f}/s)", flush=True)

    os.makedirs(OUTDIR, exist_ok=True)
    per_file = os.path.join(OUTDIR, "camera_corrupt_full_per_file.csv")
    per_clip = os.path.join(OUTDIR, "camera_corrupt_full_per_clip.csv")
    with open(per_file, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["clip_id", "sensor", "reason"])
        w.writerows(sorted(bad))

    # dedup to clip level, keep reason histogram per clip
    clip_reasons = {}
    for clip_id, _sensor, reason in bad:
        clip_reasons.setdefault(clip_id, set()).add(reason.split(":")[0])
    with open(per_clip, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["clip_id", "reasons", "bad_file_count"])
        counts = {}
        for clip_id, _s, _r in bad:
            counts[clip_id] = counts.get(clip_id, 0) + 1
        for clip_id in sorted(clip_reasons):
            w.writerow([clip_id, "|".join(sorted(clip_reasons[clip_id])), counts[clip_id]])

    print(f"\n[scan] DONE in {time.perf_counter()-t0:.0f}s", flush=True)
    print(f"[scan] total mp4s:       {len(paths):,}", flush=True)
    print(f"[scan] bad files:        {len(bad):,}", flush=True)
    print(f"[scan] distinct bad clips:{len(clip_reasons):,}", flush=True)
    # reason histogram
    from collections import Counter
    rc = Counter(r.split(":")[0] for _c, _s, r in bad)
    print(f"[scan] reason histogram: {dict(rc)}", flush=True)
    print(f"[scan] per-file -> {per_file}", flush=True)
    print(f"[scan] per-clip -> {per_clip}", flush=True)


if __name__ == "__main__":
    main()
