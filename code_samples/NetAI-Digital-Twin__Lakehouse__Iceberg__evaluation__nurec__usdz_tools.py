#!/usr/bin/env python3
"""usdz_tools.py — inspect a NuRec USDZ bundle, or build a hybrid from two bundles.

    usdz_tools.py inspect <bundle.usdz>
    usdz_tools.py inject <ours.usdz> <reference.usdz> <out.usdz> [--members map.xodr clipgt/ ...]
    usdz_tools.py fix-meta <bundle.usdz> --scene-id clipgt-<clip> [--uuid <uuid>]
    usdz_tools.py add <bundle.usdz> <out.usdz> <file> [<file> ...]   # add loose files (e.g. a post-hoc ground mesh)
    usdz_tools.py shift-time <bundle.usdz> <out.usdz> [--offset-us N | --from-reference <ref.usdz>] [--drop-before-start [--drop-members M ...]]

`inject` copies members that our export cannot produce (PAI has no map labels, so no
`map.xodr` / `clipgt/*`) from NVIDIA's bundle of the same clip into ours, leaving our
neural scene, trajectories, tracks and meshes untouched. That isolates reconstruction
quality from map availability in an AlpaSim A/B. Entries are written uncompressed, as
USDZ expects.

`shift-time` moves a bundle onto a new absolute time base. Our NCore stores are
clip-relative (the PAI converter re-bases every clip to its egomotion origin, see
ncore/tools/data_converter/pai/utils.py), so the export carries
`absoluteTimeOffsetMicroSec = 0` and the track samples that precede the sequence start
are negative microseconds. AlpaSim loads track timestamps as uint64
(alpasim_utils/scenario.py, TrafficObjects.load_from_json) and fails on the first
negative one. NVIDIA's bundles use an absolute base (`int64` offset, all timestamps
>= offset). The shift adds the same delta to every `*timestamp(s)_us` value in the JSON
members, to `metadata.yaml: time_range` and to the USDA offsets; USD time codes stay
relative to the offset, so nothing else moves. `--drop-before-start` additionally drops
track samples earlier than the new scene start (and tracks left empty by that).

Caveat: the checkpoint stores datasource microseconds too (state_dict
`*.time_embed.timestamps_us_ranges`, `*.time_input_embedding.timestamps_us_ranges`) and the
renderer (nre/grpc/serve.py) feeds request timestamps to the model unchanged, so a shifted
bundle only renders time-varying content correctly if the checkpoint is shifted with it.
`shift-time` does not touch checkpoint.ckpt; for AlpaSim use, keep the base and run
`--drop-before-start` alone (offset unchanged), which is the tested path.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import re
import sys
import uuid as uuidlib
import zipfile

import yaml

# map layers PAI lacks, plus the ground mesh in case our export produced none (only
# members missing from our bundle are ever copied)
DEFAULT_INJECT = ["map.xodr", "clipgt/", "mesh_ground.ply", "mesh_ground.usd", "pose_record.json"]


def inspect(path: str):
    z = zipfile.ZipFile(path)
    names = z.namelist()
    tot = sum(z.getinfo(n).file_size for n in names)
    print(f"{path}: {len(names)} members, {tot/1e9:.2f} GB")
    for n in names:
        if not n.startswith("clipgt/") and not n.startswith("frames/"):
            print(f"  {z.getinfo(n).file_size/1e6:9.1f} MB  {n}")
    print(f"  clipgt/ members: {sum(n.startswith('clipgt/') for n in names)}; frames/: {sum(n.startswith('frames/') for n in names)}")
    if "metadata.yaml" in names:
        m = yaml.safe_load(z.read("metadata.yaml"))
        print("  metadata:", {k: m.get(k) for k in ("uuid", "scene_id", "version_string", "time_range")},
              "cameras:", len((m.get("sensors") or {}).get("camera_ids") or []))


def _copy(zin: zipfile.ZipFile, zout: zipfile.ZipFile, name: str):
    info = zin.getinfo(name)
    zout.writestr(zipfile.ZipInfo(name, date_time=info.date_time), zin.read(name), compress_type=zipfile.ZIP_STORED)


def inject(ours: str, ref: str, out: str, members: list[str]):
    zo, zr = zipfile.ZipFile(ours), zipfile.ZipFile(ref)
    have = set(zo.namelist())
    want = [n for n in zr.namelist() if any(n == m or (m.endswith("/") and n.startswith(m)) for m in members)]
    added = [n for n in want if n not in have]
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_STORED) as zw:
        for n in zo.namelist():
            _copy(zo, zw, n)
        for n in added:
            _copy(zr, zw, n)
    print(f"{out}: {len(zo.namelist())} own members + {len(added)} injected from reference: "
          f"{[n for n in added if not n.startswith('clipgt/')]} + {sum(n.startswith('clipgt/') for n in added)} clipgt files")


def add(bundle: str, out: str, files: list[str]):
    """Copy the bundle and add loose files at the archive root (existing names are replaced)."""
    zin = zipfile.ZipFile(bundle)
    names = {os.path.basename(f) for f in files}
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_STORED) as zw:
        for n in zin.namelist():
            if n not in names:
                _copy(zin, zw, n)
        for f in files:
            zw.write(f, os.path.basename(f), compress_type=zipfile.ZIP_STORED)
    print(f"{out}: {len(zin.namelist())} members + added {sorted(names)}")

# --- time base ----------------------------------------------------------------------
TIME_KEY = re.compile(r"timestamps?_us$")          # tracks_timestamps_us, T_rig_world_timestamps_us, ...
INTERVAL_KEY = "sequence_timestamp_interval_us"     # data_info.json: {"start": .., "stop": ..}
USDA_OFFSET = re.compile(r"int(?:64)? absoluteTimeOffsetMicroSec = (-?\d+)")
TRACK_CONTAINER = ("tracks_data", "cuboidtracks_data")


def read_offset(z: zipfile.ZipFile) -> int:
    m = USDA_OFFSET.search(z.read("default.usda").decode())
    if not m:
        raise SystemExit("default.usda carries no absoluteTimeOffsetMicroSec")
    return int(m.group(1))


def _shift_vals(v, delta: int, stats: dict, path: str):
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        stats[path] = stats.get(path, 0) + 1
        return v + delta
    if isinstance(v, list):
        return [_shift_vals(x, delta, stats, path) for x in v]
    if isinstance(v, dict):  # {start, stop} intervals, or {sensor_id: [[start, end], ...]}
        return {k: _shift_vals(x, delta, stats, path) for k, x in v.items()}
    return v


def _shift_json(obj, delta: int, stats: dict, path: str = ""):
    """Add delta to every *timestamp(s)_us value (scalars, nested lists, {start,stop} intervals)."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if TIME_KEY.search(k) or k == INTERVAL_KEY:
                obj[k] = _shift_vals(v, delta, stats, f"{path}/{k}")
            else:
                _shift_json(v, delta, stats, f"{path}/{k}")
    elif isinstance(obj, list):
        for v in obj:
            _shift_json(v, delta, stats, path)


def _drop_before(obj, start_us: int, stats: dict):
    """Drop track samples earlier than start_us from every tracks_data/cuboidtracks_data pair in obj."""
    if isinstance(obj, list):
        for v in obj:
            _drop_before(v, start_us, stats)
        return
    if not isinstance(obj, dict):
        return
    if not all(k in obj for k in TRACK_CONTAINER):
        for v in obj.values():
            _drop_before(v, start_us, stats)
        return
    td, cd = obj["tracks_data"], obj["cuboidtracks_data"]
    n_tracks = len(td["tracks_timestamps_us"])
    keep_tracks = []
    for i, ts in enumerate(td["tracks_timestamps_us"]):
        keep = [j for j, t in enumerate(ts) if t >= start_us]
        stats["samples"] = stats.get("samples", 0) + len(ts) - len(keep)
        if not keep:
            stats["tracks"] = stats.get("tracks", 0) + 1
            continue
        if len(keep) != len(ts):
            td["tracks_timestamps_us"][i] = [ts[j] for j in keep]
            td["tracks_poses"][i] = [td["tracks_poses"][i][j] for j in keep]
        keep_tracks.append(i)
    if len(keep_tracks) != n_tracks:  # every per-track array shrinks together
        for d in (td, cd):
            for k, v in d.items():
                if isinstance(v, list) and len(v) == n_tracks:
                    d[k] = [v[i] for i in keep_tracks]
    stats["min_samples"] = min([stats.get("min_samples", 10**9)] + [len(ts) for ts in td["tracks_timestamps_us"]])


def shift_time(bundle: str, out: str, offset_us: int | None, reference: str | None, drop_before_start: bool,
               drop_members: list[str] | None = None):
    z = zipfile.ZipFile(bundle)
    names = z.namelist()
    cur = read_offset(z)
    if reference:
        offset_us = read_offset(zipfile.ZipFile(reference))
    if offset_us is None:
        offset_us = cur
    delta = offset_us - cur
    start_us = offset_us  # USD time code 0 == scene start == metadata time_range.start
    print(f"{bundle}: absoluteTimeOffsetMicroSec {cur} -> {offset_us} (delta {delta:+d})"
          + (f", dropping track samples < {start_us}" if drop_before_start else ""))
    edits: dict[str, bytes] = {}
    for n in names:
        if n.endswith(".usda"):
            t = z.read(n).decode()
            t2, k = USDA_OFFSET.subn(f"int64 absoluteTimeOffsetMicroSec = {offset_us}", t)
            if k and delta:
                edits[n] = t2.encode()
                print(f"  {n}: {k} offset field(s) rewritten (int64)")
        elif n == "metadata.yaml":
            m = yaml.safe_load(z.read(n))
            tr = m.get("time_range") or {}
            if delta and tr:
                tr["start"] += delta; tr["end"] += delta
                edits[n] = yaml.safe_dump(m, sort_keys=False).encode()
                print(f"  {n}: time_range -> {tr}")
        elif n.endswith(".json") and n != "pose_record.json":  # pose_record stays clip-relative in NVIDIA's bundles too
            o = json.loads(z.read(n))
            stats: dict = {}
            if delta:
                _shift_json(o, delta, stats)
            drop: dict = {}
            if drop_before_start and (not drop_members or n in drop_members):
                _drop_before(o, start_us, drop)
            if stats or drop.get("samples") or drop.get("tracks"):
                edits[n] = json.dumps(o).encode()
                print(f"  {n}: shifted {sum(stats.values())} values in {len(stats)} field(s)"
                      + (f"; dropped {drop.get('samples', 0)} pre-start sample(s), {drop.get('tracks', 0)} empty track(s), "
                         f"min samples/track now {drop.get('min_samples')}" if drop else ""))
    if not edits:
        print("  nothing to do"); return
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_STORED) as zw:
        for n in names:
            if n in edits:
                zw.writestr(zipfile.ZipInfo(n, date_time=z.getinfo(n).date_time), edits[n], compress_type=zipfile.ZIP_STORED)
            else:
                _copy(z, zw, n)
    print(f"{out}: {len(names)} members, {len(edits)} rewritten: {sorted(edits)}")


def fix_meta(path: str, scene_id: str | None, uid: str | None):
    z = zipfile.ZipFile(path)
    m = yaml.safe_load(z.read("metadata.yaml")) if "metadata.yaml" in z.namelist() else {}
    changed = False
    if scene_id and m.get("scene_id") != scene_id:
        m["scene_id"] = scene_id; changed = True
    if uid or not m.get("uuid"):
        m["uuid"] = uid or str(uuidlib.uuid4()); changed = True
    if not changed:
        print("metadata unchanged:", {k: m.get(k) for k in ("uuid", "scene_id")}); return
    tmp = path + ".tmp"
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_STORED) as zw:
        for n in z.namelist():
            if n == "metadata.yaml":
                zw.writestr("metadata.yaml", yaml.safe_dump(m, sort_keys=False))
            else:
                _copy(z, zw, n)
    z.close(); os.replace(tmp, path)
    print("metadata updated:", {k: m.get(k) for k in ("uuid", "scene_id")})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("inspect"); s.add_argument("bundle")
    s = sub.add_parser("inject"); s.add_argument("ours"); s.add_argument("reference"); s.add_argument("out")
    s.add_argument("--members", nargs="+", default=DEFAULT_INJECT)
    s = sub.add_parser("fix-meta"); s.add_argument("bundle"); s.add_argument("--scene-id"); s.add_argument("--uuid")
    s = sub.add_parser("add"); s.add_argument("bundle"); s.add_argument("out"); s.add_argument("files", nargs="+")
    s = sub.add_parser("shift-time"); s.add_argument("bundle"); s.add_argument("out")
    g = s.add_mutually_exclusive_group(); g.add_argument("--offset-us", type=int); g.add_argument("--from-reference")
    s.add_argument("--drop-before-start", action="store_true")
    s.add_argument("--drop-members", nargs="+", metavar="MEMBER",
                   help="restrict --drop-before-start to these JSON members (default: every member with track data); "
                        "AlpaSim reads sequence_tracks.json, the renderer rebuilds the model from datasource_summary.json, "
                        "whose track sample count must match the checkpoint")
    a = ap.parse_args()
    if a.cmd == "inspect":
        inspect(a.bundle)
    elif a.cmd == "inject":
        inject(a.ours, a.reference, a.out, a.members)
    elif a.cmd == "add":
        add(a.bundle, a.out, a.files)
    elif a.cmd == "shift-time":
        shift_time(a.bundle, a.out, a.offset_us, a.from_reference, a.drop_before_start, a.drop_members)
    else:
        fix_meta(a.bundle, a.scene_id, a.uuid)


if __name__ == "__main__":
    main()
