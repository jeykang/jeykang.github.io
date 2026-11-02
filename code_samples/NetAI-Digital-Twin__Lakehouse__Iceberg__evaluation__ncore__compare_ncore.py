#!/usr/bin/env python3
"""compare_ncore.py — our NCore conversion of a clip against NVIDIA's release of the same clip.

Both are opened with ncore's own V4 loader and compared level by level: files and
sizes, sequence meta, sensor sets, per-sensor frame counts and timestamps, sensor
extrinsics, ego poses sampled along the sequence, cuboid observations, and a few
decoded camera frames. Exact equality is not expected everywhere — NVIDIA's release
may come from a different converter version and a different video decoder — so each
row reports the measured difference and the script says which rows are structural
(must match) and which are numeric (report the magnitude).

    PYTHONPATH=repo .venv/bin/python compare_ncore.py out/pai_<clip>/pai_<clip>.json \
        reference/clips/<clip>/pai_<clip>.json
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
from upath import UPath

from ncore.impl.data.v4.compat import SequenceLoaderV4
from ncore.impl.data.v4.components import SequenceComponentGroupsReader


def load(meta_json: str) -> SequenceLoaderV4:
    return SequenceLoaderV4(SequenceComponentGroupsReader([UPath(meta_json)]))


def files_table(a_json: str, b_json: str):
    rows = []
    for label, p in (("ours", a_json), ("nvidia", b_json)):
        d = os.path.dirname(p)
        for f in sorted(os.listdir(d)):
            rows.append((label, f.split(".ncore4")[-1] or f, os.path.getsize(os.path.join(d, f))))
    keys = sorted({r[1] for r in rows})
    print(f"{'file':44s} {'ours MB':>9s} {'nvidia MB':>10s}")
    for k in keys:
        a = next((r[2] for r in rows if r[0] == "ours" and r[1] == k), None)
        b = next((r[2] for r in rows if r[0] == "nvidia" and r[1] == k), None)
        print(f"{k:44s} {(a or 0)/1e6:9.1f} {(b or 0)/1e6:10.1f}" + ("" if a and b else "   <- only one side"))


def cmp_scalar(name, a, b, structural=True):
    same = a == b
    flag = "OK" if same else ("MISMATCH" if structural else "differs")
    print(f"  {name:36s} ours={str(a)[:40]:40s} nvidia={str(b)[:40]:40s} {flag}")
    return same


def cmp_array(name, a, b, tol=1e-6):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != b.shape:
        print(f"  {name:36s} shape ours={a.shape} nvidia={b.shape}  MISMATCH")
        return False
    d = float(np.max(np.abs(a - b))) if a.size else 0.0
    print(f"  {name:36s} shape={a.shape} max|diff|={d:.3g}  {'OK' if d <= tol else 'differs'}")
    return d <= tol


def main(a_json: str, b_json: str):
    print("== files ==")
    files_table(a_json, b_json)
    A, B = load(a_json), load(b_json)
    print("\n== sequence ==")
    cmp_scalar("sequence_id", A.sequence_id, B.sequence_id)
    ia, ib = A.sequence_timestamp_interval_us, B.sequence_timestamp_interval_us
    cmp_scalar("interval_us", (ia.start, ia.stop), (ib.start, ib.stop))
    ma, mb = A.generic_meta_data, B.generic_meta_data
    for k in sorted(set(ma) | set(mb)):
        cmp_scalar(f"meta.{k}", ma.get(k), mb.get(k), structural=k not in ("converter_version",))

    print("\n== sensors ==")
    cams = sorted(set(A.camera_ids) & set(B.camera_ids))
    cmp_scalar("camera_ids (ours ⊆ nvidia?)", sorted(A.camera_ids), sorted(B.camera_ids),
               structural=False)
    print(f"  only in nvidia: {sorted(set(B.camera_ids) - set(A.camera_ids))}")
    cmp_scalar("lidar_ids", sorted(A.lidar_ids), sorted(B.lidar_ids))
    cmp_scalar("radar_ids", sorted(A.radar_ids), sorted(B.radar_ids), structural=False)

    print("\n== per camera ==")
    for cam in cams:
        sa, sb = A.get_camera_sensor(cam), B.get_camera_sensor(cam)
        cmp_scalar(f"{cam}.frames_count", sa.frames_count, sb.frames_count)
        ta, tb = np.asarray(sa.frames_timestamps_us), np.asarray(sb.frames_timestamps_us)
        if len(ta) == len(tb):
            cmp_array(f"{cam}.timestamps_us", ta, tb, tol=0)
        try:
            cmp_array(f"{cam}.T_sensor_rig", sa.T_sensor_rig, sb.T_sensor_rig, tol=1e-6)
        except Exception as e:
            print(f"  {cam}.T_sensor_rig: {type(e).__name__} {str(e)[:60]}")
        # a few decoded frames: pixel agreement (decoders may differ slightly)
        for idx in (0, sa.frames_count // 2, sa.frames_count - 1):
            try:
                fa = np.asarray(sa.get_frame_image_array(idx)).astype(np.int16)
                fb = np.asarray(sb.get_frame_image_array(idx)).astype(np.int16)
                if fa.shape != fb.shape:
                    print(f"  {cam}.frame[{idx}] shape ours={fa.shape} nvidia={fb.shape}  MISMATCH")
                else:
                    diff = np.abs(fa - fb)
                    print(f"  {cam}.frame[{idx}] shape={fa.shape} mean|diff|={diff.mean():.3f} "
                          f"max={diff.max()} px>8: {(diff > 8).mean()*100:.2f}%")
            except Exception as e:
                print(f"  {cam}.frame[{idx}]: {type(e).__name__} {str(e)[:80]}")
                break

    print("\n== lidar ==")
    for lid in sorted(set(A.lidar_ids) & set(B.lidar_ids)):
        la, lb = A.get_lidar_sensor(lid), B.get_lidar_sensor(lid)
        cmp_scalar(f"{lid}.frames_count", la.frames_count, lb.frames_count)
        ta, tb = np.asarray(la.frames_timestamps_us), np.asarray(lb.frames_timestamps_us)
        if len(ta) == len(tb):
            cmp_array(f"{lid}.timestamps_us", ta, tb, tol=0)
        try:
            cmp_array(f"{lid}.T_sensor_rig", la.T_sensor_rig, lb.T_sensor_rig, tol=1e-6)
        except Exception as e:
            print(f"  {lid}.T_sensor_rig: {type(e).__name__} {str(e)[:60]}")
        for idx in (0, la.frames_count // 2, la.frames_count - 1):
            try:
                na, nb = la.get_frame_ray_bundle_count(idx), lb.get_frame_ray_bundle_count(idx)
                cmp_scalar(f"{lid}.frame[{idx}].ray_bundles", na, nb)
                if na == nb:
                    cmp_array(f"{lid}.frame[{idx}].return_distance_m",
                              la.get_frame_ray_bundle_return_distance_m(idx),
                              lb.get_frame_ray_bundle_return_distance_m(idx), tol=0)
                    cmp_array(f"{lid}.frame[{idx}].return_intensity",
                              la.get_frame_ray_bundle_return_intensity(idx),
                              lb.get_frame_ray_bundle_return_intensity(idx), tol=0)
            except Exception as e:
                print(f"  {lid}.frame[{idx}]: {type(e).__name__} {str(e)[:80]}")
                break

    print("\n== ego poses ==")
    try:
        ts = np.linspace(ia.start, ia.stop - 1, 9).astype(np.int64)
        pa = np.asarray(A.pose_graph.evaluate_poses("rig", "world", ts)) if hasattr(A.pose_graph, "evaluate_poses") else None
        pb = np.asarray(B.pose_graph.evaluate_poses("rig", "world", ts)) if hasattr(B.pose_graph, "evaluate_poses") else None
        if pa is not None and pb is not None:
            cmp_array("rig->world at 9 times", pa, pb, tol=1e-4)
    except Exception as e:
        print(f"  pose_graph.evaluate_poses: {type(e).__name__} {str(e)[:100]}")
        try:
            print(f"  nodes ours={sorted(A.pose_graph.nodes)} nvidia={sorted(B.pose_graph.nodes)}")
        except Exception as e2:
            print(f"  nodes: {type(e2).__name__}")

    print("\n== cuboids ==")
    try:
        ca = list(A.get_cuboid_track_observations())
        cb = list(B.get_cuboid_track_observations())
        cmp_scalar("cuboid observations", len(ca), len(cb))
        if ca and cb:
            print(f"  first ours:   {str(ca[0])[:160]}")
            print(f"  first nvidia: {str(cb[0])[:160]}")
    except Exception as e:
        print(f"  cuboids: {type(e).__name__} {str(e)[:100]}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
