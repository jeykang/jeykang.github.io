#!/usr/bin/env python3
"""validate_ncore.py — open a converted NCore v4 sequence with ncore's own loader and
report what it contains. This is the schema-level gate for the ncore serving mode: if
NVIDIA's reader accepts the store and the sensors, time span, poses and cuboids match the
source clip, the conversion is structurally right.

    PYTHONPATH=repo .venv/bin/python validate_ncore.py out/pai_<clip>/pai_<clip>.json
"""
import sys

import numpy as np
from upath import UPath

from ncore.impl.data.v4.compat import SequenceLoaderV4
from ncore.impl.data.v4.components import SequenceComponentGroupsReader


def main(meta_json: str):
    loader = SequenceLoaderV4(SequenceComponentGroupsReader([UPath(meta_json)]))
    iv = loader.sequence_timestamp_interval_us
    print(f"sequence_id            {loader.sequence_id}")
    print(f"time span              {(iv.stop - iv.start)/1e6:.2f} s  [{iv.start} .. {iv.stop}) us")
    print(f"cameras ({len(loader.camera_ids)})           {loader.camera_ids}")
    print(f"lidars  ({len(loader.lidar_ids)})           {loader.lidar_ids}")
    api = [m for m in dir(loader) if not m.startswith("_")]
    print(f"loader api             {api}")
    # best-effort probes of the optional content, named by the compat API
    for name in ("radar_ids", "generic_meta_data"):
        if hasattr(loader, name):
            v = getattr(loader, name)
            print(f"{name:22s} {str(v)[:160]}")
    for cam in loader.camera_ids[:1]:
        for name in ("get_camera_image_timestamps_us", "camera_timestamps_us", "get_sensor_timestamps_us"):
            if hasattr(loader, name):
                try:
                    ts = np.asarray(getattr(loader, name)(cam))
                    print(f"{cam}: {len(ts)} frames via {name}, {(ts[-1]-ts[0])/1e6:.2f} s")
                    break
                except Exception as e:
                    print(f"{name}: {type(e).__name__} {str(e)[:80]}")
    print("OK: loader opened the sequence")


if __name__ == "__main__":
    main(sys.argv[1])
