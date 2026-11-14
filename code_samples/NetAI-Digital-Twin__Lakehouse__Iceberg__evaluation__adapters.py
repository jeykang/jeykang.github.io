#!/usr/bin/env python3
"""Dataset adapters. One per dataset; the rest of the pipeline is agnostic.

Implemented:
  NvidiaAdapter — NVIDIA PhysicalAI Autonomous Vehicles (the populated dataset).

To add a dataset, implement `scenario.DatasetAdapter` and register it in ADAPTERS.
You need exactly two things from the source: per-timestamp ego poses and agent
boxes. Nothing else.
"""
from __future__ import annotations

import glob
import io
import math
import os
import zipfile
from typing import Dict, List, Optional, Sequence

import pyarrow.parquet as pq

from scenario import AgentBox, EgoState, Scenario

_C = "/mnt/netai-e2e/nvidia-physicalai-av-subset"
_LOCAL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                      "netai-e2e", "nvidia-physicalai-av-subset")
NVIDIA_ROOT = os.environ.get("AV_ROOT", _C if os.path.isdir(_C) else _LOCAL)


def _yaw_from_quat(qx: float, qy: float, qz: float, qw: float) -> float:
    """Yaw about +z from a unit quaternion."""
    return math.atan2(2.0 * (qw * qz + qx * qy),
                      1.0 - 2.0 * (qy * qy + qz * qz))


class NvidiaAdapter:
    """NVIDIA PhysicalAI AV.

    Source layout:
      labels/egomotion/<chunk>/<clip>.egomotion.parquet
        timestamp(us), x,y,z, qx,qy,qz,qw, vx,vy,vz, ... — already a world/odometry pose.
      labels/obstacle.offline/obstacle.offline.chunk_XXXX.zip -> <clip>.obstacle.offline.parquet
        center_*, size_*, orientation_*, label_class, and crucially
        reference_frame='rig' + reference_frame_timestamp_us.

    The rig->world lift is the part that is easy to get wrong: each agent box is
    expressed in the ego rig frame *at its own reference timestamp*, so it must be
    rotated and translated by the ego pose at THAT timestamp, not at t0.
    """

    name = "nvidia_physicalai_av"

    def __init__(self, root: str = NVIDIA_ROOT):
        self.root = root
        self._chunk_of: Dict[str, str] = {}
        self._dims: Dict[str, tuple] = {}

    # ── discovery ────────────────────────────────────────────────────────────
    _CLIP_CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".clip_list.json")

    def list_clips(self) -> Sequence[str]:
        """All clip ids, cached to disk.

        The underlying glob walks ~33k files over NFS; through a Docker bind-mount
        that took minutes and looked like a hang. Cache it — delete
        `.clip_list.json` after the dataset changes.
        """
        import json
        if os.path.exists(self._CLIP_CACHE):
            try:
                return json.load(open(self._CLIP_CACHE))
            except Exception:
                pass
        ids = sorted(os.path.basename(p).split(".")[0]
                     for p in glob.glob(f"{self.root}/labels/egomotion/*/*.egomotion.parquet"))
        try:
            json.dump(ids, open(self._CLIP_CACHE, "w"))
        except Exception:
            pass
        return ids

    def _chunk(self, clip_id: str) -> Optional[str]:
        """Chunk id for a clip, taken from its egomotion path."""
        if clip_id not in self._chunk_of:
            m = glob.glob(f"{self.root}/labels/egomotion/*/{clip_id}.egomotion.parquet")
            if not m:
                return None
            self._chunk_of[clip_id] = os.path.basename(os.path.dirname(m[0])).split("chunk_")[-1]
        return self._chunk_of.get(clip_id)

    # ── ego ──────────────────────────────────────────────────────────────────
    def _ego(self, clip_id: str) -> List[EgoState]:
        m = glob.glob(f"{self.root}/labels/egomotion/*/{clip_id}.egomotion.parquet")
        if not m:
            return []
        d = pq.read_table(m[0], columns=["timestamp", "x", "y", "z", "qx", "qy", "qz",
                                         "qw", "vx", "vy", "ax", "ay"]).to_pydict()
        out = []
        for i in range(len(d["timestamp"])):
            yaw = _yaw_from_quat(d["qx"][i], d["qy"][i], d["qz"][i], d["qw"][i])
            # vx/vy are body-frame; rotate into world so every consumer sees one frame.
            c, s = math.cos(yaw), math.sin(yaw)
            out.append(EgoState(int(d["timestamp"][i]), float(d["x"][i]), float(d["y"][i]),
                                yaw,
                                c * d["vx"][i] - s * d["vy"][i],
                                s * d["vx"][i] + c * d["vy"][i],
                                float(d["z"][i]), float(d["qx"][i]), float(d["qy"][i]),
                                float(d["qz"][i]), float(d["qw"][i]),
                                float(d["ax"][i]), float(d["ay"][i])))
        out.sort(key=lambda e: e.t_us)
        return out

    def _footprint(self, clip_id: str) -> tuple:
        ch = self._chunk(clip_id)
        if ch is None:
            return (4.87, 2.12)
        if ch not in self._dims:
            p = f"{self.root}/calibration/vehicle_dimensions/vehicle_dimensions.chunk_{ch}.parquet"
            table = {}
            if os.path.exists(p):
                d = pq.read_table(p, columns=["clip_id", "length", "width"]).to_pydict()
                table = {c: (float(l), float(w))
                         for c, l, w in zip(d["clip_id"], d["length"], d["width"])}
            self._dims[ch] = table
        return self._dims[ch].get(clip_id, (4.87, 2.12))

    # ── agents ───────────────────────────────────────────────────────────────
    def _agents(self, clip_id: str, ego: List[EgoState]) -> Dict[str, List[AgentBox]]:
        ch = self._chunk(clip_id)
        if ch is None:
            return {}
        zp = f"{self.root}/labels/obstacle.offline/obstacle.offline.chunk_{ch}.zip"
        if not os.path.exists(zp):
            return {}
        try:
            zf = zipfile.ZipFile(zp)
            nm = f"{clip_id}.obstacle.offline.parquet"
            if nm not in zf.namelist():
                return {}
            d = pq.read_table(io.BytesIO(zf.read(nm))).to_pydict()
        except Exception:
            return {}

        by_t = {e.t_us: e for e in ego}
        ts_sorted = sorted(by_t)

        def ego_at(t: int) -> Optional[EgoState]:
            if not ts_sorted:
                return None
            return by_t[min(ts_sorted, key=lambda x: abs(x - t))]

        tracks: Dict[str, List[AgentBox]] = {}
        for i in range(len(d["timestamp_us"])):
            t = int(d["timestamp_us"][i])
            ref_t = int(d.get("reference_frame_timestamp_us", d["timestamp_us"])[i])
            e = ego_at(ref_t)
            if e is None:
                continue
            ax, ay = float(d["center_x"][i]), float(d["center_y"][i])
            a_yaw = _yaw_from_quat(d["orientation_x"][i], d["orientation_y"][i],
                                   d["orientation_z"][i], d["orientation_w"][i])
            c, s = math.cos(e.yaw), math.sin(e.yaw)
            tracks.setdefault(str(d["track_id"][i]), []).append(AgentBox(
                t_us=t, track_id=str(d["track_id"][i]),
                x=e.x + c * ax - s * ay,
                y=e.y + s * ax + c * ay,
                yaw=e.yaw + a_yaw,
                length=float(d["size_x"][i]), width=float(d["size_y"][i]),
                label=str(d["label_class"][i]),
            ))
        for k in tracks:
            tracks[k].sort(key=lambda b: b.t_us)
        return tracks

    # ── entry point ──────────────────────────────────────────────────────────
    def load(self, clip_id: str) -> Optional[Scenario]:
        ego = self._ego(clip_id)
        if len(ego) < 10:
            return None
        agents = self._agents(clip_id, ego)
        length, width = self._footprint(clip_id)
        return Scenario(clip_id=clip_id, dataset=self.name, ego=ego, agents=agents,
                        ego_length=length, ego_width=width,
                        meta={"n_tracks": len(agents)},
                        sensor_factory=self.sensor_reader,
                        sensor_span_fn=lambda cid=clip_id: self.sensor_span(cid))

    # ── sensors ──────────────────────────────────────────────────────────────
    CAMERAS = ("camera_cross_left_120fov", "camera_front_wide_120fov",
               "camera_cross_right_120fov", "camera_front_tele_30fov",
               "camera_rear_left_70fov", "camera_rear_right_70fov",
               "camera_rear_tele_30fov")

    def sensor_span(self, clip_id: str, camera: str = "camera_front_wide_120fov"):
        """(first, last) camera timestamp for a clip, from the .timestamps sidecar."""
        ch = self._chunk(clip_id)
        if ch is None:
            return None
        p = (f"{self.root}/camera/{camera}/{camera}.chunk_{ch}/"
             f"{clip_id}.{camera}.timestamps.parquet")
        if not os.path.exists(p):
            return None
        d = pq.read_table(p, columns=["timestamp"]).to_pydict()["timestamp"]
        return (int(min(d)), int(max(d))) if d else None

    def sensor_reader(self, clip_id: str, max_t_us: int):
        ch = self._chunk(clip_id)
        return None if ch is None else NvidiaSensorReader(self.root, clip_id, ch, max_t_us)


class NvidiaSensorReader:
    """Decodes camera frames from the local mp4s, bounded to `max_t_us`.

    Each clip/camera ships a `.timestamps.parquet` sidecar mapping frame_index to
    the same microsecond clock as egomotion and the obstacle labels, so frames can
    be addressed by time rather than by index — which is what a policy needs and
    what makes the history bound enforceable.
    """

    def __init__(self, root: str, clip_id: str, chunk: str, max_t_us: int):
        self.root, self.clip_id, self.chunk = root, clip_id, chunk
        self.max_t_us = max_t_us
        self._ts: Dict[str, list] = {}

    def available(self) -> Sequence[str]:
        return NvidiaAdapter.CAMERAS

    def _paths(self, camera: str):
        base = f"{self.root}/camera/{camera}/{camera}.chunk_{self.chunk}/{self.clip_id}.{camera}"
        return base + ".mp4", base + ".timestamps.parquet"

    def _timestamps(self, camera: str):
        if camera not in self._ts:
            _, tsp = self._paths(camera)
            if not os.path.exists(tsp):
                self._ts[camera] = []
            else:
                d = pq.read_table(tsp, columns=["timestamp", "frame_index"]).to_pydict()
                self._ts[camera] = sorted(zip(d["timestamp"], d["frame_index"]))
        return self._ts[camera]

    def lidar(self, t_us: int, sensor: str = "lidar_top_360fov"):
        """Decoded point cloud (N,3) at or before t_us, or None.

        Bounded by max_t_us like frames(). Sweep selection prefers a timestamp
        column if the parquet has one and otherwise falls back to position within
        the camera span — the same fractional indexing the earlier DiffusionDrive
        runner used, kept so results stay comparable to that work.
        """
        import numpy as np
        if t_us > self.max_t_us:
            raise ValueError(f"lidar request {t_us} past decision time {self.max_t_us}")
        p = (f"{self.root}/lidar/{sensor}/{sensor}.chunk_{self.chunk}/"
             f"{self.clip_id}.{sensor}.parquet")
        if not os.path.exists(p):
            return None
        try:
            import DracoPy
        except ImportError:
            raise RuntimeError("lidar decoding needs DracoPy (present in the "
                               "diffusiondrive-runner image)")
        f = pq.ParquetFile(p)
        n = f.metadata.num_rows
        if not n:
            return None
        tcol = next((c for c in f.schema_arrow.names if "time" in c.lower()), None)

        # Pick the sweep index WITHOUT materialising the point clouds. Each clip
        # holds ~200 Draco blobs totalling hundreds of MB; reading them all to use
        # one is what made the first version appear to hang over NFS.
        if tcol:
            ts = pq.read_table(p, columns=[tcol]).column(0).to_pylist()
            cand = [k for k, t in enumerate(ts) if t <= t_us]
            i = max(cand) if cand else 0
        else:
            cam_ts = self._timestamps("camera_front_wide_120fov")
            if cam_ts:
                lo, hi = cam_ts[0][0], cam_ts[-1][0]
                frac = 0.0 if hi <= lo else max(0.0, min(1.0, (t_us - lo) / (hi - lo)))
            else:
                frac = 0.5
            i = min(n - 1, int(n * frac))

        # Read only the row group containing that sweep.
        seen = 0
        for rg in range(f.metadata.num_row_groups):
            rows = f.metadata.row_group(rg).num_rows
            if seen + rows > i:
                col = f.read_row_group(rg, columns=["draco_encoded_pointcloud"]).column(0)
                blob = col[i - seen].as_py()
                return np.asarray(DracoPy.decode(blob).points, np.float32)
            seen += rows
        return None

    def frame_times(self, camera: str, t_us: Sequence[int]):
        """Actual timestamps of the frames `frames()` would return.

        Alpamayo 2 consumes absolute per-frame timestamps (it derives relative time
        and the ego-t0 offset from them), so a policy needs the real capture times,
        not the requested ones.
        """
        ts = self._timestamps(camera)
        if not ts:
            return []
        return [min(ts, key=lambda r: abs(r[0] - t))[0] for t in t_us]

    def frames(self, camera: str, t_us: Sequence[int]):
        import cv2
        import numpy as np

        over = [t for t in t_us if t > self.max_t_us]
        if over:
            raise ValueError(
                f"SensorReader is bounded to t<={self.max_t_us}; refused {len(over)} "
                "future request(s). A policy must not observe past its decision time.")
        ts = self._timestamps(camera)
        mp4, _ = self._paths(camera)
        if not ts or not os.path.exists(mp4):
            return []
        # Refuse requests outside the recorded window. Nearest-frame lookup would
        # otherwise silently clamp: a request 22 s past the end of a 20 s video
        # returns the final frame, pairing stale pixels with fresh ego history and
        # producing confidently wrong trajectories. Found exactly that way.
        lo, hi = ts[0][0], ts[-1][0]
        tol = 2 * (ts[1][0] - ts[0][0]) if len(ts) > 1 else 100_000
        out_of_range = [t for t in t_us if t < lo - tol or t > hi + tol]
        if out_of_range:
            raise ValueError(
                f"{camera} covers [{lo}, {hi}] us; refused {len(out_of_range)} "
                f"request(s) outside it (e.g. {out_of_range[0]}). Sensor coverage is "
                "much shorter than the egomotion track on this dataset.")
        cap = cv2.VideoCapture(mp4)
        out = []
        try:
            for t in t_us:
                idx = min(range(len(ts)), key=lambda i: abs(ts[i][0] - t))
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(ts[idx][1]))
                ok, fr = cap.read()
                out.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB) if ok
                           else np.zeros((1080, 1920, 3), np.uint8))
        finally:
            cap.release()
        return out


ADAPTERS = {NvidiaAdapter.name: NvidiaAdapter, "nvidia": NvidiaAdapter}


def get_adapter(name: str = "nvidia", **kw):
    if name not in ADAPTERS:
        raise KeyError(f"unknown dataset '{name}'; have {sorted(set(ADAPTERS))}. "
                       "Add one by implementing scenario.DatasetAdapter.")
    return ADAPTERS[name](**kw)
