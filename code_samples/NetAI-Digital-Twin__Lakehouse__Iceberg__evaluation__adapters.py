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
    def list_clips(self) -> Sequence[str]:
        return sorted(os.path.basename(p).split(".")[0]
                      for p in glob.glob(f"{self.root}/labels/egomotion/*/*.egomotion.parquet"))

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
        d = pq.read_table(m[0], columns=["timestamp", "x", "y", "qx", "qy", "qz",
                                         "qw", "vx", "vy"]).to_pydict()
        out = []
        for i in range(len(d["timestamp"])):
            yaw = _yaw_from_quat(d["qx"][i], d["qy"][i], d["qz"][i], d["qw"][i])
            # vx/vy are body-frame; rotate into world so every consumer sees one frame.
            c, s = math.cos(yaw), math.sin(yaw)
            out.append(EgoState(int(d["timestamp"][i]), float(d["x"][i]), float(d["y"][i]),
                                yaw,
                                c * d["vx"][i] - s * d["vy"][i],
                                s * d["vx"][i] + c * d["vy"][i]))
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
                        meta={"n_tracks": len(agents)})


ADAPTERS = {NvidiaAdapter.name: NvidiaAdapter, "nvidia": NvidiaAdapter}


def get_adapter(name: str = "nvidia", **kw):
    if name not in ADAPTERS:
        raise KeyError(f"unknown dataset '{name}'; have {sorted(set(ADAPTERS))}. "
                       "Add one by implementing scenario.DatasetAdapter.")
    return ADAPTERS[name](**kw)
