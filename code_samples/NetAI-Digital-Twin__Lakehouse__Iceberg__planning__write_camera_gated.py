"""Write the agent-gated camera-only perceptual score consumed by edge_case_scorer.

Raw camera_low_conf over-flags empty scenes (25% of clips have 0 detections -> 1.0,
but they have low agent load -> they are EMPTY, not hard). Gate: count camera
perceptual difficulty only where obstacle.offline says agents are present (via the
behavioral axes). Empties -> 0. Output: .camera_perception/camera_gated.parquet
(clip_id, low_conf, raw_low_conf, cam_ndet). See cosmos_augmentation/FINDINGS.md.
"""
import os
import pyarrow as pa, pyarrow.parquet as pq

ROOT = os.environ.get("NFS_ROOT", "netai-e2e/nvidia-physicalai-av-subset")
cam = pq.read_table(f"{ROOT}/.camera_perception/camera_perception.parquet").to_pydict()
beh = pq.read_table(f"{ROOT}/.behavioral/behavioral_shard_00_of_01.parquet").to_pydict()
B = {c: i for i, c in enumerate(beh["clip_id"])}


def present(c):
    i = B.get(c)
    if i is None:
        return True   # no behavioral info -> don't gate out
    return (beh["conflict"][i] > 0 or beh["vru"][i] > 0
            or beh["multidir"][i] > 1 or beh["closing"][i] > 0)


rows = [{"clip_id": c, "low_conf": (float(lc) if present(c) else 0.0),
         "raw_low_conf": float(lc), "cam_ndet": float(nd)}
        for c, lc, nd in zip(cam["clip_id"], cam["camera_low_conf"], cam["cam_ndet"])]
pq.write_table(pa.Table.from_pylist(rows), f"{ROOT}/.camera_perception/camera_gated.parquet")
g = [r["low_conf"] for r in rows]
print(f"wrote camera_gated.parquet: {len(rows)} clips; gated low_conf mean="
      f"{sum(g)/len(g):.3f}, hard(>0.6)={100*sum(v>0.6 for v in g)/len(g):.1f}%")
