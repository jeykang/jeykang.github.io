"""Write SparseDrive's kmeans anchor files from the pretrained checkpoint.

SparseDrive's config loads `data/kmeans/kmeans_{det,map,motion,plan}_*.npy` at
model-build time — normally generated from nuScenes (`scripts/kmeans.sh`), which
we don't have. But those anchors are stored as buffers in the stage-2 checkpoint
(and `load_checkpoint` overwrites them anyway), so we dump them straight from the
checkpoint and skip nuScenes entirely.

Run from the SparseDrive repo root (checkpoint at ckpt/sparsedrive_stage2.pth).
"""
import os
import numpy as np
import torch

CKPT = os.environ.get("SD_CKPT", "ckpt/sparsedrive_stage2.pth")
# (config: fut_mode = ego_fut_mode = 6)
ANCHORS = {
    "data/kmeans/kmeans_det_900.npy":  "head.det_head.instance_bank.anchor",
    "data/kmeans/kmeans_map_100.npy":  "head.map_head.instance_bank.anchor",
    "data/kmeans/kmeans_motion_6.npy": "head.motion_plan_head.motion_anchor",
    "data/kmeans/kmeans_plan_6.npy":   "head.motion_plan_head.plan_anchor",
}


def main():
    sd = torch.load(CKPT, map_location="cpu")
    sd = sd.get("state_dict", sd)
    os.makedirs("data/kmeans", exist_ok=True)
    for path, key in ANCHORS.items():
        arr = sd[key].cpu().numpy()
        np.save(path, arr)
        print(f"saved {path} {arr.shape}")


if __name__ == "__main__":
    main()
