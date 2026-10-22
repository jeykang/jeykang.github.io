#!/usr/bin/env python3
"""Plan an evaluation slice that OVERLAPS NuRec, so MF-PDMS and AlpaSim score the
same clips.

The first slice was planned from on-disk chunk density and hit **zero** of the 1,607
NuRec scenes — NuRec covers ~0.5% of the 306,152-clip dataset, so a blind draw was
never going to intersect it. Without overlap the per-clip MF-PDMS/closed-loop
correlation (the entire reason for adding AlpaSim) is impossible, and only an
aggregate comparison remains — which NVIDIA's published scores already provide.

So selection is inverted: start from the NuRec scene list, map each scene to its
dataset chunk, and take the densest chunks. NuRec is strongly chunk-clustered (the top
6 chunks hold ~143 scenes), so overlap is cheap to obtain.

    python plan_nurec_slice.py --chunks 6 --cameras 1
"""
from __future__ import annotations

import argparse, collections, json, os, re

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", type=int, default=6)
    ap.add_argument("--max-clips", type=int, default=0, help="0 = all in chosen chunks")
    ap.add_argument("--cameras", type=int, choices=(1, 4, 7), default=1,
                    help="1 = front_wide only (enough for VaVAM + ego baselines)")
    ap.add_argument("--out", default=os.path.join(HERE, "slice_manifest_nurec.json"))
    a = ap.parse_args()

    from huggingface_hub import HfApi, hf_hub_download
    import pyarrow.parquet as pq

    files = HfApi().list_repo_files("nvidia/PhysicalAI-Autonomous-Vehicles-NuRec",
                                    repo_type="dataset")
    nurec = set(re.findall(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
                           "\n".join(files)))
    idx = hf_hub_download("nvidia/PhysicalAI-Autonomous-Vehicles", "clip_index.parquet",
                          repo_type="dataset")
    t = pq.read_table(idx, columns=["clip_id", "chunk"]).to_pydict()
    chunk_of = {c: f"{int(ch):04d}" for c, ch in zip(t["clip_id"], t["chunk"])}

    hit = {c: chunk_of[c] for c in nurec if c in chunk_of}
    by_chunk = collections.defaultdict(list)
    for c, ch in hit.items():
        by_chunk[ch].append(c)
    ranked = sorted(by_chunk, key=lambda ch: -len(by_chunk[ch]))[:a.chunks]

    clips, per = [], {}
    for ch in ranked:
        sel = sorted(by_chunk[ch])
        per[ch] = sel
        clips += sel
    if a.max_clips:
        clips = sorted(clips)[:a.max_clips]
        per = {ch: [c for c in v if c in set(clips)] for ch, v in per.items()}

    cams = {1: ["camera_front_wide_120fov"],
            4: ["camera_cross_left_120fov", "camera_front_wide_120fov",
                "camera_cross_right_120fov", "camera_front_tele_30fov"],
            7: ["camera_cross_left_120fov", "camera_front_wide_120fov",
                "camera_cross_right_120fov", "camera_rear_left_70fov",
                "camera_rear_tele_30fov", "camera_rear_right_70fov",
                "camera_front_tele_30fov"]}[a.cameras]
    man = {"chunks": ranked, "clips_by_chunk": per, "clips": sorted(clips),
           "cameras": cams, "nurec_aligned": True,
           "estimated_download_gb": round(len(ranked) * (1.4 * len(cams) + 0.08), 1)}
    json.dump(man, open(a.out, "w"), indent=1)
    print(f"[plan] {len(clips)} NuRec-aligned clips across {len(ranked)} chunks -> {a.out}")
    print(f"[plan] chunks: {', '.join(ranked)}")
    print(f"[plan] per-chunk NuRec counts: {[len(per[c]) for c in ranked]}")
    print(f"[plan] cameras: {len(cams)}  estimated download ~{man['estimated_download_gb']} GB")
    print("[plan] every clip here is BOTH MF-PDMS-scorable and AlpaSim-scorable")


if __name__ == "__main__":
    main()
