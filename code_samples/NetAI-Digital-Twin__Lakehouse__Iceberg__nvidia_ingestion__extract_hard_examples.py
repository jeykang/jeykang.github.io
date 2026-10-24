"""Host-side step: pick 3 distinct hard-clip examples and extract a frame.

Reads Gold scores (dumped earlier to user_data/gold_top300.json), enriches
with clip_index.parquet + data_collection.parquet (read with pyarrow
directly), buckets by dominant sub-score, picks the top clip from each of
the 3 largest buckets, and extracts a middle-frame JPEG via ffmpeg.
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pyarrow.parquet as pq

ICEBERG_ROOT = Path(os.environ.get("ICEBERG_ROOT",
    "/home/netai/jeykang/NetAI-Digital-Twin/Lakehouse/Iceberg"))
SOURCE = ICEBERG_ROOT / "netai-e2e/nvidia-physicalai-av-subset"
SCORES_JSON = ICEBERG_ROOT / "user_data/gold_top5000.json"
OUT_DIR = ICEBERG_ROOT / "nvidia_ingestion/hard_examples"
OUT_DIR.mkdir(exist_ok=True)


def dominant_key(sub):
    active = {k: v for k, v in sub.items() if v is not None}
    if not active:
        return "unknown"
    return max(active.items(), key=lambda kv: kv[1])[0]


def friendly_reason(sub, hour, country, season):
    """Human-readable description of why this clip is hard."""
    bits = []
    if sub.get("time_of_day", 0) >= 0.7:
        bits.append(f"night driving (hour={hour})")
    elif sub.get("time_of_day", 0) >= 0.4:
        bits.append(f"dusk/dawn (hour={hour})")
    if sub.get("season_geography", 0) >= 0.7:
        bits.append(f"rare region+season ({country}, {season})")
    elif sub.get("season_geography", 0) >= 0.4:
        bits.append(f"uncommon region+season ({country}, {season})")
    if sub.get("sensor_coverage", 0) >= 0.5:
        bits.append("degraded sensor coverage")
    if sub.get("ego_dynamics", 0) >= 0.5:
        bits.append("aggressive ego motion")
    if sub.get("obstacle_density") and sub["obstacle_density"] >= 0.5:
        bits.append("dense obstacles")
    if sub.get("perception") and sub["perception"] >= 0.5:
        bits.append("visually complex scene (YOLO)")
    return "; ".join(bits) if bits else "balanced-high across dimensions"


def load_clip_meta():
    tbl = pq.read_table(SOURCE / "clip_index.parquet",
                        columns=["clip_id", "chunk"])
    clip_to_chunk = {r["clip_id"]: int(r["chunk"]) for r in tbl.to_pylist()}
    dc = pq.read_table(SOURCE / "metadata/data_collection.parquet",
                       columns=["clip_id", "hour_of_day", "country", "month"])
    dc_map = {r["clip_id"]: r for r in dc.to_pylist()}
    return clip_to_chunk, dc_map


def find_mp4(clip_id, chunk):
    camera_root = SOURCE / "camera/camera_front_wide_120fov"
    chunk_dir = camera_root / f"chunk_{chunk:04d}"
    if not chunk_dir.is_dir():
        for cdir in sorted(camera_root.iterdir()):
            cand = cdir / f"{clip_id}.camera_front_wide_120fov.mp4"
            if cand.exists():
                return cand
        return None
    cand = chunk_dir / f"{clip_id}.camera_front_wide_120fov.mp4"
    return cand if cand.exists() else None


def extract_middle_frame(mp4, out_jpg):
    """Save a frame from ~halfway through the clip via decord+cv2."""
    import decord
    import cv2
    try:
        vr = decord.VideoReader(str(mp4), ctx=decord.cpu(0))
        if len(vr) < 2:
            return False
        idx = len(vr) // 2
        frame = vr[idx].asnumpy()  # RGB
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        return cv2.imwrite(str(out_jpg), bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
    except Exception as e:
        print(f"  frame extract failed: {e}")
        return False


def season_from_month(m):
    if m is None:
        return "unknown"
    m = int(m)
    return {12: "winter", 1: "winter", 2: "winter",
            3: "spring", 4: "spring", 5: "spring",
            6: "summer", 7: "summer", 8: "summer",
            9: "autumn", 10: "autumn", 11: "autumn"}.get(m, "unknown")


def main():
    scored = json.loads(SCORES_JSON.read_text())
    print(f"Loaded {len(scored)} scored clips")

    clip_to_chunk, dc_map = load_clip_meta()
    print(f"Loaded {len(clip_to_chunk):,} clip_index rows, "
          f"{len(dc_map):,} data_collection rows")

    # Enrich + bucket
    camera_root = SOURCE / "camera/camera_front_wide_120fov"
    mp4_index = {}
    for cdir in sorted(camera_root.iterdir()):
        if not cdir.is_dir():
            continue
        for f in cdir.iterdir():
            if f.name.endswith(".camera_front_wide_120fov.mp4"):
                cid = f.name.split(".")[0]
                mp4_index[cid] = f
    print(f"Indexed {len(mp4_index):,} front-wide mp4s")

    # For each candidate, compute a "distinctness score" = (top sub-score)
    # minus (mean of the others). Clips with one clearly dominant factor get
    # high distinctness; clips maxed on every dimension get low distinctness.
    cands = []
    for rec in scored:
        if rec["clip_id"] not in mp4_index:
            continue
        detail = json.loads(rec["detail"])
        sub = detail.get("sub_scores", {})
        active = {k: v for k, v in sub.items() if v is not None}
        if len(active) < 2:
            continue
        top_k, top_v = max(active.items(), key=lambda kv: kv[1])
        others = [v for k, v in active.items() if k != top_k]
        distinctness = top_v - (sum(others) / len(others))
        cands.append((top_k, distinctness, rec, detail, sub))

    # For each dimension, pick the clip with highest distinctness on that
    # dimension (still high absolute score thanks to top-300 filter).
    per_dim = {}
    for top_k, dist, rec, detail, sub in cands:
        cur = per_dim.get(top_k)
        if cur is None or dist > cur[1]:
            per_dim[top_k] = (top_k, dist, rec, detail, sub)

    print("\nMost-distinct example per dominant dimension:")
    for k, v in sorted(per_dim.items(), key=lambda kv: -kv[1][1]):
        rec = v[2]
        print(f"  {k:>18s}: dist={v[1]:+.3f} score={rec['difficulty_score']:.4f} "
              f"clip={rec['clip_id']}")

    # Pick the 3 dimensions with the cleanest distinct picks.
    ranked = sorted(per_dim.values(), key=lambda v: -v[1])[:3]

    chosen = []
    for key, dist, rec, detail, sub in ranked:
        cid = rec["clip_id"]
        chunk = clip_to_chunk.get(cid)
        meta = dc_map.get(cid, {})
        hour = meta.get("hour_of_day")
        country = meta.get("country")
        month = meta.get("month")
        season = season_from_month(month)
        mp4 = mp4_index.get(cid)
        out_jpg = OUT_DIR / f"example_{len(chosen)+1}_{key}.jpg"
        ok = extract_middle_frame(mp4, out_jpg) if mp4 else False
        chosen.append({
            "rank": len(chosen) + 1,
            "clip_id": cid,
            "difficulty_score": round(rec["difficulty_score"], 4),
            "dominant_factor": key,
            "sub_scores": {k: round(v, 3) if v is not None else None
                           for k, v in sub.items()},
            "hour_of_day": hour,
            "country": country,
            "month": month,
            "season": season,
            "chunk": chunk,
            "mp4": str(mp4) if mp4 else None,
            "frame_jpg": str(out_jpg) if ok else None,
            "why_hard": friendly_reason(sub, hour, country, season),
        })
        if len(chosen) == 3:
            break

    (OUT_DIR / "examples.json").write_text(json.dumps(chosen, indent=2))
    print("\n=== 3 distinct hard-clip examples ===")
    for ex in chosen:
        print(f"\n[{ex['rank']}] {ex['clip_id']}  (score={ex['difficulty_score']})")
        print(f"    dominant factor : {ex['dominant_factor']}")
        print(f"    sub_scores      : {ex['sub_scores']}")
        print(f"    hour / country  : {ex['hour_of_day']} / {ex['country']}")
        print(f"    month / season  : {ex['month']} / {ex['season']}")
        print(f"    chunk           : {ex['chunk']}")
        print(f"    why hard        : {ex['why_hard']}")
        print(f"    frame           : {ex['frame_jpg']}")
    print(f"\nWrote {OUT_DIR}/examples.json and per-example JPEGs")


if __name__ == "__main__":
    main()
