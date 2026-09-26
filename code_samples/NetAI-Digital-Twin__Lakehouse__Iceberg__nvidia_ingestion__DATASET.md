# Nvidia PhysicalAI Autonomous Vehicles Dataset — Detailed Breakdown

**Source**: [nvidia/PhysicalAI-Autonomous-Vehicles](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles) (HuggingFace, gated)
**Version**: v25.10 (initial release), with partial v26.03 features absent
**Local subset**: 340 chunks of the full ~3,146 chunk dataset
**Total size on disk**: ~13 TB (after extraction)

> **As of 2026-04-29**, this dataset is exposed in Iceberg as the canonical
> KAIST schema (`kaist_schema_v2.dbml`) — 16 tables in `iceberg.nvidia_bronze`:
> Session, Clip, Episode, Frame, Calibration, Camera, Lidar, Radar, CanBus,
> HDMap, Session_EgoMotion, Category, DynamicObject, Occupancy, Motion,
> EgoMotion. See `MEDALLION_PROGRESS.md §11` for the full mapping spec and
> build benchmarks. The sections below describe the underlying source-data
> layout on NFS, which is what the canonical tables reshape from.

---

## 1. Dataset Overview

The Nvidia PhysicalAI AV dataset is one of the largest publicly available multi-sensor autonomous driving datasets:

| Metric | Full Dataset | Our Subset |
|--------|-------------|------------|
| Total clips | 310,895 | ~33,767 (from 340 chunks) |
| Clip duration | ~140 seconds each (egomotion span) | ~140 seconds each |
| Total driving hours | 1,727 hours | ~188 hours |
| Countries | 25 | 25 (represented in subset) |
| Sensor platform | Hyperion 8 / 8.1 | Hyperion 8 / 8.1 |
| Total size | ~133 TB | ~12-14 TB |

Each "clip" is a multi-sensor recording identified by a UUID (e.g., `25cd4769-5dcf-4b53-a351-bf2c5deb6124`). Earlier docs stated 20-second clips, but a direct probe of 32,651 clips' egomotion data (2026-05-04) shows the actual per-clip relative-time span is **~140 seconds** (egomotion `max_ts` p50 = 139,257,000 µs, p99 = 140,192,000 µs). Whether the camera mp4s span the same range or only ~20s of it is not yet confirmed; treat the 140s figure as the egomotion timestamp span and verify against video duration before using.

Each "chunk" is a batch of ~100 clips, packaged as a zip archive per sensor type.

---

## 2. Sensor Modalities

### 2.1 Camera (7 sensors)

**Format**: MP4 video files (H.264, 1080p, 30fps) + timestamp parquet sidecar

**Sensors**:
| Camera | FOV | Coverage (chunks) |
|--------|-----|-------------------|
| `camera_front_wide_120fov` | 120° front | 340/340 |
| `camera_front_tele_30fov` | 30° front telephoto | 340/340 |
| `camera_cross_left_120fov` | 120° left | 340/340 |
| `camera_cross_right_120fov` | 120° right | 340/340 |
| `camera_rear_left_70fov` | 70° rear-left | 340/340 |
| `camera_rear_right_70fov` | 70° rear-right | **307/340** (incomplete) |
| `camera_rear_tele_30fov` | 30° rear telephoto | **175/340** (incomplete) |

**File structure per chunk**:
```
camera/<sensor>/<sensor>.chunk_XXXX/
  ├── <clip_uuid>.<sensor>.mp4              # 20s video at 30fps
  ├── <clip_uuid>.<sensor>.timestamps.parquet    # per-frame timestamps
  └── <clip_uuid>.<sensor>.blurred_boxes.parquet # privacy blur regions
```

**Timestamp parquet schema** (per clip):
- Frame-level timestamps synchronized with video frames
- Used to align camera frames with lidar/radar/egomotion

**Blurred boxes parquet**: Bounding boxes for privacy-blurred regions (faces, license plates).

### 2.2 LiDAR (1 sensor)

**Sensor**: `lidar_top_360fov` — roof-mounted 360° lidar
**Format**: Parquet files with Draco-compressed point cloud blobs
**Coverage**: 340/340 chunks extracted (recovered Apr 2026 after data-loss incident)

**File structure per chunk**:
```
lidar/lidar_top_360fov/lidar_top_360fov.chunk_XXXX/
  └── <clip_uuid>.lidar_top_360fov.parquet    # ~216 MB per file
```

**Schema** (per clip parquet):
- Contains Draco-compressed 3D point cloud data
- ~200 lidar spins per clip (clip duration ~140s based on egomotion span — see §1)
- Decode with DracoPy library to get XYZ + intensity
- Each parquet file is ~216 MB (large due to point cloud density)

**Decoding example**:
```python
import DracoPy
# Read the Draco blob column from parquet
# Decode each blob to get point cloud arrays
compressed = row["lidar_data"]  # binary blob
mesh = DracoPy.decode(compressed)
points = mesh.points  # Nx3 float array
```

### 2.3 Radar (19 sensors)

**Format**: Parquet files with 3D radar detections
**Coverage**: All 19 sensors populated after Apr 2026 recovery. Per-sensor extracted-chunk counts below; the "(of design)" column shows the chunk count expected for that sensor's design coverage (subset has 340 chunks total, but each radar sensor only ships data for a subset of those — typically ~120 for `srr_0`/`mrr_2` sensors and ~134 for `srr_3`/`imaging_lrr_1`/etc.). Gaps relative to design = 49 upstream-pruned zips that are 404 on HuggingFace.

**Sensors** (grouped by position):

| Position | Sensors | Type | Chunks (extracted / design) |
|----------|---------|------|-------------------------|
| Front center | `radar_front_center_srr_0` | Short-range | 120 / 121 |
| Front center | `radar_front_center_mrr_2` | Medium-range | 130 / 134 |
| Front center | `radar_front_center_imaging_lrr_1` | Long-range imaging | 130 / 134 |
| Front left | `radar_corner_front_left_srr_0` | Short-range | 120 / 120 |
| Front left | `radar_corner_front_left_srr_3` | Short-range (alt) | 130 / 134 |
| Front right | `radar_corner_front_right_srr_0` | Short-range | 120 / 121 |
| Front right | `radar_corner_front_right_srr_3` | Short-range (alt) | 130 / 134 |
| Rear left | `radar_corner_rear_left_srr_0` | Short-range | 120 / 121 |
| Rear left | `radar_corner_rear_left_srr_3` | Short-range (alt) | 130 / 134 |
| Rear right | `radar_corner_rear_right_srr_0` | Short-range | 120 / 121 |
| Rear right | `radar_corner_rear_right_srr_3` | Short-range (alt) | 130 / 134 |
| Rear left | `radar_rear_left_mrr_2` | Medium-range | 130 / 134 |
| Rear left | `radar_rear_left_srr_0` | Short-range | 120 / 121 |
| Rear right | `radar_rear_right_mrr_2` | Medium-range | 130 / 134 |
| Rear right | `radar_rear_right_srr_0` | Short-range | 120 / 121 |
| Side left | `radar_side_left_srr_0` | Short-range | 120 / 121 |
| Side left | `radar_side_left_srr_3` | Short-range (alt) | 32 / 36 |
| Side right | `radar_side_right_srr_0` | Short-range | 120 / 121 |
| Side right | `radar_side_right_srr_3` | Short-range (alt) | 32 / 36 |

**Radar config**: Clips have either `"low"` or `"high"` radar configuration, indicated by the `radar_config` field in `sensor_presence.parquet`. Low config has ~9 sensors, high config has up to 19.

**File structure per chunk**:
```
radar/<sensor>/<sensor>.chunk_XXXX/
  └── <clip_uuid>.<sensor>.parquet
```

**Schema**: 3D radar detections with velocity estimates, signal strength, and confidence per detection.

### 2.4 Egomotion (Labels)

**Format**: Parquet files with ego vehicle trajectory
**Coverage**: 340 chunks (recovered from NFS silly-rename files)
**Size**: ~14 GB total (~40 MB per chunk zip)

**File structure per chunk**:
```
labels/egomotion/egomotion.chunk_XXXX/
  └── <clip_uuid>.egomotion.parquet
```

**Schema** (per clip parquet):
- `timestamp` — time in local frame
- `x`, `y`, `z` — position (meters, relative to origin at t=0)
- `qw`, `qx`, `qy`, `qz` — orientation quaternion
- Attitude (yaw, pitch, roll) estimated w.r.t. gravity
- Origin is ego vehicle position at timestamp 0, with 0 yaw

**Use cases**: Ego dynamics analysis, trajectory prediction, lidar-ego fusion.

### 2.5 Calibration

**Format**: Parquet files (one per chunk)
**Coverage**: 340 files each, complete

| Type | File | Description |
|------|------|-------------|
| Camera intrinsics | `calibration/camera_intrinsics/` | Focal length, principal point, distortion coefficients per camera |
| Sensor extrinsics | `calibration/sensor_extrinsics/` | 6DoF transforms between sensors and ego vehicle frame |
| Vehicle dimensions | `calibration/vehicle_dimensions/` | Vehicle length, width, height, wheelbase |

---

## 3. Metadata

### 3.1 `clip_index.parquet` (310,895 rows)

Top-level clip registry. One row per clip.

| Column | Type | Description |
|--------|------|-------------|
| `clip_id` | string | UUID, primary key (e.g., `25cd4769-5dcf-4b53-...`) |
| `chunk` | int64 | Chunk number (0-3145 full dataset, subset uses specific chunks) |
| `split` | string | `train`, `val`, or `test` |
| `clip_is_valid` | bool | Data validity flag |

### 3.2 `metadata/data_collection.parquet` (310,895 rows)

Per-clip collection context. Joined to clip_index by clip_id.

| Column | Type | Description | Example values |
|--------|------|-------------|----------------|
| `clip_id` | string | UUID, join key | |
| `country` | string | Collection country | "United States", "Germany", "Finland", ... (25 countries) |
| `month` | int64 | Month of year (1-12) | 5, 8, 12 |
| `hour_of_day` | int64 | Hour (0-23) | 17, 14, 3 |
| `platform_class` | string | Sensor platform | "hyperion_8", "hyperion_8.1" |

### 3.3 `metadata/sensor_presence.parquet` (310,895 rows)

Per-clip sensor availability matrix. Boolean flag per sensor.

| Column | Type | Description |
|--------|------|-------------|
| `clip_id` | string | UUID, join key |
| `camera_cross_left_120fov` | bool | Camera present (True for all clips) |
| `camera_cross_right_120fov` | bool | (same for all 7 cameras) |
| ... | ... | |
| `lidar_top_360fov` | bool | LiDAR present |
| `radar_corner_front_left_srr_0` | bool | Radar sensor present |
| ... | ... | (19 radar sensors total) |
| `radar_config` | string | `"low"` or `"high"` radar suite |

### 3.4 `selected_chunks.csv` (340 rows)

Metadata for each chunk in our subset.

| Column | Type | Description | Example values |
|--------|------|-------------|----------------|
| `chunk` | int | Chunk number | 3, 10, 15, ... |
| `country` | string | Predominant country | "United States", "Germany" |
| `season` | string | Season during collection | "winter", "spring", "summer", "fall" |
| `hour_bin` | string | Time-of-day bin | "morning", "afternoon", "evening", "night" |
| `platform` | string | Sensor platform | "hyperion_8", "hyperion_8.1" |
| `n_clips` | int | Clips in this chunk | ~100 |
| `split` | string | Dataset split | "train", "val", "test" |

**Distribution of our 340-chunk subset:**
- **Countries**: 25 (all represented)
- **Seasons**: winter, spring, summer, fall
- **Time bins**: morning, afternoon, evening, night
- **Platforms**: hyperion_8, hyperion_8.1
- **Total clips**: 33,767

---

## 4. Data Not Present (v26.03 Additions)

The following data was added in the March 2026 update and is **not yet downloaded**:

| Data | Path | Size (est.) | Description |
|------|------|-------------|-------------|
| Obstacle labels | `labels/obstacle.offline/` | ~50-100 GB | Machine-generated 3D obstacle detections (not ground truth). Per-frame bounding boxes with class labels. |
| Offline egomotion | `labels/egomotion.offline/` | ~14 GB | Signal-processing-optimized ego motion (smoother than online version) |
| Reasoning labels | `reasoning/ood_reasoning.parquet` | ~10 MB | Human-verified OOD reasoning labels for 1,740 clips (Chain of Causation annotations) |
| Offline calibration | `calibration/camera_intrinsics.offline/`, `lidar_intrinsics.offline/`, `sensor_extrinsics.offline/` | ~5 GB | Offline-optimized calibration for NuRec reconstruction |
| Feature presence | `metadata/feature_presence.parquet` | ~12 MB | Updated sensor presence with per-clip offline feature flags |

**Download command** (requires HuggingFace token):
```bash
hf download nvidia/PhysicalAI-Autonomous-Vehicles --repo-type dataset \
  --include "labels/obstacle.offline/*" \
  --include "labels/egomotion.offline/*" \
  --include "reasoning/*" \
  --local-dir /path/to/dataset
```

---

## 5. Data Relationships

```
clip_index.parquet ──(clip_id)──┬── data_collection.parquet
                                ├── sensor_presence.parquet
                                ├── labels/egomotion/<chunk>/<clip_id>.egomotion.parquet
                                ├── lidar/<sensor>/<chunk>/<clip_id>.lidar_top_360fov.parquet
                                ├── radar/<sensor>/<chunk>/<clip_id>.<sensor>.parquet
                                └── camera/<sensor>/<chunk>/<clip_id>.<sensor>.mp4
                                                           <clip_id>.<sensor>.timestamps.parquet
                                                           <clip_id>.<sensor>.blurred_boxes.parquet

selected_chunks.csv ──(chunk)── clip_index.parquet

calibration/ ──(per chunk, no clip_id)── global sensor parameters
```

**Join key**: `clip_id` (UUID string) is the universal join key across all sensor data and metadata.

**Chunk**: Groups of ~100 clips. Each chunk corresponds to one zip archive per sensor. The chunk number links `clip_index.chunk` to `selected_chunks.csv.chunk`.

---

## 6. Key Limitations

1. **No 3D bounding box ground truth** — obstacle labels (v26.03) are machine-generated, not human-annotated. Cannot do standard mAP evaluation against ground truth.

2. **Lidar is Draco-compressed** — raw XYZ point clouds are not directly queryable via SQL. Requires DracoPy decoding before use.

3. **Camera data is video, not frames** — MP4 files need frame extraction to align with lidar timestamps. Timestamp parquets provide the frame-timestamp mapping.

4. **Radar coverage varies** — clips with `radar_config="low"` have ~9 sensors; `"high"` has up to 19. Not all radar sensors are present for all clips.

5. **Two camera sensors incomplete** — `camera_rear_right_70fov` (307/340 chunks present, 122 with non-zero parquets) and `camera_rear_tele_30fov` (175/340 chunks present, 2 with non-zero parquets) have gaps in our subset. These were not part of the Apr 2026 redownload (radar+lidar only); a future camera-recovery pass would address them.

7. **49 upstream-pruned radar zips** — During the Apr 2026 redownload, 49 zips listed in `transfer_manifest.json` returned `404` from HuggingFace (file deleted upstream). All are radar (most: `chunk_1057`, `chunk_3109`). Affects ~0.6% of expected radar parquets; no per-clip impact has been characterised yet — handled by Silver `missing_sensors` quality check.

6. **Subset is 340/3146 chunks** — approximately 11% of the full dataset. The subset was selected to cover all 25 countries, all seasons, and all time bins.

---

## 7. Scale Reference

Canonical Bronze row counts (post-recovery + canonical reshape, 2026-04-29):

| Canonical table | Rows | Notes |
|-----------------|------|-------|
| Session | 3,116 | one per chunk in clip_index (full dataset) |
| Clip | 310,895 | one per clip |
| Calibration | 458,873 | per (clip_id, sensor_name) |
| Camera | 109,171,395 | per-frame metadata across 7 cameras |
| Lidar | 6,164,244 | per-spin metadata across 340 chunks |
| Radar | 11,730,962,796 | per-detection across 19 radar sensors |
| EgoMotion | 101,745,981 | per-timestamp ego state |
| Frame | 257,290,851 | distinct (clip_id, sensor_timestamp) |
| Episode, CanBus, HDMap, Session_EgoMotion, Category, DynamicObject, Occupancy, Motion | 0 each | empty (no Nvidia source) |
| **TOTAL** | **12,206,108,151** | across 16 canonical tables |

**Query performance** (from benchmark):
- Iceberg `count(*)` on 2.17B rows: **129ms** (O(1) via manifest metadata)
- Egomotion count: **439ms**
- Silver view creation (31 views): **14.1s**
- Gold view creation (3 views): **7.6s**
