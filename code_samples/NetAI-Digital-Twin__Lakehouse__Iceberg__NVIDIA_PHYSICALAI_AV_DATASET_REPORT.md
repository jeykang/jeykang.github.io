# NVIDIA PhysicalAI Autonomous Vehicles Dataset -- Comprehensive Analysis Report

**Date:** 2026-03-24
**Dataset Location:** `/mnt/datax/hub/datasets--nvidia--PhysicalAI-Autonomous-Vehicles/`
**Dataset Version:** Snapshot `0c8e5b78` (HuggingFace Hub format)
**Total Size:** ~90 TB on disk (~100 TB claimed by NVIDIA)

---

## 1. Executive Summary

The NVIDIA PhysicalAI Autonomous Vehicles dataset is one of the largest multi-sensor driving datasets ever released, comprising **1,727 hours** of driving data across **310,895 clips** (each 20 seconds) from **25 countries** and **2,500+ cities**. At ~100 TB, it dwarfs existing public AD datasets in raw sensor volume.

However, **the dataset ships with virtually no perception annotations**. The only "labels" present are **egomotion** (GPS/IMU-derived vehicle pose trajectories). Object-level annotations (3D bounding boxes, segmentation, lane markings, etc.) are explicitly listed as **"COMING SOON"** in the README and do not exist in this snapshot. This makes the dataset unsuitable for supervised perception tasks in its current form -- which is likely the basis for the concern raised by your organization.

---

## 2. Sensor Data Inventory

### 2.1 Camera (7 views, ~40-45 TB)

| Camera | Field of View | Chunks | Per-Chunk Size |
|--------|--------------|--------|----------------|
| `camera_front_wide_120fov` | 120 deg, forward | 3,116 | ~1.9-2.0 GB |
| `camera_front_tele_30fov` | 30 deg, forward tele | 3,116 | ~1.6 GB |
| `camera_cross_left_120fov` | 120 deg, left cross | 3,116 | ~2.2 GB |
| `camera_cross_right_120fov` | 120 deg, right cross | 3,116 | ~2.2 GB |
| `camera_rear_left_70fov` | 70 deg, rear left | 3,116 | ~similar |
| `camera_rear_right_70fov` | 70 deg, rear right | 3,116 | ~similar |
| `camera_rear_tele_30fov` | 30 deg, rear tele | 3,116 | ~1.9 GB |

- **Format:** ZIP archives (store-compressed) containing **1080p MP4 video at 30 fps** + per-frame timestamp parquet files
- **File naming:** `<clip_uuid>.camera_<view>.mp4`
- **Coverage:** All 310,895 clips have all 7 camera views
- **Estimated total:** ~40-45 TB

### 2.2 LiDAR (1 sensor, ~62 TB)

| Sensor | Chunks | Per-Chunk Size |
|--------|--------|----------------|
| `lidar_top_360fov` | 3,116 | ~20-21 GB |

- **Format:** ZIP archives containing parquet files with **Draco-encoded 3D point clouds**
- **Schema:** `spin_index` (int64), `reference_timestamp` (int64, microseconds), `draco_encoded_pointcloud` (binary)
- **Capture rate:** 10 Hz (200 spins per 20-second clip)
- **Decoding:** Requires DracoPy library for point cloud extraction
- **Coverage:** All 310,895 clips
- **Estimated total:** ~62 TB (dominates dataset size)

### 2.3 Radar (up to 10 physical positions, 19 sensor modes, ~1-2 TB)

| Radar Configuration | Sensor Variants | Chunks per Variant | Clips Covered |
|--------------------|----------------|-------------------|---------------|
| **Low** (base) | 5x `srr_0` (corners + front center) | 970 each | ~97,000 |
| **Medium** | +4x `mrr_2`/`lrr_1` (front center + rear) | 956 each | ~95,600 |
| **High** | +4x `srr_3` (corners) + 2x `srr_3` (sides) | 298-956 each | varies |

- **Format:** ZIP archives containing parquet files with radar point cloud data
- **Schema per detection:** `scan_index`, `timestamp`, `sensor_timestamp`, `num_returns`, `doppler_ambiguity`, `max_returns`, `detection_index`, `radar_model`, `azimuth`, `elevation`, `distance`, `radial_velocity`, `rcs` (radar cross-section), `snr`, `exist_probb`
- **Coverage:** 163,850 of 310,895 clips (52.7%) have radar data; configuration varies by vehicle platform
- **Estimated total:** ~1-2 TB

### 2.4 Sensor Coverage Summary

| Sensor Type | Clips Covered | Coverage Rate |
|-------------|--------------|---------------|
| All 7 cameras | 310,895 | 100% |
| LiDAR (top 360) | 310,895 | 100% |
| Radar (any) | 163,850 | 52.7% |
| Radar (full "high" config) | ~29,800 | ~9.6% |

---

## 3. Calibration Data

### 3.1 Camera Intrinsics (3,116 parquet files, ~37 KB each)

| Column | Type | Description |
|--------|------|-------------|
| `clip_id` | string | Clip UUID |
| `camera_name` | string | Camera identifier |
| `width`, `height` | int | Image dimensions (1920x1080) |
| `cx`, `cy` | float | Principal point |
| `bw_poly_0..4` | float | F-theta backward polynomial (5 coefficients) |
| `fw_poly_0..4` | float | F-theta forward polynomial (5 coefficients) |

**Note:** Uses f-theta lens model (not standard pinhole), common for wide-angle automotive cameras.

### 3.2 Sensor Extrinsics (3,116 parquet files, ~63 KB each)

- Quaternion rotation (`qx`, `qy`, `qz`, `qw`) and translation (`x`, `y`, `z`) for all sensors
- Reference frame: center of rear axle projected to ground plane
- Covers all 7 cameras, 1 LiDAR, and up to 10 radar sensors per clip

### 3.3 Vehicle Dimensions (3,116 parquet files, ~9 KB each)

- `length`, `width`, `height`, `rear_axle_to_bbox_center`, `wheelbase`, `track_width` (meters)
- Per-clip (varies by vehicle platform: Hyperion 8 and 8.1)

---

## 4. Labels and Annotations -- THE CRITICAL GAP

### 4.1 What Actually Exists: Egomotion Only

**Location:** `labels/egomotion/` -- 3,116 ZIP chunks (~36 MB each, ~112 GB total)

| Column | Type | Description |
|--------|------|-------------|
| `timestamp` | int64 | Absolute timestamp (microseconds) |
| `qx`, `qy`, `qz`, `qw` | float64 | Quaternion orientation |
| `x`, `y`, `z` | float64 | Position in local world frame (meters) |
| `vx`, `vy`, `vz` | float64 | Velocity in world frame (m/s) |
| `ax`, `ay`, `az` | float64 | Acceleration in world frame (m/s^2) |
| `curvature` | float64 | Path curvature (1/meters) |

- This is the **ego vehicle's 6-DoF pose trajectory** with full kinematics
- Derived automatically from **GPS/INS sensors** -- not human-annotated
- ~200 timestamped poses per clip (10 Hz over 20 seconds)
- Local coordinate frame origin at vehicle position at t=0

### 4.2 What Is Promised But Missing

The README (line 531) explicitly states:

> **Objects and Road Elements:** (COMING SOON)

The README also mentions "autogenerated (non-GT) machine labels" as a feature, but the only labels actually shipped are egomotion. No files with names containing "bbox", "3d_box", "segmentation", "semantic", "instance", "lane", "object", "detection", "tracking", or "annotation" exist anywhere in the 54,372-file dataset tree.

### 4.3 Annotation Gap vs. Standard AD Datasets

| Annotation Type | This Dataset | nuScenes | Waymo Open | KITTI | Argoverse 2 |
|----------------|-------------|----------|------------|-------|-------------|
| **3D bounding boxes** (vehicles, pedestrians, cyclists) | MISSING | Yes | Yes | Yes | Yes |
| **2D bounding boxes** | MISSING | Derived | Yes | Yes | Derived |
| **Object tracking IDs** | MISSING | Yes | Yes | Limited | Yes |
| **Semantic segmentation** (per-point or per-pixel) | MISSING | Yes (lidarseg) | Yes | Yes | No |
| **Instance segmentation** | MISSING | Yes (panoptic) | Yes | No | No |
| **Lane markings / drivable area** | MISSING | No | Yes | No | Yes (HD maps) |
| **Traffic signs/lights** | MISSING | No | Yes | No | No |
| **HD map data** | MISSING | Yes | No | No | Yes |
| **Depth maps** | MISSING | No | No | Yes | No |
| **Ego motion / vehicle pose** | **YES** | Yes | Yes | Yes | Yes |
| **Sensor calibration** | **YES** | Yes | Yes | Yes | Yes |

---

## 5. Metadata

### 5.1 data_collection.parquet (12 MB, 310,895 rows)

| Column | Type | Description |
|--------|------|-------------|
| `clip_id` | string | UUID |
| `country` | string | Country code |
| `month` | int | 1-12 |
| `hour_of_day` | int | 0-23 |
| `platform_class` | string | `hyperion_8` or `hyperion_8.1` |

### 5.2 sensor_presence.parquet (11 MB, 310,895 rows)

- `clip_id` + 27 boolean columns (one per sensor: 7 cameras, 1 LiDAR, 19 radar variants)
- `radar_config` column: `'NA'`, `'low'`, `'med'`, or `'high'`

### 5.3 clip_index.parquet (11 MB)

- Maps clip UUIDs to chunk numbers across all sensor directories
- Enables selective download of specific clips

---

## 6. Geographic and Temporal Distribution

- **25 countries**, ~50% US and ~50% from 24 EU countries
- **2,500+ cities**
- Temporal metadata includes month and hour-of-day, enabling analysis of seasonal and time-of-day diversity
- Two vehicle platforms: Hyperion 8 and Hyperion 8.1

---

## 7. Data Format and Access Architecture

- **Storage:** HuggingFace Hub format with content-addressed blob store (SHA-256)
- **Chunking:** All sensor data organized into ZIP/parquet chunks of ~100 clips each (3,116 chunks)
- **Selective download:** Chunk architecture enables downloading by sensor type, geography, or split without fetching the entire 100 TB
- **Compression:** Camera data stored as MP4 (H.264/H.265); LiDAR uses Draco encoding; ZIPs use store method (no additional compression)
- **License:** NVIDIA Autonomous Vehicle Dataset License -- restricts to AV/ADAS use with NVIDIA technology; no surveillance, identification, or redistribution; 12-month expiry from download

---

## 8. Suitability Assessment for Autonomous Driving Tasks

### 8.1 Tasks the Dataset IS Suitable For

| Task | Suitability | Rationale |
|------|------------|-----------|
| **Self-supervised pretraining** | Excellent | Massive scale (1,727 hrs, 7 cameras, LiDAR) provides rich unlabeled data for contrastive learning, masked prediction, etc. |
| **End-to-end driving (imitation learning)** | Good | Egomotion provides the ego trajectory as a supervision signal; camera/LiDAR provide perception inputs. This is the stated design purpose. |
| **Neural scene reconstruction** (NeRF, 3DGS) | Good | Multi-camera + LiDAR + calibration + egomotion provide the necessary inputs |
| **Ego-trajectory prediction / planning** | Good | Rich egomotion data with velocity, acceleration, and curvature |
| **Sensor fusion research** | Good | Synchronized camera + LiDAR + radar with full calibration |
| **Domain adaptation / transfer** | Excellent | Geographic diversity across 25 countries provides strong domain shift signal |
| **Foundation model pretraining** | Excellent | Scale and sensor diversity are ideal for large-scale representation learning |

### 8.2 Tasks the Dataset is NOT Suitable For (Current Snapshot)

| Task | Suitability | Reason |
|------|------------|--------|
| **3D object detection** | Not suitable | No bounding box annotations of any kind |
| **Object tracking / MOT** | Not suitable | No object annotations or track IDs |
| **Semantic segmentation** | Not suitable | No pixel/point-level semantic labels |
| **Instance / panoptic segmentation** | Not suitable | No instance labels |
| **Lane detection** | Not suitable | No lane annotations |
| **Traffic sign/light recognition** | Not suitable | No traffic element annotations |
| **Occupancy prediction** (supervised) | Not suitable | No occupancy ground truth |
| **Motion forecasting** (of other agents) | Not suitable | No annotations of other agents' trajectories |
| **Behavior prediction** | Not suitable | No labeled agent behaviors |
| **Open-loop perception benchmarking** | Not suitable | Cannot compute standard metrics (mAP, NDS, etc.) without GT |

### 8.3 Nuanced Assessment

The dataset's design philosophy is clear from the README: it targets **end-to-end driving** (sensor-in, trajectory-out) and **Physical AI** pretraining, where massive unlabeled sensor data is the primary value. This is a legitimate and increasingly important paradigm (e.g., NVIDIA's own DRIVE Thor platform, UniAD, and similar end-to-end architectures).

However, for organizations whose AD stack relies on **modular perception pipelines** (detect -> track -> predict -> plan), this dataset provides no value for training or evaluating the perception components. The egomotion-only labels mean you cannot:

1. Train a 3D detector (no box labels)
2. Train a segmentation network (no semantic labels)
3. Evaluate perception performance (no ground truth for metrics)
4. Benchmark against other datasets (no common annotation format)

The README's mention of "machine labels" and "autogenerated (non-GT) labels" is misleading in the current snapshot -- these appear to refer to planned future releases that have not materialized as of this dataset version.

---

## 9. Comparison with Established AD Datasets

| Property | NVIDIA PhysicalAI | nuScenes | Waymo Open | Argoverse 2 |
|----------|-------------------|----------|------------|-------------|
| **Hours** | 1,727 | 5.5 | 6.4 (v2) | 4.2 |
| **Clips/Scenes** | 310,895 | 1,000 | 1,150 | 1,000 |
| **Countries** | 25 | 1 (SG/US) | 1 (US) | 1 (US) |
| **Cameras** | 7 | 6 | 5 | 7 |
| **LiDAR** | 1 (Draco) | 1 | 5 (merged) | 2 |
| **Radar** | Up to 10 | 5 (v1.0) | No | No |
| **3D boxes** | No | 1.4M | 12M+ | 4.2M |
| **Segmentation** | No | Yes | Yes | No |
| **HD maps** | No | Yes | No | Yes |
| **Ego pose** | Yes | Yes | Yes | Yes |
| **Total size** | ~100 TB | ~1.4 TB | ~2.5 TB | ~1 TB |
| **Primary use** | Pretraining / E2E | Full-stack AD | Full-stack AD | Forecasting |

The NVIDIA dataset is **300x larger** than nuScenes in hours and clips, but provides **zero perception labels** compared to nuScenes' 1.4M annotated 3D bounding boxes.

---

## 10. Conclusions and Recommendations

### Your Organization's Assessment Is Correct (With Nuance)

The claim that the dataset is "inadequate for autonomous driving due to a lack of appropriate annotations" is **factually accurate for supervised perception tasks**. The dataset contains:

- **Rich sensor data:** 7 cameras (1080p@30fps), 360-degree LiDAR (10Hz), up to 10 radars, full calibration -- all at massive scale
- **Minimal annotations:** Only egomotion (vehicle pose), which is sensor-derived, not human-annotated
- **No perception ground truth:** Zero 3D boxes, zero segmentation maps, zero lane labels, zero traffic element annotations

### However, "Inadequate" Depends on the Use Case

If the intended use is **supervised perception training or benchmarking**, the dataset is indeed inadequate -- you cannot train or evaluate a single detector, segmentor, or tracker with it.

If the intended use is **self-supervised pretraining, end-to-end driving research, or foundation model development**, the dataset is arguably the most valuable public resource available due to its unprecedented scale and geographic diversity.

### Potential Mitigation Strategies

If your organization still wishes to leverage this dataset's scale:

1. **Auto-labeling pipeline:** Run a pretrained 3D detector (e.g., CenterPoint, TransFusion) on the LiDAR data to generate pseudo-labels, then refine with a human-in-the-loop process
2. **Foundation model pretraining:** Use the dataset for self-supervised pretraining, then fine-tune on a smaller labeled dataset (nuScenes, Waymo Open)
3. **Monitor NVIDIA releases:** The "Objects and Road Elements (COMING SOON)" labels may be released in a future snapshot, which would dramatically change the dataset's utility
4. **Hybrid approach:** Use this dataset for representation learning and domain adaptation, combined with labeled datasets for supervised heads

---

*Report generated by analyzing the full on-disk contents of the NVIDIA PhysicalAI Autonomous Vehicles dataset (snapshot 0c8e5b78, ~90 TB on NFS).*
