# 산자부 E2E 데이터 스키마 검토 의견서

> **Date:** March 20, 2026
> **From:** KAIST NetAI Lab — Data Lakehouse Team
> **To:** KAIST AXE Lab
> **Re:** 데이터 스키마 초안 (260316) 검토 결과
> **Document:** `카이스트_산자부E2E_데이터스키마_260316`

---

## 1. Overview

We reviewed the proposed E2E data schema draft against our Apache Iceberg lakehouse implementation (Spark 3.5 + Iceberg 1.8, medallion architecture with Bronze/Silver/Gold layers). This document summarizes compatibility issues, ambiguities, and recommendations organized by priority.

Our existing pipeline already ingests 14 tables from a prior version of this schema. We note where the new draft diverges from what we have implemented and where the draft leaves details unspecified that are required for Iceberg table creation.

**Legend:** Items are tagged **[P0]** critical / **[P1]** high / **[P2]** medium / **[P3]** low.

---

## 2. Primary Key Definitions

### 2.1 [P0] Incorrect Single-Column PKs

Several tables mark a single column as PK where a **composite key** is required for row uniqueness:

| Table | Current PK (Draft) | Required Composite PK | Rationale |
|-------|-------------------|----------------------|-----------|
| Camera | `camera_name` | `(clip_id, frame_id, camera_name)` | The same camera name (e.g. `CAM_FRONT`) repeats across every frame and clip |
| Radar | `radar_name` | `(clip_id, frame_id, radar_name)` | Same as Camera |
| Calibration | `sensor_name` | `(clip_id, sensor_name)` | Calibration is per-clip; the same sensor name appears in every clip |
| HDMap | `filename` | `(clip_id, filename)` or `clip_id` alone | Depends on cardinality (see §7) |

**Action requested:** Define composite PKs for these tables in the finalized DBML.

### 2.2 [P0] Tables with No PK Defined

The following tables have no primary key at all. Without PKs, we cannot enforce uniqueness constraints, and Iceberg sort ordering and partition pruning lose effectiveness:

| Table | Recommended PK | Notes |
|-------|---------------|-------|
| Lidar | `(clip_id, frame_id)` | Or `(clip_id, frame_id, lidar_name)` if multi-LiDAR configurations are planned |
| DynamicObject | `(clip_id, frame_id, object_id)` | Requires adding an `object_id` or `object_idx` column |
| Occupancy | `(clip_id, frame_id)` | Assuming one occupancy grid per frame |
| Motion | `(clip_id, frame_id)` | Assuming one motion record per frame |
| EgoMotion | `(clip_id, frame_id)` | `session_id` is derivable via Clip FK |
| Session_EgoMotion | `session_id` | Assuming one summary record per session |

**Action requested:** Add explicit PKs. In particular, DynamicObject needs an `object_id` or `object_idx` column — without it, individual object annotations within the same frame cannot be uniquely identified.

---

## 3. Data Type Issues

### 3.1 [P0] Timestamps Must Be `long` (64-bit), Not `int` (32-bit)

All timestamp fields across Camera, Lidar, Radar, and Frame are typed as `int` in the draft. This is insufficient:

- Unix timestamps in **seconds** overflow 32-bit signed int in 2038
- Autonomous driving datasets conventionally use **microsecond** precision (e.g., nuScenes, Waymo), which exceeds 32-bit range immediately
- Our pipeline uses `LongType` (64-bit) for all timestamps

**Affected fields (8 total):**
- `Camera.system_timestamp`, `Camera.sensor_timestamp`
- `Lidar.system_timestamp`, `Lidar.sensor_timestamp`
- `Radar.system_timestamp`, `Radar.sensor_timestamp`
- `Frame.sensor_timestamps`
- (Any future CAN bus timestamps)

**Action requested:** Change all timestamp fields to `long` / `bigint`.

### 3.2 [P0] `Frame.sensor_timestamps` — Ambiguous Type and Semantics

The draft defines this as `int` (singular), but the field name is plural. Two possible interpretations:

| Interpretation | Expected Type | Semantics |
|---------------|--------------|-----------|
| Single sync reference timestamp | `long` | A canonical timestamp that all sensors are aligned to |
| Per-sensor timestamp array | `List<long>` | One timestamp per sensor in a fixed order |

Our current implementation uses `ArrayType(LongType())` (the array interpretation). **Please clarify the intended semantics and update the type accordingly.**

### 3.3 [P1] Geometric Types Need Concrete Serialization Spec

The draft uses abstract type names (`se3`, `matrix_t`, `vec3`, `quat_t`) that have no standard Parquet/Iceberg mapping. For Iceberg table creation, we need concrete struct definitions. Below is what we currently implement — please confirm or revise:

| Draft Type | Our Iceberg Implementation | Details |
|-----------|---------------------------|---------|
| `se3` | `struct<translation: array<double>, rotation: array<double>>` | translation = `[x, y, z]`, rotation = `[qw, qx, qy, qz]` |
| `matrix_t` | `array<double>` (9 elements) | Row-major 3×3 intrinsic matrix: `[fx, 0, cx, 0, fy, cy, 0, 0, 1]` |
| `vec3` | `struct<x: double, y: double, z: double>` | Named fields for clarity |
| `quat_t` | `struct<qw: double, qx: double, qy: double, qz: double>` | Hamilton convention (scalar-first) |

**Key questions:**
1. **Quaternion convention:** Hamilton (scalar-first: `qw, qx, qy, qz`) or JPL (scalar-last: `qx, qy, qz, qw`)? Mixing conventions silently produces incorrect rotations.
2. **SE3 rotation format:** Quaternion `[qw, qx, qy, qz]` or rotation matrix (9 floats)? The draft says "SE3" which is a 4×4 matrix, but storing the full matrix is wasteful vs. translation + quaternion.
3. **Intrinsics matrix element order:** Row-major or column-major? 3×3 or 3×4 (with distortion)?

**Action requested:** Add a "Type Definitions" appendix to the finalized DBML specifying serialization format, element ordering, and conventions.

### 3.4 [P2] `Clip.date` Should Be a Date Type

Currently `varchar`. Using a proper `date` type enables:
- Iceberg partition pruning on date ranges (e.g., all clips from a specific collection campaign)
- Sort ordering without string comparison issues
- Consistent formatting without manual parsing

**Recommendation:** Change to `date` type with ISO 8601 format (`YYYY-MM-DD`).

---

## 4. Schema Completeness

### 4.1 [P1] Episode Table — New Entity, Integration Details Needed

The `Episode` table and `Frame.episode_id` FK are new additions not present in our current pipeline. We welcome this for cross-clip sequence modeling. To integrate it, we need clarity on:

1. **Frame selection semantics:** Is `Episode.frame_id_list` always a contiguous subsequence of frames from `from_clip_id` through `to_clip_id`? Or can it be an arbitrary selection?
2. **Clip spanning rules:** Can an episode span non-adjacent clips within a session? Across sessions?
3. **Cardinality:** Can a frame belong to multiple episodes simultaneously? (The single `Frame.episode_id` FK implies no — is that intentional?)
4. **Nullability:** Will all frames have an `episode_id`, or only those included in curated episodes? (If the latter, `Frame.episode_id` must be nullable.)

### 4.2 [P1] CAN Bus / Vehicle Signals Table Missing

The directory structure (slide 4) shows `can_bus/` containing `signals.parquet`, but there is no corresponding table in the ER diagram. If vehicle dynamics data (speed, steering angle, yaw rate, IMU, etc.) are part of the dataset, a schema definition is needed. Suggested minimum:

```
Table CANBus {
  clip_id           varchar  [ref: > Clip.clip_id]
  frame_id          varchar  [ref: > Frame.frame_id]
  timestamp         long
  speed             double
  steering_angle    double
  yaw_rate          double
  acceleration_x    double
  acceleration_y    double
  // ... additional signals
}
```

### 4.3 [P2] Occupancy and Motion Tables — No Fields Specified

Beyond `clip_id` and `frame_id`, these tables have no defined columns. We need to know at minimum:
- **Occupancy:** Voxel grid dimensions, resolution, value encoding (binary occupancy? semantic labels?), storage format (dense array? sparse indices?)
- **Motion:** Flow vectors? Trajectory predictions? Per-object or per-voxel?

### 4.4 [P2] Calibration — `camera_intrinsics` Nullable for Non-Camera Sensors

The Calibration table includes `camera_intrinsics` for all sensors, but only cameras have intrinsics. Two options:
- **Option A (current):** Keep as nullable; non-camera rows store `NULL`
- **Option B:** Split into `Calibration` (extrinsics only) and `CameraCalibration` (adds intrinsics)

We currently implement Option A. Please confirm this is acceptable.

### 4.5 [P3] Category Table — Hierarchy and Task Scope

The `Category` table has only a single `category` field. Given that annotations span multiple tasks (det, track, occ, planning) with independent versioning:
- Should categories be task-scoped? (e.g., detection categories vs. planning categories)
- Is a hierarchical taxonomy planned? (e.g., `vehicle > car > sedan`)
- Would a numeric `category_id` be useful for join efficiency?

---

## 5. Versioning and Provenance

### 5.1 [P2] Schema Has No Version Tracking Columns

The directory structure defines an elaborate 4-block versioning format (`v{Base}_{Map}_{Annotation}_{Episode}`), but no table columns capture this information. This creates a gap for Iceberg-based workflows:

- **Iceberg time-travel** tracks *table* snapshots, not dataset-pipeline versions
- Without a `version` or `config_id` column, SQL queries cannot filter by dataset version
- Reproducing a specific training dataset configuration requires directory path parsing, which is fragile

**Recommendation:** Add at minimum these columns:

| Table | Column | Purpose |
|-------|--------|---------|
| Episode | `config_id` | Links to `episode/config.json` (e.g., `"C001"`) |
| Episode | `base_sensor_version` | e.g., `"v1a"` |
| DynamicObject / Occupancy / Motion | `annotation_version` | e.g., `"1.2"` for det, `"2.0"` for occ |
| HDMap | `map_version` | e.g., `"2.1.0"` |

This allows queries like:
```sql
SELECT * FROM episode WHERE config_id = 'C001' AND base_sensor_version = 'v1a'
```

---

## 6. Denormalized List Columns

### 6.1 [P2] `varchar[]` ID Lists Are Redundant with FK Relationships

Three tables carry denormalized ID arrays:
- `Session.clip_id_list` — redundant with `Clip.session_id` FK
- `Clip.frame_id_list` — redundant with `Frame.clip_id` FK
- `Episode.frame_id_list` — redundant with `Frame.episode_id` FK

In an analytical lakehouse context, these are problematic:
- Cannot be used in JOIN predicates without `EXPLODE` / `LATERAL VIEW`
- Cannot benefit from predicate pushdown or Iceberg partition pruning
- Create dual sources of truth that can diverge

**Recommendation:** Keep these for compatibility with file-based tooling if needed, but:
1. Mark them as **non-authoritative** (FK relationships are canonical)
2. Document the consistency contract (are they materialized views? always in sync?)
3. Our pipeline will ignore these columns and join exclusively via FKs

---

## 7. HDMap Relationship Cardinality

### 7.1 [P3] Map-to-Clip Relationship Direction

The draft has `HDMap.clip_id` as FK, implying one map record per clip. But the directory structure shows maps are **regional** (`seoul_gangnam/v1a/2.0.0/`) and **shared across clips**.

If multiple clips from the same region use the same map, the current design forces duplicating map metadata per clip. Consider inverting the relationship:

- **Current (draft):** `HDMap → Clip` (map belongs to a clip)
- **Suggested:** `Clip → HDMap` (clip references a map via `map_id` FK)

This avoids redundancy and aligns with the versioned directory structure where maps exist independently of clips.

---

## 8. Summary of Requested Changes

| # | Priority | Request |
|---|----------|---------|
| 1 | **P0** | Define composite PKs for Camera, Radar, Calibration, HDMap |
| 2 | **P0** | Add PKs to Lidar, DynamicObject, Occupancy, Motion, EgoMotion, Session_EgoMotion |
| 3 | **P0** | Change all timestamp fields from `int` to `long` |
| 4 | **P0** | Clarify `Frame.sensor_timestamps` semantics and type |
| 5 | **P1** | Add concrete serialization spec for geometric types (se3, vec3, quat, matrix) |
| 6 | **P1** | Clarify Episode table semantics (frame selection, spanning, cardinality) |
| 7 | **P1** | Add CAN bus / vehicle signals table |
| 8 | **P2** | Add version/provenance columns (config_id, annotation_version, etc.) |
| 9 | **P2** | Specify Occupancy and Motion table fields |
| 10 | **P2** | Change `Clip.date` from `varchar` to `date` type |
| 11 | **P2** | Clarify denormalized list column contract |
| 12 | **P3** | Consider inverting HDMap ↔ Clip relationship |
| 13 | **P3** | Expand Category table (hierarchy, task scope) |

We are ready to update our Iceberg pipeline once the finalized DBML is provided. Items P0 and P1 are blocking for schema-safe ingestion; P2/P3 are recommendations.

---

*Generated by KAIST NetAI Lab — Data Lakehouse Team*
