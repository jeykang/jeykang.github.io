# KAIST E2E Dataset Ingestion Pipeline

This package provides automated ingestion of the KAIST/MOTIE E2E autonomous driving dataset into an Apache Iceberg lakehouse using the **Medallion Architecture** (Bronze → Silver → Gold).

## Quick Start: Testing with Simulated Data

Since the actual KAIST dataset may not be available yet, you can test the pipeline using **simulated data generated from nuScenes mini**:

```bash
# Step 1: Generate simulated KAIST data from nuScenes
python -m kaist_ingestion.simulate_from_nuscenes \
    --nuscenes-root ./user_data/nuscenes-mini/v1.0-mini \
    --output ./user_data/kaist-simulated

# Step 2: Run the ingestion pipeline (inside Spark container)
docker exec -it spark-iceberg bash -c "
    export KAIST_SOURCE_PATH=/user_data/kaist-simulated
    python -m kaist_ingestion.kaist_runner all
"
```

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                      KAIST Ingestion Pipeline                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐          │
│  │   Source     │    │   Bronze     │    │   Silver     │          │
│  │   (JSON/     │───▶│   Layer      │───▶│   Layer      │          │
│  │   Files)     │    │   (Raw)      │    │   (Clean)    │          │
│  └──────────────┘    └──────────────┘    └──────────────┘          │
│                                                 │                    │
│                                                 ▼                    │
│                                          ┌──────────────┐           │
│                                          │   Gold       │           │
│                                          │   Layer      │           │
│                                          │   (ML-Ready) │           │
│                                          └──────────────┘           │
└─────────────────────────────────────────────────────────────────────┘
```

## Installation

```bash
# Ensure you have PySpark 3.5+ with Iceberg 1.8+ support
pip install pyspark==3.5.5

# The package uses the existing Docker environment in the parent directory
```

## Usage

### Run the Full Pipeline

```bash
# From the Iceberg directory
cd /path/to/Lakehouse/Iceberg

# Set source data path
export KAIST_SOURCE_PATH=/user_data/kaist

# Run inside Spark container
docker exec -it spark-iceberg python -m kaist_ingestion.kaist_runner all
```

### Run Individual Layers

```bash
# Bronze only (raw ingestion)
python -m kaist_ingestion.kaist_runner bronze

# Silver only (transformations)
python -m kaist_ingestion.kaist_runner silver

# Gold only (pre-joined tables)
python -m kaist_ingestion.kaist_runner gold

# Validation only
python -m kaist_ingestion.kaist_runner validate
```

## Configuration

Configuration is managed via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `KAIST_SOURCE_PATH` | `/user_data/kaist` | Path to source data files |
| `AWS_S3_ENDPOINT` | `http://minio:9000` | S3-compatible storage endpoint |
| `AWS_ACCESS_KEY_ID` | `minioadmin` | Storage access key |
| `AWS_SECRET_ACCESS_KEY` | `minioadmin` | Storage secret key |
| `POLARIS_URI` | `http://polaris:8181/api/catalog` | Iceberg REST catalog URI |
| `POLARIS_CATALOG_NAME` | `lakehouse_catalog` | Catalog name in Polaris |

## Data Layers

### Bronze Layer (`kaist_bronze`)

Raw ingestion with 1:1 mapping from source files:
- `session`, `clip`, `frame` - Core hierarchy
- `camera`, `lidar`, `radar` - Sensor data
- `calibration` - Sensor calibration
- `dynamic_object`, `occupancy`, `motion` - Annotations
- `ego_motion`, `session_ego_motion` - Vehicle pose
- `hdmap` - Map references

### Silver Layer (`kaist_silver`)

Cleaned and partitioned for efficient queries:
- Type coercion and null handling
- Partitioning by `clip_id`, `sensor_name`
- Sort ordering for temporal scans

### Gold Layer (`kaist_gold`)

Pre-joined feature tables for ML workloads:

| Table | Use Case | Partition |
|-------|----------|-----------|
| `camera_annotations` | Object detection training | `camera_name` |
| `lidar_with_ego` | SLAM/Localization | `clip_id` |
| `sensor_fusion_frame` | Multi-modal perception | `clip_id` |

## Module Structure

```
kaist_ingestion/
├── __init__.py               # Package initialization
├── config.py                 # Environment and Spark configuration
├── schemas.py                # PySpark schema definitions
├── ingest_bronze.py          # Bronze layer ingestion
├── transform_silver.py       # Silver layer transformations
├── build_gold.py             # Gold table construction
├── validators.py             # Data quality validation
├── kaist_runner.py           # CLI entry point
└── simulate_from_nuscenes.py # Test data generator (nuScenes → KAIST)
```

## Data Simulation Details

The `simulate_from_nuscenes.py` script maps nuScenes structures to KAIST format:

| nuScenes | KAIST | Notes |
|----------|-------|-------|
| `scene` | `Session` + `Clip` | 1 scene = 1 session with 1 clip |
| `sample` | `Frame` | Keyframe with synchronized sensors |
| `sample_data` (CAM_*) | `Camera` | 6 camera channels |
| `sample_data` (LIDAR_TOP) | `Lidar` | Single LiDAR |
| `sample_data` (RADAR_*) | `Radar` | 5 radar channels |
| `calibrated_sensor` | `Calibration` | Extrinsics + intrinsics |
| `sample_annotation` | `DynamicObject` | 3D bounding boxes |
| `ego_pose` | `EgoMotion` | Vehicle pose per frame |
| `log` + `map` | `HDMap` | Map metadata |

## Extending for Ceph

For production deployment with Ceph instead of MinIO:

```python
# Set Ceph-specific endpoint
export AWS_S3_ENDPOINT=http://ceph-rgw:7480
export AWS_REGION=default
```

The S3FileIO configuration is compatible with Ceph RADOS Gateway.

## See Also

- [KAIST_INGESTION_PLAN.md](./KAIST_INGESTION_PLAN.md) - Detailed analysis and design document
- [KAIST_LAKEHOUSE_COMPATIBILITY_REPORT.md](./KAIST_LAKEHOUSE_COMPATIBILITY_REPORT.md) - Compatibility analysis and scalability projections
- Parent [README.md](../README.md) - Lakehouse architecture overview
- [nuscenes_experiment](../nuscenes_experiment/) - nuScenes scalability benchmarks (3 strategies × 4 scale factors)
- [benchmarks/](../benchmarks/) - KAIST AD workload benchmarks (Gold vs. Silver, partition pruning, temporal replay, time travel, column metrics)
