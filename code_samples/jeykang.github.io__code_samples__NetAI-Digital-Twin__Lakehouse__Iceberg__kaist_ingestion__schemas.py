"""
PySpark schema definitions for KAIST E2E dataset tables.

These schemas are used for:
1. Type-safe ingestion from JSON/Parquet sources
2. Schema validation
3. Iceberg table creation with explicit types
"""

from pyspark.sql.types import (
    ArrayType,
    DoubleType,
    IntegerType,
    LongType,
    StringType,
    StructField,
    StructType,
)


# =============================================================================
# Complex Types (Reusable Components)
# =============================================================================

# SE3 Pose: 4x4 transformation matrix or translation + rotation
SE3Type = StructType([
    StructField("translation", ArrayType(DoubleType()), nullable=False),  # [x, y, z]
    StructField("rotation", ArrayType(DoubleType()), nullable=False),     # [qw, qx, qy, qz] quaternion
])

# 3x3 Camera intrinsic matrix (flattened row-major)
Matrix3x3Type = ArrayType(DoubleType())

# Quaternion rotation
QuaternionType = StructType([
    StructField("qw", DoubleType(), nullable=False),
    StructField("qx", DoubleType(), nullable=False),
    StructField("qy", DoubleType(), nullable=False),
    StructField("qz", DoubleType(), nullable=False),
])

# 3D Translation
Translation3DType = StructType([
    StructField("x", DoubleType(), nullable=False),
    StructField("y", DoubleType(), nullable=False),
    StructField("z", DoubleType(), nullable=False),
])

# 3D Bounding Box
Box3DType = StructType([
    StructField("center_x", DoubleType(), nullable=False),
    StructField("center_y", DoubleType(), nullable=False),
    StructField("center_z", DoubleType(), nullable=False),
    StructField("length", DoubleType(), nullable=False),
    StructField("width", DoubleType(), nullable=False),
    StructField("height", DoubleType(), nullable=False),
    StructField("yaw", DoubleType(), nullable=False),
])


# =============================================================================
# Core Hierarchy Tables
# =============================================================================

SessionSchema = StructType([
    StructField("session_id", StringType(), nullable=False),
    StructField("session_name", StringType(), nullable=True),
    StructField("clip_id_list", ArrayType(StringType()), nullable=True),
])

ClipSchema = StructType([
    StructField("clip_id", StringType(), nullable=False),
    StructField("session_id", StringType(), nullable=False),
    StructField("clip_idx", IntegerType(), nullable=False),
    StructField("frame_id_list", ArrayType(StringType()), nullable=True),
    StructField("date", StringType(), nullable=True),
])

FrameSchema = StructType([
    StructField("frame_id", StringType(), nullable=False),
    StructField("clip_id", StringType(), nullable=False),
    StructField("frame_idx", IntegerType(), nullable=False),
    StructField("sensor_timestamps", ArrayType(LongType()), nullable=True),
])


# =============================================================================
# Sensor Tables
# =============================================================================

CalibrationSchema = StructType([
    StructField("clip_id", StringType(), nullable=False),
    StructField("sensor_name", StringType(), nullable=False),
    StructField("extrinsics", SE3Type, nullable=True),
    StructField("camera_intrinsics", Matrix3x3Type, nullable=True),  # Only for cameras
])

CameraSchema = StructType([
    StructField("frame_id", StringType(), nullable=False),
    StructField("clip_id", StringType(), nullable=False),
    StructField("system_timestamp", LongType(), nullable=True),
    StructField("sensor_timestamp", LongType(), nullable=True),
    StructField("camera_name", StringType(), nullable=False),
    StructField("filename", StringType(), nullable=False),
])

LidarSchema = StructType([
    StructField("frame_id", StringType(), nullable=False),
    StructField("clip_id", StringType(), nullable=False),
    StructField("system_timestamp", LongType(), nullable=True),
    StructField("sensor_timestamp", LongType(), nullable=True),
    StructField("filename", StringType(), nullable=False),
])

RadarSchema = StructType([
    StructField("frame_id", StringType(), nullable=False),
    StructField("clip_id", StringType(), nullable=False),
    StructField("system_timestamp", LongType(), nullable=True),
    StructField("sensor_timestamp", LongType(), nullable=True),
    StructField("radar_name", StringType(), nullable=False),
    StructField("filename", StringType(), nullable=False),
])


# =============================================================================
# Annotation Tables
# =============================================================================

CategorySchema = StructType([
    StructField("category", StringType(), nullable=False),
])

DynamicObjectSchema = StructType([
    StructField("frame_id", StringType(), nullable=False),
    StructField("clip_id", StringType(), nullable=False),
    StructField("boxes_3d", ArrayType(DoubleType()), nullable=True),
    StructField("category", StringType(), nullable=True),  # Added for convenience
])

OccupancySchema = StructType([
    StructField("frame_id", StringType(), nullable=False),
    StructField("clip_id", StringType(), nullable=False),
    StructField("occupancy_data", StringType(), nullable=True),  # Placeholder - actual format TBD
])

MotionSchema = StructType([
    StructField("frame_id", StringType(), nullable=False),
    StructField("clip_id", StringType(), nullable=False),
    StructField("motion_data", StringType(), nullable=True),  # Placeholder - actual format TBD
])

EgoMotionSchema = StructType([
    StructField("frame_id", StringType(), nullable=False),
    StructField("clip_id", StringType(), nullable=False),
    StructField("session_id", StringType(), nullable=True),
    StructField("translation", Translation3DType, nullable=True),
    StructField("rotation", QuaternionType, nullable=True),
])

# Session-level ego motion summary
SessionEgoMotionSchema = StructType([
    StructField("session_id", StringType(), nullable=False),
    StructField("translation", Translation3DType, nullable=True),
    StructField("rotation", QuaternionType, nullable=True),
    StructField("start", Translation3DType, nullable=True),
    StructField("goal", Translation3DType, nullable=True),
])


# =============================================================================
# Metadata Tables
# =============================================================================

HDMapSchema = StructType([
    StructField("clip_id", StringType(), nullable=False),
    StructField("filename", StringType(), nullable=True),
    StructField("city", StringType(), nullable=True),
    StructField("site", StringType(), nullable=True),
])


# =============================================================================
# Schema Registry (for dynamic access by table name)
# =============================================================================

BRONZE_SCHEMAS = {
    "session": SessionSchema,
    "clip": ClipSchema,
    "frame": FrameSchema,
    "calibration": CalibrationSchema,
    "camera": CameraSchema,
    "lidar": LidarSchema,
    "radar": RadarSchema,
    "category": CategorySchema,
    "dynamic_object": DynamicObjectSchema,
    "occupancy": OccupancySchema,
    "motion": MotionSchema,
    "ego_motion": EgoMotionSchema,
    "session_ego_motion": SessionEgoMotionSchema,
    "hdmap": HDMapSchema,
}


def get_schema(table_name: str) -> StructType:
    """Get the schema for a given table name."""
    if table_name not in BRONZE_SCHEMAS:
        raise ValueError(f"Unknown table: {table_name}. Available: {list(BRONZE_SCHEMAS.keys())}")
    return BRONZE_SCHEMAS[table_name]
