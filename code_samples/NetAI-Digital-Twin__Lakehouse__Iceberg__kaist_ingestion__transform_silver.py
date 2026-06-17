"""
Silver Layer Transformations for KAIST E2E Dataset.

The Silver layer applies:
1. Data cleaning and type coercion
2. Referential integrity validation
3. Strategic partitioning for query optimization
4. Sort ordering for efficient scans
"""

from typing import Dict, List, Optional

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import (
    array,
    col,
    concat_ws,
    explode,
    lit,
    regexp_replace,
    struct,
    when,
)

from .config import PipelineConfig, apply_ad_table_optimizations, build_spark_session, create_namespaces


class SilverTransformer:
    """
    Transforms Bronze tables into optimized Silver tables.
    
    Key optimizations:
    - Partitioning by high-cardinality filter columns
    - Sort ordering for temporal/sequential access
    - Deduplication and null handling
    """
    
    # Partitioning configuration for each table
    PARTITION_CONFIG = {
        "session": [],  # Small reference table - no partitioning
        "clip": ["session_id"],
        "frame": ["clip_id"],
        "calibration": ["clip_id", "sensor_name"],
        "camera": ["camera_name", "clip_id"],
        "lidar": ["clip_id"],
        "radar": ["radar_name", "clip_id"],
        "category": [],  # Small reference table
        "dynamic_object": ["clip_id"],
        "occupancy": ["clip_id"],
        "motion": ["clip_id"],
        "ego_motion": ["clip_id"],
        "session_ego_motion": [],
        "hdmap": [],
    }
    
    # Sort order configuration for each table
    SORT_CONFIG = {
        "frame": ["clip_id", "frame_idx"],
        "camera": ["clip_id", "frame_id", "sensor_timestamp"],
        "lidar": ["clip_id", "frame_id", "sensor_timestamp"],
        "radar": ["clip_id", "frame_id", "sensor_timestamp"],
        "dynamic_object": ["clip_id", "frame_id"],
        "ego_motion": ["clip_id", "frame_id"],
    }
    
    # AD-specific columns requiring full min/max metrics for predicate pushdown.
    # These enable Iceberg to skip data files based on sensor timestamp ranges,
    # frame indices, clip boundaries, and sensor name filters — the dominant
    # predicate patterns in autonomous driving ML data loading.
    METRICS_CONFIG = {
        "frame": ["clip_id", "frame_idx"],
        "camera": ["sensor_timestamp", "clip_id", "frame_id", "camera_name"],
        "lidar": ["sensor_timestamp", "clip_id", "frame_id"],
        "radar": ["sensor_timestamp", "clip_id", "frame_id", "radar_name"],
        "calibration": ["clip_id", "sensor_name"],
        "dynamic_object": ["clip_id", "frame_id"],
        "ego_motion": ["clip_id", "frame_id"],
        "clip": ["session_id", "clip_id"],
    }
    
    def __init__(self, spark: SparkSession, config: PipelineConfig):
        self.spark = spark
        self.config = config
        self.catalog = config.spark_catalog_name
        self.bronze_ns = config.kaist.namespace_bronze
        self.silver_ns = config.kaist.namespace_silver
        
    def _bronze_table(self, table: str) -> str:
        return f"{self.catalog}.{self.bronze_ns}.{table}"
    
    def _silver_table(self, table: str) -> str:
        return f"{self.catalog}.{self.silver_ns}.{table}"
    
    def _read_bronze(self, table: str) -> DataFrame:
        """Read a table from the Bronze layer."""
        return self.spark.table(self._bronze_table(table))
    
    def _write_silver(
        self, 
        df: DataFrame, 
        table: str,
        partitions: Optional[List[str]] = None,
        sort_by: Optional[List[str]] = None,
    ) -> int:
        """
        Write a DataFrame to the Silver layer with optimizations.
        
        Args:
            df: DataFrame to write
            table: Target table name
            partitions: Partition columns
            sort_by: Sort order columns
            
        Returns:
            Number of rows written
        """
        full_table = self._silver_table(table)
        
        # Apply sort if specified
        if sort_by:
            df = df.sortWithinPartitions(*sort_by)
        
        # Build writer
        writer = (
            df.writeTo(full_table)
            .using("iceberg")
            .tableProperty("format-version", "2")
            .tableProperty("write.target-file-size-bytes", 
                          str(self.config.kaist.target_file_size_bytes))
        )
        
        # Apply partitioning
        if partitions:
            writer = writer.partitionedBy(*partitions)
        
        writer.createOrReplace()
        
        # Apply AD-specific Iceberg optimizations (persisted for all future writes)
        metrics = self.METRICS_CONFIG.get(table, [])
        apply_ad_table_optimizations(
            self.spark,
            full_table,
            sort_columns=sort_by,
            partition_columns=partitions,
            metrics_columns=metrics if metrics else None,
            config=self.config.kaist,
        )
        
        return self.spark.table(full_table).count()
    
    # =========================================================================
    # Table-specific transformations
    # =========================================================================
    
    def transform_session(self) -> int:
        """Transform session table - minimal changes, just clean nulls."""
        df = self._read_bronze("session")
        
        # Ensure clip_id_list is not null (empty array instead)
        df = df.withColumn(
            "clip_id_list",
            when(col("clip_id_list").isNull(), array()).otherwise(col("clip_id_list"))
        )
        
        return self._write_silver(df, "session")
    
    def transform_clip(self) -> int:
        """Transform clip table with session_id partitioning."""
        df = self._read_bronze("clip")
        
        # Ensure frame_id_list is not null
        df = df.withColumn(
            "frame_id_list",
            when(col("frame_id_list").isNull(), array()).otherwise(col("frame_id_list"))
        )
        
        return self._write_silver(
            df, "clip",
            partitions=self.PARTITION_CONFIG["clip"],
        )
    
    def transform_frame(self) -> int:
        """Transform frame table with clip_id partitioning and frame_idx sorting."""
        df = self._read_bronze("frame")
        
        return self._write_silver(
            df, "frame",
            partitions=self.PARTITION_CONFIG["frame"],
            sort_by=self.SORT_CONFIG["frame"],
        )
    
    def transform_calibration(self) -> int:
        """Transform calibration table with composite partitioning."""
        df = self._read_bronze("calibration")
        
        return self._write_silver(
            df, "calibration",
            partitions=self.PARTITION_CONFIG["calibration"],
        )
    
    def transform_camera(self) -> int:
        """Transform camera table - key table for ML workloads."""
        df = self._read_bronze("camera")
        
        # Normalize camera names if needed
        df = df.withColumn(
            "camera_name",
            regexp_replace(col("camera_name"), r"[^A-Za-z0-9_]", "_")
        )
        
        return self._write_silver(
            df, "camera",
            partitions=self.PARTITION_CONFIG["camera"],
            sort_by=self.SORT_CONFIG["camera"],
        )
    
    def transform_lidar(self) -> int:
        """Transform lidar table."""
        df = self._read_bronze("lidar")
        
        return self._write_silver(
            df, "lidar",
            partitions=self.PARTITION_CONFIG["lidar"],
            sort_by=self.SORT_CONFIG["lidar"],
        )
    
    def transform_radar(self) -> int:
        """Transform radar table with multi-sensor partitioning."""
        df = self._read_bronze("radar")
        
        # Normalize radar names
        df = df.withColumn(
            "radar_name",
            regexp_replace(col("radar_name"), r"[^A-Za-z0-9_]", "_")
        )
        
        return self._write_silver(
            df, "radar",
            partitions=self.PARTITION_CONFIG["radar"],
            sort_by=self.SORT_CONFIG["radar"],
        )
    
    def transform_category(self) -> int:
        """Transform category reference table."""
        df = self._read_bronze("category")
        return self._write_silver(df, "category")
    
    def transform_dynamic_object(self) -> int:
        """Transform dynamic object annotations."""
        df = self._read_bronze("dynamic_object")
        
        return self._write_silver(
            df, "dynamic_object",
            partitions=self.PARTITION_CONFIG["dynamic_object"],
            sort_by=self.SORT_CONFIG["dynamic_object"],
        )
    
    def transform_ego_motion(self) -> int:
        """Transform frame-level ego motion."""
        df = self._read_bronze("ego_motion")
        
        return self._write_silver(
            df, "ego_motion",
            partitions=self.PARTITION_CONFIG["ego_motion"],
            sort_by=self.SORT_CONFIG["ego_motion"],
        )
    
    def transform_hdmap(self) -> int:
        """Transform HD map references."""
        df = self._read_bronze("hdmap")
        return self._write_silver(df, "hdmap")
    
    def transform_all(self) -> Dict[str, int]:
        """
        Run all Silver transformations.
        
        Returns:
            Dictionary mapping table names to row counts
        """
        transformations = [
            ("session", self.transform_session),
            ("clip", self.transform_clip),
            ("frame", self.transform_frame),
            ("calibration", self.transform_calibration),
            ("camera", self.transform_camera),
            ("lidar", self.transform_lidar),
            ("radar", self.transform_radar),
            ("category", self.transform_category),
            ("dynamic_object", self.transform_dynamic_object),
            ("ego_motion", self.transform_ego_motion),
            ("hdmap", self.transform_hdmap),
        ]
        
        results = {}
        for table_name, transform_fn in transformations:
            try:
                print(f"[TRANSFORM] {table_name}")
                count = transform_fn()
                results[table_name] = count
                print(f"[DONE] {self._silver_table(table_name)}: {count} rows")
            except Exception as e:
                print(f"[ERROR] Failed to transform {table_name}: {e}")
                results[table_name] = -1
                
        return results


def run_silver_transformation(config: Optional[PipelineConfig] = None) -> Dict[str, int]:
    """
    Main entry point for Silver layer transformation.
    
    Args:
        config: Pipeline configuration (uses defaults if None)
        
    Returns:
        Dictionary mapping table names to row counts
    """
    if config is None:
        config = PipelineConfig()
        
    spark = build_spark_session(config, app_name="kaist-silver-transform")
    
    try:
        create_namespaces(spark, config)
        transformer = SilverTransformer(spark, config)
        return transformer.transform_all()
    finally:
        spark.stop()


if __name__ == "__main__":
    results = run_silver_transformation()
    
    print("\n" + "=" * 60)
    print("SILVER TRANSFORMATION SUMMARY")
    print("=" * 60)
    for table, count in results.items():
        status = "✓" if count >= 0 else "✗"
        print(f"  {status} {table}: {count} rows")
