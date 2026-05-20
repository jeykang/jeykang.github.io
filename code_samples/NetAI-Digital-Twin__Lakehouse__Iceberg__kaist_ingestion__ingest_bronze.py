"""
Bronze Layer Ingestion for KAIST E2E Dataset.

The Bronze layer provides 1:1 mapping from source files to Iceberg tables.
No transformations are applied - this preserves the original data for lineage.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.types import StructType

from .config import PipelineConfig, build_spark_session, create_namespaces
from .schemas import BRONZE_SCHEMAS


class BronzeIngester:
    """
    Ingests raw KAIST data files into the Bronze layer of the lakehouse.
    
    Supports JSON and Parquet source files with automatic schema application.
    """
    
    # Mapping from table name to expected source file pattern
    TABLE_FILE_MAP = {
        "session": "session.json",
        "clip": "clip.json",
        "frame": "frame.json",
        "calibration": "calibration.json",
        "camera": "camera.json",
        "lidar": "lidar.json",
        "radar": "radar.json",
        "category": "category.json",
        "dynamic_object": "dynamic_object.json",
        "occupancy": "occupancy.json",
        "motion": "motion.json",
        "ego_motion": "ego_motion.json",
        "session_ego_motion": "session_ego_motion.json",
        "hdmap": "hdmap.json",
    }
    
    def __init__(self, spark: SparkSession, config: PipelineConfig):
        self.spark = spark
        self.config = config
        self.catalog = config.spark_catalog_name
        self.namespace = config.kaist.namespace_bronze
        self.source_path = Path(config.kaist.source_path)
        
    def _full_table_name(self, table: str) -> str:
        """Get fully qualified table name."""
        return f"{self.catalog}.{self.namespace}.{table}"
    
    def _read_json(self, file_path: Path, schema: Optional[StructType] = None) -> DataFrame:
        """Read a JSON file into a DataFrame."""
        reader = self.spark.read.option("multiline", "true")
        if schema:
            reader = reader.schema(schema)
        return reader.json(str(file_path))
    
    def _read_parquet(self, file_path: Path) -> DataFrame:
        """Read a Parquet file into a DataFrame."""
        return self.spark.read.parquet(str(file_path))
    
    def _find_source_file(self, table_name: str) -> Optional[Path]:
        """
        Locate the source file for a given table.
        
        Supports multiple layouts:
        - <source_path>/<table>.json
        - <source_path>/metadata/<table>.json
        - <source_path>/<table>.parquet
        """
        filename = self.TABLE_FILE_MAP.get(table_name)
        if not filename:
            return None
            
        # Check various possible locations
        candidates = [
            self.source_path / filename,
            self.source_path / "metadata" / filename,
            self.source_path / filename.replace(".json", ".parquet"),
        ]
        
        for candidate in candidates:
            if candidate.exists():
                return candidate
                
        return None
    
    def ingest_table(self, table_name: str, force: bool = False) -> int:
        """
        Ingest a single table from source to Bronze layer.
        
        Args:
            table_name: Name of the table to ingest
            force: If True, overwrite existing table
            
        Returns:
            Number of rows ingested
        """
        if table_name not in BRONZE_SCHEMAS:
            raise ValueError(f"Unknown table: {table_name}")
            
        source_file = self._find_source_file(table_name)
        if source_file is None:
            print(f"[SKIP] No source file found for table: {table_name}")
            return 0
            
        schema = BRONZE_SCHEMAS[table_name]
        full_table = self._full_table_name(table_name)
        
        print(f"[INGEST] {table_name} from {source_file}")
        
        # Read source file
        if source_file.suffix == ".json":
            df = self._read_json(source_file, schema)
        elif source_file.suffix == ".parquet":
            df = self._read_parquet(source_file)
        else:
            raise ValueError(f"Unsupported file format: {source_file.suffix}")
        
        # Write to Iceberg
        write_mode = "createOrReplace" if (force or self.config.kaist.overwrite_existing) else "create"
        
        writer = (
            df.writeTo(full_table)
            .using("iceberg")
            .tableProperty("format-version", "2")
        )
        
        if write_mode == "createOrReplace":
            writer.createOrReplace()
        else:
            writer.create()
        
        row_count = self.spark.table(full_table).count()
        print(f"[DONE] {full_table}: {row_count} rows")
        
        return row_count
    
    def ingest_all(self, tables: Optional[List[str]] = None, force: bool = False) -> Dict[str, int]:
        """
        Ingest all (or specified) tables to Bronze layer.
        
        Args:
            tables: List of table names to ingest (None = all)
            force: If True, overwrite existing tables
            
        Returns:
            Dictionary mapping table names to row counts
        """
        if tables is None:
            tables = list(BRONZE_SCHEMAS.keys())
            
        results = {}
        for table_name in tables:
            try:
                count = self.ingest_table(table_name, force=force)
                results[table_name] = count
            except Exception as e:
                print(f"[ERROR] Failed to ingest {table_name}: {e}")
                results[table_name] = -1
                
        return results


def run_bronze_ingestion(config: Optional[PipelineConfig] = None) -> Dict[str, int]:
    """
    Main entry point for Bronze layer ingestion.
    
    Args:
        config: Pipeline configuration (uses defaults if None)
        
    Returns:
        Dictionary mapping table names to row counts
    """
    if config is None:
        config = PipelineConfig()
        
    spark = build_spark_session(config, app_name="kaist-bronze-ingestion")
    
    try:
        # Ensure namespaces exist
        create_namespaces(spark, config)
        
        # Run ingestion
        ingester = BronzeIngester(spark, config)
        return ingester.ingest_all()
        
    finally:
        spark.stop()


if __name__ == "__main__":
    results = run_bronze_ingestion()
    
    print("\n" + "=" * 60)
    print("BRONZE INGESTION SUMMARY")
    print("=" * 60)
    for table, count in results.items():
        status = "✓" if count >= 0 else "✗"
        print(f"  {status} {table}: {count} rows")
