"""
KAIST E2E Dataset Ingestion Pipeline for Apache Iceberg Lakehouse.

This package provides automated ingestion of the KAIST/MOTIE E2E autonomous
driving dataset into a medallion-architecture Iceberg lakehouse.

Layers:
    - Bronze: Raw ingestion (1:1 from source)
    - Silver: Cleaned, typed, and partitioned
    - Gold: Pre-joined ML-ready feature tables
"""

__version__ = "0.1.0"
