"""Re-register Nvidia-specific aux tables zero-copy alongside canonical schema.

The canonical KAIST schema doesn't carry several fields that the Gold scorer
needs (hour_of_day, month, platform_class, radar_config, egomotion velocity
/acceleration/curvature). These re-register zero-copy via add_files() and
live alongside the canonical tables. Names are prefixed `aux_` to make their
non-canonical nature obvious to consumers.

Tables created:
    aux_data_collection   — country, month, hour_of_day, platform_class
    aux_sensor_presence   — per-clip 7-camera + 19-radar booleans, radar_config
    aux_egomotion         — per-row timestamp, q*, x/y/z, v*, a*, curvature

Run AFTER canonical_bronze.py + drop of old tables.
"""
from typing import Dict

from .config import (
    NvidiaPipelineConfig, build_spark_session, create_namespaces,
)
from .register_bronze import BronzeRegistrar


def run_aux_registration(config=None, tracker=None) -> Dict[str, int]:
    if config is None:
        config = NvidiaPipelineConfig()
    spark = build_spark_session(config, app_name="nvidia-aux-register")
    try:
        create_namespaces(spark, config)
        reg = BronzeRegistrar(spark, config, tracker=tracker)
        ns = f"{config.spark_catalog_name}.{config.nvidia.namespace_bronze}"
        results: Dict[str, int] = {}

        # data_collection — bare parquet
        spark.sql(f"DROP TABLE IF EXISTS {ns}.aux_data_collection PURGE")
        rows = reg._register(
            "aux_data_collection",
            f"file://{config.nvidia.source_path}/metadata/data_collection.parquet",
        )
        print(f"  [DONE] {ns}.aux_data_collection: {rows:,} rows")
        results["aux_data_collection"] = rows

        # sensor_presence equivalent: HF dropped sensor_presence.parquet in
        # favour of feature_presence.parquet (v26.03). Same per-clip per-sensor
        # boolean schema plus extras (egomotion, *.offline variants).
        spark.sql(f"DROP TABLE IF EXISTS {ns}.aux_sensor_presence")  # NO PURGE
        rows = reg._register(
            "aux_sensor_presence",
            f"file://{config.nvidia.source_path}/metadata/feature_presence.parquet",
        )
        print(f"  [DONE] {ns}.aux_sensor_presence: {rows:,} rows (from feature_presence.parquet)")
        results["aux_sensor_presence"] = rows

        # egomotion — chunked parquet dirs (re-use NFS sensor registration)
        spark.sql(f"DROP TABLE IF EXISTS {ns}.aux_egomotion PURGE")
        rows = reg.register_nfs_sensor("aux_egomotion", "labels/egomotion")
        print(f"  [DONE] {ns}.aux_egomotion: {rows:,} rows")
        results["aux_egomotion"] = rows

        return results
    finally:
        spark.stop()


if __name__ == "__main__":
    res = run_aux_registration()
    print("\n=== Aux registration summary ===")
    for k, v in res.items():
        print(f"  {k}: {v:,} rows")
