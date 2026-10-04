#!/usr/bin/env python3
"""
Quick integration test: verify add_files() + input_file_name() works
for the zero-copy pipeline.

Run inside the spark-iceberg container:
    python -m nvidia_ingestion.test_zerocopy
"""

import time
import sys

from .config import NvidiaPipelineConfig, build_spark_session, create_namespaces


def main():
    cfg = NvidiaPipelineConfig()
    spark = build_spark_session(cfg, app_name="nvidia-zerocopy-test")
    cat = cfg.spark_catalog_name
    ns = "nvidia_test"

    try:
        spark.sql(f"CREATE NAMESPACE IF NOT EXISTS {cat}.{ns}")
        print(f"Using namespace: {cat}.{ns}")

        SRC = cfg.nvidia.source_path

        # ── Test 1: Register bare Parquet via add_files() ──────────
        print("\n[TEST 1] Register clip_index.parquet via add_files()")
        t0 = time.time()

        clip_uri = f"file://{SRC}/clip_index.parquet"
        table = f"{cat}.{ns}.clip_index"

        # Create empty table with matching schema
        df = spark.read.parquet(clip_uri)
        print(f"  Schema: {', '.join(df.columns[:5])}...")
        df.limit(0).writeTo(table).using("iceberg").tableProperty(
            "format-version", "2"
        ).createOrReplace()

        # Register files (zero-copy)
        spark.sql(
            f"CALL {cat}.system.add_files("
            f"  table => '{table}',"
            f"  source_table => '`parquet`.`{clip_uri}`'"
            f")"
        )
        count = spark.table(table).count()
        print(f"  [OK] {table}: {count} rows in {time.time()-t0:.2f}s")

        # ── Test 2: input_file_name() on add_files() table ────────
        print("\n[TEST 2] input_file_name() on add_files() registered table")
        t0 = time.time()
        result = spark.sql(f"""
            SELECT input_file_name() AS file_path
            FROM {table}
            LIMIT 3
        """).collect()
        for r in result:
            print(f"  file_path: {r.file_path[:100]}...")
        print(f"  [OK] input_file_name() works, {time.time()-t0:.2f}s")

        # ── Test 3: Register calibration directory via add_files() ─
        print("\n[TEST 3] Register calibration/sensor_extrinsics/ dir via add_files()")
        t0 = time.time()

        cal_uri = f"file://{SRC}/calibration/sensor_extrinsics"
        cal_table = f"{cat}.{ns}.sensor_extrinsics"

        df2 = spark.read.parquet(cal_uri)
        print(f"  Schema: {', '.join(df2.columns[:5])}...")
        df2.limit(0).writeTo(cal_table).using("iceberg").tableProperty(
            "format-version", "2"
        ).createOrReplace()

        spark.sql(
            f"CALL {cat}.system.add_files("
            f"  table => '{cal_table}',"
            f"  source_table => '`parquet`.`{cal_uri}`'"
            f")"
        )
        count2 = spark.table(cal_table).count()
        print(f"  [OK] {cal_table}: {count2} rows in {time.time()-t0:.2f}s")

        # ── Test 4: input_file_name() with regex for clip_id ──────
        print("\n[TEST 4] clip_id extraction from input_file_name() on calibration")
        t0 = time.time()
        result2 = spark.sql(f"""
            SELECT
                input_file_name() AS file_path,
                regexp_extract(input_file_name(),
                    '([0-9a-f]{{8}}-[0-9a-f]{{4}}-[0-9a-f]{{4}}-[0-9a-f]{{4}}-[0-9a-f]{{12}})', 1
                ) AS clip_id
            FROM {cal_table}
            LIMIT 3
        """).collect()
        for r in result2:
            print(f"  file: ...{r.file_path[-60:]}")
            print(f"  clip_id: '{r.clip_id}'")
        print(f"  [OK] regex extraction test done, {time.time()-t0:.2f}s")

        # ── Test 5: FUSE-mounted radar Parquets (if available) ────
        print("\n[TEST 5] FUSE-mounted radar (if available)")
        import os
        fuse_test = "/tmp/nvidia-fuse/test_radar"
        if os.path.isdir(fuse_test):
            import glob
            parquets = glob.glob(os.path.join(fuse_test, "**/*.parquet"), recursive=True)
            if parquets:
                t0 = time.time()
                # Pick first chunk directory
                first_dir = os.path.dirname(parquets[0])
                radar_uri = f"file://{first_dir}"
                radar_table = f"{cat}.{ns}.radar_test"

                df3 = spark.read.parquet(radar_uri)
                print(f"  Schema: {', '.join(df3.columns[:5])}...")
                df3.limit(0).writeTo(radar_table).using("iceberg").tableProperty(
                    "format-version", "2"
                ).createOrReplace()

                spark.sql(
                    f"CALL {cat}.system.add_files("
                    f"  table => '{radar_table}',"
                    f"  source_table => '`parquet`.`{radar_uri}`'"
                    f")"
                )
                count3 = spark.table(radar_table).count()
                print(f"  [OK] {radar_table}: {count3} rows in {time.time()-t0:.2f}s")

                # Test input_file_name() on FUSE data
                result3 = spark.sql(f"""
                    SELECT
                        input_file_name() AS file_path,
                        regexp_extract(input_file_name(),
                            '([0-9a-f]{{8}}-[0-9a-f]{{4}}-[0-9a-f]{{4}}-[0-9a-f]{{4}}-[0-9a-f]{{12}})', 1
                        ) AS clip_id
                    FROM {radar_table}
                    LIMIT 3
                """).collect()
                for r in result3:
                    print(f"  file: ...{r.file_path[-80:]}")
                    print(f"  clip_id: '{r.clip_id}'")
            else:
                print("  [SKIP] No parquet files in FUSE mount")
        else:
            print("  [SKIP] FUSE mount not available yet")

        # ── Cleanup ──────────────────────────────────────────────
        print("\n[CLEANUP] Dropping test namespace")
        for t in ["clip_index", "sensor_extrinsics", "radar_test"]:
            try:
                spark.sql(f"DROP TABLE IF EXISTS {cat}.{ns}.{t} PURGE")
            except Exception:
                pass
        spark.sql(f"DROP NAMESPACE IF EXISTS {cat}.{ns}")
        print("  [OK] Cleaned up")

        print("\n" + "=" * 60)
        print("ALL TESTS PASSED")
        print("=" * 60)

    except Exception as e:
        print(f"\n[FAIL] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
