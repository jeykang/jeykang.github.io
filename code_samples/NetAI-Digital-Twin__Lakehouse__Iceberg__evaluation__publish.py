#!/usr/bin/env python3
"""Land evaluation results in Iceberg, next to the curation scores.

Kept separate from the harness on purpose: `run_eval.py` has no Spark dependency
and runs anywhere, which matters because the eval pipeline is meant to be
consumable by people who do not run this lakehouse. Publishing is the optional
step for people who do.

    python publish.py .results_constant_velocity.parquet --run-id nightly-2026-08-11

Table: <catalog>.eval.policy_runs — one row per (run_id, policy, clip_id), so
successive runs of the same policy accumulate rather than overwrite, and a run can
be joined against nvidia_gold.clip_scores on clip_id to ask the question the
lakehouse exists to answer: does this policy do worse on the clips we curated as
hard?
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

NAMESPACE = os.environ.get("EVAL_NS", "eval")
TABLE = "policy_runs"


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def build_spark(app="eval-publish"):
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    from kaist_ingestion.config import CatalogConfig, StorageConfig
    from pyspark.sql import SparkSession

    storage, cat = StorageConfig(), CatalogConfig()
    c = "iceberg"
    b = (SparkSession.builder.appName(app)
         .config("spark.sql.defaultCatalog", c)
         .config(f"spark.sql.catalog.{c}", "org.apache.iceberg.spark.SparkCatalog")
         .config(f"spark.sql.catalog.{c}.catalog-impl", "org.apache.iceberg.rest.RESTCatalog")
         .config(f"spark.sql.catalog.{c}.uri", cat.uri)
         .config(f"spark.sql.catalog.{c}.warehouse", cat.warehouse)
         .config(f"spark.sql.catalog.{c}.io-impl", "org.apache.iceberg.io.ResolvingFileIO")
         .config(f"spark.sql.catalog.{c}.s3.endpoint", storage.endpoint)
         .config(f"spark.sql.catalog.{c}.s3.path-style-access",
                 str(storage.path_style_access).lower())
         .config(f"spark.sql.catalog.{c}.s3.access-key-id", storage.access_key)
         .config(f"spark.sql.catalog.{c}.s3.secret-access-key", storage.secret_key)
         .config(f"spark.sql.catalog.{c}.oauth2-server-uri", cat.oauth2_server_uri)
         .config(f"spark.sql.catalog.{c}.credential", cat.credential)
         .config(f"spark.sql.catalog.{c}.scope", cat.scope)
         .config("spark.sql.extensions",
                 "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions"))
    return b.getOrCreate()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results", help="parquet written by run_eval.py")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--namespace", default=NAMESPACE)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    import pyarrow.parquet as pq
    rows = pq.read_table(a.results).to_pylist()
    run_id = a.run_id or f"{rows[0]['policy']}-{int(time.time())}"
    stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    sha = _git_sha()
    for r in rows:
        r.update(run_id=run_id, evaluated_at=stamp, code_sha=sha)
    print(f"[publish] {len(rows)} rows  run_id={run_id}  policy={rows[0]['policy']}")
    if a.dry_run:
        print("[publish] dry run — not writing"); return

    spark = build_spark()
    fq = f"iceberg.{a.namespace}.{TABLE}"
    spark.sql(f"CREATE NAMESPACE IF NOT EXISTS iceberg.{a.namespace}")
    df = spark.createDataFrame(rows)
    (df.writeTo(fq).tableProperty("format-version", "2")
       .partitionedBy(df.policy).createOrReplace() if not spark.catalog.tableExists(fq)
     else df.writeTo(fq).append())
    print(f"[publish] wrote {fq}")
    spark.stop()


if __name__ == "__main__":
    main()
