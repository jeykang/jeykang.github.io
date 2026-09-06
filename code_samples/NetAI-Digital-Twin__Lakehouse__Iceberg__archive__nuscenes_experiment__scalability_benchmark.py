#!/usr/bin/env python3
"""
Scalability Benchmark: SF 1-50 for Python Baseline, Silver JOIN, and Gold.

Scales the primary fact table (sample_data) from 1x to 50x the nuScenes-mini
base size, keeping reference tables (samples, categories, instances) fixed at
1x.  This models a realistic growth scenario: more sensor data collected while
the vocabulary of sensors, categories, and sessions stays stable.

For each scale factor the script:
  - Python Baseline: replicates data in memory, times the nested-loop query
  - Silver JOIN    : ingests nx sample_data into Iceberg (partitioned by
                     channel), runs a multi-table JOIN query
  - Gold           : writes a pre-joined gold table (nx rows), runs a
                     partition-pruned filter query

Outputs:
  nuscenes_experiment/scalability_results.json  - raw measurements
  nuscenes_experiment/scalability_chart.png     - 3-line chart

Usage (inside Spark container):
    export PYTHONPATH=/opt/spark:/opt/spark/python:/opt/spark/python/lib/py4j-0.10.9.7-src.zip
    python3 nuscenes_experiment/scalability_benchmark.py
"""

import json
import os
import time
import statistics

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
RAW_DATA_PATH = "/user_data/nuscenes-mini/v1.0-mini/v1.0-mini"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_JSON = os.path.join(SCRIPT_DIR, "scalability_results.json")
OUTPUT_CHART = os.path.join(SCRIPT_DIR, "scalability_chart.png")

# Dense from 1-10, then every-5 to 50 -> 18 points for smooth curves
SCALE_FACTORS = list(range(1, 11)) + [15, 20, 25, 30, 35, 40, 45, 50]

BASELINE_RUNS = 5        # pure Python, fast - median of 5
SPARK_WARMUP = 1         # JVM warmup
SPARK_TIMED = 3          # median of 3

# ---------------------------------------------------------------------------
# Polaris / Spark config
# ---------------------------------------------------------------------------
aws_access_key = os.getenv("AWS_ACCESS_KEY_ID", "minioadmin")
aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY", "minioadmin")
s3_endpoint    = os.getenv("AWS_S3_ENDPOINT", "http://minio:9000")
polaris_uri    = os.getenv("POLARIS_URI", "http://polaris:8181/api/catalog")
polaris_wh     = os.getenv("POLARIS_CATALOG_NAME", "lakehouse_catalog")
polaris_cred   = os.getenv("POLARIS_CREDENTIAL", "root:s3cr3t")
polaris_scope  = os.getenv("POLARIS_SCOPE", "PRINCIPAL_ROLE:ALL")
CATALOG   = "iceberg"
NAMESPACE = "nusc_scalability"


# ===================================================================
# Helper: load raw JSONs once
# ===================================================================
_RAW_CACHE = {}

def _raw(name):
    if name not in _RAW_CACHE:
        with open(os.path.join(RAW_DATA_PATH, name)) as f:
            _RAW_CACHE[name] = json.load(f)
    return _RAW_CACHE[name]


# ===================================================================
# 1.  Python Baseline
# ===================================================================
def run_baseline(sf):
    """Pure-Python nested-loop at scale factor sf (sample_data x sf)."""
    data_sample_data = _raw("sample_data.json") * sf        # scale fact table
    data_annotations = _raw("sample_annotation.json")        # 1x
    data_categories  = _raw("category.json")
    data_instances   = _raw("instance.json")
    data_sensors     = _raw("sensor.json")
    data_calibrated  = _raw("calibrated_sensor.json")

    # Index structures (built once, not timed)
    cat_map = {c["token"]: c["name"] for c in data_categories}
    i2c     = {i["token"]: i["category_token"] for i in data_instances}
    sch     = {s["token"]: s["channel"] for s in data_sensors}
    cch = {}
    for cs in data_calibrated:
        if cs["sensor_token"] in sch:
            cch[cs["token"]] = sch[cs["sensor_token"]]

    ann_by_sample = {}
    for a in data_annotations:
        ann_by_sample.setdefault(a["sample_token"], []).append(a)

    # Timed query runs
    timings = []
    row_count = 0
    for _ in range(BASELINE_RUNS):
        t0 = time.perf_counter()
        pairs = []
        for sd in data_sample_data:
            if cch.get(sd["calibrated_sensor_token"]) != "CAM_FRONT":
                continue
            for ann in ann_by_sample.get(sd["sample_token"], []):
                cn = cat_map.get(i2c.get(ann["instance_token"]))
                if cn == "human.pedestrian.adult":
                    pairs.append(sd["filename"])
        timings.append(time.perf_counter() - t0)
        row_count = len(pairs)

    return {
        "elapsed_seconds": round(statistics.median(timings), 6),
        "row_count": row_count,
    }


# ===================================================================
# 2.  Spark helpers
# ===================================================================
_spark = None


def _get_spark():
    global _spark
    if _spark is not None:
        return _spark
    from pyspark.sql import SparkSession

    _spark = (
        SparkSession.builder.appName("ScalabilityBenchmark")
        .config("spark.sql.defaultCatalog", CATALOG)
        .config(f"spark.sql.catalog.{CATALOG}", "org.apache.iceberg.spark.SparkCatalog")
        .config(f"spark.sql.catalog.{CATALOG}.catalog-impl", "org.apache.iceberg.rest.RESTCatalog")
        .config(f"spark.sql.catalog.{CATALOG}.uri", polaris_uri)
        .config(f"spark.sql.catalog.{CATALOG}.warehouse", polaris_wh)
        .config(f"spark.sql.catalog.{CATALOG}.io-impl", "org.apache.iceberg.aws.s3.S3FileIO")
        .config(f"spark.sql.catalog.{CATALOG}.s3.endpoint", s3_endpoint)
        .config(f"spark.sql.catalog.{CATALOG}.s3.path-style-access", "true")
        .config(f"spark.sql.catalog.{CATALOG}.s3.access-key-id", aws_access_key)
        .config(f"spark.sql.catalog.{CATALOG}.s3.secret-access-key", aws_secret_key)
        .config(f"spark.sql.catalog.{CATALOG}.credential", polaris_cred)
        .config(f"spark.sql.catalog.{CATALOG}.scope", polaris_scope)
        .config("spark.hadoop.fs.s3a.endpoint", s3_endpoint)
        .config("spark.hadoop.fs.s3a.access.key", aws_access_key)
        .config("spark.hadoop.fs.s3a.secret.key", aws_secret_key)
        .config("spark.hadoop.fs.s3a.path.style.access", "true")
        .config("spark.hadoop.fs.s3a.connection.ssl.enabled", "false")
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
        .config(
            "spark.hadoop.fs.s3a.aws.credentials.provider",
            "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider",
        )
        .getOrCreate()
    )
    return _spark


def _scale_df(spark, df, n):
    """Replicate df by n-times via crossJoin(range(n))."""
    if n <= 1:
        return df
    return df.crossJoin(spark.range(n)).drop("id")


def _timed_query(spark, sql):
    """Run sql with SPARK_WARMUP warmup + SPARK_TIMED timed; return median."""
    timings = []
    row_count = 0
    for i in range(SPARK_WARMUP + SPARK_TIMED):
        t0 = time.perf_counter()
        row_count = spark.sql(sql).count()
        elapsed = time.perf_counter() - t0
        if i >= SPARK_WARMUP:
            timings.append(elapsed)
    return {
        "elapsed_seconds": round(statistics.median(timings), 6),
        "row_count": row_count,
    }


# ===================================================================
# 3.  Silver JOIN
# ===================================================================
def _setup_silver_ref(spark):
    """Write reference tables once (1x, never scaled)."""
    ns = f"{CATALOG}.{NAMESPACE}"
    spark.sql(f"CREATE NAMESPACE IF NOT EXISTS {ns}")

    df_sample   = spark.read.option("multiLine", True).json(f"{RAW_DATA_PATH}/sample.json")
    df_category = spark.read.option("multiLine", True).json(f"{RAW_DATA_PATH}/category.json")
    df_instance = spark.read.option("multiLine", True).json(f"{RAW_DATA_PATH}/instance.json")

    df_sample.write.format("iceberg").mode("overwrite").saveAsTable(f"{ns}.samples")
    df_category.write.format("iceberg").mode("overwrite").saveAsTable(f"{ns}.category")
    df_instance.write.format("iceberg").mode("overwrite").saveAsTable(f"{ns}.instances")
    return ns


def _ingest_silver_facts(spark, ns, sf):
    """Write sample_data (sf-times) and annotations (1x)."""
    df_sd  = spark.read.option("multiLine", True).json(f"{RAW_DATA_PATH}/sample_data.json")
    df_ann = spark.read.option("multiLine", True).json(f"{RAW_DATA_PATH}/sample_annotation.json")
    df_sen = spark.read.option("multiLine", True).json(f"{RAW_DATA_PATH}/sensor.json")
    df_cal = spark.read.option("multiLine", True).json(f"{RAW_DATA_PATH}/calibrated_sensor.json")

    # Enrich sample_data with channel
    df_ch = df_cal.join(df_sen, df_cal["sensor_token"] == df_sen["token"]) \
                  .select(df_cal["token"].alias("calib_token"), df_sen["channel"])
    df_sd_enriched = df_sd.join(df_ch, df_sd["calibrated_sensor_token"] == df_ch["calib_token"]) \
                         .drop("calib_token")

    _scale_df(spark, df_sd_enriched, sf) \
        .write.format("iceberg").partitionBy("channel").mode("overwrite") \
        .saveAsTable(f"{ns}.sample_data")

    # Annotations always 1x
    df_ann.write.format("iceberg").mode("overwrite").saveAsTable(f"{ns}.annotations")


def _query_silver(spark, ns):
    sql = f"""
        SELECT sd.filename AS img_path, a.translation, a.size, a.rotation
        FROM {ns}.samples s
        JOIN {ns}.sample_data sd ON s.token = sd.sample_token
        JOIN {ns}.annotations a ON s.token = a.sample_token
        JOIN {ns}.instances  i ON a.instance_token = i.token
        JOIN {ns}.category   c ON i.category_token = c.token
        WHERE sd.channel = 'CAM_FRONT'
          AND c.name    = 'human.pedestrian.adult'
    """
    return _timed_query(spark, sql)


# ===================================================================
# 4.  Gold (Pre-Joined)
# ===================================================================
def _ingest_gold(spark, ns, sf):
    """Build a pre-joined gold table with sf-times rows."""
    df_sd  = spark.read.option("multiLine", True).json(f"{RAW_DATA_PATH}/sample_data.json")
    df_ann = spark.read.option("multiLine", True).json(f"{RAW_DATA_PATH}/sample_annotation.json")
    df_cat = spark.read.option("multiLine", True).json(f"{RAW_DATA_PATH}/category.json")
    df_ins = spark.read.option("multiLine", True).json(f"{RAW_DATA_PATH}/instance.json")
    df_sen = spark.read.option("multiLine", True).json(f"{RAW_DATA_PATH}/sensor.json")
    df_cal = spark.read.option("multiLine", True).json(f"{RAW_DATA_PATH}/calibrated_sensor.json")

    df_ch = df_cal.join(df_sen, df_cal["sensor_token"] == df_sen["token"]) \
                  .select(df_cal["token"].alias("calib_token"), df_sen["channel"])

    df_gold_raw = (
        df_sd
        .join(df_ch,  df_sd["calibrated_sensor_token"] == df_ch["calib_token"])
        .join(df_ann, df_sd["sample_token"] == df_ann["sample_token"])
        .join(df_ins, df_ann["instance_token"] == df_ins["token"])
        .join(df_cat, df_ins["category_token"] == df_cat["token"])
        .select(
            df_sd["filename"].alias("img_path"),
            df_ann["translation"], df_ann["size"], df_ann["rotation"],
            df_sen["channel"],
            df_cat["name"].alias("category_name"),
        )
    )

    _scale_df(spark, df_gold_raw, sf) \
        .write.format("iceberg").partitionBy("channel").mode("overwrite") \
        .saveAsTable(f"{ns}.gold_train_set")


def _query_gold(spark, ns):
    sql = f"""
        SELECT img_path, translation, size, rotation
        FROM {ns}.gold_train_set
        WHERE channel       = 'CAM_FRONT'
          AND category_name = 'human.pedestrian.adult'
    """
    return _timed_query(spark, sql)


# ===================================================================
# Main sweep
# ===================================================================
def main():
    t_start = time.perf_counter()
    results = []

    # --- Phase A: Python Baseline -------------------------------------------
    print("=" * 65)
    print("PHASE A  |  Python Baseline  |  SF 1 - 50")
    print("=" * 65)
    for sf in SCALE_FACTORS:
        print(f"  [Baseline] SF={sf:>2d} ... ", end="", flush=True)
        r = run_baseline(sf)
        print(f"{r['elapsed_seconds']:.4f} s  ({r['row_count']:,} rows)")
        results.append({"strategy": "Python Baseline", "scale_factor": sf, **r})

    # --- Spark init & reference tables --------------------------------------
    spark = _get_spark()
    ns = _setup_silver_ref(spark)

    # --- Phase B: Silver JOIN ------------------------------------------------
    print("=" * 65)
    print("PHASE B  |  Silver JOIN  |  SF 1 - 50")
    print("=" * 65)
    for sf in SCALE_FACTORS:
        print(f"  [Silver] SF={sf:>2d}  ingest ... ", end="", flush=True)
        _ingest_silver_facts(spark, ns, sf)
        print("query ... ", end="", flush=True)
        r = _query_silver(spark, ns)
        print(f"{r['elapsed_seconds']:.4f} s  ({r['row_count']:,} rows)")
        results.append({"strategy": "Silver JOIN", "scale_factor": sf, **r})

    # --- Phase C: Gold -------------------------------------------------------
    print("=" * 65)
    print("PHASE C  |  Gold (Pre-Joined)  |  SF 1 - 50")
    print("=" * 65)
    for sf in SCALE_FACTORS:
        print(f"  [Gold]   SF={sf:>2d}  ingest ... ", end="", flush=True)
        _ingest_gold(spark, ns, sf)
        print("query ... ", end="", flush=True)
        r = _query_gold(spark, ns)
        print(f"{r['elapsed_seconds']:.4f} s  ({r['row_count']:,} rows)")
        results.append({"strategy": "Gold", "scale_factor": sf, **r})

    spark.stop()

    elapsed_total = time.perf_counter() - t_start
    print(f"\nTotal benchmark time: {elapsed_total:.1f} s")

    # --- Write JSON ----------------------------------------------------------
    with open(OUTPUT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results  -> {OUTPUT_JSON}")

    # --- Chart ---------------------------------------------------------------
    try:
        _make_chart(results)
        print(f"Chart    -> {OUTPUT_CHART}")
    except Exception as exc:
        print(f"Chart generation failed ({exc}); JSON is still valid.")


# ===================================================================
# Matplotlib chart
# ===================================================================
def _make_chart(results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    strategies = ["Python Baseline", "Silver JOIN", "Gold"]
    colors  = {"Python Baseline": "#e74c3c", "Silver JOIN": "#3498db", "Gold": "#2ecc71"}
    markers = {"Python Baseline": "o",       "Silver JOIN": "s",       "Gold": "D"}

    fig, ax = plt.subplots(figsize=(10, 5.5))

    for strat in strategies:
        pts = sorted(
            [r for r in results if r["strategy"] == strat],
            key=lambda r: r["scale_factor"],
        )
        xs = [p["scale_factor"] for p in pts]
        ys = [p["elapsed_seconds"] for p in pts]
        ax.plot(xs, ys,
                label=strat, color=colors[strat],
                marker=markers[strat], markersize=5, linewidth=2)

    ax.set_xlabel("Scale Factor (n x base sample_data)", fontsize=12)
    ax.set_ylabel("Query Latency (seconds)", fontsize=12)
    ax.set_title("nuScenes Scalability: Query Latency vs. Data Scale Factor",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=11, loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 52)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(OUTPUT_CHART, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
