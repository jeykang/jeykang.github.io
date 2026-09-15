import os
from pathlib import Path

from pyspark.sql import SparkSession


def _env(name: str, default: str | None = None) -> str:
    value = os.environ.get(name)
    if value is None or value == "":
        if default is None:
            raise RuntimeError(f"Missing required env var: {name}")
        return default
    return value


def build_spark() -> SparkSession:
    polaris_uri = _env("POLARIS_URI", "http://polaris:8181/api/catalog")
    polaris_warehouse = _env("POLARIS_CATALOG_NAME", "lakehouse_catalog")
    polaris_credential = _env("POLARIS_CREDENTIAL", "root:s3cr3t")
    polaris_scope = _env("POLARIS_SCOPE", "PRINCIPAL_ROLE:ALL")
    polaris_oauth2_server_uri = os.environ.get(
        "POLARIS_OAUTH2_SERVER_URI", f"{polaris_uri}/v1/oauth/tokens"
    )

    s3_endpoint = _env("AWS_S3_ENDPOINT", "http://minio:9000")
    s3_bucket = _env("S3_BUCKET", "spark1")
    aws_access_key = _env("AWS_ACCESS_KEY_ID", "minioadmin")
    aws_secret_key = _env("AWS_SECRET_ACCESS_KEY", "minioadmin")
    aws_region = _env("AWS_REGION", "us-east-1")

    # Use the same catalog name as Trino's connector (`iceberg`), but it's arbitrary.
    catalog = os.environ.get("ICEBERG_SPARK_CATALOG", "iceberg")

    builder = (
        SparkSession.builder.appName("ingest-nuscenes-mini")
        # The tabulario image may preconfigure a default catalog named "rest".
        # Force Spark to use our configured catalog by default.
        .config("spark.sql.defaultCatalog", catalog)
        .config(f"spark.sql.catalog.{catalog}", "org.apache.iceberg.spark.SparkCatalog")
        .config(f"spark.sql.catalog.{catalog}.catalog-impl", "org.apache.iceberg.rest.RESTCatalog")
        .config(f"spark.sql.catalog.{catalog}.uri", polaris_uri)
        # Polaris uses the REST "warehouse" to identify the catalog to use.
        .config(f"spark.sql.catalog.{catalog}.warehouse", polaris_warehouse)
        # Use Iceberg's S3FileIO so Spark can read/write s3:// paths (instead of Hadoop s3a://).
        .config(f"spark.sql.catalog.{catalog}.io-impl", "org.apache.iceberg.aws.s3.S3FileIO")
        .config(f"spark.sql.catalog.{catalog}.s3.endpoint", s3_endpoint)
        .config(f"spark.sql.catalog.{catalog}.s3.path-style-access", "true")
        .config(f"spark.sql.catalog.{catalog}.s3.access-key-id", aws_access_key)
        .config(f"spark.sql.catalog.{catalog}.s3.secret-access-key", aws_secret_key)
        # Polaris requires auth; configure OAuth2 client-credentials flow.
        # Iceberg 1.8.x expects these exact key names.
        .config(f"spark.sql.catalog.{catalog}.oauth2-server-uri", polaris_oauth2_server_uri)
        .config(f"spark.sql.catalog.{catalog}.credential", polaris_credential)
        .config(f"spark.sql.catalog.{catalog}.scope", polaris_scope)
        # Keep older/alternate key spellings too (harmless) to reduce version sensitivity.
        .config(f"spark.sql.catalog.{catalog}.oauth2.server-uri", polaris_oauth2_server_uri)
        .config(f"spark.sql.catalog.{catalog}.oauth2.credential", polaris_credential)
        .config(f"spark.sql.catalog.{catalog}.oauth2.scope", polaris_scope)
        # S3A/MinIO
        .config("spark.hadoop.fs.s3a.endpoint", s3_endpoint)
        .config("spark.hadoop.fs.s3a.access.key", aws_access_key)
        .config("spark.hadoop.fs.s3a.secret.key", aws_secret_key)
        .config("spark.hadoop.fs.s3a.path.style.access", "true")
        .config("spark.hadoop.fs.s3a.connection.ssl.enabled", "false")
        .config("spark.hadoop.fs.s3a.aws.credentials.provider", "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider")
        .config("spark.hadoop.fs.s3a.region", aws_region)
    )

    return builder.getOrCreate()


def main() -> None:
    # Expect the official nuScenes mini extract, which includes v1.0-mini/*.json
    # Place it on the host at ./user_data/nuscenes-mini and it will be mounted to /user_data.
    nuscenes_root = Path(os.environ.get("NUSCENES_DIR", "/user_data/nuscenes-mini")).resolve()
    version_dir = nuscenes_root / "v1.0-mini"

    spark = build_spark()
    catalog = os.environ.get("ICEBERG_SPARK_CATALOG", "iceberg")
    namespace = os.environ.get("ICEBERG_NAMESPACE", "nuscenes")

    if os.environ.get("DEBUG_INGEST_CONFIG") in {"1", "true", "TRUE", "yes", "YES"}:
        keys = [
            f"spark.sql.defaultCatalog",
            f"spark.sql.catalog.{catalog}",
            f"spark.sql.catalog.{catalog}.uri",
            f"spark.sql.catalog.{catalog}.warehouse",
            f"spark.sql.catalog.{catalog}.io-impl",
            f"spark.sql.catalog.{catalog}.s3.endpoint",
            f"spark.sql.catalog.{catalog}.s3.path-style-access",
            "spark.hadoop.fs.s3a.endpoint",
        ]
        print("DEBUG_INGEST_CONFIG=1")
        for k in keys:
            try:
                v = spark.conf.get(k)
            except Exception:
                v = None
            print(f"{k}={v}")

        env_keys = [
            "AWS_S3_ENDPOINT",
            "AWS_ENDPOINT_URL",
            "AWS_ENDPOINT_URL_S3",
            "AWS_REGION",
            "AWS_ACCESS_KEY_ID",
        ]
        for k in env_keys:
            print(f"ENV:{k}={os.environ.get(k)}")

        jvm = spark.sparkContext._jvm
        sysprops = jvm.java.lang.System.getProperties()
        sysprop_keys = [
            "aws.endpointUrl",
            "aws.endpoint.url",
            "aws.s3.endpoint",
            "aws.s3.endpointUrl",
            "software.amazon.awssdk.endpoints.s3",
            "http.proxyHost",
            "http.proxyPort",
            "https.proxyHost",
            "https.proxyPort",
            "http.nonProxyHosts",
        ]
        for k in sysprop_keys:
            v = sysprops.getProperty(k)
            print(f"JVM:{k}={v}")

        if os.environ.get("DEBUG_INGEST_CONFIG_ONLY") in {"1", "true", "TRUE", "yes", "YES"}:
            print("DEBUG_INGEST_CONFIG_ONLY=1 (exiting before ingest)")
            return

    # A tiny, useful subset of the nuScenes tables.
    tables = [
        "scene",
        "sample",
        "sample_data",
        "sample_annotation",
        "ego_pose",
        "sensor",
        "calibrated_sensor",
        "log",
        "map",
        "instance",
        "category",
        "attribute",
        "visibility",
    ]

    # nuScenes mini is commonly extracted as either:
    # - <root>/v1.0-mini/*.json
    # - <root>/v1.0-mini/v1.0-mini/*.json
    # Auto-detect which layout is present.
    if not version_dir.exists():
        raise RuntimeError(
            "Could not find nuScenes mini directory. Expected: "
            f"{version_dir}\n"
            "Set NUSCENES_DIR to the extracted nuscenes-mini root."
        )

    candidate_json_dirs = [version_dir, version_dir / "v1.0-mini"]
    chosen_json_dir: Path | None = None
    for candidate in candidate_json_dirs:
        expected_paths = [candidate / f"{t}.json" for t in tables]
        if any(p.exists() for p in expected_paths):
            chosen_json_dir = candidate
            break

    if chosen_json_dir is None:
        raise RuntimeError(
            "Found nuScenes directory, but no expected JSON files were present.\n"
            f"Looked in: {candidate_json_dirs[0]} and {candidate_json_dirs[1]}\n"
            "Expected at least one of: " + ", ".join(f"{t}.json" for t in tables)
        )

    spark.sql(f"CREATE NAMESPACE IF NOT EXISTS {catalog}.{namespace}")

    loaded = 0
    for t in tables:
        json_path = str((chosen_json_dir / f"{t}.json").as_posix())
        if not Path(json_path).exists():
            # Some releases may omit some tables; skip cleanly.
            print(f"Skipping missing JSON: {json_path}")
            continue

        df = spark.read.option("multiline", "true").json(json_path)

        full_table = f"{catalog}.{namespace}.{t}"
        print(f"Writing {full_table} from {json_path}")

        (
            df.writeTo(full_table)
            .using("iceberg")
            .tableProperty("format-version", "2")
            .createOrReplace()
        )

        loaded += 1

        cnt = spark.table(full_table).count()
        print(f"{full_table}: {cnt} rows")

    print(f"Done. Loaded {loaded} table(s). Next: query via Trino and add dataset in Superset.")


if __name__ == "__main__":
    main()
