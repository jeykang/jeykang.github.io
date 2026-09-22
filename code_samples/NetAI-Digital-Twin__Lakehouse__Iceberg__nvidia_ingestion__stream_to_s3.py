"""
Byte-stream uploader: extracts Parquet files from STORED-mode zip archives
and uploads them directly to MinIO/S3 without deserialization.

This is the ingestion path for Option 2 (incoming real-world data).
After uploading, call register_bronze.py to register the files in Iceberg.

Since the zips use STORED mode (no compression), each Parquet file's bytes
are read verbatim from the zip and uploaded to S3 — no Arrow, no pandas,
no Spark, no memory accumulation.  Peak memory is bounded by the single
largest Parquet file (~200 MB for lidar).

Can also ingest bare Parquet files (not in zips) for new data that arrives
as plain files rather than archives.

Usage:
    # Upload all sensor data from NFS zips to MinIO
    python -m nvidia_ingestion.stream_to_s3 --all

    # Upload a single sensor
    python -m nvidia_ingestion.stream_to_s3 --sensor radar --max-chunks 2

    # Upload bare Parquet files (for incoming real-world data)
    python -m nvidia_ingestion.stream_to_s3 --path /data/incoming/radar_new/*.parquet --table radar_new

    # Dry run (show what would be uploaded)
    python -m nvidia_ingestion.stream_to_s3 --all --dry-run
"""

import argparse
import glob
import io
import os
import sys
import zipfile
from typing import Dict, List, Optional, Tuple

import boto3
from botocore.config import Config as BotoConfig


DEFAULT_SOURCE = (
    "/mnt/datax/hub/datasets--nvidia--PhysicalAI-Autonomous-Vehicles"
    "/snapshots/0c8e5b7813562ab6e907e55db6ead3351922073f"
)
DEFAULT_ENDPOINT = os.environ.get("AWS_S3_ENDPOINT", "http://localhost:9000")
DEFAULT_BUCKET = os.environ.get("S3_BUCKET", "spark1")
DEFAULT_PREFIX = "nvidia_bronze"


def _s3_client(endpoint: str = DEFAULT_ENDPOINT):
    """Create a boto3 S3 client for MinIO."""
    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID", "minioadmin"),
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY", "minioadmin"),
        region_name=os.environ.get("AWS_REGION", "us-east-1"),
        config=BotoConfig(signature_version="s3v4"),
    )


def _clip_id_from_filename(name: str) -> str:
    """Extract clip UUID from a filename like 'abcd1234-...-5678.egomotion.parquet'."""
    return name.split(".")[0]


# ---------------------------------------------------------------------------
# Zip-based streaming upload
# ---------------------------------------------------------------------------

def stream_zip_to_s3(
    zip_path: str,
    table_name: str,
    s3_client,
    bucket: str = DEFAULT_BUCKET,
    prefix: str = DEFAULT_PREFIX,
    suffix_filter: str = ".parquet",
    max_clips: int = 0,
    dry_run: bool = False,
) -> Tuple[int, int]:
    """Stream Parquet files from a zip archive to S3.

    Returns (files_uploaded, bytes_uploaded).
    """
    files, total_bytes = 0, 0

    with zipfile.ZipFile(zip_path, "r") as z:
        entries = [i for i in z.infolist() if i.filename.endswith(suffix_filter)]
        if max_clips:
            entries = entries[:max_clips]

        for info in entries:
            s3_key = f"{prefix}/{table_name}/{info.filename}"

            if dry_run:
                print(f"  [DRY] {info.filename} → s3://{bucket}/{s3_key} "
                      f"({info.file_size:,} bytes)")
                files += 1
                total_bytes += info.file_size
                continue

            # Read raw bytes — no deserialization.  For STORED zips this is
            # a simple read at a known offset, byte-for-byte identical to
            # the original Parquet file.
            raw = z.read(info.filename)
            s3_client.put_object(
                Bucket=bucket,
                Key=s3_key,
                Body=raw,
                ContentLength=len(raw),
            )
            files += 1
            total_bytes += len(raw)

    return files, total_bytes


def stream_bare_parquet_to_s3(
    file_path: str,
    table_name: str,
    s3_client,
    bucket: str = DEFAULT_BUCKET,
    prefix: str = DEFAULT_PREFIX,
    dry_run: bool = False,
) -> Tuple[int, int]:
    """Upload a single bare Parquet file to S3.

    Returns (1, bytes_uploaded).
    """
    basename = os.path.basename(file_path)
    s3_key = f"{prefix}/{table_name}/{basename}"
    size = os.path.getsize(file_path)

    if dry_run:
        print(f"  [DRY] {file_path} → s3://{bucket}/{s3_key} ({size:,} bytes)")
        return 1, size

    with open(file_path, "rb") as f:
        s3_client.put_object(Bucket=bucket, Key=s3_key, Body=f, ContentLength=size)

    return 1, size


# ---------------------------------------------------------------------------
# Sensor group uploaders
# ---------------------------------------------------------------------------

BARE_PARQUET_TABLES = {
    "clip_index": "clip_index.parquet",
    "data_collection": "metadata/data_collection.parquet",
    "sensor_presence": "metadata/sensor_presence.parquet",
}

CALIBRATION_TABLES = {
    "camera_intrinsics": "calibration/camera_intrinsics",
    "sensor_extrinsics": "calibration/sensor_extrinsics",
    "vehicle_dimensions": "calibration/vehicle_dimensions",
}


def upload_all(
    source: str = DEFAULT_SOURCE,
    sensor: Optional[str] = None,
    max_chunks: int = 0,
    max_clips: int = 0,
    endpoint: str = DEFAULT_ENDPOINT,
    bucket: str = DEFAULT_BUCKET,
    prefix: str = DEFAULT_PREFIX,
    dry_run: bool = False,
) -> Dict[str, Tuple[int, int]]:
    """Upload all (or filtered) sensor data from NFS to S3.

    Returns dict of table_name → (total_files, total_bytes).
    """
    s3 = _s3_client(endpoint)
    results: Dict[str, Tuple[int, int]] = {}

    # 1. Bare Parquets
    if sensor is None or sensor == "metadata":
        for table, rel_path in BARE_PARQUET_TABLES.items():
            path = os.path.join(source, rel_path)
            if os.path.exists(path):
                print(f"[UPLOAD] {table}")
                r = stream_bare_parquet_to_s3(
                    path, table, s3, bucket, prefix, dry_run
                )
                results[table] = r
                print(f"  {r[0]} files, {r[1]:,} bytes")

    # 2. Calibration directories
    if sensor is None or sensor == "calibration":
        for table, rel_dir in CALIBRATION_TABLES.items():
            src_dir = os.path.join(source, rel_dir)
            if not os.path.isdir(src_dir):
                continue
            print(f"[UPLOAD] {table}")
            total_f, total_b = 0, 0
            for p in sorted(glob.glob(os.path.join(src_dir, "*.parquet"))):
                f, b = stream_bare_parquet_to_s3(
                    p, table, s3, bucket, prefix, dry_run
                )
                total_f += f
                total_b += b
            results[table] = (total_f, total_b)
            print(f"  {total_f} files, {total_b:,} bytes")

    # 3. Egomotion
    if sensor is None or sensor == "labels":
        _upload_zipped_sensor(
            source, "labels/egomotion", "egomotion",
            s3, bucket, prefix, max_chunks, max_clips, dry_run, results
        )

    # 4. Lidar
    if sensor is None or sensor == "lidar":
        _upload_zipped_sensor(
            source, "lidar/lidar_top_360fov", "lidar",
            s3, bucket, prefix, max_chunks, max_clips, dry_run, results
        )

    # 5. Radar
    if sensor is None or sensor == "radar":
        radar_dir = os.path.join(source, "radar")
        if os.path.isdir(radar_dir):
            for sub in sorted(os.listdir(radar_dir)):
                if os.path.isdir(os.path.join(radar_dir, sub)):
                    safe = sub.replace("-", "_")
                    _upload_zipped_sensor(
                        source, f"radar/{sub}", f"radar_{safe}",
                        s3, bucket, prefix, max_chunks, max_clips,
                        dry_run, results
                    )

    # 6. Camera metadata
    if sensor is None or sensor == "camera":
        cam_dir = os.path.join(source, "camera")
        if os.path.isdir(cam_dir):
            for sub in sorted(os.listdir(cam_dir)):
                if not os.path.isdir(os.path.join(cam_dir, sub)):
                    continue
                safe = sub.replace("-", "_")
                _upload_zipped_sensor(
                    source, f"camera/{sub}", f"cam_{safe}_ts",
                    s3, bucket, prefix, max_chunks, max_clips,
                    dry_run, results,
                    suffix_filter=".timestamps.parquet",
                )
                _upload_zipped_sensor(
                    source, f"camera/{sub}", f"cam_{safe}_blur",
                    s3, bucket, prefix, max_chunks, max_clips,
                    dry_run, results,
                    suffix_filter=".blurred_boxes.parquet",
                )

    return results


def _upload_zipped_sensor(
    source: str,
    rel_dir: str,
    table_name: str,
    s3_client,
    bucket: str,
    prefix: str,
    max_chunks: int,
    max_clips: int,
    dry_run: bool,
    results: Dict[str, Tuple[int, int]],
    suffix_filter: str = ".parquet",
):
    """Upload all zips in a sensor subdirectory to S3."""
    full_dir = os.path.join(source, rel_dir)
    if not os.path.isdir(full_dir):
        return
    zips = sorted(glob.glob(os.path.join(full_dir, "*.zip")))
    if max_chunks:
        zips = zips[:max_chunks]
    if not zips:
        return

    print(f"[UPLOAD] {table_name} ({len(zips)} zips)")
    total_f, total_b = 0, 0
    for zp in zips:
        f, b = stream_zip_to_s3(
            zp, table_name, s3_client, bucket, prefix,
            suffix_filter=suffix_filter, max_clips=max_clips, dry_run=dry_run,
        )
        total_f += f
        total_b += b
    results[table_name] = (total_f, total_b)
    print(f"  {total_f} files, {total_b:,} bytes")


# ---------------------------------------------------------------------------
# Bare file ingestion (for incoming real-world data)
# ---------------------------------------------------------------------------

def upload_incoming_files(
    file_paths: List[str],
    table_name: str,
    endpoint: str = DEFAULT_ENDPOINT,
    bucket: str = DEFAULT_BUCKET,
    prefix: str = DEFAULT_PREFIX,
    dry_run: bool = False,
) -> Tuple[int, int]:
    """Upload bare Parquet files (new incoming data) to S3.

    This is the entry point for real-world data as it arrives — just point
    it at a list of Parquet files and a target table name.
    """
    s3 = _s3_client(endpoint)
    total_f, total_b = 0, 0

    print(f"[UPLOAD] {table_name} ({len(file_paths)} files)")
    for path in file_paths:
        f, b = stream_bare_parquet_to_s3(
            path, table_name, s3, bucket, prefix, dry_run
        )
        total_f += f
        total_b += b

    print(f"  {total_f} files, {total_b:,} bytes")
    return total_f, total_b


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Stream Parquet files from NFS zips to MinIO/S3"
    )
    parser.add_argument("--source", default=DEFAULT_SOURCE,
                        help="Dataset root path")
    parser.add_argument("--sensor", type=str, default=None,
                        help="Sensor group: lidar, radar, camera, labels, "
                             "metadata, calibration")
    parser.add_argument("--all", action="store_true",
                        help="Upload all sensor groups")
    parser.add_argument("--max-chunks", type=int, default=0,
                        help="Limit zip chunks per sensor (0=all)")
    parser.add_argument("--max-clips", type=int, default=0,
                        help="Limit clips per chunk (0=all)")
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--bucket", default=DEFAULT_BUCKET)
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--dry-run", action="store_true",
                        help="Show uploads without executing")

    # For incoming data
    parser.add_argument("--path", type=str, default=None,
                        help="Glob pattern for incoming Parquet files")
    parser.add_argument("--table", type=str, default=None,
                        help="Target table name (for --path)")

    args = parser.parse_args()

    if args.path and args.table:
        # Incoming data mode
        files = sorted(glob.glob(args.path))
        if not files:
            print(f"No files matching {args.path}")
            sys.exit(1)
        upload_incoming_files(
            files, args.table, args.endpoint, args.bucket,
            args.prefix, args.dry_run,
        )
    elif args.all or args.sensor:
        results = upload_all(
            source=args.source,
            sensor=args.sensor if not args.all else None,
            max_chunks=args.max_chunks,
            max_clips=args.max_clips,
            endpoint=args.endpoint,
            bucket=args.bucket,
            prefix=args.prefix,
            dry_run=args.dry_run,
        )
        print(f"\n{'=' * 60}")
        print("UPLOAD SUMMARY")
        print(f"{'=' * 60}")
        grand_files, grand_bytes = 0, 0
        for table, (f, b) in sorted(results.items()):
            print(f"  {table}: {f} files, {b:,} bytes")
            grand_files += f
            grand_bytes += b
        print(f"\n  TOTAL: {grand_files} files, {grand_bytes:,} bytes "
              f"({grand_bytes / 1e9:.1f} GB)")
    else:
        print("Specify --all, --sensor <name>, or --path <glob> --table <name>")
        sys.exit(1)


if __name__ == "__main__":
    main()
