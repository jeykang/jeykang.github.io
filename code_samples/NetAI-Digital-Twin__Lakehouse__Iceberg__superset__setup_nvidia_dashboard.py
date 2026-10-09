#!/usr/bin/env python3
"""
Provision Superset datasets, charts, and dashboard for the Nvidia PhysicalAI
autonomous-vehicles dataset.  Talks to the Superset REST API.

Usage:
    python3 superset/setup_nvidia_dashboard.py
"""

import json
import sys
import time
import requests

SUPERSET_URL = "http://localhost:8088"
USERNAME = "admin"
PASSWORD = "admin"
DATABASE_ID = 1  # Pre-existing Trino connection


# ---------------------------------------------------------------------------
# Auth — use a requests.Session so cookies (incl. CSRF session) persist
# ---------------------------------------------------------------------------

SESSION = requests.Session()


def init_session():
    """Login + fetch CSRF token; sets session-wide headers."""
    # 1. Login → get JWT
    r = SESSION.post(f"{SUPERSET_URL}/api/v1/security/login", json={
        "username": USERNAME, "password": PASSWORD,
        "provider": "db", "refresh": True,
    })
    r.raise_for_status()
    access_token = r.json()["access_token"]

    # 2. Fetch CSRF token (also sets the session cookie we need)
    SESSION.headers.update({"Authorization": f"Bearer {access_token}"})
    r = SESSION.get(f"{SUPERSET_URL}/api/v1/security/csrf_token/")
    r.raise_for_status()
    csrf = r.json()["result"]

    # 3. Set persistent headers
    SESSION.headers.update({
        "Content-Type": "application/json",
        "X-CSRFToken": csrf,
        "Referer": SUPERSET_URL,
    })
    print(f"  Authenticated as {USERNAME}, CSRF OK")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def api_get(path):
    r = SESSION.get(f"{SUPERSET_URL}{path}")
    r.raise_for_status()
    return r.json()


def api_post(path, payload):
    r = SESSION.post(f"{SUPERSET_URL}{path}", json=payload)
    if r.status_code >= 400:
        print(f"  POST {path} -> {r.status_code}: {r.text[:500]}")
    r.raise_for_status()
    return r.json()


def api_put(path, payload):
    r = SESSION.put(f"{SUPERSET_URL}{path}", json=payload)
    if r.status_code >= 400:
        print(f"  PUT {path} -> {r.status_code}: {r.text[:500]}")
    r.raise_for_status()
    return r.json()


# ---------------------------------------------------------------------------
# Dataset creation (virtual = SQL-based)
# ---------------------------------------------------------------------------

DATASETS = [
    # ── 1. Clip metadata (Gold) ──
    {
        "name": "Clip Overview (Gold)",
        "schema": "nvidia_gold_blob",
        "sql": """
SELECT
    split,
    country,
    month,
    hour_of_day,
    platform_class,
    clip_is_valid,
    radar_config,
    length  AS vehicle_length,
    width   AS vehicle_width,
    height  AS vehicle_height,
    wheelbase,
    -- sensor availability counts (cameras)
    CAST(camera_cross_left_120fov   AS INTEGER)
  + CAST(camera_cross_right_120fov  AS INTEGER)
  + CAST(camera_front_tele_30fov    AS INTEGER)
  + CAST(camera_front_wide_120fov   AS INTEGER)
  + CAST(camera_rear_left_70fov     AS INTEGER)
  + CAST(camera_rear_right_70fov    AS INTEGER)
  + CAST(camera_rear_tele_30fov     AS INTEGER) AS num_cameras,
    CAST(lidar_top_360fov AS INTEGER)           AS has_lidar,
    -- radar count
    CAST(radar_corner_front_left_srr_0  AS INTEGER)
  + CAST(radar_corner_front_left_srr_3  AS INTEGER)
  + CAST(radar_corner_front_right_srr_0 AS INTEGER)
  + CAST(radar_corner_front_right_srr_3 AS INTEGER)
  + CAST(radar_corner_rear_left_srr_0   AS INTEGER)
  + CAST(radar_corner_rear_left_srr_3   AS INTEGER)
  + CAST(radar_corner_rear_right_srr_0  AS INTEGER)
  + CAST(radar_corner_rear_right_srr_3  AS INTEGER)
  + CAST(radar_front_center_imaging_lrr_1 AS INTEGER)
  + CAST(radar_front_center_mrr_2       AS INTEGER)
  + CAST(radar_front_center_srr_0       AS INTEGER)
  + CAST(radar_rear_left_mrr_2          AS INTEGER)
  + CAST(radar_rear_left_srr_0          AS INTEGER)
  + CAST(radar_rear_right_mrr_2         AS INTEGER)
  + CAST(radar_rear_right_srr_0         AS INTEGER)
  + CAST(radar_side_left_srr_0          AS INTEGER)
  + CAST(radar_side_left_srr_3          AS INTEGER)
  + CAST(radar_side_right_srr_0         AS INTEGER)
  + CAST(radar_side_right_srr_3         AS INTEGER) AS num_radars
FROM iceberg.nvidia_gold_blob.sensor_fusion_clip
""",
    },
    # ── 2. Radar detections (Gold) ──
    {
        "name": "Radar Detections (Gold)",
        "schema": "nvidia_gold_blob",
        "sql": """
SELECT
    clip_id,
    sensor_name,
    timestamp,
    azimuth,
    elevation,
    distance,
    radial_velocity,
    rcs,
    snr,
    num_returns,
    detection_index,
    ego_sample_count,
    -- derived: range bucket
    CASE
        WHEN distance < 25  THEN '0-25 m'
        WHEN distance < 50  THEN '25-50 m'
        WHEN distance < 100 THEN '50-100 m'
        WHEN distance < 200 THEN '100-200 m'
        ELSE '200+ m'
    END AS range_bucket,
    -- derived: sensor type category
    CASE
        WHEN sensor_name LIKE '%lrr%' THEN 'LRR (Long Range)'
        WHEN sensor_name LIKE '%mrr%' THEN 'MRR (Mid Range)'
        WHEN sensor_name LIKE '%srr_0%' THEN 'SRR Mode-0'
        WHEN sensor_name LIKE '%srr_3%' THEN 'SRR Mode-3'
        ELSE 'Other'
    END AS radar_type,
    -- derived: position
    CASE
        WHEN sensor_name LIKE '%front_center%' THEN 'Front Center'
        WHEN sensor_name LIKE '%front_left%'   THEN 'Front Left'
        WHEN sensor_name LIKE '%front_right%'  THEN 'Front Right'
        WHEN sensor_name LIKE '%rear_left%'    THEN 'Rear Left'
        WHEN sensor_name LIKE '%rear_right%'   THEN 'Rear Right'
        WHEN sensor_name LIKE '%side_left%'    THEN 'Side Left'
        WHEN sensor_name LIKE '%side_right%'   THEN 'Side Right'
        ELSE 'Unknown'
    END AS radar_position
FROM iceberg.nvidia_gold_blob.radar_ego_fusion
""",
    },
    # ── 3. Ego-motion dynamics (Silver) ──
    {
        "name": "Ego-Motion Dynamics (Silver)",
        "schema": "nvidia_silver_blob",
        "sql": """
SELECT
    clip_id,
    timestamp,
    x, y, z,
    qw, qx, qy, qz,
    vx, vy, vz,
    ax, ay, az,
    curvature,
    SQRT(vx*vx + vy*vy + vz*vz)       AS speed,
    SQRT(ax*ax + ay*ay + az*az)        AS total_accel,
    SQRT(vx*vx + vy*vy + vz*vz) * 3.6 AS speed_kmh,
    ABS(curvature)                      AS abs_curvature
FROM iceberg.nvidia_silver_blob.egomotion
""",
    },
    # ── 4. Lakehouse layer comparison ──
    {
        "name": "Lakehouse Layer Row Counts",
        "schema": "nvidia_bronze_blob",
        "sql": """
SELECT 'Bronze' AS layer, 'clip_index' AS tbl, COUNT(*) AS row_count FROM iceberg.nvidia_bronze_blob.clip_index
UNION ALL SELECT 'Bronze', 'egomotion', COUNT(*) FROM iceberg.nvidia_bronze_blob.egomotion
UNION ALL SELECT 'Bronze', 'camera_intrinsics', COUNT(*) FROM iceberg.nvidia_bronze_blob.camera_intrinsics
UNION ALL SELECT 'Bronze', 'sensor_extrinsics', COUNT(*) FROM iceberg.nvidia_bronze_blob.sensor_extrinsics
UNION ALL SELECT 'Bronze', 'sensor_presence', COUNT(*) FROM iceberg.nvidia_bronze_blob.sensor_presence
UNION ALL SELECT 'Bronze', 'data_collection', COUNT(*) FROM iceberg.nvidia_bronze_blob.data_collection
UNION ALL SELECT 'Bronze', 'vehicle_dimensions', COUNT(*) FROM iceberg.nvidia_bronze_blob.vehicle_dimensions
UNION ALL SELECT 'Bronze', 'lidar', COUNT(*) FROM iceberg.nvidia_bronze_blob.lidar
UNION ALL SELECT 'Silver', 'clip_index', COUNT(*) FROM iceberg.nvidia_silver_blob.clip_index
UNION ALL SELECT 'Silver', 'egomotion', COUNT(*) FROM iceberg.nvidia_silver_blob.egomotion
UNION ALL SELECT 'Silver', 'camera_intrinsics', COUNT(*) FROM iceberg.nvidia_silver_blob.camera_intrinsics
UNION ALL SELECT 'Silver', 'sensor_extrinsics', COUNT(*) FROM iceberg.nvidia_silver_blob.sensor_extrinsics
UNION ALL SELECT 'Silver', 'sensor_presence', COUNT(*) FROM iceberg.nvidia_silver_blob.sensor_presence
UNION ALL SELECT 'Silver', 'data_collection', COUNT(*) FROM iceberg.nvidia_silver_blob.data_collection
UNION ALL SELECT 'Silver', 'vehicle_dimensions', COUNT(*) FROM iceberg.nvidia_silver_blob.vehicle_dimensions
UNION ALL SELECT 'Silver', 'lidar', COUNT(*) FROM iceberg.nvidia_silver_blob.lidar
UNION ALL SELECT 'Gold', 'sensor_fusion_clip', COUNT(*) FROM iceberg.nvidia_gold_blob.sensor_fusion_clip
UNION ALL SELECT 'Gold', 'lidar_with_ego', COUNT(*) FROM iceberg.nvidia_gold_blob.lidar_with_ego
UNION ALL SELECT 'Gold', 'radar_ego_fusion', COUNT(*) FROM iceberg.nvidia_gold_blob.radar_ego_fusion
""",
    },
    # ── 5. Iceberg table metadata ──
    {
        "name": "Iceberg Snapshot Metadata",
        "schema": "nvidia_gold_blob",
        "sql": """
SELECT
    'sensor_fusion_clip' AS table_name,
    committed_at,
    snapshot_id,
    operation,
    summary
FROM iceberg.nvidia_gold_blob."sensor_fusion_clip$snapshots"
UNION ALL
SELECT
    'lidar_with_ego',
    committed_at,
    snapshot_id,
    operation,
    summary
FROM iceberg.nvidia_gold_blob."lidar_with_ego$snapshots"
UNION ALL
SELECT
    'radar_ego_fusion',
    committed_at,
    snapshot_id,
    operation,
    summary
FROM iceberg.nvidia_gold_blob."radar_ego_fusion$snapshots"
""",
    },
    # ── 6. Radar sensor comparison ──
    {
        "name": "Radar Sensor Statistics",
        "schema": "nvidia_gold_blob",
        "sql": """
SELECT
    sensor_name,
    COUNT(*)                  AS detection_count,
    ROUND(AVG(distance), 2)   AS avg_distance_m,
    ROUND(AVG(radial_velocity), 2) AS avg_radial_vel,
    ROUND(AVG(rcs), 2)        AS avg_rcs_dbsm,
    ROUND(AVG(snr), 2)        AS avg_snr_db,
    ROUND(STDDEV(distance), 2) AS std_distance,
    ROUND(MIN(distance), 2)   AS min_distance,
    ROUND(MAX(distance), 2)   AS max_distance,
    CASE
        WHEN sensor_name LIKE '%lrr%' THEN 'LRR'
        WHEN sensor_name LIKE '%mrr%' THEN 'MRR'
        ELSE 'SRR'
    END AS radar_type,
    CASE
        WHEN sensor_name LIKE '%front_center%' THEN 'Front Center'
        WHEN sensor_name LIKE '%front_left%'   THEN 'Front Left'
        WHEN sensor_name LIKE '%front_right%'  THEN 'Front Right'
        WHEN sensor_name LIKE '%rear_left%'    THEN 'Rear Left'
        WHEN sensor_name LIKE '%rear_right%'   THEN 'Rear Right'
        WHEN sensor_name LIKE '%side_left%'    THEN 'Side Left'
        WHEN sensor_name LIKE '%side_right%'   THEN 'Side Right'
        ELSE 'Unknown'
    END AS radar_position
FROM iceberg.nvidia_gold_blob.radar_ego_fusion
GROUP BY sensor_name
""",
    },
]

# ---------------------------------------------------------------------------
# Chart definitions
# ---------------------------------------------------------------------------

CHARTS = [
    # ── 1. Clips by Country (Bar) ──
    {
        "slice_name": "Clips by Country",
        "viz_type": "echarts_timeseries_bar",
        "dataset_name": "Clip Overview (Gold)",
        "params": {
            "viz_type": "echarts_timeseries_bar",
            "x_axis": "country",
            "metrics": [{"expressionType": "SIMPLE", "column": {"column_name": "country"}, "aggregate": "COUNT", "label": "Clip Count"}],
            "groupby": [],
            "order_desc": True,
            "row_limit": 25,
            "truncate_metric": True,
            "show_legend": False,
            "rich_tooltip": True,
            "color_scheme": "supersetColors",
        },
    },
    # ── 2. Train/Val/Test Split (Pie) ──
    {
        "slice_name": "Train / Val / Test Split",
        "viz_type": "pie",
        "dataset_name": "Clip Overview (Gold)",
        "params": {
            "viz_type": "pie",
            "groupby": ["split"],
            "metric": {"expressionType": "SIMPLE", "column": {"column_name": "split"}, "aggregate": "COUNT", "label": "count"},
            "color_scheme": "supersetColors",
            "show_labels": True,
            "label_type": "key_percent",
            "show_legend": True,
            "innerRadius": 40,
            "outerRadius": 80,
            "row_limit": 10,
        },
    },
    # ── 3. Collection Hours Heatmap (by month x hour) ──
    {
        "slice_name": "Collection Time Heatmap (Month x Hour)",
        "viz_type": "heatmap",
        "dataset_name": "Clip Overview (Gold)",
        "params": {
            "viz_type": "heatmap",
            "all_columns_x": "month",
            "all_columns_y": "hour_of_day",
            "metric": {"expressionType": "SIMPLE", "column": {"column_name": "month"}, "aggregate": "COUNT", "label": "count"},
            "linear_color_scheme": "blue_white_yellow",
            "xscale_interval": 1,
            "yscale_interval": 1,
            "canvas_image_rendering": "auto",
            "normalize_across": "heatmap",
            "show_legend": True,
            "show_perc": True,
            "show_values": False,
            "row_limit": 300,
        },
    },
    # ── 4. Platform Class Split (Pie) ──
    {
        "slice_name": "Platform Class Distribution",
        "viz_type": "pie",
        "dataset_name": "Clip Overview (Gold)",
        "params": {
            "viz_type": "pie",
            "groupby": ["platform_class"],
            "metric": {"expressionType": "SIMPLE", "column": {"column_name": "platform_class"}, "aggregate": "COUNT", "label": "count"},
            "color_scheme": "supersetColors",
            "show_labels": True,
            "label_type": "key_percent",
            "show_legend": True,
            "row_limit": 10,
        },
    },
    # ── 5. Radar Detections per Sensor (Horizontal Bar) ──
    {
        "slice_name": "Radar Detection Volume by Sensor",
        "viz_type": "dist_bar",
        "dataset_name": "Radar Sensor Statistics",
        "params": {
            "viz_type": "dist_bar",
            "groupby": ["sensor_name"],
            "metrics": [{"expressionType": "SIMPLE", "column": {"column_name": "detection_count"}, "aggregate": "SUM", "label": "Total Detections"}],
            "columns": [],
            "color_scheme": "supersetColors",
            "show_legend": False,
            "y_axis_format": ",d",
            "order_bars": True,
            "row_limit": 25,
        },
    },
    # ── 6. Avg Detection Distance by Radar Type (Grouped Bar) ──
    {
        "slice_name": "Avg Detection Distance by Radar Type",
        "viz_type": "dist_bar",
        "dataset_name": "Radar Sensor Statistics",
        "params": {
            "viz_type": "dist_bar",
            "groupby": ["radar_position"],
            "metrics": [{"expressionType": "SIMPLE", "column": {"column_name": "avg_distance_m"}, "aggregate": "AVG", "label": "Avg Distance (m)"}],
            "columns": ["radar_type"],
            "color_scheme": "supersetColors",
            "show_legend": True,
            "y_axis_format": ",.1f",
            "row_limit": 50,
        },
    },
    # ── 7. Radar Range Distribution (Pie) ──
    {
        "slice_name": "Radar Detection Range Distribution",
        "viz_type": "pie",
        "dataset_name": "Radar Detections (Gold)",
        "params": {
            "viz_type": "pie",
            "groupby": ["range_bucket"],
            "metric": {"expressionType": "SIMPLE", "column": {"column_name": "range_bucket"}, "aggregate": "COUNT", "label": "Detections"},
            "color_scheme": "supersetColors",
            "show_labels": True,
            "label_type": "key_percent",
            "show_legend": True,
            "row_limit": 10,
        },
    },
    # ── 8. Radar RCS vs Distance (Scatter) ──
    {
        "slice_name": "Radar RCS vs Distance (sampled)",
        "viz_type": "echarts_timeseries_scatter",
        "dataset_name": "Radar Detections (Gold)",
        "params": {
            "viz_type": "echarts_timeseries_scatter",
            "x_axis": "distance",
            "metrics": [{"expressionType": "SIMPLE", "column": {"column_name": "rcs"}, "aggregate": "AVG", "label": "Avg RCS (dBsm)"}],
            "groupby": ["radar_type"],
            "color_scheme": "supersetColors",
            "show_legend": True,
            "row_limit": 5000,
            "truncate_metric": True,
            "rich_tooltip": True,
        },
    },
    # ── 9. Ego Speed Distribution (Histogram) ──
    {
        "slice_name": "Ego Vehicle Speed Distribution",
        "viz_type": "histogram_v2",
        "dataset_name": "Ego-Motion Dynamics (Silver)",
        "params": {
            "viz_type": "histogram_v2",
            "all_columns_x": ["speed_kmh"],
            "color_scheme": "supersetColors",
            "normalize": False,
            "cumulative": False,
            "row_limit": 50000,
        },
    },
    # ── 10. Ego Acceleration vs Curvature ──
    {
        "slice_name": "Acceleration vs Curvature",
        "viz_type": "bubble_v2",
        "dataset_name": "Ego-Motion Dynamics (Silver)",
        "params": {
            "viz_type": "bubble_v2",
            "x": {"expressionType": "SIMPLE", "column": {"column_name": "abs_curvature"}, "aggregate": "AVG", "label": "Avg |Curvature|"},
            "y": {"expressionType": "SIMPLE", "column": {"column_name": "total_accel"}, "aggregate": "AVG", "label": "Avg Total Accel"},
            "size": {"expressionType": "SIMPLE", "column": {"column_name": "speed_kmh"}, "aggregate": "AVG", "label": "Avg Speed (km/h)"},
            "entity": "clip_id",
            "color_scheme": "supersetColors",
            "show_legend": True,
            "max_bubble_size": 50,
            "row_limit": 50000,
        },
    },
    # ── 11. Medallion Architecture Row Counts (Bar) ──
    {
        "slice_name": "Medallion Architecture: Row Counts",
        "viz_type": "dist_bar",
        "dataset_name": "Lakehouse Layer Row Counts",
        "params": {
            "viz_type": "dist_bar",
            "groupby": ["tbl"],
            "metrics": [{"expressionType": "SIMPLE", "column": {"column_name": "row_count"}, "aggregate": "SUM", "label": "Rows"}],
            "columns": ["layer"],
            "color_scheme": "supersetColors",
            "show_legend": True,
            "y_axis_format": ",d",
            "row_limit": 50,
        },
    },
    # ── 12. Country x Split Heatmap ──
    {
        "slice_name": "Country x Split Coverage",
        "viz_type": "heatmap",
        "dataset_name": "Clip Overview (Gold)",
        "params": {
            "viz_type": "heatmap",
            "all_columns_x": "split",
            "all_columns_y": "country",
            "metric": {"expressionType": "SIMPLE", "column": {"column_name": "split"}, "aggregate": "COUNT", "label": "count"},
            "linear_color_scheme": "blue_white_yellow",
            "canvas_image_rendering": "auto",
            "normalize_across": "y",
            "show_legend": True,
            "show_perc": True,
            "show_values": True,
            "row_limit": 300,
        },
    },
    # ── 13. Sensor Availability per Camera ──
    {
        "slice_name": "Camera Availability Across Clips",
        "viz_type": "dist_bar",
        "dataset_name": "Clip Overview (Gold)",
        "params": {
            "viz_type": "dist_bar",
            "groupby": ["num_cameras"],
            "metrics": [{"expressionType": "SIMPLE", "column": {"column_name": "num_cameras"}, "aggregate": "COUNT", "label": "Clip Count"}],
            "columns": ["split"],
            "color_scheme": "supersetColors",
            "show_legend": True,
            "y_axis_format": ",d",
            "row_limit": 20,
        },
    },
    # ── 14. Big Numbers KPI row ──
    {
        "slice_name": "Total Clips",
        "viz_type": "big_number_total",
        "dataset_name": "Clip Overview (Gold)",
        "params": {
            "viz_type": "big_number_total",
            "metric": {"expressionType": "SIMPLE", "column": {"column_name": "split"}, "aggregate": "COUNT", "label": "Total Clips"},
            "header_font_size": 0.3,
            "subheader_font_size": 0.15,
        },
    },
    {
        "slice_name": "Total Countries",
        "viz_type": "big_number_total",
        "dataset_name": "Clip Overview (Gold)",
        "params": {
            "viz_type": "big_number_total",
            "metric": {"expressionType": "SIMPLE", "column": {"column_name": "country"}, "aggregate": "COUNT_DISTINCT", "label": "Countries"},
            "header_font_size": 0.3,
            "subheader_font_size": 0.15,
        },
    },
    {
        "slice_name": "Total Radar Detections",
        "viz_type": "big_number_total",
        "dataset_name": "Radar Sensor Statistics",
        "params": {
            "viz_type": "big_number_total",
            "metric": {"expressionType": "SIMPLE", "column": {"column_name": "detection_count"}, "aggregate": "SUM", "label": "Radar Detections"},
            "header_font_size": 0.3,
            "subheader_font_size": 0.15,
        },
    },
    {
        "slice_name": "Radar Sensors",
        "viz_type": "big_number_total",
        "dataset_name": "Radar Sensor Statistics",
        "params": {
            "viz_type": "big_number_total",
            "metric": {"expressionType": "SIMPLE", "column": {"column_name": "sensor_name"}, "aggregate": "COUNT", "label": "Radar Sensors"},
            "header_font_size": 0.3,
            "subheader_font_size": 0.15,
        },
    },
]


# ---------------------------------------------------------------------------
# Dashboard layout
# ---------------------------------------------------------------------------

def build_dashboard_layout(chart_ids):
    """Build a Superset dashboard JSON layout.

    chart_ids is a dict  { chart_name: chart_id }.
    Follows the v2 format: charts sit directly inside ROWs (no COLUMN
    wrapper), parents chain is [ROOT_ID, GRID_ID, ROW-xxx].
    """
    # Row definitions (each row is a list of (chart_name, width) tuples)
    rows = [
        # Row 0: KPI big numbers
        [("Total Clips", 3), ("Total Countries", 3), ("Total Radar Detections", 3), ("Radar Sensors", 3)],
        # Row 1: Geographic & temporal overview
        [("Clips by Country", 6), ("Collection Time Heatmap (Month x Hour)", 6)],
        # Row 2: Split & platform
        [("Train / Val / Test Split", 4), ("Platform Class Distribution", 4), ("Camera Availability Across Clips", 4)],
        # Row 3: Radar overview
        [("Radar Detection Volume by Sensor", 6), ("Radar Detection Range Distribution", 6)],
        # Row 4: Radar analysis
        [("Avg Detection Distance by Radar Type", 6), ("Radar RCS vs Distance (sampled)", 6)],
        # Row 5: Ego dynamics
        [("Ego Vehicle Speed Distribution", 6), ("Acceleration vs Curvature", 6)],
        # Row 6: Architecture & coverage
        [("Medallion Architecture: Row Counts", 6), ("Country x Split Coverage", 6)],
    ]

    components = {
        "DASHBOARD_VERSION_KEY": "v2",
        "ROOT_ID": {"type": "ROOT", "id": "ROOT_ID", "children": ["GRID_ID"]},
        "GRID_ID": {"type": "GRID", "id": "GRID_ID", "children": [], "parents": ["ROOT_ID"]},
        "HEADER_ID": {"type": "HEADER", "id": "HEADER_ID", "meta": {"text": "Nvidia PhysicalAI AV Dataset"}},
    }

    for row_idx, row_charts in enumerate(rows):
        row_id = f"ROW-nvidia-r{row_idx}"
        components[row_id] = {
            "type": "ROW",
            "id": row_id,
            "children": [],
            "parents": ["ROOT_ID", "GRID_ID"],
            "meta": {"0": "ROOT_ID", "background": "BACKGROUND_TRANSPARENT"},
        }
        components["GRID_ID"]["children"].append(row_id)

        for chart_name, width in row_charts:
            if chart_name not in chart_ids:
                continue
            cid = chart_ids[chart_name]
            chart_key = f"CHART-explore-{cid}-1"

            # KPI row is shorter
            height = 20 if row_idx == 0 else 50

            components[chart_key] = {
                "type": "CHART",
                "id": chart_key,
                "children": [],
                "parents": ["ROOT_ID", "GRID_ID", row_id],
                "meta": {
                    "chartId": cid,
                    "width": width,
                    "height": height,
                    "sliceName": chart_name,
                },
            }
            components[row_id]["children"].append(chart_key)

    return json.dumps(components)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Nvidia PhysicalAI — Superset Dashboard Provisioning")
    print("=" * 60)

    init_session()

    # 1. Create datasets
    print("\n[1/3] Creating datasets …")
    dataset_ids = {}
    for ds in DATASETS:
        name = ds["name"]
        payload = {
            "database": DATABASE_ID,
            "schema": ds["schema"],
            "table_name": name,
            "sql": ds["sql"],
        }
        try:
            resp = api_post("/api/v1/dataset/", payload)
            ds_id = resp["id"]
            dataset_ids[name] = ds_id
            print(f"  + {name} (id={ds_id})")
        except requests.HTTPError as e:
            # Dataset may already exist — search for it
            search = api_get(
                f"/api/v1/dataset/?q=(filters:!((col:table_name,opr:eq,value:'{name}')),page_size:1)",
            )
            if search.get("result"):
                ds_id = search["result"][0]["id"]
                dataset_ids[name] = ds_id
                print(f"  ~ {name} exists (id={ds_id})")
            else:
                print(f"  ! FAILED to create {name}: {e}")
                continue

    # 2. Create charts
    print("\n[2/3] Creating charts …")
    chart_ids = {}
    for ch in CHARTS:
        ds_name = ch["dataset_name"]
        ds_id = dataset_ids.get(ds_name)
        if ds_id is None:
            print(f"  ! Skipping '{ch['slice_name']}' — missing dataset '{ds_name}'")
            continue

        payload = {
            "slice_name": ch["slice_name"],
            "viz_type": ch["viz_type"],
            "datasource_id": ds_id,
            "datasource_type": "table",
            "params": json.dumps(ch["params"]),
        }
        try:
            resp = api_post("/api/v1/chart/", payload)
            cid = resp["id"]
            chart_ids[ch["slice_name"]] = cid
            print(f"  + {ch['slice_name']} (id={cid})")
        except requests.HTTPError:
            # Search existing
            safe = ch["slice_name"].replace("'", "\\'")
            search = api_get(
                f"/api/v1/chart/?q=(filters:!((col:slice_name,opr:eq,value:'{safe}')),page_size:1)",
            )
            if search.get("result"):
                cid = search["result"][0]["id"]
                chart_ids[ch["slice_name"]] = cid
                print(f"  ~ {ch['slice_name']} exists (id={cid})")
            else:
                print(f"  ! FAILED: {ch['slice_name']}")

    # 3. Create dashboard
    print("\n[3/3] Creating dashboard …")
    layout = build_dashboard_layout(chart_ids)
    dash_payload = {
        "dashboard_title": "Nvidia PhysicalAI AV Dataset",
        "slug": "nvidia-physicalai",
        "position_json": layout,
        "published": True,
    }
    try:
        resp = api_post("/api/v1/dashboard/", dash_payload)
        dash_id = resp["id"]
        print(f"  + Dashboard created (id={dash_id})")
    except requests.HTTPError:
        # Try to find existing
        search = api_get(
            "/api/v1/dashboard/?q=(filters:!((col:slug,opr:eq,value:'nvidia-physicalai')),page_size:1)",
        )
        if search.get("result"):
            dash_id = search["result"][0]["id"]
            # Update it
            api_put(f"/api/v1/dashboard/{dash_id}", dash_payload)
            print(f"  ~ Dashboard updated (id={dash_id})")
        else:
            print("  ! FAILED to create dashboard")
            dash_id = None

    print("\n" + "=" * 60)
    print("DONE")
    print(f"  Dashboard URL: {SUPERSET_URL}/superset/dashboard/nvidia-physicalai/")
    print(f"  Charts created: {len(chart_ids)}")
    print(f"  Datasets created: {len(dataset_ids)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
