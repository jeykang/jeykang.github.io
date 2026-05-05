# Superset chart recipes (nuScenes-mini via Trino)

These recipes assume your Trino connection can query the schema `iceberg.nuscenes`.

General notes:
- nuScenes `timestamp` fields are in **microseconds**. In Trino/Superset, convert with:
  - `from_unixtime(ts / 1000000.0)` (returns `timestamp with time zone`)
- For Superset time-series charts, expose a single time column called e.g. `event_time`.

---

## 1) Enriched annotations (category + visibility + scene + location)

**Use for**: most “business-friendly” charts (counts by class, visibility mix, location mix).

**Virtual dataset SQL**
```sql
SELECT
  sa.token                          AS annotation_token,
  from_unixtime(s.timestamp / 1000000.0) AS event_time,
  l.location                        AS location,
  sc.name                           AS scene_name,
  c.name                            AS category_name,
  v.level                           AS visibility_level,
  sa.num_lidar_pts                  AS num_lidar_pts,
  sa.num_radar_pts                  AS num_radar_pts,
  sa.size                           AS box_size,
  sa.translation                    AS box_translation,
  sa.rotation                       AS box_rotation
FROM iceberg.nuscenes.sample_annotation sa
JOIN iceberg.nuscenes.instance i
  ON sa.instance_token = i.token
JOIN iceberg.nuscenes.category c
  ON i.category_token = c.token
JOIN iceberg.nuscenes.visibility v
  ON sa.visibility_token = v.token
JOIN iceberg.nuscenes.sample s
  ON sa.sample_token = s.token
JOIN iceberg.nuscenes.scene sc
  ON s.scene_token = sc.token
JOIN iceberg.nuscenes.log l
  ON sc.log_token = l.token
```

**Recommended charts**
- **Bar chart**: “Annotations by category”
  - Metric: `COUNT(*)`
  - Dimension: `category_name`
  - Sort: descending, limit 15
- **Stacked bar**: “Visibility distribution by category”
  - Metric: `COUNT(*)`
  - Dimension: `category_name`
  - Stack/Series: `visibility_level`
  - Limit categories to top 10–15
- **Time-series line**: “Annotation volume over time”
  - Time column: `event_time`
  - Metric: `COUNT(*)`
  - Optional group by: `category_name`
  - Time grain: minute/hour (mini is small; hour usually works)
- **Box plot / histogram** (if enabled): “num_lidar_pts distribution”
  - Metric: use `num_lidar_pts` as numeric value
  - Group by: `category_name` (optional)

---

## 2) Sensor coverage & keyframes (sample_data + sensor)

**Use for**: what sensors produce what, and keyframe vs sweep coverage.

**Virtual dataset SQL**
```sql
SELECT
  sd.token                               AS sample_data_token,
  from_unixtime(sd.timestamp / 1000000.0) AS event_time,
  l.location                             AS location,
  sc.name                                AS scene_name,
  sen.modality                           AS modality,
  sen.channel                            AS channel,
  sd.fileformat                          AS fileformat,
  sd.is_key_frame                        AS is_key_frame,
  sd.width                               AS width,
  sd.height                              AS height
FROM iceberg.nuscenes.sample_data sd
JOIN iceberg.nuscenes.calibrated_sensor cs
  ON sd.calibrated_sensor_token = cs.token
JOIN iceberg.nuscenes.sensor sen
  ON cs.sensor_token = sen.token
JOIN iceberg.nuscenes.sample s
  ON sd.sample_token = s.token
JOIN iceberg.nuscenes.scene sc
  ON s.scene_token = sc.token
JOIN iceberg.nuscenes.log l
  ON sc.log_token = l.token
```

**Recommended charts**
- **Bar chart**: “Frames by sensor channel”
  - Metric: `COUNT(*)`
  - Dimension: `channel`
  - Filter: `is_key_frame = true` (optional)
- **Stacked bar**: “Keyframes vs sweeps by modality”
  - Metric: `COUNT(*)`
  - Dimension: `modality`
  - Series: `is_key_frame`
- **Time-series**: “Keyframes over time per modality”
  - Time: `event_time`
  - Metric: `COUNT(*)`
  - Group by: `modality`
  - Filter: `is_key_frame = true`

---

## 3) Scene summary (duration + sample count + location)

**Use for**: high-level table, sorting scenes, comparing locations.

**Virtual dataset SQL**
```sql
SELECT
  sc.token                               AS scene_token,
  sc.name                                AS scene_name,
  l.location                             AS location,
  l.date_captured                        AS date_captured,
  COUNT(*)                               AS num_samples,
  MIN(from_unixtime(s.timestamp / 1000000.0)) AS start_time,
  MAX(from_unixtime(s.timestamp / 1000000.0)) AS end_time,
  date_diff(
    'second',
    MIN(from_unixtime(s.timestamp / 1000000.0)),
    MAX(from_unixtime(s.timestamp / 1000000.0))
  )                                      AS duration_seconds
FROM iceberg.nuscenes.scene sc
JOIN iceberg.nuscenes.log l
  ON sc.log_token = l.token
JOIN iceberg.nuscenes.sample s
  ON s.scene_token = sc.token
GROUP BY 1,2,3,4
```

**Recommended charts**
- **Table**: show `scene_name`, `location`, `date_captured`, `num_samples`, `duration_seconds`
- **Bar chart**: “Total samples by location”
  - Metric: `SUM(num_samples)`
  - Dimension: `location`

---

## 4) Object instance popularity (instances + annotations)

**Use for**: which tracked objects appear most, per class.

**Virtual dataset SQL**
```sql
SELECT
  i.token                AS instance_token,
  c.name                 AS category_name,
  COUNT(sa.token)        AS num_annotations
FROM iceberg.nuscenes.instance i
JOIN iceberg.nuscenes.category c
  ON i.category_token = c.token
LEFT JOIN iceberg.nuscenes.sample_annotation sa
  ON sa.instance_token = i.token
GROUP BY 1,2
```

**Recommended charts**
- **Bar chart**: “Top instances by annotation count”
  - Metric: `MAX(num_annotations)` or just use `num_annotations` as metric
  - Dimension: `instance_token`
  - Filter: `category_name IN (...)` or group by `category_name`
  - Limit 20
- **Box plot**: “Instance annotation count distribution by category”
  - Value: `num_annotations`
  - Group by: `category_name`

---

## 5) Attributes (explode attribute_tokens)

**Use for**: semantic composition of annotations (e.g. pedestrian.moving, vehicle.parked, etc.).

**Virtual dataset SQL**
```sql
SELECT
  from_unixtime(s.timestamp / 1000000.0) AS event_time,
  l.location                             AS location,
  c.name                                 AS category_name,
  a.name                                 AS attribute_name,
  COUNT(*)                               AS attribute_count
FROM iceberg.nuscenes.sample_annotation sa
JOIN iceberg.nuscenes.sample s
  ON sa.sample_token = s.token
JOIN iceberg.nuscenes.scene sc
  ON s.scene_token = sc.token
JOIN iceberg.nuscenes.log l
  ON sc.log_token = l.token
JOIN iceberg.nuscenes.instance i
  ON sa.instance_token = i.token
JOIN iceberg.nuscenes.category c
  ON i.category_token = c.token
CROSS JOIN UNNEST(sa.attribute_tokens) AS t(attribute_token)
JOIN iceberg.nuscenes.attribute a
  ON a.token = t.attribute_token
GROUP BY 1,2,3,4
```

**Recommended charts**
- **Bar chart**: “Top attributes overall”
  - Metric: `SUM(attribute_count)`
  - Dimension: `attribute_name`
  - Sort desc, limit 20
- **Stacked bar**: “Attributes by category”
  - Metric: `SUM(attribute_count)`
  - Dimension: `attribute_name`
  - Series: `category_name`
  - Limit to a small set of categories (e.g., vehicles + pedestrians)

---

## 6) Annotations per sample (scene dynamics)

**Use for**: how busy each frame is; compare scenes.

**Virtual dataset SQL**
```sql
SELECT
  from_unixtime(s.timestamp / 1000000.0) AS event_time,
  sc.name                                AS scene_name,
  l.location                             AS location,
  COUNT(sa.token)                        AS annotations_in_sample
FROM iceberg.nuscenes.sample s
JOIN iceberg.nuscenes.scene sc
  ON s.scene_token = sc.token
JOIN iceberg.nuscenes.log l
  ON sc.log_token = l.token
LEFT JOIN iceberg.nuscenes.sample_annotation sa
  ON sa.sample_token = s.token
GROUP BY 1,2,3
```

**Recommended charts**
- **Time-series line**: “Annotations per frame over time”
  - Time: `event_time`
  - Metric: `AVG(annotations_in_sample)` or `SUM(annotations_in_sample)`
  - Group by: `scene_name` (optional)
- **Bar chart**: “Average annotations per frame by scene”
  - Metric: `AVG(annotations_in_sample)`
  - Dimension: `scene_name`

---

## Quick Superset setup tips (per virtual dataset)

- In **Datasets → + Dataset → SQL (virtual dataset)**: paste one of the queries above.
- Set `event_time` as the dataset’s **Time Column** (where present).
- For charts that feel “flat”:
  - Start by limiting categories (top 10–15), then add a breakdown (visibility/modality).
  - Prefer **COUNT(*)** metrics on joined datasets instead of trying to chart raw token tables.
