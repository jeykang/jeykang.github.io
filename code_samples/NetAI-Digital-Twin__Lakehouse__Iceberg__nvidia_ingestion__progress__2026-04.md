# Medallion Progress — April 2026 (foundational build)

*Bronze/Silver/Gold setup, NFS lidar/radar recovery, canonical schema migration,
and perception integration v3–v5. Historical log; current state lives in the
[progress index](../MEDALLION_PROGRESS.md).*

---
**Status**: Recovery complete. Bronze re-registered: lidar 6.16M rows (340/340 chunks, was 9-chunk salvage), radar 11.73B rows across 19 sensors (was 1/19 populated). Silver + Gold re-ran post-recovery; both completed but radar count() returns -1 due to expanded UINT_8 scope (see Open Issues). Top 10% Gold subset = **35,773 clips** (was 45,070 pre-recovery — tightened cohort because new radar coverage signal increased score discrimination). Total post-recovery wall time excluding download: **3h 6m** (Bronze 2h 8m + Silver 42m + Gold 16m).

**Lidar/radar recovery (2026-04-24 → 2026-04-28)**: Re-downloaded all missing zips from HuggingFace via `nvidia_ingestion/redownload_missing.py`. Four passes total: 2,446 + 46 + 5 + 0 = **2,497 of 2,546 zips recovered (98.1%)**, 10.56 TB downloaded, 89h wall time. The 49 unrecoverable zips return genuine 404 from HF (upstream-pruned). Pass 1 saw two TCP-zombie stalls during the lidar phase (~44h hang on chunk_0276; 3-day hang on radar chunk_0369) — manually killed the stuck `hf download` children, parent's `subprocess.run` returned nonzero, run continued; both stalled chunks succeeded on pass 2 retry. Recovery curve: 100 fails → 54 → 49 → 49 (converged). HF-pruned chunks cluster: chunk_1057 and chunk_3109 each fail across ~9 different radar sensors.

**FULL RECOVERY EVENT v2 (2026-04-29 → 2026-05-04)**: PURGE incident (Open Issue #11) destroyed the lidar+radar+labels+calibration+metadata source data. Full re-download from HF expanded to include calibration (bare parquets, not zips — required `redownload_missing.py` patch) + bare metadata files (clip_index.parquet, data_collection.parquet; sensor_presence.parquet replaced by feature_presence.parquet from v26.03). Five-pass recovery curve: 401 → 97 → 57 → 49 → 49 (converged at 49 truly upstream-pruned, same set as before). **3,884 of 3,933 zips recovered (98.75%)**, ~10.85 TB downloaded, ~35h wall on pass 1 (subsequent passes <30 min each).

**Lidar data-loss incident (2026-04-10, fully recovered 2026-04-28)**: The lidar Bronze table that previously reported 6.4M rows was a phantom — the Iceberg manifests pointed at NFS parquets that had been truncated to 0 bytes. Root cause: `extract_remaining.sh` checked only `unzip`'s exit status in its `if unzip; then mv; echo DONE; rm zip; fi` block, so every `mv` that failed with `Disk quota exceeded` still caused the source zip to be deleted. Across two retry passes on 2026-04-10, ~97% of lidar and nearly all radar source zips were lost this way. Initial salvage on 2026-04-21 recovered 9 chunks (172,445 rows). Full re-download on 2026-04-24 → 2026-04-28 brought the dataset back to 13 TB on NFS (340/340 lidar chunks, 19/19 radar sensors).

**Gold scoring v2 (bulk ego aggregation)**: replaced the constant `ego_dynamics=0.5` placeholder with a single-pass Spark aggregation over the 101M-row egomotion table computing per-clip `stddev(||accel||)` and `stddev(curvature)`. Result: ego_dynamics `n_distinct` went 1 → 4515, `std` went 0.00 → 0.1052.

**Gold scoring v3 (renormalized weights)**: the old fallback `obstacle_density = ego_dynamics` effectively double-counted ego signal at 30%+20%=50% combined weight. Replaced with a principled renormalisation: when obstacle data is absent, drop `obstacle_density` from the weighted sum and rescale the remaining four weights. Final distribution: `[0.1873, 0.8364]`, mean 0.4869, std 0.0891 (vs 0.0786 with the mirror, 0.0714 with the constant). Top 10% → 19 Gold views, p90 threshold 0.6125 (35,777 clips above).

**Cohort profile audit**:

| Sub-score | Top 10% mean | Bottom 10% mean | Separation |
|-----------|--------------|-----------------|------------|
| time_of_day (0.20 weight) | 0.872 | 0.227 | ✓ dark vs daytime |
| season_geography (0.15) | 0.242 | 0.259 | ≈ same (most clips are summer) |
| sensor_coverage (0.15) | 0.999 | 0.548 | ✓ full vs partial coverage |
| ego_dynamics (0.30) | 0.487 | 0.385 | ✓ dynamic vs calm |

The top cohort clusters on winter/night/full-sensor-coverage with above-average ego dynamics — matches the intuitive "hard driving clip" profile. Hardest clip is `b1ef9ae7-...` at 0.8364 (5am winter US, ego=0.614).

---

## 1. Architecture Summary

The medallion tiers were redefined (professor's directive, April 2026) from data engineering operations to **data curation tiers**:

```
Bronze = Full dataset registered as-is in Iceberg (zero-copy via add_files())
Silver = Bronze minus broken/unusable data (quality-filtered views)
Gold   = Curated edge-case subset (hardest clips via composite scoring)
```

The prior approach treated Bronze/Silver/Gold as sequential data transformations (raw → enriched → joined). The new approach treats them as selection filters: every tier contains the same data structure, just progressively narrowed to higher-quality and higher-difficulty subsets.

### Technology Stack

| Component | Technology | Details |
|-----------|-----------|---------|
| Table format | Apache Iceberg v2 | REST catalog (Polaris), format-version 2, zstd |
| Query engine | PySpark (local mode) | `local[*]`, 24-core Xeon, 32GB driver memory |
| Storage | NFS (primary) | `/mnt/netai-e2e/nvidia-physicalai-av-subset/` |
| Object store | MinIO/S3 | For Iceberg metadata warehouse |
| Container | spark-iceberg | `tabulario/spark-iceberg:3.5.5_1.8.1` |
| GPUs | RTX 6000 (24GB) + A10 (23GB) | For Gold perception scoring (BEVFusion) |

### Iceberg Namespaces

| Namespace | Purpose | Content |
|-----------|---------|---------|
| `iceberg.nvidia_bronze` | Raw data tables | All sensor data registered via `add_files()` |
| `iceberg.nvidia_silver` | Quality-filtered views | Views excluding FAIL clips from `quality_report` |
| `iceberg.nvidia_gold` | Edge-case views | Views filtering Silver to top N% hardest clips |

---

## 2. Tier-by-Tier Status

### Bronze — `register_bronze.py`

**What it does**: Zero-copy registration of on-disk parquet files into Iceberg using `add_files()`. No data is rewritten — Iceberg manifests point directly at NFS parquet files.

**Registration strategies**:
- Bare parquets (clip_index, metadata): single `add_files()` call
- Calibration (camera_intrinsics, sensor_extrinsics, vehicle_dimensions): directory-based
- Small sensor files (egomotion, radar): batched flat symlink directories (5000 files/batch) to work around `add_files()` scalability
- Large sensor files (lidar, ~216MB each): per-chunk-dir `add_files()` to avoid memory issues

**Status**: Code is tested and operational. `register_nfs` mode added (April 2026) to walk extracted NFS chunk directories directly. Last full registration was done via FUSE mounts before egomotion data was recovered. Needs re-registration now that:
1. All data is extracted to native NFS directories (no more FUSE)
2. Egomotion data has been recovered (340 chunks, 33,767 parquets from NFS silly-rename files)

**What's needed**:
- [x] Add a `register_nfs` mode to `register_bronze.py` that walks extracted chunk directories directly, instead of requiring FUSE mount points
- [x] Filter zero-byte stub files from incomplete NFS extraction (`_has_real_parquets` in `_find_nfs_chunk_dirs`)
- [x] Fix `suffix_filter` scoping bug: camera chunk dirs contain both `.timestamps.parquet` and `.blurred_boxes.parquet`, so passing the raw dir to `add_files()` picked up both schemas and silently produced 0-row tables. `_register_per_dir` now builds a per-chunk symlink staging tree under `<source>/.bronze_staging/<table>_<suffix>/chunk_*/` so `add_files()` sees only matching files. The staging symlinks must persist (Iceberg manifests store their paths)
- [x] Register bare parquets, calibration, egomotion, lidar, radar (partial), cameras (all 7 sensors × 2 suffixes). Final state: 23 Bronze tables, ~492M rows
- [ ] Row counts do NOT match original expectations — see "NFS extraction incompleteness" below

**Registered row counts (post-recovery, 2026-04-28)**:

| Table | Rows | Notes |
|-------|------|-------|
| clip_index | 310,895 | full (unchanged) |
| data_collection | 310,895 | full (unchanged) |
| sensor_presence | 310,895 | full (unchanged) |
| camera_intrinsics | 236,369 | full (unchanged) |
| sensor_extrinsics | 425,106 | full (unchanged) |
| vehicle_dimensions | 33,767 | full (unchanged) |
| egomotion | 101,745,981 | full (unchanged) |
| **lidar** | **6,164,244** | **340/340 chunks (was 172,445 on 9-chunk salvage; pre-loss phantom: 6.4M)** |
| radar_radar_corner_front_left_srr_0 | 362,783,421 | post-recovery (was 66.2M with partial data) |
| radar_radar_corner_front_left_srr_3 | 633,932,700 | post-recovery (was 0 — 100% zero-byte) |
| radar_radar_corner_front_right_srr_0 | 397,434,200 | post-recovery (was 0) |
| radar_radar_corner_front_right_srr_3 | 595,595,762 | post-recovery (was 0) |
| radar_radar_corner_rear_left_srr_0 | 450,902,561 | post-recovery (was 0) |
| radar_radar_corner_rear_left_srr_3 | 673,652,788 | post-recovery (was 0) |
| radar_radar_corner_rear_right_srr_0 | 428,043,322 | post-recovery (was 0) |
| radar_radar_corner_rear_right_srr_3 | 628,411,454 | post-recovery (was 0) |
| radar_radar_front_center_imaging_lrr_1 | 2,100,264,994 | post-recovery (was 0); largest single sensor |
| radar_radar_front_center_mrr_2 | 1,126,343,293 | post-recovery (was 0) |
| radar_radar_front_center_srr_0 | 483,157,676 | post-recovery (was 0) |
| radar_radar_rear_left_mrr_2 | 1,022,287,477 | post-recovery (was 0) |
| radar_radar_rear_left_srr_0 | 465,936,146 | post-recovery (was 0) |
| radar_radar_rear_right_mrr_2 | 1,014,377,038 | post-recovery (was 0) |
| radar_radar_rear_right_srr_0 | 458,284,032 | post-recovery (was 0) |
| radar_radar_side_left_srr_0 | 359,351,110 | post-recovery (was 0) |
| radar_radar_side_left_srr_3 | 109,159,563 | post-recovery (was 0); design 32/36 chunks |
| radar_radar_side_right_srr_0 | 326,050,630 | post-recovery (was 0) |
| radar_radar_side_right_srr_3 | 94,994,629 | post-recovery (was 0); design 32/36 chunks |
| **radar TOTAL (19 sensors)** | **11,730,962,796** | matches DATASET.md ~11.3B prediction within 4% |
| cam_camera_cross_left_120fov (ts/blur) | 20,414,240 / 25,107,548 | full (unchanged) |
| cam_camera_cross_right_120fov (ts/blur) | 20,414,425 / 22,022,313 | full (unchanged) |
| cam_camera_front_tele_30fov (ts/blur) | 20,364,585 / 58,462,844 | full (unchanged) |
| cam_camera_front_wide_120fov (ts/blur) | 20,415,369 / 44,653,338 | full (unchanged) |
| cam_camera_rear_left_70fov (ts/blur) | 20,414,360 / 40,484,385 | full (unchanged) |
| cam_camera_rear_right_70fov (ts/blur) | 7,099,487 / 12,095,848 | partial — 122/307 dirs non-empty (unchanged; not in scope of Apr 24 redownload) |
| cam_camera_rear_tele_30fov (ts/blur) | 48,929 / 227,935 | partial — 2/175 dirs non-empty (unchanged) |
| **GRAND TOTAL** | **~12.2B rows across 41 tables** | |

**Re-registration command** (one-off driver, 2h 8m wall):
```bash
docker cp /tmp/register_lidar_radar.py spark-iceberg:/tmp/
docker exec --user 1000:1007 spark-iceberg /opt/spark/bin/spark-submit \
  --master 'local[*]' --packages org.apache.hadoop:hadoop-aws:3.3.4 \
  --conf spark.driver.memory=16g \
  /tmp/register_lidar_radar.py
```
Drops then re-creates only lidar + 19 radar tables; leaves egomotion/cameras/calibration alone. Non-radar tables unchanged from 2026-04-21 numbers.

**Run commands**:
```bash
# NFS mode (recommended — walks extracted chunk directories directly)
python -m nvidia_ingestion.register_bronze nfs

# Or via unified pipeline
python -m nvidia_ingestion.pipeline --bronze --bronze-mode nfs

# Legacy FUSE mode (if using ratarmount-mounted zips)
python -m nvidia_ingestion.register_bronze fuse --fuse-root /mnt/nvidia-fuse
```

### Silver — `quality_checks.py`

**What it does**: Scans Bronze tables and raw files to identify broken/unusable data. Writes a `quality_report` Iceberg table, then creates Silver views that exclude failed clips.

**Four quality checks**:

| Check | What it detects | Method |
|-------|----------------|--------|
| `missing_sensors` | Clips where expected egomotion, lidar, or radar data is absent | Cross-reference `clip_index` against sensor tables via Spark |
| `timestamps` | Non-positive timestamps, non-monotonic sequences, gaps >1s | Window functions on egomotion table |
| `camera` | Corrupt MP4 files (too small, invalid box header) | File I/O on NFS (reads first 12 bytes per MP4) |
| `schema` | All-null columns in Bronze tables (corruption indicator) | Sample 1000 rows per table |

**Output**: 
- `nvidia_silver.quality_report` table — one row per (clip_id, check_name, status)
- Silver views: `CREATE VIEW silver.X AS SELECT * FROM bronze.X WHERE clip_id NOT IN (failed_clips)`
- Tables without `clip_id` (calibration): pass-through views

**Status**: Re-ran end-to-end on 2026-04-28 post-recovery. Wall=2,499s (42m). 100 FAIL findings (all from `missing_sensors` fallback), clip_index → silver dropped 310,895 → 310,795 = exactly the 100 FAILs. 35,773 valid clips after exclusion. Sensor table view counts came back as -1 for all radar/lidar/egomotion — see Open Issues #10.

**Check-level findings (post-recovery, 2026-04-28)**:

| Check | Status | Count | Wall | Notes |
|-------|--------|-------|------|-------|
| `missing_sensors` | FAIL | 100 | 959.8s | All 19 radar count() queries failed with UINT_8 — fell back to "0/19 radar tables populated, skip per-clip radar check" branch. 100 findings come from camera/lidar coverage gaps. The check is now ineffective for radar. |
| `timestamps` | clean | 0 | 142.4s | egomotion timestamps fine (relative microsecond convention) |
| `camera_corrupt` | WARN | (n/a captured this run) | — | MP4 extraction gaps unchanged from prior run |
| `schema_unreadable` | FAIL | (n/a captured this run) | — | UINT_8 issue now affects all 19 radar sensors, not just one |

**Bugs fixed in this run**:
1. **OOM in timestamp check on 101M-row egomotion** — replaced `Window.partitionBy(clip_id).orderBy(timestamp) + lag` with `groupBy(clip_id).agg(min, max, count)` to produce per-clip bounds without sorting.
2. **False-positive timestamp FAIL on 100% of clips** — initial check flagged all 33,767 clips for "negative timestamps". PhysicalAI egomotion uses *relative microseconds with ~200ms pre-roll* (timestamp range `[-200000, 141017307]`), so negative `min_ts` is normal. Fixed threshold to `min_ts < -1_000_000` (>1s pre-roll is anomalous).
3. **Cascading FAIL from camera mp4 corruption** — 35,913 camera_corrupt findings were excluding clips from ALL Silver views (sensor tables, calibration, etc.). Downgraded to WARN so each Silver sensor view is gated only by its own quality findings.
4. **missing_sensors false positives scoped** — `check_missing_sensors` now scopes the "expected clips" set to clips actually present in locally-available camera Bronze tables, and skips radar entirely when <10 radar sensors are populated (documented NFS extraction gap).

**Run command** (recorded for reproduction):
```bash
docker exec --user 1000:1007 -e HOME=/tmp -e PATH=/opt/spark/bin:/usr/bin:/bin spark-iceberg \
  /opt/spark/bin/spark-submit \
  --master 'local[*]' \
  --packages org.apache.hadoop:hadoop-aws:3.3.4 \
  --conf spark.jars.ivy=/tmp/.ivy2 \
  --conf spark.driver.memory=16g \
  /tmp/run_quality.py
```

**Run command**:
```bash
python -m nvidia_ingestion.quality_checks
# Or specific checks only:
python -m nvidia_ingestion.quality_checks --checks missing_sensors timestamps
# Without building views:
python -m nvidia_ingestion.quality_checks --no-views
```

### Gold — `edge_case_scorer.py`

**What it does**: Scores every clip's difficulty via a composite function, then selects the top N% hardest clips as the Gold subset.

**Composite scoring (two components)**:

#### Scene Complexity Score (CPU, no GPU)

| Sub-score | Weight | Logic |
|-----------|--------|-------|
| `time_of_day` | 20% | Night (0-5h) = 1.0, dawn/dusk = 0.7, midday = 0.2 |
| `season_geography` | 15% | Winter = 0.9 (boosted for northern European countries), summer = 0.2 |
| `sensor_coverage` | 15% | Inverted radar sensor coverage ratio (fewer sensors = harder) |
| `ego_dynamics` | 30% | Acceleration std + heading angular velocity (60/40 blend) |
| `obstacle_density` | 20% | Actor count / 20, capped at 1.0 (requires v26.03 `obstacle.offline` labels; without those, falls back to ego_dynamics value) |

#### Perception Difficulty Score (GPU, optional)

Planned via BEVFusion temporal consistency analysis:
- Run BEVFusion on each lidar+camera frame
- Measure detection jitter, ID switches, confidence drops across frames
- High jitter/low confidence = harder scene
- **Not yet operational** — BEVFusion backend is a placeholder

**Combined**: `final = 0.4 * scene + 0.6 * perception` (GPU mode) or `final = scene` (metadata-only mode)

**Three backends**:

| Backend | GPU? | Purpose | Status |
|---------|------|---------|--------|
| `MetadataBackend` | No | Bulk Spark scoring from metadata tables | Ready |
| `BEVFusionBackend` | Yes (~12GB) | mmdetection3d BEVFusion inference | Placeholder |
| `DummyBackend` | No | Deterministic hash scores for testing | Ready |

**Gold subset selection**: After scoring, creates Gold views filtering Silver tables to clips with `difficulty_score >= percentile_threshold(1 - top_pct/100)`. Default `top_pct = 10` (top 10% hardest).

**Status**: MetadataBackend re-ran end-to-end on 2026-04-28 (post-recovery). Wall=992.8s (16m), 35,773 hard clips selected (top 10%). 20 Gold views built. Per Open Issues #10, all 19 radar Gold view row counts come back as -1 — views exist but underlying reads fail.

**Gold scoring run (2026-04-28, metadata backend, post-recovery)**:
- Wall: 992.8s (16m) for full Bronze→Gold scoring + view building
- 35,773 Gold clips selected (was 45,070 pre-recovery — tighter cohort)
- Gold views populated (selected): egomotion 6.4M, lidar 417,697, cam_front_wide_blur 1.92M, cam_front_tele_blur 2.41M, cam_rear_left_70fov_blur 1.96M
- All 19 radar Gold view counts: -1 (UINT_8 read failures)

**Gold scoring run (2026-04-21, metadata backend)**:
- 310,895 clips scored in 8.6s (36,147 clips/s, bulk Spark mode)
- Score distribution: min=0.3911, max=0.7500, mean=0.5000, std=0.0714
- Top 10% threshold: `difficulty_score >= 0.59` → 45,070 Gold clips (14.5% — discreteness of sub-scores means the percentile threshold catches a larger bucket than a literal 10%)
- `iceberg.nvidia_gold.clip_scores`: 310,895 rows, per-clip `detail` column contains sub-score JSON for audit
- 21 Gold views created (sensor tables filtered to the 45,070 hardest clips). Sample row counts: egomotion=35,322,908; camera_front_tele_30fov_blur=15,381,562; camera_front_wide_120fov_blur=11,333,010; clip_index=45,070; sensor_presence=45,070

**Gold view builder bugs fixed in this run**:
1. **Row.get() does not exist** — bulk scoring path used `row.get("chunk", 0)` but PySpark `Row` objects don't support `.get()`. Changed to `row.asDict().get(...)`.
2. **Silver VIEWs not discovered** — `build_gold_subset` used `SHOW TABLES IN iceberg.nvidia_silver`, which only returns Iceberg tables (quality_report); Silver sensor objects are Iceberg VIEWs and required a separate `SHOW VIEWS` call. Now unions both.
3. **radar_corner_front_left_srr_0 Gold-view count = -1** — view creation succeeded, but counting the Gold view via `spark.table(...).count()` returns -1 because the underlying vectorized path hits the UINT_8 error. SQL aggregates run on the view work correctly when `spark.sql.iceberg.vectorization.enabled=false` is set. Downstream consumers must set this conf.

**Score distribution audit (2026-04-21)**:

| Sub-score | n_distinct | min | max | mean | std | Signal |
|-----------|-----------|-----|-----|------|-----|--------|
| `time_of_day` | 6 | 0.20 | 1.00 | 0.50 | **0.26** | good — matches `hour_of_day` histogram |
| `season_geography` | 5 | 0.20 | 1.00 | 0.24 | **0.15** | modest — 92% of clips are summer, so winter-boost fires rarely |
| `sensor_coverage` | 4 | 0.47 | 1.00 | 0.76 | **0.23** | good — driven by zero-byte radar (only 1/19 sensors populated) |
| `ego_dynamics` | **1** | 0.50 | 0.50 | 0.50 | **0.00** | **NO SIGNAL** — bulk mode skips per-clip egomotion reads, emits neutral 0.5 |
| `obstacle_density` | **1** | 0.50 | 0.50 | 0.50 | **0.00** | **NO SIGNAL** — v26.03 `obstacle.offline` labels not downloaded |

Overall `difficulty_score`: mean 0.5000, std 0.0714, range [0.3911, 0.7500]. Histogram is bimodal at [0.45, 0.50) (85K clips) and [0.55, 0.60) (68K), with a long tail above 0.60.

**Net effect**: 50% of the composite weight (ego 30% + obstacle 20%) is currently a constant 0.5. Only three sub-scores carry real signal. The top-of-distribution Gold clips are effectively selecting for "winter + northern-Europe + night + low-radar-coverage" — a legitimate hard scenario, but not diversified. The `ego_dynamics` neutral-default re-weights its mass onto `obstacle_density` inside `compute_scene_score`, which is *also* neutral, so the re-weighting is a no-op here.

Two paths to add more signal (ordered by effort):
1. **Enable per-clip egomotion**: switch from `_score_metadata_bulk` to `_score_per_clip` — reads egomotion per-clip and computes accel-std + heading-angular-velocity. Much slower (~10 clips/s vs 36K clips/s bulk), so would need to be run on a subset or in batches. Yields true ego_dynamics signal.
2. **Download v26.03 `obstacle.offline` labels** (~100 GB): unlocks `obstacle_density` via actor counts. Already documented in §4 "What's NOT on disk".

**What's needed**:
- [x] Run metadata-only scoring after Silver is ready: done.
- [x] Review score distribution: done — see above.
- [ ] Decide whether to enable per-clip ego scoring (trade throughput for signal) or wait for v26.03 download before rerunning Gold.
- [ ] Download v26.03 `obstacle.offline` labels to enable actor-count scoring (currently hardcoded to neutral)
- [ ] Install mmdetection3d + download BEVFusion checkpoint for GPU scoring
- [ ] Tune sub-score weights based on domain expert review
- [ ] Decide final Gold subset size (10% default, configurable via `--top-pct`)

**Run commands**:
```bash
# Metadata-only scoring (no GPU needed)
python -m nvidia_ingestion.edge_case_scorer --backend metadata --top-pct 10

# With limit for testing
python -m nvidia_ingestion.edge_case_scorer --backend metadata --top-pct 10 --limit 1000

# Pipeline testing with dummy scores
python -m nvidia_ingestion.edge_case_scorer --backend dummy --top-pct 10

# GPU scoring (when BEVFusion is set up)
python -m nvidia_ingestion.edge_case_scorer --backend bevfusion --gpu 0 --top-pct 10 \
    --bevfusion-config /path/to/config.py --bevfusion-checkpoint /path/to/ckpt.pth
```

---

## 3. Unified Pipeline Runner — `pipeline.py`

Orchestrates all three stages sequentially:

```bash
# Full pipeline (NFS mode, metadata scoring, no GPU)
python -m nvidia_ingestion.pipeline --all

# Individual stages
python -m nvidia_ingestion.pipeline --bronze
python -m nvidia_ingestion.pipeline --silver
python -m nvidia_ingestion.pipeline --gold --backend metadata --top-pct 10

# Full pipeline with BEVFusion
python -m nvidia_ingestion.pipeline --all --backend bevfusion --gpu 0

# Bronze with legacy FUSE mode
python -m nvidia_ingestion.pipeline --bronze --bronze-mode fuse --fuse-root /mnt/nvidia-fuse
```

Saves a JSON report to `/tmp/nvidia_pipeline_report.json` with per-stage timing and results.

Defaults: `--bronze-mode nfs`, `--backend metadata`.

---

## 4. Data Readiness

### What's on disk (NFS)

| Data | Location | Count | Status |
|------|----------|-------|--------|
| clip_index | `clip_index.parquet` | 310,895 clips | Ready |
| data_collection | `metadata/data_collection.parquet` | 310,895 rows | Ready |
| sensor_presence | `metadata/sensor_presence.parquet` | 310,895 rows | Ready |
| selected_chunks | `selected_chunks.csv` | 340 rows | Ready |
| Egomotion | `labels/egomotion/` | 340 chunk dirs, 33,767 parquets | Ready (recovered from .nfs files) |
| LiDAR | `lidar/lidar_top_360fov/` | 338 chunk dirs, 33,673 parquets | Ready |
| Radar (19 sensors) | `radar/*/` | 36-134 chunk dirs per sensor | Ready |
| Cameras (7 sensors) | `camera/*/` | 175-340 chunk dirs per sensor | Ready (2 sensors incomplete) |
| Calibration | `calibration/` | 1,020 parquets (3 types x 340) | Ready |

### What's NOT on disk

| Data | Impact | How to get it |
|------|--------|---------------|
| `obstacle.offline` labels (v26.03) | obstacle_density sub-score falls back to ego_dynamics | `hf download nvidia/PhysicalAI-Autonomous-Vehicles --repo-type dataset --include "labels/obstacle.offline/*"` |
| `egomotion.offline` (v26.03) | No impact — online egomotion is sufficient | Same HF download |
| `reasoning/ood_reasoning.parquet` (v26.03) | No impact — OOD reasoning is supplementary | Same HF download |
| BEVFusion checkpoint | Perception scoring backend unavailable | `pip install mmdet3d`, download from model zoo |

---

## 5. Known Issues

1. **NFS extraction incompleteness (2026-04-21)**: Much of the NFS-extracted dataset is zero-byte stub files from an incomplete extraction. Per-sensor per-suffix breakdown from `find ... -size 0`:
   - **Lidar**: 16 of 338 chunk dirs are 100% zero-byte stubs. 322 chunks have real data. Registered row count: 6.4M.
   - **Radar**: **18 of 19 sensors are 100% zero-byte**. Only `radar_corner_front_left_srr_0` has any real data (1,274 non-zero of 8,727 parquets). Registered row count for that one sensor: 66.2M. The other 18 radar tables will register 0 rows (correctly skipped by the zero-byte filter).
   - **Cameras**: 5 of 7 sensors are fully populated (33,685–33,767 parquets per suffix, zero zero-byte). `camera_rear_right_70fov` has 11,736 real / 18,734 zero-byte (~38% complete) in both timestamp and blurred_boxes suffixes. `camera_rear_tele_30fov` has 141 real / 17,184 zero-byte (~1% complete).
   - **Impact**: The registered Bronze tables represent a fraction of the full dataset. Silver quality checks and Gold scoring will run on whatever survived extraction. A re-extraction pass would be needed to approach the "expected" row counts (~101M egomotion is hit, but ~11.3B radar target is off by two orders of magnitude because 18/19 sensors have no data).

2. **Camera data incomplete for 2 sensors**: `camera_rear_right_70fov` (307/340 chunks), `camera_rear_tele_30fov` (175/340 chunks). All other cameras have 340/340. This is partly a dataset property and partly the zero-byte extraction issue above.

2. **Egomotion chunk numbering mismatch**: Recovered from NFS silly-rename files with sequential numbering (chunk_0000 to chunk_0339), which may not match original Nvidia chunk IDs. The parquet filenames inside each chunk contain the correct clip UUIDs, so downstream joins by `clip_id` work correctly.

3. **No ground truth 3D bounding boxes**: The dataset does not include human-annotated 3D labels. Even v26.03 `obstacle.offline` labels are machine-generated. Difficulty scoring relies on self-consistency and metadata, not GT comparison.

4. **NFS throughput**: ~35 MB/s aggregate write, which bottlenecks large operations. Camera integrity checks (reading headers of ~236K files) will be slow.

5. **~~`pipeline.py` default backend~~**: Fixed — now defaults to `"metadata"` for production.

6. **Ego dynamics scoring without bulk egomotion**: MetadataBackend bulk scoring skips per-clip egomotion reads for performance — the ego_dynamics sub-score defaults to 0.5 (neutral) and its weight redistributes to obstacle_density. For more accurate scoring, per-clip scoring mode (`_score_per_clip`) reads egomotion but is much slower.

7. **Egomotion Bronze was pointing at ephemeral `/tmp/nvidia-fuse-flat/egomotion/`** (2026-04-21): The first Silver rerun had all 101M egomotion rows reachable by Iceberg but partial scan failures because the referenced `/tmp/nvidia-fuse-flat/egomotion/batch_0005/` directory no longer existed. Re-registered egomotion from NFS at `labels/egomotion/` (340 chunks, 33,767 parquets). Bronze paths now point at `file:/mnt/netai-e2e/nvidia-physicalai-av-subset/labels/egomotion/...`. Row count unchanged at 101,745,981.

8. **UINT_8 radar column breaks Python collect path** (2026-04-21): `radar_radar_corner_front_left_srr_0` contains a column with Parquet logical type UINT_8 (likely `max_returns`). Iceberg's `GenericArrowVectorAccessorFactory` raises `UnsupportedOperationException: Unsupported logical type: UINT_8` during dictionary-encoded vectorized Arrow reads. SQL aggregates (count, sum, groupBy) work; any path that returns rows to Python driver fails. **Workaround**: `spark.conf.set("spark.sql.iceberg.vectorization.enabled", "false")` before reading; or cast the column in a CTAS rewrite. Affects the single `schema_unreadable` FAIL in Silver's quality_report — it's a global finding that doesn't exclude clips, but downstream Python consumers must apply the workaround.

9. **Egomotion timestamp encoding** (2026-04-21): PhysicalAI egomotion timestamps are **per-clip relative microseconds with a pre-roll window**. Observed range across 101M samples: `[-200_000, 141_017_307]`, with 671,858 negative rows, 33,767 zero rows, 101,040,356 positive rows. Every clip has ~200ms of negative-time pre-roll followed by positive main-recording time. Any check that treats `ts < 0` as corruption will flag 100% of clips as FAIL — use `min_ts < -1_000_000` (>1s pre-roll is truly anomalous) instead.

11. **PURGE on add_files-registered tables destroyed source NFS data** (2026-04-29, recovered 2026-05-04): Iceberg `DROP TABLE PURGE` walks the manifest's file list and deletes every referenced file — including external parquets registered via `add_files()`. Running this on the 41 pre-canonical Bronze tables in `nvidia_bronze` deleted ~10.5 TB of source data on NFS: all 33,673 lidar parquets (6.4 TB), all radar parquets (~150 GB), all 33,767 egomotion parquets (~14 GB), all calibration parquets, plus `clip_index.parquet`, `metadata/data_collection.parquet`, `metadata/sensor_presence.parquet`. Camera mp4/timestamps/blurred_boxes survived because the cam_*_ts/cam_*_blur Bronze tables had been registered through `.bronze_staging/` symlink dirs — PURGE deleted the symlinks but not the symlink targets. Canonical Bronze tables (16 of them) survived because they were CTAS-written via `writeTo()` and live in Iceberg-managed storage (S3/MinIO via the catalog), not on NFS. Recovery: 5-pass redownload (2026-04-29 → 2026-05-04) recovered 3,884 / 3,933 zips (98.75%, 10.85 TB downloaded, ~35h on pass 1) — the same 49 upstream-pruned chunks remain unrecoverable. v26.03 `metadata/feature_presence.parquet` substituted for the now-removed v25.10 `sensor_presence.parquet`.

   Mitigation: `canonical_bronze.drop_old_tables()` and `register_lidar_only.py` patched 2026-04-29 to use plain `DROP TABLE` (no PURGE). Memory note saved at `feedback_iceberg_drop_purge_destroys_add_files.md`.

10. **Sensor Bronze tables have no clip_id column** (2026-04-28, supersedes prior UINT_8-only diagnosis): All 19 radar tables, lidar, egomotion, and the 14 camera tables (`*_ts`/`*_blur`) lack a `clip_id` column entirely. clip_id is encoded only in the parquet filename pattern `<clip_id>.<sensor>.parquet`. The current `add_files()` zero-copy registration captures parquet schemas as-is; the source parquets don't have clip_id rows. This — not UINT_8 — is the actual reason `check_missing_sensors` fails and Silver/Gold view counts return -1 for all sensor tables.

   Sub-issue: a UINT_8 column does exist (probably `max_returns` or `exist_probb`) and would block column reads even if clip_id were present, but the immediate failure mode is `[UNRESOLVED_COLUMN]` on `clip_id`.

   Spark conf trials (2026-04-28): all 8 attempted overrides (vectorization on/off, parquet.legacy.fallback.allowUnsupportedTypes, iceberg reader-type=hadoop, etc.) had no effect because they don't add the missing column. `count(*)` worked under all configs because Iceberg uses manifest metadata, not column reads.

   Mitigation: the canonical-schema migration (§11) makes clip_id a first-class column on every sensor table, sourced from filename via `regexp_extract(input_file_name(), …)` during reshape. That's the planned fix.

   For any code that needs clip_id from a sensor table BEFORE migration completes, use:
   ```python
   df.withColumn("clip_id",
       F.regexp_extract(F.input_file_name(),
                        r"/([^/]+)\.<sensor>\.parquet$", 1))
   ```

12. **Dataset version skew: camera data (v25.10) vs metadata (v26.03) — ~1,185 "orphan" clips** (2026-06-16): The camera MP4s on disk are from the **original ~v25.10 extraction (mtime 2026-04-20)**, but `clip_index.parquet` / `feature_presence.parquet` were re-downloaded fresh after the PURGE and are **v26.03 (mtime 2026-04-29)**. NVIDIA prunes clips between releases (README license: *"from time to time update the Dataset… delete any prior versions upon NVIDIA's written request"* — PII/consent/quality/legal). Net effect on our 340 selected chunks: `selected_chunks.csv` declared **33,767** clips, but current `clip_index` lists only **32,582** within those chunks → **1,185 clips removed upstream**. Their camera files survived the PURGE (registered via `.bronze_staging/` symlinks, see §11) so their clip_ids persist on disk while absent from all current metadata (`clip_index`, `data_collection`, `feature_presence`, `transfer_manifest`).

    Evidence: per-chunk, on-disk camera clips are a strict **superset** of metadata (`meta_minus_disk = 0`, `disk_minus_meta = 8–18` per chunk — never the reverse). These are *not* corruption or a foreign dataset.

    Impact: `check_missing_sensors` derives its clip universe from camera Bronze (old clip_ids), so these orphans surface as bogus "missing sensor" FAILs (26 of them landed in the 190-clip Silver exclusion of the 2026-05-04 run; all confirmed non-genuine). Any clip list exchanged with external teams (e.g. KAIST curation) **must be pinned to a dataset version**, since clip membership differs across releases. Cleaner fix: filter Bronze/Silver to clips present in current `clip_index` (drop the version-orphaned tail).

    **Cleaner fix implemented (2026-06-19, code only — not yet re-run; stack is down).** Rewrote `check_missing_sensors` to be driven by `aux_sensor_presence` (= v26.03 `feature_presence.parquet`) instead of hardcoded floors. Changes: (a) universe is now `Clip ∩ Camera ∩ feature_presence` via inner-join, which drops the 26 version-orphans (absent from feature_presence) and the ~9 out-of-scope/not-downloaded clips (absent from Camera); (b) lidar is only checked when the per-clip `lidar_top_360fov` flag is true, eliminating the 23 `BY_DESIGN_no_lidar` FAILs; (c) radar FAILs only when actual distinct `radar_name` < the *per-clip expected* radar count (sum of the 19 `radar_*` booleans), eliminating the 95 `BY_DESIGN_radar_config` FAILs (low config legitimately has ~9); (d) lidar candidates that are missing from Bronze but present non-empty on disk are downgraded FAIL→WARN (registration gap, not a clip-quality failure), eliminating the 32 `FALSE_POSITIVE_bronze_bug` FAILs. Net: of the 190 prior FAILs, ~185 become non-FAIL by construction, mapped 1:1 to `bad_clip_lists/missing_sensor_classified.csv` categories. Expected Silver retention back to ~100% with only genuinely-broken clips excluded. **Verification pending an env bring-up** (NFS mount + polaris/minio/spark-iceberg containers, all currently down).

---

## 6. Open Design Decisions

| Decision | Options | Current Default | Notes |
|----------|---------|-----------------|-------|
| Gold subset size | 5%, 10%, 20% | 10% (`--top-pct 10`) | Configurable. Domain expert should review score distributions |
| Scene score weights | Adjustable in `_SCENE_WEIGHTS` dict | time_of_day=20%, season=15%, sensor=15%, ego=30%, obstacle=20% | Initial estimates. Tune after first full scoring run |
| Perception model | BEVFusion, CenterPoint, or other | BEVFusion (placeholder) | Need mmdetection3d + checkpoint. ~12GB VRAM |
| Scene+perception blend | Adjustable `scene_w`/`percep_w` | 40% scene / 60% perception | Only applies when GPU backend is active |
| Ego dynamics: bulk vs per-clip | Bulk (fast, neutral score) vs per-clip (accurate, slow) | Bulk | Trade-off: ~310K clips/s bulk vs ~10 clips/s per-clip |
| v26.03 download priority | Download now vs wait | Wait | Enables obstacle_density scoring but is ~100GB+ download |

---

## 7. Execution Order

When ready to run the full pipeline:

```
1. Bronze re-registration
   └─ Prerequisite: None (all data on NFS)
   └─ Mode: `python -m nvidia_ingestion.register_bronze nfs`

2. Silver quality checks
   └─ Prerequisite: Bronze tables exist in Iceberg
   └─ Outputs: quality_report table + Silver views

3. Gold metadata scoring
   └─ Prerequisite: Silver views exist (falls back to Bronze if not)
   └─ Outputs: clip_scores table + Gold views

4. (Optional) Download v26.03 obstacle.offline labels
   └─ Re-run Gold scoring with obstacle_density enabled

5. (Optional) BEVFusion perception scoring
   └─ Prerequisite: mmdetection3d installed, checkpoint downloaded
   └─ Re-run Gold scoring with --backend bevfusion --gpu 0
```

Each stage can be run independently via `python -m nvidia_ingestion.pipeline --<stage>` or all at once via `--all`.

---

## 8. File Reference

| File | Purpose | Status |
|------|---------|--------|
| `config.py` | Pipeline config: paths, Spark, Iceberg, namespace names | Stable |
| `register_bronze.py` | Bronze zero-copy registration via `add_files()` | Ready (nfs/fuse/s3 modes). NFS mode also creates persistent symlink staging dirs under `<source>/.bronze_staging/` for suffix-filtered tables (cameras). Do not delete — Iceberg manifests reference those paths. |
| `quality_checks.py` | Silver quality pipeline (4 checks + filtered views) | Run at scale 2026-04-21, 100% retention |
| `edge_case_scorer.py` | Gold composite scoring (3 backends) | MetadataBackend run at scale 2026-04-21 (310K clips in 8.6s, 45K Gold clips); BEVFusion placeholder |
| `pipeline.py` | Unified Bronze->Silver->Gold runner | Implemented |
| `transform_silver.py` | Legacy Silver transforms (view + inplace enrichment) | Superseded by `quality_checks.py` |
| `build_gold.py` | Legacy Gold pre-joined tables | Superseded by `edge_case_scorer.py` |
| `quick_bench.py` | Query benchmarks | Tested |
| `benchmark.py` | Benchmark tracking utilities | Stable |
| `DATASET.md` / `DATASET_KO.md` | Full dataset breakdown | Complete |
| `HANDOFF.md` / `HANDOFF_KO.md` | Cosmos pipeline handoff (for colleague) | Complete |

---

## 9. Benchmark Reference (from prior runs)

| Operation | Result |
|-----------|--------|
| Iceberg `count(*)` on 2.17B rows (radar) | 129ms (O(1) via manifest metadata) |
| Egomotion count | 439ms |
| Silver view creation (31 views) | 14.1s |
| Gold view creation (3 views) | 7.6s |
| MetadataBackend scoring rate | ~310K clips/s (bulk Spark mode) |

### Post-recovery benchmarks (2026-04-28, in `nvidia_ingestion/benchmark_results.json`)

Captured stage-level via `BenchmarkTracker` from `/tmp/run_silver_gold_bench.py`:

| Stage | Wall | rows_out | Peak RSS | Notes |
|-------|------|----------|----------|-------|
| Bronze re-reg (lidar + 19 radar) | 7,672s (2h 8m) | 11,737,127,040 | — | Backfilled from log mtimes; per-step timing not captured |
| Silver (all checks + views) | 2,499s (42m) | 1,627,627† | 226.9 MB | †Only metadata/calibration table counts; sensor tables returned -1 from UINT_8 reads |
| Gold (metadata backend, top 10%) | 992.8s (16m) | 23,819,551 | 555.7 MB | 20 views; 19 radar views created but uncountable |
| **Total (excl. download)** | **11,164s (3h 6m)** | — | — | |
| HF redownload (4 passes total) | ~89h wall | 10.56 TB | — | 2,497 of 2,546 zips (98.1%); ~50-60 MB/s sustained on lidar phase, ~5-6 MB/s on radar phase |

---

## 10. Perception Integration (2026-04-22)

Added a camera-only perception sub-score to Gold scoring via
`camera_perception_scorer.py` (YOLOv8n over 6 sampled frames per clip, on
`camera_front_wide_120fov` MP4s). Output schema per clip: `n_frames_sampled`,
`mean_det_count`, `std_det_count`, `mean_conf`, `max_conf`, `class_diversity`,
`driving_obj_count`, `perception_score` (composite ∈ [0,1]).

Coverage ceiling: only 33,767 of 310,895 valid clips (10.9%) have a front-wide
MP4, so perception is a partial signal. `compute_scene_score` handles this by
treating `perception_score=None` as "dimension dropped" and renormalising the
active weights.

### v3 → v4 → v5 evolution

| Rev | Perception coverage | Max score | Pearson vs v3 | Top-10% Jaccard vs v3 | Demoted |
|-----|---------------------|-----------|---------------|------------------------|---------|
| v3 (no perception baseline) | 0% | 0.836 | 1.000 | 1.000 | — |
| v4 (partial, buggy sharding) | 8.2% (25,340) | 0.812 | 0.981 | 0.955 | 1,599 |
| v5 (full front-wide coverage) | 10.9% (33,767) | 0.701 | 0.975 | 0.941 | 2,124 |

Dominant factors of v5-demoted clips: `sensor_coverage` (912), `time_of_day`
(695), `season_geography` (517). Interpretation: perception acts as a damper —
clips that look hard by metadata (missing sensors, night, rare region) but
show low visual complexity get pulled out of the top cohort. That's the
intended behaviour; YOLO catches "metadata-says-hard-but-actually-empty-road"
false positives.

### Sharding bug and fix

v4's two-GPU run used Python `hash()` to partition clips, which is randomised
per-process via `PYTHONHASHSEED=random`. The two shards rolled different hash
functions and silently overlapped: 8,390 clips scored twice, 8,427 never
scored (combined parquet: 33,730 rows, 25,340 unique). Fixed in
`camera_perception_scorer.py` by switching to `hashlib.md5(cid.encode())[:4]`
and added a `--skip-existing-glob` flag for fill-in runs. Post-fix fill-in
covered exactly the missing 8,427 clips with zero overlap, zero failures.

### Three distinct hard-clip examples (v5 top-5000)

Selected via "distinctness" = top_sub_score − mean(other active subs), one
per dominant dimension. Frame JPEGs + JSON metadata in
`nvidia_ingestion/hard_examples/`.

| # | Dominant | Score | Hour/Country | Season | Perception |
|---|----------|-------|--------------|--------|------------|
| 1 | time_of_day | 0.667 | 5 / United States | summer | 0.127 |
| 2 | sensor_coverage | 0.690 | 21 / Spain | winter | 0.414 |
| 3 | season_geography | 0.695 | 6 / Finland | spring | 0.497 |

All three have degraded sensor coverage + aggressive ego motion + night hour
but low-to-moderate perception scores, confirming they're hard by
environmental/sensor profile rather than visual scene complexity.

---

## 11. Canonical Schema Migration (Bronze build complete 2026-04-29)

**Directive (2026-04-28)**: Reshape Bronze tables to match `kaist_schema_v2.dbml` (the canonical cross-dataset schema). Replace existing tables in `iceberg.nvidia_bronze` (no new namespace). Keep empty tables for schema completeness on entities Nvidia doesn't source.

### Mapping spec — current Bronze → canonical

| Canonical table | PK | Source | Strategy |
|---|---|---|---|
| **Session** | session_id | none | empty table; Nvidia has no session concept |
| **Clip** | clip_id | clip_index ⨝ data_collection ⨝ selected_chunks | extract: `clip_id, session_id=NULL, clip_idx=row_number_within_chunk, frame_id_list=array of derived frame_ids, date=NULL, city=country, site=NULL`. `chunk` becomes a foreign-key into a Nvidia-only side table (or absorbed into clip_idx) |
| **Episode** | episode_id | none | empty |
| **Frame** | frame_id | egomotion (drives the time grid) | `frame_id = sha256(clip_id + ":" + sensor_timestamp)[:16]; clip_id from filename; episode_id=NULL; frame_idx=row_number per clip; sensor_timestamps=collect_list across sensors at this timestamp` |
| **Calibration** | (clip_id, sensor_name) | camera_intrinsics ∪ sensor_extrinsics ∪ vehicle_dimensions | one row per (clip_id, sensor_name). camera_intrinsics expands to JSON for `camera_intrinsics`; sensor_extrinsics → `extrinsics`. vehicle_dimensions → sensor_name='vehicle' |
| **Camera** | (clip_id, frame_id, camera_name) | cam_*_ts (×7) | clip_id from filename via `regexp_extract(input_file_name(), …)`; frame_id from sensor_timestamp; system_timestamp = sensor_timestamp; sensor_timestamp = `timestamp`; filename = mp4 sibling path (derived) |
| **Lidar** | (clip_id, frame_id) | lidar | clip_id from filename; frame_id from `spin_start_timestamp`; system_timestamp/sensor_timestamp = spin_start; filename = parquet path |
| **Radar** | (clip_id, frame_id, radar_name) | radar_* (×19, unioned) | clip_id from filename; frame_id from `sensor_timestamp`; radar_name = sensor name; filename = parquet path |
| **CanBus** | (clip_id, frame_id) | none | empty (Nvidia has no CAN bus data) |
| **HDMap** | filename | none | empty |
| **Session_EgoMotion** | session_id | none | empty |
| **Category** | category | none | empty |
| **DynamicObject** | (clip_id, frame_id) | (v26.03 obstacle.offline, not downloaded) | empty pending v26.03 download |
| **Occupancy** | (clip_id, frame_id) | none | empty |
| **Motion** | (clip_id, frame_id) | none | empty |
| **EgoMotion** | (session_id, clip_id, frame_id) | egomotion | clip_id from filename; session_id=NULL; frame_id = sha256(clip_id + ":" + timestamp); translation = (x,y,z); rotation = (qx,qy,qz,qw) |

### Synthesizing frame_id

The schema treats frame_id as a global PK that all sensor tables reference. Nvidia data has clips with continuous timestamps. Strategy:

```python
frame_id = sha256(f"{clip_id}:{sensor_timestamp}".encode()).hexdigest()[:16]
```

This yields a 16-char hex frame_id that is stable per (clip_id, sensor_timestamp). Different sensors writing the same timestamp produce the same frame_id, so all sensors at the same instant share a Frame row. **Caveat**: real-world sensors don't fire at exactly synchronized timestamps, so distinct sensors will mostly produce distinct frame_ids — this means the Frame table will have ~(# sensors × # timestamps) rows rather than just # timestamps. Open question whether to bucket timestamps to a coarser grid (e.g. 30 Hz) before hashing.

### Implementation sequence

1. Write `nvidia_ingestion/canonical_bronze.py` — new registrar that produces canonical-schema tables instead of zero-copy `add_files()`. This is NOT zero-copy; it does Spark transformations and CTAS.
2. Drop existing 41 Bronze tables; replace with the 16 canonical tables.
3. Update `quality_checks.py` (Silver) to operate on canonical schema — `missing_sensors` joins Clip ⨝ Camera/Lidar/Radar by clip_id properly.
4. Update `edge_case_scorer.py` (Gold) to score against canonical schema.
5. Update `pipeline.py`, `register_bronze.py` (deprecate or replace), `DATASET.md`.
6. Re-run end-to-end with benchmark capture.

### Decisions (settled 2026-04-28)

- **frame_id grid**: per-timestamp (no quantization) — schema is designed for a future synced-LiDAR/camera dataset; Nvidia data is unsynced so we accept multiple Frames per "moment in time"
- **Clip.session_id**: one session per chunk (3,116 sessions covering all chunks in clip_index, not just the on-disk 340)
- **Clip.date/city/site**: best-effort — `date = "XXXX-MM-01"` placeholder for unknown year, `city = country`, `site = NULL`
- **Calibration vehicle_dimensions row**: included as `(clip_id, sensor_name='vehicle')` rows
- **UINT_8 on radar**: not a blocker for the metadata-only canonical Radar (no payload columns referenced); projection-pruned reads succeeded

### Build run (2026-04-28 → 2026-04-29)

Implementation in [`canonical_bronze.py`](canonical_bronze.py). Run via `/tmp/run_canonical_bench.py`. **Total wall: 3.92h** (under the 6-12h estimate).

| Canonical table | Rows | Wall | Notes |
|-----------------|------|------|-------|
| Session | 3,116 | 5.5s | one per unique `chunk` in `clip_index` |
| Clip | 310,895 | 3.6s | full join clip_index ⨝ data_collection |
| Calibration | 458,873 | 4.7s | sensor_extrinsics ⨝ camera_intrinsics + vehicle_dimensions |
| Camera | 109,171,395 | 4m 23s* | 7 cam_*_ts unioned; *original was 5m 6s with broken filename, rebuilt 2026-04-29 |
| Lidar | 6,164,244 | 44m 27s | slow due to Draco-blob row-group scans even with projection |
| Radar | 11,730,962,796 | 98m | 19 radar tables unioned; UINT_8 column not projected so reads succeed |
| EgoMotion | 101,745,981 | 1m 33s | + clip_index join for session_id |
| Frame | 257,290,851 | 85m 31s | distinct (clip_id, sensor_timestamp) across egomotion+lidar+camera+radar |
| Episode, CanBus, HDMap, Session_EgoMotion, Category, DynamicObject, Occupancy, Motion | 0 each | <1s each | empty schema-only tables |
| **TOTAL** | **12,206,108,151** | **3.92h** | |

### Validation (2026-04-29)

15 sanity checks all pass after Camera rebuild — Camera filenames now resolve to actual mp4 files on NFS (initial build had filenames pointing into `.bronze_staging` symlink dir).

### Storage overhead (2026-04-29)

Per-table sizes via Iceberg `<table>.files` metadata. Total canonical Bronze is **16.02 GB across 12.2B rows** — **0.12% of the source NFS data size** (13 TB). Payload (lidar Draco blobs, radar detection vectors, MP4 video) stays in source files; canonical keeps only per-row metadata.

| Table | Rows | Bytes | GB | bytes/row | Files |
|-------|------|-------|----|-----------|-------|
| Session | 3,116 | 6.4 MB | 0.01 | 2,058 | 11 |
| Clip | 310,895 | 6.6 MB | 0.01 | 21.3 | 15 |
| Frame | 257,290,851 | 3.78 GB | 3.78 | 14.7 | 200 |
| Calibration | 458,873 | 27.0 MB | 0.03 | 58.9 | 110 |
| Camera | 109,171,395 | 1.82 GB | 1.82 | 16.6 | 45,145 |
| Lidar | 6,164,244 | 339.8 MB | 0.34 | 55.1 | 81,749 |
| Radar | 11,730,962,796 | 2.05 GB | 2.05 | **0.2** | 19,425 |
| EgoMotion | 101,745,981 | 7.99 GB | 8.00 | 78.6 | 2,111 |
| 8 empty tables | 0 | 0 | 0 | — | 0 |
| **TOTAL** | **12,206,108,151** | **16.02 GB** | **16.02** | **1.3** | **148,766** |

Notable points:
- **Radar at 0.2 bytes/row** — repetitive metadata columns (clip_id/frame_id/timestamps/radar_name/filename) compress to almost nothing under zstd
- **EgoMotion at 78.6 bytes/row** is the heaviest, driven by JSON-encoded `translation` / `rotation`. Switching to native struct encoding would likely cut it to ~30 bytes/row
- Lidar's 81K files for 6M rows (~74 rows/file) and Camera's 45K files (~2.4K rows/file) are good candidates for a future `OPTIMIZE` / compaction pass

### Camera filename fix

`cam_*_ts` Bronze tables were registered through `.bronze_staging/` symlink dirs with sequential chunk numbers, so `input_file_name()` returns the staging path (chunk_0268) not the real chunk path (chunk_2431). Camera filename is now reconstructed via Camera ⨝ clip_index on clip_id to retrieve the original chunk number, then formed as `file:<source_root>/camera/<sensor>/<sensor>.chunk_<chunk:04d>/<clip_id>.<sensor>.mp4`.

### Schema-table mapping

| Canonical table | PK | Source | Strategy |
|---|---|---|---|
| **Session** | session_id | clip_index group by chunk | session_id=`session_chunk_NNNN`; clip_id_list=JSON-encoded list |
| **Clip** | clip_id | clip_index ⨝ data_collection | session_id from chunk; clip_idx via row_number per chunk; date=`XXXX-MM-01` placeholder; city=country |
| **Episode** | episode_id | (none) | empty |
| **Frame** | frame_id | UNION of (clip_id, ts) across egomotion+lidar+camera+radar; per-source distinct first | frame_id=`substr(sha2(clip_id+":"+ts, 256), 1, 16)`; frame_idx=row_number per clip; sensor_timestamps=JSON `[ts]` (single-element since per-timestamp) |
| **Calibration** | (clip_id, sensor_name) | sensor_extrinsics FULL OUTER JOIN camera_intrinsics on (clip_id, sensor_name=camera_name), UNION vehicle_dimensions | extrinsics=JSON of qx/qy/qz/qw/x/y/z; camera_intrinsics=JSON of width/height/cx/cy/poly_coefs; vehicle row has dims-as-extrinsics |
| **Camera** | (clip_id, frame_id, camera_name) | cam_*_ts (×7) UNION ALL ⨝ clip_index | filename = derived mp4 path |
| **Lidar** | (clip_id, frame_id) | lidar | filename = parquet path; sensor_timestamp=spin_start; system_timestamp=spin_end |
| **Radar** | (clip_id, frame_id, radar_name) | radar_* (×19) UNION ALL | filename = parquet path |
| **CanBus** | (clip_id, frame_id) | (none) | empty |
| **HDMap** | filename | (none) | empty |
| **Session_EgoMotion** | session_id | (none) | empty |
| **Category** | category | (none) | empty |
| **DynamicObject** | (clip_id, frame_id) | v26.03 obstacle.offline (not downloaded) | empty pending |
| **Occupancy** | (clip_id, frame_id) | (none) | empty |
| **Motion** | (clip_id, frame_id) | (none) | empty |
| **EgoMotion** | (session_id, clip_id, frame_id) | egomotion ⨝ clip_index | translation/rotation as JSON of (x,y,z) / (qx,qy,qz,qw) |

### Pending work

- [x] Drop the 41 old Bronze tables (done 2026-04-29; slow due to per-table REST calls). Caveat: PURGE was used inadvertently → see Open Issue #11.
- [x] Update `quality_checks.py` (Silver) to query canonical schema — done 2026-04-29, 750 → 285 lines.
- [x] Update DATASET.md / DATASET_KO.md to describe canonical schema
- [x] Update `edge_case_scorer.py` (Gold) for canonical schema — done 2026-04-29 via aux-table strategy: `aux_data_collection`, `aux_sensor_presence` (now from `feature_presence.parquet` since v26.03 dropped `sensor_presence.parquet`), `aux_egomotion` registered zero-copy alongside canonical. Scorer pulls hour_of_day/month/country from aux_data_collection, radar booleans from aux_sensor_presence, ax/ay/az/curvature from aux_egomotion (canonical EgoMotion only kept JSON-encoded translation+rotation).
- [x] Run end-to-end Silver + Gold canonical with bench (done 2026-05-04 post-recovery). Total wall 69.1 min: aux 21.5m, Silver 23.2m, Gold 2.8m (+ summary write). 2.7× faster than pre-canonical 3h 6m baseline. Silver: 28,224 clips excluded → 282,671 valid. Gold: 33,719 Gold clips selected (top 10%, score range [0.187, 0.809], mean 0.491 ± 0.085).
- [x] **Re-tune timestamps Silver check** (done 2026-05-04). First fix: switched source from canonical Frame to aux_egomotion (dense per-clip timestamps, not cross-sensor-deduped). Second fix: raised `max_ts > 60M` to `max_ts > 200M` and `n < 100` to `n < 1000` after probing actual distribution — clips are ~140s long (median max_ts 139M), not 20s as docs claimed. Old thresholds flagged 100% of healthy clips. New thresholds yield 0 timestamp findings on healthy data. Silver re-run: **190 clips excluded** (was 28,224), Silver Clip 310,705 (~99.94% retention).
- [x] **Verify camera filenames resolve to mp4s** (done 2026-05-04 via `verify_cameras.py`): 140/140 sample mp4s resolve across all 7 cameras. Cameras intact through PURGE event because cam_*_ts/cam_*_blur Bronze tables were registered through `.bronze_staging/` symlinks; PURGE deleted symlinks but not symlink targets.

