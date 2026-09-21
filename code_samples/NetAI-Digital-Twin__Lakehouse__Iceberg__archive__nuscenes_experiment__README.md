# Query Performance & Scalability Analysis: Apache Iceberg Lakehouse

This project investigates the impact of **data modeling strategies** on query performance and **scalability** within an **Apache Iceberg Data Lakehouse**.

> **Infrastructure:** Spark 3.5.5, Iceberg 1.8.1, Polaris REST catalog, MinIO (S3). All results collected on the current Polaris-based stack.

The experiments utilize the **[nuScenes v1.0-mini](https://www.nuscenes.org/nuscenes)** autonomous driving dataset to compare performance across three distinct data processing strategies: **Python Baseline (Nested Loop)**, **Silver JOIN (Normalized)**, and **Gold (Pre-Joined)**.

---

## 📂 1. Dataset Overview

We utilized a subset of the **[nuScenes v1.0](https://www.nuscenes.org/nuscenes)** dataset (nuScenes v1.0-mini). The raw data consists of directory-based sensor files (images, pcd) and highly structured metadata stored in JSON format.

### Source Data Structure (Raw Input)

The legacy system relies on parsing 7 core JSON tables relevant to object detection to reconstruct the dataset metadata. To generate a training dataset, the system must iteratively join these files based on token keys.

* **Core Metadata:** `sample.json`, `sample_data.json`
* **Sensor Metadata:** `sensor.json`, `calibrated_sensor.json`
* **Annotation Metadata:** `sample_annotation.json`, `instance.json`, `category.json`

> **⚠️ Python Loop Bottleneck:** Retrieving specific data (e.g., *"Front Camera only"*) requires scanning unrelated JSON objects and performing heavy in-memory joins using Python loops.

---

## 🏗️ 2. Experimental Strategies

This research compares three architectural approaches to handling this complex metadata, evaluating how data modeling affects scalability.

---

### Phase 1: Pure Python (Nested Loop)

* **Modeling Strategy:** **Unstructured (Schema-on-Read)**.
* The dataset remains as scattered sensor files and **7 distinct JSON metadata files**. No indexing or partitioning is applied.

* **Processing Method:** **Iterative Parsing & Nested Loops**.
* The system loads all JSON files into memory as Python dictionaries.
* Data linkage relies on **nested loops** to manually associate `sample` → `sample_data` → `annotation`.

* **Limitation:** Performance is severely **bottlenecked by CPU overhead** due to Python's iterative processing. Query time increases aggressively as data scale grows.

---

### Phase 2: Spark Iceberg — Silver JOIN (Normalized)

* **Modeling Strategy:** **Normalized Tables (Runtime Join)**.
* Data is migrated 1:1 into separate Iceberg tables (e.g., `iceberg.nusc_exp.samples`, `iceberg.nusc_exp.annotations`).
* **Optimization:** The `sample_data` table is **partitioned by `channel`**, enabling **Partition Pruning** during sensor data retrieval.

* **Processing Method:** **Distributed Runtime Joins**.
* Queries must execute complex **multi-way joins** at runtime (Compute-on-Read) to link the filtered sensor data with annotations.

* **Limitation:** Despite partition pruning on the sensor table, performance is **bottlenecked by Shuffle Join** overhead. The engine must scan the large, unpartitioned `annotations` table and shuffle data across the cluster to match records.

---

### Phase 3: Spark Iceberg — Gold (Pre-Joined)

* **Modeling Strategy:** **Denormalized Table (Pre-joined)**.
* All necessary features (Image paths, 3D Boxes, Categories, Channels) are **pre-computed** into a single table: `iceberg.nusc_exp.gold_train_set`.
* **Optimization:** The entire table is physically **partitioned by `channel`**, strictly aligning data storage with the query access pattern.

* **Processing Method:** **Zero-Join & Partition Pruning**.
* The query engine leverages **Partition Pruning** to read *only* the specific data files (e.g., `channel=CAM_FRONT`), skipping all unrelated data.
* No runtime joins are required.

* **Advantage:** **Eliminates Shuffle Joins** entirely and **minimizes I/O**, resulting in consistent sub-second query latency suitable for high-throughput ML data loading.

---

## ❄️ 3. Gold Table Schema Design (Denormalized)

The **Gold (pre-joined)** table consolidates all features into a flat schema optimized for ML data loading.

| Column Name | Data Type | Description | Source (Origin) |
| --- | --- | --- | --- |
| `img_path` | `string` | File path derived from `filename` | `sample_data` |
| `translation` | `array<double>` | 3D Global coordinates (x, y, z) | `sample_annotation` |
| `size` | `array<double>` | Object dimensions (w, l, h) | `sample_annotation` |
| `rotation` | `array<double>` | Orientation (Quaternion) | `sample_annotation` |
| `category_name` | `string` | Object classification | `category` |
| **`channel`** | **`string`** | **Sensor Channel (Partition Key)** | `sensor` |

> **Note on Partitioning:** The table is partitioned by the **`channel`** column (e.g., `CAM_FRONT`, `LIDAR_TOP`). This allows the query engine to instantly skip unrelated sensor partitions during file scanning.

---

## 🎯 4. Experimental Workload

We designed a specific query workload to measure the performance gap between the three strategies.

### Target Scenario

The experiment simulates a real-world Autonomous Driving ML data preparation task:

> *"Retrieve image paths and 3D bounding box parameters for **Adult Pedestrians** captured by the **Front Camera**."*

### Query Filters

To evaluate **Partition Pruning** and **Column Projection** capabilities, we applied the following filters:

1. **Sensor Filter:** `channel = 'CAM_FRONT'` (Tests Partition Pruning)
2. **Category Filter:** `category_name = 'human.pedestrian.adult'` (Tests Row Filtering)

### Final Output Data

All three experiments produce the exact same dataset required for training 3D object detection models:

```json
{
  "img_path": "samples/CAM_FRONT/n015-2018...jpg",
  "bbox_translation": [373.21, 1130.48, 1.25],
  "bbox_size": [0.62, 0.67, 1.64],
  "bbox_rotation": [0.98, 0.00, 0.00, -0.18]
}

```

---

## 📊 5. Performance Benchmarks

We measured query execution time across **18 scale factors from 1× to 50×** to evaluate scalability. The scale factor represents linear growth of observation data (`sample_data` replicated n-times) while reference tables (`category`, `sensor`, etc.) remain at 1×, reflecting realistic data growth patterns where sensor observations accumulate faster than metadata.

> **Methodology:** Python Baseline — median of 5 timed runs; Spark strategies — 1 warmup + median of 3 timed runs. Benchmark script: [`scalability_benchmark.py`](scalability_benchmark.py). Full data: [`scalability_results.json`](scalability_results.json).

### Scalability Chart

![Query Latency vs. Scale Factor](scalability_chart.png)

### Result Table

| SF | Effective Rows | Python Baseline (s) | Silver JOIN (s) | Gold (s) |
|---:|---:|---:|---:|---:|
| **1×** | 27,483 | 0.015 | 0.267 | 0.042 |
| **2×** | 54,966 | 0.031 | 0.227 | 0.047 |
| **3×** | 82,449 | 0.048 | 0.184 | 0.050 |
| **4×** | 109,932 | 0.063 | 0.214 | 0.054 |
| **5×** | 137,415 | 0.078 | 0.221 | 0.063 |
| **6×** | 164,898 | 0.092 | 0.200 | 0.055 |
| **7×** | 192,381 | 0.106 | 0.202 | 0.060 |
| **8×** | 219,864 | 0.121 | 0.199 | 0.050 |
| **9×** | 247,347 | 0.136 | 0.213 | 0.058 |
| **10×** | 274,830 | 0.153 | 0.198 | 0.060 |
| **15×** | 412,245 | 0.220 | 0.242 | 0.069 |
| **20×** | 549,660 | 0.297 | 0.297 | 0.068 |
| **25×** | 687,075 | 0.373 | 0.351 | 0.073 |
| **30×** | 824,490 | 0.447 | 0.385 | 0.075 |
| **35×** | 961,905 | 0.492 | 0.397 | 0.082 |
| **40×** | 1,099,320 | 0.622 | 0.437 | 0.091 |
| **45×** | 1,236,735 | 0.623 | 0.481 | 0.082 |
| **50×** | 1,374,150 | **0.733** | **0.499** | **0.087** |

### Key Findings

1. **Python Baseline — Linear Degradation:**
    * Performance degrades linearly from 0.015 s → 0.733 s (SF 1→50). Dictionary lookups and sequential processing become the dominant bottleneck. Scales at ~O(n).

2. **Silver JOIN — Sub-Linear Growth with JIT Floor:**
    * At low scale factors (SF ≤ 10), Spark's JIT warmup overhead dominates (~0.20 s floor). Beyond SF 20, the strategy surpasses the Python baseline and grows sub-linearly to 0.499 s at SF 50. The crossover occurs around **SF ≈ 20** where both strategies tie at ~0.297 s.

3. **Gold Table — Near-Constant Latency:**
    * The Gold (pre-joined) layout maintains **sub-100 ms latency** across the entire 1×–50× range (0.042 s → 0.087 s), achieving **~8.4× speedup** over Python and **~5.7× over Silver JOIN** at SF 50.
    * Partition pruning by `channel` eliminates 83.3% of data at scan time; pre-computation eliminates all runtime joins. Growth is bounded by I/O scan of a single partition.

4. **Scalability Gap Widens with Scale:**
    * The performance gap between Gold and the other strategies grows with scale — at SF 1, Python is actually faster (0.015 s vs. 0.042 s) due to Spark's overhead. But by SF 10 the gap is 2.6× and by SF 50 it reaches 8.4×. This demonstrates that pre-computation costs are amortized quickly.

> **See also:** For KAIST AD-specific benchmarks (object detection, SLAM, sensor fusion), see [`benchmarks/benchmark_results.json`](../benchmarks/benchmark_results.json).
