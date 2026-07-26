/*
 * data.js — baked snapshot of real NetAI Lakehouse figures.
 *
 * These numbers are pulled from nvidia_ingestion/MEDALLION_PROGRESS.md,
 * benchmark_results.json, and hard_examples/examples.json. They are baked in
 * (not queried live) so the hallway display can NEVER show a broken chart if
 * Spark / Trino / Polaris restart. To refresh, edit this file and reload —
 * see refresh_snapshot.py for a helper that re-queries Trino.
 */
window.DEMO_DATA = {
  snapshot_date: "2026-06-01",
  dataset: {
    name: "NVIDIA PhysicalAI — Autonomous Vehicles",
    source: "Hugging Face Datasets",
    tb_downloaded: 10.85,
    archives_recovered: 3884,
    archives_total: 3933,
    clips: 310895,
    countries: 30,
    sensors: [
      { kind: "camera", label: "Cameras", count: 7, note: "120° / 70° / 30° FOV" },
      { kind: "lidar", label: "LiDAR", count: 1, note: "360° top, Draco-encoded" },
      { kind: "radar", label: "Radar", count: 19, note: "corner · front · side" },
      { kind: "ego", label: "EgoMotion", count: 1, note: "pose · velocity · accel" },
    ],
  },
  // Stage 2 — medallion processing. Canonical Bronze in Apache Iceberg.
  process: {
    bronze_rows: 12206108151,
    bronze_tables: 16,
    bronze_storage_gb: 16.02,
    source_tb: 13,
    bronze_build_hours: 3.92,
    tables: [
      { name: "Radar", rows: 11730962796 },
      { name: "Frame", rows: 257290851 },
      { name: "Camera", rows: 109171395 },
      { name: "EgoMotion", rows: 101745981 },
      { name: "Lidar", rows: 6164244 },
      { name: "Calibration", rows: 458873 },
      { name: "Clip", rows: 310895 },
      { name: "Session", rows: 3116 },
    ],
    // curation funnel (clips)
    funnel: [
      { tier: "Bronze", label: "Ingested as-is", clips: 310895, color: "#c87f3a" },
      { tier: "Silver", label: "Quality-filtered", clips: 310705, color: "#b9c2cc" },
      { tier: "Gold", label: "Edge-case subset", clips: 33719, color: "#e8b923" },
    ],
    silver_excluded: 190,
    silver_retention_pct: 99.94,
    silver_gold_minutes: 69.1,
    speedup_x: 2.7,
  },
  // Stage 3 — GPU perception scoring.
  gpu: {
    devices: [
      { name: "Quadro RTX 6000", vram_gb: 24, arch: "Turing" },
      { name: "NVIDIA A10", vram_gb: 23, arch: "Ampere" },
    ],
    models: ["BEVFusion (LiDAR + camera)", "YOLOv8 (camera)"],
    clips_scored: 33719,
    score_min: 0.187,
    score_max: 0.809,
    score_mean: 0.491,
    score_std: 0.085,
    // synthetic detections drawn over the camera frame (normalized 0..1 boxes)
    detections: [
      { x: 0.70, y: 0.70, w: 0.10, h: 0.10, label: "car", conf: 0.91 },
      { x: 0.62, y: 0.71, w: 0.05, h: 0.06, label: "car", conf: 0.78 },
      { x: 0.07, y: 0.55, w: 0.16, h: 0.22, label: "building", conf: 0.66 },
      { x: 0.86, y: 0.58, w: 0.12, h: 0.20, label: "vegetation", conf: 0.61 },
    ],
    // difficulty-score histogram (approx, 12 bins over [0.18, 0.81])
    histogram: [4, 11, 28, 96, 220, 410, 560, 470, 280, 120, 38, 9],
  },
  // Stage 4 — serving layer.
  serve: {
    services: [
      { name: "Apache Superset", role: "Dashboards & BI", port: 8088 },
      { name: "Trino", role: "Interactive SQL over Iceberg", port: 8080 },
      { name: "Apache Iceberg", role: "Table format (Polaris REST)", port: 8181 },
      { name: "MinIO", role: "S3 object store", port: 9001 },
    ],
    // the three real hardest-clip exemplars (real decoded front-wide frames)
    hard_clips: [
      {
        img: "assets/cam1.jpg",
        score: 0.667,
        factor: "time of day",
        where: "United States · summer · 05:00",
        why: "Pre-dawn low light, degraded sensor coverage, aggressive ego motion",
      },
      {
        img: "assets/cam2.jpg",
        score: 0.690,
        factor: "sensor coverage",
        where: "Spain · winter · 21:00",
        why: "Night driving, rare region+season, partial radar coverage",
      },
      {
        img: "assets/cam3.jpg",
        score: 0.695,
        factor: "season & geography",
        where: "Finland · spring · 06:00",
        why: "Rare northern-latitude scene, dawn light, dynamic egomotion",
      },
    ],
  },
  stack: ["Apache Iceberg", "Apache Spark", "Trino", "Polaris", "MinIO S3", "Apache Superset", "Docker"],

  // ── Per-stage wall-clock time. All figures are the CANONICAL pipeline runs
  // recorded in MEDALLION_PROGRESS.md §11 (the current 16-table schema), to stay
  // consistent with the 12.2 B-row / 16-table counts shown above.
  //   Ingest  = canonical Bronze build .............. 3.92 h
  //   Curate  = Silver (23.2m) + Gold (2.8m) tiers ... 26 min
  //   GPU     = perception is NOT yet benchmarked end-to-end (BEVFusion mid-
  //             integration), so we show the hardware, not a fabricated time.
  //   Serve   = Iceberg view creation (14.1s + 7.6s) . < 15 s
  stage_time: {
    ingest: "⏱ 3.9 h",
    curate: "⏱ 26 min",
    gpu: "2× GPU",
    serve: "⏱ < 15 s",
  },
  // Curate sub-step breakdown (decomposes the 26 min Curate badge).
  pipeline_steps: [
    { step: "Silver", time: "23 min", note: "quality filter" },
    { step: "Gold", time: "2.8 min", note: "edge-case scoring" },
  ],
  // Bronze (3.9 h) is the Ingest stage's time; download (10.85 TB, ~35 h) is
  // network-bound and shown on the Ingest progress bar, kept out of compute.

  // ── Hardware / runtime environment the lakehouse runs on ─────────────────
  environment: [
    { icon: "🖥", text: "24-core Xeon Silver 4310 · 188 GB RAM" },
    { icon: "⚡", text: "2× GPU — RTX 6000 24 GB + A10 23 GB" },
    { icon: "🗄", text: "NFS 639 MB/s · MinIO S3 · 100 GbE" },
    { icon: "◆", text: "Spark 3.5.5 · Iceberg v2 · Polaris" },
  ],

  // ── Dataset-level framing ────────────────────────────────────────────────
  // The flow is told in terms of WHOLE DATASETS: a dataset is uploaded, gets
  // ingested + curated ("trim the fat"), and lands in a catalog beside others.
  // NVIDIA PhysicalAI is the fully-processed flagship (real figures); the other
  // datasets are honest catalog entries the lakehouse is built to hold.
  active_dataset: {
    name: "NVIDIA PhysicalAI",
    uploader: "NVIDIA · Hugging Face",
    raw_size: "13 TB",
    modalities: "7 cameras · LiDAR · 19 radar · egomotion",
  },
  // Curate stage — "trim the fat": raw → curated reduction for the dataset.
  curate: {
    raw: "13 TB", raw_label: "raw sensor data",
    curated: "16 GB", curated_label: "curated metadata",
    pct_kept: "0.12%",
    clips_in: 310895, clips_out: 33719, clips_label: "clips kept after curation",
  },
  // Dataset catalog shown at the "Serve" stage — processed datasets side by side.
  catalog: [
    {
      name: "NVIDIA PhysicalAI", source: "NVIDIA · Hugging Face",
      status: "Active", kind: "active", color: "#76e36b",
      modalities: "7 cam · LiDAR · 19 radar", scale: "310,895 clips · 12.2 B rows",
      curated: "16 GB curated", note: "Edge-case Gold subset: 33,719 clips",
      thumb: "assets/cam1.jpg",
    },
    {
      name: "KAIST E2E", source: "KAIST / MOTIE", status: "Ingesting", kind: "progress",
      color: "#38d6ff", modalities: "cam · LiDAR · radar · HD map",
      scale: "Session → Clip → Frame", curated: "canonical schema v2",
      note: "Pipeline ready · shared canonical schema",
    },
    {
      name: "nuScenes v1.0", source: "Motional · public", status: "Benchmark", kind: "registered",
      color: "#b98bff", modalities: "6 cam · LiDAR · 5 radar",
      scale: "1,000 scenes · 1.4 M images", curated: "query-scalability study",
      note: "Reference dataset for lakehouse benchmarks",
    },
    {
      name: "NVIDIA Cosmos", source: "Synthetic · Cosmos NIM", status: "Synthetic", kind: "synthetic",
      color: "#e8b923", modalities: "multi-cam · simulation",
      scale: "scene augmentation", curated: "on-demand generation",
      note: "Synthetic scenes to enrich real datasets",
    },
  ],
};
