#!/usr/bin/env python3
"""Generate all figures for the slide deck presentation."""

import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import numpy as np
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
OUT = BASE / "paper" / "figures"
OUT.mkdir(exist_ok=True)

# ============================================================================
# Color palette (professional, presentation-ready)
# ============================================================================
C_GOLD = "#E8A838"
C_SILVER = "#6C8EBF"
C_PYTHON = "#D4526E"
C_BRONZE = "#CD7F32"
C_BG = "#FAFAFA"
C_GRID = "#E0E0E0"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 13,
    "axes.facecolor": C_BG,
    "figure.facecolor": "white",
    "axes.grid": True,
    "grid.color": C_GRID,
    "grid.alpha": 0.6,
})


# ============================================================================
# Figure 1: Data Flow Diagram — new AD dataset → ML-ready tables
# ============================================================================
def fig1_data_flow():
    fig, ax = plt.subplots(figsize=(18, 9))
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 9)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    # --- Helper: rounded box ---
    def draw_box(x, y, w, h, label, color, sublabel=None, fontsize=12,
                 text_color="white", border_color="#333333"):
        rect = mpatches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.12",
            facecolor=color, edgecolor=border_color, linewidth=1.5, alpha=0.92
        )
        ax.add_patch(rect)
        ty = y + h/2 + (0.20 if sublabel else 0)
        ax.text(x + w/2, ty, label,
                ha="center", va="center", fontsize=fontsize, fontweight="bold",
                color=text_color,
                path_effects=[pe.withStroke(linewidth=2, foreground="#00000020")])
        if sublabel:
            ax.text(x + w/2, y + h/2 - 0.25, sublabel,
                    ha="center", va="center", fontsize=8.5, color=text_color,
                    alpha=0.8)

    # --- Helper: arrow ---
    def arrow(x1, y1, x2, y2, color="#555555", lw=2.0, style="-|>"):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                     arrowprops=dict(arrowstyle=style, color=color, lw=lw,
                                     mutation_scale=16))

    # --- Helper: label on arrow ---
    def arrow_label(x, y, text, fontsize=8, color="#555"):
        ax.text(x, y, text, ha="center", va="center", fontsize=fontsize,
                color=color, fontstyle="italic",
                bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                          edgecolor="none", alpha=0.85))

    # ====================== TITLE ======================
    ax.text(9.0, 8.6, "Data Flow: New AD Dataset → ML-Ready Tables",
            ha="center", va="center", fontsize=16, fontweight="bold", color="#333")

    # ====================== STAGE 0: DATA SOURCE ======================
    # Vehicle / raw data source
    draw_box(0.2, 5.6, 2.4, 1.6, "Vehicle Data", "#7F8C8D",
             "14 JSON files", fontsize=11)
    # Show the sensor types fanning out
    sensor_labels = ["camera ×6", "lidar", "radar", "ego_pose",
                     "annotations", "calibration", "hdmap", "..."]
    for i, sl in enumerate(sensor_labels):
        row = i // 2
        col = i % 2
        ax.text(0.35 + col * 1.2, 5.35 - row * 0.3, sl,
                fontsize=7, color="#666", family="monospace")

    # ====================== STAGE 1: BRONZE ======================
    bx = 3.8
    draw_box(bx, 5.2, 2.6, 2.4, "Bronze", C_BRONZE, fontsize=14)
    # Sub-detail inside the box
    bronze_lines = [
        "Schema enforcement",
        "1 JSON → 1 Iceberg table",
        "14 tables, immutable",
        "Raw data preserved",
    ]
    for i, line in enumerate(bronze_lines):
        ax.text(bx + 1.3, 6.9 - i * 0.35, line, ha="center", va="center",
                fontsize=7.5, color="white", alpha=0.9)

    arrow(2.6, 6.4, 3.8, 6.4, color=C_BRONZE, lw=2.5)
    arrow_label(3.2, 6.75, "ingest_bronze.py")

    # ====================== STAGE 2: SILVER ======================
    sx = 7.6
    draw_box(sx, 5.2, 2.8, 2.4, "Silver", C_SILVER, fontsize=14)
    silver_lines = [
        "Partition by access pattern",
        "Sort by timestamp",
        "Column min/max stats",
        "11 tables optimized",
    ]
    for i, line in enumerate(silver_lines):
        ax.text(sx + 1.4, 6.9 - i * 0.35, line, ha="center", va="center",
                fontsize=7.5, color="white", alpha=0.9)

    arrow(6.4, 6.4, 7.6, 6.4, color=C_SILVER, lw=2.5)
    arrow_label(7.0, 6.75, "transform_silver.py")

    # ====================== STAGE 3: GOLD ======================
    gx = 11.6
    draw_box(gx, 5.2, 2.8, 2.4, "Gold", C_GOLD, fontsize=14)
    gold_lines = [
        "Pre-joined per ML task",
        "camera_annotations (6⋈)",
        "lidar_with_ego (3⋈)",
        "sensor_fusion_frame (5⋈+3agg)",
    ]
    for i, line in enumerate(gold_lines):
        ax.text(gx + 1.4, 6.9 - i * 0.35, line, ha="center", va="center",
                fontsize=7.5, color="white" if i == 0 else "#333",
                alpha=0.9 if i == 0 else 0.85,
                fontweight="bold" if i == 0 else "normal")

    arrow(10.4, 6.4, 11.6, 6.4, color=C_GOLD, lw=2.5)
    arrow_label(11.0, 6.75, "build_gold.py")

    # ====================== STAGE 4: VALIDATE ======================
    vx = 15.4
    draw_box(vx, 5.6, 2.2, 1.6, "Validate", "#27AE60",
             "20 checks", fontsize=12)
    val_lines = ["PK unique", "FK intact", "‖q‖≈1", "ts≥0", "row counts"]
    for i, vl in enumerate(val_lines):
        ax.text(vx + 1.1, 6.5 - i * 0.28, vl, ha="center", va="center",
                fontsize=7, color="white", alpha=0.85)

    arrow(14.4, 6.4, 15.4, 6.4, color="#27AE60", lw=2.5)
    arrow_label(14.9, 6.75, "validators.py")

    # ====================== BOTTOM ROW: CONSUMERS ======================
    # Three consumption paths from Gold

    # Consumer 1: ML Training
    draw_box(7.0, 1.0, 2.6, 1.5, "ML Training", "#8E44AD",
             "DataLoader reads 1 table", fontsize=11)
    # Consumer 2: SQL / Trino
    draw_box(10.2, 1.0, 2.4, 1.5, "SQL Queries", "#7B68EE",
             "Trino interactive", fontsize=11)
    # Consumer 3: Dashboard / Superset
    draw_box(13.2, 1.0, 2.6, 1.5, "Dashboards", "#E67E22",
             "Superset BI", fontsize=11)

    # Arrows from Gold down to consumers
    for cx in [8.3, 11.4, 14.5]:
        arrow(13.0, 5.2, cx, 2.5, color="#888888", lw=1.2, style="-|>")

    # Label the consumption section
    ax.text(11.7, 3.2, "Consumption", ha="center", va="center",
            fontsize=11, fontweight="bold", color="#888", fontstyle="italic")
    ax.text(11.7, 2.8, "(all engines read via Polaris catalog — same tables, no copies)",
            ha="center", va="center", fontsize=8, color="#aaa")

    # ====================== INFRASTRUCTURE STRIP (bottom) ======================
    infra_y = 0.15
    ax.text(9.0, infra_y,
            "Infrastructure:  Spark 3.5.5  ·  Iceberg 1.8.1  ·  Polaris (REST catalog)"
            "  ·  MinIO (S3)  ·  Trino 479  ·  Superset  ·  Docker Compose",
            ha="center", va="center", fontsize=8.5, color="#999", fontstyle="italic")

    # ====================== TIMING ANNOTATION ======================
    # Show the full pipeline time
    ax.annotate("", xy=(15.4, 8.1), xytext=(3.8, 8.1),
                arrowprops=dict(arrowstyle="|-|", color="#CCC", lw=1.5,
                                mutation_scale=8))
    ax.text(9.6, 8.25, "~24 s end-to-end (simulated dataset, single node)",
            ha="center", va="center", fontsize=9, color="#999")

    # ====================== SIDE ANNOTATION: what Iceberg provides ======================
    # Small callout on the right
    iceberg_features = [
        "Iceberg provides:",
        "• ACID transactions",
        "• Time travel / snapshots",
        "• Schema evolution",
        "• Partition pruning",
        "• Column-level statistics",
    ]
    for i, feat in enumerate(iceberg_features):
        ax.text(16.0, 4.7 - i * 0.32, feat, fontsize=7.5,
                color="#34495E" if i == 0 else "#666",
                fontweight="bold" if i == 0 else "normal")

    # ====================== SIDE ANNOTATION: what stays on disk ======================
    ax.add_patch(mpatches.FancyBboxPatch(
        (0.2, 1.0), 2.4, 2.8, boxstyle="round,pad=0.1",
        facecolor="#F8F8F8", edgecolor="#DDD", linewidth=1.0
    ))
    ax.text(1.4, 3.5, "On MinIO (S3)", ha="center", fontsize=8.5,
            fontweight="bold", color="#555")
    storage_lines = [
        "kaist_bronze/  (14 tables)",
        "kaist_silver/  (11 tables)",
        "kaist_gold/    (3 tables)",
        "user_data/     (raw JSON)",
    ]
    for i, sl in enumerate(storage_lines):
        ax.text(1.4, 3.1 - i * 0.35, sl, ha="center", fontsize=7.5,
                color="#666", family="monospace")
    ax.text(1.4, 1.55, "Storage pluggable:\nMinIO → Ceph / AWS S3",
            ha="center", fontsize=7, color="#999", fontstyle="italic")

    # Arrow from raw data down to storage
    arrow(1.4, 5.6, 1.4, 3.85, color="#AAA", lw=1.0, style="->")
    arrow_label(0.7, 4.7, "raw JSON\npreserved")

    fig.savefig(OUT / "data_flow.png", dpi=200, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)
    print(f"  ✓ {OUT / 'data_flow.png'}")


# ============================================================================
# Figure 2: Data Model — 3-Level Hierarchy
# ============================================================================
def fig2_data_model():
    fig, ax = plt.subplots(figsize=(15, 8.5))
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 8.5)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    def draw_entity(x, y, w, h, label, color, sublabel=None, fontsize=12):
        rect = mpatches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.12",
            facecolor=color, edgecolor="#333333", linewidth=1.5, alpha=0.92
        )
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2 + (0.15 if sublabel else 0), label,
                ha="center", va="center", fontsize=fontsize, fontweight="bold", color="white")
        if sublabel:
            ax.text(x + w/2, y + h/2 - 0.22, sublabel,
                    ha="center", va="center", fontsize=8.5, color="white", alpha=0.8)

    # Level 1: Session
    draw_entity(6.0, 6.8, 3.0, 0.9, "Session", "#8E44AD", "Long-duration drive")

    # Level 2: Clip
    draw_entity(6.0, 4.8, 3.0, 0.9, "Clip", "#2980B9", "Contiguous segment")

    # Level 3: Frame
    draw_entity(6.0, 2.8, 3.0, 0.9, "Frame", "#27AE60", "Single time-step")

    # Sensors & Annotations (Level 4) — spread wider
    sensor_y = 0.8
    draw_entity(0.3, sensor_y, 2.3, 0.9, "Camera", "#E74C3C", "6 views")
    draw_entity(3.1, sensor_y, 2.3, 0.9, "LiDAR", "#D35400", "Point cloud")
    draw_entity(5.9, sensor_y, 2.3, 0.9, "Radar", "#F39C12", "Doppler")
    draw_entity(8.7, sensor_y, 2.5, 0.9, "Ego Motion", "#16A085", "SE3 pose")
    draw_entity(11.7, sensor_y, 2.8, 0.9, "Annotations", "#7F8C8D", "3D boxes, cat.")

    # Arrows
    arrow_kw = dict(arrowstyle="-|>", color="#555555", lw=1.8, mutation_scale=14)
    ax.annotate("", xy=(7.5, 5.7), xytext=(7.5, 6.8), arrowprops=arrow_kw)
    ax.annotate("", xy=(7.5, 3.7), xytext=(7.5, 4.8), arrowprops=arrow_kw)
    # Frame to sensors
    for tx in [1.45, 4.25, 7.05, 9.95, 13.1]:
        ax.annotate("", xy=(tx, 1.7), xytext=(7.5, 2.8), arrowprops=dict(
            arrowstyle="-|>", color="#888888", lw=1.0, mutation_scale=10
        ))

    # Multiplicity labels — offset from arrows
    ax.text(7.8, 6.2, "1 : N", fontsize=9, color="#555", fontstyle="italic")
    ax.text(7.8, 4.25, "1 : N", fontsize=9, color="#555", fontstyle="italic")
    ax.text(4.5, 2.2, "1 : N per sensor", fontsize=9, color="#555", fontstyle="italic")

    # Title / legend
    ax.text(7.5, 8.1, "KAIST 3-Level Data Hierarchy: Session \u2192 Clip \u2192 Frame \u2192 Sensors",
            ha="center", fontsize=14, fontweight="bold", color="#333333")
    ax.text(7.5, 0.2, "14 entity types  \u00b7  Named geometric structs (SE3, Quaternion, Box3D)  \u00b7  Generalizes nuScenes 2-level model",
            ha="center", fontsize=9.5, color="#888888", fontstyle="italic")

    fig.savefig(OUT / "data_model.png", dpi=200, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)
    print(f"  ✓ {OUT / 'data_model.png'}")


# ============================================================================
# Figure 3: Medallion Pipeline (Bronze → Silver → Gold)
# ============================================================================
def fig3_medallion():
    fig, ax = plt.subplots(figsize=(16, 7))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 7)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    def draw_layer_box(x, y, w, h, title, color, items, title_fontsize=14):
        # Main box
        rect = mpatches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.15",
            facecolor=color, edgecolor="#333333", linewidth=2, alpha=0.15
        )
        ax.add_patch(rect)
        # Header bar
        header = mpatches.FancyBboxPatch(
            (x, y+h-0.7), w, 0.7, boxstyle="round,pad=0.1",
            facecolor=color, edgecolor="none", alpha=0.85
        )
        ax.add_patch(header)
        ax.text(x + w/2, y + h - 0.35, title, ha="center", va="center",
                fontsize=title_fontsize, fontweight="bold", color="white")
        for i, item in enumerate(items):
            ax.text(x + 0.3, y + h - 1.2 - i*0.45, f"• {item}",
                    fontsize=9, color="#333333", va="center")

    # Bronze
    draw_layer_box(0.3, 0.7, 4.2, 5.5, "Bronze Layer", C_BRONZE, [
        "1:1 JSON \u2192 Iceberg tables",
        "14 tables, all entity types",
        "Schema-enforced ingestion",
        "No transformations",
        "Preserves lineage",
        "ACID guarantees",
    ])

    # Silver
    draw_layer_box(5.7, 0.7, 4.6, 5.5, "Silver Layer", C_SILVER, [
        "Domain-aware partitioning",
        "camera \u2192 by camera_name, clip_id",
        "Iceberg sort orders (temporal)",
        "Column-level min/max metrics",
        "Predicate pushdown enabled",
        "Snapshot retention (time travel)",
        "11 tables optimized",
    ])

    # Gold
    draw_layer_box(11.5, 0.7, 4.2, 5.5, "Gold Layer", C_GOLD, [
        "camera_annotations (Obj. Det.)",
        "  \u2192 6-table pre-join, by camera",
        "lidar_with_ego (SLAM)",
        "  \u2192 3-table pre-join, by clip",
        "sensor_fusion_frame (Fusion)",
        "  \u2192 5-table pre-join + 3 aggs",
        "Zero runtime joins",
    ])

    # Arrows between layers
    arrow_kw = dict(arrowstyle="-|>", color="#333333", lw=2.5, mutation_scale=20)
    ax.annotate("", xy=(5.6, 3.45), xytext=(4.6, 3.45), arrowprops=arrow_kw)
    ax.annotate("", xy=(11.4, 3.45), xytext=(10.4, 3.45), arrowprops=arrow_kw)

    ax.text(5.1, 4.0, "Partition &\nOptimize", ha="center", fontsize=8.5, color="#555",
            fontweight="bold")
    ax.text(10.9, 4.0, "Pre-join &\nDenormalize", ha="center", fontsize=8.5, color="#555",
            fontweight="bold")

    ax.text(8.0, 0.2, "Medallion Architecture \u2014 Each layer adds optimizations for AD workloads",
            ha="center", fontsize=10, color="#888888", fontstyle="italic")

    fig.savefig(OUT / "medallion_pipeline.png", dpi=200, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)
    print(f"  ✓ {OUT / 'medallion_pipeline.png'}")


# ============================================================================
# Figure 4: Gold vs Silver Benchmark (3 Workloads) — Grouped Bar Chart
# ============================================================================
def fig4_workload_benchmark():
    with open(BASE / "benchmarks" / "benchmark_results.json") as f:
        data = json.load(f)

    three_wl = [d for d in data if d["experiment"] == "Three-Workload"]

    workloads = ["Object Detection", "SLAM / Localization", "Multi-Modal Fusion"]
    gold_times = []
    silver_times = []
    row_counts = []

    for wl in workloads:
        gold = [d for d in three_wl if d["variant"].startswith("Gold:") and wl.split("/")[0].strip() in d["variant"]][0]
        silver = [d for d in three_wl if d["variant"].startswith("Silver JOIN:") and wl.split("/")[0].strip() in d["variant"]][0]
        gold_times.append(gold["elapsed_seconds"] * 1000)  # ms
        silver_times.append(silver["elapsed_seconds"] * 1000)
        row_counts.append(gold["row_count"])

    x = np.arange(len(workloads))
    width = 0.32

    fig, ax = plt.subplots(figsize=(10, 5.5))
    bars_gold = ax.bar(x - width/2, gold_times, width, label="Gold (pre-joined)",
                        color=C_GOLD, edgecolor="#333", linewidth=0.8, zorder=3)
    bars_silver = ax.bar(x + width/2, silver_times, width, label="Silver (runtime JOIN)",
                          color=C_SILVER, edgecolor="#333", linewidth=0.8, zorder=3)

    # Speedup annotations
    speedups = ["3.2×", "2.2×", "2.0×"]
    for i, sp in enumerate(speedups):
        ax.annotate(sp, xy=(x[i] + width/2, silver_times[i]),
                     xytext=(0, 8), textcoords="offset points",
                     ha="center", fontsize=11, fontweight="bold", color=C_PYTHON)

    # Value labels on bars
    for bar in bars_gold:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f"{bar.get_height():.0f}", ha="center", va="bottom", fontsize=9, color="#555")
    for bar in bars_silver:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f"{bar.get_height():.0f}", ha="center", va="bottom", fontsize=9, color="#555")

    ax.set_ylabel("Query Latency (ms)", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(workloads, fontsize=11)
    ax.set_title("Gold vs. Silver Query Latency — Three AD Workloads", fontsize=14, fontweight="bold", pad=15)
    ax.legend(fontsize=11, loc="upper right")
    ax.set_ylim(0, max(silver_times) * 1.35)
    fig.tight_layout(pad=1.5)

    fig.savefig(OUT / "workload_benchmark.png", dpi=200, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)
    print(f"  ✓ {OUT / 'workload_benchmark.png'}")


# ============================================================================
# Figure 5: Scalability Line Chart (nuScenes 1×–50×)
# ============================================================================
def fig5_scalability():
    with open(BASE / "benchmarks" / "kaist_scalability_results.json") as f:
        data = json.load(f)

    strategies = {"Python Baseline": C_PYTHON, "Silver JOIN": C_SILVER, "Gold": C_GOLD}

    fig, ax = plt.subplots(figsize=(11, 6))

    for strat, color in strategies.items():
        pts = sorted([d for d in data if d["strategy"] == strat], key=lambda d: d["scale_factor"])
        sfs = [d["scale_factor"] for d in pts]
        times = [d["elapsed_seconds"] * 1000 for d in pts]  # ms
        marker = "o" if strat == "Python Baseline" else ("s" if strat == "Silver JOIN" else "D")
        ax.plot(sfs, times, marker=marker, label=strat, color=color,
                linewidth=2.2, markersize=5, zorder=3)

    # Highlight reference lines
    ax.axhline(y=100, color="#999", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.text(1020, 100, "100 ms", fontsize=8, color="#999", va="bottom")
    ax.axhline(y=1000, color="#999", linestyle="--", linewidth=0.8, alpha=0.4)
    ax.text(1020, 1000, "1 s", fontsize=8, color="#999", va="bottom")

    # Annotations at SF=1000
    ax.annotate("4,661 ms", xy=(1000, 4661), xytext=(850, 4200), fontsize=9, color=C_PYTHON,
                fontweight="bold", arrowprops=dict(arrowstyle="-", color=C_PYTHON, lw=0.8))
    ax.annotate("1,204 ms", xy=(1000, 1204), xytext=(850, 1600), fontsize=9, color=C_SILVER,
                fontweight="bold", arrowprops=dict(arrowstyle="-", color=C_SILVER, lw=0.8))
    ax.annotate("33 ms", xy=(1000, 33), xytext=(900, 350), fontsize=9, color=C_GOLD,
                fontweight="bold", arrowprops=dict(arrowstyle="->", color=C_GOLD, lw=0.8))

    # Speedup box at SF=1000
    ax.text(30, 3800, "At SF 1000× (23.4 M rows):\n"
                      "Gold 140× faster than Python\n"
                      "Gold  36× faster than Silver",
            fontsize=10, fontweight="bold", color=C_GOLD,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#F0FFF0", edgecolor=C_GOLD, alpha=0.85))

    # Gold flat-line annotation
    ax.annotate("Gold remains flat:\n29–49 ms across all SFs",
                xy=(500, 40), xytext=(300, 700),
                fontsize=9, color="#555", ha="center",
                arrowprops=dict(arrowstyle="->", color="#555", lw=0.8))

    ax.set_xlabel("Scale Factor", fontsize=12)
    ax.set_ylabel("Query Latency (ms)", fontsize=12)
    ax.set_title("Scalability: Latency vs. Data Scale (KAIST Tiers, 1\u00d7\u20131000\u00d7)", fontsize=14, fontweight="bold", pad=15)
    ax.legend(fontsize=11, loc="upper left")
    ax.set_xlim(-20, 1080)
    ax.set_ylim(0, 5500)
    fig.tight_layout(pad=1.5)

    fig.savefig(OUT / "scalability.png", dpi=200, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)
    print(f"  ✓ {OUT / 'scalability.png'}")


# ============================================================================
# Figure 6: Supplementary Benchmarks Summary (Partition Pruning, Time Travel, etc.)
# ============================================================================
def fig6_supplementary():
    with open(BASE / "benchmarks" / "benchmark_results.json") as f:
        data = json.load(f)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))

    # --- Panel A: Partition Pruning ---
    ax = axes[0]
    pruning = [d for d in data if d["experiment"] == "Partition Pruning"]
    labels = ["No filter\n(all parts.)", "1 partition\n(sensor)", "Combined\n(sensor+clip)"]
    vals = [pruning[0]["elapsed_seconds"]*1000, pruning[1]["elapsed_seconds"]*1000, pruning[2]["elapsed_seconds"]*1000]
    rows = [pruning[0]["row_count"], pruning[1]["row_count"], pruning[2]["row_count"]]
    colors_p = ["#95A5A6", C_SILVER, C_GOLD]
    bars = ax.bar(labels, vals, color=colors_p, edgecolor="#333", linewidth=0.8, width=0.6, zorder=3)
    ax.set_ylim(0, max(vals) * 1.35)
    for bar, rc in zip(bars, rows):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+max(vals)*0.04,
                f"{rc:,} rows", ha="center", fontsize=8, color="#555")
    ax.set_ylabel("Latency (ms)", fontsize=10)
    ax.set_title("Partition Pruning", fontsize=12, fontweight="bold", pad=10)
    ax.text(1.0, vals[1]*0.7, "83.3%\nskipped", ha="center", fontsize=9, color="white", fontweight="bold")

    # --- Panel B: Temporal Replay (Gold vs Silver) ---
    ax = axes[1]
    temporal = [d for d in data if d["experiment"] == "Temporal Replay"]
    gold_t = [d for d in temporal if d["variant"].startswith("Gold")][0]
    silver_t = [d for d in temporal if d["variant"].startswith("Silver JOIN")][0]
    labels_t = ["Gold\n(pre-sorted)", "Silver JOIN\n(runtime sort)"]
    vals_t = [gold_t["elapsed_seconds"]*1000, silver_t["elapsed_seconds"]*1000]
    bars = ax.bar(labels_t, vals_t, color=[C_GOLD, C_SILVER], edgecolor="#333", linewidth=0.8, width=0.5, zorder=3)
    ax.set_title("Temporal Replay", fontsize=12, fontweight="bold", pad=10)
    ax.set_ylabel("Latency (ms)", fontsize=10)
    ax.set_ylim(0, max(vals_t) * 1.3)
    ax.annotate("1.8\u00d7", xy=(1, vals_t[1]), xytext=(0, 10), textcoords="offset points",
                ha="center", fontsize=11, fontweight="bold", color=C_PYTHON)

    # --- Panel C: Column Metrics ---
    ax = axes[2]
    metrics = [d for d in data if d["experiment"] == "Column Metrics"]
    narrow = [d for d in metrics if "Narrow" in d["variant"]][0]
    full = [d for d in metrics if "Full" in d["variant"]][0]
    labels_m = ["Narrow\ntimestamp", "Full scan\n(no filter)"]
    vals_m = [narrow["elapsed_seconds"]*1000, full["elapsed_seconds"]*1000]
    bars = ax.bar(labels_m, vals_m, color=[C_GOLD, "#95A5A6"], edgecolor="#333", linewidth=0.8, width=0.5, zorder=3)
    ax.set_title("Column-Level Metrics", fontsize=12, fontweight="bold", pad=10)
    ax.set_ylabel("Latency (ms)", fontsize=10)
    ax.set_ylim(0, max(vals_m) * 1.3)
    ax.annotate("4.8\u00d7", xy=(1, vals_m[1]), xytext=(0, 10), textcoords="offset points",
                ha="center", fontsize=11, fontweight="bold", color=C_PYTHON)

    fig.tight_layout(pad=3.0)
    fig.subplots_adjust(left=0.06)
    fig.suptitle("Supplementary Iceberg Features — Validated on KAIST Dataset", fontsize=13, fontweight="bold", y=1.03)
    fig.savefig(OUT / "supplementary_benchmarks.png", dpi=200, bbox_inches="tight", pad_inches=0.5)
    plt.close(fig)
    print(f"  ✓ {OUT / 'supplementary_benchmarks.png'}")


# ============================================================================
# Figure 7: Gold Table Design Overview
# ============================================================================
def fig7_gold_tables():
    fig, ax = plt.subplots(figsize=(16, 5.5))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 5.5)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    table_data = [
        {
            "name": "camera_annotations",
            "workload": "Object Detection",
            "joins": "6 Silver tables",
            "partition": "camera_name",
            "rows": "140K annotations",
            "color": "#E74C3C",
            "x": 0.3,
        },
        {
            "name": "lidar_with_ego",
            "workload": "SLAM / Localization",
            "joins": "3 Silver tables",
            "partition": "clip_id",
            "rows": "3,935 frames",
            "color": "#2980B9",
            "x": 5.4,
        },
        {
            "name": "sensor_fusion_frame",
            "workload": "Multi-Modal Fusion",
            "joins": "5 Silver + 3 aggs",
            "partition": "clip_id",
            "rows": "3,935 frames",
            "color": "#27AE60",
            "x": 10.5,
        },
    ]

    for td in table_data:
        x = td["x"]
        w = 4.8
        # Main card
        rect = mpatches.FancyBboxPatch(
            (x, 0.5), w, 4.0, boxstyle="round,pad=0.15",
            facecolor="white", edgecolor=td["color"], linewidth=2.5
        )
        ax.add_patch(rect)
        # Header
        header = mpatches.FancyBboxPatch(
            (x, 3.6), w, 0.9, boxstyle="round,pad=0.1",
            facecolor=td["color"], edgecolor="none", alpha=0.9
        )
        ax.add_patch(header)
        ax.text(x+w/2, 4.05, td["name"], ha="center", va="center",
                fontsize=11, fontweight="bold", color="white", family="monospace")
        # Content
        lines = [
            f"ML Workload:  {td['workload']}",
            f"Joins eliminated:  {td['joins']}",
            f"Partition key:  {td['partition']}",
            f"Scale:  {td['rows']}",
        ]
        for i, line in enumerate(lines):
            ax.text(x + 0.35, 3.1 - i*0.6, line, fontsize=10, color="#333", va="center")

    ax.text(8.0, 5.15, "Gold Table Designs \u2014 Pre-Joined, ML-Ready", ha="center",
            fontsize=14, fontweight="bold", color="#333333")

    fig.savefig(OUT / "gold_tables.png", dpi=200, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)
    print(f"  ✓ {OUT / 'gold_tables.png'}")


# ============================================================================
# Figure 8: Validation Summary
# ============================================================================
def fig8_validation():
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.axis("off")
    fig.patch.set_facecolor("white")

    # Table data
    cols = ["Check Type", "Count", "Status"]
    rows = [
        ["PK Uniqueness", "6", "PASS"],
        ["FK Integrity", "4", "PASS"],
        ["Quaternion Normalization", "2", "PASS"],
        ["Non-negative Timestamps", "4", "PASS"],
        ["Gold Row Consistency", "4", "PASS"],
        ["TOTAL", "20 checks", "ALL PASS"],
    ]

    table = ax.table(cellText=rows, colLabels=cols, loc="center",
                     cellLoc="center", colColours=["#34495E"]*3)
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.1, 1.8)

    # Style header
    for j in range(3):
        table[0, j].set_text_props(color="white", fontweight="bold")
        table[0, j].set_facecolor("#34495E")

    # Style last row (totals)
    for j in range(3):
        table[len(rows), j].set_facecolor("#E8F5E9")
        table[len(rows), j].set_text_props(fontweight="bold")

    # Alternate row colors
    for i in range(1, len(rows)):
        for j in range(3):
            if i < len(rows):
                table[i, j].set_facecolor("#F8F8F8" if i % 2 == 0 else "white")

    ax.set_title("Data Quality Validation Summary", fontsize=13, fontweight="bold", pad=20)

    fig.savefig(OUT / "validation_summary.png", dpi=200, bbox_inches="tight", pad_inches=0.4)
    plt.close(fig)
    print(f"  ✓ {OUT / 'validation_summary.png'}")


# ============================================================================
# Main
# ============================================================================
if __name__ == "__main__":
    print("Generating slide deck figures...")
    fig1_data_flow()
    fig2_data_model()
    fig3_medallion()
    fig4_workload_benchmark()
    fig5_scalability()
    fig6_supplementary()
    fig7_gold_tables()
    fig8_validation()
    print(f"\nAll figures saved to: {OUT}/")
