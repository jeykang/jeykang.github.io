#!/usr/bin/env python3
"""fig_scenario_space.py — the scenario x episode space as one figure (WS3).

    .skip_venv/bin/python fig_scenario_space.py      # -> figures/fig_scenario_space.{png,pdf}

Rows are serving modes (open-loop over recorded logs, open-loop over each Cosmos augmentation,
closed-loop over NVIDIA's NuRec scenes, over our NuRec twins, over our HUGS twins); columns are
the recording condition from the dataset's hour_of_day (episodes.condition_of). A cell shows
its episode count and clips; closed-loop cells also show how many of those clips have at least
one rollout in eval.rollout (evaluation/rollouts.py). Counts come from the same parquet files
nvidia_ingestion/build_episode_tables.py lands, with the same de-duplication, so they match
nvidia_gold.scenario.
"""
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, LogNorm

HERE = os.path.dirname(os.path.abspath(__file__))
UD = os.path.join(HERE, "..", "user_data")
BLUE = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
        "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]       # sequential ramp 100..700
INK, MUTED, EMPTY, SURFACE = "#1f1f1e", "#5f5e5a", "#f0efec", "#ffffff"


def _lum(rgb):
    c = [x / 12.92 if x <= 0.03928 else ((x + 0.055) / 1.055) ** 2.4 for x in rgb[:3]]
    return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]


def _contrast(a, b):
    la, lb = sorted((_lum(a), _lum(b)), reverse=True)
    return (la + 0.05) / (lb + 0.05)


def text_colors(fc):
    """(value ink, sub ink) with the higher contrast against the cell fill."""
    white, ink = (1, 1, 1), matplotlib.colors.to_rgb(INK)
    if _contrast(fc, white) >= _contrast(fc, ink):
        return "#ffffff", "#ffffff"
    # muted ink only where it still clears 4.5:1 on this fill; mid-tone fills keep full ink
    return INK, (MUTED if _contrast(fc, matplotlib.colors.to_rgb(MUTED)) >= 4.5 else INK)

parts = []
for f in sorted(glob.glob(os.path.join(UD, "episodes_*.parquet"))):
    d = pd.read_parquet(f).rename(columns={"validator_mode": "serving_mode", "n_agents": "n_actors"})
    parts.append(d)
ep = pd.concat(parts, ignore_index=True).drop_duplicates("episode_id")
ro = pd.read_parquet(os.path.join(UD, "rollouts.parquet"))
cond_of = ep.drop_duplicates("clip_id").set_index("clip_id")["condition"]
ro["condition"] = ro["clip_id"].map(cond_of)

rows = [("openloop-mfpdms", "none", "open-loop, recorded logs"),
        ("augmented-openloop", "cosmos-fog", "open-loop, Cosmos fog"),
        ("augmented-openloop", "cosmos-night", "open-loop, Cosmos night"),
        ("augmented-openloop", "cosmos-rain", "open-loop, Cosmos rain"),
        ("closedloop-nurec", "none", "closed-loop, NVIDIA's NuRec scene"),
        ("closedloop-nurec-ours", "none", "closed-loop, our NuRec twin"),
        ("closedloop-hugsim", "none", "closed-loop, our HUGS twin")]
cols = [("day", "day"), ("dawn_dusk", "dawn / dusk"), ("night", "night"), ("unknown", "unknown")]

n_ep = np.zeros((len(rows), len(cols)), int); n_clip = np.zeros_like(n_ep); n_roll = np.full_like(n_ep, -1)
for i, (mode, aug, _) in enumerate(rows):
    for j, (c, _) in enumerate(cols):
        sel = ep[(ep.serving_mode == mode) & (ep.augmentation == aug) & (ep.condition == c)]
        n_ep[i, j], n_clip[i, j] = len(sel), sel.clip_id.nunique()
        if mode.startswith("closedloop"):
            n_roll[i, j] = ro[(ro.serving_mode == mode) & (ro.condition == c)].clip_id.nunique()
assert n_ep.sum() == len(ep), (n_ep.sum(), len(ep))

cmap = LinearSegmentedColormap.from_list("blue", BLUE)
norm = LogNorm(vmin=1, vmax=max(2, n_ep.max()))
fig, ax = plt.subplots(figsize=(8.6, 4.9), dpi=200)
gap = 0.04                                                      # surface gap between cells
for i in range(len(rows)):
    for j in range(len(cols)):
        v = n_ep[i, j]
        fc = cmap(norm(v)) if v else EMPTY
        ax.add_patch(plt.Rectangle((j + gap, i + gap), 1 - 2 * gap, 1 - 2 * gap, facecolor=fc, edgecolor="none"))
        if not v:
            ax.text(j + 0.5, i + 0.5, "none", ha="center", va="center", fontsize=8, color=MUTED)
            continue
        tc, sc = text_colors(fc)
        unit = "scene" if rows[i][0].startswith("closedloop") else "clip"
        ax.text(j + 0.5, i + 0.38, f"{v:,}", ha="center", va="center", fontsize=10.5, fontweight="bold", color=tc)
        sub = f"{n_clip[i, j]:,} {unit}{'s' if n_clip[i, j] != 1 else ''}"
        if n_roll[i, j] >= 0:
            sub += f" · {n_roll[i, j]} rolled out"
        ax.text(j + 0.5, i + 0.68, sub, ha="center", va="center", fontsize=7.2, color=sc)
ax.set_xlim(0, len(cols)); ax.set_ylim(len(rows), 0)
ax.set_xticks(np.arange(len(cols)) + 0.5, [c[1] for c in cols], fontsize=9, color=INK)
ax.set_yticks(np.arange(len(rows)) + 0.5, [r[2] for r in rows], fontsize=9, color=INK)
ax.xaxis.tick_top(); ax.tick_params(length=0, pad=4)
for s in ax.spines.values():
    s.set_visible(False)
ax.axhline(4, color=MUTED, lw=0.8, xmin=-0.62, clip_on=False)   # open-loop | closed-loop
n_cls = int(((n_ep > 0)).sum())
fig.suptitle(f"The scenario × episode space: {len(ep):,} episodes in {n_cls} scenario classes",
             x=0.02, ha="left", fontsize=11.5, fontweight="bold", color=INK)
fig.text(0.02, 0.875, "Cell: episodes, then the clips they come from; closed-loop cells also count the clips with at least\n"
         "one rollout so far. Shade: episodes, log scale. Recording condition from the dataset's hour of day.",
         fontsize=7.8, color=MUTED, ha="left", linespacing=1.4)
fig.subplots_adjust(left=0.30, right=0.985, top=0.77, bottom=0.03)
os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(HERE, "figures", f"fig_scenario_space.{ext}"), facecolor=SURFACE)
print("episodes", len(ep), "classes", n_cls, "rollout clips", int(n_roll[n_roll > 0].sum()))
