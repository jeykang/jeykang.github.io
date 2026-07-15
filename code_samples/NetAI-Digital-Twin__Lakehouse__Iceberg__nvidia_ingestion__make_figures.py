"""Generate presentation figures for the June 2026 progress meeting.
All values are from this month's analyses (see MEETING_FACTSHEET_2026-06.md).
Run inside the spark-iceberg container (has matplotlib). Outputs to ./figures/.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 150})
GREEN, GRAY, RED, DARK, LIGHT = "#2a9d8f", "#9aa0a6", "#e76f51", "#264653", "#a8c5d6"


def col(auc):
    return RED if auc < 0.5 else (GREEN if auc >= 0.55 else GRAY)


# ---- Figure 1: validity battery per-signal AUC ----
sig = [("sensor_coverage", 0.432, False), ("composite — OLD", 0.450, True),
       ("time_of_day", 0.477, False), ("ego_dynamics", 0.498, False),
       ("season_geography", 0.507, False), ("perception", 0.564, False),
       ("conflict (behavioral)", 0.651, False), ("composite — NEW", 0.655, True)]
fig, ax = plt.subplots(figsize=(8.2, 4.6))
names = [s[0] for s in sig]; vals = [s[1] for s in sig]
bars = ax.barh(names, vals, color=[col(v) for v in vals],
               edgecolor=["black" if s[2] else "none" for s in sig],
               linewidth=[1.6 if s[2] else 0 for s in sig])
ax.axvline(0.5, ls="--", color="black", lw=1); ax.text(0.5, 7.7, " chance", fontsize=9)
for b, v in zip(bars, vals):
    ax.text(v + 0.004, b.get_y() + b.get_height()/2, f"{v:.3f}", va="center", fontsize=9.5)
ax.set_xlim(0.40, 0.70); ax.set_xlabel("OOD AUC  (vs 1,740 human-flagged hard clips)")
fig.suptitle("Which difficulty signals actually track human-judged hard clips",
             fontweight="bold", fontsize=12.5, y=1.04)
ax.set_title("0.5 = random · <0.5 = backwards · metadata heuristics fail; agent-conflict is the one that works",
             fontsize=9, color="#555", pad=8)
plt.tight_layout(); plt.savefig(f"{OUT}/fig1_validity_auc.png", bbox_inches="tight"); plt.close()


# ---- Figure 2: per-cluster conflict AUC ----
clu = [("PEDESTRIAN_DENSITY", 0.866, 52), ("SPECIAL_VEHICLE", 0.654, 34),
       ("ROAD_DEBRIS", 0.646, 1), ("CYCLISTS", 0.621, 9), ("EMERGENCY_INCIDENT", 0.588, 3),
       ("WORK_ZONES", 0.561, 88), ("ANIMALS", 0.541, 4), ("COMPLEX_INTERSECTION", 0.499, 5),
       ("OTHER_LONGTAIL", 0.212, 4)]
clu = clu[::-1]
fig, ax = plt.subplots(figsize=(8.2, 4.8))
names = [c[0] for c in clu]; vals = [c[1] for c in clu]; ns = [c[2] for c in clu]
bars = ax.barh(names, vals, color=[col(v) for v in vals])
ax.axvline(0.5, ls="--", color="black", lw=1)
for b, v, n in zip(bars, vals, ns):
    b.set_alpha(1.0 if n >= 20 else 0.45)
    ax.text(v + 0.008, b.get_y() + b.get_height()/2, f"{v:.3f}  (n={n})", va="center", fontsize=9.5)
ax.set_xlim(0.0, 1.0); ax.set_xlabel("agent-conflict OOD AUC")
fig.suptitle("Agent-conflict validates on human hard-event categories",
             fontweight="bold", fontsize=12.5, y=1.04)
ax.set_title("Per event cluster · bars faded where n<20 (indicative) · pedestrian-density & work-zones are the reliable ones",
             fontsize=8.5, color="#555", pad=8)
plt.tight_layout(); plt.savefig(f"{OUT}/fig2_conflict_by_cluster.png", bbox_inches="tight"); plt.close()


# ---- Figure 3: darkness degrades perception ----
fig, axes = plt.subplots(1, 2, figsize=(8.2, 4.2))
for ax, (title, day, dark, unit, pct) in zip(axes, [
        ("Detection confidence", 0.505, 0.456, "", "−10%"),
        ("Detections / frame", 11.53, 8.72, "", "−24%")]):
    b = ax.bar(["Day", "Dark"], [day, dark], color=[LIGHT, DARK], width=0.6)
    for bi, v in zip(b, [day, dark]):
        ax.text(bi.get_x()+bi.get_width()/2, v, f"{v:.2f}" if v < 5 else f"{v:.1f}",
                ha="center", va="bottom", fontsize=10)
    ax.set_title(title, fontsize=11.5)
    ax.annotate(pct, xy=(1, dark), xytext=(0.5, max(day, dark)*0.55),
                fontsize=13, fontweight="bold", color=RED, ha="center")
fig.suptitle("Low light measurably degrades perception  (n = 3,334 clips)",
             fontweight="bold", fontsize=12.5, y=1.02)
plt.tight_layout(); plt.savefig(f"{OUT}/fig3_darkness_degrades_perception.png", bbox_inches="tight"); plt.close()


# ---- Figure 4: inversion fix ----
fig, ax = plt.subplots(figsize=(6.2, 4.4))
labels = ["conflict-only\n(behavioral)", "union composite\n(final)"]
vals = [-0.14, 0.61]
b = ax.bar(labels, vals, color=[RED, GREEN], width=0.55)
ax.axhline(0, color="black", lw=1)
for bi, v in zip(b, vals):
    ax.text(bi.get_x()+bi.get_width()/2, v + (0.03 if v > 0 else -0.05),
            f"{v:+.2f}", ha="center", va="bottom" if v > 0 else "top", fontsize=12, fontweight="bold")
ax.set_ylim(-0.3, 0.75); ax.set_ylabel("rank-correlation of score with darkness")
fig.suptitle("Dark clips: from discarded to kept", fontweight="bold", fontsize=12.5, y=1.04)
ax.set_title("Negative = perceptually-hard dark clips ranked low (stripped); positive = kept",
             fontsize=9, color="#555", pad=8)
plt.tight_layout(); plt.savefig(f"{OUT}/fig4_inversion_fix.png", bbox_inches="tight"); plt.close()


# ---- Figure 5 (optional): Gold composition before/after rank-norm ----
import numpy as np
cats = ["dark", "high-conflict", "perceptual-rescued"]
raw = [89, 38, 70]; norm = [78, 70, 46]
x = np.arange(len(cats)); w = 0.38
fig, ax = plt.subplots(figsize=(7.0, 4.4))
b1 = ax.bar(x - w/2, raw, w, label="raw darkness (saturated)", color=GRAY)
b2 = ax.bar(x + w/2, norm, w, label="rank-normalized (final)", color=GREEN)
for bs in (b1, b2):
    for bi in bs:
        ax.text(bi.get_x()+bi.get_width()/2, bi.get_height()+1, f"{int(bi.get_height())}%",
                ha="center", fontsize=9.5)
ax.set_xticks(x); ax.set_xticklabels(cats); ax.set_ylabel("% of Gold clips"); ax.set_ylim(0, 100)
ax.legend(frameon=False, fontsize=9.5)
ax.set_title("Both axes contribute after balancing  (Gold = 3,176 clips)",
             fontweight="bold", fontsize=12)
plt.tight_layout(); plt.savefig(f"{OUT}/fig5_gold_composition.png", bbox_inches="tight"); plt.close()

print("WROTE:", sorted(os.listdir(OUT)))
