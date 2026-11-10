#!/usr/bin/env python3
"""storage_sizing.py — size accumulation (축적) and serving (진열) storage separately.

The lakehouse registers raw data in place, so Bronze/Silver/Gold cost no storage of
their own; what the serving tier really holds is the *derived* artifacts a validator
needs — a NuRec twin per Gold scene, augmented variants, rollout outputs. This model
separates the two tiers, sizes each from measured constants, and puts a number on the
store-vs-regenerate question for augmented variants: below a break-even re-curation
interval it is cheaper to keep variants; above it, to regenerate them.

Every constant is either measured in this repo (cited) or an explicit assumption
(flagged). Run it to print the tables and write STORAGE_SIZING.md next to it; use
the CLI to move the assumptions.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import statistics

HERE = os.path.dirname(os.path.abspath(__file__))
EVAL = os.path.join(HERE, "..", "evaluation")

# ----------------------------------------------------------------------------- measured constants
MEASURED = {
    # accumulation tier
    "raw_full_sensor_clip_mb": 600,     # camera 235 + lidar 356 + radar 8 + labels 0.4 MB, clip 2daf9698 (NFS, 2026-09-21)
    "raw_on_disk_mean_mb": 409,         # 13.5 TB / 32,986 clips on disk (MEDALLION_PROGRESS.md); below the 600 MB full-sensor clip although 31,812 on-disk clips do have LiDAR (gap not attributed)
    "corpus_clips": 306_152,            # clip_index.parquet
    "on_disk_clips": 32_986,            # MEDALLION_PROGRESS.md
    "gold_clips": 3_176,                # MEDALLION_PROGRESS.md (noisy-OR union, sensor-covered)
    # serving tier
    "nurec_scene_gb_local": 1.60,       # 65 GB / 41 cached 26.04 artifacts (alpasim/repo/data/nre-artifacts)
    "nurec_scene_gb_catalog": 1.79,     # FEASIBILITY.md, 26.04 release average
    "variant_single_cam_window_mb": 8,  # 121-frame (4 s) single-camera Cosmos output; 39 MB per 20 s camera clip x 0.2 (estimate)
    "variant_full_clip_6cam_mb": 235,   # one condition rendered for all 6 cameras, full clip = the camera set size
    "rollout_output_mb": 16,            # 474 MB / 30 scenes, alpasim/runs/vavam-batch2 (videos + metrics)
    # regeneration
    "cosmos_window_gpu_min": 33.6,      # 7 h / 50 windows on 4 x A100-40GB = 8.4 wall-min x 4 GPUs (E-B, git 167a3f4)
    "cosmos_keep_rate": 0.86,           # 43 / 50 passed the hallucination gate (E-B)
    "openloop_s_per_clip": 0.46,        # track-only, 8 workers (BENCHMARKS.md)
    "openloop_sut_s_per_clip": 11.6,    # VaVAM's own open-loop pass, 1 worker on one GPU (evaluation/.results_nurec_vavam.runmeta.json)
    "closedloop_s_per_scene": 38,       # VaVAM, 40 scenes in 25 min wall incl. downloads (ALPASIM.md batch 3)
    # the two twin kinds (per clip, on the A10; hours are read from the pipelines' timing files when present)
    "hugs_scene_gb": 1.0,               # LiDAR-seeded HUGSIM export of ac73935a (HUGSIM.md)
    "nurec_psnr_db": 29.5,              # median held-out PSNR of the nine NuRec twins (evaluation/nurec/README.md)
    "hugs_psnr_db": 26.4,               # median held-out PSNR of the nine HUGS twins, 24.2-28.8 dB (hugsim/data/models/*/results.json)
    "aux_semantics_a10_h": 2.9,         # NRE aux store (Mask2Former on 6 cameras + LiDAR seg), median of the twin queue; the HUGS twin reuses its labels
    # Cosmos-Transfer2.5 on the DGX Spark (cosmos_augmentation/FINDINGS.md)
    "cosmos25_window_mb": 4.0,          # 4 s single-camera window, 1080p/30 fps
    "cosmos25_gb10_min_per_video_s": 22.5,  # 90 min wall for 4 s of video, one condition
}


def twin_hours() -> dict:
    """Per-clip A10 hours of each twin kind, from the pipelines' own timing files.

    NuRec: nurec/out/<short>.twin.json, time to the AlpaSim bundle, clips whose aux store was
    built in the same run (one resumed run excluded). HUGS: hugsim/runs/twin_queue/<short>.json,
    time to the export, clips that went through every step in one run.
    """
    nurec, hugs = [], []
    for f in glob.glob(os.path.join(EVAL, "nurec", "out", "*.twin.json")):
        st = json.load(open(f))["steps"]
        if st.get("aux", {}).get("t_s", 0) > 3600 and "bundle" in st:
            nurec.append(st["bundle"]["t_s"] / 3600)
    for f in glob.glob(os.path.join(EVAL, "hugsim", "runs", "twin_queue", "*.json")):
        st = json.load(open(f)).get("steps", {})
        if st.get("convert", {}).get("step_s", 0) > 0 and st.get("train", {}).get("status") == "ok":
            hugs.append(st["train"]["t_s"] / 3600)
    return {"nurec_h": statistics.median(nurec) if nurec else 5.9, "nurec_n": len(nurec),
            "hugs_h": statistics.median(hugs) if hugs else 2.7, "hugs_n": len(hugs)}


def fmt_tb(gb): return f"{gb/1000:.2f} TB" if gb >= 1000 else f"{gb:.0f} GB"   # decimal units (1 TB = 1,000 GB) throughout


def model(a):
    m = MEASURED
    out = []
    P = out.append

    # ---------------------------------------------------------------- accumulation tier
    P("## Accumulation tier (축적)\n")
    P("Raw logs as recorded. This tier is sized by the fleet, not by curation; the "
      "lakehouse only registers it in place.\n")
    fleet_tb_yr = a.cars * a.hours_per_car_day * 365 * a.clips_per_hour * m["raw_full_sensor_clip_mb"] / 1e6
    P("| quantity | value | basis |\n|---|---|---|")
    P(f"| full-sensor clip | {m['raw_full_sensor_clip_mb']} MB | measured, one clip on NFS |")
    P(f"| on-disk mean clip | {m['raw_on_disk_mean_mb']} MB | 13.5 TB / 32,986 clips |")
    P(f"| NVIDIA corpus, full-sensor | {fmt_tb(m['corpus_clips']*m['raw_full_sensor_clip_mb'] / 1000)} | 306,152 clips x 600 MB |")
    P(f"| project fleet, raw per year | {fleet_tb_yr:.0f} TB | {a.cars} cars x {a.hours_per_car_day} h/day x {a.clips_per_hour} clips/h x 600 MB (assumption) |")
    P(f"| retained after redundancy cull | {fleet_tb_yr*a.retain:.0f} TB/yr | retain {a.retain:.0%} (the professor's 100-to-10 rule; our on-disk to Gold is {m['gold_clips']/m['on_disk_clips']:.1%}) |")
    P("")

    # ---------------------------------------------------------------- serving tier
    P("## Serving tier (진열)\n")
    P("Derived artifacts per Gold clip, for a Gold tier of N clips. Bronze/Silver/Gold views "
      "themselves are metadata only (register-in-place).\n")
    P("| Gold clips N | twin (NuRec) | + variants stored, v cond x 6 cam | + rollout outputs, p policies | total (store) | total (regenerate) |\n|---|---|---|---|---|---|")
    for N in a.gold_sizes:
        twin = N * m["nurec_scene_gb_catalog"]
        var = N * a.variants * m["variant_full_clip_6cam_mb"] / 1000
        roll = N * a.policies * (1 + a.variants) * m["rollout_output_mb"] / 1000
        P(f"| {N:,} | {fmt_tb(twin)} | {fmt_tb(var)} | {fmt_tb(roll)} | {fmt_tb(twin+var+roll)} | {fmt_tb(twin+roll)} |")
    P("")
    P(f"Assumptions: v = {a.variants} augmentation conditions per Gold clip, p = {a.policies} policies "
      f"evaluated per re-curation, twin = {m['nurec_scene_gb_catalog']} GB/scene (catalog mean; local mean "
      f"{m['nurec_scene_gb_local']} GB), variant = {m['variant_full_clip_6cam_mb']} MB per condition for all six "
      f"cameras over the whole clip (the camera set size; the E-B batch rendered 4 s single-camera windows of "
      f"about {m['variant_single_cam_window_mb']} MB).\n")

    # ---------------------------------------------------------------- store vs regenerate
    P("## Store or regenerate the variants?\n")
    gpu_h_per_clip = m["cosmos_window_gpu_min"] / 60 * (20 / 4) * 6 / m["cosmos_keep_rate"]   # full clip, 6 cams, gate loss
    P(f"Regenerating one condition for one Gold clip (20 s, 6 cameras) costs about "
      f"**{gpu_h_per_clip:.1f} A100-GPU-hours** (E-B: {m['cosmos_window_gpu_min']} GPU-min per 4 s single-camera "
      f"window, x5 duration x6 cameras, / {m['cosmos_keep_rate']:.0%} gate keep-rate). Storing it costs "
      f"{m['variant_full_clip_6cam_mb']} MB.\n")
    store_usd_yr = m["variant_full_clip_6cam_mb"] / 1e6 * a.tb_month_usd * 12
    regen_usd = gpu_h_per_clip * a.gpu_hour_usd
    breakeven_yr = store_usd_yr / regen_usd if regen_usd else float("inf")
    P("| | per clip-condition | basis |\n|---|---|---|")
    P(f"| store, per year | ${store_usd_yr:.4f} | {m['variant_full_clip_6cam_mb']} MB x ${a.tb_month_usd}/TB-month (assumption) |")
    P(f"| regenerate, once | ${regen_usd:.2f} | {gpu_h_per_clip:.1f} GPU-h x ${a.gpu_hour_usd}/GPU-h (assumption) |")
    P(f"| break-even storage horizon | {1/breakeven_yr:.0f} years | storing beats regenerating unless a variant is kept that long unused |")
    P("")
    P(f"At these prices a stored variant pays for its regeneration after {1/breakeven_yr:.0f} years, so "
      f"**store every variant that was rendered**; the tension the professor describes is not about the "
      f"variants that exist but about the ones that do not: the space of harder situations (a pedestrian "
      f"stepping out, a different weather) is open-ended, so it cannot be pre-rendered, and what must be sized "
      f"is the GPU budget per re-curation, not the disk.\n")
    P("| re-curation interval | new Gold fraction per cycle (turnover) | GPU-h per cycle for v conditions, N Gold | GPU-h per year |\n|---|---|---|---|")
    for months in a.recuration_months:
        for turnover in (0.1, 0.3):
            for N in (a.gold_sizes[0], a.gold_sizes[-1]):
                per_cycle = N * turnover * a.variants * gpu_h_per_clip
                P(f"| {months} mo | {turnover:.0%} | N={N:,}: {per_cycle:,.0f} | {per_cycle*12/months:,.0f} |")
    P("")
    P("Reading: with Gold shifting every 6 months and 30% of it new each time (his 6-month drift), "
      f"a {a.gold_sizes[-1]:,}-clip Gold with {a.variants} conditions needs roughly "
      f"{a.gold_sizes[-1]*0.3*a.variants*gpu_h_per_clip*2:,.0f} A100-GPU-hours a year of Cosmos-Transfer1 "
      f"generation, which is the number to negotiate for, against a serving disk of "
      f"{fmt_tb(a.gold_sizes[-1]*(m['nurec_scene_gb_catalog'] + a.variants*m['variant_full_clip_6cam_mb'] / 1000))}.\n")

    # ---------------------------------------------------------------- twin reconstruction
    P("## The twin itself\n")
    th = twin_hours()
    P("Two twin kinds now exist (`evaluation/EPISODES.md`, serving modes `closedloop-nurec-ours` and "
      "`closedloop-hugsim`), with measured per-clip costs on the A10 (23 GB; an L40S would be faster):\n")
    P("| twin | A10 hours per clip | storage per scene | held-out PSNR | basis |\n|---|---|---|---|---|")
    P(f"| NuRec (NRE prod config + aux store) | {th['nurec_h']:.1f} | {m['nurec_scene_gb_catalog']} GB | {m['nurec_psnr_db']} dB (median) "
      f"| median of {th['nurec_n']} uninterrupted runs of `nurec/twin_pipeline.sh` (aux ~2.9 h + training ~2.8 h) |")
    hb = (f"median of {th['hugs_n']} runs of `hugsim/pai/twin_hugsim.sh`" if th["hugs_n"]
          else "ac73935a: preprocessing ~0.3 h + ground 26 min + scene 1 h 56 min")
    hugs_fresh = th["hugs_h"] + m["aux_semantics_a10_h"]
    P(f"| HUGS (LiDAR-seeded), semantics already built | {th['hugs_h']:.1f} | {m['hugs_scene_gb']} GB | {m['hugs_psnr_db']} dB (median of nine) | {hb} |")
    P(f"| HUGS (LiDAR-seeded), fresh clip | {hugs_fresh:.1f} | {m['hugs_scene_gb']} GB | — | + the NRE aux store for its semantic labels "
      f"(~{m['aux_semantics_a10_h']} h; HUGSIM's own InverseForm path is unmeasured) |")
    P("")
    P("| Gold clips N | NuRec: storage | NuRec: A10-years | HUGS: storage | HUGS, fresh clips: A10-years |\n|---|---|---|---|---|")
    for N in a.gold_sizes:
        P(f"| {N:,} | {fmt_tb(N*m['nurec_scene_gb_catalog'])} | {N*th['nurec_h']/8766:.2f} | "
          f"{fmt_tb(N*m['hugs_scene_gb'])} | {N*hugs_fresh/8766:.2f} |")
    P("")
    P(f"The twin, not the variants, dominates both serving storage and GPU time, and it is what re-curation "
      f"churns: every clip that enters Gold needs a reconstruction, every clip that leaves holds its scene until "
      f"evicted. At {m['gold_clips']:,} Gold clips a NuRec twin of everything is "
      f"{fmt_tb(m['gold_clips']*m['nurec_scene_gb_catalog'])} and {m['gold_clips']*th['nurec_h']/8766:.1f} A10-years. "
      f"The HUGS twin needs about half the storage, at ~3 dB lower fidelity. Its GPU time is "
      f"{th['hugs_h']/th['nurec_h']:.0%} of NuRec's where the clip's semantic labels already exist, and it varies far "
      f"more from clip to clip (1.8 to 8.2 h over eight clips; the slowest scenes trained at about a third of the usual "
      f"iteration rate, 1.35 it/s on 44c3b4d5); where the NRE aux tool has to make the labels it costs about as much as NuRec ({hugs_fresh:.1f} h). "
      f"Its case is coverage, not cost: it needs no HD map and no NVIDIA scene. With {a.recuration_months[0]}-month "
      f"re-curation and 30 % turnover, keeping a NuRec twin of every Gold clip costs "
      f"{m['gold_clips']*0.3*th['nurec_h']*12/a.recuration_months[0]:,.0f} A10-hours a year. Which clips earn a "
      f"twin, and of which kind, is therefore a triage decision of the same shape as which clips earn a rollout.\n")

    # ---------------------------------------------------------------- validation cost
    P("## Validation cost per re-curation\n")
    P("| step | per clip | N = 3,176 Gold, one policy | basis |\n|---|---|---|---|")
    G = m["gold_clips"]
    P(f"| open-loop reference ladder (five track-only policies) | {m['openloop_s_per_clip']} s | {G*m['openloop_s_per_clip']/3600:.1f} h CPU | BENCHMARKS.md, 8 workers |")
    P(f"| open-loop pass of the policy under test (VaVAM) | {m['openloop_sut_s_per_clip']} s | {G*m['openloop_sut_s_per_clip']/3600:.1f} GPU-h | `.results_nurec_vavam.runmeta.json`, 1 worker |")
    P(f"| closed-loop rollout (VaVAM) | {m['closedloop_s_per_scene']} s | {G*m['closedloop_s_per_scene']/3600:.1f} GPU-h | ALPASIM.md batch 3 |")
    P(f"| closed-loop at a {a.budget:.0%} triage budget | — | {G*a.budget*m['closedloop_s_per_scene']/3600:.1f} GPU-h | evaluation/skip.py |")
    P("")
    full = G * m["closedloop_s_per_scene"] / 3600
    sunk = full * (1 - a.budget)
    paid = full - (G * m["openloop_sut_s_per_clip"] / 3600 + G * a.budget * m["closedloop_s_per_scene"] / 3600)
    P(f"The screen's signal comes from the policy's own open-loop pass, not from the track-only ladder "
      f"(`evaluation/SKIP.md`), so what a {a.budget:.0%} budget saves depends on whether that pass is run anyway. "
      f"If the proving ground scores every curated clip open-loop regardless (its first rung), the screen is free "
      f"and the budget saves **{sunk:.1f} of {full:.1f} GPU-hours** per policy per re-curation. If the open-loop pass "
      f"is run only to triage, it costs {G*m['openloop_sut_s_per_clip']/3600:.1f} GPU-h and the saving shrinks to "
      f"**{paid:.1f} GPU-hours ({paid/full:.0%})**.\n")

    # ---------------------------------------------------------------- measured since 2026-09-21
    P("## Measured constants since the model was written\n")
    P("| constant | modelled before | measured | where |\n|---|---|---|---|")
    P(f"| NuRec twin, per clip | L40S number missing; regeneration ~19.5 A100-h per clip-condition (all steps) | "
      f"{th['nurec_h']:.1f} h on one A10 end to end (aux ~2.9 h, training ~2.8 h; ac73935a's v3 training alone "
      f"2 h 05 m at 4.0–4.4 it/s plus ~10 min validation/export) | `evaluation/nurec/README.md` |")
    P(f"| HUGS twin, per clip | — | {th['hugs_h']:.1f} h on one A10 given the clip's aux semantics "
      f"({th['hugs_h'] + m['aux_semantics_a10_h']:.1f} h with them), {m['hugs_scene_gb']} GB export | `evaluation/HUGSIM.md` |")
    P(f"| Cosmos variant, single-camera 4 s window | 235 MB per condition for six cameras over a whole clip (assumption kept) | "
      f"{m['cosmos25_window_mb']} MB at 1080p/30 fps (Transfer2.5 encode), 90 min wall on the DGX Spark's GB10 ≈ "
      f"{m['cosmos25_gb10_min_per_video_s']:.0f} GB10-min per second of video per condition | `cosmos_augmentation/FINDINGS.md` |")
    P(f"| policy under test, open-loop | ~0.5 s per clip assumed for every rung of the screen | {m['openloop_sut_s_per_clip']} s "
      f"per clip (VaVAM, one GPU) | `evaluation/.results_nurec_vavam.runmeta.json` |")
    P("")
    P("The A10 figures are lower bounds on an L40S; the Spark figure is ~10× the wall time of one 4×A100 node per "
      "second of video, on hardware that queues nothing. None of them changes the store-vs-regenerate verdict "
      "(break-even stays in the hundreds of years).\n")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cars", type=int, default=21, help="project fleet size (assumption)")
    ap.add_argument("--hours-per-car-day", type=float, default=8)
    ap.add_argument("--clips-per-hour", type=float, default=180, help="20 s clips per recording hour")
    ap.add_argument("--retain", type=float, default=0.10, help="fraction kept after redundancy cull")
    ap.add_argument("--gold-sizes", type=lambda s: [int(x) for x in s.split(",")], default=[500, 1000, 3176, 10000])
    ap.add_argument("--variants", type=int, default=3, help="augmentation conditions per Gold clip (night, rain, fog)")
    ap.add_argument("--policies", type=int, default=3)
    ap.add_argument("--recuration-months", type=lambda s: [int(x) for x in s.split(",")], default=[6, 12])
    ap.add_argument("--tb-month-usd", type=float, default=20.0, help="storage price assumption")
    ap.add_argument("--gpu-hour-usd", type=float, default=2.0, help="A100 price assumption")
    ap.add_argument("--budget", type=float, default=0.5, help="rollout-triage budget (fraction of clips rolled out)")
    ap.add_argument("--out", default=os.path.join(HERE, "STORAGE_SIZING.md"))
    a = ap.parse_args()

    body = model(a)
    head = ("# Storage sizing — accumulation vs serving (2026-09-21, regenerated 2026-10-06)\n\n"
            "Generated by `storage_sizing.py`; rerun with different assumptions rather than editing.\n\n")
    open(a.out, "w").write(head + body)
    print(body)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
