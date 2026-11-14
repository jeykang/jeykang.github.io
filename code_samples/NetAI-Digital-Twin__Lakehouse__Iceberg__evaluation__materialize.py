#!/usr/bin/env python3
"""materialize.py — serve a Gold selection in a validator's serving mode (호환 모드로 진열).

Gold is a selection of clips, not a file format. Each validator wants that selection
in its own shape, so the serving step is: take a selection, emit it in a *mode*:

  openloop   a clip list for run_eval.py (--clips-file). Needs only tracks + ego
             poses, so every clip qualifies.
  nurec      a directory of NuRec USDZ scenes that alpasim/run_scene.sh runs with
             LOCAL_USDZ_DIR= — NVIDIA mode. A clip qualifies if the NuRec catalog
             has an artifact for it (cached ones are hardlinked, missing ones
             downloaded with the gated HF token when --download is given). Clips
             without an artifact are listed as candidates for reconstruction.
  ncore      NCore v4 stores for reconstruction (NCore -> NuRec -> USDZ): each clip is
             staged in the pai-clip-dl layout (ncore/stage_pai_clip.py, which also
             fetches the four .offline features from HuggingFace) and converted with
             NVIDIA's own PAI converter vendored at ncore/repo (see ncore/README.md).
  hugsim     our HUGS twins of the selection for HUGSIM (no HD map, no catalog scene
             needed): a clip with a twin is served as its recorded-traffic scenario
             (hugsim/configs/pai/<short>_lidar-rec-02.yaml); a clip without one is listed
             with what it still needs, and --build makes the twins it can
             (hugsim/pai/twin_hugsim.sh, ~2.7 h per clip on the A10 given the clip's
             staged inputs and aux semantics). The fallback for clips NVIDIA has no
             NuRec scene for.

Selection: --clips-file, or --clips-from-parquet with --rank-col/--top-frac like
run_eval.py. Rollout triage (the dial): --triage <parquet with clip_id, p_fail> (e.g.
`skip.py predict` output) with --budget <fraction or count> keeps only the clips the
screen ranks most likely to fail; clips the screen has no score for are always kept
(nothing is skipped that the screen cannot judge). Every mode writes a manifest.json
beside its output naming what was served, what was skipped and why, so the serving step
is auditable.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time

import pyarrow.parquet as pq

HERE = os.path.dirname(os.path.abspath(__file__))
ALPASIM = os.path.join(HERE, "alpasim")
SIM_SCENES = os.path.join(ALPASIM, "repo", "data", "scenes", "sim_scenes.csv")
USDZ_CACHE = os.path.join(ALPASIM, "repo", "data", "nre-artifacts", "all-usdzs")
HF_REPO = "nvidia/PhysicalAI-Autonomous-Vehicles-NuRec"
NCORE_DIR = os.path.join(HERE, "ncore")
NCORE_REPO = os.path.join(NCORE_DIR, "repo")
NCORE_PY = os.path.join(NCORE_DIR, ".venv", "bin", "python")
HUGSIM = os.path.join(HERE, "hugsim")


# ----------------------------------------------------------------------------- selection
def select_clips(a) -> list[str]:
    if a.clips_file:
        return [l.strip() for l in open(a.clips_file) if l.strip()]
    t = pq.read_table(a.clips_from_parquet).to_pydict()
    ids = t["clip_id"]
    if a.rank_col:
        score = t[a.rank_col]
        order = sorted(range(len(ids)), key=lambda i: -(score[i] if score[i] is not None else -1e9))
        ids = [ids[i] for i in order]
    if a.top_frac:
        ids = ids[: max(1, int(round(a.top_frac * len(ids))))]
    if a.limit:
        ids = ids[: a.limit]
    return ids


def triage(clips: list[str], path: str, budget: float) -> tuple[list[str], dict]:
    """The dial: keep the top `budget` (fraction <= 1, or a count) of clips by p_fail."""
    t = pq.read_table(path, columns=["clip_id", "p_fail"]).to_pydict()
    p = dict(zip(t["clip_id"], t["p_fail"]))
    scored = sorted([c for c in clips if p.get(c) is not None], key=lambda c: -p[c])
    unscored = [c for c in clips if p.get(c) is None]
    k = int(round(budget * len(scored))) if budget <= 1 else min(len(scored), int(budget))
    keep = scored[:k] + unscored
    record = {"triage_file": os.path.abspath(path), "budget": budget, "scored": len(scored),
              "kept_by_screen": scored[:k], "skipped_by_screen": scored[k:], "kept_unscored": unscored}
    print(f"[triage] budget {budget}: {k} of {len(scored)} scored clips kept, {len(scored) - k} skipped; "
          f"{len(unscored)} unscored clips kept")
    return keep, record


# ----------------------------------------------------------------------------- nurec
def newest_artifacts() -> dict[str, dict]:
    best = {}
    for r in csv.DictReader(open(SIM_SCENES)):
        cid = r["scene_id"].replace("clipgt-", "")
        if cid not in best or r["last_modified"] > best[cid]["last_modified"]:
            best[cid] = r
    return best


def mode_nurec(clips, out_dir, download: bool):
    cat = newest_artifacts()
    os.makedirs(out_dir, exist_ok=True)
    served, missing_artifact, to_download = [], [], []
    for c in clips:
        r = cat.get(c)
        if r is None:
            missing_artifact.append(c); continue
        src = os.path.join(USDZ_CACHE, f"{r['uuid']}.usdz")
        dst = os.path.join(out_dir, f"{r['uuid']}.usdz")
        if os.path.exists(src):
            if not os.path.exists(dst):
                os.link(src, dst)           # hardlink: symlinks do not resolve inside the bind mount
            served.append({"clip_id": c, "uuid": r["uuid"], "hf_revision": r["hf_revision"], "from": "cache"})
        else:
            to_download.append((c, r))
    if to_download and download:
        from huggingface_hub import hf_hub_download
        tok = open(os.path.expanduser("~/.cache/huggingface/token")).read().strip()
        for c, r in to_download:
            p = hf_hub_download(HF_REPO, r["path"], repo_type="dataset", revision=r["hf_revision"],
                                token=tok, local_dir=os.path.join(out_dir, ".hf"))
            dst = os.path.join(out_dir, f"{r['uuid']}.usdz")
            os.link(p, dst) if not os.path.exists(dst) else None
            served.append({"clip_id": c, "uuid": r["uuid"], "hf_revision": r["hf_revision"], "from": "huggingface"})
        to_download = []
    manifest = {
        "mode": "nurec", "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "out_dir": os.path.abspath(out_dir),
        "no_artifact_fallback": "materialize.py hugsim (our HUGS twin: no HD map or catalog scene needed)",
        "run": f"LOCAL_USDZ_DIR={os.path.abspath(out_dir)} alpasim/run_scene.sh <policy-spec>",
        "served": served,
        "not_cached_pass_--download": [c for c, _ in to_download],
        "no_nurec_artifact_reconstruct_these": missing_artifact,
    }
    json.dump(manifest, open(os.path.join(out_dir, "manifest.json"), "w"), indent=1)
    print(f"[nurec] served {len(served)} scenes -> {out_dir}; "
          f"{len(to_download)} cached-miss (use --download); "
          f"{len(missing_artifact)} have no NuRec artifact (reconstruction candidates)")
    return manifest


# ----------------------------------------------------------------------------- openloop
def mode_openloop(clips, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    p = os.path.join(out_dir, "clips.txt")
    open(p, "w").write("\n".join(clips) + "\n")
    manifest = {"mode": "openloop", "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "run": f"run_eval.py --policy <spec> --clips-file {os.path.abspath(p)}",
                "served": clips}
    json.dump(manifest, open(os.path.join(out_dir, "manifest.json"), "w"), indent=1)
    print(f"[openloop] {len(clips)} clips -> {p}")
    return manifest


# ----------------------------------------------------------------------------- ncore
def mode_ncore(clips, out_dir, root):
    if not (os.path.isdir(NCORE_REPO) and os.path.exists(NCORE_PY)):
        sys.exit(f"NCore converter not set up: need {NCORE_REPO} and {NCORE_PY} (see ncore/README.md)")
    staged = os.path.join(out_dir, "staged")
    conv = os.path.join(out_dir, "ncore")
    os.makedirs(conv, exist_ok=True)
    served, failed = [], []
    for c in clips:
        st = subprocess.run([NCORE_PY, os.path.join(NCORE_DIR, "stage_pai_clip.py"), "--root", root,
                             "--clip", c, "--out", staged], capture_output=True, text=True)
        if st.returncode != 0:
            failed.append({"clip_id": c, "step": "stage", "rc": st.returncode, "log": (st.stdout + st.stderr)[-600:]})
            continue
        cv = subprocess.run([NCORE_PY, "-m", "tools.data_converter.pai.converter", "--root-dir", staged,
                             "--output-dir", conv, "pai-v4", "--clip-id", c],
                            cwd=NCORE_REPO, env={**os.environ, "PYTHONPATH": NCORE_REPO},
                            capture_output=True, text=True)
        if cv.returncode == 0:
            served.append({"clip_id": c, "ncore": os.path.join(conv, f"pai_{c}")})
        else:
            failed.append({"clip_id": c, "step": "convert", "rc": cv.returncode, "log": (cv.stdout + cv.stderr)[-600:]})
    manifest = {"mode": "ncore", "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "out_dir": os.path.abspath(out_dir),
                "next": "NuRec reconstruction (NVIDIA container, >24 GB VRAM) -> USDZ -> nurec mode",
                "served": served, "failed": failed}
    json.dump(manifest, open(os.path.join(out_dir, "manifest.json"), "w"), indent=1)
    print(f"[ncore] converted {len(served)}, failed {len(failed)} -> {conv}")
    return manifest


# ----------------------------------------------------------------------------- hugsim
def mode_hugsim(clips, out_dir, build: bool):
    def state(c):
        short = c[:8]
        scene = os.path.join(HUGSIM, "data", "scenes", "pai", f"{short}_lidar")
        scen = os.path.join(HUGSIM, "configs", "pai", f"{short}_lidar-rec-02.yaml")
        staged = os.path.isdir(os.path.join(NCORE_DIR, "staged", c))
        aux = os.path.exists(os.path.join(NCORE_DIR, "out", f"pai_{c}", f"pai_{c}.aux.sseg.zarr.itar"))
        return short, scene, scen, staged, aux

    if build:
        for c in clips:
            short, scene, scen, staged, aux = state(c)
            if not os.path.exists(os.path.join(scene, "scene.pth")) and staged and aux:
                print(f"[hugsim] building the twin of {short} (hugsim/pai/twin_hugsim.sh)", flush=True)
                subprocess.run([os.path.join(HUGSIM, "pai", "twin_hugsim.sh"), c])
    os.makedirs(os.path.join(out_dir, "scenarios"), exist_ok=True)
    served, buildable, needs_inputs = [], [], []
    for c in clips:
        short, scene, scen, staged, aux = state(c)
        if os.path.exists(os.path.join(scene, "scene.pth")) and os.path.exists(scen):
            dst = os.path.join(out_dir, "scenarios", os.path.basename(scen))
            if not os.path.exists(dst):
                try:
                    os.link(scen, dst)
                except OSError:                     # another filesystem
                    shutil.copy2(scen, dst)
            served.append({"clip_id": c, "scene": scene, "scenario": dst})
        elif staged and aux:
            buildable.append(c)
        else:
            needs_inputs.append({"clip_id": c, "staged": staged, "aux_semantics": aux})
    manifest = {
        "mode": "hugsim", "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "out_dir": os.path.abspath(out_dir),
        "run": {"vavam": f"CAMERA={HUGSIM}/configs/pai_camera_front100.yaml TAG=front100 {HUGSIM}/run_vavam.sh <scenario> <seed>",
                "constant_velocity": f"{HUGSIM}/run_closed_loop.sh <scenario> ltf {HUGSIM}/configs/pai_base_local_cv.yaml",
                "any HUGSIM client": f"{HUGSIM}/run_closed_loop.sh <scenario> <ad> {HUGSIM}/configs/pai_base_local.yaml"},
        "served": served,
        "build_with_--build_(twin_hugsim.sh)": buildable,
        "needs_staging_and_aux_semantics_first": needs_inputs,
    }
    json.dump(manifest, open(os.path.join(out_dir, "manifest.json"), "w"), indent=1)
    print(f"[hugsim] served {len(served)} twins -> {out_dir}/scenarios; {len(buildable)} buildable "
          f"(--build); {len(needs_inputs)} need staging + aux semantics first (ncore mode, then the NRE aux tool)")
    return manifest


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["openloop", "nurec", "ncore", "hugsim"])
    ap.add_argument("--clips-file")
    ap.add_argument("--clips-from-parquet", help="any parquet with a clip_id column, e.g. a score shard")
    ap.add_argument("--rank-col")
    ap.add_argument("--top-frac", type=float)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", required=True, help="output directory for the mode")
    ap.add_argument("--download", action="store_true", help="nurec: fetch uncached artifacts from HF")
    ap.add_argument("--build", action="store_true", help="hugsim: build the missing twins it can (hours each, A10)")
    ap.add_argument("--triage", help="rollout triage scores: parquet with clip_id, p_fail (skip.py predict)")
    ap.add_argument("--budget", type=float, help="with --triage: fraction (<=1) or count of scored clips to keep")
    ap.add_argument("--root", default=os.environ.get("AV_ROOT", "/mnt/netai-e2e/nvidia-physicalai-av-subset"))
    a = ap.parse_args()
    if not (a.clips_file or a.clips_from_parquet):
        sys.exit("give --clips-file or --clips-from-parquet")
    clips = select_clips(a)
    print(f"[materialize] {len(clips)} clips selected, mode={a.mode}")
    tri = None
    if a.triage:
        if a.budget is None:
            sys.exit("--triage needs --budget")
        clips, tri = triage(clips, a.triage, a.budget)
    m = {"openloop": lambda: mode_openloop(clips, a.out),
     "nurec": lambda: mode_nurec(clips, a.out, a.download),
     "ncore": lambda: mode_ncore(clips, a.out, a.root),
     "hugsim": lambda: mode_hugsim(clips, a.out, a.build)}[a.mode]()
    if tri:
        m["triage"] = tri
        json.dump(m, open(os.path.join(a.out, "manifest.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
