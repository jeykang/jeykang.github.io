# EAD-Data Lakehouse — Hallway Wall Display

> EAD = **Evolutionary Autonomous Driving**.

An auto-looping, full-screen **data-flow** visualization for a wall-mounted
monitor. The story is told at the **dataset** level — a dataset is uploaded,
ingested, curated ("trim the fat"), scored, and lands in a catalog beside other
processed datasets — in four stages, left → right:

```
  01 Ingest  →  02 Curate  →  03 GPU Perception  →  04 Serve
  dataset       trim the fat     BEVFusion + YOLO     dataset catalog
  uploaded      13 TB → 16 GB    on 2× GPU            (PhysicalAI · KAIST ·
  → NFS         Bronze/Silver/Gold                     nuScenes · Cosmos)
```

The lakehouse is genuinely multi-dataset: **NVIDIA PhysicalAI** is the fully
processed flagship (all real figures); **KAIST E2E**, **nuScenes**, and **NVIDIA
Cosmos** appear in the Serve-stage catalog with honest status badges (Active /
Ingesting / Benchmark / Synthetic). Per-clip LiDAR + camera + detections still
appear in the GPU stage as a representative *sample* of the active dataset. Edit
the `catalog` / `active_dataset` / `curate` blocks in `js/data.js` to adjust.

It is built to run **unattended and permanently**:

- **Pure HTML / CSS / Canvas** — no React, no WebGL libs, no CDN, no backend.
  Nothing to break on the wall; works with no internet.
- **Manifest-driven, so it doesn't look like a fixed loop.** If a real-clip
  asset library exists (`assets/clips/manifest.json`), every loop streams a
  *different* real clip through the pipeline — new dashcam frame, new decoded
  LiDAR sweep, new metadata, new difficulty score. Counters tick continuously
  as if data is still arriving.
- **Always-valid fallback.** With no manifest it plays three baked real
  exemplar frames + a procedural LiDAR sweep, so the screen is never blank.

The autonomous-driving subject is unmistakable: real dashcam frames, a rotating
360° LiDAR point cloud with a sweeping scan line, detection boxes, sensor-suite
badges (7 cameras · LiDAR · 19 radar · egomotion), and a per-clip difficulty score.

---

## 1. Serve it (the display half)

The `demo-wall` service is defined in `docker-compose.override.yml`:

```bash
cd Lakehouse/Iceberg
docker compose up -d demo-wall
```

Then on the **demo PC**, open a browser full-screen (kiosk) at:

```
http://<server-ip>:8090
```

Find `<server-ip>` with `ip -4 addr` on the server. The page auto-starts; no
clicks needed. For a true kiosk:

```bash
# Chrome/Chromium kiosk, auto-fullscreen, no chrome UI, no screensaver
chromium --kiosk --noerrdialogs --disable-infobars \
         --incognito http://<server-ip>:8090
```

Disable the demo PC's screen blanking (`xset s off -dpms`).

---

## 2. Make it "live" with real clips (recommended)

This builds the real-clip playlist so each loop is a different clip. It reads
the NFS subset on the **server**, so run it on the server.

### Install the extractor deps (one-time)

```bash
cd Lakehouse/Iceberg/demo_wall
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt          # av DracoPy pyarrow numpy pillow
# optional extras:
pip install ultralytics                    # real YOLOv8 2D detections  (--yolo)
pip install "trino[sqlalchemy]"            # real Gold difficulty scores (--trino)
```

### Extract ~40 real clips

```bash
# from Lakehouse/Iceberg  (note --nfs default is ./netai-e2e/...):
python demo_wall/extract_assets.py --n 40 --yolo --trino
```

Outputs into `demo_wall/assets/clips/`:
`<clip_id>.jpg` (frame), `<clip_id>.bin` (Float32 xyz LiDAR), and `manifest.json`.
The running nginx serves them immediately — just refresh the wall browser.

Re-run anytime (e.g. nightly via cron) to rotate in a fresh set of clips:

```bash
0 4 * * *  cd /path/to/Lakehouse/Iceberg && demo_wall/.venv/bin/python demo_wall/extract_assets.py --n 60 --yolo --trino
```

If `--nfs` is wrong, the script tells you "no candidates" — point it at whichever
of `./netai-e2e` or `./netai-e2e-orig` holds the dataset.

---

## 2b. Portable / USB build (no server, offline)

If the wall PC can't reach this server, build a fully self-contained copy that
runs by **double-clicking `index.html`** — no web server, no internet, no Chrome
flags. It inlines the manifest + LiDAR clouds as JavaScript (browsers block
`fetch()` under `file://`, so inlining is what makes a USB copy work).

```bash
# after extract_assets.py has populated assets/clips/
python demo_wall/build_portable.py             # default: up to 500 clips (~4.6 h loop)
python demo_wall/build_portable.py --size 1500  # ~1.5 GB, ~2.7-day non-repeating loop
python demo_wall/build_portable.py --clips 2000 # exactly ~2000 clips
python demo_wall/build_portable.py --all        # package the ENTIRE extracted library
```

Copy the whole `demo_wall_portable/` folder to the USB stick. On the demo PC:

- **Easiest:** double-click `index.html`, then press **F11** for full screen.
- **Kiosk:** `run_windows.bat` (Windows) or `./run_linux.sh` (Linux) — launches
  Chrome full-screen automatically.

**How it scales:** the manifest stays tiny; each clip's LiDAR cloud is a small
per-clip `*.cloud.js` lazy-loaded on demand via `<script>` injection (allowed
under `file://`, unlike `fetch`). So the loop can be hours or **days** long
without one giant file. Budget ≈ **0.25 MB and ~33 s of loop per clip** — a
128 GB stick holds far more than you'd ever loop through. If neither `--size`
nor `--clips` nor `--all` is given, it caps at 500 clips so you never
accidentally emit a multi-GB folder. `--size`/`--clips` auto-extract more clips
first if the library is smaller than the target (needs the venv + NFS);
`--no-extract` packages only what's already there. The folder is offline and
self-contained, and also works if served over HTTP.

## 3. Refresh the headline numbers from Trino (optional)

The baked figures live in `js/data.js` (pulled from `MEDALLION_PROGRESS.md`).
`extract_assets.py --trino` already updates the live clip/score totals in the
manifest. To also refresh the big aggregate counts, edit `js/data.js`.

---

## Files

| File | Role |
|------|------|
| `index.html` | 4-stage layout |
| `css/style.css` | wall-scale dark theme (clamp()-based responsive type) |
| `js/data.js` | baked real lakehouse figures (fallback + headline numbers) |
| `js/lidar.js` | dependency-free Canvas point-cloud renderer (real or procedural) |
| `js/app.js` | loop orchestration, counters, histogram, detection overlay, particle flow |
| `extract_assets.py` | samples real NFS clips → frames + LiDAR clouds + manifest |
| `requirements.txt` | core extractor deps |
| `assets/cam1–3.jpg` | three real baked exemplar frames (always-valid fallback) |
| `assets/clips/` | real-clip library produced by the extractor |

## Tuning

- Stage dwell time: `STAGE_MS` in `js/app.js` (default 8200 ms → ~33 s/loop).
- Points per LiDAR sweep: `--cloud-points` (default 12000; lower if the GPU-less
  wall PC struggles with the Canvas render).
- Number of clips in rotation: `--n`.
