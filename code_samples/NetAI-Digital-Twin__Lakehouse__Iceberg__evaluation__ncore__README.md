# ncore/ — NCore v4 serving mode (NVIDIA mode of the 진열 layer)

The chain that makes a curated clip closed-loop capable is

    on-disk clip (+ offline features) -> NCore v4 -> NuRec -> USDZ -> AlpaSim (LOCAL_USDZ_DIR)

This directory holds the first hop. `repo/` is an unmodified checkout of
[NVIDIA/ncore](https://github.com/NVIDIA/ncore) (Apache-2.0), whose
`tools/data_converter/pai` is the converter NVIDIA itself used to produce the
`PhysicalAI-Autonomous-Vehicles-NCore` release — so our conversion *is* the
reference conversion, not a re-implementation. `stage_pai_clip.py` is the only code
of ours: it lays one on-disk clip out in the per-clip `pai-clip-dl` layout the
converter's local mode reads, and fetches what the NFS subset lacks.

## The offline-feature constraint, measured 2026-09-21

The converter refuses any clip without four `.offline` features (offline egomotion,
offline camera / sensor / lidar calibration; `data_provider.REQUIRED_OFFLINE_FEATURES`).
Our NFS subset carries none of them (only `obstacle.offline`). That looked like it
would confine conversion to NVIDIA's ~1.1k NCore clips. It does not:

| set | clips | with all four offline features on HuggingFace |
|---|---|---|
| corpus (`feature_presence.parquet`) | 306,152 | 298,326 (97.4%) |
| on-disk subset | 32,651 | 31,861 (97.6%) |
| Gold top-300 (`user_data/gold_top300.json`) | 300 | 290 |
| on-disk ∩ NCore release | 130 | 130 |

The offline features are small per-chunk files in the gated *main* dataset
(`calibration/<name>.offline/<name>.offline.chunk_XXXX.parquet`,
`labels/egomotion.offline/egomotion.offline.chunk_XXXX.zip`), which our token can
read; the NCore release itself is a separate gated repo we are **not** approved for
(403), so a byte-level diff against NVIDIA's output needs an access request. The
data property the twin depends on is therefore "offline-calibrated multi-camera rig
+ offline ego pose", and the corpus has it almost everywhere.

## Setup (no Bazel)

```bash
git clone https://github.com/NVIDIA/ncore repo          # or the vendored checkout
uv venv --python 3.12 .venv
VIRTUAL_ENV=$PWD/.venv uv pip install -r repo/deps/pip/requirements_ncore.in \
    -r repo/deps/pip/requirements_pai.in -r repo/deps/pip/requirements_tools.in "zarr<3"
```
`zarr<3` because `ncore` imports `zarr._storage`; `PyNvVideoCodec` needs an NVIDIA GPU
for decode.

## Stage, then convert

```bash
# 1. per-clip layout: filtered calibration/metadata, offline features from HF,
#    obstacle.offline from the NFS chunk zip, media symlinked
.venv/bin/python stage_pai_clip.py --clip <clip_id> --out staged

# 2. NVIDIA's converter, local mode
PYTHONPATH=repo .venv/bin/python -m tools.data_converter.pai.converter \
    --root-dir staged --output-dir out pai-v4 --clip-id <clip_id>
# -> out/pai_<clip_id>/pai_<clip_id>.ncore4.zarr.itar (+ one per sensor)
```
`evaluation/materialize.py ncore` wraps both for a Gold selection. NuRec
reconstruction of the result needs NVIDIA's NuRec container and >24 GB VRAM
(FEASIBILITY.md 3c) — an L40S job, not this host.

## Verified 2026-09-21: one Gold-eligible clip converted and read back by NVIDIA's loader

Clip `ac73935a-548f-402f-8a6b-16688261b219` (on disk, in the NCore release and the
NuRec 26.04 catalog): staged in 4 s with the four offline features fetched from HF,
converted in 2 min 10 s on the Quadro RTX 6000 (NVDEC decode, `CUDA_VISIBLE_DEVICES=0`),
output 3.0 GB — six camera stores of 134–386 MB, a 1.1 GB lidar store, the main store
and the sequence meta JSON, named exactly as the NVIDIA release names them.
`validate_ncore.py` then opened it with `SequenceLoaderV4`: sequence id
`pai_ac73935a…`, span 20.00 s, six cameras, one lidar, `platform_class hyperion_8.1`,
`calibration_type pai-calibration`, provenance carried in `generic_meta_data`.

Two caveats found on the way. All seven on-disk ∩ NCore ∩ NuRec clips have one or two
cameras whose NFS files are 0 bytes (the April extraction bug), which crashes the
converter; `stage_pai_clip.py` now refuses such clips unless `--allow-empty`, and the
conversion is run with `--camera-id` for the good cameras (this clip: rear_tele
excluded). And this host's cameras decode at ~60 fps, so a 7-camera clip is ~2.5 min —
fine for a Gold suite of hundreds, an overnight job for thousands.

## Reference diff: our conversion vs NVIDIA's release of the same clip (2026-09-21)

Access to the NCore release was granted the same day, so `compare_ncore.py` opened
both stores of `ac73935a` with NVIDIA's loader and compared them level by level:

| level | ours vs NVIDIA |
|---|---|
| files | same names; per-camera stores within 0.2% in size (e.g. cross_left 377.7 vs 377.1 MB); lidar 1216.1 vs 1216.4 MB; main store 1.5 vs 1.5 MB; NVIDIA has `rear_tele` (277.7 MB), which our NFS copy lost |
| sequence meta | identical span [0, 20000001) us, `converter_version 1.0.0`, calibration/egomotion types, platform, vehicle bbox; only the three `source_*` provenance keys were empty on our side (now written by `stage_pai_clip.py`) |
| cameras (6 shared) | 599 frames each, timestamps identical to the microsecond, `T_sensor_rig` identical |
| decoded pixels | mean abs difference 1.0–2.1 of 255, max 13–34, 0.00–0.87% of pixels off by more than 8: decoder noise, not content |
| lidar | 199 frames, timestamps and extrinsics identical; 337,839 / 342,443 / 339,806 ray bundles in frames 0 / 100 / 198, return distance and intensity max diff **0.0** |
| ego poses | rig→world at nine sample times, max diff 0 |
| cuboids | 1,186 observations on both sides, first observation identical |

So the NCore mode of the serving layer reproduces NVIDIA's own conversion for the
same clip on every structured quantity, with images equal up to the decoder. The
`rear_tele` gap is our data loss (the April extraction bug), which NVIDIA's copy of
the raw clip did not have.

## The DGX Spark (aarch64) runs this hop too — verified 2026-09-21

NuRec itself (`nre-ga`, `nre-tools-ga`, AlpaSim's renderer) is amd64-only, but the
converter is plain Python + NVDEC. On the lab's DGX Spark (GB10, aarch64, CUDA 13,
`torch 2.14+cu130`, `PyNvVideoCodec` aarch64 wheel; `point-cloud-utils` has no aarch64
wheel and is not needed by the converter) the same clip converted in **54 s** (2 min 10 s
on the Quadro RTX 6000 here). `compare_ncore.py` against the x86 output: all structural
rows identical, decoded pixels bit-identical (both NVDEC), lidar return distances differ
by 3.05e-5 m (fp32 rounding in the aarch64 build). Setup lives in
`~/netai-lakehouse/` on the Spark (`.venv`, `ncore/`, `staged/`, `out/`); clips are
staged with `rsync -aL` of a `stage_pai_clip.py` directory (symlinks resolved, ~540 MB) or,
once the NFS share is mounted there (the user is in group 1007, `nfs-common` is
installed, port 2049 is reachable), directly.

## Files
| file | role |
|---|---|
| `stage_pai_clip.py` | NFS chunk layout + HF offline features -> `pai-clip-dl` per-clip layout |
| `repo/` | NVIDIA/ncore checkout (gitignored) |
| `staged/`, `out*/` | staging and conversion outputs (gitignored) |
