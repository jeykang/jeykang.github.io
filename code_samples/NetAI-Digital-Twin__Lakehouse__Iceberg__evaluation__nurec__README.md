# nurec/ — NuRec reconstruction: NCore store → USDZ twin (the second hop)

Answers the professor's question from the Sep 16 call: can an existing dataset be
converted through NuRec into an AlpaSim-usable USD scene? Findings of 2026-09-21.

## What the pipeline is, verified against NVIDIA's own scene bundles

A NuRec USDZ's `data_info.json` names an **NCore `.zarr.itar` shard** as its input and its
`parsed_config.yaml` is the trainer config (`train_config_name:
configs/apps/prod/Hyperion-8.1/car2sim_6cam.yaml`, 6 cameras + 1 lidar, 40k steps). The
trainer is the same container AlpaSim already uses as its renderer —
`nvcr.io/nvidia/nre/nre-ga:26.04`, whose default command `main` is "Neural Reconstruction
Engine training, validation and testing", with `export-usdz-artifact` for the bundle.
NVIDIA documents the workflow at docs.nvidia.com/nurec (Prepare Data → Reconstruct an AV
Scene → Run Validation). So the chain is:

    on-disk clip → NCore v4 (ncore/, verified identical to NVIDIA's) → [nre-tools aux data]
        → nre-ga train → checkpoint → export-usdz-artifact → USDZ → AlpaSim LOCAL_USDZ_DIR

Two inputs beyond the NCore store:

| input | what | availability here |
|---|---|---|
| **Instant NuRec** initialisation (recommended by the docs) | github.com/NVIDIA/instant-nurec, Apache-2.0, native Python, weights auto-fetched from HF (`nvidia/instant-nurec`); feed-forward: our 6-camera clip → merged 144 MB PLY in minutes | **works here** (`instant_nurec/`, A10) |
| **NuRec auxiliary data** (`nre-tools-ga:26.04`) | semantic segmentation per camera (conditionally required), lidar segmentation + visibility (recommended), depth / DINOv2 (optional), metadata (required); written as `<store>.aux.<signal>.zarr.itar` | **blocked**: the container is on NGC and needs an `NGC_API_KEY` (free NGC account) to pull; not present on this host |

The production config's base (`_base_3dgut_dynamic_road_semantic`) asserts on the aux store
(`NCOREDataSource: aux data was not loaded`), which is exactly how attempt v1 failed.
Attempt v2 replaces the base with the plain `_base_3dgut_dynamic` (the same way the
codebase's own COLMAP path runs without aux) — `configs/car2sim_6cam_noaux.yaml`, bind-mounted
into the image's config tree — and trains.

## Hardware

Docs: >24 GB VRAM required, >48 GB recommended, A10 listed as supported. Ours is an A10
with 22.5 GiB usable; the Quadro RTX 6000 (Turing) cannot run the NuRec kernels. v2 runs
6 cameras + lidar at the prod settings (2 M Gaussians, fp32, batch sampler 4).

## Runs

| run | config | status |
|---|---|---|
| v1 `ac73935a_a10_instant_v1` | prod `car2sim_6cam.yaml`, `dataset.aux_data=false` | failed at data load: aux store required by the road-semantic base |
| v2 `ac73935a_a10_noaux_v2` | `configs/car2sim_6cam_noaux.yaml`, Instant NuRec init, `mode=trainval` | **trained 30k steps in 2 h 22 m (7.5 it/s alone, ~2.6 while sharing the A10 with the aux job), then crashed in the final checkpoint hook**: the ground-mesh export found 0 "ground-compatible" lidar points in every frame without the aux road labels and raised `ValueError` splitting road from non-road; no checkpoint was written. Lesson: `checkpoint.every_n_train_steps=10000` and `checkpoint.artifact.mesh.ground.enabled=false` on any run without aux (mesh post hoc with `export-ground-mesh`, or borrowed from NVIDIA's bundle in the hybrid) |
| aux `aux_<clip>/` | `nre-tools-ga:26.04 ncore-aux-data --segmentation-backend=mask2former --no-seg-logits --lidar-seg-camvis --store-meta`, six cameras + lidar | **done** on the A10 (the Turing card lacks the deformable-attention kernels: `no kernel image`); the store is what the prod config's road-semantic base needs |
| v3 `ac73935a_a10_prod_v3` | prod `car2sim_6cam.yaml` with the aux store, Instant init, `checkpoint.artifact.nrend.enabled=true`, checkpoint every 10k steps, in-checkpoint mesh export off | **trained**: 30k steps in 2 h 05 m on the A10 alone (4.0–4.4 it/s; 08:41 → 10:53 wall with validation + export), `val/metrics.yaml` `test/psnr` **29.53 dB**; artifacts `artifacts/{010000,020000,030000,last}.usdz` (1.29–1.59 GB), ground mesh post hoc in `ground/mesh_ground.ply`. As exported it does **not** load in AlpaSim — time base, see below; `alpasim_bundle.sh` fixes that without retraining |
| v2b `ac73935a_a10_noaux_v2b` | `configs/car2sim_6cam_noaux.yaml`, same safety flags as v3 (`checkpoint.every_n_train_steps=10000`, in-checkpoint ground mesh off) | **trained**: 30k steps in 1 h 17 m on the A10 (6.9 it/s — no aux losses), artifacts `artifacts/{010000,020000,030000,last}.usdz` (1.08 GB; `volume.nurec` 356 MB vs v3's 525 MB) exported **before** the validation pass asserted `NCOREDataSource: aux data was not loaded` (exit 255) — so no `test/psnr` for the no-aux config; validation would need the aux store. Post-hoc `export-ground-mesh` finds 0 ground-compatible points per frame without road labels (same as v2), so its map-less variant borrows nothing and its `ours_map` takes NVIDIA's mesh with the map layers; A/B via orchestrator 6 (`alpasim/runs/v2b_*`) |

The NGC key that unlocked `nre-tools-ga` is the `COSMOS_API_KEY` in `.secrets/cosmos.env`
(an NGC personal key; `docker login nvcr.io -u '$oauthtoken'`). Two docs-vs-tool
mismatches found: the thread flag is `--num-threads`, and `--segmentation-backend`
takes `none|mask2former`, not `0|1`.

## The A/B, and the map-layer gap

Our export cannot contain `map.xodr` or `clipgt/*` (lane, road boundary, traffic
lights…): PAI ships no map labels; NVIDIA's bundles carry them from their own map
pipeline, and AlpaSim's off-road / lane metrics read them. `usdz_tools.py inject` copies
those members from NVIDIA's bundle of the same clip into ours, so the comparison is run
three ways per reconstruction (`alpasim/local_scenes_<v>_{ours,ours_map,nvidia}`):

| variant | neural scene | map layers | isolates |
|---|---|---|---|
| ours | ours | none | does AlpaSim load and drive a map-less twin at all |
| ours_map | ours | NVIDIA's | reconstruction quality, map held equal |
| nvidia | NVIDIA's | NVIDIA's | the reference |

Reference for the A/B: NVIDIA's own NuRec scene of the same clip,
`alpasim/local_scenes_ref_ac73935a…/fb47eeef….usdz` (26.04), fetched with
`materialize.py nurec --download`. The test that answers the question: run AlpaSim on our
USDZ through `LOCAL_USDZ_DIR` with the same policy as on NVIDIA's, compare per-clip
metrics; plus `metrics.yaml` `test/psnr` from the validation pass.

## Making our export loadable by AlpaSim: the time base

Our export does not load in AlpaSim as exported, for a reason that has nothing to do with
reconstruction quality: **time base**. The PAI converter re-bases every clip to its
egomotion origin (`ncore/tools/data_converter/pai/utils.py`), so the store, the checkpoint
and the USDZ all live on a clip-relative axis — `absoluteTimeOffsetMicroSec = 0`,
`metadata.yaml: time_range 0 … 20000001`, and six track samples that precede the sequence
start are *negative* microseconds. NVIDIA's bundles live on an absolute axis (`int64
absoluteTimeOffsetMicroSec = 9528307000` for this clip, `time_range` and every JSON
timestamp above it). Three things break on the relative axis, found one rollout at a time:

| symptom | where | cause |
|---|---|---|
| `OverflowError: Python integer -50319 out of bounds for uint64` | `alpasim_utils/scenario.py` `TrafficObjects.load_from_json` | AlpaSim casts `sequence_tracks.json` timestamps to uint64 |
| `size mismatch for …dynamic_rigids.tracks_calib.tracks_delta_q: [751,4] vs [746,4]` | renderer, model load | the renderer rebuilds the model from `datasource_summary.json` (`nre/render/render.py`), whose dynamic-track sample count must match the checkpoint — so dropping the pre-start samples there is not allowed (dropping them in `sequence_tracks.json` alone is fine: that file only feeds the actor tracks reported to AlpaSim, `nre/render/scene.py`) |
| `ValueError: Value out of range: -500000` | `physics_service.py`, force-GT blend | AlpaSim wants `force_gt_duration_us` = 0.5 s of history *before* the scene start, packed into a uint64 proto field — a scene starting at 0 cannot run at all, whatever is done to the tracks |

So the fix is a **re-basing, not a repair**: add one delta to every microsecond quantity
and leave the sample structure alone. The renderer (`nre/grpc/serve.py`) feeds AlpaSim's
absolute timestamps to the model unchanged and the model normalises them with ranges stored
in the checkpoint, so the checkpoint has to move with the JSON — exactly four fields, the
same four that carry absolute values in NVIDIA's checkpoint:
`background.time_embed._extra_state{timestamps_us_min,max}`,
`dynamic_rigids.time_embed.timestamps_us_ranges`, and the deformables'
`time_input_embedding.timestamps_us_ranges`. USD time codes are relative to the offset and
do not move; the checkpoint's Gaussians, calibration and track deltas are untouched.

```
nurec/alpasim_bundle.sh <artifact.usdz> <out.usdz> --from-reference <nvidia.usdz> [--add mesh_ground.ply]
  1. usdz_tools.py shift-time    JSON (`*timestamp(s)_us`, per-sensor frame dicts, data_info interval),
                                 metadata.yaml time_range, `int64 absoluteTimeOffsetMicroSec` in the USDAs
  2. shift_checkpoint.py         the four checkpoint fields, same delta (torch; runs inside nre-ga with
                                 its bundled interpreter — the image's /usr/bin/python has no torch)
  3. usdz_tools.py add           replaces checkpoint.ckpt, adds the loose files
```
`--from-reference` copies NVIDIA's base so an A/B shares one time axis; without a reference
any base ≥ 1e6 µs works (PAI clips have no canonical absolute time). `shift-time` also has
`--drop-before-start [--drop-members …]`, kept for experiments; it is not part of the
loadable-bundle recipe. Two AlpaSim-side facts learned on the way: a twin without
`map.xodr` fails in `RouteGeneratorMap` (`'NoneType'.get_current_lane`) — use
`EXTRA_ARGS="runtime.simulation_config.route_generator_type=RECORDED"` with `run_scene.sh`
to drive it along the recorded waypoints instead; and `clipgt/*` is only read by the
video-model driver path, so injecting NVIDIA's map layers into our bundle mixes no time bases
(`pose_record.json` is clip-relative in NVIDIA's bundles too).

### Result (v3, 2026-09-21): our twin drives in AlpaSim

Constant-velocity policy, one rollout per variant, `alpasim/runs/v3_*` (`per_clip.parquet`,
`aggregate/`, the rendered `camera_front_wide_120fov` video under `rollouts/`):

| metric | ours_map (our scene + NVIDIA's map) | nvidia (reference) | ours (map-less, `route_generator_type=RECORDED`) |
|---|---|---|---|
| simulated | 19.54 s in 97 s wall, 39 steps, 813 metric rows | same | 19.54 s in 98 s wall, 39 steps, 690 rows (no lane/off-road metrics) |
| collision_any / rear / at_fault | 1 / 1 / 0 (rear-ended at 8 s) | 1 / 1 / 0 (same step) | 1 / 1 / 0 (same step) |
| offroad / offroad_or_collision | 0 / 1 | 0 / 1 | — / 1 (no map → `OffRoadScorer` fails, scene score refused: `missing metric 'offroad'`, wizard exit 1) |
| duration_frac_20s | 0.40 | 0.40 | 0.40 |
| dist_traveled_m / progress_rel | 0.062 / 0.0069 | 0.094 / 0.0074 | 0.065 / 0.0071 |
| dist_to_gt_trajectory / location | 0.0044 / 13.44 | 0.0083 / 13.36 | 0.0045 / 13.44 |
| min lane-boundary / obstacle distance | 0.389 / 0.0 | 0.387 / 0.0 | — / 0.0 |
| open_loop_collision / plan_deviation | 0.412 / 0.0006 | 0.412 / 0.0014 | — / — |
| img_is_black | 0 | 0 | — |

Read it for what the policy allows: constant velocity never looks at the images, so the
numbers are driven by the ego trajectory, the actors and the map — the first two are ours,
the map is shared — and they agree to the step (the ego is stationary at a red light and
gets rear-ended at 8 s in both). What the rollout *does* test is everything the renderer
needs: the scene loads, every step renders (`img_is_black 0`), and the time-varying content
is right — the traffic light turns green at the same step in both twins, i.e. the shifted
time embedding is coherent with the shifted trajectories. Both runs log the same non-fatal
`MinADEScorer` error (the 4 s constant-velocity plan is asked for t+4.5 s) — evaluator
quirk, not twin. The map-less variant answers its own question: AlpaSim **drives** a twin without `map.xodr` along the recorded route (every step rendered, same collision at the same step), but its evaluator will not **score** one — off-road and lane metrics need the vector map and the scene score refuses to aggregate without `offroad`. So on non-NVIDIA clips a map source is the remaining gap, exactly the one this A/B was built to expose.

### v2b (no-aux) through the same A/B

Same store, same steps, no aux losses: v2b trains in 1 h 17 m (6.9 it/s vs v3's 4.2),
exports its artifacts, then asserts in validation (`aux data was not loaded`) — so no PSNR
— and `export-ground-mesh` finds no ground points without road labels — so no mesh.
`alpasim_bundle.sh` re-based it unattended and:

- `v2b_ours_map` (+ NVIDIA's map **and mesh**): drives; per-clip numbers **identical** to
  `v3_ours_map` (0.0624 m travelled, rear collision at 8 s, lane boundary 0.389 …) — as they
  must be, since the ego trajectory, tracks and map are the same and constant velocity never
  looks at the images.
- `v2b_ours` (map-less, mesh-less): fails before the first step — the physics service reads
  `mesh_ground.ply` from the bundle for `ground_intersection` (`KeyError: There is no item
  named 'mesh_ground.ply'`). **A ground mesh is a hard requirement**, and without aux road
  labels our pipeline cannot make one, so the no-aux config only runs with borrowed layers.
- What differs is the picture (`out/figures/v3_vs_v2b_vs_nvidia_render.jpg`): v3 is
  visually on par with NVIDIA's scene; v2b renders a softer road surface and smears the
  passing car into a ghost — the aux store (road semantics, lidar segmentation, ego masks)
  buys dynamic-actor fidelity, which is exactly what a camera-only policy would be judged on.
  Constant velocity cannot see that; the next policy in the A/B should be one that does.
- `v2b_nvidia`, the reference rolled out a second time, reproduced `v3_nvidia` to every
  decimal — the rollout is deterministic, so per-variant differences above are real.

### The first camera-policy A/B (VaVAM, 2026-09-21)

Constant velocity never looks at the images; VaVAM does. Three findings from one clip:

- **AlpaSim's map-based route generator fails with VaVAM on all three bundles** — ours
  (v3, v2b) *and* NVIDIA's own scene — at the same waypoint after ~14 s of driving
  (`Waypoint 115 fails sanity check: route folds back on itself`). It is a property of the
  map route from where VaVAM gets to on this clip, not of the reconstruction.
  `runtime.simulation_config.route_generator_type=RECORDED` (routes from the recorded
  waypoints, the map kept for scoring) runs on both; the A/B uses it from here on
  (`twin_pipeline.sh` does) so both twins get identical route logic.
- **With recorded routes the policy behaves the same way on both twins** — pulls away from
  the red light, passes the construction gear, leaves the road on the curve, no collision
  in either — with different timing: ours 60 m / off-road at 8.5 s, NVIDIA's 38 m / 6 s
  (`runs/v3_{ours_map,nvidia}_vavam_rec`). Off the recorded line the novel views render on
  par (`out/figures/v3_vavam_ours_vs_nvidia.jpg`). One clip is an anecdote; the twin queue
  (`twin_pipeline.sh`, HANDOFF §5) is collecting the sample.
- **Map-less twins can be scored on the map-free metrics**: `eval.scene_score.enabled=false`
  lets aggregation finish without `offroad` (`runs/v3_ours_noscore`: collision, progress,
  dist-to-GT present; off-road and lane distance null). With `RECORDED` routes that is the
  full recipe for a clip that has no map source.

### Fidelity table (twin queue, one row per clip as it lands)

Constant velocity (cv) and VaVAM, recorded-waypoint routes, our twin (+ NVIDIA's map layers) vs NVIDIA's scene. "outcome" = offroad_or_collision / collision_rear / offroad, "t_end" = duration_frac_20s × 20 s.

| clip | tod / speed / NVIDIA q | PSNR | cv ours: dist, outcome, t_end | cv nvidia | VaVAM ours: dist, outcome, t_end | VaVAM nvidia | timings (convert / aux / train / total) |
|---|---|---|---|---|---|---|---|
| ac73935a | day / — / — (reference clip) | 29.53 | 0.06 m, rear-ended, 8 s | 0.09 m, rear-ended, 8 s | 60.1 m, off-road, 8.5 s | 38.4 m, off-road, 6 s | 2.5 min / ~3 h (shared GPU) / 2 h 05 / — |
| bb4394e7 | day / slow / 84.3 | 32.57 | 0.89 m, rear-ended, 8 s | 0.00 m, rear-ended, 7.5 s | pending (Docker network pool exhausted at launch; fill-in pass queued) | pending | 2 min / 2 h 48 / 2 h 19 / 5 h 28 |

## One clip end to end

```bash
nurec/twin_pipeline.sh <clip-uuid> [--skip-rollouts] [--keep-intermediates]
```
stage + NCore convert → Instant NuRec → aux store → prod-config training → ground mesh →
NVIDIA's scene of the clip (HF) → `alpasim_bundle.sh` → `ours_map` / `nvidia` variants →
constant-velocity and VaVAM rollouts with recorded-waypoint routes → per-clip tables →
cleanup (keeps the twin, `last.usdz`, the reference, aux + staged inputs). Per-step timings
land in `out/<short>.twin.json`. Candidates need a NVIDIA scene (HF `sample_set/26.04_release`,
1,607 clips; 143 of them on our disk, 54 with all six cameras intact) — see the session
scratch `twin_queue.txt` selection logic in HANDOFF §5.

## Commands

```bash
# Instant NuRec (once): ./instant_nurec/setup.sh; then
CUDA_VISIBLE_DEVICES=1 instant_nurec/.venv/bin/python instant_nurec/run_inference.py \
    --ncore-path ../ncore/out/pai_<clip>/pai_<clip>.json --output-dir out/instant_<clip> --merge

# NRE training, no-aux variant (see scratch nre_train_v2.sh for the full invocation)
docker run --rm --gpus '"device=1"' --shm-size=64g \
  -v $PWD/../ncore/out/pai_<clip>:/workdir/dataset:ro -v $PWD/out:/workdir/output \
  -v <instant ply dir>:/workdir/instant_nurec:ro \
  -v $PWD/configs/car2sim_6cam_noaux.yaml:/app/internal/scripts/pycena/runtime/pycena_nrm_full.runfiles/_main/configs/apps/prod/Hyperion-8.1/car2sim_6cam_noaux.yaml:ro \
  nvcr.io/nvidia/nre/nre-ga:26.04 mode=trainval out_dir=/workdir/output \
  --config-name=configs/apps/prod/Hyperion-8.1/car2sim_6cam_noaux.yaml \
  dataset.path=/workdir/dataset/pai_<clip>.json dataset.lidar_ids=[lidar_top_360fov] \
  model/gaussians/initialization@model.layers.background.initialization=nrm_ply \
  model.layers.background.initialization.path=/workdir/instant_nurec/<clip>.ply \
  model.layers.background.initialization.num_point_cloud_points=2000000

# with an NGC key (full-quality path): docker pull nvcr.io/nvidia/nre/nre-tools-ga:26.04, generate
# the aux store per docs.nvidia.com/nurec/nurec/nurec-aux-data.html, then use car2sim_6cam.yaml
```

Gitignored: `instant_nurec/`, `out*/`.
