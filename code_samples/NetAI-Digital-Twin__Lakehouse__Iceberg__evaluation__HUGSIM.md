# HUGSIM — is it less strict about input data than AlpaSim? (2026-09-24, updated 2026-10-01)

The professor's question is whether existing driving datasets can be turned into closed-loop
scenes. With AlpaSim the answer so far is "yes, for PhysicalAI-AV clips, with a gap": our
NuRec twins drive (`nurec/README.md`), but AlpaSim's route generator and its off-road / lane
metrics need an HD map (`map.xodr`, `clipgt/*`), which PAI does not ship, and its physics
needs a ground mesh, which our pipeline can only make with the aux store. HUGSIM
(Zhou et al., arXiv 2412.01718, TPAMI 2026; github.com/hyzhou404/HUGSIM, MIT) is the
obvious alternative. This note records what it needs, checked against the paper and the
code on 2026-09-24, and then installed and run on this workstation (see "Test on this machine").

## Short answer

**Yes, on the data side HUGSIM is markedly less strict.** It reconstructs from RGB images,
camera poses and intrinsics, plus 3D boxes for the moving vehicles (dataset labels, or a
monocular 3D tracker's noisy output in the paper); LiDAR is optional, and 2D semantics and
optical flow are pseudo-labels it computes itself (InverseForm). **No HD map is needed
for any of its scores**: the shipped benchmark scenarios run with `load_HD_map: false`; the
map only enables IDM "normal" behaviour for inserted actors, and only nuScenes has one.
What it asks in return is a perspective camera model, its own reconstruction, its own
metric, and a policy that speaks its client protocol. **Tested 2026-09-24:** one of our
PhysicalAI-AV clips was converted (f-theta → pinhole, our aux semantics), reconstructed in
2 h 10 min on the A10 (test PSNR 25.5 dB) and driven closed-loop with no map, by our own
constant-velocity client and by LTF (results table below). **2026-10-01:** the same VaVAM
checkpoint we run in AlpaSim now drives HUGSIM through AlpaSim's own wrapper, and the
recorded traffic can be replayed. On that clip the two simulators fail VaVAM at the same
place, the left bend after the work zone (37–46 m in HUGSIM, 38–43 m in AlpaSim on
NVIDIA's scene), once HUGSIM renders a front view as wide as AlpaSim's. With the default
nuScenes-style view it crashes earlier, into the work zone itself ("Same policy, same clip"
below).

## Side by side

| | AlpaSim + NuRec (what we run now) | HUGSIM |
|---|---|---|
| scene | NVIDIA NuRec USDZ, reconstructed with `nre-ga` from an NCore v4 store | its own 3DGS model (HUGS), trained by `train_ground.py` + `train.py` from its per-dataset preprocessing |
| reconstruction inputs | cameras + LiDAR + egomotion + cuboid tracks; the prod config also needs the aux store (road semantics, LiDAR seg, ego masks) | RGB + poses + intrinsics + 3D boxes; semantics and flow derived; LiDAR optional |
| HD map | default route generator and off-road / lane metrics need it; our workaround (`RECORDED` routes, scene score off) drives map-less twins but drops off-road and lane metrics | not used by any score; only for IDM actor behaviour |
| drivable area / off-road | map lanes and road boundaries | ego footprint over the reconstructed road-class Gaussians (`ground.ply`, 2×2 grid, <30 % supported → 0) — `score_calculator.py`, `hug_sim.py` |
| collision | actors from the log / trafficsim; static scenery has no collision geometry (driving into it shows up only as off-road, through the map) | >100 reconstructed background points inside the ego box (any scenery that is not road, sidewalk or sky), or polygon overlap with actor boxes |
| ground | `mesh_ground.ply` is a hard requirement (physics); needs aux road labels to generate | multi-plane ground model learned from ground-class Gaussians |
| camera model | f-theta + rolling shutter natively (NVIDIA's own rig) | perspective; fisheye cameras are rectified in preprocessing (KITTI-360 is); custom rig via `configs/sim/*_camera.yaml` (count, intrinsics, extrinsics) |
| actors | log replay (+ traffic model), with each vehicle's reconstructed appearance | 3DRealCar vehicle assets inserted per scenario, constant-speed / IDM / aggressive (optimisation-based) planners; recorded vehicles can be replayed along the paths fitted to their tracks (`UnicyclePlanner`, original timing), as 3DRealCar stand-ins or, after a fixed re-orientation (`hugsim/pai/export_actors.py`), in their own reconstructed appearance |
| policies | VaVAM, Alpamayo, our policy bridge | UniAD, VAD, LTF (NAVSIM) clients; waypoints → LQR → kinematic bicycle; **VaVAM through AlpaSim's own wrapper** (`hugsim/clients/vavam_client.py`, ours). A third party's Alpamayo port is unresolved (NVlabs/alpamayo#54: frame convention x-forward vs HUGSIM's x-right/y-forward, rate mismatch) |
| metric | AlpaSim eval (collision / off-road / progress / at-fault …) | HD-Score = Rc × mean over time of NC · DAC · weighted(TTC, comfort) |
| platform | amd64 + Ampere (NRE); ~6 h per clip on our A10 incl. aux | `linux-64` only (pixi), PyTorch 2.4.1 + CUDA 11.8 → not the DGX Spark as packaged; renders ~89 FPS on a 3090; per-scene training time not stated |
| released scenes | 1,607 PAI scenes (NuRec 26.04) | 70+ sequences, 400+ scenarios from KITTI-360, Waymo, nuScenes, PandaSet (HF `XDimLab/HUGSIM`); used for RealADSim @ ICCV 2025 |

## What a PAI clip would need

1. **A `data/pai/` preprocessor**, like their per-dataset `load.py`. It would read our NCore
   store rather than raw PAI, since that already holds the decoded frames, egomotion poses,
   calibrations and `obstacle.offline` cuboid tracks. The one real conversion is optics:
   all seven PAI cameras are **f-theta with a rolling shutter**
   (`camera_intrinsics.offline`). They must be rectified to pinhole images, which costs field of view on
   the 120° cameras (the 30° / 70° ones are near-pinhole), and the rolling-shutter skew is
   ignored by a global-shutter 3DGS.
2. **Their pseudo-labels** (InverseForm semantics, flow), then `train_ground.py` +
   `train.py` per scene on the A10 (time to be measured; CUDA 11.8 supports sm_86).
3. **A scenario YAML** per clip: ego start from the log, optional inserted actors.
4. **A policy client**: our policy bridge would need a HUGSIM client (waypoints in its frame
   and rate) to run VaVAM or Alpamayo; UniAD / VAD / LTF work out of the box. *Done for VaVAM
   (2026-10-01): `hugsim/clients/vavam_client.py`; Alpamayo would follow the same pattern.*

## Caveats

- DAC is coarser than AlpaSim's off-road check. At simulation time `ground.ply` is the set of
  scene Gaussians whose semantic argmax is **road** (class 0; `sim/hugsim_env/envs/hug_sim.py`),
  and DAC asks whether the ego footprint (a fixed 1.6 × 3.0 m box) sits over enough of them.
  Road and sidewalk (classes 0–1, `train_ground.py`) are used only to fit the ground's
  *height*. So leaving the road onto a sidewalk does count, but there are no lanes, road
  edges or driving direction. Collision uses the scene Gaussians that are neither road,
  sidewalk nor sky.
- HD-Score and AlpaSim's metrics are not interchangeable; a cross-simulator comparison
  needs a mapping (collision ↔ NC, off-road ↔ DAC, progress ↔ Rc). The same physical
  failure can also land in different columns. Running wide into a roadside barrier is a
  background collision in HUGSIM, where every non-road Gaussian is an obstacle; in AlpaSim
  it is off-road, since scenery has no collision geometry there. Compare *where* and *when*
  a policy fails (distance driven, sim time), not the metric names.
- The camera a policy gets is part of the simulator setup. AlpaSim feeds VaVAM the 120°
  f-theta front-wide image, as NVIDIA configured it. HUGSIM renders pinhole views from a
  rig file, and the nuScenes rig's 65° front camera changed VaVAM's behaviour on our clip
  (below).
- Research code (a few dozen commits, a competition release); no aarch64 or CUDA 12+ build.
- Reconstruction quality is below NuRec's on the one clip measured: 26.4 dB held-out PSNR
  with LiDAR seeding (25.5 without) against NuRec's 29.5, with different splits and
  resolutions. NuRec models the PAI rig exactly; HUGSIM sees rectified, cropped views.

## Reconstruction input format (for a PAI converter)

From `data/nusc/load.py` and `scene/dataset_readers.py`: a source directory with
`meta_data.json` (`camera_model: OPENCV`; per frame `rgb_path`, 4×4 `camtoworld`, 3×3
`intrinsics`, `width`, `height`, `timestamp`, and `dynamics` = per-object poses; per object
`verts` = box corners), `images/<cam>/…`, an initial point cloud `points3d.ply` (required)
and `ground_points3d.ply` for the ground model, plus optional `semantics/<cam>/*.npy`
(20-class Cityscapes-style; effectively required, since the ground model trains only on
road/sidewalk pixels), `flow/…_flow.npy` and `depth/…pt`. For PAI: poses, intrinsics of the
rectified cameras, boxes and LiDAR points all come from the NCore store; semantics come
from InverseForm (their pipeline, needs apex), or could be mapped from the Mask2Former labels
our aux store already holds.

## Test on this machine (2026-09-24)

Everything lives in `evaluation/hugsim/` (gitignored: `repo/`, `NAVSIM/`, `data/`, `runs/`,
`.pixi-cache/`; committed: the scripts and configs below, so the whole thing detaches by
deleting the directory).

- **Container, not host.** HUGSIM compiles gsplat, tiny-cuda-nn, pytorch3d and simple-knn
  against PyTorch 2.4.1+cu118, which needs nvcc 11.x and GCC ≤ 11. The host has no CUDA
  toolkit and GCC 12/13 (Ubuntu 24.04), so `Dockerfile` builds `hugsim-dev:cu118`
  (CUDA 11.8 devel, Ubuntu 22.04, pixi, uid 1000), and `build_env.sh` creates both pixi
  environments inside it, mounted at the same absolute path. Build time was dominated by a
  slow package mirror; the HUGSIM environment is 8.3 GB. The CUDA extensions are built
  for sm_86 (the A10); GPU 0 (Turing, running Isaac Sim) is not used.
- **Data** (1.6 GB of the 61 GB release): nuScenes `scene-0383`, the scenario pack, the two
  3DRealCar assets `scene-0383-hard-00` inserts, and NAVSIM's LTF checkpoint. The scene's
  `cfg.yaml` carries the authors' absolute paths; `model_path` is repointed (original kept
  as `cfg.yaml.orig`), and `configs/nuscenes_base_local.yaml` holds this machine's paths.
- **First closed loop, no HD map:** `scene-0383-easy-00` (no inserted actors,
  `load_HD_map: false`) driven by `clients/cv_client.py`, a constant-velocity client written
  against HUGSIM's named-pipe protocol (the same reference policy as our AlpaSim runs): 17
  steps, then background collision on the left verge where the road bends. NC 0.47,
  DAC 0.53, TTC 0.47, comfort 1.0, Rc 0.16, **HD-Score 0.077**; 29 s wall for the whole
  episode on the A10 (`runs/nusc_cv_ltf/scene-0383_easy_00/`: `eval.json`, `video.mp4`,
  `data.pkl`). The simulator, renderer, ground and collision checks all work, and a client
  of our own drives it, which is the policy-bridge path VaVAM or Alpamayo would take.

**All closed-loop runs so far** (HD-Score = Rc × mean_t(NC · DAC · weighted(TTC, comfort)); no
HD map in any of them):

| scene | scenario | policy | NC | DAC | TTC | comfort | Rc | HD-Score | wall |
|---|---|---|---|---|---|---|---|---|---|
| nuScenes scene-0383 (released) | easy-00, no actors | constant velocity (ours) | 0.47 | 0.53 | 0.47 | 1.0 | 0.16 | 0.077 | 29 s |
| nuScenes scene-0383 (released) | easy-00, no actors | LTF | 1.0 | 1.0 | 1.0 | 1.0 | 0.56 | **0.56** | 45 s |
| nuScenes scene-0383 (released) | hard-00, 2 inserted cars | LTF | 0.25 | 1.0 | 0.19 | 1.0 | 0.26 | 0.053 | 28 s |
| **our PAI clip ac73935a** (converted) | easy-00, no actors | constant velocity (ours) | 0.18 | 0.73 | 0.18 | 1.0 | 0.15 | 0.027 | 17 s |
| **our PAI clip ac73935a** (converted) | easy-00, no actors | LTF | 0.43 | 0.48 | 0.24 | 1.0 | 0.28 | 0.083 | 27 s |
| ac73935a | rec-00: standstill start, no actors | VaVAM, 5 seeds (65° front, nuScenes rig) | 0.35–0.53 | 1.0 | 0.28–0.41 | 0–0.1 | 0.36–0.38 | 0.074–0.108 | ~1 min each |
| ac73935a | rec-00 | VaVAM, 5 seeds (100° front) | 0–0.12 | 0.62–0.67 | 0–0.05 | 0 | 0.53–0.63 | 0–0.018 | ~1 min each |
| ac73935a | rec-01: rec-00 + recorded traffic | VaVAM, seed 0 (65°) | 0.41 | 1.0 | 0.35 | 0 | 0.37 | 0.077 | ~1 min |
| ac73935a | rec-01 | constant velocity (ours) | 0.81 | 1.0 | 0.81 | 1.0 | 0.00 | 0.001 | ~1 min |
| ac73935a | rec-02: rec-01 with the recorded cars' own appearance | constant velocity (ours) | 0.81 | 1.0 | 0.81 | 1.0 | 0.00 | 0.001 | ~1 min |

HD-Score multiplies by route completion, so a policy that never moves scores ~0 even with
no collision for most of the episode; the VaVAM comfort term is 0 because it accelerates
from standstill at ~3 m/s² and keeps pushing (to 11 m/s on the 100° runs). Per-episode
rows: `hugsim/summarize_runs.py`, which reads the end reason from `data.pkl`. The final
step's info is not in `infos.pkl`.

LTF drives the released scene cleanly and is caught by the inserted cars in the hard
scenario. On our clip it turns left early, meets reconstructed background at 2.5 s and leaves
the road at 2.75 s. That clip is a hard case for a policy trained on urban nuPlan driving
(stopped at a red light, construction zone ahead, narrow winding forest road), seen through a
25.5 dB reconstruction and a nuScenes rig. One clip cannot separate these causes.

Fixes needed on the way, all mechanical:
- **NAVSIM's `pixi.lock` pins a mirror.** Every PyPI package points at `mirrors.zju.edu.cn`
  (the authors' university mirror, ~5 KB/s from here). It is kept as `pixi.lock.zju`, and
  re-solving against PyPI built the environment in 90 s.
- **3DRealCar layout mismatch.** All 430 vehicle references in the released scenarios use
  `<id>/postprocess/shadow.pth/` as the asset directory, while the Hugging Face release is
  flat (`<id>/gs.pth`, `<id>/wlh.json`). Hardlinks recreate the expected layout.
- **Authors' paths.** Scene `cfg.yaml` files and the client script carry absolute paths from
  the authors' machines; they are repointed.

```
docker build -t hugsim-dev:cu118 evaluation/hugsim
docker run --rm -v $H:$H -w $H -e FORCE_CUDA=1 hugsim-dev:cu118 bash build_env.sh   # $H = evaluation/hugsim (absolute)
evaluation/hugsim/run_closed_loop.sh evaluation/hugsim/data/nuscenes/scene-0383-easy-00.yaml ltf \
    evaluation/hugsim/configs/nuscenes_base_local_cv.yaml        # constant velocity
evaluation/hugsim/run_closed_loop.sh evaluation/hugsim/data/nuscenes/scene-0383-hard-00.yaml ltf   # LTF
```

## Converting one of our clips (ac73935a, 2026-09-24)

The same clip as our AlpaSim twin and NVIDIA's reference (ac73935a: the only one of the nine
whose NCore store still has its decoded camera streams).

1. **`hugsim/pai/load_ncore.py`** (the dataset-specific `load.py`, mirroring HUGSIM's
   `data/nusc/load.py`) reads the NCore v4 store and presents the clip as `data_type:
   nuscenes`. That means six images per 10 Hz sample, in nuScenes order, because HUGSIM steps
   back six frames to find the previous image of the same camera:
   - **Camera mapping:** CAM_FRONT ← front wide 120°, CAM_FRONT_LEFT/RIGHT ← cross 120°,
     CAM_BACK_LEFT/RIGHT ← rear 70°, CAM_BACK ← front tele 30°. The names are labels; the
     poses carry the geometry.
   - **Optics:** every f-theta camera is resampled to a pinhole through NCore's own camera
     model (fx = fy, 100° horizontal field of view for the 120° cameras, principal point
     placed low because PAI's is at y ≈ 747/1080; 100 % valid pixels).
   - **Ego car:** rows or columns where it is visible are cropped: the hood (front −42 rows,
     tele −28) and the body side on the rear cameras (−81 / −75 columns).
   - **Semantics:** taken from the NRE aux store we already have for AlpaSim. Its
     Mask2Former labels use exactly HUGSIM's Cityscapes ids (0 road … 18 bicycle, 19 ego car;
     PNG, id = R/3), so InverseForm and apex are not needed.
   - **Boxes and world frame:** cuboid tracks become HUGSIM's box frame; the world is the
     first front-camera pose. HUGSIM's own `vis_bbox_2d.py` puts the boxes on the cars in
     every camera.
   - **Cost:** 200 samples × 6 cameras (1,200 images) in 1 min on the CPU.
2. **`hugsim/pai/prepare.sh`** runs HUGSIM's own steps: dynamic masks, UniDepth V2 depth
   (8.7 frames/s on the A10) and the two merged point clouds. The PyPI xformers wheel targets
   CUDA 12.1 and cannot run under the environment's cu118 torch, so `estimate_depth_noxf.py`
   switches UniDepth to PyTorch attention. Geometry check: the ground points lie 1.54 m below
   the front camera (the LiDAR ground fit gave 1.545 m), and the camera path is 78.7 m, the
   same ~79 m the clip shows in AlpaSim.
3. **`hugsim/pai/train.sh`** runs `train_ground.py`, then `train.py` (`configs/nusc.yaml`, 30k
   iterations), then `export_scene.py` into `data/scenes/pai/<short>`. On the A10 the ground
   model takes 22 min and the scene 1 h 45 min (~4.9 it/s after densification). The export
   is 872 MB: the Gaussian scene, six moving vehicles with unicycle tracks, and the ground
   model. Held-out images: **PSNR 25.5 dB, SSIM 0.74, LPIPS 0.23** (train 26.2). NuRec reached
   29.5 dB on this clip, measured on full-resolution f-theta frames with a different split,
   so the gap (~4 dB) is indicative only. It is expected from monocular-depth
   initialisation against LiDAR plus NVIDIA's production pipeline. Seeding from LiDAR
   instead closes about 0.9 dB of it (below).
4. The closed loop reuses the nuScenes rendering rig (`nuscenes_camera.yaml`), so LTF sees
   the views it was trained on. The configs are `configs/pai_base_local{,_cv}.yaml` and
   `configs/pai/ac73935a-easy-00.yaml` (no inserted actors, no map). Note that HUGSIM
   episodes contain the static reconstruction plus the vehicles a scenario inserts
   (3DRealCar). The shipped scenarios do not replay the recorded traffic, but HUGSIM can
   replay it with stand-in vehicles (`UnicyclePlanner`; corrected 2026-10-01, see
   `ac73935a-rec-01` below).
5. **Result: our clip drives in HUGSIM with no HD map** (results table above). `ac73935a-easy-00` with the
   constant-velocity client ran 11 steps before a background collision: the road bends left
   and constant velocity heads into the right-hand rock face. NC 0.18, DAC 0.73, Rc 0.15,
   HD-Score 0.027; 17 s wall (`runs/pai_cv_ltf/ac73935a_easy_00/`; frames in
   `nurec/out/figures/hugsim_pai_cv.jpg`). Front views render cleanly. Rear views are
   smeared: the nuScenes rig has a rear-centre camera, while PAI has none in our clips (the
   rear tele stream is empty), so the area behind was barely observed, and the ground under
   the ego car never was. LTF uses only the front cameras. The static reconstruction shows
   the traffic light in one state (green), where NuRec's time-conditioned model reproduced
   red → green.

## Same policy, same clip: VaVAM in AlpaSim and HUGSIM (2026-10-01)

**The clip.** ac73935a is a one-lane work zone on a winding mountain road. A temporary
signal ("STOP HERE ON RED") stands before an excavator, a portable toilet and barrels that
close the right lane. The recorded driver waits ~6.5 s at red, passes the work zone in the
opposing lane, then follows a left bend lined with concrete barriers. In AlpaSim (recorded
route, recorded traffic) VaVAM does not wait, passes the work zone in the opposing lane too
(`wrong_lane` from 6.5 s, at most 1.6 m off the recorded path), and leaves the road on the
bend.

**Bridge.** `hugsim/clients/vavam_client.py` drives HUGSIM with AlpaSim's own VaVAM wrapper
(`alpasim_driver.models.vam_model.VAMModel`), run inside `alpasim-harness-base:0.134.0`.
The checkpoint (`VAM_width_1024_pretrained_139k` + VQ tokenizer), the preprocessing
(resize to 900×1600, NeuroNCAP transform to 288×512), the command encoding and the fp16
autocast are therefore those of our AlpaSim rollouts; only the transport differs.
`hugsim/run_vavam.sh` starts the policy on GPU 0 and the simulator on the A10, and the two
meet at the episode's named pipes. GPU 0 is the RTX 6000: VaVAM's torch 2.8/cu128 runs on
Turing, while HUGSIM's extensions are built for sm_86 only. HUGSIM's own launcher gets a
stub (`clients/external_client.sh`). Per 0.25 s step:
- **Image:** the CAM_FRONT render, with context 1, as `driver-config.yaml` sets it in our
  AlpaSim runs.
- **Command:** HUGSIM's command (0 right / 1 left / 2 forward, from the recorded path 2 s
  ahead) maps onto VaVAM's. AlpaSim derives its command from route waypoints ≥ 5 m ahead;
  both say STRAIGHT until the bend.
- **Plan:** VaVAM's 2 Hz waypoints (x forward, y left) become HUGSIM's (x right, y forward,
  0.5 s apart).

AlpaSim does not seed VaVAM's sampler. The client seeds every step, so an episode is
reproducible and different seeds measure the policy's own spread. Cost: ~200 ms per
inference, ~1 min per episode.

**Scenarios.** `configs/pai/ac73935a-rec-00.yaml` starts the ego at standstill, like the
log and AlpaSim, with no other vehicles. `ac73935a-rec-01.yaml` adds the recorded traffic:
all six moving tracks in the clip start behind the ego (the queue at the signal), and the
four longest are replayed with HUGSIM's `UnicyclePlanner`. It drives an actor along the
unicycle path fitted to a recorded track during training (`unicycle_<track>.pth`), at the
original timing. In rec-01 the appearance is a 3DRealCar stand-in, because the planner poses actors by
yaw alone in the assets' frame, while our reconstructed vehicles (`dynamic_<track>.pth`)
live in the training box frame. The two frames differ by one constant rotation per track,
C = Ry(90°)·Rz(pitch)·Rx(roll), from roma's extrinsic `xzy` (`unicycle_b2w`) against the
planner's Ry(−yaw). `hugsim/pai/export_actors.py` bakes C into each vehicle's Gaussians
(means and rotations; view-dependent colour dropped). After it, every car's longest axis
is the heading axis, as the planner needs. It writes them as assets with the
box-centre height for the scenario. `ac73935a-rec-02.yaml` replays the same four vehicles
in their own appearance; they render soft from behind, a view no training camera had. A second rig,
`configs/pai_camera_front100.yaml`, keeps the nuScenes rig but widens CAM_FRONT to a
100° pinhole. That is the widest view the reconstruction was trained on, and the closest
pinhole stand-in for the 120° f-theta image AlpaSim gives VaVAM.

| simulator · scene · front camera | scenario | policy | outcome | distance driven, sim time |
|---|---|---|---|---|
| AlpaSim · NVIDIA's NuRec scene · 120° f-theta | recorded route + traffic | VaVAM, 3 rollouts (original, exact repeat, PNG images) | passes the work zone in the opposing lane; first off-road on the left bend | **37.6 / 38.4 / 42.7 m**; 9.0–9.5 s |
| AlpaSim · our NuRec twin · 120° f-theta | same | VaVAM, 1 rollout | same; first off-road on the bend | 60.1 m; 11.5 s |
| HUGSIM · our HUGS reconstruction · 100° pinhole | rec-00 | VaVAM, 5 seeds | passes the work zone in the opposing lane; runs wide on the left bend into the barrier (collision; DAC falls to 0.62–0.67), 5/5 | **37.0–37.9 m** (4 seeds), 46.1 m (1); 5.25–6.0 s |
| HUGSIM · same · 65° pinhole (nuScenes rig) | rec-00 | VaVAM, 5 seeds | holds the closed right lane; collision with the work-zone equipment, 5/5 | 20.9–22.2 m; 4.25–4.5 s |
| HUGSIM · same · 65° | rec-01 | VaVAM, seed 0 | as rec-00 | 21.3 m; 4.25 s |
| AlpaSim · NVIDIA's scene / our twin | recorded route + traffic | constant velocity | stays stopped; rear-ended by the queue | 0.18 / 0.11 m; 11.0 s |
| HUGSIM · 65° | rec-01 (stand-ins) | constant velocity | stays stopped; rear-ended by the replayed queue | 0 m; 10.75 s |
| HUGSIM · 65° | rec-02 (own appearance) | constant velocity | same | 0 m; 10.5 s |

AlpaSim's first off-road and first collision steps are read from `metrics.parquet` (0.5 s
resolution, ±3.5 m at these speeds). The seconds in the nurec README's fidelity table
(VaVAM 8.5 s / 6 s, cv 8 s) are scored spans after AlpaSim's truncation rules, not event
times. Front views at matched distances:
`nurec/out/figures/vavam_alpasim_vs_hugsim_ac73935a.jpg`.

What this shows, on one clip:
- **With comparable inputs, the two simulators fail VaVAM at the same place.** HUGSIM with
  a 100° front view puts the failure at 37–38 m in four of five seeds. AlpaSim on NVIDIA's
  scene puts it at 37.6–42.7 m over three rollouts. The scenes were reconstructed by
  different methods (HUGS from rectified views and monocular depth; NuRec from the full
  f-theta rig and LiDAR), the vehicle models differ (kinematic bicycle vs AlpaSim physics),
  and one simulator has a map while the other has none. Our own NuRec twin lets VaVAM
  run 20 m further (60.1 m), so here HUGSIM sits closer to NVIDIA's scene than our twin
  does, though n = 1 clip.
- **The failure is labelled differently.** In AlpaSim it is off-road, because the barrier
  has no collision geometry. In HUGSIM it is a collision, because the barrier's Gaussians
  are obstacles, and DAC drops at the same step.
- **The camera matters more than the simulator.** With the nuScenes rig's 65° front view,
  the nuScenes front-camera geometry VaVAM's NeuroNCAP preprocessing is written for, VaVAM
  never swings into the opposing lane and hits the excavator in all five seeds. Seeds barely matter by comparison
  (±0.7 m at the crash in both rigs, except one 100° seed that got 9 m further).
- **Timing differs, location agrees.** HUGSIM's ego leaves at t = 0 and accelerates at
  ~3 m/s², to 7 m/s (65° runs) or 11 m/s (100° runs). AlpaSim's ego stays put until
  ~3 s (its 3.0 s recorded warm-up) and then cruises at ~7.5 m/s.
  So the bend comes at 5–6 s in HUGSIM and at 9–9.5 s in AlpaSim.
- **The traffic light differs.** It is red at t = 0 in both AlpaSim scenes and green in
  the static HUGS reconstruction, so VaVAM runs a red light only in AlpaSim. (AlpaSim's ego
  waits there for its first 3 s only because the recorded stop is replayed during the 3.0 s
  warm-up; corrected 2026-10-07, this bullet used to blame a 1.5 s warm-up.) The failure
  itself comes ~38 m past the signal in both simulators.
- **Recorded traffic, replayed.** Constant velocity stays stopped and is rear-ended by the
  queue in both simulators: at 10.5–10.75 s in HUGSIM (the recorded cars or stand-ins on
  their fitted tracks) and at 11.0 s in AlpaSim.

Runs: `hugsim/runs/pai_vavam_{,front100_}s<seed>_ltf/ac73935a_rec_00/` and
`pai_{vavam_s0,cv}_ltf/ac73935a_rec_01/`, `pai_cv_ltf/ac73935a_rec_02/`. Each holds `eval.json`, `video.mp4`, and
`vavam_steps.jsonl` (per-step command, speed, plan, inference time).

```
cd evaluation/hugsim
./run_vavam.sh configs/pai/ac73935a-rec-00.yaml 0                                   # VaVAM, seed 0, nuScenes rig
CAMERA=configs/pai_camera_front100.yaml TAG=front100 ./run_vavam.sh configs/pai/ac73935a-rec-00.yaml 0
./run_closed_loop.sh configs/pai/ac73935a-rec-01.yaml ltf configs/pai_base_local_cv.yaml   # cv + recorded traffic (stand-ins)
docker run --rm --gpus '"device=1"' -v $PWD:$PWD -w $PWD/repo hugsim-dev:cu118 pixi run python $PWD/pai/export_actors.py $PWD/data/scenes/pai/ac73935a
./run_closed_loop.sh configs/pai/ac73935a-rec-02.yaml ltf configs/pai_base_local_cv.yaml   # ... in their own appearance
docker run --rm -v $PWD:$PWD -w $PWD/repo hugsim-dev:cu118 pixi run python $PWD/summarize_runs.py 'runs/pai*_ltf/*'
```

## LiDAR-seeded reconstruction (2026-10-01)

HUGSIM initialises its Gaussians from `points3d.ply` (scene) and `ground_points3d.ply`
(ground). The stock pipeline makes both from UniDepth monocular depth. Seen from above,
that cloud is a fog around the true surfaces, while the clip's LiDAR is crisp; both are
aligned in the same frame (`nurec/out/figures/hugsim_seed_depth_vs_lidar.jpg`).
`hugsim/pai/lidar_points.py` builds both files from the NCore store's LiDAR instead and
hardlinks everything else from the depth-seeded source, so only the initialisation
changes:
1. **Points:** every LiDAR frame's motion-compensated points, minus the ego box, points
   beyond 80 m, and points inside moving vehicles' boxes (+0.3 m).
2. **Labels and colour:** each point takes them from the nearest sample's six rectified
   cameras, the same aux Mask2Former labels the training uses, with a loose occlusion test.
3. **Split and sampling:** road and sidewalk go to the ground cloud, flattened to the
   camera height like `merge_depth_ground.py` does; everything else except sky and ego goes
   to the scene cloud. Then a 5 cm voxel grid and the same point budgets as the depth
   pipeline (600k / 200k).

Of 67.2 M LiDAR points, 27.5 M are scene and 3.0 M ground, 34 M are seen by no camera,
1.9 M are beyond 80 m and 0.8 M are on moving vehicles (`data/pai/ac73935a_lidar/lidar_points.json`).
Making the cloud takes 2.5 min on the CPU. Training is otherwise identical
(`pai/train.sh data/pai/ac73935a_lidar data/models/pai_ac73935a_lidar`). On the A10 it took
26 min for the ground model and 1 h 56 min for the scene (4.3 it/s), sharing the GPU with
~25 closed-loop episodes. The export is 1.0 GB.

| held-out images (same split) | iteration 7k | 15k | 30k: PSNR / SSIM / LPIPS |
|---|---|---|---|
| depth-seeded (UniDepth) | 23.0 dB | 24.5 dB | 25.5 dB / 0.741 / 0.227 |
| **LiDAR-seeded** | 24.0 dB | 25.1 dB | **26.4 dB / 0.782 / 0.176** |

LPIPS (perceptual error) falls by 22 %. NuRec's 29.5 dB is still ~3 dB ahead, on a
different split.

Closed loop on the LiDAR-seeded scene: the same configs with `scene_name: ac73935a_lidar`
(`configs/pai/ac73935a_lidar-{easy,rec}-00.yaml`).

| policy · scenario | depth-seeded | LiDAR-seeded |
|---|---|---|
| constant velocity · easy-00 | collision at 6.5 m, 2.75 s (right-hand side, where the road bends) | identical: 6.5 m, 2.75 s |
| LTF · easy-00 | turns left early, collision at 16.2 m, 5.25 s; HD 0.083 | turns left harder, ends against the left guardrail at 6.5 m, 9.25 s; HD 0.040 |
| VaVAM · rec-00 · 65° front, 5 seeds | 5/5 collisions with the work zone at 20.9–22.2 m | 3/5 with the work zone at 21.4–21.8 m; 2/5 pass it and run into the bend's barrier at 50.7 m |
| VaVAM · rec-00 · 100° front, 5 seeds | 5/5 fail on the bend at 37.0–37.9 m (4) and 46.1 m (1) | 5/5 fail on the bend at **36.6–41.7 m** |
| *AlpaSim, NVIDIA's scene (VaVAM, 3 rollouts)* | *off-road on the bend at 37.6–42.7 m* | |

The better reconstruction changes no verdict in kind. Constant velocity is unchanged. LTF
fails the same way. VaVAM with the AlpaSim-like wide view still fails on the bend, now in a
36.6–41.7 m band that nearly coincides with AlpaSim's 37.6–42.7 m on NVIDIA's scene. The
one shift is VaVAM with the narrow nuScenes view, where two of five seeds now get past the
work zone. That fits the reading above: with 65° the work zone is a close call for VaVAM,
and the image detail tips it.

## The nine twin clips in HUGSIM (2026-10-06, running)

The question one clip cannot answer: does a map-free HUGS twin reproduce the reference
simulator's verdicts as often as our NuRec twin does (7/9 for VaVAM, `nurec/README.md`)? The
other eight clips of the twin queue now go through one script, with one protocol for all nine.

**Pipeline** (`hugsim/pai/twin_hugsim.sh <clip>`, idempotent; `pai/twin_queue.sh` runs clips in
sequence, logs and per-step timings in `runs/twin_queue/`):
1. re-convert the clip's NCore store into a temporary directory (the twin queue's cleanup had
   dropped the decoded camera streams; staging and the aux store are reused), ~2.5 min;
2. `pai/load_ncore.py --aux-sseg` (pinhole rectification, aux Mask2Former semantics), ~1 min;
3. `pai/prepare.sh` (dynamic masks, UniDepth, merged clouds), ~4 min, then
   `pai/lidar_points.py` (LiDAR seeding), ~3 min;
4. `pai/train.sh` on the LiDAR-seeded source → `data/scenes/pai/<short>_lidar`, ~2 h 20 min;
5. `pai/export_actors.py`, then `pai/make_scenario.py`: the clip replayed as recorded — ego at
   the first front-camera pose with the recorded start speed, every reconstructed moving vehicle
   whose track spans ≥ 1 s replayed by `UnicyclePlanner` on its own path and timing, in its own
   appearance (`configs/pai/<short>_lidar-rec-02.yaml`; on ac73935a it reproduces the hand-made
   `ac73935a-rec-02.yaml`, plus track 19, a 1.1 s passing car the hand-made file left out);
6. VaVAM with seeds 0–4 on the 100° front view (`configs/pai_camera_front100.yaml`, the closest
   pinhole stand-in for AlpaSim's 120° f-theta image), and constant velocity;
7. cleanup of what the same run created (temporary NCore store, checkpoints, depth, masks).

**Comparison.** `rollouts.py` collects every AlpaSim and HUGSIM rollout into one table keyed by
episode (`eval.rollout`, `EPISODES.md`), and `compare_simulators.py` puts, per clip, NVIDIA's
scene in AlpaSim (the original run, the exact repeat and the PNG-frame run), our NuRec twin in
AlpaSim, and our HUGS twin in HUGSIM side by side: failures over rollouts and the distance
driven when the episode ended. A twin agrees when its majority verdict (fail or not) matches
the reference's. "Fail" is `offroad_or_collision` in AlpaSim and a collision in HUGSIM; as above,
the same physical failure can carry different labels, so where the policy fails matters more
than the label. The starts also differ: AlpaSim replays 3.0 s of the recorded trajectory before
the policy takes over, HUGSIM hands the policy the recorded start speed at t = 0. The policy also runs on
different GPUs (the A10 under AlpaSim, the RTX 6000 as HUGSIM's client), and seeded VaVAM is not
reproducible across GPUs (`SKIP.md`, 2026-10-06), so hardware is part of the run-to-run noise here.

**The reference clip on this protocol:**

| clip | AlpaSim, NVIDIA's scene | AlpaSim, our NuRec twin | HUGSIM, our HUGS twin |
|---|---|---|---|
| ac73935a, VaVAM | 3/3 failed, 37.6–42.7 m | 1/1 failed, 60.1 m | 5/5 failed (collision on the bend), 38.8–42.1 m |
| ac73935a, constant velocity | 2/2 failed (rear-ended), 0.1 m | 1/1 failed, 0.1 m | 1/1 failed (rear-ended at 10.75 s), 0.0 m |

The eight other clips are queued (~3 h each on the A10, started 2026-10-06 00:37 UTC); their rows
replace this paragraph when the queue ends.

**Decision rule, fixed before the eight new clips' results exist** (written 2026-10-06 ~01:25 UTC,
while the first of them was still training; only ac73935a's HUGSIM runs existed):
1. *Verdict per clip and set-up:* the majority of "failed" over its rollouts (NVIDIA's scene: the
   original, repeat and PNG runs; HUGSIM: the five seeds; our NuRec twin: its one run). A tie
   counts as not failed.
2. *Agreement:* a twin agrees on a clip when its verdict equals NVIDIA's. Report agreements out of
   nine for each twin, with a Wilson 95 % interval.
3. *Reading, set in advance:* the HUGS twin is "as faithful as the NuRec twin" if its agreements
   are at least the NuRec twin's minus one, and "less faithful" if two or more short. One clip is
   within noise at n = 9, and the result is descriptive, not a test.
4. *Location:* on clips where NVIDIA's scene and a twin both fail, the twin fails "at the same
   place" if the medians of distance driven differ by at most the larger of 10 m and the spread
   of NVIDIA's own runs.
5. *Plumbing check:* constant velocity should agree on all nine clips in both twins, as it does
   for the NuRec twin (9/9). Any disagreement there is a conversion or scenario error to fix
   before the VaVAM comparison is read.
6. *Reported regardless of outcome:* all nine rows, including clips where a twin fails to build;
   a clip that cannot be built counts as not agreeing.

**Rule 5 fired (2026-10-07, interim look at six clips).** Constant velocity disagreed with
NVIDIA's scene on 2 of 6 clips in HUGSIM (0 of 6 for the NuRec twin), so, as the rule says, the
protocol was examined before any VaVAM verdict was read. Three causes, none of them the HUGS
reconstruction:
1. *Off-road did not count in HUGSIM.* HUGSIM never ends an episode for leaving the road, and its
   DAC term scores the planned trajectory; its per-frame check on the ego's footprint is commented
   out in the scorer. `summarize_runs.py` now reproduces that check post hoc, and `rollouts.py`
   counts the first of off-road and collision as the failure, with the distance driven by then.
2. *AlpaSim replays 3.0 s of the recorded drive before handing over* (`force_gt_duration_us`; the
   "1.5 s" earlier in this file was wrong), and HUGSIM handed over at t = 0. The clients now take
   `--warmup 3.0` (`clients/warmup.py`: the recorded front-camera path, sent as the plan until 3 s).
3. *Neither constant velocity held its speed.* HUGSIM's client re-read the simulator's speed each
   step and drifted up with the controller (5.0 → 7.4 m/s on a07e81de); AlpaSim's braked at
   handover, because AlpaSim reports zero speed during the recorded warm-up and our policy bridge
   passed it on (`ALPASIM.md`, Batch 2 correction; bridge fixed). The HUGSIM client now holds the
   recorded speed at handover (`--hold-speed`).
One difference is definitional and stays: AlpaSim's off-road check uses map lanes and road edges,
HUGSIM's the reconstructed road surface, so a drift that crosses a lane edge on a wide road fails
in AlpaSim only (constant velocity on 0ec48454). The comparison is reported under both protocols:
the original runs (`compare_simulators.py --protocol original`) and the corrected ones
(`--protocol warmup`: HUGSIM episodes in `runs/pai_cvwu_ltf/` and `runs/pai_vavam_front100wu_s*_ltf/`,
AlpaSim constant velocity in `alpasim/runs/<short>_nvidia_cv2/`; run by `../rerun_after_fixes.sh`).
The verdict and location rules above are unchanged.

Two follow-ups from the first warm-up episodes (2026-10-07): the first version of the warm-up aimed
the plan at the recorded positions, so a stopped ego chased centimetre offsets and turned 12° on the
spot (ac73935a); it now sends the recorded *displacements*, all zeros while the recording stands
still, and those seven episodes were discarded. And bb4394e7's HUGS twin has reconstructed
background (non-road Gaussians) ~0.3–0.6 m ahead of the start pose: any policy that moves collides
there (VaVAM after 0.56 m, constant velocity after 0.35 m), while in AlpaSim VaVAM drives 10–13 m.
That is a twin artifact (collision geometry comes from semantic labels and floaters in HUGSIM), to be
inspected before the clip's verdict is read; under rule 6 it still counts as a row.

```bash
cd evaluation/hugsim && nohup setsid pai/twin_queue.sh <clip-uuid> [...] > runs/twin_queue/queue.log 2>&1 &
cd evaluation && .skip_venv/bin/python rollouts.py && .skip_venv/bin/python compare_simulators.py [--policy constant_velocity]
```

## Cheapest decisive test

1. Install it on the workstation (pixi, x86 + A10) and run one released scene from
   `XDimLab/HUGSIM` in closed loop with LTF, which validates the install and runtime (half a
   day).
2. Write the NCore → HUGSIM preprocessor for one of our nine twin clips, train it, and drive
   it (2–3 days). The same clip then exists in both simulators, and in AlpaSim we already have
   NVIDIA's reference for it, which gives a three-way comparison on a single clip.

Sources: arXiv 2412.01718 (paper); github.com/hyzhou404/HUGSIM (README, `data/README.md`,
`data/nusc/run.sh`, `pixi.toml`, `sim/utils/score_calculator.py`,
`configs/benchmark/nuscenes/scene-0383-hard-00.yaml`); github.com/NVlabs/alpamayo/issues/54.
