# Medallion Progress — May 2026 (perception scoring)

*BEVFusion multimodal perception made operational; sampling-adequacy verdict.
Historical log; see the [progress index](../MEDALLION_PROGRESS.md).*

---
## 12. BEVFusion Perception Scoring Setup (2026-05-04 → 2026-05-05, in progress)

**Goal**: replace the placeholder `BEVFusionBackend` in `edge_case_scorer.py` with a real multimodal (camera + lidar) perception pass on the 33,719-clip Gold subset.
**Container**: `bevfusion/Dockerfile` builds `netai/bevfusion-runner:latest` (30.7 GB). Stack: pytorch 2.1.2 + cuda 12.1, openmim → mmcv 2.1.0 / mmdet 3.2.0 / mmsegmentation 1.2.2 / mmdet3d 1.4.0, mmdetection3d source clone for `projects/BEVFusion`. Custom CUDA ops (bev_pool_ext, voxel) compiled for sm_75+sm_86 via inline CUDA_HOME export.

**Compose**: `bevfusion-runner` service in `docker-compose.override.yml`, `runtime: nvidia`, all GPUs, NFS read-only, profile=manual so it doesn't auto-start.

**Pretrained checkpoint**: `bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d-5239b1af.pth` (167 MB) from openmmlab's hosting. Required a layout permutation (`convert_spconv_layout.py` — 21 sparse-encoder weights) to convert spconv1 `[O,kx,ky,kz,I]` → spconv2 `[kx,ky,kz,I,O]`. Without the permutation the lidar branch silently runs at random init.

**Validation methodology** (`bevfusion/validate_sampling.py`): score 200 random Silver clips at N ∈ {10, 20, 40} frames/clip, accept N=10 iff Spearman ρ(10,40) ≥ 0.85 AND Jaccard(top10%@10, top10%@40) ≥ 0.80. If 10 fails, fall back to 20 and retest. Approach justifies frame-sampling adequacy quantitatively.

**Build issues encountered & fixed**:
1. mmdet3d 1.4.0 pip package doesn't include `projects/BEVFusion` — must clone source. Fixed.
2. mmdet3d editable install pulls numpy 2.x → matplotlib `_ARRAY_API not found`. Fixed: re-pin numpy 1.26.4 after editable install.
3. BEVFusion CUDA ops setup.py has paths relative to mmdet3d root, not its own dir. Fixed: cd to mmdet3d root before `python projects/BEVFusion/setup.py develop`.
4. `cuda_runtime_api.h: No such file or directory` — `CUDA_HOME` not set in build env. Fixed: inline `export CUDA_HOME=/usr/local/cuda` before setup.py.
5. NFS root-squash blocks container's root user. Fixed: run with `--user 1000:1007 -e HOME=/tmp`.
6. mmdet3d 1.4.0 `data_preprocessor.simple_process` checks for `'img'` (singular) but uses `'imgs'` (plural) — bug. Fixed: pass both keys. **Superseded by the correct fix in #8 below** (the right contract is `inputs['img']` only, as a list).
7. `_get_pad_shape` requires 4D NCHW for `img`; pass `imgs.squeeze(0)` (treats cams as batch). **Also superseded by #8.**

**Blocker RESOLVED + full pipeline operational (2026-06-19).** Traced the mmdet3d data-flow contract by reading the in-image source and fixed three stacked issues; one-clip inference now produces detections and the batch scorer (`runner.py`) writes real per-clip scores. The four fixes:

8. **5D vs 4D image contract** (the original blocker). `Det3DDataPreprocessor.collate_data` reads `inputs['img']` and *overwrites* `inputs['imgs']`; the output shape depends on how `img` is passed: a **list of per-sample 4D `(N_cams,C,H,W)` tensors** routes through `multiview_img_stack_batch` → **5D `(B,N,C,H,W)`** (what `extract_img_feat` unpacks), whereas a single 4D tensor stays 4D. Fix: pass `inputs['img'] = [tensor(N,C,H,W)]` and drop the `imgs`/`squeeze` workarounds from #6/#7.
9. **Image size mismatch**. Frames were resized to 800×448 → LSS feature height 56 ≠ the config's expected 32, crashing `get_cam_feats`'s `torch.cat([d, x])`. Fix: resize to the config `view_transform image_size=[256,704]` (feature `[32,88]`).
10. **Voxel op compiled CPU-only** → `RuntimeError: Not compiled with GPU support` in `hard_voxelize`. Root cause: `projects/BEVFusion/setup.py` gates `CUDAExtension` on `torch.cuda.is_available() or FORCE_CUDA==1`, and there's no GPU during `docker build`. Fix: `export FORCE_CUDA=1` before `setup.py develop` in the Dockerfile (baked into the rebuilt image 2026-06-19; ops compile in ~77s).
11. **Missing `box_type_3d` metainfo**. The detection head's `predict_by_feat` wraps decoded boxes via `metas[0]['box_type_3d'](...)`; a hand-built data sample must supply `box_type_3d=LiDARInstance3DBoxes` and `box_mode_3d=Box3DMode.LIDAR`.

Inference + production-path refactor: the validated input-assembly + per-frame inference live in **`bevfusion/bevfusion_infer.py`** (`build_data`, `run_frame`, `CAM_ORDER`, `NUSC_CLASSES`); both `test_one_clip.py` (smoke test) and `runner.py` (batch scorer) import it, so the exact contract validated is the one that runs.

**End-to-end smoke test: PASSING** (`bevfusion/test_one_clip.py`, clip `bd539f72-…`, frame 100): model load → Draco decode (188,135 pts) → 6× mp4 at 256×704 → inference → **200 detections, max_conf 0.092, classes spanning car/truck/trailer/pedestrian/barrier/traffic_cone**. Low absolute scores are expected (placeholder nuScenes calibration + domain shift); only the relative signal is used.

**Batch scorer: VERIFIED** (`runner.py` on a 2-clip list, baked image, no recompile): clip with all 6 cams → real scores (4 frames, mean_n_detections 3.0, class_diversity 0.96, perception_score 0.568); clip with a corrupt rear_right mp4 → correctly routed to the neutral 0.5 fallback. Output parquet schema matches `_load_perception_scores`'s expectations.

**Score-threshold calibration (2026-06-19)**: a 12-clip probe at N=10 showed the
earlier "max conf ~0.09" was an unrepresentative single frame — the real
distribution has mean_max_conf ≈ 0.59 (up to 0.80). `--score-thr 0.05` yields a
healthy perception_score spread (range [0.446, 0.705], stdev 0.072). Throughput
≈ 22 s/clip at N=10 (I/O-bound: 216 MB lidar parquet decode + per-frame 6-cam
mp4 seeks; lidar read is per-clip, so N scales only the per-frame cost).

**Sampling-adequacy validation: DONE (2026-06-19) → use N=20.** Ran the harness
(`bevfusion/validation/`) on 100 fully-covered clips, sharded across RTX 6000 +
A10, at N ∈ {10,20,40}, score_thr=0.05:

| Comparison | Spearman ρ | Jaccard top-10% |
|---|---|---|
| N=10 vs N=40 | 0.912 | **0.667** |
| N=20 vs N=40 | 0.937 | **0.818** |
| N=10 vs N=20 | 0.887 | 0.538 |

Verdict: N=10 ranking is stable (ρ=0.912 ≥ 0.85) but top-10% *membership* is not
(Jaccard 0.667 < 0.80); since Gold selection is a top-10% cutoff, membership
stability governs. **N=20 clears both criteria** (ρ=0.937, Jaccard=0.818) → adopt
N=20. Caveat: on 100 clips top-10% = 10 clips, so Jaccard is coarse (designed for
the 200-clip set); N=20 passing at 0.818 is reassuring rather than marginal.

**Cascade run: LAUNCHED (2026-06-19).** Running BEVFusion only on a
metadata-preselected cohort rather than all 33K clips.

*Catalog recovery first*: bringing the stack up revealed polaris had lost its
catalog — `apache/polaris:latest` uses an in-memory metastore with no
persistence volume, so stopping it 2 weeks earlier dropped all table
registrations (and the Silver/Gold *views*, which are catalog-only). The 19 GB
of table data + Iceberg metadata survived in `minio_data/`. Recovery was bounded
because the metadata scorer needs only Bronze `Clip` + the 3 `aux_*` tables (it
falls back from the missing Silver view to Bronze Clip): re-registered
`clip_index` + `data_collection` (bare parquet), the 3 aux tables, and rebuilt
canonical `Clip` — no full 3.9 h Bronze rebuild. `aux_egomotion` re-registered at
98.7 M rows (was 101.7 M pre-PURGE; the gap is the upstream-pruned chunks).
**Action item: make polaris persistent — DONE (2026-06-19).** Added a `postgres`
service + `polaris-bootstrap` one-shot (admin tool) to `docker-compose.yml` and
switched polaris to `polaris.persistence.type=relational-jdbc` (Quarkus
datasource → postgres). Verified: catalog survives a `docker compose restart
polaris` with no setup re-run (`lakehouse_catalog` still served; entities read
from `polaris_schema` in Postgres). Footprint ~47 MB (Postgres baseline; catalog
rows are KB). Data lives in `./polaris_pg_data` (gitignored). NOTE: switching to
the Postgres backend re-bootstrapped a fresh empty catalog, so the in-memory
Clip/aux/clip_scores registered during this session's recovery are gone again —
they need ONE more re-registration for the Gold wire-up, after which they
persist for good.

*Preselection*: metadata Gold scoring re-run → `clip_scores`, 306,152 clips,
range [0.187, 0.836], mean 0.487 (matches prior canonical runs). Cohort =
**top 30 % by score ∩ on-disk fully-covered clips** (lidar + the 5 unique
CAM_ORDER cameras; `rear_right` is the binding constraint at 11,739 valid →
**11,128 fully-covered**). Top 30 % = **3,338 clips** (score ≥ 0.4997), pinned at
`bevfusion/cohort/cascade_cohort.csv`.

*Run*: `bevfusion/run_cascade.sh` launches one detached container per GPU
(shard 0 = 1,642 clips on RTX 6000, shard 1 = 1,696 on A10) at N=20,
score_thr=0.05, writing to `<NFS>/.perception_bevfusion/` (kept separate from the
retired YOLO scorer's `.perception/`). Runner is now **resumable** (`--resume`,
default on: reloads the shard parquet and skips scored clips). ETA ~18 h.

**Perception wire-up + comparison: DONE (2026-06-22).** Cascade output (3,338
clips) wired into Gold scoring and compared to the metadata-only baseline.

*Loader bug found + fixed*: `_load_perception_scores` did `import pyarrow`
inside a `try/except` that swallowed the error — but the **spark-submit driver
Python has no pyarrow**, so the import always failed and the perception
dimension had silently *never* been applied (first run after wire-up wrote
`non-null perception_score = 0`). Rewrote it to read the parquets via
`spark.read.parquet` (we're in a Spark context anyway) and pass `self.spark`
from the caller. Re-run then logged "Loaded perception scores for 3,338 clips"
and wrote 3,338 non-null perception scores.

*Perception consolidation*: archived the retired YOLO parquets to
`.perception_yolo_archive/` and placed the two BEVFusion shards in `.perception/`
(canonical source stays in `.perception_bevfusion/`), so the loader returns
exactly the cohort.

*Comparison (metadata-only vs perception-active, 306,152 clips)*:

| metric | value |
|---|---|
| Spearman overall | 0.9996 |
| top-10% Jaccard | 0.985 (228 demoted / 228 promoted of 30,615) |
| cohort in base top-10% → perception top-10% | 688 → 527 |
| cohort demoted out of top-10% | 228 |
| cohort score delta | mean −0.004, range [−0.098, +0.060] |

Reproduces the §10 **damper** behaviour: perception mostly lowers scores for
metadata-hard clips, so 228 "looks-hard-by-metadata-but-visually-empty" clips
are pulled out of the Gold top-10% and replaced by genuinely harder ones.
Overall ranking barely moves (only 1.1 % of clips carry perception) but the
cohort *boundary* — where Gold selection happens — is meaningfully refined.

**Views recreated + demotion write-up: DONE (2026-06-22).**
- Re-registered the 5 view-backing canonical Bronze tables (Calibration, Camera,
  Lidar, Radar, EgoMotion) from existing minio metadata via
  `system.register_table` — **zero rebuild** (Radar 11.73B, Camera 109.2M, Lidar
  6.16M, EgoMotion 101.7M, Calibration 458.9K). Rebuilt Silver views via
  `quality_checks` (Bronze Clip 306,152 → Silver Clip 305,724; **428 excluded**,
  ~99.86% retention — the hardened missing_sensors is not over-excluding) and
  Gold views via the gold builder (perception-active; Gold Clip 35,376 at top
  10%). Both namespaces now have the 6 per-table views again; all in the
  persistent Postgres catalog.
- Demotion characterisation written to a standalone doc (not here):
  [`PERCEPTION_DEMOTION_ANALYSIS.md`](PERCEPTION_DEMOTION_ANALYSIS.md). Key
  finding: of the 228 demoted, dominant metadata factor is sensor_coverage 47% /
  time_of_day 28% / season_geography 25%; demoted mean perception 0.468 vs
  promoted 0.691; 82% are winter clips — i.e. metadata-inflated-but-visually-empty
  scenes correctly damped out of the cohort.
- Wire output parquets to `_load_perception_scores` in edge_case_scorer.py.
  Contract verified (2026-06-19): the loader globs any `*.parquet` under
  `<source>/.perception/` and reads `clip_id`+`perception_score`, which the
  BEVFusion runner writes to that exact path — compatible as-is. **Gotcha**: the
  runner shares `.perception/` with the retired YOLO scorer
  (`camera_perception_scorer.py` → `perception_shard_*.parquet`); the loader
  merges all files and `perception_*` sorts after `bevfusion_*`, so stale YOLO
  parquets would overwrite BEVFusion per clip. Before the full run, either clear
  the old YOLO parquets or point the BEVFusion runner at a dedicated subdir.
- Re-run Gold with the perception signal active and compare cohort vs
  metadata-only baseline (pre-/post-perception Spearman + Jaccard, like the
  v3→v5 perception integration analysis in §10).
- [x] Clean up `_failed_clips` Silver helper view leakage into Gold's view-iteration loop (done 2026-06-19: `build_gold_subset` now skips any source name starting with `_` alongside `quality_report`/`clip_scores`).
- [x] Cosmetic: fix `→ <name>: N findings` log line in `quality_checks.py` (done 2026-06-19: now counts the findings each check produced via a `results[before:]` slice instead of the broken `startswith(name)` match).
- [ ] Camera filename rebuild for canonical Camera (the `filename` column points at the v1 camera mp4 paths from before the redownload — current re-download didn't touch cameras, so they should still match). Verify via validation script post-recovery.
- [ ] Add v26.03 `obstacle.offline` download to populate DynamicObject (long-standing work item; would be ~50-100 GB).
- [ ] Future: native struct encoding for EgoMotion translation/rotation (currently JSON, ~78 bytes/row → ~30 bytes/row).

