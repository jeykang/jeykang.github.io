# Driving-model evaluation pipeline — preliminary feasibility (2026-08-11)

Goal: a consumer of the Gold-curated dataset should be able to evaluate a driving
policy against a curated slice without hand-rolling a harness. Question asked:
*is there still no closed-loop validation system that can consume this kind of
dataset?*

**Answer: that has changed, and decisively — the stack is now NVIDIA-native to this
exact dataset family.** But the coverage numbers against *our* clips are the problem,
not the tooling.

## 1. What now exists (all post-dating the last look)

| Component | What it is | License | Status |
|---|---|---|---|
| [AlpaSim](https://github.com/NVlabs/alpasim) | open-source closed-loop AV sim; gRPC policy interface, Docker Compose, pluggable renderers | **Apache-2.0** | released |
| [AlpaGym](https://github.com/NVlabs/alpagym) | closed-loop **RL training** harness — wires AlpaSim (env) + Cosmos-RL (trainer) to a policy. **Not a successor to AlpaSim**; it sits on top of it, and is for post-training rather than evaluation | open | released |
| [OmniDreams](https://huggingface.co/nvidia/omni-dreams-models) | generative world-model renderer; closed-loop observations **without** per-scene reconstruction | weights published | released |
| NuRec | Omniverse neural-reconstruction engine (3DGS / 3DGUT), real logs -> simulatable USDZ | NGC container, NVIDIA account | released |
| [InstantNuRec](https://github.com/NVIDIA/instant-nurec) | feed-forward reconstruction, ~1.5 s per 10-20 s multi-camera scene | **Apache-2.0** | Jul 2026 |
| NCore v4 | canonical multi-sensor interchange format; the ingestion layer for NuRec | open | released |
| [PhysicalAI-AV-NuRec](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles-NuRec) | 1,607 pre-reconstructed scenes (26.04), ~20 s each | NVIDIA AV NuRec license (gated) | released |
| [PhysicalAI-AV-NCore](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles-NCore) | 1,147 clips already in NCore v4 | gated | released |

AlpaSim emits per-rollout `metrics.parquet` plus violation-categorised videos
(`collision_at_fault`, `offroad`, `dist_to_gt_trajectory`, `collision_rear`), and
decouples metric computation from rollout — which is the right shape for us, since
it means metrics can land in Iceberg alongside `clip_scores`.

Critically for the long-standing map gap: the NuRec scene USDZ **bundles `map.xodr`
(OpenDRIVE) and `mesh.ply`**. Our raw `labels/` has only `egomotion` and
`obstacle.offline` — no map layer at all — which is exactly why
`planning/README.md` splits "rung 2: map-free PDMS" from "rung 3: map-dependent
PDMS (needs an estimated drivable-area map)". Reconstruction supplies rung 3's map.

## 2. Coverage against OUR data — measured, and this is the catch

| set | count |
|---|---|
| NuRec reconstructed scenes (26.04) | 1,607 |
| clips on local disk | 33,767 |
| parent `clip_index` | 306,152 |
| **NuRec ∩ our on-disk clips** | **160** |
| NuRec ∩ clips with `conflict` scored | 140 |
| NCore ∩ our on-disk clips | 131 |
| NuRec ∩ parent clip_index | 1,607 (all of them) |

So closed-loop covers **0.47% of the clips we hold**. Every NuRec scene belongs to
the parent dataset, so the other 1,447 are obtainable — they simply were not in our
subset selection. But 1,607 of 306,152 is what NVIDIA has reconstructed, full stop.

**This is the whole feasibility question.** A Gold tier of 3,174 clips whose eval
suite can only ever contain ~160 of them is not an evaluation pipeline for *our*
curation — it is an evaluation pipeline for NVIDIA's chosen 1,607. Two ways out:

- **(a) Reconstruct our own Gold clips.** NCore conversion -> NuRec -> USDZ. The
  format mapping is proven for this dataset (the `-NCore` release *is* these clips),
  and our data has everything reconstruction needs: 7 cameras
  (front_wide/front_tele/cross_left/cross_right/rear_left/rear_right/rear_tele),
  `calibration/camera_intrinsics`, `sensor_extrinsics`, `vehicle_dimensions`, and
  ego poses from `labels/egomotion`. InstantNuRec's profiles want 1, 3 or 5 cameras
  — we have 7, so the 5-camera profile is satisfiable.
- **(b) Constrain the eval suite to the 1,607 reconstructable clips.** Immediate and
  cheap, but it inverts the system: curation would be selecting from NVIDIA's list
  rather than the lakehouse driving the selection. Defeats the stated purpose.

(a) is the real project; (b) is a legitimate week-one shortcut to get an end-to-end
rollout working before committing to the reconstruction pipeline.

## 3. NAVSIM: useful, but not the closed-loop answer

Three findings, in order of importance:

1. **NAVSIM is non-reactive by construction.** It is "data-driven *non-reactive*
   simulation"; v2.1+ added two-stage reactive agents and CoRL'25 added
   *pseudo*-simulation, explicitly positioned as a midpoint between open- and
   closed-loop. It is not a closed-loop validator, so it does not answer the ask on
   its own.
2. **EPDMS needs a map for 4 of 9 sub-metrics** — DAC, DDC, TLC (all *multipliers*)
   and LK (weighted). Because DAC/DDC/TLC are multipliers, without a map the score
   is not merely degraded, it is **undefined**. Only NC, EP, TTC, HC, EC are
   computable from `obstacle.offline` alone.
3. **Format lock.** v2.2 consumes OpenScene/nuPlan layouts; there is no documented
   custom-dataset ingestion path. Adapting our clips is a real conversion project on
   top of the map problem.

Where NAVSIM-style scoring *does* fit is the cheap tier — and we already have
working prototypes of exactly that: `planning/pdms_test.py` (map-free trajectory
feasibility over `obstacle.offline`) and `planning/alpamayo/pdms_planner_gate.py`
(collision/TTC scoring of a real planner's output).

## 3b. Is this NVIDIA-dataset-locked? No — but the dependency is real

The lock is not AlpaSim and not the PhysicalAI dataset. It is **NCore v4**, which
NVIDIA positions explicitly as a canonical interchange format: *"any NuRec user can
convert their own proprietary fleet data into NCore as the common ingestion layer"*,
covering cameras, lidar, radar, IMU, depth and stereo, with a converter template.
The chain is:

```
any dataset with a calibrated multi-camera rig + ego poses
      -> NCore v4 -> NuRec -> USDZ -> AlpaSim
```

So the binding constraint is a **data property** — a calibrated multi-view rig with
known extrinsics and ego trajectory — not a vendor dataset. That is a much better
fit for a multi-dataset lakehouse: "normalise any dataset into NCore" is precisely a
lakehouse-shaped job, and it slots in as a sim-facing sibling of Bronze/Silver/Gold.

Three honest caveats. It is still an **NVIDIA-controlled format**; each new dataset
needs its **own converter**; and datasets lacking calibrated multi-view coverage
cannot be reconstructed at all, no matter how good the labels are. Worth noting the
current state of the multi-dataset claim as well: `nvidia_ingestion` is populated
(`nvidia_bronze/silver/gold` in MinIO), while `kaist_ingestion` is presently a
scaffold (`config.py` + `__init__.py` only). The generality requirement is a design
goal here, not yet a second populated dataset — which affects how much the
constraint should be allowed to steer near-term work.

## 3c. The cost problem, and the renderer that may dissolve it

Two facts sharpen this. **NuRec is per-scene optimisation, not feed-forward** —
confirmed in its docs (training pipeline, per-scene checkpoints). And
**InstantNuRec does not replace it**: it is feed-forward at ~1.5 s/scene but emits
only *static* scene Gaussians as PLY, no dynamic actors and no USDZ; its documented
role is to *initialise* downstream NuRec training. So sim-ready scenes cost a
per-scene optimisation run, and NVIDIA publishes no wall-clock figure for it.

Hardware bounds that run: NuRec requires **>24 GB VRAM minimum, >48 GB recommended**,
Ampere supported (A100, A10, A40, RTX A6000). Our A10 is 23 GB — **below the stated
minimum** — so reconstruction is A100-cluster work, and the cluster is one node at a
time. That is the real source of the "prohibitively long at scale" worry, and it is
about *reconstruction*, not about rollout.

**The alternative: skip reconstruction entirely.** AlpaSim's renderer is pluggable,
and one of the supported renderers is a generative world model — OmniDreams, via
FlashDreams. OmniDreams conditions on past frames, current simulator state and the
policy's immediate actions to generate observations autoregressively; the paper's
explicit motivation is that "reconstruction-based neural simulators ... are
fundamentally constrained by their initial captured data". It reports real-time
operation and was demonstrated closed-loop with Alpamayo 1 under the AlpaSim
orchestrator. Weights are published (`nvidia/omni-dreams-models`).

If that path works on our clips it removes the per-scene reconstruction step — which
simultaneously answers the scale objection *and* weakens the dataset-lock objection,
since there is no USDZ scene catalogue to be a member of. Unverified for us:
generalisation beyond its 21k-hour training distribution, VRAM, and licence. This is
the highest-value thing to test in this whole area.

## 4. Proposed shape: two tiers

**Tier 1 — open-loop / pseudo-sim, all 33,767 clips, available now.**
Map-free PDMS (NC, EP, TTC, HC, EC) over the dataset's own agent tracks. No new
heavy dependencies, no reconstruction, no GPU beyond the policy itself. This is the
plug-and-play default any consumer gets on any Gold slice, and most of the code
exists. Ceiling: no drivable-area/lane metrics, no reactivity, no counterfactuals.

Tier 1 is also the tier that actually serves the multi-dataset thesis: it needs only
agent tracks and ego poses, which every AV dataset has, and carries no NVIDIA
dependency of any kind. Closed-loop is inherently reconstruction- or model-bound and
will always be a curated-suite affair.

**Tier 2 — closed-loop via AlpaSim, with two possible renderers.**
- *Reconstruction path* (NuRec USDZ): highest fidelity, map metrics included,
  per-scene optimisation cost, A100-only, storage-heavy.
- *Generative path* (OmniDreams/FlashDreams): no per-scene reconstruction, no scene
  catalogue, plausibly dataset-agnostic — but fidelity and generalisation unproven
  for our clips.

Start on the 160 overlap (or pull `public_2604` wholesale) to prove a rollout, then
choose a renderer path on measured numbers rather than on the assumption that
reconstruction is mandatory.

**Scale, reframed:** closed-loop evaluation is not run over a dataset, it is run over
a *suite*. Producing that suite is exactly what Gold curation is for — the simulator
only ever sees the selection, so the relevant number is hundreds of scenes, not
33,767. The reframe is only partial, though: if every curation change requires
re-reconstructing scenes, iteration is slow and the curation/eval loop is coupled.
The generative renderer is what would decouple them.

## 5. Risks and blockers, ranked

1. **UNVERIFIED, AND IT GATES EVERYTHING: can AlpaSim load self-reconstructed
   scenes?** Its catalog is CSV-driven (`data/scenes/sim_scenes.csv`) with an
   `artifact_repository` column, but the only documented value is `huggingface`, and
   `docs/DATA_PIPELINE.md` covers rollout output rather than scene ingestion. If
   custom USDZ cannot be registered, Tier 2 can *never* cover our curated clips and
   option (a) collapses. **Verify before any other work.**
2. **Storage.** 1.74 GB/scene. The 160-clip overlap alone is **278 GB** against
   **217 GB free**. A 1,000-scene suite is ~1.7 TB. Needs a storage decision before
   this becomes real.
3. **VRAM, and it is tight everywhere.** NuRec needs >24 GB (48 GB recommended) —
   the A10's 23 GB is below minimum, so reconstruction is A100-cluster-only, one
   node at a time (pod09). AlpaSim itself is ~20 GB, and its tutorial cites ~40 GB
   for Alpamayo 1 / 1.5 (~60 GB with CFG navigation) — so sim + a large policy does
   not co-locate on one 40 GB A100 either. Smaller reference policies (VaVAM,
   Transfuser) are the sane starting point.
4. **Licensing, and this one matters for a system meant for consumers.** AlpaSim and
   InstantNuRec are Apache-2.0, but NuRec *scenes* are under a gated NVIDIA AV
   dataset licence, and NuRec itself is an NGC container. A consumer of our Gold
   data cannot be handed an eval suite without accepting NVIDIA terms of their own.
   Tier 1 has no such constraint — another reason to build it first.
5. **Reconstruction quality is not uniform.** The NuRec release ships
   `clip_ratings_26.04.csv` and AlpaSim's suites explicitly exclude "known-invalid
   scenes". Any curated eval suite must join reconstruction quality against our
   difficulty scores, or we will select hard clips that render badly.

## 6. Recommended next steps

1. **Build Tier 1.** It is the only part that is genuinely multi-dataset, needs no
   NVIDIA component, runs on all 33,767 clips today, and is mostly assembly of
   `planning/pdms_test.py` + `planning/alpamayo/pdms_planner_gate.py`. Results land
   in Iceberg beside `clip_scores`. This is the plug-and-play deliverable.
2. **Two cheap measurements that decide Tier 2's shape**, in this order:
   a. *Can AlpaSim register a locally-supplied USDZ?* (risk #1) — if not, the
      reconstruction path can never cover our curated clips.
   b. *What does one NuRec reconstruction actually cost on an A100?* One scene
      converts "prohibitively long" from a guess into a number. If it is minutes, a
      500-scene suite is an overnight job and the objection dissolves; if it is
      hours, the generative renderer becomes the only viable path.
3. **Evaluate the generative renderer (OmniDreams).** Highest-value unknown in this
   area: it would remove per-scene reconstruction, remove the scene-catalogue
   membership requirement, and make closed-loop plausibly dataset-agnostic.
4. Only after 2 and 3 decide whether to build the NCore conversion + reconstruction
   pipeline at all.
