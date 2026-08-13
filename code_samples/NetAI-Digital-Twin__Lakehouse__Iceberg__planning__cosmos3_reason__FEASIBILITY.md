# Cosmos 3 Edge — feasibility for this pipeline (2026-08-10)

NVIDIA released **Cosmos3-Edge** (4B, OpenMDW1.1, not gated) on 2026-07-20, the
third tier of the Cosmos 3 family after Nano (16B) and Super (64B, 2026-05-31).
Motivation for evaluating it: both of our Cosmos-adjacent efforts were bounded by
**model size**, not by idea quality —
augmentation needs a 4x A100-40GB cluster booking per clip
(`cosmos_augmentation/FINDINGS.md`), and the VLM difficulty scorer was shelved
partly as "infeasible on 24 GB" (`planning/alpamayo/FINDINGS.md`).

**Verdict: split, and the split is structural.**
- Generation / augmentation — **NO-GO**. Edge has no video-to-video path at all.
- Reasoning / difficulty scoring — **GO**. Fits one local A10 with headroom.

## 1. Generation (augmentation): NO-GO

Cosmos3-Edge's **generator input is text + image + action trajectory only — no
video input**. NVIDIA states it plainly: *"Cosmos3-Edge currently doesn't support
video-to-video transfer."* Nano and Super do expose `transfer-control
video-to-video` (edge / blur / depth / seg / wsm hints) through vLLM-Omni; that
capability was cut from the Edge tier.

This is disqualifying rather than merely inconvenient, because the entire
label-validity argument of our augmentation pipeline rests on **depth-controlled**
generation preserving geometry so `obstacle.offline` labels and ego pose transfer
for free. Edge offers no control-video conditioning:
- image-to-video from frame 0 would re-invent every non-ego agent — precisely the
  hallucination failure already caught and gated by `safety.hallucination_gate`;
- action conditioning (AV embodiment, 9D) steers **ego** motion only, not other
  agents, so it does not rescue label validity.

Secondary limits, moot given the above but worth recording: **480p max** output
(we render at 704) and 50-150 frames (our window is 121, which would have fit).

### Corollary: "smaller model" != "smaller VRAM" for video diffusion
Cosmos-Transfer2.5-**2B** — a third of Transfer1-7B's parameters — still documents
**65.4 GB VRAM** for single-GPU inference, with no offload path. Its
`--model=edge/distilled` flag denotes the *Canny-edge control modality*, not an
edge device (a naming collision worth not tripping over). So the parameter count
of a video-diffusion model is a poor proxy for its footprint, and no released
Cosmos generation checkpoint currently gets augmentation off the A100 cluster.
If the generation half is revisited, the candidate to measure is **Cosmos3-Nano's
unified transfer-control v2v** (one model consuming the control video in-sequence,
instead of Transfer1's per-modality ControlNet stack) — not Edge.

## 2. Reasoning (difficulty axis): GO

This is the workload `planning/alpamayo/FINDINGS.md` recorded as
*"infeasible on 24 GB"*: Alpamayo-1.5-10B at 22 GB resident, forced to a degraded
1-frame / 64-token config, 66.5 s/clip -> ~610 h for the 33k cohort.

| | Alpamayo-1.5-10B (what we ran) | Cosmos3-Edge |
|---|---|---|
| Params | 10B | 4B |
| Reasoner weights on disk | ~20 GB | **4.87 GB** (BF16, MoT AR tower + ViT) |
| Context | forced to 1 frame / 64 tok | 131k positions, video input (4 fps rec.) |
| Throughput | 66.5 s/clip | 3.2 s e2e video on Jetson Thor; 10.3 req/s @ conc-64 on RTX PRO 6000 |
| Serving | vendored NVlabs repo | stock `transformers` / vLLM |

Reasoning quality (NVIDIA's numbers, vs 2B-class peers):

| | General | Robotics | Smart infra. | **Driving** |
|---|---|---|---|---|
| Cosmos3-Nano (16B) | 69.6 | 55.1 | 61.0 | **76.0** |
| **Cosmos3-Edge (4B)** | 60.7 | 48.5 | 50.3 | **61.8** |
| Cosmos-Reason2-2B | 57.5 | 42.5 | 47.7 | 55.7 |
| Qwen3-2B-VL-Instruct | 60.3 | 38.7 | 42.5 | 42.7 |

Driving breakdown for Edge: LingoQA 61.0, AVSpecialCollision 67.3,
AVSpecialStopBehavior 57.1.

The **logit-EV trick transfers unchanged** — the reasoner is a standard
autoregressive tower with next-token decoding, so restricting the next-token
distribution to `'0'..'9'` after a steered prefix still yields a continuous,
100%-parseable, deterministic score from a single forward pass. That is the one
reusable artefact from the Alpamayo work and it survives the model swap.

### The honest caveat
The VLM scorer was shelved for a **construct** reason, not a compute reason: CoC
AUC 0.604 < conflict's 0.651, and the negative control was only +0.03 (blanked
frames scored almost the same -> substantially prior-driven). **A cheaper model
does not fix a construct problem.** What Edge buys is that the ">=40 GB retry"
listed under "If revisited" becomes a ~single-shift job on hardware we already
own, at *full* config rather than the degraded one — which is the only way to
distinguish "the construct is wrong" from "we never gave it a fair run".

## 3. Hardware fit on this host

Supported microarchitectures are **Ampere / Hopper / Blackwell**, and **BF16 only**
(FP4/FP8/FP16 explicitly untested). Local inventory:

| GPU | Arch | VRAM | Usable |
|---|---|---|---|
| Quadro RTX 6000 | Turing | 24 GB | **No** — pre-Ampere, no native BF16 |
| NVIDIA A10 | Ampere | 23 GB | **Yes** |

So this runs single-GPU on `cuda:1`, no cluster booking. Reasoner weights are
4.87 GB, leaving ~18 GB for activations + KV — the exact headroom Alpamayo lacked.
Download is ~7.8 GB (`transformer/*.safetensors` carry both MoT towers, plus
`vision_encoder/`); the 1.4 GB `vae/` is generator-only and is skipped.

## 4. Decision

Adopt Edge for the reasoning/gating half only; leave generation on Cosmos-Transfer1.
Implemented as the detachable module in this directory — see `README.md`.

**Measured outcome (2026-08-10), see `RESULTS.md`:** the caveat above was half right.
A cheaper model did not fix the construct — the *cold* path reaches 0.636 at full
config and still adds nothing to production. But the *reasoned* path reaches 0.670
and lifts the production union 0.630 -> 0.690 (bootstrap CI [+0.024, +0.096]), which
no Alpamayo variant came close to. It is scene-grounded under a swapped-frame control
(AUC collapses to 0.501). Blocked on a circularity check before adoption.

## Sources
- Model card: https://huggingface.co/nvidia/Cosmos3-Edge
- Launch post: https://huggingface.co/blog/nvidia/cosmos3edge
- Cosmos 3 platform repo: https://github.com/nvidia/cosmos
- Transfer2.5 inference docs: https://github.com/nvidia-cosmos/cosmos-transfer2.5/blob/main/docs/inference.md
- Cosmos 3 technical report: https://arxiv.org/pdf/2606.02800
