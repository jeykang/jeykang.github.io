#!/usr/bin/env python3
"""VLM driving-difficulty scorer — Cosmos3-Edge (4B) and out-of-family controls.

Successor to planning/alpamayo/difficulty_qa.py. Same construct, same scoring
trick, one-sixth the model: rate per-clip driving difficulty by reading the
next-token distribution over the digits '0'..'9' after a steered answer prefix
(logit expected value). Single forward pass -> continuous, always-numeric and
DETERMINISTIC, no sampling and no parse failures.

Why re-test a shelved axis: Alpamayo-1.5-10B was 22 GB resident on a 24 GB card,
so it only ran at a degraded 1-frame / 64-token config at 66.5 s/clip. Cosmos3-Edge
is 4.87 GB of reasoner weights, which affords the FULL config (multi-frame, full
reasoning budget) that the original verdict never got to test. See FEASIBILITY.md.

Backend-pluggable (`MODELS` below, `--model` on gate_runner). Everything except the
checkpoint is held constant across backends — same frames, same prompts, same steer,
same logit-EV readout — because the reason a second model exists here is the
circularity control described in RESULTS.md: the OOD labels are Alpamayo-lineage and
Cosmos3-Edge is the same family, so an out-of-family model has to be measured on a
byte-identical battery for its result to mean anything.

Detachable: self-contained module. Nothing in the base lakehouse imports it; frames
are read straight off the NFS mp4s (no physical_ai_av / vendored-repo dependency).

Run inside the module venv:  ./c3_venv/bin/python gate_runner.py
"""
import functools
import glob
import json
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F

# Short name -> HF checkpoint. Sizes are the comparable ones: Cosmos3-Edge is 4B,
# Gemma-4-E4B is 4B-effective. gemma-e2b is the variant that appears in NVIDIA's own
# reasoning table (Driving 24.4 vs Cosmos3-Edge 61.8), kept as a weak-model reference.
MODELS = {
    "cosmos3-edge": "nvidia/Cosmos3-Edge",
    "gemma4-e4b":   "google/gemma-4-E4B-it",
    "gemma4-e2b":   "google/gemma-4-E2B-it",
}
MODEL = os.environ.get("C3_MODEL", "cosmos3-edge")
MODEL_ID = MODELS.get(MODEL, MODEL)
N_FRAMES = int(os.environ.get("C3_FRAMES", "4"))  # Alpamayo was capped at 1 by VRAM
LONG_SIDE = int(os.environ.get("C3_LONG_SIDE", "640"))

_HERE = os.path.dirname(os.path.abspath(__file__))
_C = "/mnt/netai-e2e/nvidia-physicalai-av-subset"
ROOT = _C if os.path.isdir(_C) else os.path.join(
    _HERE, "..", "..", "netai-e2e", "nvidia-physicalai-av-subset")
SENSOR = "camera_front_wide_120fov"
_INDEX = os.path.join(_HERE, ".clip_index.json")   # gitignored cache

cv2.setNumThreads(2)


def pick_device():
    """First BF16-capable (sm_80+) GPU.

    This host has an A10 (Ampere) and a Quadro RTX 6000 (Turing). Turing has no
    BF16 and no flash/SDPA-flash kernel, so the model must not land there — and
    torch's default CUDA_DEVICE_ORDER=FASTEST_FIRST does NOT match nvidia-smi's
    index order, so a hardcoded "cuda:1" picks the wrong card. Select by capability.
    """
    env = os.environ.get("C3_DEV")
    if env:
        return env
    for i in range(torch.cuda.device_count()):
        if torch.cuda.get_device_properties(i).major >= 8:
            return f"cuda:{i}"
    raise RuntimeError("no BF16-capable (sm_80+) GPU found; Cosmos3-Edge is BF16-only")


# ── frames ──────────────────────────────────────────────────────────────────
@functools.lru_cache(maxsize=1)
def clip_index():
    """clip_id -> front-wide mp4 path (cached; the NFS glob is slow)."""
    if os.path.exists(_INDEX):
        return json.load(open(_INDEX))
    idx = {os.path.basename(p).split(".")[0]: p
           for p in glob.glob(f"{ROOT}/camera/{SENSOR}/*/*.mp4")}
    json.dump(idx, open(_INDEX, "w"))
    return idx


def load_frames(clip_id, n=None, long_side=None):
    """Evenly-spaced RGB frames from the clip's front-wide mp4, downscaled.

    Sampling at fractions rather than the first N frames matters: the augmentation
    work found the first-121-frame window is frequently empty (see
    cosmos_augmentation/FINDINGS.md, find_agent_window).
    """
    n = n or N_FRAMES
    long_side = long_side or LONG_SIDE
    path = clip_index().get(clip_id)
    if path is None:
        raise FileNotFoundError(f"no {SENSOR} mp4 for clip {clip_id}")
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out = []
    for frac in [(i + 0.5) / n for i in range(n)]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(total * frac))
        ok, f = cap.read()
        if not ok:
            continue
        h, w = f.shape[:2]
        s = long_side / max(h, w)
        if s < 1.0:
            f = cv2.resize(f, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
        out.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    cap.release()
    if not out:
        raise RuntimeError(f"decoded 0 frames for {clip_id}")
    return out


# ── model ───────────────────────────────────────────────────────────────────
def load_model(device=None, model_id=None):
    """Load a vision-language backend in BF16.

    BF16 for both: NVIDIA tests no other precision for Cosmos3, and holding dtype
    constant keeps the control honest.
    """
    from transformers import AutoModelForImageTextToText, AutoProcessor
    device = device or pick_device()
    model_id = model_id or MODEL_ID
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForImageTextToText.from_pretrained(
        model_id, dtype=torch.bfloat16, device_map=device)
    model.eval()
    return model, processor


# ── prompts ─────────────────────────────────────────────────────────────────
RATING_QUESTION = (
    "Rate how difficult this scene is for an autonomous vehicle to drive "
    "(0 = trivial empty road, 9 = extreme edge case).")
STEER = ("The overall driving difficulty of this scene "
         "on a scale of 0 to 9 is ")

REASON_QUESTION = (
    "In one sentence, identify the factors that make this scene easy or hard for an "
    "autonomous vehicle to drive (agents/pedestrians/cyclists, work zones, "
    "visibility, maneuvers).")
REASON_STEER = " Therefore, on a scale of 0 to 9, the overall driving difficulty is "


@functools.lru_cache(maxsize=16)
def _digit_ids(tok, steer):
    """Token ids for '0'..'9' AS THEY APPEAR after `steer`.

    Tokenizing the digit in isolation is family-dependent and wrong for at least one
    of these backends (SentencePiece merges a leading space into the digit token,
    byte-level BPE does not). Encoding `steer + digit` and taking the last token is
    correct for both. Distinctness is asserted, so a family whose tokenizer breaks
    this assumption fails loudly instead of silently scoring noise.
    """
    ids = [tok.encode(steer + str(k), add_special_tokens=False)[-1] for k in range(10)]
    if len(set(ids)) != 10:
        raise RuntimeError(f"digit tokens not distinct for {MODEL_ID}: {ids}")
    return ids


def _build(processor, frames, question, steer, blank=False):
    """Render the chat template with thinking disabled, then append the steer prefix.

    enable_thinking=False makes the template emit a closed `<think></think>`, so the
    next token after `steer` is the answer digit itself.
    """
    imgs = [np.zeros_like(f) for f in frames] if blank else frames
    msgs = [{"role": "user", "content":
             [{"type": "image"} for _ in imgs] + [{"type": "text", "text": question}]}]
    text = processor.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    return processor(text=[text + steer], images=imgs, return_tensors="pt")


def _digit_ev(model, processor, inputs, steer):
    ids = _digit_ids(processor.tokenizer, steer)
    inputs = {k: (v.to(model.device) if hasattr(v, "to") else v) for k, v in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs).logits[0, -1, :].float()
    p = F.softmax(logits[ids], dim=0)
    ev = float((p * torch.arange(10, device=p.device, dtype=p.dtype)).sum()) / 9.0
    return ev, float(p.max())


def score_clip(model, processor, clip_id, frames=None, blank=False):
    """Cold logit-EV difficulty in [0,1]. blank=True zeros the frames (neg control)."""
    frames = load_frames(clip_id) if frames is None else frames
    inputs = _build(processor, frames, RATING_QUESTION, STEER, blank=blank)
    ev, pmax = _digit_ev(model, processor, inputs, STEER)
    return {"clip_id": clip_id, "score": ev, "digit_pmax": pmax}


def reasoned_score(model, processor, clip_id, frames=None, blank=False, max_new_tokens=96):
    """Reason-then-rate: free-text rationale, then logit-EV conditioned on it.

    On Alpamayo this path was the one that moved the needle (0.437 cold -> 0.565
    reasoned -> 0.604 CoC), so it is the like-for-like comparison to carry over.
    Greedy decoding keeps it deterministic.
    """
    frames = load_frames(clip_id) if frames is None else frames
    inputs = _build(processor, frames, REASON_QUESTION, "", blank=blank)
    inputs = {k: (v.to(model.device) if hasattr(v, "to") else v) for k, v in inputs.items()}
    with torch.no_grad():
        out = model.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens)
    rationale = processor.tokenizer.decode(
        out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    steer2 = rationale + REASON_STEER
    inputs2 = _build(processor, frames, REASON_QUESTION, steer2, blank=blank)
    ev, pmax = _digit_ev(model, processor, inputs2, steer2)
    return {"clip_id": clip_id, "score": ev, "digit_pmax": pmax, "rationale": rationale}


if __name__ == "__main__":
    import sys
    m, p = load_model()
    for cid in sys.argv[1:] or list(clip_index())[:3]:
        print(cid, score_clip(m, p, cid))
