"""Alpamayo-1.5 VLM driving-difficulty scorer (Phase-0 gate).

Uses the model's VQA path (`generate_text`) to rate per-clip driving difficulty
0-10 -> [0,1]. Detachable module, same spirit as planning/conflict_runner.py: a
candidate axis for the Gold difficulty union, subject to the validity battery.

Run inside the alpamayo1.5 uv venv (a1_5_venv); SDPA attention (no flash-attn on
this host). Needs HF auth + gated access to nvidia/Alpamayo-1.5-10B AND the
nvidia/PhysicalAI-Autonomous-Vehicles dataset (physical_ai_av streams frames).
"""
import re
import torch
import physical_ai_av
from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5
from alpamayo1_5.load_physical_aiavdataset import load_physical_aiavdataset
from alpamayo1_5 import helper

MODEL_ID = "nvidia/Alpamayo-1.5-10B"

DIFFICULTY_PROMPT = (
    "Assess how difficult this scene is for an autonomous vehicle to DRIVE. "
    "Weigh agent interactions (pedestrians, cyclists, vehicles, cut-ins, dense traffic), "
    "work zones / unusual road layouts, visibility and lighting, and maneuver complexity. "
    "Reply with ONLY a one-line JSON object: "
    '{"difficulty": <integer 0-10>, "reason": "<one short clause>"} '
    "where 0 = trivial empty road and 10 = extreme edge case. No other text."
)

_AVDI = None
def _avdi():
    global _AVDI
    if _AVDI is None:
        _AVDI = physical_ai_av.PhysicalAIAVDatasetInterface()
    return _AVDI


def load_model():
    """Load Alpamayo-1.5 on cuda (bf16, SDPA). ~24 GB VRAM single-sample."""
    model = Alpamayo1_5.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda")
    model.eval()
    processor = helper.get_processor(model.tokenizer)
    return model, processor


def load_frames(clip_id, camera=None):
    avdi = _avdi()
    cam = camera or avdi.features.CAMERA.CAMERA_FRONT_WIDE_120FOV
    return load_physical_aiavdataset(clip_id, camera_features=[cam])


# Steered answer prefix: the model continues with a difficulty digit, so we read
# the next-token distribution over 0..9 instead of parsing free text. This makes
# the score continuous, parse-failure-free, and DETERMINISTIC (single forward pass,
# no sampling) — sidestepping the VLM format + nondeterminism problems.
RATING_QUESTION = (
    "Rate how difficult this scene is for an autonomous vehicle to drive "
    "(0 = trivial empty road, 9 = extreme edge case).")
STEER = ("<|answer_start|>The overall driving difficulty of this scene "
         "on a scale of 0 to 9 is ")

import torch.nn.functional as F  # noqa: E402

_DIGIT_IDS = None
def _digit_ids(tokenizer):
    global _DIGIT_IDS
    if _DIGIT_IDS is None:
        _DIGIT_IDS = [tokenizer.encode(str(k), add_special_tokens=False)[0]
                      for k in range(10)]
    return _DIGIT_IDS


def score_clip(model, processor, clip_id, blank=False, data=None, **_):
    """Logit expected-value difficulty in [0,1] from a single forward pass.

    score = E[digit] / 9 over softmax(next-token logits restricted to '0'..'9').
    blank=True zeros the frames (negative control). Deterministic.
    """
    if data is None:
        data = load_frames(clip_id)
    frames = data["image_frames"].flatten(0, 1)
    if blank:
        frames = torch.zeros_like(frames)
    msg = helper.create_vqa_message(frames, question=RATING_QUESTION,
                                    camera_indices=data["camera_indices"])
    msg[-1]["content"][0]["text"] = STEER
    inputs = processor.apply_chat_template(
        msg, tokenize=True, add_generation_prompt=False,
        continue_final_message=True, return_dict=True, return_tensors="pt")
    inputs = helper.to_device(inputs, "cuda")
    digit_ids = _digit_ids(model.tokenizer)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model.vlm(**inputs).logits[0, -1, :].float()
    p = F.softmax(logits[digit_ids], dim=0)
    ev = float((p * torch.arange(10, device=p.device)).sum()) / 9.0
    return {"clip_id": clip_id, "score": ev, "digit_pmax": float(p.max())}


def reason(model, processor, clip_id, data=None, temperature=0.1, max_len=128):
    """Optional free-text rationale (the model identifies hazards well in prose)."""
    if data is None:
        data = load_frames(clip_id)
    msg = helper.create_vqa_message(data["image_frames"].flatten(0, 1),
                                    question=DIFFICULTY_PROMPT,
                                    camera_indices=data["camera_indices"])
    inp = processor.apply_chat_template(
        msg, tokenize=True, add_generation_prompt=False,
        continue_final_message=True, return_dict=True, return_tensors="pt")
    inp = helper.to_device({"tokenized_data": inp}, "cuda")
    torch.cuda.manual_seed_all(0)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        extra = model.generate_text(data=inp, temperature=temperature,
                                    num_samples=1, max_generation_length=max_len)
    return str(extra["answer"][0][0])
