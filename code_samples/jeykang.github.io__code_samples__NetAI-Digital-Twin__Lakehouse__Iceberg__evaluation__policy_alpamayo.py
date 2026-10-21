#!/usr/bin/env python3
"""Alpamayo VLA driving models as evaluation policies — real models through the contract.

Two variants share every line of input construction and differ only in checkpoint and
message format:
    AlpamayoPolicy    nvidia/Alpamayo-1.5-10B  (newer; supersedes R1 per NVlabs/alpamayo)
    AlpamayoR1Policy  nvidia/Alpamayo-R1-10B   (the 1.0 release, no RL post-training)
Running both checks the metric as well as the models: NVIDIA states 1.5 supersedes
R1, so a metric that ranked them the other way round would be suspect.

The point of this file is to prove the Tier 1 contract admits an actual
vision-language-action driving model, not just trajectory heuristics. It therefore
builds the model's inputs **from the harness `Observation` only** — ego history from
`obs.ego_history`, images from `obs.sensors` — rather than calling Alpamayo's own
`load_physical_aiavdataset`. Using the model's loader would have been three lines,
but it fetches by clip_id with its own notion of time and would have validated
nothing about the harness (and could silently read past the decision point).

Why 1.5 and not R1: Alpamayo-R1-10B states a 24 GB VRAM minimum. This host has a
23 GB A10 (Ampere, usable) and a 24 GB Quadro RTX 6000 (Turing — no BF16), so R1
needs the A100 cluster plus the clip data staged there. Alpamayo-1.5-10B is the same
family and task, already resident locally (~21 GB), and runs here. See RESULTS.md.

Run inside the vendored venv:
    planning/alpamayo/alpamayo1.5/a1_5_venv/bin/python run_eval.py \
        --policy policy_alpamayo:AlpamayoPolicy --limit 60 --workers 1
"""
from __future__ import annotations

import math
import os
from typing import List, Tuple

MODEL_ID = os.environ.get("ALPAMAYO_MODEL", "nvidia/Alpamayo-1.5-10B")
# Alpamayo's own ordering: cross_left(0), front_wide(1), cross_right(2), front_tele(6),
# sorted by camera index. Must match training or the per-image name tags are wrong.
CAMERAS = (("camera_cross_left_120fov", 0), ("camera_front_wide_120fov", 1),
           ("camera_cross_right_120fov", 2), ("camera_front_tele_30fov", 6))
N_HISTORY = 16            # 1.6 s at 10 Hz, ending at t0
HISTORY_DT_US = 100_000
# Frames per camera. 4 is the trained configuration and it matters: at 1 frame the
# model cannot infer motion from vision and mispredicts direction of travel
# (measured ADE 1.62 m -> 0.34 m on the same clip when raised to 4).
N_FRAMES = int(os.environ.get("ALPAMAYO_FRAMES", "4"))
# Canonical inference settings from the model's own test_inference.py. The June
# work in planning/alpamayo/ used max_generation_length=64 / temperature=0.1 to fit
# a 24 GB card; that truncates the chain-of-causation the action expert conditions
# on, and it produced trajectories pointing BACKWARDS on low-speed clips here.
# Do not lower these to save VRAM without re-checking ADE against ground truth.
TEMPERATURE = float(os.environ.get("ALPAMAYO_TEMP", "0.6"))
TOP_P = float(os.environ.get("ALPAMAYO_TOP_P", "0.98"))
MAX_GEN = int(os.environ.get("ALPAMAYO_MAX_GEN", "256"))


def _pick_device():
    import torch
    best, best_free = None, -1
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        if p.major < 8:                     # needs BF16 -> Ampere or newer
            continue
        free = torch.cuda.mem_get_info(i)[0]
        if free > best_free:
            best, best_free = i, free
    if best is None:
        raise RuntimeError("no BF16-capable GPU (sm_80+) available")
    return f"cuda:{best}"


class _AlpamayoBase:
    """Vision-language-action planner. Reads only history-bounded observation data."""

    name = "alpamayo"
    is_oracle = False
    needs_scenario = False
    needs_sensors = True        # restricts decision times to camera coverage

    def _load(self, model_id, device):
        raise NotImplementedError

    def _message(self, frames, cam_idx):
        raise NotImplementedError

    def __init__(self, model_id: str | None = None, device: str | None = None):
        import torch
        self._torch = torch
        self.device = device or _pick_device()
        self._load(model_id or self.default_model, self.device)

    # ── input construction, entirely from the Observation ────────────────────
    def _ego_history(self, obs):
        """16 poses at 10 Hz ending at t0, expressed in the ego frame at t0."""
        import numpy as np
        import scipy.spatial.transform as spt

        want = [obs.t0_us - (N_HISTORY - 1 - i) * HISTORY_DT_US for i in range(N_HISTORY)]
        hist = obs.ego_history
        picked = [min(hist, key=lambda e: abs(e.t_us - t)) for t in want]

        xyz = np.array([[e.x, e.y, e.z] for e in picked], dtype=np.float64)
        quat = np.array([[e.qx, e.qy, e.qz, e.qw] for e in picked], dtype=np.float64)
        r0 = spt.Rotation.from_quat(quat[-1]).inv()          # pose at t0
        xyz_local = r0.apply(xyz - xyz[-1])
        rot_local = (r0 * spt.Rotation.from_quat(quat)).as_matrix()

        t = self._torch
        return (t.from_numpy(xyz_local).float()[None, None],
                t.from_numpy(rot_local).float()[None, None])

    def _images(self, obs):
        """(N_cam*N_frames, 3, H, W) uint8 + camera indices, newest frame last."""
        import numpy as np
        t = self._torch
        if obs.sensors is None:
            return None, None
        want = [obs.t0_us - (N_FRAMES - 1 - i) * HISTORY_DT_US for i in range(N_FRAMES)]
        stacks, idxs = [], []
        for cam, cam_idx in CAMERAS:
            fr = obs.sensors.frames(cam, want)     # raises if any t > t0
            if len(fr) != len(want):
                return None, None
            arr = np.stack(fr)                     # (T,H,W,3)
            stacks.append(t.from_numpy(arr).permute(0, 3, 1, 2))
            idxs.append(cam_idx)
        order = sorted(range(len(idxs)), key=lambda i: idxs[i])
        frames = t.stack([stacks[i] for i in order])          # (N_cam,T,3,H,W)
        return frames.flatten(0, 1), t.tensor([idxs[i] for i in order], dtype=t.int64)

    # ── policy interface ─────────────────────────────────────────────────────
    def plan(self, obs) -> List[Tuple[float, float]] | None:
        t = self._torch
        frames, cam_idx = self._images(obs)
        if frames is None:
            return None
        xyz, rot = self._ego_history(obs)

        msg = self._message(frames, cam_idx)
        inp = self.processor.apply_chat_template(
            msg, tokenize=True, add_generation_prompt=False,
            continue_final_message=True, return_dict=True, return_tensors="pt")
        data = self._helper.to_device(
            {"tokenized_data": inp, "ego_history_xyz": xyz, "ego_history_rot": rot},
            self.device)

        t.cuda.manual_seed_all(0)
        with t.no_grad(), t.autocast("cuda", dtype=t.bfloat16):
            pred, _, _ = self.model.sample_trajectories_from_data_with_vlm_rollout(
                data=data, top_p=TOP_P, temperature=TEMPERATURE, num_traj_samples=1,
                max_generation_length=MAX_GEN, return_extra=True)
        P = pred[0, 0, 0, :, :2].float().cpu().numpy()        # (64,2) @10Hz, ego frame at t0
        del pred, data
        t.cuda.empty_cache()

        # Resample the model's 10 Hz output onto the harness's decision grid.
        out = []
        for k in range(obs.n_steps):
            i = int(round((k + 1) * obs.dt_s / 0.1)) - 1
            i = max(0, min(len(P) - 1, i))
            out.append((float(P[i, 0]), float(P[i, 1])))
        return out


class AlpamayoPolicy(_AlpamayoBase):
    """Alpamayo-1.5-10B — the current model in the family."""

    name = "alpamayo_1_5"
    default_model = MODEL_ID

    def _load(self, model_id, device):
        from alpamayo1_5 import helper
        from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5
        t = self._torch
        self.model = Alpamayo1_5.from_pretrained(
            model_id, dtype=t.bfloat16, attn_implementation="sdpa").to(device)
        self.model.eval()
        self._helper = helper
        self.processor = helper.get_processor(self.model.tokenizer)

    def _message(self, frames, cam_idx):
        return self._helper.create_message(frames=frames, camera_indices=cam_idx,
                                           num_frames_per_camera=N_FRAMES)


class AlpamayoR1Policy(_AlpamayoBase):
    """Alpamayo-R1-10B — the 1.0 release. Its create_message takes frames only.

    Needs the NVlabs/alpamayo sources on the path; they are vendored (gitignored) at
    planning/alpamayo/alpamayo_r1/src. Deps are identical to the 1.5 venv (torch
    2.8.0, transformers 4.57.1), so no second environment is needed. flash-attn is
    the package default but is not installed here, hence attn_implementation="sdpa".
    """

    name = "alpamayo_r1"
    default_model = "nvidia/Alpamayo-R1-10B"

    def _load(self, model_id, device):
        import sys
        src = os.environ.get("ALPAMAYO_R1_SRC", os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "planning", "alpamayo",
            "alpamayo_r1", "src"))
        if src not in sys.path:
            sys.path.insert(0, src)
        from alpamayo_r1 import helper
        from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
        t = self._torch
        attn = os.environ.get("ALPAMAYO_R1_ATTN", "eager")
        self.model = AlpamayoR1.from_pretrained(
            model_id, dtype=t.bfloat16, attn_implementation=attn).to(device)
        self.model.eval()
        self._helper = helper
        self.processor = helper.get_processor(self.model.tokenizer)

    def _message(self, frames, cam_idx):
        return self._helper.create_message(frames)


# Alpamayo 2 Super validates that the SOURCE contains the full canonical camera ring
# in canonical order (validate_source_camera_ring), and only then selects the driving
# profile's six (ids 0,1,2,3,5,6 — rear_tele is excluded from driving but must still
# be PRESENT in the source). So all seven are supplied, in this exact order, from
# alpamayo2_super.common.constants.CAMERA_NAMES_TO_INDICES.
A2_CAMERAS = (("camera_cross_left_120fov", 0), ("camera_front_wide_120fov", 1),
              ("camera_cross_right_120fov", 2), ("camera_rear_left_70fov", 3),
              ("camera_rear_tele_30fov", 4), ("camera_rear_right_70fov", 5),
              ("camera_front_tele_30fov", 6))
A2_FRAMES = 4


class Alpamayo2Policy(_AlpamayoBase):
    """Alpamayo2-Super (34B) — 32B VLM backbone + 2.3B diffusion action decoder.

    Differs from 1.5/R1 in three ways that matter here:
      * six cameras, not four (see A2_CAMERAS);
      * inputs are assembled by the package itself — `select_task_input` slices a
        source dict by the task's input profile and `helper.prepare_model_inputs`
        builds the messages — so this policy produces that source dict rather than
        a chat message;
      * `sample_trajectories_from_data` (no `_with_vlm_rollout` suffix) and it
        returns four values, including a logprob.

    NVIDIA measured 72,115 MiB peak on one H100 80GB, so on 40 GB A100s it must be
    sharded: loaded with device_map="auto" across the GPUs the job requests (use 4).
    """

    name = "alpamayo2_super"
    default_model = "nvidia/Alpamayo2-Super"

    def _load(self, model_id, device):
        import sys
        src = os.environ.get("ALPAMAYO2_SRC")
        if src and src not in sys.path:
            sys.path.insert(0, src)
        from alpamayo2_super import helper
        from alpamayo2_super.models.alpamayo2_super import Alpamayo2Super
        t = self._torch
        # Sharding metadata: the Qwen3-VL backbone declares its own _no_split_modules,
        # but the Alpamayo2Super wrapper does not re-export them, so accelerate split
        # transformer blocks across devices and generation died with
        #   "tensors is on cuda:3, different from other tensors on cuda:0"
        # inside the KV cache concat. Declaring the block classes keeps each block
        # whole on one device. NVIDIA only ever tested this model on a single
        # H100-80GB, so multi-GPU is our own configuration, not a supported path.
        if not getattr(Alpamayo2Super, "_no_split_modules", None):
            Alpamayo2Super._no_split_modules = ["Qwen3VLTextDecoderLayer",
                                                "Qwen3VLVisionBlock"]
        # A2_QUANT=int8 loads the 34B in 8-bit (~34 GB) so it fits ONE A100-40GB.
        # That matters beyond memory: the released code assumes a single device (NVIDIA
        # tested on one H100-80GB), and multi-GPU sharding fails in the KV cache
        # ("tensors on cuda:3 vs cuda:0"). Quantising restores single-device semantics
        # rather than fighting the sharding. It CHANGES NUMERICS — report any result
        # from this path as Alpamayo2-Super int8, never as the released model.
        quant = os.environ.get("A2_QUANT", "").lower()
        kw = {}
        if quant in ("int8", "int4"):
            from transformers import BitsAndBytesConfig
            # Quantise the VLM backbone only. The 2.3B diffusion action expert is a
            # custom module and bitsandbytes left its output as int8, which blew up in
            # SiLU ("silu_cuda not implemented for 'Char'"). Skipping it keeps the
            # expert in bf16 (~4.6 GB) while the 32B backbone shrinks, which is also
            # the part that actually needs to fit.
            skip = [m for m in os.environ.get("A2_SKIP", "expert,lm_head").split(",") if m]
            if quant == "int8":
                cfg = BitsAndBytesConfig(load_in_8bit=True, llm_int8_skip_modules=skip)
            else:
                cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                         bnb_4bit_compute_dtype=t.bfloat16,
                                         llm_int8_skip_modules=skip)
            kw["quantization_config"] = cfg
            kw["device_map"] = {"": 0}
        else:
            kw["device_map"] = os.environ.get("A2_DEVICE_MAP", "auto")
        self.model = Alpamayo2Super.from_pretrained(model_id, dtype=t.bfloat16, **kw)
        self.model.eval()
        self._helper = helper
        self.processor = None

    def _message(self, frames, cam_idx):        # unused; A2 builds its own messages
        raise NotImplementedError

    def _source_data(self, obs):
        """The dict `select_task_input` expects, built only from the Observation."""
        import numpy as np
        t = self._torch
        want = [obs.t0_us - (A2_FRAMES - 1 - i) * HISTORY_DT_US for i in range(A2_FRAMES)]
        stacks, ids, abs_ts = [], [], []
        for cam, cam_id in A2_CAMERAS:
            fr = obs.sensors.frames(cam, want)
            if len(fr) != A2_FRAMES:
                return None
            stacks.append(t.from_numpy(np.stack(fr)).permute(0, 3, 1, 2))   # (T,3,H,W)
            abs_ts.append(obs.sensors.frame_times(cam, want))
            ids.append(cam_id)
        xyz, rot = self._ego_history(obs)
        return {
            "image_frames": t.stack(stacks),                       # (cam,frame,3,H,W)
            "camera_indices": t.tensor(ids, dtype=t.int64),
            "camera_names": [c for c, _ in A2_CAMERAS],   # ring validation needs these
            "absolute_timestamps": t.tensor(abs_ts, dtype=t.int64),
            "relative_timestamps": t.zeros(len(ids), A2_FRAMES),   # recomputed downstream
            "ego_history_xyz": xyz,
            "ego_history_rot": rot,
            "ego_t0": t.tensor([obs.t0_us], dtype=t.int64),
            # Index of t0 within the frame axis. select_input_profile overwrites the
            # value with len(frame_indices)-1 but reuses this tensor's dtype/device,
            # so it must be present and be a tensor.
            "ego_t0_frame_idx": t.tensor([A2_FRAMES - 1], dtype=t.int64),
        }

    def plan(self, obs):
        t = self._torch
        if obs.sensors is None:
            return None
        src = self._source_data(obs)
        if src is None:
            return None
        from alpamayo2_super.input_profiles import select_task_input
        data = select_task_input(src, "trajectory")
        mi = self._helper.prepare_model_inputs(data, self.model.config, self.model.tokenizer)
        mi = self._helper.to_device(mi, "cuda")
        t.cuda.manual_seed_all(0)
        with t.no_grad(), t.autocast("cuda", dtype=t.bfloat16):
            pred, _rot, _lp, _extra = self.model.sample_trajectories_from_data(
                data=mi, top_p=TOP_P, temperature=TEMPERATURE, num_traj_samples=1,
                diffusion_kwargs={"inference_step": int(os.environ.get("A2_STEPS", "10"))},
                return_extra=True)
        P = pred[0, 0, 0, :, :2].float().cpu().numpy()   # (64,2) @10Hz, ego frame at t0
        del pred
        t.cuda.empty_cache()
        out = []
        for k in range(obs.n_steps):
            i = max(0, min(len(P) - 1, int(round((k + 1) * obs.dt_s / 0.1)) - 1))
            out.append((float(P[i, 0]), float(P[i, 1])))
        return out
