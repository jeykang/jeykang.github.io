#!/usr/bin/env python3
"""VaVAM (valeoai VideoActionModel) as an evaluation policy.

The reason to run this is not its score — it is trained on OpenDV/nuPlan/nuScenes and
scored here on NVIDIA PhysicalAI, so a domain gap is expected and the number is hard
to read. The reason is that VaVAM is an **out-of-lab, out-of-architecture** policy:
an autoregressive video GPT plus a diffusion action expert, from a different group,
consuming a different input representation. Getting it to run through the same
`Policy` contract as Alpamayo is direct evidence that the harness is transferable
rather than an Alpamayo-shaped wrapper.

Pipeline (all details verified against the upstream repo, not guessed):
  frames -> resize by 3.75 -> 512x288 -> scale to [-1,1] -> VQ encoder (TorchScript,
  ds16/16384) -> (8, 18, 32) int64 tokens -> VaVAM -> 6 waypoints at 2 Hz.

  * resize_factor 3.75 is VaVAM's own **nuplan** preset, and nuPlan frames are
    1920x1080 — exactly this dataset's resolution, so no geometry is invented.
  * normalisation `2.0*x - 1.0` is taken from vam/datalib/token_creator.py.
  * commands are RIGHT=0 / LEFT=1 / STRAIGHT=2; with no route available this sends
    STRAIGHT, the same assumption the DiffusionDrive integration makes.

HORIZON: the action expert emits exactly 6 steps at 2 Hz = 3.0 s. Run the harness at
EVAL_HORIZON_S=3.0 for this policy (and for anything compared against it) rather than
extrapolating the missing second.
"""
from __future__ import annotations

import os
import pathlib
import sys
from typing import List, Optional, Tuple

# Assets live next to this file under `.vavam/` (repo checkout + checkpoints).
# The originals were on the A100 cluster, which is no longer reachable; env vars
# still override for anyone running elsewhere.
_VAM_HOME = os.environ.get("VAM_HOME", str(pathlib.Path(__file__).resolve().parent / ".vavam"))
VAM_REPO = os.environ.get("VAM_REPO", f"{_VAM_HOME}/repo")
VAM_CKPT = os.environ.get("VAM_CKPT", f"{_VAM_HOME}/ckpt/VAM_width_1024_pretrained_139k.pt")
VAM_ENC = os.environ.get("VAM_ENC", f"{_VAM_HOME}/ckpt/VQ_ds16_16384_llamagen_encoder.jit")

CAMERA = "camera_front_wide_120fov"
N_FRAMES = 8              # VaVAM's sequence_length
FRAME_DT_US = 500_000     # 2 Hz, matching the action expert's output rate
RESIZE_FACTOR = 3.75      # nuplan preset: 1920x1080 -> 512x288
COMMAND_STRAIGHT = 2


class VaVAMPolicy:
    name = "vavam"
    is_oracle = False
    needs_scenario = False
    needs_sensors = True

    def __init__(self, ckpt: str = VAM_CKPT, enc: str = VAM_ENC, device: str = "cuda"):
        import torch
        if VAM_REPO not in sys.path:
            sys.path.insert(0, VAM_REPO)
        from vam.action_expert import load_inference_VAM
        self._torch = torch
        self.device = device

        # torch>=2.6 defaults torch.load(weights_only=True); this checkpoint embeds
        # omegaconf config objects, so the strict unpickler rejects it. The upstream
        # loader does not expose the flag, so relax it only for the duration of this
        # call. Safe here: the file is the official valeoai v1.0.0 release artifact we
        # downloaded ourselves. Enumerating safe globals instead cascades through
        # omegaconf internals (ListConfig, DictConfig, ContainerMetadata, ...).
        _orig_load = torch.load

        def _load_permissive(*a, **k):
            k["weights_only"] = False
            return _orig_load(*a, **k)

        torch.load = _load_permissive
        try:
            self.vam = load_inference_VAM(ckpt, device)
        finally:
            torch.load = _orig_load

        self.enc = torch.jit.load(enc).to(device).eval()

    def _tokens(self, obs):
        """8 history frames -> VQ token grid, exactly as the upstream preprocessing."""
        import numpy as np
        import torch
        import torchvision.transforms.v2.functional as TF

        want = [obs.t0_us - (N_FRAMES - 1 - i) * FRAME_DT_US for i in range(N_FRAMES)]
        want = [t for t in want if t <= obs.t0_us]
        fr = obs.sensors.frames(CAMERA, want)
        if len(fr) != N_FRAMES:
            return None
        x = torch.from_numpy(np.stack(fr)).permute(0, 3, 1, 2)          # (T,3,H,W) uint8
        h, w = x.shape[2], x.shape[3]
        x = TF.resize(x, (int(h / RESIZE_FACTOR), int(w / RESIZE_FACTOR)), antialias=True)
        x = x.to(torch.float32) / 255.0
        x = 2.0 * x - 1.0                                               # [-1,1]
        with torch.no_grad():
            return self.enc(x.to(self.device))                          # (T,h,w) int64

    def plan(self, obs) -> Optional[List[Tuple[float, float]]]:
        torch = self._torch
        if obs.sensors is None:
            return None
        toks = self._tokens(obs)
        if toks is None:
            return None
        vt = toks.unsqueeze(0)                                          # (1,T,h,w)
        cmd = torch.tensor([[COMMAND_STRAIGHT]], device=self.device, dtype=torch.long)
        # The action expert is a diffusion sampler: unseeded it returns a different
        # trajectory each call. Measured run-to-run spread on the same 55 clips was
        # ~0.03 MF-PDMS (0.632 cluster vs 0.660 local) — larger than the entire gap
        # between Alpamayo generations (0.005). Seed per decision so a score is
        # reproducible and differences mean something.
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            traj = self.vam(vt, cmd, torch.bfloat16)                    # (s,1,t,a)
        P = traj[0, 0].float().cpu().numpy()                            # (6,2) @ 2 Hz
        del traj
        torch.cuda.empty_cache()
        out = []
        for k in range(obs.n_steps):
            i = min(len(P) - 1, max(0, int(round((k + 1) * obs.dt_s / 0.5)) - 1))
            out.append((float(P[i][0]), float(P[i][1])))
        return out
