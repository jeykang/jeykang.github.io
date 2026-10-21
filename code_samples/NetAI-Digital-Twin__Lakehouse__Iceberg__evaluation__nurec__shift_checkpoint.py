#!/usr/bin/env python3
"""shift_checkpoint.py — move a NuRec checkpoint's stored time base by a constant.

    shift_checkpoint.py <in.ckpt> <out.ckpt> --delta-us <N> [--dry-run]

Companion to `usdz_tools.py shift-time`. The renderer (nre/grpc/serve.py) feeds AlpaSim's
absolute microsecond timestamps to the model unchanged, and the model normalises them with
ranges it stores in the checkpoint (`*.timestamps_us_ranges` tensors, `timestamps_us_min/max`
in the time-embedding extra state), so a bundle whose JSON/USDA base moves must have its
checkpoint moved by the same delta. Every state-dict entry whose key, or extra-state dict
key, contains `timestamps_us` is shifted; nothing else is touched. Needs torch — run it
with the renderer image's interpreter (see nurec/README.md), not the host python.
"""
import argparse
import re

import torch

PAT = re.compile(r"timestamps?_us")


def shift(obj, delta, path, log):
    if isinstance(obj, torch.Tensor):
        if obj.dtype in (torch.int64, torch.int32, torch.float64):
            before = obj.flatten()[:4].tolist()
            out = (obj.to(torch.int64) + delta).to(obj.dtype) if not obj.dtype.is_floating_point else obj + delta
            log.append(f"{path}: {obj.dtype} {tuple(obj.shape)} {before} -> {out.flatten()[:4].tolist()}")
            return out
        log.append(f"{path}: SKIPPED tensor dtype {obj.dtype}")
        return obj
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, int):
        log.append(f"{path}: int {obj} -> {obj + delta}")
        return obj + delta
    if isinstance(obj, float):
        log.append(f"{path}: float {obj} -> {obj + delta}")
        return obj + delta
    if isinstance(obj, dict):
        return {k: (shift(v, delta, f"{path}/{k}", log) if PAT.search(str(k)) else walk(v, delta, f"{path}/{k}", log)) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        t = type(obj)
        return t(shift(v, delta, f"{path}[{i}]", log) for i, v in enumerate(obj))
    log.append(f"{path}: SKIPPED {type(obj).__name__}")
    return obj


def walk(obj, delta, path, log):
    """Descend into containers; only keys matching PAT get shifted (by `shift`)."""
    if isinstance(obj, dict):
        return {k: (shift(v, delta, f"{path}/{k}", log) if PAT.search(str(k)) else walk(v, delta, f"{path}/{k}", log)) for k, v in obj.items()}
    return obj


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inp"); ap.add_argument("out"); ap.add_argument("--delta-us", type=int, required=True)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    ck = torch.load(a.inp, map_location="cpu", weights_only=False)
    log: list[str] = []
    sd = ck["state_dict"]
    for k in list(sd):
        if PAT.search(k):
            sd[k] = shift(sd[k], a.delta_us, f"state_dict/{k}", log)
        elif k.endswith("_extra_state") and isinstance(sd[k], dict):
            sd[k] = walk(sd[k], a.delta_us, f"state_dict/{k}", log)
    for k in list(ck):
        if k != "state_dict" and isinstance(ck[k], dict):
            ck[k] = walk(ck[k], a.delta_us, k, log)
    print(f"{a.inp}: delta {a.delta_us:+d}, {len(log)} field(s):")
    for line in log:
        print("  " + line)
    if a.dry_run:
        return
    torch.save(ck, a.out)
    chk = torch.load(a.out, map_location="cpu", weights_only=False)["state_dict"]
    k0 = "model.gaussians_nodes.dynamic_rigids.time_embed.timestamps_us_ranges"
    print(f"{a.out}: written; {k0} min={int(chk[k0].min())} max={int(chk[k0].max())}" if k0 in chk else f"{a.out}: written")


if __name__ == "__main__":
    main()
