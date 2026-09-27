#!/usr/bin/env python3
"""Create the hf-token Secret and the eval-harness ConfigMap, in one SSH session.

kubectl runs on the control plane but the harness files are local, so
`kubectl create configmap --from-file` cannot see them; the ConfigMap is built here
and piped to a remote `apply` instead. Re-running is safe -- both objects are applied,
not created, so this is idempotent.

    python deploy/k8s/bootstrap.py
"""
from __future__ import annotations

import base64
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kctl import kubectl  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
NS = "netai-l40-test"

# Flat keys: ConfigMap keys cannot contain "/", and the eval pods put /harness on
# PYTHONPATH, so `import adapters` resolves regardless of the original subdirectory.
SOURCES = sorted(glob.glob(os.path.join(ROOT, "evaluation", "*.py"))) + [
    os.path.join(ROOT, "evaluation", "cluster", "fetch_slice_local.py"),
    os.path.join(ROOT, "evaluation", "cluster", "slice_manifest_burnin16.json"),
]


def hf_token() -> str:
    for name in ("cluster.env", "hf.env"):
        p = os.path.join(ROOT, ".secrets", name)
        if not os.path.exists(p):
            continue
        for line in open(p):
            line = line.strip()
            if line.startswith("HF_TOKEN=") and not line.startswith("#"):
                tok = line.split("=", 1)[1].strip().strip('"').strip("'")
                if tok:
                    return tok
    raise SystemExit("no HF_TOKEN in .secrets/cluster.env or .secrets/hf.env")


def main() -> int:
    data, total = {}, 0
    for p in SOURCES:
        if not os.path.exists(p):
            print(f"  SKIP (missing) {p}")
            continue
        body = open(p, encoding="utf-8").read()
        data[os.path.basename(p)] = body
        total += len(body.encode())
    if total > 1_000_000:
        raise SystemExit(f"ConfigMap would be {total} bytes, over the 1 MB cap")
    print(f"eval-harness: {len(data)} keys, {total/1024:.0f} KiB")

    cm = {"apiVersion": "v1", "kind": "ConfigMap",
          "metadata": {"name": "eval-harness", "namespace": NS}, "data": data}
    rc, out, err = kubectl(f"apply -n {NS} -f -", stdin=json.dumps(cm))
    print((out or err).strip())
    if rc != 0:
        return rc

    sec = {"apiVersion": "v1", "kind": "Secret", "type": "Opaque",
           "metadata": {"name": "hf-token", "namespace": NS},
           "data": {"HF_TOKEN": base64.b64encode(hf_token().encode()).decode()}}
    rc, out, err = kubectl(f"apply -n {NS} -f -", stdin=json.dumps(sec))
    print((out or err).strip())
    return rc


if __name__ == "__main__":
    sys.exit(main())
