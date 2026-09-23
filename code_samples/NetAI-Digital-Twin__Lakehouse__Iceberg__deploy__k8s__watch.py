#!/usr/bin/env python3
"""Poll the eval Job over one persistent SSH session and print progress.

Each kctl.py invocation opens a fresh SSH connection (~10-20 s), which is far too slow
to watch a Job with. This holds one connection open instead.

    python deploy/k8s/watch.py [job-label] [minutes]
"""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kctl import _client  # noqa: E402

NS = "netai-l40-test"


def main() -> int:
    label = sys.argv[1] if len(sys.argv) > 1 else "l40-eval-r1-nodisk"
    minutes = float(sys.argv[2]) if len(sys.argv) > 2 else 45.0
    c = _client()

    def run(cmd: str) -> str:
        _, o, e = c.exec_command(cmd, timeout=120)
        return (o.read().decode() + e.read().decode()).strip()

    deadline = time.time() + minutes * 60
    last = ""
    while time.time() < deadline:
        pods = run(f"kubectl get pods -n {NS} -l app={label} --no-headers")
        node = run(
            "kubectl get --raw /api/v1/nodes/datax-l40-gpu/proxy/stats/summary 2>/dev/null | "
            "python3 -c \"import json,sys;d=json.load(sys.stdin)['node'];"
            "print('nodefs avail %.0f GiB'%(d['fs']['availableBytes']/2**30))\""
        )
        if pods != last:
            print(f"\n[{time.strftime('%H:%M:%S')}] {node}")
            print(pods, flush=True)
            last = pods
        if pods and all(w in pods for w in ("Completed",)) and "Running" not in pods:
            break
        if "Error" in pods or "CrashLoopBackOff" in pods:
            print("\n!! failure detected", flush=True)
            break
        time.sleep(30)

    print("\n===== shard 0 tail =====", flush=True)
    print(run(f"kubectl logs -n {NS} -l app={label} --tail=40 --prefix=false 2>&1 | tail -40"))
    c.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
