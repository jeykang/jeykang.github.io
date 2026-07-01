"""Thin SSH/SFTP helper for the A100 cluster (feature C: Cosmos augmentation).

Reads credentials from ../.secrets/cluster.env (untracked) — no secrets in code.
Usage:
    python cosmos_augmentation/cluster.py "<remote command>"     # run + print
    from cluster import run, put, get                            # programmatic
"""
import os, sys, paramiko

_SECRET = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".secrets", "cluster.env")


def load_env(path=_SECRET):
    env = {}
    for line in open(path):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            env[k] = v.strip().strip("'\"")
    return env


def client():
    e = load_env()
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(e["CLUSTER_HOST"], port=int(e["CLUSTER_PORT"]), username=e["CLUSTER_USER"],
              password=e["CLUSTER_PASS"], timeout=25, banner_timeout=25, auth_timeout=25)
    return c


def run(cmd, timeout=180):
    c = client()
    try:
        _, o, er = c.exec_command(cmd, timeout=timeout)
        return o.read().decode("utf-8", "replace"), er.read().decode("utf-8", "replace")
    finally:
        c.close()


def put(local, remote):
    c = client()
    try:
        s = c.open_sftp(); s.put(local, remote); s.close()
    finally:
        c.close()


def get(remote, local):
    c = client()
    try:
        s = c.open_sftp(); s.get(remote, local); s.close()
    finally:
        c.close()


if __name__ == "__main__":
    out, err = run(sys.argv[1] if len(sys.argv) > 1 else "hostname")
    sys.stdout.write(out)
    if err.strip():
        sys.stderr.write("\n[stderr]\n" + err)
