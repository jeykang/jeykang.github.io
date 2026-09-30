#!/usr/bin/env bash
# Re-mount the NFS share the NVIDIA/Argoverse pipeline depends on.
# It does not survive a server reboot (intentionally not in /etc/fstab so a
# dead NFS server can't hang boot). Run this after a reboot.
#
#   ./mount_netai_e2e.sh          # mount
#   ./mount_netai_e2e.sh --status # show current mount state only
#
# 2026-09-21: the NFS server was migrated. The share now lives on 10.38.36.222
# (export /exports/test/shared/netai-e2e) and is mounted system-wide at
# /mnt/netai-e2e, then bind-mounted onto ./netai-e2e so every path in this repo
# (and the spark-iceberg container's /mnt/netai-e2e bind) keeps working.
# Access requires membership of the shared group GID 1007 (netaie2e).
# The old second share (netai-e2e-orig, 10.38.36.222:/exports/datax/...) no
# longer exists on the server and is not mounted.
set -u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVER="10.38.36.222"
EXPORT="/exports/test/shared/netai-e2e"
SYS_TARGET="/mnt/netai-e2e"
REPO_TARGET="$HERE/netai-e2e"
OPTS="vers=4.1,proto=tcp,rw,hard"

status() {
  for t in "$SYS_TARGET" "$REPO_TARGET"; do
    if mountpoint -q "$t"; then echo "  [mounted] $t"; else echo "  [ABSENT ] $t"; fi
  done
  if ! id -nG | tr ' ' '\n' | grep -qx netaie2e; then
    echo "  [warn] $(id -un) is not in group netaie2e (GID 1007); the share will be unreadable" >&2
  fi
}

if [[ "${1:-}" == "--status" ]]; then
  echo "NFS mount status:"; status; exit 0
fi

sudo mkdir -p "$SYS_TARGET"
if mountpoint -q "$SYS_TARGET"; then
  echo "[skip] $SYS_TARGET already mounted"
else
  echo "[mount] $SERVER:$EXPORT -> $SYS_TARGET"
  if ! sudo mount -t nfs -o "$OPTS" "$SERVER:$EXPORT" "$SYS_TARGET"; then
    echo "[warn] first attempt failed, retrying once..." >&2
    sudo mount -t nfs -o "$OPTS" "$SERVER:$EXPORT" "$SYS_TARGET" \
      || { echo "[FAIL] could not mount $SYS_TARGET" >&2; exit 1; }
  fi
fi

mkdir -p "$REPO_TARGET"
if mountpoint -q "$REPO_TARGET"; then
  echo "[skip] $REPO_TARGET already mounted"
else
  echo "[bind ] $SYS_TARGET -> $REPO_TARGET"
  sudo mount --bind "$SYS_TARGET" "$REPO_TARGET" || echo "[FAIL] bind mount failed" >&2
fi

echo "Final status:"; status
