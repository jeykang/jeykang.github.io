#!/usr/bin/env bash
# Re-mount the NFS shares the NVIDIA/Argoverse pipeline depends on.
# These do not survive a server reboot (intentionally not in /etc/fstab so a
# dead NFS server can't hang boot). Run this after a reboot.
#
#   ./mount_netai_e2e.sh          # mount both
#   ./mount_netai_e2e.sh --status # show current mount state only
set -u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# target_dir  server:export
MOUNTS=(
  "netai-e2e|10.38.36.221:/exports/test/shared/netai-e2e"
  "netai-e2e-orig|10.38.36.222:/exports/datax/shared/netai-e2e"
)
OPTS="nfsvers=4.1"

status() {
  for m in "${MOUNTS[@]}"; do
    d="${m%%|*}"
    if mountpoint -q "$HERE/$d"; then
      echo "  [mounted] $d"
    else
      echo "  [ABSENT ] $d"
    fi
  done
}

if [[ "${1:-}" == "--status" ]]; then
  echo "NFS mount status:"; status; exit 0
fi

for m in "${MOUNTS[@]}"; do
  d="${m%%|*}"; src="${m##*|}"
  target="$HERE/$d"
  if mountpoint -q "$target"; then
    echo "[skip] $d already mounted"
    continue
  fi
  mkdir -p "$target"
  echo "[mount] $src -> $d"
  if ! sudo mount -t nfs -o "$OPTS" "$src" "$target"; then
    echo "[warn] first attempt failed, retrying once..." >&2
    sudo mount -t nfs -o "$OPTS" "$src" "$target" \
      || echo "[FAIL] could not mount $d" >&2
  fi
done

echo "Final status:"; status
