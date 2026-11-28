#!/usr/bin/env bash
# R-A prep: copy up to 4994 radar+egomotion parquets per sensor from NFS to local SSD
# (COPY, not symlink, so the bench measures NVMe-latency registration). ~53 GB.
set -eo pipefail
SRC=netai-e2e/nvidia-physicalai-av-subset
DST=/tmp/nvidia-extract/scale_50
MAX=4994
mkdir -p "$DST"
stage_sensor(){ # $1=label $2=glob-root
  local out="$DST/$1"; mkdir -p "$out"
  local n=0
  for f in $(find "$2" -name '*.parquet' 2>/dev/null | sort | head -$MAX); do
    cp -n "$f" "$out/$(basename "$f")"; n=$((n+1))
  done
  echo "staged $1: $n files"
}
for s in "$SRC"/radar/*/; do
  name="radar_$(basename "$s")"; stage_sensor "$(basename "$s")" "$s"
done
stage_sensor egomotion "$SRC/labels/egomotion"
echo "TOTAL: $(du -sh $DST | cut -f1) in $(ls -d $DST/*/ | wc -l) sensor dirs"
