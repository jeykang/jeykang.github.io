#!/usr/bin/env bash
# Extract all zip archives in the Nvidia PhysicalAI dataset in-place.
# Since the zips are STORED (uncompressed), this is just a copy of bytes
# with no decompression overhead.  After extraction, parquet files are
# directly accessible on NFS without ratarmount FUSE mounts.
#
# Usage (inside spark-iceberg container, as uid 1000:1007):
#   bash nvidia_ingestion/extract_zips.sh [SOURCE_DIR]
#
# Default source: /mnt/netai-e2e/nvidia-physicalai-av-subset

set -euo pipefail

SRC="${1:-/mnt/netai-e2e/nvidia-physicalai-av-subset}"
LOG="/tmp/extract_zips_$(date +%Y%m%d_%H%M%S).log"
PARALLEL="${EXTRACT_PARALLEL:-4}"   # parallel unzip processes

echo "======================================================================"
echo "ZIP EXTRACTION — Nvidia PhysicalAI AV dataset"
echo "  Source: $SRC"
echo "  Parallel workers: $PARALLEL"
echo "  Log: $LOG"
echo "======================================================================"
echo ""

# Verify source exists
if [ ! -d "$SRC" ]; then
    echo "ERROR: Source directory not found: $SRC"
    exit 1
fi

# Find all zip files
ZIPS=$(find "$SRC" -name "*.zip" | sort)
TOTAL=$(echo "$ZIPS" | wc -l)
echo "Found $TOTAL zip archives to extract"
echo ""

# Check available space — each zip is stored uncompressed so we need ~same space again
USED_KB=$(du -sk "$SRC" 2>/dev/null | awk '{print $1}')
AVAIL_KB=$(df -k "$SRC" 2>/dev/null | awk 'NR==2{print $4}')
echo "Dataset size: $(( USED_KB / 1048576 )) GB"
echo "Available:    $(( AVAIL_KB / 1048576 )) GB"
echo ""
if [ "$AVAIL_KB" -lt "$USED_KB" ]; then
    echo "WARNING: Available space (${AVAIL_KB} KB) may be less than needed (${USED_KB} KB)"
    echo "         Since zips are STORED, extraction is approximately 1:1 in size."
    echo "         Press Ctrl+C within 10s to abort, or wait to proceed..."
    sleep 10
fi

extract_one() {
    local zip="$1"
    local dir
    dir="$(dirname "$zip")"
    local base
    base="$(basename "$zip" .zip)"
    local out_dir="${dir}/${base}"

    # Skip if already extracted
    if [ -d "$out_dir" ] && [ "$(ls -A "$out_dir" 2>/dev/null | wc -l)" -gt 0 ]; then
        echo "[SKIP] $zip (already extracted to $out_dir)"
        return 0
    fi

    mkdir -p "$out_dir"
    if unzip -q -o "$zip" -d "$out_dir" 2>>"$LOG"; then
        local nfiles
        nfiles=$(find "$out_dir" -name "*.parquet" | wc -l)
        echo "[DONE] $base → $nfiles parquet files"
        # Remove zip after successful extraction to free space
        rm -f "$zip"
    else
        echo "[FAIL] $zip — see $LOG"
        return 1
    fi
}
export -f extract_one
export LOG

T0=$(date +%s)
echo "$ZIPS" | xargs -P "$PARALLEL" -I{} bash -c 'extract_one "$@"' _ {}

T1=$(date +%s)
ELAPSED=$(( T1 - T0 ))
echo ""
echo "======================================================================"
echo "Extraction complete in ${ELAPSED}s ($(( ELAPSED / 60 )) min)"
echo "Remaining zips: $(find "$SRC" -name "*.zip" | wc -l)"
echo "Extracted parquets: $(find "$SRC" -name "*.parquet" | wc -l)"
echo "======================================================================"
