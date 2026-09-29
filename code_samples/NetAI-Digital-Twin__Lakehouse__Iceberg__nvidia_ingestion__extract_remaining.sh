#!/usr/bin/env bash
# Extract remaining (broken) zip archives by extracting to local /tmp first,
# then moving to NFS. This avoids NFS concurrent read+write contention
# that causes spurious "disk full" errors.
#
# Deletes zip after successful extraction + move.

set -euo pipefail

SRC="${1:-/home/netai/jeykang/NetAI-Digital-Twin/Lakehouse/Iceberg/netai-e2e/nvidia-physicalai-av-subset}"
PARALLEL="${EXTRACT_PARALLEL:-2}"
LOCAL_TMP="/tmp/extract_staging"

mkdir -p "$LOCAL_TMP"

echo "======================================================================"
echo "ZIP EXTRACTION (local staging) — remaining broken chunks"
echo "  Source: $SRC"
echo "  Parallel workers: $PARALLEL"
echo "  Local staging: $LOCAL_TMP"
echo "======================================================================"

extract_one() {
    local zip="$1"
    local dir base out_dir local_out
    dir="$(dirname "$zip")"
    base="$(basename "$zip" .zip)"
    out_dir="${dir}/${base}"

    # Skip if already extracted
    if [ -d "$out_dir" ] && [ "$(find "$out_dir" -maxdepth 1 -type f | head -1)" != "" ]; then
        echo "[SKIP] $base"
        return 0
    fi

    # Extract to local disk first
    local_out="$LOCAL_TMP/$base"
    rm -rf "$local_out" 2>/dev/null
    mkdir -p "$local_out"

    if unzip -q -o "$zip" -d "$local_out" 2>/dev/null; then
        local nfiles
        nfiles=$(find "$local_out" -type f | wc -l)

        # Move to NFS. If mv fails (e.g. quota exceeded), keep staging
        # and the zip so the extraction can be retried.
        rm -rf "$out_dir" 2>/dev/null
        if ! mv "$local_out" "$out_dir" 2>&1; then
            echo "[FAIL-MOVE] $base (staging kept at $local_out, zip kept)"
            return 1
        fi

        # Verify destination has the expected file count before deleting zip
        local moved
        moved=$(find "$out_dir" -type f 2>/dev/null | wc -l)
        if [ "$moved" -lt "$nfiles" ]; then
            echo "[FAIL-VERIFY] $base (staged=$nfiles, moved=$moved, zip kept)"
            return 1
        fi

        echo "[DONE] $base → $nfiles files"
        rm -f "$zip"
    else
        echo "[FAIL] $base"
        rm -rf "$local_out" 2>/dev/null
        return 1
    fi
}
export -f extract_one
export LOCAL_TMP

# Find only zips that don't have a corresponding extracted directory
ZIPS=""
while IFS= read -r zip; do
    base="$(basename "$zip" .zip)"
    dir="$(dirname "$zip")"
    out_dir="${dir}/${base}"
    if [ ! -d "$out_dir" ] || [ -z "$(find "$out_dir" -maxdepth 1 -type f 2>/dev/null | head -1)" ]; then
        ZIPS="$ZIPS
$zip"
    fi
done < <(find "$SRC" -name "*.zip" | sort)

TOTAL=$(echo "$ZIPS" | grep -c . || true)
echo "Found $TOTAL zips needing extraction"
echo ""

if [ "$TOTAL" -eq 0 ]; then
    echo "All zips already extracted!"
    exit 0
fi

T0=$(date +%s)
echo "$ZIPS" | grep . | xargs -P "$PARALLEL" -I{} bash -c 'extract_one "$@"' _ {}

T1=$(date +%s)
ELAPSED=$(( T1 - T0 ))
echo ""
echo "======================================================================"
echo "Extraction complete in ${ELAPSED}s ($(( ELAPSED / 60 )) min)"
echo "Remaining zips: $(find "$SRC" -name '*.zip' | wc -l)"
echo "======================================================================"

# Keep LOCAL_TMP so that any [FAIL-MOVE]/[FAIL-VERIFY] staging dirs remain
# recoverable. Remove only if empty.
rmdir "$LOCAL_TMP" 2>/dev/null || echo "Staging retained: $(ls "$LOCAL_TMP" 2>/dev/null | wc -l) dirs in $LOCAL_TMP"
