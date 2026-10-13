#!/usr/bin/env bash
# alpasim_bundle.sh — make one of our NuRec USDZ exports loadable by AlpaSim.
#
#   alpasim_bundle.sh <artifact.usdz> <out.usdz> (--offset-us N | --from-reference <ref.usdz>) [--add <file> ...]
#
# Our NCore stores are clip-relative, so the export sits at absoluteTimeOffsetMicroSec=0
# with track samples before t=0; AlpaSim casts track timestamps to uint64 and wants 0.5 s
# of force-GT history before the scene start (`force_gt_duration_us`), so a zero-based
# scene cannot run at all. Three steps, all pure re-basing (nothing is dropped):
#   1. usdz_tools.py shift-time   JSON / metadata.yaml / USDA offsets onto the new base
#   2. shift_checkpoint.py        the checkpoint's stored time ranges, same delta (torch,
#                                 run inside the renderer image with its own interpreter)
#   3. usdz_tools.py add          replace checkpoint.ckpt; add loose files (--add, e.g. the
#                                 post-hoc ground mesh)
# --from-reference takes the base from NVIDIA's bundle of the same clip so an A/B shares
# one time axis; without a reference any base >= 1e6 us works (there is no canonical
# absolute time for PAI clips: the converter re-bases to the egomotion origin).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
NRE=${NRE_IMAGE:-nvcr.io/nvidia/nre/nre-ga:26.04}
IN=$1; OUT=$2; shift 2
ARGS=(); ADD=()
while [ $# -gt 0 ]; do case $1 in --add) ADD+=("$(readlink -f "$2")"); shift 2;; *) ARGS+=("$1"); shift;; esac; done
W=$(mktemp -d "${TMPDIR:-/tmp}/alpasim_bundle.XXXXXX"); trap 'rm -rf "$W"' EXIT
offset(){ python3 -c "import sys, zipfile; sys.path.insert(0, '$HERE'); from usdz_tools import read_offset; print(read_offset(zipfile.ZipFile(sys.argv[1])))" "$1"; }
CUR=$(offset "$IN")
python3 "$HERE/usdz_tools.py" shift-time "$IN" "$W/shifted.usdz" "${ARGS[@]}"
NEW=$(offset "$W/shifted.usdz"); DELTA=$((NEW - CUR))
if [ "$DELTA" -ne 0 ]; then
  mkdir -p "$W/in" "$W/out"
  python3 - "$IN" "$W/in/checkpoint.ckpt" <<'PY'
import sys, zipfile, shutil
with zipfile.ZipFile(sys.argv[1]).open("checkpoint.ckpt") as src, open(sys.argv[2], "wb") as dst:
    shutil.copyfileobj(src, dst, 1 << 24)
PY
  docker run --rm --user "$(id -u):$(id -g)" --entrypoint bash \
    -v "$W:/w" -v "$HERE/shift_checkpoint.py:/w/shift_checkpoint.py:ro" "$NRE" -c '
    R=/app/internal/scripts/pycena/runtime/pycena_nrm_full.runfiles
    PY=$R/rules_python++python+python_3_11_x86_64-unknown-linux-gnu/bin/python3
    SP=$(find $R -maxdepth 4 -type d -name site-packages 2>/dev/null | tr "\n" ":")
    PYTHONPATH=$SP HOME=/tmp $PY /w/shift_checkpoint.py /w/in/checkpoint.ckpt /w/out/checkpoint.ckpt --delta-us '"$DELTA"' 2>&1 | grep -v -i warning'
  [ -s "$W/out/checkpoint.ckpt" ] || { echo "checkpoint shift produced no file" >&2; exit 1; }
  ADD=("$W/out/checkpoint.ckpt" "${ADD[@]}")
fi
if [ ${#ADD[@]} -gt 0 ]; then
  python3 "$HERE/usdz_tools.py" add "$W/shifted.usdz" "$OUT" "${ADD[@]}"
else
  mv "$W/shifted.usdz" "$OUT"
fi
echo "$OUT: time base $CUR -> $NEW (delta $DELTA), checkpoint $([ "$DELTA" -ne 0 ] && echo shifted || echo untouched)"
