#!/usr/bin/env bash
# run_vavam.sh — one HUGSIM closed-loop episode driven by VaVAM (AlpaSim's VAM wrapper).
#
#   run_vavam.sh <scenario.yaml> [seed=0] [base=configs/pai_base_local_vavam.yaml]
#
# Two containers meet at the named pipes in the episode's output dir: the simulator
# (hugsim-dev:cu118 on the A10, via run_closed_loop.sh — its CUDA extensions are built for
# sm_86 only) and the policy (clients/vavam_client.py in alpasim-harness-base:0.134.0 on
# CLIENT_GPU, default 0 = the RTX 6000; as root, since the image's venv interpreter lives under
# /root). Extra client flags via VAVAM_ARGS (e.g. "--context 8"); CAMERA=<rig.yaml> renders another
# camera rig, TAG=<name> keeps such variants apart.
# Output: runs/pai_vavam_[<tag>_]s<seed>_ltf/<scene>_<mode>/ (eval.json, video.mp4, vavam_steps.jsonl).
set -euo pipefail
H=$(cd "$(dirname "$0")" && pwd)
A=$(cd "$H/../alpasim" && pwd)
SCEN=$(readlink -f "$1"); SEED=${2:-0}
BASE0=$(readlink -f "${3:-$H/configs/pai_base_local_vavam.yaml}")
scene=$(awk '/^scene_name:/{print $2}' "$SCEN"); mode=$(awk '/^mode:/{print $2}' "$SCEN")
prefix=$(awk '/^output_dir:/{print $2}' "$BASE0")

# one output dir per seed: closed_loop.py writes to <output_dir>ltf/<scene>_<mode>
mkdir -p "$H/runs/.cfg"
tag=${TAG:+${TAG}_}
BASE="$H/runs/.cfg/$(basename "$BASE0" .yaml)_${tag}s$SEED.yaml"
sed "s#^output_dir: .*#output_dir: ${prefix}${tag}s${SEED}_#" "$BASE0" > "$BASE"
OUT="${prefix}${tag}s${SEED}_ltf/${scene}_${mode}"
mkdir -p "$OUT"; rm -f "$OUT/obs_pipe" "$OUT/plan_pipe"; mkfifo "$OUT/obs_pipe" "$OUT/plan_pipe"

name="vavam-client-${scene}-${mode}-${tag}s$SEED"
docker rm -f "$name" >/dev/null 2>&1 || true
docker run -d --name "$name" --gpus "\"device=${CLIENT_GPU:-0}\"" \
  --entrypoint "" -v "$H:$H" -v "$A/repo/data/drivers:/mnt/drivers:ro" \
  alpasim-harness-base:0.134.0 \
  /repo/.venv/bin/python "$H/clients/vavam_client.py" "$OUT" --seed "$SEED" ${VAVAM_ARGS:-} >/dev/null

# a policy that dies mid-episode leaves the simulator blocked on plan_pipe: stop it as well
sim="hugsim-run-$(basename "$SCEN" .yaml)"
( code=$(docker wait "$name" 2>/dev/null || echo 1)
  if [ "$code" != 0 ]; then echo "[run_vavam] client exited $code, stopping $sim"; docker rm -f "$sim" >/dev/null 2>&1; fi ) &
watch=$!
"$H/run_closed_loop.sh" "$SCEN" ltf "$BASE" ${CAMERA:+"$(readlink -f "$CAMERA")"} || echo "[run_vavam] simulator exited $?"
kill "$watch" 2>/dev/null || true
docker logs "$name" > "$OUT/vavam_client.log" 2>&1 || true
docker rm -f "$name" >/dev/null 2>&1 || true
cat "$OUT/eval.json" 2>/dev/null || echo "[run_vavam] no eval.json in $OUT"
