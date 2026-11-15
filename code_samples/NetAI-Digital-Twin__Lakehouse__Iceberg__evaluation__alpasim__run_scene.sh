#!/usr/bin/env bash
# Run AlpaSim on one or more NuRec scenes with a harness Policy.
#
#   ./run_scene.sh <policy-spec> <scene_id> [scene_id ...]
#   ./run_scene.sh constant_velocity clipgt-01d503d4-449b-46fc-8d78-9085e70d3554
#
# EXTRA_ARGS="key=value ..." appends further wizard/hydra overrides, word-split, e.g.
#   EXTRA_ARGS="runtime.simulation_config.route_generator_type=RECORDED" for a map-less
#   twin (no map.xodr: the default MAP route generator fails on it).
# LOCAL_USDZ_DIR=<dir> runs every *.usdz found in <dir> (recursively) instead of
# catalog scene_ids -- AlpaSim's `local` suite, for scenes we produced ourselves.
# The directory becomes the scene cache bind-mounted into the containers, so files
# that live elsewhere must be hardlinked in, not symlinked. No scene_id arguments
# are needed in this mode.
#
# DRIVER selects the AlpaSim driver config (default `harness`, our plugin). Set
# DRIVER=vavam to run a native AlpaSim driver instead; the policy-spec argument is
# then ignored, since the checkpoint comes from that driver's own config.
#
# Every flag here was required to get a rollout to run; the comments say why so they
# are not dropped by accident.
set -eo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DRIVER="${DRIVER:-harness}"
POLICY="${1:?usage: run_scene.sh <policy-spec> <scene_id> [...]}"; shift
[ $# -gt 0 ] || [ -n "${LOCAL_USDZ_DIR:-}" ] || { echo "need at least one scene_id (form: clipgt-<clip-uuid>) or LOCAL_USDZ_DIR"; exit 2; }
SCENES="$(IFS=,; echo "$*")"
# 5. scene selection: catalog ids, or a local directory. scene_ids must be nulled
#    explicitly in local mode because base_config.yaml carries a default scene_id,
#    and the wizard resolves scene_ids before it falls back to the local suite.
SCENE_ARGS=("scenes.scene_ids=[$SCENES]")
if [ -n "${LOCAL_USDZ_DIR:-}" ]; then
  [ $# -eq 0 ] || { echo "LOCAL_USDZ_DIR and scene_id arguments are mutually exclusive"; exit 2; }
  SCENE_ARGS=("scenes.local_usdz_dir=$(cd "$LOCAL_USDZ_DIR" && pwd)" "scenes.scene_ids=null")
fi
RUN_DIR="${RUN_DIR:-$HERE/runs/$(date +%Y%m%d-%H%M%S)}"

# 1. Orphaned compose stacks from an interrupted run stay up and exhaust host RAM;
#    the renderer of the NEXT run then dies with exit 137 (SIGKILL/OOM). Always clean.
# `|| true` matters: with `set -e`, grep finding nothing (the normal, clean case)
# would otherwise abort the script before it ever runs.
{ docker ps --format '{{.Names}}' | grep -E '^alpasim' || true; } | sed 's/-[a-z]*-0-1$//' | sort -u |
  while read -r p; do docker ps -q --filter "name=^${p}-" | xargs -r docker rm -f >/dev/null 2>&1 || true; done
# 1b. Every run creates a compose network; finished runs leave theirs behind and after
#     ~25 runs Docker refuses ("all predefined address pools have been fully subnetted").
#     Prune only touches networks with no container attached.
docker network prune -f >/dev/null 2>&1 || true

mkdir -p "$RUN_DIR"
cd "$HERE/repo"
. "$HOME/.cargo/env" 2>/dev/null || true
export HF_TOKEN="${HF_TOKEN:-$(cat ~/.cache/huggingface/token 2>/dev/null)}"
[ -n "$HF_TOKEN" ] || { echo "HF_TOKEN required (gated NuRec dataset)"; exit 3; }

# 2. defines.base_image  -> derived image that has the harness plugin INSTALLED.
#    Mounting plugins/ is not enough: entry points need an installed distribution.
# 3. services.*.gpus=[1] -> this host pairs a Quadro RTX 6000 (sm_75) at index 0 with
#    an A10 (sm_86) at index 1. AlpaSim's topology hardcodes GPU 0, and NuRec kernels
#    need Ampere+, so GPU 0 fails with "no kernel image is available".
# 4. HARNESS_POLICY travels via model.checkpoint_path: the generated compose passes
#    NO environment to the driver container.
# Only the harness driver takes a policy spec; native drivers carry their own checkpoint.
# Plain `[ ... ] && ...` would abort under `set -e` for any non-harness driver.
CKPT_ARG=()
if [ "$DRIVER" = harness ]; then
  CKPT_ARG=(driver.model.checkpoint_path="$POLICY")
fi

exec uv run alpasim_wizard \
  deploy=local topology=1gpu "driver=$DRIVER" \
  defines.base_image=alpasim-harness-base:0.134.0 \
  services.renderer.gpus="[1]" services.driver.gpus="[1]" \
  services.physics.gpus="[1]" services.trafficsim.gpus="[1]" \
  "${CKPT_ARG[@]}" \
  "${SCENE_ARGS[@]}" \
  wizard.log_dir="$RUN_DIR" \
  runtime.simulation_config.n_rollouts="${N_ROLLOUTS:-1}" \
  ${EXTRA_ARGS:-}
