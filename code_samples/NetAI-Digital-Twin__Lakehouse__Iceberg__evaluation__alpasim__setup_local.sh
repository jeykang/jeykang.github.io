#!/usr/bin/env bash
# Local AlpaSim setup. Idempotent; safe to re-run after `git pull` in repo/.
#
# The casadi pin is load-bearing: casadi 3.8.0 (2026-08-25) removed
# `casadi.tools.SX`, which do_mpc still imports, so BOTH AlpaSim MPC controllers
# fail to load and `alpasim-info` reports `MPC: (none)`. Rollouts need a
# controller, so this is a hard blocker, not cosmetic. Re-apply after any
# `uv sync`, which resolves casadi back to 3.8.
set -eo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE/repo"
. "$HOME/.cargo/env" 2>/dev/null || true

command -v cargo >/dev/null || { echo "cargo missing (builds utils_rs): https://rustup.rs"; exit 1; }
[ -d .venv ] || uv sync --extra all
uv pip install -q 'casadi<3.8'          # see note above
# The bridge lives at repo/plugins/harness_driver — AlpaSim's documented extension
# point, and the ONLY place the driver container can see it (the generated compose
# mounts repo/plugins, repo/src and repo/data/drivers; nothing else, and no env).
uv pip install -q -e plugins/harness_driver

# Harness sources must also be reachable inside the container; repo/data/drivers is
# mounted at /mnt/drivers, so stage them there.
mkdir -p data/drivers/harness
cp -u ../../*.py data/drivers/harness/ 2>/dev/null || true

echo "--- registry ---"
uv run alpasim-info 2>&1 | grep -E "^Models|^MPC|^Configs"
echo
echo "expect: Models ... harness ...   MPC: linear, nonlinear   Configs: harness"
