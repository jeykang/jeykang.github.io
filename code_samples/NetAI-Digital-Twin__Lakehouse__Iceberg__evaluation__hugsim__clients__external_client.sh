#!/usr/bin/env zsh
# external_client.sh — ltf_path stub for policies that run in their own container.
# closed_loop.py launches `zsh <ltf_path> <ad_cuda> <output>` and only polls it at the end;
# the real client (run_vavam.sh starts vavam_client.py in alpasim-harness-base) meets the
# simulator at the named pipes in <output>.
echo "external client expected on $2/obs_pipe, $2/plan_pipe"
