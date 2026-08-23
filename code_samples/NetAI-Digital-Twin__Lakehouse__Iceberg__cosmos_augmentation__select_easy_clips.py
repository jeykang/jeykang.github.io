"""Select easy daytime clips to augment + the 121-frame AGENT WINDOW of each (so the
target actually contains agents to obscure), and write per-clip depth specs with rotated
conditions (night/rain/fog).

Easy target = low behavioral_score (easy interactions). For each candidate we locate the
121-frame window with the most agents (safety.find_agent_window) and keep it only if that
window has >=2 agents — the naive first-121-frame trim is often empty (agents elsewhere in
the clip). Records the window start so staging trims the right segment.

Writes batch_specs/<short>_<cond>.json + batch_manifest.json (with window_start/fps).
Recipe: depth control + condition-only prompts (see make_refine_specs.PROMPTS / FINDINGS.md).
"""
import glob, json, os, sys
import pyarrow.parquet as pq
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_refine_specs import PROMPTS
import safety

ROOT = "netai-e2e/nvidia-physicalai-av-subset"
HERE = os.path.dirname(os.path.abspath(__file__))
N = int(sys.argv[1]) if len(sys.argv) > 1 else 9
MAX_CHECK = int(sys.argv[2]) if len(sys.argv) > 2 else 60   # candidates to probe
CONDS = ["night", "rain", "fog"]

beh = pq.read_table(f"{ROOT}/.behavioral/behavioral_shard_00_of_01.parquet").to_pydict()
B = {c: beh["behavioral_score"][i] for i, c in enumerate(beh["clip_id"])}

# easy candidates with a front camera, sorted easiest-first
cands = []
for c in sorted(B, key=lambda k: B[k]):
    if B[c] >= 0.30:
        break
    m = glob.glob(f"{ROOT}/camera/camera_front_wide_120fov/*/{c}.camera_front_wide_120fov.mp4")
    if m:
        cands.append((c, B[c], m[0]))

sel = []
print(f"probing up to {MAX_CHECK} easy candidates for agent windows...")
for c, b, mp4 in cands[:MAX_CHECK]:
    start, agents, fps = safety.find_agent_window(mp4)
    if agents >= 2:
        sel.append({"clip": c, "short": c[:8], "behavioral": round(b, 3),
                    "window_start": int(start), "fps": round(float(fps), 3),
                    "window_agents": int(agents), "local_mp4": mp4})
        if len(sel) >= N:
            break

os.makedirs(f"{HERE}/batch_specs", exist_ok=True)
for f in glob.glob(f"{HERE}/batch_specs/*.json"):
    os.remove(f)
for i, e in enumerate(sel):
    cond = CONDS[i % len(CONDS)]; e["cond"] = cond
    spec = {"prompt": PROMPTS[cond],
            "input_video_path": f"/scratch/autodr_test/aug_test/batch_inputs/{e['short']}_121.mp4",
            "depth": {"control_weight": 1.0}}
    json.dump(spec, open(f"{HERE}/batch_specs/{e['short']}_{cond}.json", "w"), indent=2)
json.dump(sel, open(f"{HERE}/batch_manifest.json", "w"), indent=2)
print(f"selected {len(sel)} easy daytime clips with agent windows:")
for e in sel:
    print(f"  {e['short']}  beh={e['behavioral']}  window@frame{e['window_start']} "
          f"({e['window_agents']} agents)  -> {e['cond']}")
