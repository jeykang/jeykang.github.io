"""Stage the batch to the cluster: upload each clip, trim its AGENT WINDOW (window_start
from the manifest) to 121 frames via the SIF's ffmpeg, upload specs + sbatch, submit.
Run with the conda python (has paramiko). Idempotent-ish (overwrites inputs/specs)."""
import glob, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cluster

HERE = os.path.dirname(os.path.abspath(__file__))
H = "/scratch/autodr_test/aug_test"
man = json.load(open(f"{HERE}/batch_manifest.json"))

cluster.run(f"mkdir -p {H}/batch_inputs_full {H}/batch_inputs {H}/batch_specs")
for e in man:
    cluster.put(e["local_mp4"], f"{H}/batch_inputs_full/{e['short']}.mp4")
for f in glob.glob(f"{HERE}/batch_specs/*.json"):
    cluster.put(f, f"{H}/batch_specs/{os.path.basename(f)}")
cluster.put(f"{HERE}/../cosmos_batch.sbatch", f"{H}/cosmos_batch.sbatch")
print(f"uploaded {len(man)} clips + specs")

# trim each clip's agent window (start = window_start/fps seconds) to 121 frames
lines = ["S=/scratch/autodr_test"]
for e in man:
    ss = e["window_start"] / max(1.0, e["fps"])
    lines.append(
        f"singularity exec -B $S:$S $S/cosmos_transfer1.sif ffmpeg -y -ss {ss:.3f} "
        f"-i {H}/batch_inputs_full/{e['short']}.mp4 -frames:v 121 "
        f"{H}/batch_inputs/{e['short']}_121.mp4 >/dev/null 2>&1 "
        f"&& echo trimmed {e['short']}@{ss:.1f}s || echo TRIMFAIL {e['short']}")
out, _ = cluster.run("\n".join(lines), timeout=600)
print(out.strip())

out2, _ = cluster.run(f"cd {H} && rm -rf batch_out batch_*.out batch_*.err && sbatch cosmos_batch.sbatch")
print(out2.strip())
