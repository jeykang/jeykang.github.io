#!/usr/bin/env bash
# Wait for both DiffusionDrive shards, then install their parquets into
# <NFS>/.planning/ so edge_case_scorer picks them up. Because rung-0's
# planning_shard_00_of_01 sorts before DD's *_of_02 files, the loader merge
# makes DiffusionDrive the primary planning signal with the rung-0 CV score as
# the per-clip fallback (clips DD couldn't score). Then re-score Gold.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/../.."          # -> Lakehouse/Iceberg
SUB=netai-e2e/nvidia-physicalai-av-subset
STAGE="$SUB/.planning_dd_staging"

echo "[finalize] waiting for both shards ..."
docker wait dd_run_0 dd_run_1 || true
for s in 0 1; do echo "  shard $s tail:"; docker logs --tail 2 dd_run_$s 2>&1 | sed 's/^/    /'; done

cp "$STAGE"/planning_shard_*.parquet "$SUB/.planning/"
echo "[finalize] installed DiffusionDrive shards into .planning/"

python3 - <<'PY'
import glob, statistics as st, pyarrow.parquet as pq
sub="netai-e2e/nvidia-physicalai-av-subset"
dd=[r for f in glob.glob(f"{sub}/.planning_dd_staging/*.parquet") for r in pq.read_table(f).to_pylist()]
ps=[r["planning_score"] for r in dd]; sp=[r["mode_spread"] for r in dd]
print(f"[finalize] DiffusionDrive scored {len(dd):,} clips")
if ps:
    print(f"  planning_score: mean={st.mean(ps):.3f} stdev={st.pstdev(ps):.3f} range=[{min(ps):.2f},{max(ps):.2f}]")
    print(f"  mode_spread:    mean={st.mean(sp):.2f} stdev={st.pstdev(sp):.2f}")
PY
echo "[finalize] DONE — next: re-run Gold scoring to blend planning (DiffusionDrive)"
