#!/usr/bin/env bash
# Wait for both BEVFusion validation shards to finish, merge per-N shard
# parquets, then run the sampling-adequacy analysis. Idempotent.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/../.."   # -> Lakehouse/Iceberg
ROOT="$PWD"
VAL="$ROOT/bevfusion/validation"

echo "[finish] waiting for both shards (ALL_DONE markers) ..."
for _ in $(seq 1 600); do          # up to ~5h
  if grep -q ALL_DONE_SHARD_0 /tmp/val_gpu0.log 2>/dev/null \
     && grep -q ALL_DONE_SHARD_1 /tmp/val_gpu1.log 2>/dev/null; then
    echo "[finish] both shards done"; break
  fi
  sleep 30
done

echo "[finish] merging shard parquets per N ..."
python3 - <<'PY'
import glob, os
import pyarrow as pa, pyarrow.parquet as pq
val = os.path.join(os.getcwd(), "bevfusion/validation")
os.makedirs(os.path.join(val, "merged"), exist_ok=True)
for N in (10, 20, 40):
    fs = sorted(glob.glob(os.path.join(val, f"n{N}", "*.parquet")))
    if not fs:
        print(f"  N={N}: NO shard parquets found"); continue
    tabs = [pq.read_table(f) for f in fs]
    out = os.path.join(val, "merged", f"n{N}.parquet")
    pq.write_table(pa.concat_tables(tabs), out)
    print(f"  N={N}: merged {len(fs)} shards -> {sum(t.num_rows for t in tabs)} rows")
PY

echo "[finish] running sampling-adequacy analysis ..."
docker run --rm -v "$VAL:/work/validation" \
  --entrypoint python netai/bevfusion-runner:latest \
  /workspace/validate_sampling.py \
    --scores-10 /work/validation/merged/n10.parquet \
    --scores-20 /work/validation/merged/n20.parquet \
    --scores-40 /work/validation/merged/n40.parquet \
  | tee "$VAL/VERDICT.txt"
echo "[finish] verdict written to $VAL/VERDICT.txt"
