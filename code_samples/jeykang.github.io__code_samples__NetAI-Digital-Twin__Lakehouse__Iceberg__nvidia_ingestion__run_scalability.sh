#!/bin/bash
# Scalability sweep — runs each level as a separate docker exec for JVM isolation
# Usage: bash nvidia_ingestion/run_scalability.sh [max_level] [driver_memory]

MAX_LEVEL="${1:-6}"
DRIVER_MEM="${2:-4g}"
START_LEVEL="${3:-0}"
RESULTS_FILE="/tmp/nvidia_scalability_results.json"

if [ "$START_LEVEL" -eq 0 ]; then
    echo "[]" > "$RESULTS_FILE"
fi

LEVELS=("1,2" "1,5" "1,10" "1,25" "1,50" "1,60" "1,75" "1,100" "2,100" "5,100" "10,100")

echo "======================================================================"
echo "NVIDIA PHYSICALAI - SCALABILITY BENCHMARK"
echo "  Levels: $START_LEVEL to $((MAX_LEVEL-1)), driver_memory: $DRIVER_MEM"
echo "======================================================================"

for i in $(seq $START_LEVEL $((MAX_LEVEL-1))); do
    IFS=',' read -r MC ML <<< "${LEVELS[$i]}"
    echo ""
    echo "############################################################"
    echo "# Level $i: max_chunks=$MC, max_clips=$ML"
    echo "############################################################"

    LEVEL_FILE="/tmp/_scale_level_${i}.json"

    docker exec -u 1010 -e HOME=/tmp -e SPARK_DRIVER_MEMORY=$DRIVER_MEM spark-iceberg bash -c "
export PYTHONPATH=/opt/spark/python:/opt/spark/python/lib/py4j-0.10.9.7-src.zip:/opt/spark
export SPARK_DRIVER_MEMORY=$DRIVER_MEM
cd /opt/spark
python3 -m nvidia_ingestion.scalability_benchmark \\
    --run-level $i --driver-memory $DRIVER_MEM \\
    --level-output $LEVEL_FILE
" 2>&1 | grep -E '^  (Bronze|Silver|Gold|>>>|!!!)|ERROR'

    EXIT_CODE=$?

    # Copy level results from container
    docker cp spark-iceberg:$LEVEL_FILE /tmp/_scale_level_${i}.json 2>/dev/null

    if [ -f "/tmp/_scale_level_${i}.json" ]; then
        # Merge into combined results
        python3 -c "
import json
with open('$RESULTS_FILE') as f:
    combined = json.load(f)
with open('/tmp/_scale_level_${i}.json') as f:
    level = json.load(f)
combined.extend(level)
with open('$RESULTS_FILE', 'w') as f:
    json.dump(combined, f, indent=2)

# Print total line
totals = [r for r in level if r['phase'] == 'total']
errors = [r for r in level if r.get('error')]
if totals:
    t = totals[0]
    print(f\">>> L$i TOTAL: {t['wall_s']}s, {t['rows_out']} rows, RSS {t['peak_rss_mb']}MB\")
if errors:
    for e in errors:
        print(f\"!!! ERROR in {e['phase']}: {e['error'][:100]}\")
    exit(1)
"
        MERGE_EXIT=$?
        rm -f "/tmp/_scale_level_${i}.json"
        if [ $MERGE_EXIT -ne 0 ]; then
            echo "!!! Level $i had errors - stopping sweep"
            break
        fi
    else
        echo "!!! Level $i crashed (exit=$EXIT_CODE) - stopping sweep"
        break
    fi
done

echo ""
echo "======================================================================"
echo "RESULTS: $RESULTS_FILE"
echo "======================================================================"

# Print summary table
python3 -c "
import json
with open('$RESULTS_FILE') as f:
    data = json.load(f)
print(f\"{'Lvl':>3} {'Chnk':>4} {'Clip':>4} {'Phase':>8} {'Wall(s)':>8} {'CPU_U':>7} {'CPU_S':>7} {'RSS_MB':>8} {'Rows':>12}\")
print('-' * 80)
for r in data:
    err = ' ERR' if r.get('error') else ''
    print(f\"{r['level']:>3} {r['max_chunks']:>4} {r['max_clips']:>4} {r['phase']:>8} {r['wall_s']:>8.1f} {r['cpu_user_s']:>7.1f} {r['cpu_sys_s']:>7.1f} {r['peak_rss_mb']:>8.1f} {r['rows_out']:>12}{err}\")
print('=' * 80)
"
