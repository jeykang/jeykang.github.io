#!/usr/bin/env bash
# Submit an evaluation job to an ASSIGNED node, pairing node -> partition correctly.
#
#   ./submit.sh <node12|13> MODEL=<name> [KEY=VAL ...] [-- extra sbatch args]
#
# pod12 and pod15 are both in partition `jobs`. Access ends 2026-08-24 00:00 KST.
set -eo pipefail
NODE_SHORT=${1:?usage: submit.sh <12|13> MODEL=... [KEY=VAL ...]}
shift
case "$NODE_SHORT" in
  12) NODE=hpc-pr-a-pod12; PART=jobs ;;
  15) NODE=hpc-pr-a-pod15; PART=jobs ;;
  *)  echo "node must be 12 or 15 (assigned until 2026-08-24 00:00 KST)"; exit 2 ;;
esac

EXPORTS="ALL"
SB_EXTRA=()
for arg in "$@"; do
  case "$arg" in
    -*) SB_EXTRA+=("$arg") ;;
    *=*) EXPORTS="$EXPORTS,$arg" ;;
    *) SB_EXTRA+=("$arg") ;;
  esac
done

echo "[submit] node=$NODE partition=$PART exports=$EXPORTS"
sbatch --nodelist="$NODE" --partition="$PART" --export="$EXPORTS" "${SB_EXTRA[@]}" \
  /scratch/autodr_test/alpamayo_eval/eval.sbatch
