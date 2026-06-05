"""Thin launcher for Gold edge-case scoring under spark-submit."""
import argparse
import sys

from nvidia_ingestion.edge_case_scorer import run_gold_scoring


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--backend", default="metadata")
    p.add_argument("--top-pct", type=float, default=10.0)
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args()

    scores_df, gold_results = run_gold_scoring(
        backend_name=args.backend,
        top_pct=args.top_pct,
        limit=args.limit,
    )
    sys.exit(0)
