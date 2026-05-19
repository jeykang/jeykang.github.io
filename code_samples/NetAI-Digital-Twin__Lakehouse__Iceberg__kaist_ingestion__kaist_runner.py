#!/usr/bin/env python3
"""
CLI Runner for KAIST E2E Dataset Ingestion Pipeline.

Usage:
    # Run full pipeline
    python kaist_runner.py all
    
    # Run individual layers
    python kaist_runner.py bronze
    python kaist_runner.py silver
    python kaist_runner.py gold
    
    # Run validation
    python kaist_runner.py validate
    
    # Run with custom config
    KAIST_SOURCE_PATH=/data/kaist python kaist_runner.py all
"""

import argparse
import sys
import time
from typing import Dict


def run_bronze() -> Dict[str, int]:
    """Run Bronze layer ingestion."""
    from kaist_ingestion.ingest_bronze import run_bronze_ingestion
    return run_bronze_ingestion()


def run_silver() -> Dict[str, int]:
    """Run Silver layer transformation."""
    from kaist_ingestion.transform_silver import run_silver_transformation
    return run_silver_transformation()


def run_gold() -> Dict[str, int]:
    """Run Gold layer construction."""
    from kaist_ingestion.build_gold import run_gold_build
    return run_gold_build()


def run_validate(layers: list = None) -> bool:
    """Run validation and return success status."""
    from kaist_ingestion.validators import run_validation
    report = run_validation(layers=layers)
    print(report.summary())
    return report.passed


def run_all() -> bool:
    """Run the complete pipeline: Bronze → Silver → Gold → Validate."""
    print("\n" + "=" * 70)
    print("KAIST E2E DATASET INGESTION PIPELINE")
    print("=" * 70)
    
    total_start = time.time()
    success = True
    
    # Bronze Layer
    print("\n[PHASE 1/4] Bronze Layer Ingestion")
    print("-" * 50)
    start = time.time()
    bronze_results = run_bronze()
    bronze_time = time.time() - start
    print(f"Bronze completed in {bronze_time:.2f}s")
    
    if all(v >= 0 for v in bronze_results.values()):
        # Silver Layer
        print("\n[PHASE 2/4] Silver Layer Transformation")
        print("-" * 50)
        start = time.time()
        silver_results = run_silver()
        silver_time = time.time() - start
        print(f"Silver completed in {silver_time:.2f}s")
        
        if all(v >= 0 for v in silver_results.values()):
            # Gold Layer
            print("\n[PHASE 3/4] Gold Layer Construction")
            print("-" * 50)
            start = time.time()
            gold_results = run_gold()
            gold_time = time.time() - start
            print(f"Gold completed in {gold_time:.2f}s")
            
            # Validation
            print("\n[PHASE 4/4] Data Validation")
            print("-" * 50)
            start = time.time()
            validation_passed = run_validate()
            validation_time = time.time() - start
            print(f"Validation completed in {validation_time:.2f}s")
            
            success = validation_passed
        else:
            print("\n[ERROR] Silver transformation failed - skipping subsequent phases")
            success = False
    else:
        print("\n[ERROR] Bronze ingestion failed - skipping subsequent phases")
        success = False
    
    total_time = time.time() - total_start
    
    print("\n" + "=" * 70)
    print(f"PIPELINE {'COMPLETED' if success else 'FAILED'}")
    print(f"Total time: {total_time:.2f}s")
    print("=" * 70)
    
    return success


def main():
    parser = argparse.ArgumentParser(
        description="KAIST E2E Dataset Ingestion Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    parser.add_argument(
        "command",
        choices=["all", "bronze", "silver", "gold", "validate", "benchmark"],
        help="Pipeline command to run",
    )
    
    parser.add_argument(
        "--layers",
        nargs="+",
        choices=["bronze", "silver", "gold"],
        default=None,
        help="Layers to validate (only used with 'validate' command)",
    )
    
    args = parser.parse_args()
    
    try:
        if args.command == "all":
            success = run_all()
        elif args.command == "bronze":
            results = run_bronze()
            success = all(v >= 0 for v in results.values())
        elif args.command == "silver":
            results = run_silver()
            success = all(v >= 0 for v in results.values())
        elif args.command == "gold":
            results = run_gold()
            success = all(v >= 0 for v in results.values())
        elif args.command == "validate":
            success = run_validate(layers=args.layers)
        elif args.command == "benchmark":
            from benchmarks.ad_workload_benchmark import main as run_benchmarks
            run_benchmarks()
            success = True
        else:
            parser.print_help()
            success = False
            
        sys.exit(0 if success else 1)
        
    except Exception as e:
        print(f"\n[FATAL ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
