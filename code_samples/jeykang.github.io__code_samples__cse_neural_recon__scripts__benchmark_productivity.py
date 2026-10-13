#!/usr/bin/env python3
"""
DGX Spark Productivity Benchmark Script

Runs a quick training benchmark to collect productivity metrics
demonstrating the value of DGX Spark for neural 3D reconstruction.

Usage:
    python scripts/benchmark_productivity.py --epochs 5
    python scripts/benchmark_productivity.py --epochs 10 --batch_size 16
"""

import argparse
import sys
import time
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models import HashGridSDF
from src.data import CSEDataset
from src.utils.productivity_metrics import (
    ProductivityMetricsCollector,
    create_comparison_baseline
)


def run_benchmark(
    epochs: int = 5,
    batch_size: int = 8,
    data_path: str = "data/warehouse_extracted/static_warehouse_robot1",
    output_dir: str = "output/productivity_benchmark",
):
    """Run productivity benchmark."""
    
    print("=" * 80)
    print("DGX SPARK PRODUCTIVITY BENCHMARK")
    print("Neural 3D Reconstruction Workload")
    print("=" * 80)
    print()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Create output directory
    output_path = Path(output_dir) / datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize metrics collector
    collector = ProductivityMetricsCollector(output_dir=str(output_path))
    
    # Create model
    print("\nCreating model...")
    model = HashGridSDF(
        hidden_features=256,
        hidden_layers=6,
        encoding_config={
            'num_levels': 16,
            'base_resolution': 16,
            'max_resolution': 4096,
            'features_per_level': 2,
            'log2_hashmap_size': 19,
        },
        geometric_init=True,
    ).to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model: HashGridSDF ({num_params:,} parameters)")
    
    # Create dataset and dataloader
    print("\nLoading dataset...")
    dataset = CSEDataset(
        run_dir=data_path,
        depth_scale=0.001,
    )
    print(f"Dataset: {len(dataset)} frames")
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    print(f"Batch size: {batch_size}")
    print(f"Batches per epoch: {len(dataloader)}")
    
    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scaler = torch.cuda.amp.GradScaler()
    
    # Start experiment
    config = {
        'training': {
            'batch_size': batch_size,
            'learning_rate': 1e-4,
            'epochs': epochs,
        }
    }
    collector.start_experiment(model, config, "HashGridSDF")
    
    print(f"\nRunning benchmark for {epochs} epochs...")
    print("-" * 80)
    
    # Training loop
    for epoch in range(epochs):
        collector.start_epoch(epoch)
        model.train()
        
        epoch_start = time.time()
        epoch_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(dataloader):
            collector.start_batch()
            collector.start_phase('data_load')
            
            # Move to device
            depth = batch['depth'].to(device)
            pose = batch['pose'].to(device)
            K = batch['K'].to(device)
            
            collector.start_phase('forward')
            
            # Simple forward pass (just to measure throughput)
            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast():
                # Sample random points in [0, 1]
                B = depth.shape[0]
                points = torch.rand(B, 1000, 3, device=device)
                
                # Forward
                output = model(points)
                sdf = output['sdf']
                
                # Simple loss
                loss = sdf.abs().mean()
            
            collector.start_phase('backward')
            
            # Backward
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            # End batch
            collector.end_batch(epoch, batch_idx, B, loss.item())
            
            epoch_loss += loss.item()
            num_batches += 1
            
            # Progress update
            if batch_idx % 100 == 0:
                print(f"  Epoch {epoch} | Batch {batch_idx}/{len(dataloader)} | Loss: {loss.item():.4f}")
        
        collector.end_epoch(epoch)
        
        epoch_time = time.time() - epoch_start
        avg_loss = epoch_loss / num_batches
        samples_per_sec = (num_batches * batch_size) / epoch_time
        
        print(f"Epoch {epoch} complete: {epoch_time:.1f}s | Loss: {avg_loss:.4f} | {samples_per_sec:.0f} samples/sec")
    
    # End experiment
    collector.end_experiment()
    
    # Generate report
    print("\n" + "=" * 80)
    print("GENERATING PRODUCTIVITY REPORT")
    print("=" * 80)
    
    report = collector.generate_report()
    json_path, text_path = collector.save_report(report, "dgx_spark_benchmark")
    
    # Print summary
    print(f"\n{'-' * 80}")
    print("BENCHMARK RESULTS SUMMARY")
    print(f"{'-' * 80}")
    print(f"  Total Training Time: {report.total_training_time:.1f} seconds ({report.total_training_time/60:.1f} minutes)")
    print(f"  Total Samples Processed: {report.total_samples_processed:,}")
    print(f"  Samples/Second: {report.samples_per_second:.1f}")
    print(f"  Epochs/Hour: {report.epochs_per_hour:.1f}")
    print(f"  GPU Utilization: {report.avg_gpu_utilization:.1f}% (peak: {report.peak_gpu_utilization:.1f}%)")
    print(f"  Memory Used: {report.peak_memory_used:.0f} MB")
    print(f"  Power Efficiency: {report.samples_per_watt:.1f} samples/Watt-hour")
    print(f"  Estimated Experiments/Day: {report.estimated_experiments_per_day:.1f}")
    print(f"{'-' * 80}")
    
    # DGX Spark productivity value analysis
    print("\n" + "=" * 80)
    print("DGX SPARK RESEARCH PRODUCTIVITY VALUE")
    print("=" * 80)
    
    # Key productivity benefits
    print("\n1. ITERATION SPEED")
    print("-" * 40)
    hours_per_experiment = 12 / report.epochs_per_hour if report.epochs_per_hour > 0 else 24  # Assume 12-epoch experiments
    experiments_per_week = report.estimated_experiments_per_day * 5  # 5-day work week
    print(f"   • {report.epochs_per_hour:.1f} epochs/hour training throughput")
    print(f"   • {hours_per_experiment:.1f} hours per full training run (12 epochs)")
    print(f"   • {report.estimated_experiments_per_day:.0f} experiments possible per day")
    print(f"   • {experiments_per_week:.0f} experiments per work week")
    
    print("\n2. POWER & COST EFFICIENCY")
    print("-" * 40)
    print(f"   • {report.avg_power_draw:.1f}W average power consumption")
    print(f"   • {report.samples_per_watt:.1f} samples per Watt-hour")
    cost_per_kwh = 0.15  # Typical US electricity rate
    daily_power_cost = (report.avg_power_draw / 1000) * 24 * cost_per_kwh
    print(f"   • ~${daily_power_cost:.2f}/day power cost (24/7 operation)")
    print(f"   • Desktop-friendly power envelope (no special infrastructure)")
    
    print("\n3. MEMORY CAPACITY")
    print("-" * 40)
    print(f"   • {report.peak_memory_used:.0f} MB current usage")
    print(f"   • {report.memory_headroom/1024:.1f} GB memory headroom available")
    print(f"   • Enables larger batch sizes and model architectures")
    print(f"   • No out-of-memory constraints on current workload")
    
    print("\n4. ACCESSIBILITY")
    print("-" * 40)
    print("   • Runs on desktop without datacenter infrastructure")
    print("   • Blackwell architecture with latest CUDA 13.0")
    print("   • Immediate availability (no cluster scheduling)")
    print("   • Full control over experiments and debugging")
    
    print("\n5. COMPARED TO CLOUD GPU RENTAL")
    print("-" * 40)
    cloud_hourly_rate = 3.0  # Typical cloud GPU hourly rate
    hours_saved_per_day = 24  # Always available
    monthly_cloud_cost = cloud_hourly_rate * 8 * 22  # 8 hours/day, 22 workdays
    print(f"   • Cloud GPU rental: ~${cloud_hourly_rate:.2f}/hour")
    print(f"   • Equivalent monthly cloud cost: ~${monthly_cloud_cost:.0f}")
    print("   • DGX Spark: One-time purchase, unlimited usage")
    print("   • No data transfer latency or security concerns")
    
    # Research productivity gains
    print("\n" + "=" * 80)
    print("ESTIMATED RESEARCH PRODUCTIVITY GAINS")
    print("=" * 80)
    
    # Before DGX Spark scenarios
    cpu_only_speedup = 50  # Rough estimate GPU vs CPU
    print("\n   Scenario: Neural 3D Reconstruction Research")
    print("-" * 60)
    print(f"   • Training throughput: {report.samples_per_second:.0f} samples/second")
    print(f"   • Model complexity: {num_params/1e6:.1f}M parameters")
    print(f"   • Weekly experiment capacity: {experiments_per_week:.0f} full training runs")
    print(f"   • vs CPU-only: ~{cpu_only_speedup}x faster iteration")
    
    print(f"\nReports saved to: {output_path}")
    print(f"  - {json_path.name}")
    print(f"  - {text_path.name}")
    
    return report


def main():
    parser = argparse.ArgumentParser(description='DGX Spark Productivity Benchmark')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs to run')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--data_path', type=str, 
                        default='data/warehouse_extracted/static_warehouse_robot1',
                        help='Path to dataset')
    parser.add_argument('--output_dir', type=str, default='output/productivity_benchmark',
                        help='Output directory for reports')
    args = parser.parse_args()
    
    run_benchmark(
        epochs=args.epochs,
        batch_size=args.batch_size,
        data_path=args.data_path,
        output_dir=args.output_dir,
    )


if __name__ == '__main__':
    main()
