#!/usr/bin/env python3
"""
Cross-Device Comparison Tool

Aggregates and compares training metrics from multiple devices/GPUs.
Use this to compare DGX Spark performance against other hardware.

Usage:
    python scripts/compare_devices.py output/metrics/*/productivity_*.json
    python scripts/compare_devices.py --dir output/metrics
    
The script will find all *_comparison.json files and generate a comparison report.
"""

import argparse
import json
import os
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime


def find_comparison_files(search_paths: List[str]) -> List[Path]:
    """Find all comparison JSON files."""
    files = []
    
    for path in search_paths:
        p = Path(path)
        
        if p.is_file() and p.name.endswith('_comparison.json'):
            files.append(p)
        elif p.is_file() and p.name.endswith('.json'):
            # Check if it's a valid metrics file
            files.append(p)
        elif p.is_dir():
            # Search recursively for comparison files
            files.extend(p.rglob('*_comparison.json'))
            
    return sorted(set(files))


def load_metrics(filepath: Path) -> Dict[str, Any]:
    """Load metrics from a JSON file."""
    with open(filepath) as f:
        data = json.load(f)
    
    # Normalize the data format
    if 'samples_per_second' in data:
        # Already in comparison format
        return data
    elif 'device_info' in data:
        # Full report format - extract key metrics
        return {
            'device_id': data.get('device_info', {}).get('hostname', 'unknown'),
            'gpu_name': data.get('gpu_name', 'unknown'),
            'gpu_memory_gb': data.get('device_info', {}).get('gpu_total_memory_gb', 0),
            'cuda_version': data.get('cuda_version', 'unknown'),
            'pytorch_version': data.get('pytorch_version', 'unknown'),
            'model_name': data.get('model_name', 'unknown'),
            'model_parameters': data.get('model_parameters', 0),
            'batch_size': data.get('batch_size', 0),
            'epochs': data.get('total_epochs', 0),
            'samples_per_second': data.get('samples_per_second', 0),
            'epochs_per_hour': data.get('epochs_per_hour', 0),
            'avg_gpu_utilization': data.get('avg_gpu_utilization', 0),
            'peak_memory_mb': data.get('peak_memory_used', 0),
            'avg_power_watts': data.get('avg_power_draw', 0),
            'samples_per_watt': data.get('samples_per_watt', 0),
            'final_loss': data.get('final_loss', 0),
            'timestamp': data.get('report_generated', ''),
            'source_file': str(filepath),
        }
    else:
        # Basic format
        return {**data, 'source_file': str(filepath)}


def generate_comparison_table(metrics_list: List[Dict[str, Any]]) -> str:
    """Generate a formatted comparison table."""
    if not metrics_list:
        return "No metrics files found."
    
    lines = []
    lines.append("=" * 100)
    lines.append("CROSS-DEVICE TRAINING PERFORMANCE COMPARISON")
    lines.append("=" * 100)
    lines.append(f"Generated: {datetime.now().isoformat()}")
    lines.append(f"Files compared: {len(metrics_list)}")
    lines.append("")
    
    # Sort by samples per second (descending)
    sorted_metrics = sorted(metrics_list, key=lambda x: x.get('samples_per_second', 0), reverse=True)
    
    # Key metrics table
    lines.append("-" * 100)
    lines.append("THROUGHPUT COMPARISON (sorted by samples/sec)")
    lines.append("-" * 100)
    
    header = f"{'Device':<20} {'GPU':<25} {'Samples/s':>12} {'Epochs/hr':>12} {'GPU %':>8} {'Power(W)':>10}"
    lines.append(header)
    lines.append("-" * 100)
    
    baseline_throughput = None
    for m in sorted_metrics:
        device = m.get('device_id', 'unknown')[:18]
        gpu = m.get('gpu_name', 'unknown')[:23]
        sps = m.get('samples_per_second', 0)
        eph = m.get('epochs_per_hour', 0)
        gpu_util = m.get('avg_gpu_utilization', 0)
        power = m.get('avg_power_watts', 0)
        
        if baseline_throughput is None:
            baseline_throughput = sps
            
        row = f"{device:<20} {gpu:<25} {sps:>12.1f} {eph:>12.1f} {gpu_util:>7.1f}% {power:>10.1f}"
        lines.append(row)
    
    lines.append("")
    
    # Relative performance
    if baseline_throughput and baseline_throughput > 0:
        lines.append("-" * 100)
        lines.append("RELATIVE PERFORMANCE (vs fastest)")
        lines.append("-" * 100)
        
        for m in sorted_metrics:
            device = m.get('device_id', 'unknown')[:30]
            gpu = m.get('gpu_name', 'unknown')[:25]
            sps = m.get('samples_per_second', 0)
            relative = sps / baseline_throughput if baseline_throughput > 0 else 0
            
            bar_len = int(relative * 40)
            bar = "█" * bar_len + "░" * (40 - bar_len)
            
            lines.append(f"{device:<30} {gpu:<25} {bar} {relative:.2f}x")
    
    lines.append("")
    
    # Efficiency comparison
    lines.append("-" * 100)
    lines.append("EFFICIENCY METRICS")
    lines.append("-" * 100)
    
    header = f"{'Device':<20} {'GPU':<25} {'Samples/W':>12} {'Peak Mem(MB)':>14} {'Mem(GB)':>10}"
    lines.append(header)
    lines.append("-" * 100)
    
    for m in sorted_metrics:
        device = m.get('device_id', 'unknown')[:18]
        gpu = m.get('gpu_name', 'unknown')[:23]
        spw = m.get('samples_per_watt', 0)
        peak_mem = m.get('peak_memory_mb', 0)
        total_mem = m.get('gpu_memory_gb', 0)
        
        row = f"{device:<20} {gpu:<25} {spw:>12.1f} {peak_mem:>14.0f} {total_mem:>10.1f}"
        lines.append(row)
    
    lines.append("")
    
    # Configuration details
    lines.append("-" * 100)
    lines.append("CONFIGURATION DETAILS")
    lines.append("-" * 100)
    
    for m in sorted_metrics:
        device = m.get('device_id', 'unknown')
        gpu = m.get('gpu_name', 'unknown')
        cuda = m.get('cuda_version', 'unknown')
        pytorch = m.get('pytorch_version', 'unknown')
        model = m.get('model_name', 'unknown')
        params = m.get('model_parameters', 0)
        batch = m.get('batch_size', 0)
        epochs = m.get('epochs', 0)
        timestamp = m.get('timestamp', 'unknown')
        
        lines.append(f"\n  {device} - {gpu}")
        lines.append(f"    CUDA: {cuda}, PyTorch: {pytorch}")
        lines.append(f"    Model: {model} ({params:,} params)")
        lines.append(f"    Batch Size: {batch}, Epochs: {epochs}")
        lines.append(f"    Timestamp: {timestamp}")
    
    lines.append("")
    lines.append("=" * 100)
    
    return "\n".join(lines)


def generate_csv(metrics_list: List[Dict[str, Any]], output_path: Path):
    """Generate a CSV for spreadsheet analysis."""
    if not metrics_list:
        return
    
    headers = [
        'device_id', 'gpu_name', 'gpu_memory_gb', 'cuda_version', 'pytorch_version',
        'model_name', 'model_parameters', 'batch_size', 'epochs',
        'samples_per_second', 'epochs_per_hour', 'avg_gpu_utilization',
        'peak_memory_mb', 'avg_power_watts', 'samples_per_watt', 'final_loss', 'timestamp'
    ]
    
    with open(output_path, 'w') as f:
        f.write(','.join(headers) + '\n')
        for m in metrics_list:
            row = [str(m.get(h, '')) for h in headers]
            f.write(','.join(row) + '\n')
    
    print(f"CSV saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Compare training metrics across different devices',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Compare all metrics in output directory
    python scripts/compare_devices.py --dir output/metrics
    
    # Compare specific files
    python scripts/compare_devices.py output/metrics/*/productivity_*_comparison.json
    
    # Save comparison to specific location
    python scripts/compare_devices.py --dir output/metrics --output comparison_report.txt
        """
    )
    parser.add_argument('files', nargs='*', help='Metric files to compare')
    parser.add_argument('--dir', '-d', type=str, help='Directory to search for metrics')
    parser.add_argument('--output', '-o', type=str, help='Output file for comparison report')
    parser.add_argument('--csv', action='store_true', help='Also generate CSV output')
    
    args = parser.parse_args()
    
    # Collect search paths
    search_paths = args.files if args.files else []
    if args.dir:
        search_paths.append(args.dir)
    if not search_paths:
        search_paths = ['output/metrics']
    
    # Find files
    files = find_comparison_files(search_paths)
    
    if not files:
        print(f"No comparison files found in: {search_paths}")
        print("Run training first to generate metrics, then use this script to compare.")
        return
    
    print(f"Found {len(files)} metric files:")
    for f in files:
        print(f"  - {f}")
    print()
    
    # Load metrics
    metrics_list = []
    for f in files:
        try:
            metrics = load_metrics(f)
            metrics['source_file'] = str(f)
            metrics_list.append(metrics)
        except Exception as e:
            print(f"Warning: Could not load {f}: {e}")
    
    # Generate comparison
    report = generate_comparison_table(metrics_list)
    print(report)
    
    # Save outputs
    if args.output:
        output_path = Path(args.output)
        with open(output_path, 'w') as f:
            f.write(report)
        print(f"\nReport saved to: {output_path}")
    
    if args.csv:
        csv_path = Path(args.output).with_suffix('.csv') if args.output else Path('device_comparison.csv')
        generate_csv(metrics_list, csv_path)


if __name__ == '__main__':
    main()
