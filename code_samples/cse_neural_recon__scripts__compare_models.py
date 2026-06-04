#!/usr/bin/env python3
"""
Compare SDF statistics between two trained models.
Useful for verifying improvements after training changes.
"""

import argparse
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from src.models import HashGridSDF


def load_model(checkpoint_path: str) -> HashGridSDF:
    """Load a trained model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
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
        geometric_init=False
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model


def analyze_sdf_distribution(model: HashGridSDF, resolution: int = 64) -> dict:
    """Analyze SDF value distribution on a dense grid."""
    with torch.no_grad():
        x = torch.linspace(0, 1, resolution)
        xx, yy, zz = torch.meshgrid(x, x, x, indexing='ij')
        grid_pts = torch.stack([xx.flatten(), yy.flatten(), zz.flatten()], dim=-1)
        
        output = model(grid_pts.unsqueeze(0))
        sdf_values = output['sdf'].squeeze().numpy()
        
        return {
            'mean': float(sdf_values.mean()),
            'std': float(sdf_values.std()),
            'min': float(sdf_values.min()),
            'max': float(sdf_values.max()),
            'negative_pct': float((sdf_values < 0).sum() / len(sdf_values) * 100),
            'positive_pct': float((sdf_values > 0).sum() / len(sdf_values) * 100),
            'near_surface': {
                thresh: float((np.abs(sdf_values) < thresh).sum() / len(sdf_values) * 100)
                for thresh in [0.001, 0.005, 0.01, 0.02, 0.05]
            }
        }


def print_comparison(name1: str, stats1: dict, name2: str, stats2: dict):
    """Print a formatted comparison table."""
    print("\n" + "="*70)
    print(f"{'Metric':<25} {name1:<20} {name2:<20}")
    print("="*70)
    
    def delta_arrow(v1, v2, better_higher=True):
        diff = v2 - v1
        if abs(diff) < 0.0001:
            return ""
        if better_higher:
            return "↑" if diff > 0 else "↓"
        return "↓" if diff > 0 else "↑"
    
    # Basic stats
    print(f"{'Mean SDF':<25} {stats1['mean']:<20.6f} {stats2['mean']:<20.6f}")
    print(f"{'Std SDF':<25} {stats1['std']:<20.6f} {stats2['std']:<20.6f}")
    print(f"{'Min SDF':<25} {stats1['min']:<20.6f} {stats2['min']:<20.6f}")
    print(f"{'Max SDF':<25} {stats1['max']:<20.6f} {stats2['max']:<20.6f}")
    print("-"*70)
    
    # Interior/exterior
    neg_arrow = delta_arrow(stats1['negative_pct'], stats2['negative_pct'], better_higher=True)
    print(f"{'Negative (interior) %':<25} {stats1['negative_pct']:>8.2f}% {'':<10} {stats2['negative_pct']:>8.2f}% {neg_arrow}")
    print(f"{'Positive (exterior) %':<25} {stats1['positive_pct']:>8.2f}% {'':<10} {stats2['positive_pct']:>8.2f}%")
    print("-"*70)
    
    # Near-surface stats
    print("Near-surface points (|sdf| < threshold):")
    for thresh in [0.001, 0.005, 0.01, 0.02, 0.05]:
        v1 = stats1['near_surface'][thresh]
        v2 = stats2['near_surface'][thresh]
        print(f"  |sdf| < {thresh:<6} {v1:>12.2f}% {'':<6} {v2:>12.2f}%")
    
    print("="*70)


def main():
    parser = argparse.ArgumentParser(description='Compare two trained SDF models')
    parser.add_argument('model1', help='Path to first model checkpoint')
    parser.add_argument('model2', help='Path to second model checkpoint')
    parser.add_argument('--name1', default='Model 1', help='Name for first model')
    parser.add_argument('--name2', default='Model 2', help='Name for second model')
    parser.add_argument('--resolution', type=int, default=64, help='Grid resolution for analysis')
    
    args = parser.parse_args()
    
    print(f"Loading {args.name1} from {args.model1}...")
    model1 = load_model(args.model1)
    stats1 = analyze_sdf_distribution(model1, args.resolution)
    
    print(f"Loading {args.name2} from {args.model2}...")
    model2 = load_model(args.model2)
    stats2 = analyze_sdf_distribution(model2, args.resolution)
    
    print_comparison(args.name1, stats1, args.name2, stats2)
    
    # Summary assessment
    print("\nAssessment:")
    if stats2['negative_pct'] > stats1['negative_pct'] * 1.5:
        print(f"  ✓ {args.name2} has significantly more interior points ({stats2['negative_pct']:.1f}% vs {stats1['negative_pct']:.1f}%)")
    if abs(stats2['min']) > abs(stats1['min']) * 1.5:
        print(f"  ✓ {args.name2} has deeper negative values (min={stats2['min']:.4f} vs {stats1['min']:.4f})")
    if stats2['mean'] < stats1['mean']:
        print(f"  ✓ {args.name2} has lower mean SDF ({stats2['mean']:.4f} vs {stats1['mean']:.4f})")


if __name__ == '__main__':
    main()
