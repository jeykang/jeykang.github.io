# Full training (100 epochs from config)
python scripts/train.py --config config/experiment/cse_warehouse.yaml

# Quick test with 5 epochs
python scripts/train.py --config config/experiment/cse_warehouse.yaml --epochs 5

# Comparison
python scripts/compare_devices.py --dir output/metrics