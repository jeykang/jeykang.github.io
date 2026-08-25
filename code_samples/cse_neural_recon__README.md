# MobileX Poles Neural 3D Reconstruction Pipeline

A comprehensive post-processing pipeline for converting multi-camera surveillance video feeds into high-fidelity, watertight 3D point clouds for coverage analysis.

## Overview

This pipeline processes RGB-D video from mobile surveillance robots (e.g., MobileX Poles with 6x e-Con AR0234CS cameras) and produces dense, error-corrected 3D reconstructions suitable for surveillance coverage analysis.

### Key Features

- **Multi-Camera Support**: Handles 6-camera rigs with temporal synchronization
- **Neural SDF Reconstruction**: SIREN-based implicit surface representation with planar priors
- **Manhattan World Constraints**: Axis-aligned regularization for indoor environments
- **Multi-Stage Error Correction**: Classical filtering + Neural point filtering + Score-based denoising
- **Point Cloud Completion**: Hole filling using planar extension and neural completion

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd cse_neural_recon

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

## Dataset Setup

This project uses the CSE (Collaborative SLAM in Service Environments) dataset for training and evaluation.

```
data/
├── hospital_extracted/
│   ├── dynamic_hospital_robot1/
│   ├── static_hospital_robot1/
│   └── ...
├── office_extracted/
├── warehouse_extracted/
└── lifelong_extracted/
```

Each sequence contains:
- `rgb_left_compressed/` and `rgb_right_compressed/`: Stereo RGB images
- `depth_left/` and `depth_right/`: Depth images
- `camera_info_left_intrinsics.json`: Camera intrinsics
- `ground_truth.txt`: Poses in TUM format

## Quick Start

### Training

```bash
# Train on a single sequence
python scripts/train.py --config config/experiment/cse_warehouse.yaml

# Train with custom settings
python scripts/train.py \
    --data_path data/warehouse_extracted/static_warehouse_robot1 \
    --epochs 30 \
    --batch_size 4
```

### Inference

```bash
# Run full reconstruction pipeline
python scripts/inference.py \
    --data_path data/warehouse_extracted/static_warehouse_robot1 \
    --checkpoint output/checkpoints/best_model.pt \
    --output_dir output/results/warehouse_robot1
```

### Evaluation

```bash
# Evaluate reconstruction quality
python scripts/evaluate.py \
    --prediction output/results/warehouse_robot1/point_cloud.ply \
    --ground_truth data/warehouse_extracted/static_warehouse_robot1/gt_mesh.ply
```

## Architecture

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for detailed architecture documentation.

### Pipeline Overview

```
Input (RGB-D + Poses) 
    → Feature Extraction (Semantics, Planes, Normals)
    → Neural SDF Fitting (SIREN + Planar Attention)
    → Mesh Extraction (Marching Cubes)
    → Error Correction (SOR + Neural Filter + Score Denoise)
    → Completion (Planar Extension + Neural Inpainting)
    → Output (Dense Point Cloud + Watertight Mesh)
```

## Project Structure

```
cse_neural_recon/
├── config/                 # Configuration files
│   ├── default.yaml       # Default training config
│   └── experiment/        # Experiment-specific configs
├── scripts/               # Main entry point scripts
│   ├── train.py          # Training script
│   ├── infer.py          # Inference script
│   ├── evaluate.py       # Evaluation script
│   └── preprocess.py     # Data preprocessing
├── src/                   # Source code
│   ├── data/             # Data loading and preprocessing
│   │   ├── dataset.py    # CSE dataset classes
│   │   ├── multi_camera.py # Multi-camera synchronization
│   │   ├── transforms.py # Data augmentation
│   │   └── depth_processing.py # Depth filtering
│   ├── models/           # Neural network architectures
│   │   ├── neural_sdf.py # Neural SDF with SIREN
│   │   ├── planar_attention.py # Planar attention module
│   │   ├── point_filter.py # 3DMambaIPF-style filtering
│   │   ├── score_network.py # Score-based denoising
│   │   └── encodings/    # Positional & hash encodings
│   ├── losses/           # Loss functions
│   │   ├── sdf_losses.py # Surface, freespace, eikonal
│   │   ├── planar_losses.py # Planar consistency, Manhattan
│   │   └── regularization.py # Smoothness, TV losses
│   ├── training/         # Training infrastructure
│   │   ├── trainer.py    # Main trainer class
│   │   ├── scheduler.py  # LR schedulers with warmup
│   │   ├── checkpoint.py # Checkpoint management
│   │   └── samplers.py   # Point cloud samplers
│   ├── refinement/       # Post-processing
│   │   ├── mesh_extraction.py # Marching cubes
│   │   ├── statistical.py # Outlier removal, filtering
│   │   ├── neural_refinement.py # Neural point refinement
│   │   └── point_completion.py # Completion network
│   └── utils/            # Utilities
│       ├── visualization.py # Point cloud/mesh viz
│       ├── io.py         # File I/O utilities
│       └── metrics.py    # Evaluation metrics
├── scripts/               # Executable scripts
├── tests/                 # Unit tests
├── notebooks/             # Jupyter notebooks
├── output/                # Training outputs
├── data/                  # Dataset directory
└── docs/                  # Documentation
```

## Configuration

The pipeline uses YAML configuration files. Key settings:

```yaml
# config/default.yaml
model:
  hidden_features: 256
  hidden_layers: 4
  encoding: hashgrid  # or 'positional'

training:
  epochs: 30
  batch_size: 4
  learning_rate: 1e-4
  
losses:
  surface_weight: 1.0
  freespace_weight: 0.5
  eikonal_weight: 0.1
  planar_weight: 0.3
  manhattan_weight: 0.1

refinement:
  use_sor: true
  use_neural_filter: true
  use_score_denoise: true
  filter_iterations: 15
```

## Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | RTX 3080 (10GB) | RTX 4090 (24GB) |
| RAM | 32GB | 64GB |
| Storage | 100GB SSD | 500GB NVMe |

## License

MIT License - see LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{mobilex_neural_recon,
  title={MobileX Poles Neural 3D Reconstruction Pipeline},
  year={2025},
  url={<repository-url>}
}
```

## Acknowledgments

This project incorporates ideas from:
- Neural 3D Scene Reconstruction with Indoor Planar Priors (TPAMI 2024)
- 3DMambaIPF for point cloud filtering
- Score-Based Generative Models for denoising
- GenPC for point cloud completion
