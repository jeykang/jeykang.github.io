# MobileX Poles Neural 3D Reconstruction Pipeline

## Architecture Overview

This document outlines the comprehensive architecture for converting multi-camera surveillance video feeds into high-fidelity, watertight 3D point clouds for coverage analysis.

---

## 1. System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              INPUT LAYER                                        │
├─────────────────────────────────────────────────────────────────────────────────┤
│  6x e-Con AR0234CS Cameras (105° HFOV, 1280x720)                               │
│  └─> RGB Streams + Depth Streams + IMU/Odometry                                 │
│  └─> Ground Truth Poses (TUM Format: timestamp tx ty tz qx qy qz qw)           │
└─────────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          DATA PROCESSING LAYER                                  │
├─────────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐                 │
│  │ Multi-Camera    │  │ Temporal        │  │ Depth           │                 │
│  │ Synchronization │─>│ Alignment       │─>│ Preprocessing   │                 │
│  │ (±10ms)         │  │ (Interpolation) │  │ (Filtering)     │                 │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘                 │
│           │                    │                    │                           │
│           └────────────────────┼────────────────────┘                           │
│                                ▼                                                │
│                   ┌─────────────────────────┐                                   │
│                   │  Unified Frame Buffer   │                                   │
│                   │  (6-cam synchronized)   │                                   │
│                   └─────────────────────────┘                                   │
└─────────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                       FEATURE EXTRACTION LAYER                                  │
├─────────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────┐  ┌─────────────────────┐  ┌─────────────────────────┐ │
│  │ Semantic Segmentor  │  │ Plane Detector      │  │ Normal Estimator        │ │
│  │ (DeepLabV3+/SegFormer│  │ (PlaneRCNN-style)   │  │ (Omnidata/Metric3D)     │ │
│  │  - Walls, Floor,    │  │  - RANSAC + Neural  │  │  - Per-pixel normals    │ │
│  │    Ceiling, Objects)│  │  - Manhattan Prior  │  │  - Surface orientation  │ │
│  └─────────────────────┘  └─────────────────────┘  └─────────────────────────┘ │
│           │                        │                          │                 │
│           └────────────────────────┼──────────────────────────┘                 │
│                                    ▼                                            │
│                   ┌─────────────────────────────┐                               │
│                   │  Feature Fusion Module      │                               │
│                   │  (Geometric + Semantic)     │                               │
│                   └─────────────────────────────┘                               │
└─────────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                     NEURAL RECONSTRUCTION LAYER                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌───────────────────────────────────────────────────────────────────────────┐ │
│  │                    Neural SDF Network (SIREN + Planar)                    │ │
│  │  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐    │ │
│  │  │ Positional  │──>│ SIREN       │──>│ Planar      │──>│ SDF + Color │    │ │
│  │  │ Encoding    │   │ Backbone    │   │ Attention   │   │ Output      │    │ │
│  │  │ (HashGrid)  │   │ (256x4)     │   │ Module      │   │ Head        │    │ │
│  │  └─────────────┘   └─────────────┘   └─────────────┘   └─────────────┘    │ │
│  └───────────────────────────────────────────────────────────────────────────┘ │
│                                                                                 │
│  Training Objectives:                                                           │
│  ├─ L_surface: |SDF(p_surf)| = 0                                               │
│  ├─ L_freespace: SDF(p_free) = d_ray                                           │
│  ├─ L_eikonal: ||∇SDF|| = 1                                                    │
│  ├─ L_planar: enforce planarity in detected regions                            │
│  ├─ L_normal: align SDF gradients with estimated normals                       │
│  └─ L_manhattan: axis-aligned constraint for walls/floor/ceiling               │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                       MESH EXTRACTION LAYER                                     │
├─────────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐                 │
│  │ Adaptive        │  │ FlexiCubes      │  │ Dense Point     │                 │
│  │ Marching Cubes  │─>│ Mesh Extraction │─>│ Cloud Sampling  │                 │
│  │ (Multi-res)     │  │ (Differentiable)│  │ (Poisson Disk)  │                 │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘                 │
└─────────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                      ERROR CORRECTION LAYER                                     │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  Stage 1: Geometric Filtering                                                   │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │  Statistical Outlier Removal (SOR) → Radius Outlier Removal (ROR)       │   │
│  │  └─> Remove obvious noise and isolated points                            │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                      │                                          │
│                                      ▼                                          │
│  Stage 2: Neural Point Cloud Filtering (3DMambaIPF-inspired)                   │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │  ┌───────────────┐   ┌───────────────┐   ┌───────────────┐              │   │
│  │  │ Local Feature │──>│ State Space   │──>│ Displacement  │              │   │
│  │  │ Extraction    │   │ Model (Mamba) │   │ Prediction    │              │   │
│  │  │ (PointNet++)  │   │ O(n) complex. │   │ Δp for each pt│              │   │
│  │  └───────────────┘   └───────────────┘   └───────────────┘              │   │
│  │                                                                          │   │
│  │  Iterative refinement: p_{t+1} = p_t + Δp_t (10-20 iterations)          │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                      │                                          │
│                                      ▼                                          │
│  Stage 3: Score-Based Denoising (SGM-inspired)                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │  Score Network s_θ(p, σ) estimates: ∇_p log p_σ(p)                      │   │
│  │  Langevin Dynamics: p_{t+1} = p_t + ε·s_θ(p_t, σ_t) + √(2ε)·z          │   │
│  │  └─> Gradually moves points toward high-density manifold                 │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                       COMPLETION LAYER                                          │
├─────────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │  GenPC-inspired Zero-Shot Completion                                     │   │
│  │  ├─ Detect holes/occlusions via visibility analysis                      │   │
│  │  ├─ Local latent diffusion for missing regions                           │   │
│  │  └─ Seamless blending with existing geometry                             │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
│  Alternative: Planar Inpainting for indoor environments                         │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │  ├─ Extend detected planes through occluded regions                      │   │
│  │  ├─ Fill floor/wall/ceiling gaps using Manhattan constraints             │   │
│  │  └─ Maintain watertight mesh topology                                    │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          OUTPUT LAYER                                           │
├─────────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐                 │
│  │ Dense Point     │  │ Watertight      │  │ Coverage        │                 │
│  │ Cloud (PLY)     │  │ Mesh (OBJ/GLB)  │  │ Analysis        │                 │
│  │ ~10M points     │  │ with textures   │  │ Visibility maps │                 │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Project Structure

```
cse_neural_recon/
├── config/
│   ├── default.yaml              # Default training configuration
│   ├── experiment/               # Experiment-specific configs
│   │   ├── cse_hospital.yaml
│   │   ├── cse_warehouse.yaml
│   │   └── cse_office.yaml
│   └── model/                    # Model architecture configs
│       ├── neural_sdf.yaml
│       ├── error_correction.yaml
│       └── completion.yaml
│
├── src/
│   ├── __init__.py
│   │
│   ├── data/                     # Data Loading & Processing
│   │   ├── __init__.py
│   │   ├── dataset.py            # Enhanced CSE dataset loader
│   │   ├── multi_camera.py       # Multi-camera synchronization
│   │   ├── transforms.py         # Data augmentation & preprocessing
│   │   └── depth_processing.py   # Depth filtering & hole filling
│   │
│   ├── features/                 # Feature Extraction
│   │   ├── __init__.py
│   │   ├── semantic.py           # Semantic segmentation wrapper
│   │   ├── plane_detection.py    # Neural plane detection
│   │   ├── normal_estimation.py  # Surface normal estimation
│   │   └── feature_fusion.py     # Multi-modal feature fusion
│   │
│   ├── models/                   # Neural Network Models
│   │   ├── __init__.py
│   │   ├── encodings/
│   │   │   ├── __init__.py
│   │   │   ├── positional.py     # Fourier/sinusoidal encoding
│   │   │   └── hashgrid.py       # Multi-resolution hash encoding
│   │   ├── neural_sdf.py         # Enhanced Neural SDF (SIREN + Planar)
│   │   ├── planar_attention.py   # Planar prior attention module
│   │   ├── point_filter.py       # 3DMambaIPF-style point filtering
│   │   ├── score_network.py      # Score-based denoising network
│   │   └── completion_net.py     # Point cloud completion network
│   │
│   ├── losses/                   # Loss Functions
│   │   ├── __init__.py
│   │   ├── sdf_losses.py         # Surface, freespace, eikonal losses
│   │   ├── planar_losses.py      # Planar consistency losses
│   │   ├── manhattan_losses.py   # Manhattan world constraints
│   │   ├── normal_losses.py      # Normal alignment losses
│   │   └── regularization.py     # Smoothness & sparsity terms
│   │
│   ├── geometry/                 # Geometric Operations
│   │   ├── __init__.py
│   │   ├── camera.py             # Camera models & projection
│   │   ├── transforms.py         # SE(3) transformations
│   │   ├── point_cloud.py        # Point cloud operations
│   │   ├── mesh_extraction.py    # Marching cubes & FlexiCubes
│   │   └── sampling.py           # Point sampling strategies
│   │
│   ├── refinement/               # Post-processing & Refinement
│   │   ├── __init__.py
│   │   ├── outlier_removal.py    # SOR, ROR implementations
│   │   ├── neural_filter.py      # Neural point filtering
│   │   ├── score_denoise.py      # Score-based denoising
│   │   ├── completion.py         # Hole filling & completion
│   │   └── watertight.py         # Watertight mesh enforcement
│   │
│   ├── training/                 # Training Infrastructure
│   │   ├── __init__.py
│   │   ├── trainer.py            # Main training loop
│   │   ├── scheduler.py          # Learning rate schedulers
│   │   ├── checkpointing.py      # Model checkpointing
│   │   └── distributed.py        # Multi-GPU training support
│   │
│   ├── evaluation/               # Evaluation Metrics
│   │   ├── __init__.py
│   │   ├── metrics.py            # Chamfer, F-score, etc.
│   │   ├── coverage.py           # Coverage analysis
│   │   └── visualization.py      # Result visualization
│   │
│   └── utils/                    # Utilities
│       ├── __init__.py
│       ├── config.py             # Configuration management
│       ├── logging.py            # Logging utilities
│       ├── io.py                 # File I/O helpers
│       └── visualization.py      # Debug visualization
│
├── scripts/                      # Executable Scripts
│   ├── train.py                  # Main training script
│   ├── inference.py              # Inference pipeline
│   ├── evaluate.py               # Evaluation script
│   ├── preprocess_data.py        # Data preprocessing
│   └── export_model.py           # Model export for deployment
│
├── tests/                        # Unit Tests
│   ├── test_dataset.py
│   ├── test_geometry.py
│   ├── test_models.py
│   └── test_refinement.py
│
├── notebooks/                    # Jupyter Notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_debugging.ipynb
│   └── 03_results_analysis.ipynb
│
├── output/                       # Training outputs
│   ├── checkpoints/
│   ├── logs/
│   └── results/
│
├── data/                         # Dataset directory (existing)
│
├── docs/                         # Documentation
│   ├── ARCHITECTURE.md           # This file
│   ├── TRAINING.md               # Training guide
│   └── DEPLOYMENT.md             # Deployment guide
│
├── requirements.txt              # Python dependencies
├── setup.py                      # Package setup
└── README.md                     # Project README
```

---

## 3. Core Module Specifications

### 3.1 Data Pipeline (`src/data/`)

#### Enhanced Dataset Loader
```python
# Key Features:
- Multi-camera support (6 cameras simultaneously)
- Stereo pair handling (left/right)
- Flexible timestamp synchronization (configurable tolerance)
- On-the-fly depth filtering and hole filling
- Memory-efficient frame caching
- Support for both training (random access) and inference (sequential)
```

#### Multi-Camera Synchronization
```python
# Synchronization Strategy:
- Master timestamp from odometry/IMU (highest frequency)
- Soft sync: Find nearest frames within tolerance window
- Hard sync: Interpolate poses for exact timestamps
- Camera extrinsics transformation to common frame
```

### 3.2 Feature Extraction (`src/features/`)

#### Semantic Segmentation
- **Model**: SegFormer (lightweight) or DeepLabV3+ (accurate)
- **Classes**: Wall, Floor, Ceiling, Door, Window, Furniture, Dynamic Objects
- **Purpose**: Guide planar prior application, mask dynamic objects

#### Plane Detection
- **Method**: Hybrid RANSAC + Neural refinement
- **Features**: Per-plane normal, offset, boundary mask
- **Manhattan Prior**: Cluster planes into 3 orthogonal groups

#### Normal Estimation
- **Model**: Omnidata or Metric3D pretrained
- **Output**: Dense normal map aligned with depth
- **Use**: Supervise SDF gradient direction

### 3.3 Neural SDF Model (`src/models/`)

#### Architecture
```python
Input: 3D coordinate (x, y, z) + optional features
       ↓
Positional Encoding: Multi-resolution Hash Grid (16 levels, 2^14 to 2^24)
       ↓
SIREN Backbone: 4 layers × 256 units, ω₀=30
       ↓
Planar Attention: Cross-attention with detected plane features
       ↓
Output Heads:
  - SDF value (scalar)
  - RGB color (3-dim)
  - Semantic logits (N classes)
```

#### Training Losses
| Loss | Weight | Description |
|------|--------|-------------|
| L_surface | 1.0 | |SDF(p_surface)| → 0 |
| L_freespace | 0.5 | SDF(p_free) → d_ray |
| L_eikonal | 0.1 | ||∇SDF|| → 1 |
| L_planar | 0.3 | Enforce planarity in plane regions |
| L_normal | 0.2 | ∇SDF ∥ estimated_normal |
| L_manhattan | 0.1 | Axis-alignment for structural planes |
| L_color | 0.1 | RGB reconstruction (optional) |

### 3.4 Error Correction Pipeline (`src/refinement/`)

#### Stage 1: Classical Filtering
```python
# Statistical Outlier Removal
- k_neighbors: 50
- std_ratio: 2.0

# Radius Outlier Removal  
- radius: 0.05m
- min_neighbors: 10
```

#### Stage 2: Neural Point Filtering (3DMambaIPF-style)
```python
Architecture:
- PointNet++ encoder for local features
- State Space Model (S4/Mamba) for global context
- MLP decoder for per-point displacement

Training:
- Input: Noisy point cloud
- Target: Ground truth clean surface
- Loss: Chamfer Distance + Earth Mover's Distance

Inference:
- Iterative: 10-20 refinement steps
- p_{t+1} = p_t + α * Δp_t (α=0.5 for stability)
```

#### Stage 3: Score-Based Denoising
```python
# Score Network s_θ(p, σ)
Architecture:
- Point cloud encoder (PointNet++ or DGCNN)
- Noise level embedding (σ → 256-dim)
- Score prediction head

# Annealed Langevin Dynamics
σ_levels = [0.1, 0.05, 0.02, 0.01, 0.005]
for σ in σ_levels:
    for t in range(T_per_level):
        score = s_θ(points, σ)
        noise = torch.randn_like(points)
        points = points + (ε * score) + sqrt(2*ε) * noise
```

### 3.5 Point Cloud Completion (`src/refinement/completion.py`)

#### Hole Detection
```python
# Visibility-based hole detection
1. Render point cloud from training camera poses
2. Find pixels with no points (holes)
3. Back-project hole regions to 3D
4. Cluster into completion regions
```

#### Completion Methods
1. **Planar Extension**: For walls/floor/ceiling, extend detected planes
2. **Neural Completion**: GenPC-style diffusion for complex geometry
3. **Hybrid**: Use semantic labels to choose method per region

---

## 4. Training Strategy

### 4.1 Multi-Stage Training

```
Stage 1: Coarse SDF (10 epochs)
├─ Resolution: 64³ grid
├─ Losses: L_surface + L_freespace
├─ LR: 1e-3
└─ Purpose: Learn rough scene geometry

Stage 2: Fine SDF + Planar (20 epochs)
├─ Resolution: 256³ grid
├─ Losses: All SDF losses + L_planar + L_manhattan
├─ LR: 1e-4
└─ Purpose: Sharp edges, flat surfaces

Stage 3: Error Correction Training (10 epochs)
├─ Freeze SDF network
├─ Train point filter on extracted clouds
├─ Augment with synthetic noise
└─ Purpose: Learn to denoise/refine

Stage 4: End-to-End Fine-tuning (5 epochs)
├─ Unfreeze all
├─ Lower LR: 1e-5
└─ Purpose: Joint optimization
```

### 4.2 Data Augmentation
- Random rotation around gravity axis (z)
- Random translation (±0.5m)
- Depth noise injection (Gaussian σ=0.01m)
- Random camera dropout (simulate occlusion)
- Color jittering (brightness, contrast)

### 4.3 Hardware Requirements
| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | RTX 3080 (10GB) | RTX 4090 (24GB) |
| RAM | 32GB | 64GB |
| Storage | 100GB SSD | 500GB NVMe |
| Training Time | ~24h (1 scene) | ~8h (1 scene) |

---

## 5. Inference Pipeline

```python
def inference_pipeline(sequence_path, config):
    # 1. Load calibration and poses
    cameras = load_multi_camera_calibration(sequence_path)
    poses = load_poses(sequence_path)
    
    # 2. Load pretrained models
    sdf_model = load_sdf_model(config.sdf_checkpoint)
    filter_model = load_filter_model(config.filter_checkpoint)
    
    # 3. Incremental reconstruction
    global_cloud = PointCloud()
    
    for frame_idx in tqdm(range(len(poses))):
        # Get synchronized frame from all cameras
        frame = get_synchronized_frame(cameras, poses[frame_idx])
        
        # Extract features
        semantics = semantic_model(frame.rgb)
        planes = plane_detector(frame.depth, frame.rgb)
        normals = normal_estimator(frame.rgb)
        
        # Generate points from depth
        points = depth_to_world_points(
            frame.depth, frame.intrinsics, frame.pose
        )
        
        # Filter points by semantics (remove dynamic objects)
        static_mask = semantics != DYNAMIC_CLASS
        points = points[static_mask]
        
        # Accumulate
        global_cloud.add(points)
        
        # Periodic cleanup
        if frame_idx % 100 == 0:
            global_cloud = voxel_downsample(global_cloud, voxel_size=0.02)
    
    # 4. Neural SDF fitting (for watertight mesh)
    sdf_model.fit(global_cloud, planes, normals)
    
    # 5. Extract mesh
    mesh = marching_cubes(sdf_model, resolution=512)
    
    # 6. Sample dense point cloud
    dense_cloud = mesh.sample_points_poisson_disk(n_points=10_000_000)
    
    # 7. Error correction
    refined_cloud = filter_model.refine(dense_cloud, iterations=15)
    refined_cloud = score_denoise(refined_cloud, score_model)
    
    # 8. Completion
    holes = detect_holes(refined_cloud, cameras, poses)
    completed_cloud = complete_holes(refined_cloud, holes, planes)
    
    # 9. Final watertight mesh
    final_mesh = poisson_reconstruction(completed_cloud)
    
    return final_mesh, completed_cloud
```

---

## 6. Evaluation Metrics

### 6.1 Reconstruction Quality
- **Chamfer Distance (CD)**: Mean nearest-neighbor distance
- **F-Score**: Precision/recall at distance threshold
- **Normal Consistency (NC)**: Dot product of predicted vs GT normals

### 6.2 Watertightness
- **Hole Count**: Number of boundary edges in mesh
- **Volume Validity**: Check if mesh is closed manifold

### 6.3 Coverage Analysis
- **Visibility Coverage**: % of floor area visible from camera poses
- **Reconstruction Coverage**: % of GT scene reconstructed
- **Surveillance Score**: Custom metric for security coverage

---

## 7. Key Innovations

1. **Manhattan-World Planar Priors**: Enforce axis-aligned planes for indoor structural elements, dramatically improving wall/floor/ceiling reconstruction.

2. **Hybrid Error Correction**: Combine classical filtering (fast, robust) with neural refinement (accurate, adaptive) in a multi-stage pipeline.

3. **Multi-Camera Fusion**: Proper handling of 6-camera rig with temporal synchronization and extrinsic calibration.

4. **Semantic-Guided Completion**: Use semantic labels to choose appropriate completion strategy per region.

5. **Iterative Point Filtering**: 3DMambaIPF-inspired approach with linear complexity for large-scale point clouds.

---

## 8. Implementation Roadmap

### Phase 1: Foundation (Week 1-2)
- [x] Enhanced data loader with multi-camera support
- [x] Camera geometry utilities
- [x] Basic Neural SDF with SIREN
- [ ] Configuration system

### Phase 2: Core Features (Week 3-4)
- [ ] Planar detection and Manhattan constraints
- [ ] Normal estimation integration
- [ ] Extended loss functions
- [ ] Mesh extraction pipeline

### Phase 3: Error Correction (Week 5-6)
- [ ] Classical filtering module
- [ ] Neural point filter (3DMambaIPF-style)
- [ ] Score-based denoising
- [ ] Integration testing

### Phase 4: Completion & Polish (Week 7-8)
- [ ] Hole detection
- [ ] Planar inpainting
- [ ] Neural completion
- [ ] End-to-end evaluation

### Phase 5: Deployment (Week 9-10)
- [ ] Jetson optimization
- [ ] Real-time processing pipeline
- [ ] Coverage analysis tools
- [ ] Documentation & tutorials

---

*Document Version: 1.0*
*Last Updated: 2025*
