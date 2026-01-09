# OpenPCSeg - Fusion Architecture Guide

This document focuses on the **fusion** components of OpenPCSeg, an open-source point cloud segmentation toolbox for autonomous driving.

## Project Structure (Fusion-Relevant)

```
D:\OpenPCSeg/
├── pcseg/
│   ├── model/
│   │   └── segmentor/
│   │       └── fusion/                    # FUSION MODELS
│   │           ├── rpvnet/
│   │           │   ├── rpvnet.py          # Main RPVNet implementation
│   │           │   ├── swin_range_branch.py  # Swin Transformer range branch
│   │           │   ├── utils.py           # Voxel-Point conversion utilities
│   │           │   └── range_lib/         # Custom CUDA kernels
│   │           │       ├── denselize.py   # Point features to range images
│   │           │       └── map_count.py   # Projection counting
│   │           └── spvcnn/
│   │               └── spvcnn.py          # Point-Voxel CNN (no range branch)
│   └── data/
│       └── dataset/
│           ├── semantickitti/
│           │   └── semantickitti_fusion.py  # SemanticKITTI fusion dataset
│           └── waymo/
│               └── waymo_fusion.py          # Waymo fusion dataset
├── tools/
│   └── cfgs/
│       └── fusion/                        # CONFIGURATION FILES
│           ├── semantic_kitti/
│           │   ├── rpvnet_mk18_cr10.yaml
│           │   ├── rpvnet_mk34_cr17_5.yaml
│           │   ├── rpvnet_swin_tiny_cr17_5.yaml
│           │   └── spvcnn_*.yaml
│           └── waymo/
└── train.py, infer.py
```

## Fusion Architecture Overview

### Multi-Modal Fusion Strategy

OpenPCSeg implements **Range-Point-Voxel (RPV) Fusion** - a three-branch architecture that combines:

1. **Range View** (2D projection): Spherical projection of LiDAR as 2D image
2. **Voxel View** (3D grid): Regular 3D voxel grid (typically 0.05m resolution)
3. **Point Cloud** (3D scattered): Original sparse point representation

**Key Insight**: Points serve as a **communication hub** between voxel and range representations, enabling efficient feature exchange without expensive view-specific transformations.

### RPVNet Architecture

```
INPUT: Point Cloud (xyz, intensity, ring_id) + Range Image [B, 5, 64, 2048]
         │
         ├────────────────┬────────────────┬─────────────────
         │                │                │
    VOXEL BRANCH     POINT BRANCH     RANGE BRANCH
    (Sparse 3D)      (MLPs only)      (2D CNN or Swin)
         │                │                │
    [MinkUNet]       [Linear+BN+ReLU]  [SalsaNext/Swin]
         │                │                │
         └────────────────┼────────────────┘
                          │
                   POINT AS HUB ──────────────────
                   (Fusion Center)                │
                          │                       │
              ┌───────────┴───────────┐          │
              │                       │          │
         Voxel→Point             Range→Point     │
         (Trilinear)             (Grid Sample)   │
              │                       │          │
              └───────────┬───────────┘          │
                          │                      │
              z.F = voxel_feat + range_feat + point_mlp
                          │
                    CLASSIFIER
                          │
                    PREDICTIONS
```

## Key Components

### 1. Main Classes

| Class | File | Description |
|-------|------|-------------|
| `RPVNet` | `rpvnet.py:422` | Main fusion model with three branches |
| `SPVCNN` | `spvcnn/spvcnn.py:189` | Simpler point-voxel fusion (no range) |
| `SwinRangeBranch` | `swin_range_branch.py:132` | Swin Transformer range branch |
| `SwinRangeBranchWrapper` | `swin_range_branch.py:369` | API adapter for RPVNet |

### 2. Fusion Operations

Located in `rpvnet/utils.py`:

| Function | Purpose | Method |
|----------|---------|--------|
| `voxel_to_point()` | V→P transfer | Trilinear interpolation |
| `point_to_voxel()` | P→V transfer | Spatial hashing + pooling |
| `range_to_point()` | R→P transfer | `F.grid_sample` bilinear |
| `point_to_range()` | P→R transfer | Custom CUDA kernels |

### 3. Fusion Stages

RPVNet performs fusion at **4 stages** throughout the network:

| Stage | Location | Description |
|-------|----------|-------------|
| Stage 0 | After stem | Early feature fusion |
| Stage 1 | Bottleneck | Deep semantic fusion |
| Stage 2 | Mid-decoder | Mid-resolution fusion |
| Stage 3 | Final | High-resolution final fusion |

**Fusion Operation**: Simple element-wise addition
```python
z_fused.F = voxel_features + range_features + point_mlp_output
```

## Data Flow

### Input Preparation

The fusion datasets (`semantickitti_fusion.py`, `waymo_fusion.py`) prepare:

```python
batch_dict = {
    'lidar': SparseTensor,           # [N_pts, 5] features + [N_pts, 4] coords
    'range_image': Tensor,           # [B, 5, 64, 2048] for SemanticKITTI
    'range_pxpy': Tensor,            # [N_pts, 3] = [batch_id, px, py]
    'targets': SparseTensor,         # Semantic labels
    'offset': Tensor,                # Cumulative point counts
}
```

### Range Image Channels

```
Channel 0: Inverse depth (1/distance)
Channel 1: Reflectivity (intensity)
Channels 2-4: xyz coordinates
```

### Forward Pass Summary

1. **Stem**: Process voxel and range inputs through initial layers
2. **Fusion Stage 0**: Combine early features at point locations
3. **Encoder**: Progressive downsampling in both branches
4. **Fusion Stage 1**: Combine bottleneck features
5. **Decoder**: Progressive upsampling with skip connections
6. **Fusion Stages 2-3**: Combine mid and final features
7. **Classifier**: Multi-scale concat → class logits

## Configuration

### Key Model Parameters

```yaml
MODEL:
    NAME: RPVNet
    IN_FEATURE_DIM: 5                  # xyz, intensity, ring_id
    BLOCK: Bottleneck                  # ResBlock or Bottleneck
    NUM_LAYER: [2, 3, 4, 6, 2, 2, 2, 2]
    PLANES: [32, 32, 64, 128, 256, 256, 128, 96, 96]
    cr: 1.75                           # Channel multiplier

    # Range Branch Options
    RANGE_BRANCH: 'SwinRangeBranch'    # or 'SalsaNext'
    SWIN_VARIANT: 'swin_tiny_patch4_window7_224'
    SWIN_PRETRAINED: True
    RANGE_IMG_SIZE: [64, 2048]
```

### Training Settings

```yaml
OPTIM:
    BATCH_SIZE_PER_GPU: 8
    NUM_EPOCHS: 50
    OPTIMIZER: adamw
    LR_RANGE_RATIO: 0.1                # Differential LR for range branch
    FREEZE_RANGE_EPOCHS: 10            # Two-stage training strategy
```

## Dependencies

### Core Libraries

| Library | Purpose | Usage |
|---------|---------|-------|
| `torchsparse` | Sparse 3D convolutions | Voxel branch (custom build in `package/`) |
| `timm` | Vision Transformers | Swin range branch |
| `range_lib` | Range image ops | Custom CUDA kernels |

### Import Patterns

```python
# Sparse voxel operations
import torchsparse
import torchsparse.nn as spnn
from torchsparse import PointTensor, SparseTensor

# Swin Transformer
import timm

# Custom range operations
import range_utils.nn.functional as rnf
```

## Performance

| Model | Dataset | mIoU | Training Time |
|-------|---------|------|---------------|
| RPVNet (MK34, cr=1.75) | SemanticKITTI | 68.86 | 14.5h (2x A100) |
| SPVCNN (MK34, cr=1.6) | SemanticKITTI | 68.58 | - |
| SPVCNN | Waymo Open | 69.37 | 28h |

## Common Development Tasks

### Adding a New Range Branch

1. Implement in `pcseg/model/segmentor/fusion/rpvnet/`
2. Ensure it outputs multi-scale features matching expected dimensions
3. Create a wrapper with `stem()`, `encoder()`, `decoder()` methods
4. Register in `rpvnet.py` initialization

### Modifying Fusion Strategy

The fusion logic is in `RPVNet.forward()` (rpvnet.py:644+):
- Modify how `z*.F` is computed to change fusion method
- Current: additive fusion (`z.F = v + r + p`)
- Alternatives: concatenation, attention-based, gating

### Creating New Configs

Copy existing config from `tools/cfgs/fusion/` and modify:
- `cr`: Channel multiplier (affects model size)
- `NUM_LAYER`: Depth per stage
- `RANGE_BRANCH`: CNN vs Transformer
- `FREEZE_RANGE_EPOCHS`: Training schedule

## File Quick Reference

| Task | Key File(s) |
|------|-------------|
| Modify fusion model | `pcseg/model/segmentor/fusion/rpvnet/rpvnet.py` |
| Change Swin config | `pcseg/model/segmentor/fusion/rpvnet/swin_range_branch.py` |
| Edit voxel↔point ops | `pcseg/model/segmentor/fusion/rpvnet/utils.py` |
| Modify data loading | `pcseg/data/dataset/semantickitti/semantickitti_fusion.py` |
| Add training config | `tools/cfgs/fusion/semantic_kitti/*.yaml` |
| Custom CUDA kernels | `pcseg/model/segmentor/fusion/rpvnet/range_lib/` |
