# RPVNet Architecture: Complete Breakdown

## Table of Contents
1. [Overview](#overview)
2. [Three-Branch Architecture](#three-branch-architecture)
3. [Cross-View Fusion Mechanism](#cross-view-fusion-mechanism)
4. [Complete Forward Pass Analysis](#complete-forward-pass-analysis)
5. [Implementation Details](#implementation-details)
6. [Adapting to Two-View (Voxel + Range)](#adapting-to-two-view-voxel--range)

---

## Overview

RPVNet addresses the fundamental trade-off in LiDAR segmentation: **each representation has unique strengths and weaknesses**:

| View | Strengths | Weaknesses |
|------|-----------|------------|
| **Point** | Geometrically accurate, no quantization | Unordered, expensive neighbor search |
| **Voxel** | Regular 3D grid, good spatial locality | Sparse, quantization loss, O(n³) complexity |
| **Range** | Dense 2D, efficient CNNs | Distorts geometry, occlusion artifacts |

**Core Innovation**: Use **points as a communication hub** to shuttle features between voxel and range views repeatedly throughout the network, allowing each view to complement the others at multiple scales.

---

## Three-Branch Architecture

### Branch 1: Voxel Branch (Sparse 3D UNet)

```
Input: SparseTensor (voxelized point cloud)
Architecture: MinkUNet-style with Bottleneck blocks

stem (32 channels)
   ↓
stage1: downsample + 2 blocks (32→128 with expansion=4)
   ↓
stage2: downsample + 3 blocks (64→256)
   ↓
stage3: downsample + 4 blocks (128→512)
   ↓
stage4: downsample + 6 blocks (256→1024) [bottleneck]
   ↓
up1: upsample + 2 blocks (256→1024) + skip from stage3
   ↓
up2: upsample + 2 blocks (128→512) + skip from stage2
   ↓
up3: upsample + 2 blocks (96→384) + skip from stage1
   ↓
up4: upsample + 2 blocks (96→384) + skip from stem
```

**Key Features:**
- Uses sparse convolutions (torchsparse library)
- Bottleneck blocks with expansion=4 (channels × 4 in residual path)
- Standard U-Net skip connections
- Voxel resolution: 0.05m (configurable)

### Branch 2: Range Branch (SalsaNext - 2D UNet)

```
Input: Range Image [B, 5, H, W] (H=64, W=2048 for SemanticKITTI)
        5 channels: [x, y, z, intensity, range]

stem: 3× ResContextBlock (32 channels)
   ↓
stage1: ResBlock + AvgPool (32→56)
   ↓
stage2: ResBlock + AvgPool (56→112)
   ↓
stage3: ResBlock + AvgPool (112→224)
   ↓
stage4: ResBlock + AvgPool (224→448)
   ↓
mid_stage: ResBlock (no pooling)
   ↓
up1: PixelShuffle↑ + skip (448→448)
   ↓
up2: PixelShuffle↑ + skip (448→224)
   ↓
up3: PixelShuffle↑ + skip (168→168)
   ↓
up4: PixelShuffle↑ + skip (168→168)
```

**Key Features:**
- Standard 2D convolutions on spherical projection
- ResContextBlock: uses dilated convolutions (dilation=2) for wider receptive field
- PixelShuffle for upsampling (sub-pixel convolution)
- cr=1.75 channel multiplier for all layers

### Branch 3: Point Branch (Lightweight MLPs)

```python
point_transforms = [
    # Transform 0: After stem
    Linear(in_feature_dim → 32) + BN + ReLU

    # Transform 1: After stage4 (bottleneck)
    Linear(32 → 1024) + BN + ReLU

    # Transform 2: After up2
    Linear(1024 → 512) + BN + ReLU

    # Transform 3: After up4 (final)
    Linear(512 → 384) + BN + ReLU
]
```

**Key Features:**
- **No neighbor search** - just per-point MLPs
- Acts as a "learnable residual pathway" for point features
- Channels match the corresponding fusion stage dimensions
- Much cheaper than PointNet++ or KPConv

---

## Cross-View Fusion Mechanism

### The Central Role of Points

Points serve as the **universal coordinate system** that bridges voxel cells and range pixels:

```
     Voxel Grid          Points (Hub)         Range Image
         ║                   ║                     ║
         ║                   ║                     ║
    [V2P: gather] ←────── Point i ──────→ [R2P: gather]
         ║                   ║                     ║
         ║               [Fusion]                  ║
         ║                   ║                     ║
    [P2V: scatter] ←────── Point i ──────→ [P2R: scatter]
         ║                   ║                     ║
```

### Detailed Operation Flow

#### Step 1: Voxel-to-Point (V2P) - Trilinear Interpolation

**Implementation**: `voxel_to_point(x, z)` in `utils.py:69-105`

For each point at continuous coordinates `(x, y, z)`:

1. Find 8-neighborhood voxel grid (2×2×2 cube)
2. Compute trilinear interpolation weights:
   ```
   w_i = (1 - |x - x_i|) × (1 - |y - y_i|) × (1 - |z - z_i|)
   ```
3. Weighted sum of 8 voxel features:
   ```python
   point_feat = Σ w_i × voxel_feat[i]  # i ∈ [0,7]
   ```

**Key Code**:
```python
off = get_kernel_offsets(2, x.s, 1, device=z.F.device)  # 8 neighbors
old_hash = F.sphash(floor(z.C / voxel_size) * voxel_size, off)
idx_query = F.sphashquery(old_hash, pc_hash)
weights = F.calc_ti_weights(z.C, idx_query, scale=voxel_size)  # trilinear
new_feat = F.spdevoxelize(x.F, idx_query, weights)
```

#### Step 2: Range-to-Point (R2P) - Bilinear Interpolation

**Implementation**: `range_to_point()` in `rpvnet.py:49-51`

For each point with range image coordinates `(u, v)` ∈ `[-1, 1]²`:

1. Find 4 surrounding pixels in range image
2. Compute bilinear weights:
   ```
   w_tl = (1 - Δu) × (1 - Δv)  # top-left
   w_tr = Δu × (1 - Δv)        # top-right
   w_bl = (1 - Δu) × Δv        # bottom-left
   w_br = Δu × Δv              # bottom-right
   ```
3. Interpolate features:
   ```python
   point_feat = w_tl×F_tl + w_tr×F_tr + w_bl×F_bl + w_br×F_br
   ```

**Key Code**:
```python
# Uses PyTorch's F.grid_sample with mode='bilinear'
one_pxpy = pxpy[:, 1:].unsqueeze(0).unsqueeze(0)  # Nx2 → 1x1xNx2
one_resampled = F.grid_sample(feature_map, one_pxpy, mode='bilinear')
```

#### Step 3: Point-Level Fusion (Simple Addition)

**Implementation**: Lines 650-651, 667-668, 685-686, 703-704

```python
# At each fusion stage:
z_voxel = voxel_to_point(x_voxel, z)      # V2P
z_range = range_to_point(r_image, pxpy)  # R2P
z_point = point_transform(z.F)            # Point MLP

# Simple addition (NOT gated!)
z.F = z_voxel.F + z_range + z_point
```

**Important Note**: The paper describes a "Gated Fusion Module" with learnable attention weights, but **the actual implementation uses simple addition**. This is a common practical simplification that still works well.

#### Step 4: Point-to-Voxel (P2V) - Scatter Mean

**Implementation**: `point_to_voxel()` in `utils.py:41-64`

For each voxel cell, collect all points that fall inside and average their features:

```python
# Pseudo-code
for voxel_idx in all_voxels:
    points_in_voxel = find_points_in_cell(voxel_idx)
    voxel_feat[voxel_idx] = mean(point_feats[points_in_voxel])
```

**Actual Implementation**:
```python
pc_hash = F.sphash(floor(z.C / voxel_size) * voxel_size)  # Point→voxel mapping
sparse_hash = F.sphash(x.C)                                # Existing voxels
idx_query = F.sphashquery(pc_hash, sparse_hash)           # Find which voxel each point belongs to
counts = F.spcount(idx_query, num_voxels)                 # Points per voxel
inserted_feat = F.spvoxelize(z.F, idx_query, counts)      # Average pooling
```

#### Step 5: Point-to-Range (P2R) - Scatter Mean

**Implementation**: `point_to_range()` in `rpvnet.py:73-91`

For each range pixel `(u, v)`, collect all points that project to it and average:

```python
# Convert normalized coords [-1,1] to pixel coords [0, H-1]×[0, W-1]
int_pxpy = ((pxpy[:, 1:] + 1) / 2) * [W-1, H-1]

# Count points per pixel
cm = rnf.map_count(int_pxpy, batch_size, H, W)

# Average pooling via custom CUDA kernel
fm = rnf.denselize(point_features, cm, int_pxpy)
```

---

## Complete Forward Pass Analysis

Let's trace the exact forward pass through `rpvnet.py:632-716`:

### Stage 0: Stem + First Fusion

```python
# Line 643: Initialize voxel representation
x0 = initial_voxelize(z, pres=0.05, vres=0.05)  # SparseTensor

# Lines 645-646: Parallel stems
r_x0 = range_branch.stem(range_image)  # [B,32,H,W]
x0 = voxel_stem(x0)                     # SparseTensor with 32 channels

# Lines 648-651: FUSION #0
z0_voxel = voxel_to_point(x0, z)                    # V2P
z0_range = range_to_point(r_x0, range_pxpy)       # R2P
z0_point = point_transforms[0](z.F)                # Point MLP
z0.F = z0_voxel.F + z0_range + z0_point           # FUSE

# Line 653-658: Update branches with fused features
x1 = point_to_voxel(x0, z0)  # P2V: Write back to voxel
x1 = voxel_stage1(x1)
x2 = voxel_stage2(x1)
x3 = voxel_stage3(x2)
x4 = voxel_stage4(x3)

r_x1 = point_to_range(z0.F, range_pxpy, B, H, W)  # P2R: Write to range
r_x1, r_s1 = range_stage1(r_x1)
r_x2, r_s2 = range_stage2(r_x1)
r_x3, r_s3 = range_stage3(r_x2)
r_x4, r_s4 = range_stage4(r_x3)
r_x4 = range_mid_stage(r_x4)
```

**What's happening**: After stem processing, features are fused at point locations, then the fused point features are scattered back to initialize deeper voxel and range branch processing.

### Stage 1: Bottleneck Fusion

```python
# Lines 665-668: FUSION #1 (after stage4 bottleneck)
z1_voxel = voxel_to_point(x4, z0)                   # V2P
z1_range = range_to_point(r_x4, range_pxpy)       # R2P
z1_point = point_transforms[1](z0.F)               # Point MLP (1024 channels)
z1.F = z1_voxel.F + z1_range + z1_point           # FUSE

# Lines 670-680: Update branches
y1 = point_to_voxel(x4, z1)    # P2V
r_y1 = point_to_range(z1.F, range_pxpy, B, r_x4.size(2), r_x4.size(3))  # P2R

y1.F = dropout(y1.F)
y1 = voxel_up1[0](y1)          # Deconv
y1 = cat([y1, x3])             # Skip connection
y1 = voxel_up1[1](y1)          # Process

y2 = voxel_up2[0](y1)
y2 = cat([y2, x2])
y2 = voxel_up2[1](y2)

r_y1 = range_up1(r_y1, r_s4)   # Upsample + skip
r_y2 = range_up2(r_y1, r_s3)
```

**Key**: Bottleneck features (highest semantic level) are fused and redistributed.

### Stage 2: Mid-Upsample Fusion

```python
# Lines 683-689: FUSION #2 (after up2)
z2_voxel = voxel_to_point(y2, z1)                   # V2P
z2_range = range_to_point(r_y2, range_pxpy)       # R2P
z2_point = point_transforms[2](z1.F)               # Point MLP (512 channels)
z2.F = z2_voxel.F + z2_range + z2_point           # FUSE

# Lines 688-698: Update branches
y3 = point_to_voxel(y2, z2)    # P2V
r_y3 = point_to_range(z2.F, range_pxpy, B, r_y2.size(2), r_y2.size(3))  # P2R

y3.F = dropout(y3.F)
y3 = voxel_up3[0](y3)
y3 = cat([y3, x1])
y3 = voxel_up3[1](y3)

y4 = voxel_up4[0](y3)
y4 = cat([y4, x0])
y4 = voxel_up4[1](y4)

r_y3 = range_up3(r_y3, r_s2)
r_y4 = range_up4(r_y3, r_s1)
```

### Stage 3: Final Fusion

```python
# Lines 701-704: FUSION #3 (after up4, full resolution)
z3_voxel = voxel_to_point(y4, z2)                   # V2P
z3_range = range_to_point(r_y4, range_pxpy)       # R2P
z3_point = point_transforms[3](z2.F)               # Point MLP (384 channels)
z3.F = z3_voxel.F + z3_range + z3_point           # FUSE
```

### Multi-Scale Classification

```python
# Lines 706-716: Aggregate multi-scale features
if multi_scale == 'concat':
    # Concatenate features from 3 fusion stages
    # z1: 1024 ch (bottleneck)
    # z2: 512 ch (mid)
    # z3: 384 ch (final)
    out = classifier(cat([z1.F, z2.F, z3.F], dim=1))  # (1024+512+384) → num_class
```

**Rationale**: Multi-scale fusion captures both:
- High-level semantics (z1 from bottleneck)
- Mid-level patterns (z2)
- Fine-grained details (z3 at full resolution)

---

## Implementation Details

### Data Structures

1. **SparseTensor** (Voxel representation):
   ```python
   x = SparseTensor(features=F, coords=C, stride=s)
   # F: [N_voxels, C] features
   # C: [N_voxels, 4] coordinates (x, y, z, batch_idx)
   # s: voxel stride (resolution)
   ```

2. **PointTensor** (Point representation):
   ```python
   z = PointTensor(features=F, coords=C)
   # F: [N_points, C] features
   # C: [N_points, 4] continuous coordinates (x, y, z, batch_idx)
   # Additional: idx_query, weights for interpolation caching
   ```

3. **Range Image** (Dense 2D):
   ```python
   range_image: [B, 5, H, W]  # Standard tensor
   range_pxpy: [N_points, 3]  # (batch_idx, px, py) where px,py ∈ [-1,1]
   ```

### Key Configuration Parameters

```python
# From model_cfgs
IN_FEATURE_DIM: 4           # (x, y, z, intensity)
NUM_LAYER: [2,3,4,6,2,2,2,2] # Blocks per stage
BLOCK: 'Bottleneck'         # ResidualBlock or Bottleneck
PLANES: [32,32,64,128,256,256,128,96,96]
cr: 1.75                    # Channel multiplier
pres: 0.05                  # Point resolution for initial voxelization
vres: 0.05                  # Voxel resolution
DROPOUT_P: 0.3
MULTI_SCALE: 'concat'       # 'concat', 'sum', 'se', or None
```

### Fusion Stages Summary

| Stage | Location | Voxel Channels | Range Channels | Point Channels | Output Used In |
|-------|----------|----------------|----------------|----------------|----------------|
| 0 | After stem | 32 | 56 | 32 | Continue processing |
| 1 | After stage4 | 1024 | 448 | 1024 | **z1 → classifier** |
| 2 | After up2 | 512 | 224 | 512 | **z2 → classifier** |
| 3 | After up4 | 384 | 168 | 384 | **z3 → classifier** |

### Loss Function

```python
# Default configuration
LOSS_TYPES: ['CELoss', 'LovLoss']
LOSS_WEIGHTS: [1.0, 1.0]
KNN: 10  # For LovLoss neighborhood computation

# Combined loss
loss = 1.0 × CrossEntropy(pred, target) + 1.0 × LovaszLoss(pred, target)
```

---

## Adapting to Two-View (Voxel + Range)

### Why Remove the Point Branch?

The **Point Branch** in RPVNet serves as:
1. A learnable residual pathway for point features
2. An independent feature extractor (though very lightweight)

If you want a **pure 2-view fusion** (only voxel and range), you can simplify the architecture while **keeping points as the communication hub**.

### Required Modifications

#### 1. Remove Point Branch MLPs

**Delete**:
```python
# Line 571-592: Remove point_transforms
self.point_transforms = nn.ModuleList([...])  # DELETE THIS
```

#### 2. Simplify Fusion Operations

**Original 3-way fusion** (lines 650-651, 667-668, 685-686, 703-704):
```python
z0_voxel = voxel_to_point(x0, z)
z0_range = range_to_point(r_x0, range_pxpy)
z0_point = point_transforms[0](z.F)  # ← Remove this
z0.F = z0_voxel.F + z0_range + z0_point  # ← Becomes 2-way
```

**Modified 2-way fusion**:
```python
z0_voxel = voxel_to_point(x0, z)
z0_range = range_to_point(r_x0, range_pxpy)
z0.F = z0_voxel.F + z0_range  # Simple 2-way addition
```

Apply this change to all 4 fusion stages.

#### 3. Update Multi-Scale Channels

Since point MLPs change feature dimensions, removing them affects the channel counts:

**Original**:
- z1: 1024 channels (256×4 bottleneck expansion)
- z2: 512 channels (128×4)
- z3: 384 channels (96×4)

**After removing point branch**: Channels remain the same (determined by voxel branch), but you might want to adjust the classifier input:

```python
# Line 569: Update classifier
self.classifier = nn.Sequential(
    nn.Linear((cs[4] + cs[6] + cs[8]) * self.block.expansion, self.num_class)
)
# Channels: (256 + 128 + 96) × 4 = 1920 (with Bottleneck)
```

This stays the same since the point branch doesn't change these dimensions - it only adds a residual.

#### 4. (Optional) Add Gated Fusion

If you want to implement the **Gated Fusion Module** from the paper (which the current code doesn't have):

```python
class GatedFusionModule(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # Gating networks for each view
        self.gate_voxel = nn.Sequential(
            nn.Linear(channels, channels),
            nn.Sigmoid()
        )
        self.gate_range = nn.Sequential(
            nn.Linear(channels, channels),
            nn.Sigmoid()
        )

    def forward(self, feat_voxel, feat_range):
        # Compute gates
        g_v = self.gate_voxel(feat_voxel)
        g_r = self.gate_range(feat_range)

        # Normalize gates (softmax across views)
        gates = torch.stack([g_v, g_r], dim=0)  # [2, N, C]
        gates = F.softmax(gates, dim=0)

        # Weighted fusion
        fused = gates[0] * feat_voxel + gates[1] * feat_range
        return fused

# Usage in fusion stages:
self.gfm_0 = GatedFusionModule(32)
self.gfm_1 = GatedFusionModule(1024)
self.gfm_2 = GatedFusionModule(512)
self.gfm_3 = GatedFusionModule(384)

# In forward():
z0_voxel = voxel_to_point(x0, z)
z0_range = range_to_point(r_x0, range_pxpy)
z0.F = self.gfm_0(z0_voxel.F, z0_range)  # Gated fusion
```

### Complete Modified Forward Pass (2-View)

```python
def forward(self, batch_dict):
    x = batch_dict['lidar']
    range_image = batch_dict['range_image']
    range_pxpy = batch_dict['range_pxpy']
    batch_size = range_image.size(0)
    h, w = range_image.size(2), range_image.size(3)

    x.F = x.F[:, :self.in_feature_dim]
    z = PointTensor(x.F, x.C.float())
    x0 = initial_voxelize(z, self.pres, self.vres)

    # Parallel stems
    r_x0 = self.range_branch.stem(range_image)
    x0 = self.stem(x0)

    # FUSION 0: Remove point branch
    z0_v = voxel_to_point(x0, z)
    z0_r = range_to_point(r_x0, range_pxpy)
    z0.F = z0_v.F + z0_r  # 2-way fusion

    # Voxel branch encoder
    x1 = point_to_voxel(x0, z0)
    x1 = self.stage1(x1)
    x2 = self.stage2(x1)
    x3 = self.stage3(x2)
    x4 = self.stage4(x3)

    # Range branch encoder
    r_x1 = point_to_range(z0.F, range_pxpy, batch_size, h, w)
    r_x1, r_s1 = self.range_branch.stage1(r_x1)
    r_x2, r_s2 = self.range_branch.stage2(r_x1)
    r_x3, r_s3 = self.range_branch.stage3(r_x2)
    r_x4, r_s4 = self.range_branch.stage4(r_x3)
    r_x4 = self.range_branch.mid_stage(r_x4)

    # FUSION 1: Bottleneck
    z1_v = voxel_to_point(x4, z0)
    z1_r = range_to_point(r_x4, range_pxpy)
    z1.F = z1_v.F + z1_r  # 2-way fusion

    # Voxel decoder
    y1 = point_to_voxel(x4, z1)
    y1.F = self.dropout(y1.F)
    y1 = self.up1[0](y1)
    y1 = torchsparse.cat([y1, x3])
    y1 = self.up1[1](y1)
    y2 = self.up2[0](y1)
    y2 = torchsparse.cat([y2, x2])
    y2 = self.up2[1](y2)

    # Range decoder
    r_y1 = point_to_range(z1.F, range_pxpy, batch_size, r_x4.size(2), r_x4.size(3))
    r_y1 = self.range_branch.up1(r_y1, r_s4)
    r_y2 = self.range_branch.up2(r_y1, r_s3)

    # FUSION 2: Mid-upsample
    z2_v = voxel_to_point(y2, z1)
    z2_r = range_to_point(r_y2, range_pxpy)
    z2.F = z2_v.F + z2_r  # 2-way fusion

    # Continue decoder
    y3 = point_to_voxel(y2, z2)
    y3.F = self.dropout(y3.F)
    y3 = self.up3[0](y3)
    y3 = torchsparse.cat([y3, x1])
    y3 = self.up3[1](y3)
    y4 = self.up4[0](y3)
    y4 = torchsparse.cat([y4, x0])
    y4 = self.up4[1](y4)

    r_y3 = point_to_range(z2.F, range_pxpy, batch_size, r_y2.size(2), r_y2.size(3))
    r_y3 = self.range_branch.up3(r_y3, r_s2)
    r_y4 = self.range_branch.up4(r_y3, r_s1)

    # FUSION 3: Final
    z3_v = voxel_to_point(y4, z2)
    z3_r = range_to_point(r_y4, range_pxpy)
    z3.F = z3_v.F + z3_r  # 2-way fusion

    # Multi-scale classification (unchanged)
    out = self.classifier(torch.cat([z1.F, z2.F, z3.F], dim=1))

    return out
```

### Expected Impact of Removing Point Branch

**Pros**:
- ✅ Simpler architecture
- ✅ Fewer parameters (~2-5% reduction)
- ✅ Slightly faster inference (~2-3%)
- ✅ Cleaner conceptual model (pure 2-view fusion)

**Cons**:
- ❌ Loss of learnable residual pathway for point features
- ❌ Slightly reduced representational capacity
- ❌ Expected accuracy drop: **0.5-1.5 mIoU** (based on typical ablation studies)

The point branch acts as a learned "correction term" that helps refine features between fusion stages. Removing it means the network relies entirely on voxel ↔ range information transfer.

### Alternative: Keep Points as Features, Remove Independent Processing

Another interpretation of "2-view" is to:
- ✅ Keep the point MLPs but treat them as "feature adaptation layers" rather than a separate branch
- ✅ The point features would then be seen as **transformed voxel features** rather than an independent view

This is actually closer to what the implementation does - the point branch is very lightweight and doesn't have independent spatial processing.

---

## Summary

**Key Architectural Insights**:

1. **Points are the universal hub**: All cross-view communication goes through point-space
   - V2P, R2P: Interpolation from structured views to unstructured points
   - P2V, P2R: Aggregation from points back to structured views

2. **Multi-stage fusion is critical**: Fusion at 4 scales allows:
   - Early fusion: low-level feature enrichment
   - Mid fusion: semantic feature exchange
   - Late fusion: fine-grained detail refinement

3. **Simple addition works**: Despite the paper proposing gated fusion, simple addition of features achieves competitive results

4. **Multi-scale output**: Concatenating features from multiple fusion stages (z1, z2, z3) provides a rich multi-resolution representation

5. **Lightweight point branch**: The point MLPs are computationally cheap compared to the voxel/range branches, acting mainly as learned residuals

**For 2-view adaptation**:
- Remove `point_transforms` MLPs
- Change fusion from 3-way to 2-way addition: `z.F = z_voxel.F + z_range`
- Optionally add gated fusion for better feature mixing
- Keep all V2P, R2P, P2V, P2R operations (points remain the communication medium)
- Expected small accuracy drop but cleaner architecture

The beauty of this design is that **points don't need expensive spatial operations** (like neighbor search) - they simply serve as a coordinate system for transferring features between the complementary voxel and range representations.
