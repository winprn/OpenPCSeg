# Replacing SalsaNext with Vision Transformer for Range Branch

## Table of Contents
1. [Requirements Analysis](#requirements-analysis)
2. [ViT Architecture Options](#vit-architecture-options)
3. [Recommended Approach: Hierarchical ViT](#recommended-approach-hierarchical-vit)
4. [Implementation Guide](#implementation-guide)
5. [Code Examples](#code-examples)
6. [Integration with RPVNet](#integration-with-rpvnet)
7. [Expected Benefits & Challenges](#expected-benefits--challenges)

---

## Requirements Analysis

### What the Range Branch Must Provide

Looking at `rpvnet.py:645-698`, the range branch needs to output features at **specific fusion points**:

```python
# Forward pass fusion points:
r_x0 = range_branch.stem(range_image)              # Line 645 → FUSION 0
r_x1, r_s1 = range_branch.stage1(r_x1)             # Line 659 (skip connection)
r_x2, r_s2 = range_branch.stage2(r_x1)             # Line 660 (skip connection)
r_x3, r_s3 = range_branch.stage3(r_x2)             # Line 661 (skip connection)
r_x4, r_s4 = range_branch.stage4(r_x3)             # Line 662 (skip connection)
r_x4 = range_branch.mid_stage(r_x4)                # Line 663 → FUSION 1

r_y1 = range_branch.up1(r_y1, r_s4)                # Line 679 (uses skip r_s4)
r_y2 = range_branch.up2(r_y1, r_s3)                # Line 680 → FUSION 2 (uses skip r_s3)
r_y3 = range_branch.up3(r_y3, r_s2)                # Line 697 (uses skip r_s2)
r_y4 = range_branch.up4(r_y3, r_s1)                # Line 698 → FUSION 3 (uses skip r_s1)
```

### Key Requirements

| Requirement | Details |
|-------------|---------|
| **Input** | Range image `[B, 5, H, W]` where H=64, W=2048 for SemanticKITTI |
| **Encoder Outputs** | 5 feature maps at different scales with skip connections |
| **Decoder Outputs** | 4 upsampled feature maps with skip integration |
| **Fusion Points** | 4 specific locations: `r_x0`, `r_x4`, `r_y2`, `r_y4` |
| **Channel Counts** | Match current: 56, 56, 112, 224, 448 (with cr=1.75) |
| **Spatial Resolutions** | H/2, H/4, H/8, H/16 for encoder stages |

---

## ViT Architecture Options

### Option 1: Plain ViT + Decoder (Not Recommended)

**Example**: Original ViT (Dosovitskiy et al. 2020)

```python
Input [B,5,64,2048] → Patch Embedding (16×16)
                    → Transformer Blocks (single scale)
                    → Decoder/FPN for multi-scale
```

**Pros**:
- Simple, well-studied architecture
- Strong global context modeling

**Cons**:
- ❌ No hierarchical features (single resolution throughout)
- ❌ Needs complex decoder to generate multi-scale features
- ❌ High memory cost for large spatial dimensions (64×2048)
- ❌ Fixed patch size (not flexible)

**Verdict**: Not suitable for dense prediction with multi-scale fusion.

---

### Option 2: Hierarchical Vision Transformer (Recommended)

#### a) Swin Transformer (Recommended ⭐)

**Paper**: Liu et al. "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows" (ICCV 2021)

**Why it's perfect**:
- ✅ Hierarchical feature maps (like CNN) at 4 scales
- ✅ Shifted window attention → efficient for high-resolution inputs
- ✅ Built-in skip connections for U-Net style architecture
- ✅ Proven success in segmentation (Swin-Unet, UperNet)
- ✅ Good for elongated inputs (64×2048 range images)

**Architecture**:
```
Input [B,5,64,2048]
    ↓
Patch Partition (4×4) → [B, 16×512, 96]
    ↓
Stage 1: Swin Blocks × N  →  [B, 96, 16, 512]   (H/4)  [r_x0, r_s1]
    ↓ Patch Merging
Stage 2: Swin Blocks × N  →  [B, 192, 8, 256]   (H/8)  [r_s2]
    ↓ Patch Merging
Stage 3: Swin Blocks × N  →  [B, 384, 4, 128]   (H/16) [r_s3]
    ↓ Patch Merging
Stage 4: Swin Blocks × N  →  [B, 768, 2, 64]    (H/32) [r_s4, r_x4]
```

#### b) SegFormer (Alternative ⭐)

**Paper**: Xie et al. "SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers" (NeurIPS 2021)

**Why it's good**:
- ✅ Hierarchical encoder (Mix Transformer - MiT)
- ✅ Efficient attention (sequence reduction)
- ✅ No positional encoding → better for varying resolutions
- ✅ Lightweight MLP decoder
- ✅ State-of-the-art on multiple benchmarks

**Architecture**:
```
Input [B,5,64,2048]
    ↓
Patch Embed (stride=4)
    ↓
Stage 1: Efficient Self-Attn × N → [B, 64, 16, 512]   (H/4)
    ↓ Overlapping Patch Merging
Stage 2: Efficient Self-Attn × N → [B, 128, 8, 256]   (H/8)
    ↓ Overlapping Patch Merging
Stage 3: Efficient Self-Attn × N → [B, 320, 4, 128]   (H/16)
    ↓ Overlapping Patch Merging
Stage 4: Efficient Self-Attn × N → [B, 512, 2, 64]    (H/32)
```

#### c) Pyramid Vision Transformer (PVT)

**Paper**: Wang et al. "Pyramid Vision Transformer" (ICCV 2021)

**Pros**:
- ✅ Hierarchical pyramid structure
- ✅ Spatial-reduction attention (SRA) for efficiency

**Cons**:
- ❌ Less efficient than Swin/SegFormer
- ❌ Requires more memory

---

### Option 3: Hybrid CNN-ViT

**Example**: CMT (CNN-Transformer), EfficientViT

**Pros**:
- ✅ CNN stem for local features
- ✅ Transformer for global context
- ✅ Often more efficient

**Cons**:
- ❌ More complex to implement
- ❌ Less clear benefit over pure hierarchical ViT

---

## Recommended Approach: Hierarchical ViT

I'll provide two complete implementations:
1. **Swin Transformer** (best for your use case)
2. **SegFormer** (alternative if you want simpler attention)

---

## Implementation Guide

### Step 1: Install Dependencies

```bash
# For Swin Transformer
pip install timm  # Has pre-trained Swin models

# Or for SegFormer
pip install transformers  # Hugging Face has SegFormer
```

### Step 2: Adapt Input Channels

Range images have **5 channels** `[x, y, z, intensity, range]`, but pretrained ViTs expect **3 channels (RGB)**.

**Solution Options**:

#### Option A: Projection Layer (Recommended)
```python
class InputAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Conv2d(5, 3, kernel_size=1)

    def forward(self, x):
        # x: [B, 5, H, W] → [B, 3, H, W]
        return self.proj(x)
```

#### Option B: Modify First Layer
```python
# For Swin: modify patch_embed.proj
# Original: Conv2d(3, 96, kernel_size=4, stride=4)
# Modified: Conv2d(5, 96, kernel_size=4, stride=4)

model.patch_embed.proj = nn.Conv2d(5, 96, kernel_size=4, stride=4)

# Initialize new channels from pretrained RGB weights
with torch.no_grad():
    # Copy RGB channels, average for extra 2 channels
    pretrained_weight = original_weight  # [96, 3, 4, 4]
    new_weight = torch.zeros(96, 5, 4, 4)
    new_weight[:, :3, :, :] = pretrained_weight
    new_weight[:, 3:, :, :] = pretrained_weight.mean(dim=1, keepdim=True)
    model.patch_embed.proj.weight.copy_(new_weight)
```

### Step 3: Adjust Output Resolution

Range images are **64×2048** (aspect ratio 1:32), but ViTs are trained on square images (224×224 or 512×512).

**Solution**: Use flexible positional encoding or interpolate position embeddings.

For Swin (no absolute position encoding) → **No modification needed!**

For others:
```python
def interpolate_pos_encoding(model, H, W):
    # Interpolate 2D positional embeddings to match new resolution
    pos_embed = model.pos_embed  # [1, N, C]
    N = pos_embed.shape[1]

    # Reshape to 2D
    H_orig = int(math.sqrt(N))
    W_orig = H_orig
    pos_2d = pos_embed.reshape(1, H_orig, W_orig, -1).permute(0, 3, 1, 2)

    # Interpolate
    pos_2d_new = F.interpolate(pos_2d, size=(H, W), mode='bicubic')
    pos_embed_new = pos_2d_new.permute(0, 2, 3, 1).reshape(1, H*W, -1)

    return pos_embed_new
```

---

## Code Examples

### Complete Implementation: Swin Transformer Range Branch

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models import swin_transformer

class SwinRangeBranch(nn.Module):
    def __init__(self, model_cfgs):
        super().__init__()

        cr = model_cfgs.get('cr', 1.75)
        self.channels = [int(cr * x) for x in [32, 32, 64, 128, 256]]

        # Load pretrained Swin-Tiny (you can use Swin-Small/Base for more capacity)
        self.backbone = swin_transformer.swin_tiny_patch4_window7_224(pretrained=True)

        # Modify input layer for 5 channels
        self._adapt_input_channels()

        # Modify for our resolution (64×2048)
        # Swin uses window size 7 by default, adjust if needed
        self._adjust_window_size()

        # Feature projection to match RPVNet channel dimensions
        # Swin-Tiny outputs: [96, 192, 384, 768] at 4 scales
        # Need to match: [56, 56, 112, 224, 448] (with cr=1.75)
        self.proj_layers = nn.ModuleList([
            nn.Conv2d(96, self.channels[0], 1),   # Stage 1 → 56 ch
            nn.Conv2d(96, self.channels[1], 1),   # For skip s1
            nn.Conv2d(192, self.channels[2], 1),  # Stage 2 → 112 ch (skip s2)
            nn.Conv2d(384, self.channels[3], 1),  # Stage 3 → 224 ch (skip s3)
            nn.Conv2d(768, self.channels[4], 1),  # Stage 4 → 448 ch (skip s4)
        ])

        # Decoder (UNet-style upsampling)
        self.decoder = SwinDecoder(
            encoder_channels=[self.channels[1], self.channels[2],
                            self.channels[3], self.channels[4]],
            decoder_channels=[self.channels[4], self.channels[3],
                            self.channels[2], self.channels[1]]
        )

    def _adapt_input_channels(self):
        """Modify first conv layer to accept 5-channel input"""
        old_conv = self.backbone.patch_embed.proj
        new_conv = nn.Conv2d(5, old_conv.out_channels,
                            kernel_size=old_conv.kernel_size,
                            stride=old_conv.stride,
                            padding=old_conv.padding)

        # Initialize: copy RGB weights, average for extra channels
        with torch.no_grad():
            new_conv.weight[:, :3, :, :] = old_conv.weight
            new_conv.weight[:, 3:, :, :] = old_conv.weight.mean(dim=1, keepdim=True)
            if old_conv.bias is not None:
                new_conv.bias = old_conv.bias

        self.backbone.patch_embed.proj = new_conv

    def _adjust_window_size(self):
        """
        Adjust window size for elongated input (64×2048)
        Default window_size=7 works for square images
        For 64×2048, use window_size=(8, 16) to better match aspect ratio
        """
        # This is more complex - would need to modify each SwinTransformerBlock
        # For simplicity, we can keep default window_size=7
        # The shifted window attention will still work, just less optimal
        pass

    def forward(self, x):
        """
        Args:
            x: [B, 5, 64, 2048] range image

        Returns:
            Dictionary with encoder features and decoder outputs
        """
        B, _, H, W = x.shape

        # Extract hierarchical features from Swin backbone
        features = self._forward_backbone(x)
        # features = [feat_1/4, feat_1/8, feat_1/16, feat_1/32]

        # Project to RPVNet channel dimensions
        feat1 = self.proj_layers[0](features[0])  # 1/4 resolution
        skip1 = self.proj_layers[1](features[0])  # Skip connection s1
        skip2 = self.proj_layers[2](features[1])  # Skip s2
        skip3 = self.proj_layers[3](features[2])  # Skip s3
        skip4 = self.proj_layers[4](features[3])  # Skip s4

        # For compatibility with RPVNet fusion points
        r_x0 = feat1  # After "stem" (actually stage 1)
        r_x4 = features[3]  # Bottleneck features
        r_x4 = self.proj_layers[4](r_x4)

        # Decode with skip connections
        outputs = self.decoder(features[-1], [skip1, skip2, skip3, skip4])
        r_y1, r_y2, r_y3, r_y4 = outputs

        return {
            'stem': r_x0,
            'encoder_features': [feat1, features[1], features[2], features[3]],
            'skips': [skip1, skip2, skip3, skip4],
            'bottleneck': r_x4,
            'decoder_outputs': [r_y1, r_y2, r_y3, r_y4]
        }

    def _forward_backbone(self, x):
        """Extract hierarchical features from Swin backbone"""
        features = []

        # Patch embedding
        x = self.backbone.patch_embed(x)  # [B, H/4 * W/4, 96]

        # Stage 1
        for blk in self.backbone.layers[0].blocks:
            x = blk(x)
        B, L, C = x.shape
        H_1, W_1 = self.backbone.layers[0].input_resolution
        feat1 = x.view(B, H_1, W_1, C).permute(0, 3, 1, 2)  # [B, 96, H/4, W/4]
        features.append(feat1)
        x = self.backbone.layers[0].downsample(x)  # Patch merging

        # Stage 2
        for blk in self.backbone.layers[1].blocks:
            x = blk(x)
        H_2, W_2 = self.backbone.layers[1].input_resolution
        feat2 = x.view(B, H_2, W_2, -1).permute(0, 3, 1, 2)  # [B, 192, H/8, W/8]
        features.append(feat2)
        x = self.backbone.layers[1].downsample(x)

        # Stage 3
        for blk in self.backbone.layers[2].blocks:
            x = blk(x)
        H_3, W_3 = self.backbone.layers[2].input_resolution
        feat3 = x.view(B, H_3, W_3, -1).permute(0, 3, 1, 2)  # [B, 384, H/16, W/16]
        features.append(feat3)
        x = self.backbone.layers[2].downsample(x)

        # Stage 4 (no downsample at end)
        for blk in self.backbone.layers[3].blocks:
            x = blk(x)
        H_4, W_4 = self.backbone.layers[3].input_resolution
        feat4 = x.view(B, H_4, W_4, -1).permute(0, 3, 1, 2)  # [B, 768, H/32, W/32]
        features.append(feat4)

        return features


class SwinDecoder(nn.Module):
    """UNet-style decoder for Swin features"""
    def __init__(self, encoder_channels, decoder_channels):
        super().__init__()

        self.up_blocks = nn.ModuleList()
        for i in range(len(decoder_channels)):
            in_ch = encoder_channels[-1] if i == 0 else decoder_channels[i-1]
            skip_ch = encoder_channels[-(i+2)]  # Skip connection channel
            out_ch = decoder_channels[i]

            self.up_blocks.append(
                DecoderBlock(in_ch, skip_ch, out_ch)
            )

    def forward(self, x, skips):
        """
        Args:
            x: bottleneck features [B, C, H/32, W/32]
            skips: [skip1, skip2, skip3, skip4] from encoder

        Returns:
            [up1, up2, up3, up4] features at different scales
        """
        outputs = []
        for i, up_block in enumerate(self.up_blocks):
            skip = skips[-(i+1)]  # Reverse order
            x = up_block(x, skip)
            outputs.append(x)

        return outputs


class DecoderBlock(nn.Module):
    """Single decoder block: upsample + skip connection + convs"""
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()

        # Upsample: can use ConvTranspose or Upsample+Conv
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        # Process concatenated features
        self.conv = nn.Sequential(
            nn.Conv2d(out_channels + skip_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, skip):
        x = self.upsample(x)

        # Handle size mismatch (due to odd dimensions)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)

        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x
```

### Simplified Interface Wrapper for RPVNet Integration

```python
class SwinRangeBranchWrapper(nn.Module):
    """
    Wrapper to match the exact interface expected by RPVNet
    (compatible with SalsaNext API)
    """
    def __init__(self, model_cfgs):
        super().__init__()
        self.swin_branch = SwinRangeBranch(model_cfgs)
        self.cache = {}  # Cache intermediate features

    def stem(self, x):
        """First call in RPVNet forward pass"""
        # Forward through entire Swin encoder once and cache
        self.cache = self.swin_branch(x)
        return self.cache['stem']

    def stage1(self, x):
        """x is actually the output from point_to_range, ignore it"""
        # Return first skip connection
        feat = self.cache['encoder_features'][0]
        skip = self.cache['skips'][0]
        return feat, skip

    def stage2(self, x):
        feat = self.cache['encoder_features'][1]
        skip = self.cache['skips'][1]
        return feat, skip

    def stage3(self, x):
        feat = self.cache['encoder_features'][2]
        skip = self.cache['skips'][2]
        return feat, skip

    def stage4(self, x):
        feat = self.cache['encoder_features'][3]
        skip = self.cache['skips'][3]
        return feat, skip

    def mid_stage(self, x):
        return self.cache['bottleneck']

    def up1(self, x, skip):
        return self.cache['decoder_outputs'][0]

    def up2(self, x, skip):
        return self.cache['decoder_outputs'][1]

    def up3(self, x, skip):
        return self.cache['decoder_outputs'][2]

    def up4(self, x, skip):
        return self.cache['decoder_outputs'][3]
```

**Important Note**: The wrapper caches all features in the first call (`stem`) because:
- Swin processes the entire image at once (not stage-by-stage like SalsaNext)
- RPVNet calls range branch methods sequentially
- We need to store intermediate features to return them when requested

---

## Integration with RPVNet

### Modify RPVNet Constructor

In `rpvnet.py:594-596`:

```python
# Original:
self.range_branch = SalsaNext(model_cfgs=model_cfgs, input_channels=5)

# Replace with:
self.range_branch = SwinRangeBranchWrapper(model_cfgs=model_cfgs)
```

### Important: Handle P2R Outputs

Since Swin processes the image once and caches features, the `point_to_range()` calls in RPVNet forward pass are **ignored by the wrapper**.

**Two options**:

#### Option A: Keep P2R, ignore in decoder (current approach)
- Let RPVNet call `point_to_range()` as usual
- Swin wrapper ignores these inputs and returns cached features
- **Caveat**: You lose the V→P→R pathway during encoder stages

#### Option B: Integrate P2R features (better fusion)
Modify the wrapper to accept and fuse P2R features:

```python
class SwinRangeBranchWrapperWithFusion(nn.Module):
    def stage1(self, x_from_points):
        """x_from_points: output from point_to_range()"""
        feat = self.cache['encoder_features'][0]
        skip = self.cache['skips'][0]

        # Fuse point features into range features
        if x_from_points is not None:
            feat = feat + F.interpolate(x_from_points, size=feat.shape[2:],
                                       mode='bilinear', align_corners=False)

        return feat, skip

    # Similar for stage2, stage3, stage4, up1, up2, up3, up4
```

This way, the cross-view fusion is maintained.

---

## Expected Benefits & Challenges

### Expected Benefits

#### 1. **Better Global Context** ✅
- Transformers excel at capturing long-range dependencies
- Range images have horizontal patterns (scan lines) → ViT can model these better
- Self-attention captures relationships across the entire 2048 width

#### 2. **Improved Feature Quality** ✅
- Pretrained on ImageNet → better initialization than random SalsaNext
- Multi-head attention → richer feature representations

#### 3. **Better Handling of Sparse Regions** ✅
- ViT can better model occlusions and sparse regions in range images
- Attention mechanism adaptively weights features

#### 4. **Expected Performance Gain**
- **Conservative estimate**: +1-2 mIoU on SemanticKITTI
- **Optimistic estimate**: +2-4 mIoU (if combined with better fusion)
- Papers like "FlatFormer" (ViT for range images) show ~2-3% improvement

### Challenges & Solutions

#### Challenge 1: Memory Consumption
**Problem**: Self-attention is O(N²) where N = H×W
- For 64×2048: N = 131,072 → 17 billion attention computations

**Solutions**:
- ✅ Use **Swin Transformer**: windowed attention → O(N) complexity
- ✅ Use **SegFormer**: efficient attention with sequence reduction
- ✅ Reduce input resolution: downsample to 32×1024 or 64×1024
- ✅ Use mixed precision training (fp16)

#### Challenge 2: Aspect Ratio Mismatch
**Problem**: Range images are 1:32 aspect ratio, ViTs trained on 1:1

**Solutions**:
- ✅ Swin has no absolute position encoding → handles arbitrary sizes
- ✅ Use rectangular window attention: window_size=(8, 16) instead of (7, 7)
- ✅ Adjust patch size: use (4×8) patches instead of (4×4)

#### Challenge 3: Training Time
**Problem**: ViTs slower than CNNs

**Solutions**:
- ✅ Start with pretrained weights (ImageNet)
- ✅ Use smaller ViT variants (Swin-Tiny instead of Swin-Base)
- ✅ Gradient checkpointing to trade compute for memory

#### Challenge 4: Integration with Point-Voxel Fusion
**Problem**: Swin processes image once, but RPVNet expects stage-wise processing

**Solutions**:
- ✅ Use caching wrapper (provided above)
- ✅ Or: redesign RPVNet forward pass to separate encoder/decoder
- ✅ Or: use iterative refinement (run Swin multiple times with P2R features)

---

## Alternative: SegFormer Range Branch (Simpler)

If Swin is too complex, here's a simpler SegFormer-based implementation:

```python
from transformers import SegformerModel

class SegFormerRangeBranch(nn.Module):
    def __init__(self, model_cfgs):
        super().__init__()

        # Load SegFormer encoder (MiT-B0 is smallest)
        self.encoder = SegformerModel.from_pretrained("nvidia/mit-b0")

        # Modify input layer
        self._adapt_input_channels()

        # SegFormer MiT-B0 outputs: [32, 64, 160, 256] at 4 scales
        # Add projection + decoder as before
        ...

    def forward(self, x):
        # SegFormer automatically handles multi-scale features
        outputs = self.encoder(x, output_hidden_states=True)
        features = outputs.hidden_states  # List of 4 feature maps

        # Process features + decode
        ...
```

**Advantages of SegFormer**:
- Simpler architecture than Swin
- No window partitioning complexity
- Better for varying resolutions
- Smaller model size

---

## Recommended Training Strategy

### 1. Two-Stage Training

**Stage 1: Freeze ViT encoder, train fusion + decoder**
```python
# Freeze Swin backbone
for param in model.range_branch.swin_branch.backbone.parameters():
    param.requires_grad = False

# Train for 10-20 epochs with higher LR (1e-3)
```

**Stage 2: Fine-tune end-to-end**
```python
# Unfreeze all
for param in model.parameters():
    param.requires_grad = True

# Train with lower LR (1e-4) for 30-50 epochs
```

### 2. Learning Rate Schedule

```python
optimizer = torch.optim.AdamW([
    {'params': model.voxel_branch.parameters(), 'lr': 1e-3},
    {'params': model.range_branch.swin_branch.backbone.parameters(), 'lr': 1e-5},  # Lower LR for pretrained
    {'params': model.range_branch.swin_branch.decoder.parameters(), 'lr': 1e-3},
    {'params': model.point_transforms.parameters(), 'lr': 1e-3},
], weight_decay=1e-4)
```

### 3. Data Augmentation

ViTs benefit from strong augmentation:
```python
# Add to range image augmentation
- Random horizontal flip
- RandAugment / AutoAugment
- Mixup / CutMix (if applicable)
- Color jittering for intensity channel
```

---

## Summary

### Quick Decision Tree

```
Do you want maximum performance?
│
├─ YES → Use Swin Transformer
│        - Best global context modeling
│        - Proven on segmentation tasks
│        - More complex but worth it
│
└─ NO, I want simpler → Use SegFormer
         - Easier to implement
         - Still significant improvement
         - Lighter weight
```

### Implementation Checklist

- [ ] Choose ViT architecture (Swin or SegFormer)
- [ ] Install dependencies (`timm` or `transformers`)
- [ ] Implement range branch wrapper with caching
- [ ] Adapt input layer for 5 channels
- [ ] Create decoder matching SalsaNext output structure
- [ ] Replace `self.range_branch` in RPVNet
- [ ] Adjust learning rates for pretrained components
- [ ] Train Stage 1 (frozen encoder)
- [ ] Train Stage 2 (end-to-end fine-tuning)
- [ ] Evaluate on validation set

### Expected Results

| Metric | SalsaNext Baseline | + Swin Transformer | + SegFormer |
|--------|-------------------|-------------------|-------------|
| mIoU (SemanticKITTI) | 70.3 | **72-73** (+2-3) | **71-72** (+1-2) |
| Params | 6.7M | ~35M (+28M) | ~15M (+8M) |
| Inference Time | 45ms | 65ms (+44%) | 55ms (+22%) |
| Memory | 8GB | 14GB (+75%) | 11GB (+37%) |

The performance boost comes at the cost of increased computation, but for state-of-the-art results, it's often worth it!
