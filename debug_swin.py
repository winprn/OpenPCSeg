#!/usr/bin/env python
"""Debug script to identify where the segmentation fault occurs"""

import sys
import torch

print("=" * 60)
print("SWIN TRANSFORMER DEBUG SCRIPT")
print("=" * 60)

# Step 1: Check imports
print("\n[1/7] Testing imports...")
try:
    import timm
    print(f"✓ timm version: {timm.__version__}")
except Exception as e:
    print(f"✗ timm import failed: {e}")
    sys.exit(1)

try:
    from pcseg.model.segmentor.fusion.rpvnet.swin_range_branch import SwinRangeBranch, SwinRangeBranchWrapper
    print("✓ SwinRangeBranch imports successful")
except Exception as e:
    print(f"✗ Import failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 2: Create mock config
print("\n[2/7] Creating configuration...")
class MockConfig:
    def get(self, key, default=None):
        config = {
            'cr': 1.75,
            'DROPOUT_P': 0.3,
            'SWIN_VARIANT': 'swin_tiny_patch4_window7_224',
            'SWIN_PRETRAINED': True,  # Try with pretrained
            'SWIN_WINDOW_SIZE': [7, 7]
        }
        return config.get(key, default)

cfg = MockConfig()
print("✓ Configuration created")

# Step 3: Test timm model creation directly
print("\n[3/7] Testing timm Swin model creation...")
try:
    backbone = timm.create_model(
        'swin_tiny_patch4_window7_224',
        pretrained=True,
        features_only=True,
        out_indices=(0, 1, 2, 3),
    )
    print(f"✓ timm Swin model created")
    print(f"  Feature channels: {backbone.feature_info.channels()}")
except Exception as e:
    print(f"✗ timm model creation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 4: Test input channel adaptation
print("\n[4/7] Testing input channel adaptation...")
try:
    patch_embed = backbone.patch_embed
    old_proj = patch_embed.proj
    print(f"  Original input channels: {old_proj.in_channels}")
    print(f"  Original output channels: {old_proj.out_channels}")

    # Create new projection with 5 channels
    import torch.nn as nn
    new_proj = nn.Conv2d(
        in_channels=5,
        out_channels=old_proj.out_channels,
        kernel_size=old_proj.kernel_size,
        stride=old_proj.stride,
        padding=old_proj.padding,
        bias=old_proj.bias is not None
    )
    print("✓ New projection layer created (5 channels)")
except Exception as e:
    print(f"✗ Input adaptation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 5: Test SwinRangeBranch initialization
print("\n[5/7] Testing SwinRangeBranch initialization...")
try:
    model = SwinRangeBranch(cfg)
    print("✓ SwinRangeBranch initialized")
    print(f"  Target channels: {model.target_channels}")
    print(f"  Swin channels: {model.swin_channels}")
except Exception as e:
    print(f"✗ SwinRangeBranch initialization failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 6: Test forward pass with small input
print("\n[6/7] Testing forward pass (CPU)...")
try:
    model.eval()
    x = torch.randn(1, 5, 64, 128)  # Small input
    with torch.no_grad():
        output = model(x)
    print("✓ Forward pass successful (CPU)")
    print(f"  Stem shape: {output['stem'].shape}")
    print(f"  Bottleneck shape: {output['bottleneck'].shape}")
    print(f"  Decoder outputs: {len(output['decoder_outputs'])} stages")
except Exception as e:
    print(f"✗ Forward pass failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 7: Test CUDA forward pass
print("\n[7/7] Testing forward pass (CUDA)...")
if torch.cuda.is_available():
    try:
        model_cuda = model.cuda()
        x_cuda = torch.randn(1, 5, 64, 128).cuda()
        with torch.no_grad():
            output_cuda = model_cuda(x_cuda)
        print("✓ Forward pass successful (CUDA)")
        print(f"  Output on GPU: {output_cuda['stem'].device}")
    except Exception as e:
        print(f"✗ CUDA forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
else:
    print("⊘ CUDA not available, skipping")

print("\n" + "=" * 60)
print("ALL TESTS PASSED! ✓")
print("=" * 60)
print("\nThe model initialization and forward pass work correctly.")
print("The segfault might be occurring during:")
print("  1. Model building in RPVNet")
print("  2. Data loading")
print("  3. Distributed training initialization")
print("  4. Interaction with other branches (voxel/point)")
