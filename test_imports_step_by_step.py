#!/usr/bin/env python
"""Step-by-step import test to isolate the segfault"""

import sys
import torch

print("Step 1: Import torch")
print(f"  PyTorch version: {torch.__version__}")
print(f"  CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  CUDA version: {torch.version.cuda}")

print("\nStep 2: Import timm")
import timm
print(f"  timm version: {timm.__version__}")

print("\nStep 3: Test timm.create_model (this might fail)")
try:
    # This is what SwinRangeBranch.__init__ does
    test_model = timm.create_model(
        'swin_tiny_patch4_window7_224',
        pretrained=False,  # Try without pretrained first
        features_only=True,
        out_indices=(0, 1, 2, 3),
    )
    print(f"  ✓ timm.create_model works (pretrained=False)")
    del test_model
except Exception as e:
    print(f"  ✗ timm.create_model failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\nStep 4: Test timm.create_model with pretrained=True")
try:
    test_model = timm.create_model(
        'swin_tiny_patch4_window7_224',
        pretrained=True,  # Now with pretrained
        features_only=True,
        out_indices=(0, 1, 2, 3),
    )
    print(f"  ✓ timm.create_model works (pretrained=True)")
    del test_model
except Exception as e:
    print(f"  ✗ timm.create_model with pretrained failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\nStep 5: Import torch.nn")
import torch.nn as nn
print("  ✓ torch.nn imported")

print("\nStep 6: Import the actual module")
try:
    from pcseg.model.segmentor.fusion.rpvnet.swin_range_branch import SwinRangeBranch, SwinRangeBranchWrapper
    print("  ✓ SwinRangeBranch imported successfully")
except Exception as e:
    print(f"  ✗ Import failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n✓ ALL STEPS PASSED!")
