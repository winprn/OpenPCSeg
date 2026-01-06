"""
Test script for Swin Transformer Range Branch

This script validates:
1. Model initialization
2. Forward pass with dummy input
3. Feature caching mechanism
4. Output shapes and compatibility
"""

import torch
import torch.nn as nn
from easydict import EasyDict


def test_swin_range_branch():
    """Test SwinRangeBranch basic functionality"""
    print("=" * 80)
    print("Testing Swin Range Branch")
    print("=" * 80)

    # Import the module
    try:
        from pcseg.model.segmentor.fusion.rpvnet.swin_range_branch import (
            SwinRangeBranch,
            SwinRangeBranchWrapper
        )
        print("✓ Successfully imported SwinRangeBranch modules")
    except ImportError as e:
        print(f"✗ Failed to import: {e}")
        return False

    # Create dummy config
    model_cfgs = EasyDict({
        'cr': 1.75,
        'DROPOUT_P': 0.3,
        'SWIN_VARIANT': 'swin_tiny_patch4_window7_224',
        'SWIN_PRETRAINED': False,  # Use False for faster testing
        'SWIN_WINDOW_SIZE': [7, 7],
    })

    # Test 1: Model initialization
    print("\nTest 1: Model Initialization")
    try:
        model = SwinRangeBranch(model_cfgs)
        print(f"✓ SwinRangeBranch initialized successfully")
        print(f"  - Target channels: {model.target_channels}")
        print(f"  - Swin channels: {model.swin_channels}")
        print(f"  - Output features: {model.num_point_features}")
    except Exception as e:
        print(f"✗ Failed to initialize: {e}")
        return False

    # Test 2: Forward pass
    print("\nTest 2: Forward Pass")
    try:
        # Create dummy range image: [B, 5, H, W]
        batch_size = 2
        height = 64
        width = 512  # Smaller for testing (real is 2048)
        dummy_input = torch.randn(batch_size, 5, height, width)
        print(f"  Input shape: {dummy_input.shape}")

        with torch.no_grad():
            output = model(dummy_input)

        # Check output keys
        expected_keys = ['stem', 'encoder_features', 'projected_features',
                        'skips', 'bottleneck', 'decoder_outputs']
        for key in expected_keys:
            if key not in output:
                print(f"✗ Missing output key: {key}")
                return False

        print("✓ Forward pass successful")
        print(f"  - Stem shape: {output['stem'].shape}")
        print(f"  - Bottleneck shape: {output['bottleneck'].shape}")
        print(f"  - Num encoder features: {len(output['encoder_features'])}")
        print(f"  - Num decoder outputs: {len(output['decoder_outputs'])}")

    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test 3: Wrapper compatibility
    print("\nTest 3: Wrapper API Compatibility")
    try:
        wrapper = SwinRangeBranchWrapper(model_cfgs, input_channels=5)
        print("✓ SwinRangeBranchWrapper initialized")

        with torch.no_grad():
            # Simulate RPVNet's sequential API calls
            r_x0 = wrapper.stem(dummy_input)
            print(f"  - stem() output shape: {r_x0.shape}")

            r_x1, r_s1 = wrapper.stage1(None)
            print(f"  - stage1() output shapes: {r_x1.shape}, {r_s1.shape}")

            r_x2, r_s2 = wrapper.stage2(None)
            r_x3, r_s3 = wrapper.stage3(None)
            r_x4, r_s4 = wrapper.stage4(None)

            r_mid = wrapper.mid_stage(None)
            print(f"  - mid_stage() output shape: {r_mid.shape}")

            r_y1 = wrapper.up1(None, None)
            r_y2 = wrapper.up2(None, None)
            r_y3 = wrapper.up3(None, None)
            r_y4 = wrapper.up4(None, None)
            print(f"  - up4() output shape: {r_y4.shape}")

        print("✓ All wrapper methods work correctly")

    except Exception as e:
        print(f"✗ Wrapper test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test 4: Channel dimensions match expectations
    print("\nTest 4: Channel Dimension Validation")
    try:
        expected_stem_channels = int(1.75 * 32)  # cr * base_channel
        expected_bottleneck_channels = int(1.75 * 256)

        if output['stem'].shape[1] != expected_stem_channels:
            print(f"✗ Stem channels mismatch: {output['stem'].shape[1]} vs {expected_stem_channels}")
            return False

        if output['bottleneck'].shape[1] != expected_bottleneck_channels:
            print(f"✗ Bottleneck channels mismatch: {output['bottleneck'].shape[1]} vs {expected_bottleneck_channels}")
            return False

        print("✓ Channel dimensions correct")
        print(f"  - Stem: {output['stem'].shape[1]} channels")
        print(f"  - Bottleneck: {output['bottleneck'].shape[1]} channels")

    except Exception as e:
        print(f"✗ Channel validation failed: {e}")
        return False

    print("\n" + "=" * 80)
    print("All tests passed! ✓")
    print("=" * 80)
    return True


def test_rpvnet_integration():
    """Test integration with RPVNet"""
    print("\n" + "=" * 80)
    print("Testing RPVNet Integration")
    print("=" * 80)

    try:
        from pcseg.model.segmentor.fusion.rpvnet.rpvnet import RPVNet
        from easydict import EasyDict

        # Create config for RPVNet with Swin
        model_cfgs = EasyDict({
            'IN_FEATURE_DIM': 5,
            'NUM_LAYER': [2, 2, 2, 2, 2, 2, 2, 2],
            'BLOCK': 'Bottleneck',
            'PLANES': [32, 32, 64, 128, 256, 256, 128, 96, 96],
            'cr': 1.0,  # Use 1.0 for faster testing
            'pres': 0.05,
            'vres': 0.05,
            'IF_DIST': False,
            'MULTI_SCALE': 'concat',
            'DROPOUT_P': 0.3,
            'LABEL_SMOOTHING': 0.0,
            'RANGE_BRANCH': 'SwinRangeBranch',
            'SWIN_VARIANT': 'swin_tiny_patch4_window7_224',
            'SWIN_PRETRAINED': False,
            'SWIN_WINDOW_SIZE': [7, 7],
        })

        num_class = 20  # SemanticKITTI has 20 classes (excluding ignore)

        print("\nTest: RPVNet with Swin Range Branch")
        model = RPVNet(model_cfgs, num_class)
        print(f"✓ RPVNet initialized with Swin range branch")
        print(f"  - Range branch type: {type(model.range_branch).__name__}")

        # Check if it's using Swin
        if hasattr(model.range_branch, 'swin_branch'):
            print(f"  - Confirmed: Using SwinRangeBranchWrapper")
        else:
            print(f"  - Using: {type(model.range_branch).__name__}")

        print("\n✓ RPVNet integration successful")

    except Exception as e:
        print(f"✗ RPVNet integration failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("=" * 80)
    return True


if __name__ == '__main__':
    print("\n🚀 Starting Swin Transformer Range Branch Tests\n")

    # Run tests
    test1_passed = test_swin_range_branch()
    test2_passed = test_rpvnet_integration()

    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"Swin Range Branch Tests: {'✓ PASSED' if test1_passed else '✗ FAILED'}")
    print(f"RPVNet Integration Tests: {'✓ PASSED' if test2_passed else '✗ FAILED'}")
    print("=" * 80)

    if test1_passed and test2_passed:
        print("\n🎉 All tests passed! Ready for training.")
        print("\nNext steps:")
        print("1. Install timm: pip install timm>=0.9.0")
        print("2. Run training:")
        print("   python train.py --cfg_file tools/cfgs/fusion/semantic_kitti/rpvnet_swin_tiny_cr17_5.yaml")
    else:
        print("\n❌ Some tests failed. Please fix the issues before training.")
