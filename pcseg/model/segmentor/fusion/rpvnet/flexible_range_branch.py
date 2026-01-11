"""
Flexible Vision Transformer Range Branch

Provides multiple backbone options that work well with non-square range images (64x2048).
Includes ConvNeXt, MaxViT, and other architectures that don't have strict input size requirements.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import timm
except ImportError:
    raise ImportError("timm library is required. Install with: pip install timm>=0.9.0")


class FlexibleViTRangeBranch(nn.Module):
    """
    Flexible Vision Transformer range branch supporting multiple backbone architectures.

    Works better than Swin for non-square images like 64x2048 range images.

    Supported backbones:
    - ConvNeXt: CNN-based, very flexible with input sizes, excellent performance
    - MaxViT: Hybrid CNN-ViT, handles arbitrary sizes well
    - EfficientNet: Efficient CNN, scales well
    - ResNet: Classic CNN, very stable

    Args:
        model_cfgs: Model configuration dictionary
        input_channels: Number of input channels (default: 5 for range images)
    """

    def __init__(self, model_cfgs, input_channels=5):
        super().__init__()

        # Configuration
        backbone_type = model_cfgs.get('VIT_BACKBONE', 'convnext_tiny')
        pretrained = model_cfgs.get('VIT_PRETRAINED', True)
        output_scale = model_cfgs.get('VIT_OUTPUT_SCALE', 4)  # 1/4 resolution
        output_channels = model_cfgs.get('VIT_OUTPUT_CHANNELS', 256)

        self.backbone_type = backbone_type
        self.output_scale = output_scale

        # Input projection: 5 channels → 3 channels (for pretrained models)
        self.input_proj = nn.Conv2d(input_channels, 3, kernel_size=1, bias=True)

        # Initialize input projection
        with torch.no_grad():
            self.input_proj.weight.zero_()
            self.input_proj.weight[0, 0] = 1.0  # Red ← inverse_depth
            self.input_proj.weight[1, 1] = 1.0  # Green ← reflectivity
            self.input_proj.weight[2, 2:5].fill_(1.0/3.0)  # Blue ← mean(x,y,z)
            self.input_proj.bias.zero_()

        # Create backbone based on type
        print(f"Creating {backbone_type} backbone for range branch...")

        try:
            self.backbone = timm.create_model(
                backbone_type,
                pretrained=pretrained,
                features_only=True,
                out_indices=[0, 1, 2, 3]  # Multi-scale features
            )
        except Exception as e:
            raise RuntimeError(f"Failed to create backbone {backbone_type}: {e}")

        # Get channel dimensions
        backbone_channels = self.backbone.feature_info.channels()
        print(f"Backbone output channels: {backbone_channels}")

        # Select feature scale
        scale_to_idx = {4: 0, 8: 1, 16: 2, 32: 3}
        if output_scale not in scale_to_idx:
            raise ValueError(f"output_scale must be one of {list(scale_to_idx.keys())}, got {output_scale}")

        self.feature_idx = scale_to_idx[output_scale]
        backbone_ch = backbone_channels[self.feature_idx]

        # Output projection to target channels
        self.output_proj = nn.Conv2d(backbone_ch, output_channels, kernel_size=1, bias=True)
        self.output_channels = output_channels

    def forward(self, range_image):
        """
        Forward pass through backbone.

        Args:
            range_image: [B, 5, H, W] - Range image tensor

        Returns:
            features: [B, C, H_out, W_out] - Single feature tensor
        """
        # Project 5ch → 3ch
        x = self.input_proj(range_image)  # [B, 3, H, W]

        # Backbone forward pass
        features = self.backbone(x)  # List of multi-scale features

        # Select target scale feature
        feat = features[self.feature_idx]  # [B, C_backbone, H/scale, W/scale]

        # Output projection
        feat = self.output_proj(feat)  # [B, C_out, H_out, W_out]

        return feat

    def get_output_shape(self, input_h, input_w):
        """Calculate output feature map shape."""
        output_h = input_h // self.output_scale
        output_w = input_w // self.output_scale
        return (output_h, output_w, self.output_channels)


class ConvNextRangeBranch(nn.Module):
    """
    ConvNeXt-based range branch (RECOMMENDED for non-square images).

    ConvNeXt is a modernized CNN that achieves ViT-level performance
    without the strict input size requirements. Works excellently with
    64x2048 range images.

    Benefits:
    - No strict input size requirements
    - Efficient computation
    - Strong performance (competitive with Swin)
    - Stable training

    Available variants:
    - convnext_tiny (28M params) - Recommended
    - convnext_small (50M params)
    - convnext_base (89M params)
    """

    def __init__(self, model_cfgs, input_channels=5):
        super().__init__()

        # Configuration
        variant = model_cfgs.get('CONVNEXT_VARIANT', 'convnext_tiny')
        pretrained = model_cfgs.get('CONVNEXT_PRETRAINED', True)
        output_scale = model_cfgs.get('CONVNEXT_OUTPUT_SCALE', 4)
        output_channels = model_cfgs.get('CONVNEXT_OUTPUT_CHANNELS', 256)

        self.output_scale = output_scale

        # Input projection
        self.input_proj = nn.Conv2d(input_channels, 3, kernel_size=1, bias=True)
        with torch.no_grad():
            self.input_proj.weight.zero_()
            self.input_proj.weight[0, 0] = 1.0
            self.input_proj.weight[1, 1] = 1.0
            self.input_proj.weight[2, 2:5].fill_(1.0/3.0)
            self.input_proj.bias.zero_()

        # ConvNeXt backbone
        print(f"Creating {variant} backbone for range branch...")
        self.backbone = timm.create_model(
            variant,
            pretrained=pretrained,
            features_only=True,
            out_indices=[0, 1, 2, 3]
        )

        # Get channels and setup output projection
        backbone_channels = self.backbone.feature_info.channels()
        print(f"ConvNeXt output channels: {backbone_channels}")

        scale_to_idx = {4: 0, 8: 1, 16: 2, 32: 3}
        self.feature_idx = scale_to_idx[output_scale]
        backbone_ch = backbone_channels[self.feature_idx]

        self.output_proj = nn.Conv2d(backbone_ch, output_channels, kernel_size=1)
        self.output_channels = output_channels

    def forward(self, range_image):
        """Forward pass."""
        x = self.input_proj(range_image)
        features = self.backbone(x)
        feat = features[self.feature_idx]
        feat = self.output_proj(feat)
        return feat

    def get_output_shape(self, input_h, input_w):
        """Calculate output shape."""
        output_h = input_h // self.output_scale
        output_w = input_w // self.output_scale
        return (output_h, output_w, self.output_channels)


class ResNetRangeBranch(nn.Module):
    """
    ResNet-based range branch (STABLE and FAST).

    Classic ResNet architecture - very stable, fast, and works well
    with any input size. Good baseline for comparison.

    Available variants:
    - resnet18 (11M params) - Fast
    - resnet34 (21M params) - Good balance
    - resnet50 (25M params) - Recommended
    """

    def __init__(self, model_cfgs, input_channels=5):
        super().__init__()

        variant = model_cfgs.get('RESNET_VARIANT', 'resnet50')
        pretrained = model_cfgs.get('RESNET_PRETRAINED', True)
        output_scale = model_cfgs.get('RESNET_OUTPUT_SCALE', 4)
        output_channels = model_cfgs.get('RESNET_OUTPUT_CHANNELS', 256)

        self.output_scale = output_scale

        # Input projection
        self.input_proj = nn.Conv2d(input_channels, 3, kernel_size=1, bias=True)
        with torch.no_grad():
            self.input_proj.weight.zero_()
            self.input_proj.weight[0, 0] = 1.0
            self.input_proj.weight[1, 1] = 1.0
            self.input_proj.weight[2, 2:5].fill_(1.0/3.0)
            self.input_proj.bias.zero_()

        # ResNet backbone
        print(f"Creating {variant} backbone for range branch...")
        self.backbone = timm.create_model(
            variant,
            pretrained=pretrained,
            features_only=True,
            out_indices=[0, 1, 2, 3]
        )

        # Get channels and setup output projection
        backbone_channels = self.backbone.feature_info.channels()
        print(f"ResNet output channels: {backbone_channels}")

        scale_to_idx = {4: 0, 8: 1, 16: 2, 32: 3}
        self.feature_idx = scale_to_idx[output_scale]
        backbone_ch = backbone_channels[self.feature_idx]

        self.output_proj = nn.Conv2d(backbone_ch, output_channels, kernel_size=1)
        self.output_channels = output_channels

    def forward(self, range_image):
        """Forward pass."""
        x = self.input_proj(range_image)
        features = self.backbone(x)
        feat = features[self.feature_idx]
        feat = self.output_proj(feat)
        return feat

    def get_output_shape(self, input_h, input_w):
        """Calculate output shape."""
        output_h = input_h // self.output_scale
        output_w = input_w // self.output_scale
        return (output_h, output_w, self.output_channels)


# Convenience mapping for easy selection
RANGE_BRANCH_REGISTRY = {
    'FlexibleViT': FlexibleViTRangeBranch,
    'ConvNext': ConvNextRangeBranch,
    'ResNet': ResNetRangeBranch,
}


def build_flexible_range_branch(model_cfgs, input_channels=5):
    """
    Factory function to build range branch.

    Args:
        model_cfgs: Model configuration
        input_channels: Input channels (default: 5)

    Returns:
        Range branch module
    """
    branch_type = model_cfgs.get('FLEXIBLE_RANGE_BRANCH', 'ConvNext')

    if branch_type not in RANGE_BRANCH_REGISTRY:
        raise ValueError(
            f"Unknown range branch type: {branch_type}. "
            f"Available: {list(RANGE_BRANCH_REGISTRY.keys())}"
        )

    return RANGE_BRANCH_REGISTRY[branch_type](model_cfgs, input_channels)
