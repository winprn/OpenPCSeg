'''
Swin Transformer Range Branch for RPVNet

Replaces SalsaNext with hierarchical Vision Transformer for better global context modeling.
Uses Swin Transformer architecture with windowed attention for efficiency.

References:
    [1] Liu et al. "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows" (ICCV 2021)
    [2] timm library: https://github.com/rwightman/pytorch-image-models
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import timm
except ImportError:
    raise ImportError("timm library is required. Install with: pip install timm>=0.9.0")


__all__ = ['SwinRangeBranch', 'SwinRangeBranchWrapper']


class DecoderBlock(nn.Module):
    """
    Single decoder block for UNet-style upsampling with skip connections.

    Architecture:
        Upsample(2x) → Conv → BN → ReLU → Concat(skip) → Conv → BN → ReLU
    """

    def __init__(self, in_channels, skip_channels, out_channels, dropout_rate=0.2):
        super().__init__()

        # Upsample path
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        # Refinement after concatenation
        self.conv = nn.Sequential(
            nn.Conv2d(out_channels + skip_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_rate),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, skip):
        """
        Args:
            x: Input features [B, C_in, H, W]
            skip: Skip connection features [B, C_skip, H*2, W*2]

        Returns:
            Upsampled and refined features [B, C_out, H*2, W*2]
        """
        x = self.upsample(x)

        # Handle size mismatch due to odd dimensions
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)

        # Concatenate and refine
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)

        return x


class SwinDecoder(nn.Module):
    """
    UNet-style decoder for Swin Transformer features.

    Takes hierarchical encoder features and progressively upsamples with skip connections.
    """

    def __init__(self, encoder_channels, decoder_channels, dropout_rate=0.2):
        """
        Args:
            encoder_channels: List of encoder output channels [C1, C2, C3, C4]
            decoder_channels: List of decoder output channels [D1, D2, D3, D4]
            dropout_rate: Dropout probability for regularization
        """
        super().__init__()

        self.up_blocks = nn.ModuleList()

        for i in range(len(decoder_channels)):
            if i == 0:
                # First upsampling from bottleneck
                in_ch = encoder_channels[-1]
                skip_ch = encoder_channels[-(i+2)] if i+2 <= len(encoder_channels) else encoder_channels[-1]
            else:
                # Subsequent upsampling blocks
                in_ch = decoder_channels[i-1]
                skip_ch = encoder_channels[-(i+2)] if i+2 <= len(encoder_channels) else encoder_channels[0]

            out_ch = decoder_channels[i]

            self.up_blocks.append(
                DecoderBlock(in_ch, skip_ch, out_ch, dropout_rate)
            )

    def forward(self, x, skips):
        """
        Args:
            x: Bottleneck features [B, C, H, W]
            skips: List of skip connection features [skip1, skip2, skip3, skip4]
                   From low to high resolution

        Returns:
            List of decoder outputs at different scales [up1, up2, up3, up4]
        """
        outputs = []

        for i, up_block in enumerate(self.up_blocks):
            # Skip connections in reverse order (high level to low level)
            skip = skips[-(i+1)]
            x = up_block(x, skip)
            outputs.append(x)

        return outputs


class SwinRangeBranch(nn.Module):
    """
    Swin Transformer-based range branch for RPVNet.

    Architecture:
        Input [B, 5, H, W] → Patch Embed → Swin Stages (4 scales) → Decoder → Outputs

    Features:
        - Hierarchical multi-scale features via Swin Transformer encoder
        - UNet-style decoder with skip connections
        - Channel projection to match RPVNet dimensions
        - Pretrained on ImageNet-1K for better initialization
    """

    def __init__(self, model_cfgs):
        super().__init__()

        # Configuration
        self.cr = model_cfgs.get('cr', 1.75)
        self.dropout_rate = model_cfgs.get('DROPOUT_P', 0.3)

        # Target channel dimensions (matching SalsaNext with cr multiplier)
        base_channels = [32, 32, 64, 128, 256]
        self.target_channels = [int(self.cr * x) for x in base_channels]

        # Swin Transformer configuration
        swin_variant = model_cfgs.get('SWIN_VARIANT', 'swin_tiny_patch4_window7_224')
        swin_pretrained = model_cfgs.get('SWIN_PRETRAINED', True)
        window_size = model_cfgs.get('SWIN_WINDOW_SIZE', [7, 7])

        # Range image size (H, W) - can be configured per dataset
        # SemanticKITTI: (64, 2048), nuScenes: (32, 2048) after resize
        self.range_img_size = model_cfgs.get('RANGE_IMG_SIZE', (64, 2048))

        # Automatically adjust window size if default doesn't work
        # After patch embedding (patch_size=4): H/4, W/4
        # Window size must divide evenly into these dimensions
        h_feat, w_feat = self.range_img_size[0] // 4, self.range_img_size[1] // 4

        if window_size == [7, 7]:
            # Auto-adjust window size to be compatible
            # For 64×2048 → 16×512 after patch embed
            # Use 8×8 (divides 16 and 512 evenly: 16/8=2, 512/8=64)
            if h_feat % 7 != 0 or w_feat % 7 != 0:
                window_size = [8, 8]
                print(f"INFO: Auto-adjusted window size to {window_size} for compatibility")
                print(f"      Feature map size after patch embed: ({h_feat}, {w_feat})")

        self.window_size = window_size

        # Load pretrained Swin Transformer from timm
        self.backbone = timm.create_model(
            swin_variant,
            pretrained=swin_pretrained,
            features_only=True,
            out_indices=(0, 1, 2, 3),  # 4 scales: 1/4, 1/8, 1/16, 1/32
            img_size=self.range_img_size,  # Set to range image size (H, W)
            window_size=self.window_size,  # Use compatible window size
        )

        # Get Swin output channels (e.g., [96, 192, 384, 768] for Swin-Tiny)
        self.swin_channels = self.backbone.feature_info.channels()

        # Disable strict image size checking
        self._disable_strict_img_size()

        # Adapt input layer from 3 channels (RGB) to 5 channels (range image)
        self._adapt_input_channels()

        # Feature projection layers to match RPVNet channel dimensions
        # Note: SalsaNext mid_stage is at 1/16 scale, so we use swin_features[2] for bottleneck
        self.proj_layers = nn.ModuleList([
            nn.Conv2d(self.swin_channels[0], self.target_channels[0], 1),  # Stage 1 (stem)
            nn.Conv2d(self.swin_channels[0], self.target_channels[1], 1),  # Skip 1 (1/4 scale)
            nn.Conv2d(self.swin_channels[1], self.target_channels[2], 1),  # Skip 2 (1/8 scale)
            nn.Conv2d(self.swin_channels[2], self.target_channels[3], 1),  # Skip 3 (1/16 scale)
            nn.Conv2d(self.swin_channels[2], self.target_channels[4], 1),  # Bottleneck (1/16 scale, matches SalsaNext)
        ])

        # Decoder (UNet-style upsampling)
        # Match SalsaNext decoder channel pattern: cs[5:9] = [256, 128, 96, 96] scaled by cr
        # Note: We use 3 scales from Swin (1/4, 1/8, 1/16) + bottleneck at 1/16
        decoder_channels_list = [int(self.cr * 256), int(self.cr * 128),
                                int(self.cr * 96), int(self.cr * 96)]
        self.decoder = SwinDecoder(
            encoder_channels=[self.target_channels[1], self.target_channels[2],
                            self.target_channels[3], self.target_channels[4]],
            decoder_channels=decoder_channels_list,
            dropout_rate=self.dropout_rate
        )

        # Output feature dimension (for RPVNet compatibility)
        self.num_point_features = decoder_channels_list[-1]  # Final decoder output (up4)

    def _disable_strict_img_size(self):
        """
        Disable strict image size checking in patch embedding layer.

        This allows the model to accept range images of different sizes
        instead of the default ImageNet size (224×224).
        """
        if hasattr(self.backbone, 'patch_embed'):
            patch_embed = self.backbone.patch_embed
            # Set strict_img_size to False if the attribute exists
            if hasattr(patch_embed, 'strict_img_size'):
                patch_embed.strict_img_size = False
            # Also update img_size to our range image size
            patch_embed.img_size = self.range_img_size

    def _adapt_input_channels(self):
        """
        Modify Swin's patch embedding layer to accept 5-channel input.

        Strategy:
            - Copy pretrained RGB weights to first 3 channels
            - Initialize channels 4-5 with average of RGB weights
            - Preserves pretrained knowledge while extending to new inputs
        """
        patch_embed = self.backbone.patch_embed
        old_proj = patch_embed.proj

        # Create new projection layer with 5 input channels
        new_proj = nn.Conv2d(
            in_channels=5,  # Range image channels
            out_channels=old_proj.out_channels,
            kernel_size=old_proj.kernel_size,
            stride=old_proj.stride,
            padding=old_proj.padding,
            bias=old_proj.bias is not None
        )

        # Initialize weights
        with torch.no_grad():
            # Copy RGB channel weights
            new_proj.weight[:, :3, :, :] = old_proj.weight

            # Initialize extra channels with average of RGB
            avg_weight = old_proj.weight.mean(dim=1, keepdim=True)
            new_proj.weight[:, 3:, :, :] = avg_weight.repeat(1, 2, 1, 1)

            # Copy bias if exists
            if old_proj.bias is not None:
                new_proj.bias.copy_(old_proj.bias)

        # Replace patch embedding projection
        patch_embed.proj = new_proj

    def forward(self, x):
        """
        Forward pass through Swin encoder and decoder.

        Args:
            x: Range image [B, 5, H, W]

        Returns:
            Dictionary containing:
                - 'stem': Early features for fusion
                - 'encoder_features': List of 4 encoder outputs
                - 'skips': List of 4 skip connection features
                - 'bottleneck': Deepest encoder features
                - 'decoder_outputs': List of 4 decoder outputs
        """
        B, _, H, W = x.shape

        # Extract hierarchical features from Swin backbone
        # Returns list of features at 4 scales: [1/4, 1/8, 1/16, 1/32]
        swin_features_raw = self.backbone(x)

        # Convert from [B, H, W, C] to [B, C, H, W] if needed
        # timm's Swin returns features in channels-last format
        swin_features = []
        for i, feat in enumerate(swin_features_raw):
            # Check if feature is in channels-last format [B, H, W, C]
            # Expected channel dimension: self.swin_channels[i] (e.g., 96, 192, 384, 768)
            if len(feat.shape) == 4:
                # If last dimension matches expected channels, it's [B, H, W, C]
                if feat.shape[-1] == self.swin_channels[i]:
                    feat = feat.permute(0, 3, 1, 2).contiguous()
                # If second dimension matches, it's already [B, C, H, W]
                elif feat.shape[1] != self.swin_channels[i]:
                    raise RuntimeError(
                        f"Unexpected feature shape at scale {i}: {feat.shape}. "
                        f"Expected channels: {self.swin_channels[i]}"
                    )
            swin_features.append(feat)

        # Project to target channel dimensions
        feat0 = self.proj_layers[0](swin_features[0])  # Stem output (1/4 scale)
        skip1 = self.proj_layers[1](swin_features[0])  # Skip connection 1 (1/4 scale)
        skip2 = self.proj_layers[2](swin_features[1])  # Skip connection 2 (1/8 scale)
        skip3 = self.proj_layers[3](swin_features[2])  # Skip connection 3 (1/16 scale)
        skip4 = self.proj_layers[4](swin_features[2])  # Bottleneck (1/16 scale, matches SalsaNext mid_stage)

        # Decode with skip connections
        # Note: skip4 is the bottleneck input, skips list contains intermediate features only
        decoder_outputs = self.decoder(skip4, [skip1, skip2, skip3])

        return {
            'stem': feat0,                           # Early features (for fusion 0)
            'encoder_features': swin_features,       # Raw Swin features
            'projected_features': [skip1, skip2, skip3, skip4],  # Stage outputs
            'skips': [skip1, skip2, skip3, skip4],  # Skip connections
            'bottleneck': skip4,                     # Bottleneck (for fusion 1)
            'decoder_outputs': decoder_outputs,      # [up1, up2, up3, up4]
        }


class SwinRangeBranchWrapper(nn.Module):
    """
    Wrapper to match SalsaNext API for drop-in replacement in RPVNet.

    Problem: RPVNet calls range branch methods sequentially (stem, stage1, stage2, ...),
            but Swin Transformer processes the entire image at once.

    Solution: Cache all features on first call (stem), then return cached features
             for subsequent method calls.

    API Compatibility:
        - stem(x) → features
        - stage1(x) → (features, skip)
        - stage2(x) → (features, skip)
        - stage3(x) → (features, skip)
        - stage4(x) → (features, skip)
        - mid_stage(x) → features
        - up1(x, skip) → features
        - up2(x, skip) → features
        - up3(x, skip) → features
        - up4(x, skip) → features
    """

    def __init__(self, model_cfgs, input_channels=5):
        super().__init__()

        self.swin_branch = SwinRangeBranch(model_cfgs)
        self.swin_cache = None
        self.num_point_features = self.swin_branch.num_point_features

        # Get channel dimensions for fusion layers
        cr = model_cfgs.get('cr', 1.75)
        target_channels = [int(cr * x) for x in [32, 32, 64, 128, 256]]
        decoder_channels = [int(cr * 256), int(cr * 128), int(cr * 96), int(cr * 96)]

        # Fusion layers to blend cached Swin features with RPVNet's fused inputs
        # One fusion layer for each stage (stage1-4 + mid_stage = 5 total)
        self.fusion_conv_stages = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(target_channels[i], target_channels[i], 3, padding=1),
                nn.BatchNorm2d(target_channels[i]),
                nn.ReLU(inplace=True)
            )
            for i in range(1, 5)  # stage1-4: indices 1-4
        ])

        # Add fusion layer for mid_stage (bottleneck)
        self.fusion_conv_mid = nn.Sequential(
            nn.Conv2d(target_channels[4], target_channels[4], 3, padding=1),
            nn.BatchNorm2d(target_channels[4]),
            nn.ReLU(inplace=True)
        )

        # Fusion layers for decoder (up1-4)
        self.fusion_conv_decoder = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(decoder_channels[i], decoder_channels[i], 3, padding=1),
                nn.BatchNorm2d(decoder_channels[i]),
                nn.ReLU(inplace=True)
            )
            for i in range(len(decoder_channels))
        ])

    def stem(self, x):
        """
        First method called by RPVNet. Process entire image and cache features.

        Args:
            x: Range image [B, 5, H, W]

        Returns:
            Stem features for first fusion point
        """
        # Clear any stale cache from previous batch
        self.swin_cache = None

        # Forward through entire Swin network ONCE and cache all features
        self.swin_cache = self.swin_branch(x)
        return self.swin_cache['stem']

    def stage1(self, x):
        """
        Accept and process fusion input.

        Args:
            x: Fused input from RPVNet (point-to-range converted features) [B, C, H, W]

        Returns:
            (features, skip): Stage 1 output and skip connection
        """
        if self.swin_cache is None:
            raise RuntimeError("Must call stem() before stage1()")

        # Get cached Swin features
        swin_feat = self.swin_cache['projected_features'][0]
        skip = self.swin_cache['skips'][0]

        # Blend cached Swin features with RPVNet's fused input using residual connection
        fused = swin_feat + self.fusion_conv_stages[0](x)

        return fused, skip

    def stage2(self, x):
        """Accept and process fusion input."""
        if self.swin_cache is None:
            raise RuntimeError("Must call stem() before stage2()")

        # Get cached Swin features and blend with fused input
        swin_feat = self.swin_cache['projected_features'][1]
        skip = self.swin_cache['skips'][1]

        # Blend with RPVNet's fused input
        fused = swin_feat + self.fusion_conv_stages[1](x)

        return fused, skip

    def stage3(self, x):
        """Accept and process fusion input."""
        if self.swin_cache is None:
            raise RuntimeError("Must call stem() before stage3()")

        # Get cached Swin features and blend with fused input
        swin_feat = self.swin_cache['projected_features'][2]
        skip = self.swin_cache['skips'][2]

        # Blend with RPVNet's fused input
        fused = swin_feat + self.fusion_conv_stages[2](x)

        return fused, skip

    def stage4(self, x):
        """Accept and process fusion input."""
        if self.swin_cache is None:
            raise RuntimeError("Must call stem() before stage4()")

        # Get cached Swin features and blend with fused input
        swin_feat = self.swin_cache['projected_features'][3]
        skip = self.swin_cache['skips'][3]

        # Blend with RPVNet's fused input
        fused = swin_feat + self.fusion_conv_stages[3](x)

        return fused, skip

    def mid_stage(self, x):
        """
        Accept and process bottleneck fusion input.

        Note: In SalsaNext, mid_stage processes stage4 output.
              Here we blend the bottleneck with fused input.
        """
        if self.swin_cache is None:
            raise RuntimeError("Must call stem() before mid_stage()")

        # Get cached bottleneck features
        bottleneck = self.swin_cache['bottleneck']

        # Blend with RPVNet's fused input
        fused = bottleneck + self.fusion_conv_mid(x)

        return fused

    def up1(self, x, skip):
        """
        Accept and process decoder fusion input.

        Args:
            x: Fused input from RPVNet (point-to-range converted features) [B, C, H, W]
            skip: Skip connection (ignored, we use cached skips from encoder)

        Returns:
            Upsampled and fused features
        """
        if self.swin_cache is None:
            raise RuntimeError("Must call stem() before up1()")

        # Get cached decoder output
        swin_decoder = self.swin_cache['decoder_outputs'][0]

        # Blend with RPVNet's fused input
        fused = swin_decoder + self.fusion_conv_decoder[0](x)

        return fused

    def up2(self, x, skip):
        """Accept and process decoder fusion input."""
        if self.swin_cache is None:
            raise RuntimeError("Must call stem() before up2()")

        # Get cached decoder output and blend with fused input
        swin_decoder = self.swin_cache['decoder_outputs'][1]
        fused = swin_decoder + self.fusion_conv_decoder[1](x)

        return fused

    def up3(self, x, skip):
        """Accept and process decoder fusion input."""
        if self.swin_cache is None:
            raise RuntimeError("Must call stem() before up3()")

        # Get cached decoder output and blend with fused input
        swin_decoder = self.swin_cache['decoder_outputs'][2]
        fused = swin_decoder + self.fusion_conv_decoder[2](x)

        return fused

    def up4(self, x, skip):
        """Accept and process decoder fusion input."""
        if self.swin_cache is None:
            raise RuntimeError("Must call stem() before up4()")

        # Get cached decoder output and blend with fused input
        swin_decoder = self.swin_cache['decoder_outputs'][3]
        fused = swin_decoder + self.fusion_conv_decoder[3](x)

        return fused
