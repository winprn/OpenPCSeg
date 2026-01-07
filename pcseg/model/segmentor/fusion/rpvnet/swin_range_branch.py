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
    from timm.models import swin_transformer
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
                skip_ch = encoder_channels[-(i+1)]  # Fixed: was -(i+2), now -(i+1)
            else:
                # Subsequent upsampling blocks
                in_ch = decoder_channels[i-1]
                skip_ch = encoder_channels[-(i+1)]  # Fixed: was -(i+2), now -(i+1)

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

        # Load pretrained Swin Transformer from timm
        self.backbone = timm.create_model(
            swin_variant,
            pretrained=swin_pretrained,
            features_only=True,
            out_indices=(0, 1, 2, 3),  # 4 scales: 1/4, 1/8, 1/16, 1/32
        )

        # Get Swin output channels (e.g., [96, 192, 384, 768] for Swin-Tiny)
        self.swin_channels = self.backbone.feature_info.channels()

        # Adapt input layer from 3 channels (RGB) to 5 channels (range image)
        self._adapt_input_channels()

        # Optionally modify window size for aspect ratio
        if window_size != [7, 7]:
            self._modify_window_size(window_size)

        # Feature projection layers to match RPVNet channel dimensions
        self.proj_layers = nn.ModuleList([
            nn.Conv2d(self.swin_channels[0], self.target_channels[0], 1),  # Stage 1 (stem)
            nn.Conv2d(self.swin_channels[0], self.target_channels[1], 1),  # Skip 1
            nn.Conv2d(self.swin_channels[1], self.target_channels[2], 1),  # Skip 2
            nn.Conv2d(self.swin_channels[2], self.target_channels[3], 1),  # Skip 3
            nn.Conv2d(self.swin_channels[3], self.target_channels[4], 1),  # Skip 4 / Bottleneck
        ])

        # Decoder (UNet-style upsampling)
        self.decoder = SwinDecoder(
            encoder_channels=[self.target_channels[1], self.target_channels[2],
                            self.target_channels[3], self.target_channels[4]],
            decoder_channels=[self.target_channels[4], self.target_channels[3],
                            self.target_channels[2], self.target_channels[1]],
            dropout_rate=self.dropout_rate
        )

        # Output feature dimension (for RPVNet compatibility)
        self.num_point_features = self.target_channels[1]  # Final decoder output

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

    def _modify_window_size(self, window_size):
        """
        Modify window size for rectangular attention windows.

        WARNING: This may reduce effectiveness of pretrained weights.
        Use only if you need to adapt to very elongated aspect ratios.

        Args:
            window_size: List [H, W] for window dimensions
        """
        # Note: Full implementation would require modifying each SwinTransformerBlock
        # This is complex and may break pretrained weights
        # For now, we use default 7x7 and document the option
        print(f"WARNING: Custom window size {window_size} requested but not implemented.")
        print("Using default 7x7 windows. To implement, modify SwinTransformerBlock directly.")

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
        swin_features = self.backbone(x)

        # Project to target channel dimensions
        feat0 = self.proj_layers[0](swin_features[0])  # Stem output
        skip1 = self.proj_layers[1](swin_features[0])  # Skip connection 1
        skip2 = self.proj_layers[2](swin_features[1])  # Skip connection 2
        skip3 = self.proj_layers[3](swin_features[2])  # Skip connection 3
        skip4 = self.proj_layers[4](swin_features[3])  # Skip connection 4 (bottleneck)

        # Decode with skip connections
        decoder_outputs = self.decoder(skip4, [skip1, skip2, skip3, skip4])

        return {
            'stem': feat0,                           # Early features (for fusion 0)
            'encoder_features': swin_features,       # Raw Swin features
            'projected_features': [feat0, skip2, skip3, skip4],  # Projected features
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
        self.cache = None
        self.num_point_features = self.swin_branch.num_point_features

    def stem(self, x):
        """
        First method called by RPVNet. Process entire image and cache features.

        Args:
            x: Range image [B, 5, H, W]

        Returns:
            Stem features for first fusion point
        """
        # Forward through entire Swin network ONCE
        self.cache = self.swin_branch(x)
        return self.cache['stem']

    def stage1(self, x):
        """
        Return cached stage 1 features.

        Args:
            x: Ignored (output from point_to_range, but we use cached Swin features)

        Returns:
            (features, skip): Stage 1 output and skip connection
        """
        if self.cache is None:
            raise RuntimeError("Must call stem() before stage1()")

        # Return projected first stage features
        feat = self.cache['projected_features'][0]
        skip = self.cache['skips'][0]

        return feat, skip

    def stage2(self, x):
        """Return cached stage 2 features."""
        if self.cache is None:
            raise RuntimeError("Must call stem() before stage2()")

        # Note: Swin features are at indices [0,1,2,3] for scales [1/4, 1/8, 1/16, 1/32]
        # We map to RPVNet stages appropriately
        feat = self.cache['projected_features'][1]
        skip = self.cache['skips'][1]

        return feat, skip

    def stage3(self, x):
        """Return cached stage 3 features."""
        if self.cache is None:
            raise RuntimeError("Must call stem() before stage3()")

        feat = self.cache['projected_features'][2]
        skip = self.cache['skips'][2]

        return feat, skip

    def stage4(self, x):
        """Return cached stage 4 features."""
        if self.cache is None:
            raise RuntimeError("Must call stem() before stage4()")

        feat = self.cache['projected_features'][3]
        skip = self.cache['skips'][3]

        return feat, skip

    def mid_stage(self, x):
        """
        Return cached bottleneck features.

        Note: In SalsaNext, mid_stage processes stage4 output.
              Here we return the bottleneck directly from cache.
        """
        if self.cache is None:
            raise RuntimeError("Must call stem() before mid_stage()")

        return self.cache['bottleneck']

    def up1(self, x, skip):
        """
        Return cached decoder up1 output.

        Args:
            x, skip: Ignored (we return cached decoder features)

        Returns:
            Upsampled features from decoder
        """
        if self.cache is None:
            raise RuntimeError("Must call stem() before up1()")

        return self.cache['decoder_outputs'][0]

    def up2(self, x, skip):
        """Return cached decoder up2 output."""
        if self.cache is None:
            raise RuntimeError("Must call stem() before up2()")

        return self.cache['decoder_outputs'][1]

    def up3(self, x, skip):
        """Return cached decoder up3 output."""
        if self.cache is None:
            raise RuntimeError("Must call stem() before up3()")

        return self.cache['decoder_outputs'][2]

    def up4(self, x, skip):
        """Return cached decoder up4 output."""
        if self.cache is None:
            raise RuntimeError("Must call stem() before up4()")

        return self.cache['decoder_outputs'][3]
