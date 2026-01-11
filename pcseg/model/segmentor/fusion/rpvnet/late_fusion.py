"""
Late Fusion Module for 2-Branch RPVNet

This module fuses voxel and range features at the point level for semantic segmentation.
"""

import torch
import torch.nn as nn


class LateFusionModule(nn.Module):
    """
    Fuses voxel and range features at point level.

    Strategy:
    1. Convert voxel features → point features (trilinear interpolation)
    2. Sample range features → point features (bilinear grid_sample via range_pxpy)
    3. Fuse in point space (concatenate + MLP)

    Args:
        voxel_ch (int): Number of channels in voxel features
        range_ch (int): Number of channels in range features
        output_ch (int): Number of output channels (typically same as voxel_ch)
        dropout (float): Dropout rate for regularization (default: 0.3)
    """

    def __init__(self, voxel_ch, range_ch, output_ch, dropout=0.3):
        super().__init__()

        # Project features to common dimension
        self.voxel_proj = nn.Linear(voxel_ch, output_ch)
        self.range_proj = nn.Linear(range_ch, output_ch)

        # Fusion MLP
        # Takes concatenated features [N, 2*output_ch] and fuses them to [N, output_ch]
        self.fusion_mlp = nn.Sequential(
            nn.Linear(output_ch * 2, output_ch),
            nn.BatchNorm1d(output_ch),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(output_ch, output_ch)
        )

        self.output_ch = output_ch

    def forward(self, voxel_features, range_features, range_pxpy, z, grid_sample_mode='bilinear'):
        """
        Forward pass: fuse voxel and range features at point level.

        Args:
            voxel_features: SparseTensor (final voxel decoder output)
                Shape: SparseTensor with features [N_voxels, voxel_ch]
            range_features: Tensor [B, C_r, H, W] from Swin
                Shape: [Batch, range_ch, Height, Width]
            range_pxpy: Tensor [N, 3] (batch_idx, px_normalized, py_normalized)
                Point-to-range-image mapping with normalized coordinates [-1, 1]
            z: PointTensor for coordinate tracking
                Used for voxel-to-point interpolation
            grid_sample_mode: str, interpolation mode for grid_sample (default: 'bilinear')

        Returns:
            fused: Tensor [N, output_ch] - fused point features
        """
        # Import here to avoid circular dependency
        from .utils import voxel_to_point, range_to_point

        # 1. Voxel → Point (trilinear interpolation)
        # Converts sparse voxel features to dense point features using interpolation
        voxel_point_feat = voxel_to_point(voxel_features, z, nearest=False)
        voxel_point_feat = self.voxel_proj(voxel_point_feat.F)  # [N, output_ch]

        # 2. Range → Point (bilinear grid_sample)
        # Samples 2D range image features at 3D point locations using range_pxpy mapping
        range_point_feat = range_to_point(range_features, range_pxpy, grid_sample_mode)
        range_point_feat = self.range_proj(range_point_feat)  # [N, output_ch]

        # 3. Fuse (concatenate + MLP)
        # Concatenate voxel and range features and fuse them with MLP
        fused = torch.cat([voxel_point_feat, range_point_feat], dim=1)  # [N, 2*output_ch]
        fused = self.fusion_mlp(fused)  # [N, output_ch]

        return fused


class AttentionLateFusionModule(nn.Module):
    """
    Alternative fusion module using attention-based fusion.

    Instead of simple concatenation, this module uses learned attention
    weights to dynamically weight voxel and range features.

    This can be used as a drop-in replacement for LateFusionModule
    if the simple concat+MLP fusion doesn't work well.
    """

    def __init__(self, voxel_ch, range_ch, output_ch, dropout=0.3):
        super().__init__()

        # Project features to common dimension
        self.voxel_proj = nn.Linear(voxel_ch, output_ch)
        self.range_proj = nn.Linear(range_ch, output_ch)

        # Attention mechanism
        # Learns to weight voxel vs range features
        self.attention = nn.Sequential(
            nn.Linear(output_ch * 2, output_ch),
            nn.ReLU(),
            nn.Linear(output_ch, 2),  # 2 attention weights (voxel, range)
            nn.Softmax(dim=-1)
        )

        # Optional refinement MLP after attention fusion
        self.refinement = nn.Sequential(
            nn.Linear(output_ch, output_ch),
            nn.BatchNorm1d(output_ch),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.output_ch = output_ch

    def forward(self, voxel_features, range_features, range_pxpy, z, grid_sample_mode='bilinear'):
        """
        Forward pass with attention-based fusion.

        Args:
            Same as LateFusionModule

        Returns:
            fused: Tensor [N, output_ch] - attention-fused point features
        """
        from .utils import voxel_to_point, range_to_point

        # 1. Voxel → Point
        voxel_point_feat = voxel_to_point(voxel_features, z, nearest=False)
        voxel_point_feat = self.voxel_proj(voxel_point_feat.F)  # [N, output_ch]

        # 2. Range → Point
        range_point_feat = range_to_point(range_features, range_pxpy, grid_sample_mode)
        range_point_feat = self.range_proj(range_point_feat)  # [N, output_ch]

        # 3. Attention-based fusion
        # Learn attention weights for each modality
        concat_feat = torch.cat([voxel_point_feat, range_point_feat], dim=1)  # [N, 2*output_ch]
        attn_weights = self.attention(concat_feat)  # [N, 2]

        # Apply attention weights
        fused = (attn_weights[:, 0:1] * voxel_point_feat +
                 attn_weights[:, 1:2] * range_point_feat)  # [N, output_ch]

        # Optional refinement
        fused = self.refinement(fused)

        return fused
