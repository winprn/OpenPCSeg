import torch
import torch.nn.functional as F
import torchsparse.nn.functional as tsF
from torchsparse import PointTensor, SparseTensor
from torchsparse.nn.utils import get_kernel_offsets

__all__ = ['initial_voxelize', 'point_to_voxel', 'voxel_to_point', 'range_to_point', 'point_to_range']


# z: PointTensor
# return: SparseTensor
def initial_voxelize(z, init_res, after_res):
    new_float_coord = torch.cat(
        [(z.C[:, :3] * init_res) / after_res, z.C[:, -1].view(-1, 1)], 1,
    )

    pc_hash = tsF.sphash(torch.floor(new_float_coord).int())
    sparse_hash = torch.unique(pc_hash)
    idx_query = tsF.sphashquery(pc_hash, sparse_hash)
    counts = tsF.spcount(idx_query.int(), len(sparse_hash))

    inserted_coords = tsF.spvoxelize(
        torch.floor(new_float_coord),
        idx_query,
        counts,
    )
    inserted_coords = torch.round(inserted_coords).int()
    inserted_feat = tsF.spvoxelize(z.F, idx_query, counts)

    new_tensor = SparseTensor(inserted_feat, inserted_coords, 1)
    new_tensor._caches.cmaps.setdefault(new_tensor.stride, new_tensor.coords)

    z.additional_features['idx_query'][1] = idx_query
    z.additional_features['counts'][1] = counts
    z.C = new_float_coord

    return new_tensor


# x: SparseTensor, z: PointTensor
# return: SparseTensor
def point_to_voxel(x, z):
    if z.additional_features is None or z.additional_features.get(
            'idx_query') is None or z.additional_features['idx_query'].get(
                x.s) is None:
        pc_hash = tsF.sphash(
            torch.cat([
                torch.floor(z.C[:, :3] / x.s[0]).int() * x.s[0],
                z.C[:, -1].int().view(-1, 1)
            ], 1))
        sparse_hash = tsF.sphash(x.C)
        idx_query = tsF.sphashquery(pc_hash, sparse_hash)
        counts = tsF.spcount(idx_query.int(), x.C.shape[0])
        z.additional_features['idx_query'][x.s] = idx_query
        z.additional_features['counts'][x.s] = counts
    else:
        idx_query = z.additional_features['idx_query'][x.s]
        counts = z.additional_features['counts'][x.s]

    inserted_feat = tsF.spvoxelize(z.F, idx_query, counts)
    new_tensor = SparseTensor(inserted_feat, x.C, x.s)
    new_tensor._caches.cmaps = x._caches.cmaps
    new_tensor._caches.kmaps = x._caches.kmaps

    return new_tensor


# x: SparseTensor, z: PointTensor
# return: PointTensor
def voxel_to_point(x, z, nearest=False):
    if z.idx_query is None or z.weights is None or z.idx_query.get(
            x.s) is None or z.weights.get(x.s) is None:
        off = get_kernel_offsets(2, x.s, 1, device=z.F.device)
        old_hash = tsF.sphash(
            torch.cat([
                torch.floor(z.C[:, :3] / x.s[0]).int() * x.s[0],
                z.C[:, -1].int().view(-1, 1)
            ], 1), off)
        pc_hash = tsF.sphash(x.C.to(z.F.device))
        idx_query = tsF.sphashquery(old_hash, pc_hash)
        weights = tsF.calc_ti_weights(z.C, idx_query,
                                    scale=x.s[0]).transpose(0, 1).contiguous()
        idx_query = idx_query.transpose(0, 1).contiguous()
        if nearest:
            weights[:, 1:] = 0.
            idx_query[:, 1:] = -1
        new_feat = tsF.spdevoxelize(x.F, idx_query, weights)
        new_tensor = PointTensor(new_feat,
                                 z.C,
                                 idx_query=z.idx_query,
                                 weights=z.weights)
        new_tensor.additional_features = z.additional_features
        new_tensor.idx_query[x.s] = idx_query
        new_tensor.weights[x.s] = weights
        z.idx_query[x.s] = idx_query
        z.weights[x.s] = weights

    else:
        new_feat = tsF.spdevoxelize(x.F, z.idx_query.get(x.s), z.weights.get(x.s))
        new_tensor = PointTensor(new_feat,
                                 z.C,
                                 idx_query=z.idx_query,
                                 weights=z.weights)
        new_tensor.additional_features = z.additional_features

    return new_tensor


def resample_grid_stacked(predictions, pxpy, grid_sample_mode='bilinear'):
    """
    Resample 2D feature map to point locations using grid_sample.

    Args:
        predictions: [B, C, H, W] - 2D feature map (range image features)
        pxpy: [N, 3] - Point to pixel mapping (batch_idx, px_normalized, py_normalized)
            px, py range: -1 to 1 (normalized coordinates for grid_sample)
        grid_sample_mode: Interpolation mode ('bilinear' or 'nearest')

    Returns:
        resampled: [N, C] - Point features sampled from 2D feature map
    """
    resampled = []
    for cnt, one_batch in enumerate(predictions):
        bs_mask = (pxpy[:, 0] == cnt)
        one_batch = one_batch.unsqueeze(0)  # [1, C, H, W]
        one_pxpy = pxpy[bs_mask][:, 1:].unsqueeze(0).unsqueeze(0)  # [1, 1, N_batch, 2]

        one_resampled = F.grid_sample(one_batch, one_pxpy, mode=grid_sample_mode, align_corners=True)
        one_resampled = one_resampled.squeeze().transpose(0, 1)  # [N_batch, C]
        resampled.append(one_resampled)

    return torch.cat(resampled, dim=0)  # [N, C]


def range_to_point(feature_map, pxpy, grid_sample_mode='bilinear'):
    """
    Convert 2D range image features to point features.

    Args:
        feature_map: [B, C, H, W] - Range image feature map
        pxpy: [N, 3] - Point to pixel mapping (batch_idx, px, py in [-1, 1])
        grid_sample_mode: Interpolation mode

    Returns:
        point_features: [N, C] - Point features sampled from range image
    """
    return resample_grid_stacked(feature_map, pxpy, grid_sample_mode)
