import torch
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform


class TorsionFeatures(BaseTransform):
    def __call__(self, data: Data):
        p = data.pos  # [N, 3]

        diffs = p[1:] - p[:-1]  # [N-1, 3]
        diffs = diffs / diffs.norm(dim=1, keepdim=True)

        normals = torch.cross(diffs[:-1], diffs[1:])  # [N-2, 3]

        cos_torsion = torch.sum(normals[:-1] * normals[1:], dim=1)  # [N-3]
        sin_torsion = torch.sum(torch.cross(normals[:-1], normals[1:]) * diffs[1:-1], dim=1)  # [N-3]

        # extend to [N] by repeating the first and last values
        cos_torsion = torch.cat([cos_torsion[:1], cos_torsion, cos_torsion[-1:], cos_torsion[-1:]])
        sin_torsion = torch.cat([sin_torsion[:1], sin_torsion, sin_torsion[-1:], sin_torsion[-1:]])

        # modify inplace (not sure if this is a good idea)
        data.x = torch.cat([data.x, cos_torsion.unsqueeze(1), sin_torsion.unsqueeze(1)], dim=1)

        return data


class AnglesFeatures(BaseTransform):
    def __call__(self, data: Data):
        p = data.pos  # [N, 3]

        diffs = p[1:] - p[:-1]  # [N-1, 3]
        diffs = diffs / diffs.norm(dim=1, keepdim=True)

        angles_cos = torch.sum(diffs[:-1] * diffs[1:], dim=1)  # [N-2]
        angles_sin = torch.sqrt(1 - angles_cos**2)

        # extend to [N] by repeating the first and last values
        angles_cos = torch.cat([angles_cos[:1], angles_cos, angles_cos[-1:]])
        angles_sin = torch.cat([angles_sin[:1], angles_sin, angles_sin[-1:]])

        # modify inplace (not sure if this is a good idea)
        data.x = torch.cat([data.x, angles_cos.unsqueeze(1), angles_sin.unsqueeze(1)], dim=1)
        return data


class CenterDistance(BaseTransform):
    def __init__(self, normalize=False):
        self.normalize = normalize

    def __call__(self, data: Data):
        p = data.pos  # [N, 3]

        center = p.mean(dim=0, keepdim=True)
        dist = (p - center).norm(dim=1, keepdim=True)
        if self.normalize:
            dist = dist / dist.max()

        data.x = torch.cat([data.x, dist], dim=1)

        return data


class MahalanobisCenterDistance(BaseTransform):
    def __init__(self, rescale=False, normalize=False):
        self.rescale = rescale
        self.normalize = normalize
        assert not (self.rescale and self.normalize)

    def __call__(self, data: Data):
        p = data.pos  # [N, 3]
        p = p - p.mean(dim=0, keepdim=True)

        cov = (p.t() @ p) / p.shape[0]
        inv_cov = torch.inverse(cov)
        dist = torch.sqrt(torch.sum(p @ inv_cov * p, dim=1, keepdim=True))

        if self.rescale:
            dist = dist * (p.norm(dim=1).max() / dist.max())
        if self.normalize:
            dist = dist / dist.max()

        data.x = torch.cat([data.x, dist], dim=1)

        return data


class PositionInSequence(BaseTransform):
    def __call__(self, data: Data):
        t = torch.linspace(0, 1, data.num_nodes)
        data.x = torch.cat([data.x, t.unsqueeze(1)], dim=1)
        return data
