import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.pointnet2 import pointnet2_utils
from lib.pointnet2.pointnet2_modules import PointnetSAModuleVotes, PointnetFPModule


# -------------------------
# Utils: memory-friendly KNN (indices only)
# -------------------------
@torch.no_grad()
def knn_indices_chunked(xyz: torch.Tensor, new_xyz: torch.Tensor, k: int, chunk_size: int = 1024) -> torch.Tensor:
    """
    xyz:     (B, N, 3)
    new_xyz: (B, S, 3)
    return:  (B, S, k) indices in [0, N-1]
    """
    B, N, _ = xyz.shape
    _, S, _ = new_xyz.shape

    best_dist = torch.full((B, S, k), float("inf"), device=xyz.device, dtype=xyz.dtype)
    best_idx = torch.zeros((B, S, k), device=xyz.device, dtype=torch.long)

    for start in range(0, N, chunk_size):
        end = min(N, start + chunk_size)
        chunk = xyz[:, start:end, :]  # (B, nC, 3)

        # (B, S, nC)
        dist2 = (new_xyz[:, :, None, :] - chunk[:, None, :, :]).pow(2).sum(-1)

        kk = min(k, end - start)
        cand_dist, cand_local = torch.topk(dist2, k=kk, dim=-1, largest=False, sorted=True)  # (B,S,kk)
        cand_idx = cand_local + start  # (B,S,kk)

        merged_dist = torch.cat([best_dist, cand_dist], dim=-1)
        merged_idx = torch.cat([best_idx, cand_idx], dim=-1)

        best_dist, sel = torch.topk(merged_dist, k=k, dim=-1, largest=False, sorted=True)
        best_idx = merged_idx.gather(-1, sel)

    return best_idx


@torch.no_grad()
def knn_query(xyz: torch.Tensor, new_xyz: torch.Tensor, k: int, chunk_size: int = 1024) -> torch.Tensor:
    """
    Prefer CUDA KNN kernel if your pointnet2_utils provides it, otherwise chunked torch KNN.
    Return (B,S,k) int indices.
    """
    # Some repos provide knn_point(k, xyz, new_xyz) -> (dist, idx) or idx
    if hasattr(pointnet2_utils, "knn_point"):
        try:
            out = pointnet2_utils.knn_point(k, xyz, new_xyz)
            if isinstance(out, (tuple, list)) and len(out) == 2:
                _, idx = out
            else:
                idx = out
            return idx.long()
        except Exception:
            pass
    return knn_indices_chunked(xyz, new_xyz, k, chunk_size=chunk_size).long()


def safe_ball_query(radius: float, nsample: int, xyz: torch.Tensor, new_xyz: torch.Tensor, knn_fallback_chunk: int = 1024):
    """
    Return (B,S,nsample) indices for ball neighborhood.
    If ball_query is not available, fallback to KNN + radius mask.
    """
    if hasattr(pointnet2_utils, "ball_query"):
        idx = pointnet2_utils.ball_query(radius, nsample, xyz, new_xyz).long()  # (B,S,nsample)
        idx = torch.clamp(idx, min=0)
        return idx

    # Fallback: take KNN candidates and keep those within radius; pad if insufficient
    cand_k = max(nsample * 2, nsample)
    cand_idx = knn_query(xyz, new_xyz, cand_k, chunk_size=knn_fallback_chunk)  # (B,S,cand_k)

    xyz_trans = xyz.transpose(1, 2).contiguous()  # (B,3,N)
    cand_xyz = pointnet2_utils.grouping_operation(xyz_trans, cand_idx.int())  # (B,3,S,cand_k)
    rel = cand_xyz - new_xyz.transpose(1, 2).contiguous().unsqueeze(-1)
    dist2 = (rel ** 2).sum(1)  # (B,S,cand_k)

    r2 = float(radius * radius) + 1e-6
    dist2_masked = dist2.clone()
    dist2_masked[dist2 > r2] = float("inf")

    sel_dist2, sel = torch.topk(dist2_masked, k=nsample, dim=-1, largest=False, sorted=True)
    idx = cand_idx.gather(-1, sel)  # (B,S,nsample)

    # if some are inf (not enough within radius), fill with nearest overall
    fill = cand_idx[..., :1].expand_as(idx)
    idx = torch.where(torch.isinf(sel_dist2), fill, idx)
    return idx.long()


# -------------------------
# PAM: Point Augmented Aggregation Module (paper-style)
# -------------------------
class PAModuleVotes(nn.Module):
    """
    Paper-style PAM:
      - FPS samples query points
      - Ball Query gets local spherical neighbors (nsample)
      - Feature aggregation augments with J (=n_aug) nearest points excluded from the sphere
      - Vector attention aggregation (channel-wise attention) instead of max-pooling

    Output matches PointnetSAModuleVotes: (new_xyz, new_features, fps_inds)
      new_xyz      : (B, npoint, 3)
      new_features : (B, out_channel, npoint)
      fps_inds     : (B, npoint) indices w.r.t input xyz of this layer
    """
    def __init__(
        self,
        npoint: int,
        radius: float,
        nsample: int,
        in_channel: int,     # feature dim (excluding xyz)
        out_channel: int,
        hidden_dim: int = 64,
        n_aug: int = 20,     # J in paper (augmented neighbors outside sphere)
        use_xyz: bool = True,
        normalize_xyz: bool = True,
        knn_chunk_size: int = 1024,
    ):
        super().__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.n_aug = n_aug
        self.use_xyz = use_xyz
        self.normalize_xyz = normalize_xyz
        self.knn_chunk_size = knn_chunk_size

        full_in = in_channel + (3 if use_xyz else 0)
        self.full_in = full_in

        # phi, psi: map point features -> hidden_dim
        self.phi = nn.Sequential(
            nn.Conv2d(full_in, hidden_dim, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(full_in, hidden_dim, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )

        # omega(delta): relative position encoding -> hidden_dim
        self.delta_mlp = nn.Sequential(
            nn.Conv2d(3, hidden_dim, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )

        # project delta to feature space for alpha(x_j + delta)
        self.delta_to_in = nn.Sequential(
            nn.Conv2d(hidden_dim, full_in, 1, bias=False),
            nn.BatchNorm2d(full_in),
        )

        # gamma: attention logits (vector attention, channel-wise)
        self.gamma = nn.Sequential(
            nn.Conv2d(hidden_dim, out_channel, 1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, 1, bias=False),
            nn.BatchNorm2d(out_channel),
        )

        # alpha: value projection
        self.alpha = nn.Sequential(
            nn.Conv2d(full_in, out_channel, 1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
        )

        self.out_mapper = nn.Sequential(
            nn.Conv1d(out_channel, out_channel, 1, bias=False),
            nn.BatchNorm1d(out_channel),
            nn.ReLU(inplace=True),
        )

    def forward(self, xyz: torch.Tensor, features: torch.Tensor):
        """
        xyz:      (B, N, 3)
        features: (B, C, N) or None
        """
        B, N, _ = xyz.shape

        # -------- 1) FPS --------
        if self.npoint < N:
            fps_inds = pointnet2_utils.furthest_point_sample(xyz, self.npoint).long()  # (B, npoint)
        else:
            fps_inds = torch.arange(N, device=xyz.device, dtype=torch.long)[None, :].repeat(B, 1)

        new_xyz = pointnet2_utils.gather_operation(
            xyz.transpose(1, 2).contiguous(), fps_inds.int()
        ).transpose(1, 2).contiguous()  # (B, npoint, 3)

        # -------- 2) Ball Query neighbors (inside sphere) --------
        idx_ball = safe_ball_query(self.radius, self.nsample, xyz, new_xyz, knn_fallback_chunk=self.knn_chunk_size)  # (B,S,nsample)

        # -------- 3) Augment neighbors (outside sphere): J = n_aug --------
        if self.n_aug > 0:
            cand_k = max(self.nsample + self.n_aug, self.n_aug)
            idx_knn = knn_query(xyz, new_xyz, k=cand_k, chunk_size=self.knn_chunk_size)  # (B,S,cand_k)

            xyz_trans = xyz.transpose(1, 2).contiguous()  # (B,3,N)
            knn_xyz = pointnet2_utils.grouping_operation(xyz_trans, idx_knn.int())  # (B,3,S,cand_k)
            rel_knn = knn_xyz - new_xyz.transpose(1, 2).contiguous().unsqueeze(-1)
            dist2 = (rel_knn ** 2).sum(1)  # (B,S,cand_k)

            r2 = float(self.radius * self.radius) + 1e-6
            dist2_masked = dist2.clone()
            dist2_masked[dist2 <= r2] = float("inf")  # keep ONLY outside-sphere points

            sel_dist2, sel = torch.topk(dist2_masked, k=self.n_aug, dim=-1, largest=False, sorted=True)  # (B,S,J)
            idx_aug = idx_knn.gather(-1, sel)  # (B,S,J)

            # if insufficient outside points (inf), fill with nearest overall
            fill = idx_knn[..., :1].expand_as(idx_aug)
            idx_aug = torch.where(torch.isinf(sel_dist2), fill, idx_aug).long()

            idx_all = torch.cat([idx_ball, idx_aug], dim=-1)  # (B,S,nsample+J)
        else:
            idx_all = idx_ball  # (B,S,nsample)

        # -------- 4) Grouping xyz/features --------
        xyz_trans = xyz.transpose(1, 2).contiguous()  # (B,3,N)
        grouped_xyz = pointnet2_utils.grouping_operation(xyz_trans, idx_all.int())  # (B,3,S,K)

        center_xyz = new_xyz.transpose(1, 2).contiguous().unsqueeze(-1)  # (B,3,S,1)
        relative_xyz = grouped_xyz - center_xyz  # (B,3,S,K)

        if self.normalize_xyz and self.radius > 0:
            relative_xyz = relative_xyz / float(self.radius)

        if features is not None:
            grouped_features = pointnet2_utils.grouping_operation(features, idx_all.int())  # (B,C,S,K)
            center_features = pointnet2_utils.gather_operation(features, fps_inds.int()).unsqueeze(-1)  # (B,C,S,1)
        else:
            grouped_features = None
            center_features = None

        # Build x_j and x_i (feature vectors), consistent with PointNet++ use_xyz behavior
        if self.use_xyz:
            if grouped_features is None:
                xj = relative_xyz  # (B,3,S,K)
                xi = torch.zeros((B, 3, self.npoint, 1), device=xyz.device, dtype=xyz.dtype)  # (B,3,S,1)
            else:
                xj = torch.cat([relative_xyz, grouped_features], dim=1)  # (B,3+C,S,K)
                zeros = torch.zeros((B, 3, self.npoint, 1), device=xyz.device, dtype=xyz.dtype)
                xi = torch.cat([zeros, center_features], dim=1)  # (B,3+C,S,1)
        else:
            if grouped_features is None or center_features is None:
                raise ValueError("PAModuleVotes: use_xyz=False requires input features != None.")
            xj = grouped_features
            xi = center_features

        # -------- 5) Vector Attention Aggregation --------
        delta_h = self.delta_mlp(relative_xyz)  # (B,hidden,S,K)

        phi_x = self.phi(xi)                   # (B,hidden,S,1)
        psi_x = self.psi(xj)                   # (B,hidden,S,K)

        energy = phi_x - psi_x + delta_h       # (B,hidden,S,K)
        attn_logits = self.gamma(energy)       # (B,out,S,K)
        attn = F.softmax(attn_logits, dim=-1)  # (B,out,S,K)

        # alpha(x_j + delta)
        delta_in = self.delta_to_in(delta_h)   # (B,full_in,S,K)
        val = self.alpha(xj + delta_in)        # (B,out,S,K)

        new_features = torch.sum(attn * val, dim=-1)  # (B,out,S)
        new_features = self.out_mapper(new_features)  # (B,out,S)

        return new_xyz, new_features, fps_inds


# -------------------------
# Backbone: apply PAM to backbone_module.py
# -------------------------
class Pointnet2Backbone(nn.Module):
    """
    Backbone network for point cloud feature learning.
    Based on PointNet++ SSG backbone, but replace SA1 with paper-style PAM
    to reduce downsampling information loss (N -> 2048).

    If you want to replace SA2-4 as well, you can swap them to PAModuleVotes similarly.
    """
    def __init__(self, input_feature_dim=0, pam_J: int = 20):
        super().__init__()
        self.input_feature_dim = input_feature_dim

        # --------- SA1 replaced by PAM (critical stage) ---------
        self.sa1 = PAModuleVotes(
            npoint=2048,
            radius=0.2,
            nsample=64,
            in_channel=input_feature_dim,  # excluding xyz; PAM internally adds 3 if use_xyz=True
            out_channel=128,
            hidden_dim=64,
            n_aug=pam_J,                   # J in paper (default 20)
            use_xyz=True,
            normalize_xyz=True,
            knn_chunk_size=1024,
        )

        # --------- keep SA2-SA4 as standard PointNet++ for efficiency ---------
        self.sa2 = PointnetSAModuleVotes(
            npoint=1024,
            radius=0.4,
            nsample=32,
            mlp=[128, 128, 128, 256],
            use_xyz=True,
            normalize_xyz=True
        )

        self.sa3 = PointnetSAModuleVotes(
            npoint=512,
            radius=0.8,
            nsample=16,
            mlp=[256, 128, 128, 256],
            use_xyz=True,
            normalize_xyz=True
        )

        self.sa4 = PointnetSAModuleVotes(
            npoint=256,
            radius=1.2,
            nsample=16,
            mlp=[256, 128, 128, 256],
            use_xyz=True,
            normalize_xyz=True
        )

        # --------- FP layers ---------
        self.fp1 = PointnetFPModule(mlp=[256 + 256, 256, 256])
        self.fp2 = PointnetFPModule(mlp=[256 + 256, 256, 256])

    def _break_up_pc(self, pc):
        xyz = pc[..., :3].contiguous()
        features = pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None
        return xyz, features

    def forward(self, data_dict):
        pointcloud = data_dict["point_clouds"]
        xyz, features = self._break_up_pc(pointcloud)

        # --------- SA1 (PAM) ---------
        xyz, features, fps_inds = self.sa1(xyz, features)
        data_dict["sa1_inds"] = fps_inds
        data_dict["sa1_xyz"] = xyz
        data_dict["sa1_features"] = features

        # --------- SA2-SA4 (standard) ---------
        xyz, features, fps_inds2 = self.sa2(xyz, features)
        data_dict["sa2_inds"] = fps_inds2
        data_dict["sa2_xyz"] = xyz
        data_dict["sa2_features"] = features

        xyz, features, _ = self.sa3(xyz, features)
        data_dict["sa3_xyz"] = xyz
        data_dict["sa3_features"] = features

        xyz, features, _ = self.sa4(xyz, features)
        data_dict["sa4_xyz"] = xyz
        data_dict["sa4_features"] = features

        # --------- FP layers ---------
        features = self.fp1(
            data_dict["sa3_xyz"], data_dict["sa4_xyz"],
            data_dict["sa3_features"], data_dict["sa4_features"]
        )
        features = self.fp2(
            data_dict["sa2_xyz"], data_dict["sa3_xyz"],
            data_dict["sa2_features"], features
        )
        data_dict["fp2_features"] = features
        data_dict["fp2_xyz"] = data_dict["sa2_xyz"]

        # --------- Correct index mapping (1024 seeds -> original input indices) ---------
        # sa1_inds: (B,2048) indices in [0, N-1]
        # sa2_inds: (B,1024) indices in [0,2047] w.r.t SA1 output
        data_dict["fp2_inds"] = data_dict["sa1_inds"].gather(1, data_dict["sa2_inds"].long())

        return data_dict


if __name__ == "__main__":
    # Quick sanity test (shape only)
    backbone_net = Pointnet2Backbone(input_feature_dim=3, pam_J=20).cuda()
    backbone_net.eval()
    with torch.no_grad():
        out = backbone_net({"point_clouds": torch.rand(2, 20000, 6).cuda()})
    for k in sorted(out.keys()):
        if torch.is_tensor(out[k]):
            print(k, out[k].shape)
