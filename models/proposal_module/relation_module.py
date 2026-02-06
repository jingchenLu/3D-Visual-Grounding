import torch
import torch.nn as nn
import torch.nn.functional as F
from models.transformer.attention import MultiHeadAttention
from models.transformer.utils import PositionWiseFeedForward
import random
from models.jointnet.gsa_module import GlobalBiasGSA
from models.jointnet.object_gsa_module import ObjectnessAwareGSA


class FusionConcatDelta(nn.Module):
    def __init__(self, c=128, hidden=256, init_scale=0.1, dropout=0.1):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(init_scale))
        self.mlp = nn.Sequential(
            nn.LayerNorm(2*c),
            nn.Linear(2*c, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, c),
        )
        self.out_ln = nn.LayerNorm(c)

    def forward(self, rel_feat, gsa_feat):
        # delta: (B,N,C)
        delta = self.mlp(torch.cat([rel_feat, gsa_feat], dim=-1))
        out = rel_feat + self.scale * delta
        return self.out_ln(out), self.scale.detach()

class RelationModule(nn.Module):
    """
    Geometry-aware object relation reasoning module.

    - 使用改进的几何编码 (距离 / 高度差 / 尺寸差 / 尺寸比)，兼顾视角敏感和部分视角不变信息；
    - 通过几何 MLP 生成 per-head attention bias，作为 MultiHeadAttention 的加性权重；
    - 可选 kNN 稀疏图，减少远距离 proposal 之间的噪声关系；
    - 融合 3D box embedding 和多视图 / RGB 对象特征 obj_embedding。
    """

    def __init__(
        self,
        num_proposals: int = 256,
        hidden_size: int = 128,
        lang_num_size: int = 300,   # 保留接口，当前实现未显式使用
        det_channel: int = 128,
        head: int = 4,
        depth: int = 2,
        k_neighbors: int = None,      # 若为 None 或 >= num_proposals 则退化为全连接图
        use_obj_embedding=True
    ):
        super().__init__()
        self.use_box_embedding = True
        self.use_dist_weight_matrix = True

        self.num_proposals = num_proposals
        self.hidden_size = hidden_size
        self.head = head
        self.depth = depth
        self.k_neighbors = k_neighbors
        self.use_obj_embedding = use_obj_embedding

        # 将 detection head 输出的 bbox feature 映射到 relation hidden space
        self.features_concat = nn.Sequential(
            nn.Conv1d(det_channel, hidden_size, 1),
            nn.BatchNorm1d(hidden_size),
            nn.PReLU(hidden_size),
            nn.Conv1d(hidden_size, hidden_size, 1),
        )

        # 几何特征维度：
        # rel_vec(3) + dist(1) + horiz_dist(1) + cos_elev(1)
        # + size_diff(3) + size_ratio(3) = 12
        self.geom_dim = 12

        # 几何 -> attention bias (per head)
        self.self_attn_fc = nn.ModuleList(
            nn.Sequential(
                nn.Linear(self.geom_dim, 64),
                nn.ReLU(inplace=True),
                nn.LayerNorm(64),
                nn.Linear(64, head)
            ) for _ in range(depth)
        )

        # 关系 self-attention
        self.self_attn = nn.ModuleList(
            MultiHeadAttention(
                d_model=hidden_size,
                d_k=hidden_size // head,
                d_v=hidden_size // head,
                h=head
            ) for _ in range(depth)
        )

        # 3D box embedding (center + 8 corner offset，共 27 维)
        self.bbox_embedding = nn.ModuleList(
            nn.Linear(27, hidden_size) for _ in range(depth)
        )

        # 多视图 / RGB 对象特征 embedding（point_clouds 中的 128 维特征）
        self.obj_embedding = nn.ModuleList(
            nn.Linear(128, hidden_size) for _ in range(depth)
        )

        self.gsa_branch = ObjectnessAwareGSA(hidden_size=hidden_size)

        # [NEW] 使用 Concat 融合 (你验证过比较好的结构)
        self.rel_global_fusion = FusionConcatDelta(
            c=hidden_size, 
            hidden=hidden_size*2, 
            init_scale=0.1,  # 从头训可以设0.1，微调建议0.01
            dropout=0.1
        )

        # [NEW] 可选 warmup：前 N 个 epoch 不用 GSA（默认 0：不 warmup）
        self.gsa_warmup_epochs = 0


    @staticmethod
    def _get_bbox_centers(self, corners: torch.Tensor) -> torch.Tensor:
        """
        corners: (B, N, 8, 3)
        return:  (B, N, 3)
        """
        coord_min = torch.min(corners, dim=2)[0]
        coord_max = torch.max(corners, dim=2)[0]
        return (coord_min + coord_max) / 2

    def _get_bbox_centers(self, corners):
        # 复用你原来的中心计算实现
        coord_min = torch.min(corners, dim=2)[0]
        coord_max = torch.max(corners, dim=2)[0]
        return (coord_min + coord_max) / 2
    

    def _build_relation_geom(self, centers: torch.Tensor, corners: torch.Tensor) -> torch.Tensor:
        """
        构造旋转部分不变 + 视角敏感混合的几何特征

        centers: (B, N, 3)
        corners: (B, N, 8, 3)
        return:  (B, N, N, geom_dim)
        """
        B, N, _ = centers.shape

        # pairwise center difference
        center_i = centers[:, :, None, :]   # B, N, 1, 3
        center_j = centers[:, None, :, :]   # B, 1, N, 3
        rel_vec = center_j - center_i       # B, N, N, 3   (dx, dy, dz)

        # 距离相关
        dist = torch.norm(rel_vec, dim=-1, keepdim=True)                     # B, N, N, 1
        horiz_dist = torch.norm(rel_vec[..., :2], dim=-1, keepdim=True)      # B, N, N, 1
        delta_z = rel_vec[..., 2:3]                                          # B, N, N, 1
        cos_elev = delta_z / (dist + 1e-6)                                   # B, N, N, 1

        # bbox 尺寸：dx, dy, dz
        sizes = corners.max(dim=2)[0] - corners.min(dim=2)[0]                # B, N, 3
        size_i = sizes[:, :, None, :]                                        # B, N, 1, 3
        size_j = sizes[:, None, :, :]                                        # B, 1, N, 3
        size_diff = size_j - size_i                                          # B, N, N, 3
        size_ratio = size_j / (size_i + 1e-6)                                # B, N, N, 3

        geom = torch.cat(
            [rel_vec, dist, horiz_dist, cos_elev, size_diff, size_ratio],
            dim=-1
        )  # B, N, N, 12

        return geom

    def forward(self, data_dict):
        """
        Args:
            data_dict 需要包含:
                - pred_bbox_feature: (B, C_det, N)
                - pred_bbox_corner:  (B, N, 8, 3)
                - point_clouds:      (B, P, 3 + feat)
                - seed_inds:         (B, Ns)
                - aggregated_vote_inds: (B, N)
        """
        # 1) proposal 粗特征 -> relation hidden space
        features = data_dict["pred_bbox_feature"].permute(0, 2, 1)   # (B, N, C_det)
        features = self.features_concat(features).permute(0, 2, 1)  # (B, N, hidden)
        batch_size, num_proposal = features.shape[:2]

        corners = data_dict["pred_bbox_corner"]                        # (B, N, 8, 3)
        centers = self._get_bbox_centers(corners)                      # (B, N, 3)

        # 2) 构造几何特征 & 可选 kNN mask
        if self.use_dist_weight_matrix:
            geom = self._build_relation_geom(centers, corners)         # (B, N, N, geom_dim)

            if self.k_neighbors is not None and self.k_neighbors < num_proposal:
                with torch.no_grad():
                    # pairwise 距离用于 kNN
                    pairwise_dist = torch.norm(
                        centers[:, :, None, :] - centers[:, None, :, :],
                        dim=-1
                    )  # (B, N, N)
                    knn_idx = torch.topk(
                        pairwise_dist,
                        k=self.k_neighbors,
                        dim=-1,
                        largest=False
                    )[1]  # (B, N, K)

                    # mask 初始化为极小值，表示 "不相连"
                    mask = torch.ones_like(pairwise_dist) * -1e4       # (B, N, N)
                    mask.scatter_(2, knn_idx, 0.0)                     # 仅保留 K 个最近邻
                knn_mask = mask.unsqueeze(1)                           # (B, 1, N, N)
            else:
                knn_mask = None
        else:
            geom = None
            knn_mask = None

        dist_weights = None
        attention_matrix_way = "mul"

        # 3) 多层 relation reasoning
        for i in range(self.depth):
            # 3.1 几何 -> per-head attention bias
            if self.use_dist_weight_matrix and geom is not None:
                # (B, N, N, head)
                attn_bias = self.self_attn_fc[i](geom.detach())
                # -> (B, head, N, N)
                dist_weights = attn_bias.permute(0, 3, 1, 2).contiguous()

                if knn_mask is not None:
                    dist_weights = dist_weights + knn_mask

                attention_matrix_way = "add"
            else:
                dist_weights = None
                attention_matrix_way = "mul"

            # 3.2 多视图 / RGB 对象特征
            if self.use_obj_embedding:
                obj_feat = data_dict["point_clouds"][..., 6:6 + 128].permute(0, 2, 1)
                obj_feat_dim = obj_feat.shape[1]
                obj_feat_id_seed = data_dict["seed_inds"]
                obj_feat_id_seed = obj_feat_id_seed.long() + (
                    (torch.arange(batch_size) * obj_feat.shape[1])[:, None].to(obj_feat_id_seed.device))
                obj_feat_id_seed = obj_feat_id_seed.reshape(-1)
                obj_feat_id_vote = data_dict["aggregated_vote_inds"]
                obj_feat_id_vote = obj_feat_id_vote.long() + (
                    (torch.arange(batch_size) * data_dict["seed_inds"].shape[1])[:, None].to(
                        obj_feat_id_vote.device))
                obj_feat_id_vote = obj_feat_id_vote.reshape(-1)
                obj_feat_id = obj_feat_id_seed[obj_feat_id_vote]
                obj_feat = obj_feat.reshape(-1, obj_feat_dim)[obj_feat_id].reshape(batch_size, num_proposal,
                                                                                   obj_feat_dim)
                obj_embedding = self.obj_embedding[i](obj_feat)
                features = features + 0.1 * obj_embedding

            # 3.3 box 几何 embedding (center + corner offset)
            if self.use_box_embedding:
                # batch_size, num_proposals, 3
                centers = self._get_bbox_centers(corners)
                num_proposals = centers.shape[1]
                manual_bbox_feat = torch.cat(
                    [centers, (corners - centers[:, :, None, :]
                               ).reshape(batch_size, num_proposals, -1)],
                    dim=-1).float()
                bbox_embedding = self.bbox_embedding[i](manual_bbox_feat)
                features = features + bbox_embedding

            # 3.4 基于几何 bias 的 self-attention 更新关系特征
            features = self.self_attn[i](
                features, features, features,
                attention_weights=dist_weights,
                way=attention_matrix_way
            )

        data_dict["dist_weights"] = dist_weights
        data_dict["attention_matrix_way"] = attention_matrix_way
        data_dict["bbox_feature"] = features

        # data_dict["gsa_scale"] = gsa_gate_mean

        return data_dict
