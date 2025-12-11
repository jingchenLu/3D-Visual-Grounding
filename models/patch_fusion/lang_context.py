import torch
import torch.nn as nn
import torch.nn.functional as F


class LangRelContextBlock(nn.Module):
    """
    语言引导的关系上下文块（4 维几何稳妥版）：
    - feat:        (B*L, N, H)    proposal 特征（已经过 cross-attn）
    - centers:     (B*L, N, 3)    proposal 中心坐标
    - text_global: (B*L, H)       句子级语义向量（从 lang_fea 池化而得）

    输出：
    - feat_out:    (B*L, N, H)    融合了“邻居关系 + 语言语义”的增强特征
    """
    def __init__(self, hidden_size, k_neighbors=16):
        super().__init__()
        self.hidden_size = hidden_size
        self.k = k_neighbors

        # 几何特征维度：dx, dy, dz, dist(取 log(1+d))
        geom_dim = 4

        # 几何编码 -> hidden
        self.geom_mlp = nn.Sequential(
            nn.Linear(geom_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )

        # 文本生成 FiLM 门控参数：gate, bias
        self.text_gate_proj = nn.Linear(hidden_size, hidden_size)  # γ
        self.text_bias_proj = nn.Linear(hidden_size, hidden_size)  # β

        # 边注意力 MLP：输入是一个 hidden 向量
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )

        # 邻居消息聚合前的投影
        self.msg_mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )

        # 自身 + 上下文 拼接后，还原回 hidden
        self.out_proj = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU()
        )

        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, feat, centers, text_global):
        """
        feat:        (B*L, N, H)
        centers:     (B*L, N, 3)
        text_global: (B*L, H)
        """
        B_L, N, H = feat.shape

        # 1) 基于中心坐标构建几何 kNN 图
        # dist_matrix: (B*L, N, N)
        dist_matrix = torch.cdist(centers, centers)  # L2 距离

        k = min(self.k, N)
        knn_dist, knn_idx = torch.topk(
            dist_matrix, k=k, dim=-1, largest=False
        )  # (B*L, N, k)

        # 2) 按 knn_idx 取出邻居特征 & 邻居中心
        idx_flat = knn_idx.reshape(B_L, -1)  # (B*L, N*k)

        # 邻居特征
        feat_neigh_flat = torch.gather(
            feat, 1,
            idx_flat.unsqueeze(-1).expand(-1, -1, H)
        )  # (B*L, N*k, H)
        feat_neigh = feat_neigh_flat.view(B_L, N, k, H)  # (B*L, N, k, H)

        # 邻居中心
        centers_neigh_flat = torch.gather(
            centers, 1,
            idx_flat.unsqueeze(-1).expand(-1, -1, centers.shape[-1])
        )  # (B*L, N*k, 3)
        centers_neigh = centers_neigh_flat.view(B_L, N, k, -1)  # (B*L, N, k, 3)

        # 3) 几何特征：dx, dy, dz, dist
        center_i = centers.unsqueeze(2)                         # (B*L, N, 1, 3)
        rel = centers_neigh - center_i                          # (B*L, N, k, 3)  dx,dy,dz
        dist = torch.norm(rel, dim=-1, keepdim=True) + 1e-6     # (B*L, N, k, 1)

        # 为了数值稳定，距离用 log(1+d)
        dist_norm = torch.log1p(dist)                           # (B*L, N, k, 1)

        # 几何 4 维: [dx, dy, dz, log(1+d)]
        geom = torch.cat([rel, dist_norm], dim=-1)              # (B*L, N, k, 4)
        geom_emb = self.geom_mlp(geom)                          # (B*L, N, k, H)

        # 4) 文本 FiLM：根据 text_global 生成 gate & bias
        # text_global: (B*L, H)
        text_gate = torch.sigmoid(self.text_gate_proj(text_global))  # (B*L, H), ∈ (0,1)
        text_bias = self.text_bias_proj(text_global)                 # (B*L, H)

        # broadcast 到每条边
        text_gate = text_gate.unsqueeze(1).unsqueeze(2).expand(-1, N, k, -1)  # (B*L, N, k, H)
        text_bias = text_bias.unsqueeze(1).unsqueeze(2).expand(-1, N, k, -1)  # (B*L, N, k, H)

        # 文本门控几何特征：geom_conditioned = geom_emb * gate + bias
        geom_conditioned = geom_emb * text_gate + text_bias          # (B*L, N, k, H)

        # 5) 构建边特征：自身 + 邻居 + 文本调制几何
        feat_i = feat.unsqueeze(2).expand(-1, N, k, -1)              # (B*L, N, k, H)
        edge_feat = torch.tanh(feat_i + feat_neigh + geom_conditioned)  # (B*L, N, k, H)

        # 6) 边注意力
        edge_logits = self.edge_mlp(edge_feat).squeeze(-1)           # (B*L, N, k)
        edge_alpha = F.softmax(edge_logits, dim=-1).unsqueeze(-1)    # (B*L, N, k, 1)

        # 7) 邻居消息聚合
        msg = self.msg_mlp(feat_neigh)                               # (B*L, N, k, H)
        ctx = (edge_alpha * msg).sum(dim=2)                          # (B*L, N, H)

        # 8) 自身 + 上下文 融合
        out = torch.cat([feat, ctx], dim=-1)                         # (B*L, N, 2H)
        out = self.out_proj(out)                                     # (B*L, N, H)

        # 9) 残差 + LayerNorm，防止过度平滑
        out = self.norm(feat + out)
        return out
