import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.transformer.attention import MultiHeadAttention
from models.transformer.mmattention import MultiModalAttention, CrossAttentionDecoderLayer
from models.transformer.utils import PositionWiseFeedForward
import random
import numpy as np


class LangCondGeomBiasSelfAttn(nn.Module):
    """
    Language-Conditioned Geometric Attention Bias Self-Attention
    - x:       (BL, N, C)
    - lang_fea:(BL, T, C)   (tokens, without [CLS] is ok)
    - corners: (BL, N, 8, 3)
    - obj_mask:(BL, N)      bool
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 4,
        geom_dim: int = 12,
        k_rel: int = 32,
        dropout: float = 0.3,
        warmup_epochs: int = 10,
        alpha_init: float = 0.05,
    ):
        super().__init__()
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.geom_dim = geom_dim
        self.k_rel = k_rel
        self.warmup_epochs = warmup_epochs

        # pre-norm for stability
        self.ln = nn.LayerNorm(hidden_size)

        # qkv + out projection
        self.qkv = nn.Linear(hidden_size, 3 * hidden_size, bias=True)
        self.proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.drop = nn.Dropout(p=dropout, inplace=False)
        self.attn_drop = nn.Dropout(p=min(0.1, dropout), inplace=False)

        # geom -> per-head bias
        # output: (BL, N, N, H)
        self.geom2bias = nn.Sequential(
            nn.Linear(geom_dim, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, num_heads),
        )

        # lang -> per-head weights in (0,1)
        self.txt2head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, num_heads),
        )

        # learnable global scale (starts small)
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))

        # init: start from almost "no effect"
        nn.init.zeros_(self.geom2bias[-1].weight)
        nn.init.zeros_(self.geom2bias[-1].bias)
        nn.init.zeros_(self.txt2head[-1].weight)
        nn.init.zeros_(self.txt2head[-1].bias)

    @staticmethod
    def build_relation_geom(centers: torch.Tensor, corners: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """
        centers: (BL, N, 3)
        corners: (BL, N, 8, 3)
        return:  (BL, N, N, 12)
        """
        # pairwise center difference
        center_i = centers[:, :, None, :]          # (BL, N, 1, 3)
        center_j = centers[:, None, :, :]          # (BL, 1, N, 3)
        rel_vec = center_j - center_i              # (BL, N, N, 3)

        # distances
        dist = torch.norm(rel_vec, dim=-1, keepdim=True)                         # (BL, N, N, 1)
        horiz_dist = torch.norm(rel_vec[..., :2], dim=-1, keepdim=True)          # (BL, N, N, 1)
        delta_z = rel_vec[..., 2:3]                                              # (BL, N, N, 1)
        cos_elev = delta_z / (dist + eps)                                        # (BL, N, N, 1)

        # bbox sizes (axis-aligned from corners)
        sizes = corners.max(dim=2)[0] - corners.min(dim=2)[0]                    # (BL, N, 3)
        size_i = sizes[:, :, None, :]                                            # (BL, N, 1, 3)
        size_j = sizes[:, None, :, :]                                            # (BL, 1, N, 3)
        size_diff = size_j - size_i                                              # (BL, N, N, 3)
        size_ratio = size_j / (size_i + eps)                                     # (BL, N, N, 3)

        geom = torch.cat([rel_vec, dist, horiz_dist, cos_elev, size_diff, size_ratio], dim=-1)  # (BL, N, N, 12)
        return geom

    @staticmethod
    def _topk_mask_from_centers(centers: torch.Tensor, k: int) -> torch.Tensor:
        """
        centers: (BL, N, 3)
        return:  (BL, N, N) bool, each i keeps top-k nearest j (including self)
        """
        BL, N, _ = centers.shape
        k = int(min(k, N))
        # cdist: (BL, N, N)
        dist = torch.cdist(centers, centers, p=2)
        idx = torch.topk(dist, k=k, dim=-1, largest=False).indices  # (BL, N, k)
        mask = torch.zeros((BL, N, N), device=centers.device, dtype=torch.bool)
        mask.scatter_(dim=-1, index=idx, value=True)
        return mask

    def forward(
        self,
        x: torch.Tensor,                 # (BL, N, C)
        lang_fea: torch.Tensor,           # (BL, T, C)
        centers: torch.Tensor,            # (BL, N, 3)
        corners: torch.Tensor,            # (BL, N, 8, 3)
        obj_mask: torch.Tensor = None,    # (BL, N) bool
        epoch: int = None,
        return_attn: bool = True,
    ):
        BL, N, C = x.shape
        assert C == self.hidden_size, f"x last dim {C} != hidden_size {self.hidden_size}"

        # ---- build geom bias ----
        geom = self.build_relation_geom(centers, corners)             # (BL, N, N, 12)
        bias = self.geom2bias(geom)                                   # (BL, N, N, H)
        bias = bias.permute(0, 3, 1, 2).contiguous()                  # (BL, H, N, N)

        # ---- language -> head weights ----
        # sentence embedding: mean over tokens
        sent = lang_fea.mean(dim=1)                                  # (BL, C)
        head_w = torch.sigmoid(self.txt2head(sent))                   # (BL, H)
        head_w = head_w[:, :, None, None]                             # (BL, H, 1, 1)

        # ---- warmup scale ----
        warm = 1.0
        if (epoch is not None) and (self.warmup_epochs is not None) and (self.warmup_epochs > 0):
            warm = float(min(1.0, max(0.0, epoch / float(self.warmup_epochs))))

        # ---- attention mask (top-k + objectness) ----
        knn_mask = self._topk_mask_from_centers(centers, self.k_rel)  # (BL, N, N)
        attn_mask = knn_mask
        if obj_mask is not None:
            # valid pairs: i and j both valid
            pair = (obj_mask[:, :, None] & obj_mask[:, None, :])      # (BL, N, N)
            attn_mask = attn_mask & pair

        # ---- self-attn with logit bias ----
        residual = x
        x = self.ln(x)

        qkv = self.qkv(x)                                             # (BL, N, 3C)
        qkv = qkv.view(BL, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]            # (BL, N, H, Dh)
        q = q.permute(0, 2, 1, 3).contiguous()                        # (BL, H, N, Dh)
        k = k.permute(0, 2, 1, 3).contiguous()                        # (BL, H, N, Dh)
        v = v.permute(0, 2, 1, 3).contiguous()                        # (BL, H, N, Dh)

        attn_logits = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)  # (BL, H, N, N)
        attn_logits = attn_logits + (warm * self.alpha) * head_w * bias               # add LC-GeomBias

        # apply mask: invalid -> very negative
        if attn_mask is not None:
            attn_logits = attn_logits.masked_fill(~attn_mask[:, None, :, :], -1e9)

        attn = F.softmax(attn_logits, dim=-1)                          # (BL, H, N, N)
        attn = self.attn_drop(attn)

        out = torch.matmul(attn, v)                                    # (BL, H, N, Dh)
        out = out.permute(0, 2, 1, 3).contiguous().view(BL, N, C)       # (BL, N, C)
        out = self.drop(self.proj(out))
        x = residual + out

        if return_attn:
            # 返回 attn + bias（便于可视化）
            # bias_eff: (BL, H, N, N)
            bias_eff = (warm * self.alpha) * head_w * bias
            return x, attn.detach(), bias_eff.detach()
        return x, None, None