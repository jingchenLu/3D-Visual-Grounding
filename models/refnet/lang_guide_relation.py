import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class LangGuidedRelationEncoder(nn.Module):
    def __init__(self, hidden_size=128, lang_dim=128,
                 num_heads=4, geom_dim=12, rel_extra_dim=0 ):
        super().__init__()
        self.hidden_size = hidden_size
        self.lang_dim = lang_dim
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        # 对象 self-attn 的 Q/K/V
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

        # 关系特征：几何 + 其它关系特征 → 中间维
        rel_in_dim = geom_dim + rel_extra_dim
        self.rel_proj = nn.Linear(rel_in_dim, hidden_size)

        # 文本 token 投影
        self.txt_k = nn.Linear(lang_dim, hidden_size)
        self.txt_v = nn.Linear(lang_dim, hidden_size)

        # 关系+语言 → per-head bias（每对 (i,j) 每个 head 一个标量）
        self.rel_lang_to_bias = nn.Linear(hidden_size, num_heads)

    def _build_relation_geom(self, centers, corners):
        """
        你给的几何函数，稍微改成类里的版本：
        centers: (B, N, 3)
        corners: (B, N, 8, 3)
        return:  (B, N, N, 12)
        """
        B, N, _ = centers.shape

        center_i = centers[:, :, None, :]   # B, N, 1, 3
        center_j = centers[:, None, :, :]   # B, 1, N, 3
        rel_vec = center_j - center_i       # B, N, N, 3

        dist = torch.norm(rel_vec, dim=-1, keepdim=True)                # B, N, N, 1
        horiz_dist = torch.norm(rel_vec[..., :2], dim=-1, keepdim=True) # B, N, N, 1
        delta_z = rel_vec[..., 2:3]                                     # B, N, N, 1
        cos_elev = delta_z / (dist + 1e-6)                              # B, N, N, 1

        # 将 [0, 10] 映射到 [-inf, 2.3]，更适合神经网络处理
        dist_log = torch.log(dist + 1e-6)

        sizes = corners.max(dim=2)[0] - corners.min(dim=2)[0]           # B, N, 3
        size_i = sizes[:, :, None, :]                                   # B, N, 1, 3
        size_j = sizes[:, None, :, :]                                   # B, 1, N, 3
        size_diff = size_j - size_i                                     # B, N, N, 3
        size_ratio = size_j / (size_i + 1e-6)                           # B, N, N, 3

        geom = torch.cat(
            [rel_vec, dist_log, horiz_dist, cos_elev, size_diff, size_ratio],
            dim=-1
        )  # B, N, N, 12

        return geom

    def forward(self, feat, centers, corners, lang_tokens,
                extra_rel=None, return_attn=False):
        """
        feat:        (B', N, C)
        centers:     (B', N, 3)
        corners:     (B', N, 8, 3)
        lang_tokens: (B', T, D_lang)
        extra_rel:   (B', N, N, D_extra) or None
        """
        B, N, C = feat.shape
        H = self.num_heads
        D = self.head_dim

        # ---------- 1) 关系几何 + 其它关系特征 ----------
        geom = self._build_relation_geom(centers, corners)   # (B, N, N, 12)
        if extra_rel is not None:
            rel_in = torch.cat([geom, extra_rel], dim=-1)    # (B, N, N, 12+D_extra)
        else:
            rel_in = geom                                    # (B, N, N, 12)

        rel_flat = rel_in.view(B, N * N, -1)                 # (B, N^2, 12+...)
        rel_feat = self.rel_proj(rel_flat)                   # (B, N^2, C)  C=hidden_size

        # ---------- 2) 每个关系对 (i,j) 去 attend 语言 token ----------
        #   q_rel: (B, N^2, C)
        #   k_txt, v_txt: (B, T, C)
        q_rel = rel_feat
        k_txt = self.txt_k(lang_tokens)                      # (B, T, C)
        v_txt = self.txt_v(lang_tokens)                      # (B, T, C)

        # scores_rel_token: (B, N^2, T)
        scores_rel_token = torch.bmm(q_rel, k_txt.transpose(1, 2))
        scores_rel_token = scores_rel_token / math.sqrt(C)
        alpha_rel = torch.softmax(scores_rel_token, dim=-1)  # 对每个 (i,j) 在 token 维度 softmax

        # ctx_rel_flat: (B, N^2, C)
        ctx_rel_flat = torch.bmm(alpha_rel, v_txt)

        # 关系 + 语言上下文
        rel_lang_flat = rel_feat + ctx_rel_flat              # (B, N^2, C)

        # ---------- 3) 映射成 per-head 的 bias ----------
        bias_flat = self.rel_lang_to_bias(rel_lang_flat)     # (B, N^2, H)
        bias = bias_flat.view(B, N, N, H).permute(0, 3, 1, 2)  # (B, H, N, N)

        # 给一个用于可视化的单通道关系分数（比如取均值）
        rel_score = bias.mean(dim=1)                         # (B, N, N)

        # ---------- 4) 用这个 bias 做对象 self-attn ----------
        q = self.q_proj(feat).view(B, N, H, D).transpose(1, 2)  # (B, H, N, D)
        k = self.k_proj(feat).view(B, N, H, D).transpose(1, 2)  # (B, H, N, D)
        v = self.v_proj(feat).view(B, N, H, D).transpose(1, 2)  # (B, H, N, D)

        attn_logits = torch.matmul(q, k.transpose(-2, -1))      # (B, H, N, N)
        attn_logits = attn_logits / math.sqrt(D) + bias         # 加上关系+语言的 bias
        attn = torch.softmax(attn_logits, dim=-1)               # (B, H, N, N)

        out = torch.matmul(attn, v)                             # (B, H, N, D)
        out = out.transpose(1, 2).contiguous().view(B, N, C)    # (B, N, C)
        out = self.out_proj(out)                                # (B, N, C)

        if return_attn:
            return out, attn, rel_score
        else:
            return out, None, None
