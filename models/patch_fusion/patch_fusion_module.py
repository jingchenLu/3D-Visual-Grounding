import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchLanguageFusion(nn.Module):
    """
    在 SA2 patch 和语言 token 之间做 cross-attention（token -> patch），
    以“残差增强”的方式融合到原始 lang_fea 上，避免直接覆盖导致 ref 分支崩掉。

    假设：
        data_dict["sa2_features"]: (B_pc, Cpc=256, Np)
        data_dict["lang_fea"]:     (B_L = B_pc * L, T, D_lang)
    """

    def __init__(self,
                 pc_feat_dim=256,        # sa2_features 的通道数
                 lang_hidden_size=128,   # 与 LangBertModule 的 lang_hidden_size 一致
                 depth=1,
                 num_heads=4):
        super().__init__()

        self.lang_hidden_size = lang_hidden_size
        self.depth = depth
        self.num_heads = num_heads

        # SA2 patch → 语言维度
        self.pc_proj = nn.Linear(pc_feat_dim, lang_hidden_size)

        # 多层 cross-attn + FFN
        self.layers = nn.ModuleList()
        for _ in range(depth):
            self.layers.append(
                nn.ModuleDict({
                    "attn": nn.MultiheadAttention(
                        embed_dim=lang_hidden_size,
                        num_heads=num_heads,
                        batch_first=True  # 输入输出 (B, seq, D)
                    ),
                    "ffn": nn.Sequential(
                        nn.Linear(lang_hidden_size, lang_hidden_size * 4),
                        nn.ReLU(inplace=True),
                        nn.Linear(lang_hidden_size * 4, lang_hidden_size)
                    ),
                    "norm1": nn.LayerNorm(lang_hidden_size),
                    "norm2": nn.LayerNorm(lang_hidden_size)
                })
            )

        # 🔑 可学习 gate，控制融合强度，初始化很小：
        # sigmoid(-3) ≈ 0.047，相当于一开始几乎不改变原来的 lang_fea
        self.gate_logit = nn.Parameter(torch.tensor(-3.0))

    def forward(self, data_dict):

        # 安全检查
        if ("sa2_features" not in data_dict) or ("lang_fea" not in data_dict):
            return data_dict

        # -------- SA2 patch 特征 --------
        # sa2: (B_pc, 256, Np)
        sa2 = data_dict["sa2_features"]
        B_pc, Cpc, Np = sa2.shape
        # → (B_pc, Np, 256)
        sa2 = sa2.permute(0, 2, 1)

        # -------- 语言特征 --------
        # lang_fea: (B_L, T, D_lang)
        lang_fea = data_dict["lang_fea"]
        B_L, T, D_lang = lang_fea.shape

        assert D_lang == self.lang_hidden_size, \
            f"lang_fea dim = {D_lang}, 但 PatchLanguageFusion 设定的是 {self.lang_hidden_size}"

        # 这里我们只要求 B_L 是 B_pc 的整数倍
        assert B_L % B_pc == 0, f"lang batch {B_L} 不是 sa2 batch {B_pc} 的整数倍"
        L = B_L // B_pc   # 每个场景的句子数
        D = self.lang_hidden_size

        # -------- 投影 SA2 为与 lang_dim 相同 --------
        # (B_pc, Np, Cpc) -> (B_pc, Np, D)
        sa2_proj = self.pc_proj(sa2)

        # -------- 每条句子复制 SA2 patch --------
        # (B_pc, Np, D) -> (B_pc, L, Np, D)
        sa2_proj = sa2_proj.unsqueeze(1).repeat(1, L, 1, 1)
        # -> (B_pc*L, Np, D) == (B_L, Np, D)
        sa2_proj = sa2_proj.reshape(B_L, Np, D)

        # -------- Cross-Attention: token -> patch --------
        # 保存原始 BERT 语言特征
        orig_lang = lang_fea

        x = lang_fea  # (B_L, T, D)

        for layer in self.layers:
            residual = x
            attn_out, _ = layer["attn"](
                query=x,        # (B_L, T, D)
                key=sa2_proj,   # (B_L, Np, D)
                value=sa2_proj  # (B_L, Np, D)
            )
            x = layer["norm1"](residual + attn_out)

            residual = x
            ffn_out = layer["ffn"](x)
            x = layer["norm2"](residual + ffn_out)

        # -------- 残差式融合，而不是直接覆盖 --------
        gate = torch.sigmoid(self.gate_logit)  # 初始约 0.05
        # lang_new = 原始 + gate * (patch_fused - 原始)
        lang_new = orig_lang + gate * (x - orig_lang)

        data_dict["lang_fea"] = lang_new  # (B_L, T, D)
        return data_dict
