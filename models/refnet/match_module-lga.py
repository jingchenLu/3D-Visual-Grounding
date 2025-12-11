import torch
import torch.nn as nn
import torch.nn.functional as F
from models.transformer.attention import MultiHeadAttention
from models.transformer.mmattention import MultiModalAttention, CrossAttentionDecoderLayer
from models.transformer.utils import PositionWiseFeedForward
import random
import numpy as np

class MatchModule(nn.Module):
    def __init__(self, num_proposals=256, lang_size=256, hidden_size=128, lang_num_size=300, det_channel=128, head=4, use_lang_emb=False, use_pc_encoder=False, use_match_con_loss=False, depth=2, use_reg_head=False):
        super().__init__()
        self.num_proposals = num_proposals
        self.lang_size = lang_size
        self.hidden_size = hidden_size
        self.use_lang_emb = use_lang_emb
        self.use_pc_encoder = use_pc_encoder
        self.depth = depth
        self.use_reg_head = use_reg_head

        # self.match = nn.Sequential(
        #     nn.Conv1d(hidden_size, hidden_size, 1),
        #     nn.BatchNorm1d(hidden_size),
        #     nn.PReLU(),
        #     nn.Conv1d(hidden_size, hidden_size, 1),
        #     nn.BatchNorm1d(hidden_size),
        #     nn.PReLU(),
        #     nn.Conv1d(hidden_size, 1, 1)
        # )
        self.match = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            # nn.BatchNorm1d(hidden_size),
            nn.GELU(),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(hidden_size, hidden_size),
            # nn.BatchNorm1d(hidden_size),
            nn.GELU(),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(hidden_size, 1)
        )
        if self.use_reg_head:
            self.reg_head = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.GELU(),
                nn.Linear(hidden_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.GELU(),
                nn.Linear(hidden_size, 6),
                nn.Sigmoid()
            )
        self.lang_emb_proj = nn.Sequential(
            nn.Conv1d(hidden_size, hidden_size, 1),
            nn.BatchNorm1d(hidden_size),
            nn.PReLU(),
            nn.Conv1d(hidden_size, hidden_size, 1),
            nn.BatchNorm1d(hidden_size),
            nn.PReLU(),
            nn.Conv1d(hidden_size, num_proposals, 1)
        )

        # det_channel = 128
        self.pc_mlp = nn.Sequential(
            nn.Conv1d(det_channel, hidden_size, 1),
            nn.BatchNorm1d(hidden_size),
            nn.PReLU(),
            nn.Conv1d(hidden_size, hidden_size, 1)
        )
        self.pc_fusion = MultiModalAttention(hidden_size=hidden_size)
        self.pc_cls = nn.Sequential(
            nn.Conv1d(hidden_size, hidden_size, 1),
            nn.BatchNorm1d(hidden_size),
            nn.PReLU(),
            nn.Conv1d(hidden_size, num_proposals, 1)
        )

        # ====== 语言 self-attn（原有） ======
        self.lang_self_attn = nn.ModuleList(
            MultiHeadAttention(
                d_model=hidden_size,
                d_k=hidden_size // head,
                d_v=hidden_size // head,
                h=head
            ) for _ in range(2)
        )
        self.lang_ffn_net = nn.ModuleList(
            PositionWiseFeedForward(
                d_model=hidden_size,
                d_ff=hidden_size,
                dropout=0.1
            ) for _ in range(2)
        )

        # ====== text <- proposal 的 cross-attn（LGRF 的一部分，新加） ======
        self.text_cross_attn = MultiHeadAttention(
            d_model=hidden_size,
            d_k=hidden_size // head,
            d_v=hidden_size // head,
            h=head
        )

        # ====== LGRF 中的 FiLM 风格 gating（新加） ======
        # 输入: concat([box_feat, lang_global_refined]) ∈ R^{2H} -> gate ∈ R^{H}
        self.lang_gate = nn.Linear(hidden_size * 2, hidden_size)

        # ====== LGRF 残差缩放系数 alpha（新加，初始很小） ======
        self.lgrf_alpha = nn.Parameter(torch.tensor(0.1))

        # ====== objectness bias（保留形式，但初始为 0 = 关闭） ======
        self.obj_bias = nn.Parameter(torch.tensor(0.0))

        # self.conf_proj = nn.Sequential(
        #     nn.Linear(num_proposals, num_proposals)
        # )
        # self.grounding_cross_attn = MultiHeadAttention(
        #     d_model=hidden_size, d_k=hidden_size // head, d_v=hidden_size // head, h=head)  # k, q, v
        self.grounding_cross_attn = nn.ModuleList(
            CrossAttentionDecoderLayer(hidden_size=hidden_size)for _ in range(self.depth))
        self.lang_emb_cross_attn = MultiHeadAttention(
            d_model=hidden_size, d_k=hidden_size // head, d_v=hidden_size // head, h=head)  # k, q, v
        self.loss_fn = nn.CrossEntropyLoss()
        self.box_con_proj = nn.Linear(hidden_size, hidden_size)
        self.lang_con_proj = nn.Linear(hidden_size, hidden_size)
        self.temp = nn.Parameter(torch.ones([]) * 0.07)
        self.use_match_con_loss = use_match_con_loss


    def forward(self, data_dict):
        """
        只改中间的跨模态融合和匹配部分：
        - 保留原有 cross-attn + match MLP 框架；
        - 在其之上加 LGRF（text<-box co-attn + FiLM gate + residual alpha）；
        - objectness bias 初始不生效（obj_bias=0）。
        """

        # --------- 取基础变量 ---------
        objectness_masks = data_dict['objectness_scores'].max(2)[1].float().unsqueeze(2)  # (B, N, 1)
        features = data_dict["bbox_feature"]                                              # (B, N, H)
        batch_size, num_proposal = features.shape[:2]
        len_nun_max = data_dict["input_ids"].shape[1]                                    # L
        data_dict["random"] = random.random()

        # 语言特征（兼容 mlm/lang 两种）
        if self.training and "mlm_lang_fea" in data_dict:
            lang_fea = data_dict["mlm_lang_fea"]      # (B*L, T, H)
        else:
            lang_fea = data_dict["lang_fea"]          # (B*L, T, H)
        lang_emb = data_dict["lang_emb"]              # (B*L, H)

        # --------- 语言 self-attn（原有） ---------
        for i in range(2):
            lang_fea = self.lang_self_attn[i](lang_fea, lang_fea, lang_fea)
            lang_fea = self.lang_ffn_net[i](lang_fea)

        # --------- copy proposals 为每条句子扩展 ---------
        feature0 = features.clone()  # (B, N, H)
        feature1 = feature0[:, None, :, :].repeat(1, len_nun_max, 1, 1)  # (B, L, N, H)
        feature1 = feature1.reshape(batch_size * len_nun_max, num_proposal, -1)  # (B*L, N, H)

        # 去掉 [CLS]（如果你原来就是这么做的）
        if lang_fea.size(1) > 1:
            lang_tokens = lang_fea[:, 1:]      # (B*L, T-1, H)
        else:
            lang_tokens = lang_fea             # (B*L, 1, H)

        # --------- 主干：proposal <- text CrossAttention（原有） ---------
        for i in range(self.depth):
            feature1 = self.grounding_cross_attn[i](feature1, lang_tokens, lang_tokens)  # (B*L, N, H)

        data_dict["cross_box_feature"] = feature1

        # --------- LGRF Step1: text <- proposal（反向 co-attn，新加） ---------
        # text 作为 Query，proposal 作为 Key/Value
        lang_refined = self.text_cross_attn(lang_tokens, feature1, feature1)            # (B*L, T', H)

        # --------- LGRF Step2: 句子级语义 + FiLM gating（新加，温和缩放） ---------
        lang_global = lang_refined.mean(dim=1)                                          # (B*L, H)
        lang_global_exp = lang_global.unsqueeze(1).expand(-1, num_proposal, -1)         # (B*L, N, H)

        fusion_input = torch.cat([feature1, lang_global_exp], dim=-1)                   # (B*L, N, 2H)
        gate_raw = self.lang_gate(fusion_input)                                         # (B*L, N, H)

        # gate ∈ (0.5, 1.5)，只做轻量缩放，不会把特征压成 0
        gate = torch.tanh(gate_raw)                                                     # (-1, 1)
        gate = 1.0 + 0.5 * gate                                                         # (0.5, 1.5)

        delta = feature1 * gate                                                         # (B*L, N, H)
        feature_fused = feature1 + self.lgrf_alpha * delta                              # 残差融合

        # --------- 匹配 MLP（基于融合特征） ---------
        feature1_agg = feature_fused.view(batch_size * len_nun_max * num_proposal, -1)  # (B*L*N, H)
        confidence1 = self.match(feature1_agg).squeeze(1)                               # (B*L*N,)
        confidence1 = confidence1.view(batch_size * len_nun_max, num_proposal)          # (B*L, N)

        # --------- objectness 先验（形式保留，初始 obj_bias=0 ≈ 关闭） ---------
        obj_mask = objectness_masks.squeeze(2)                                          # (B, N)
        obj_mask = obj_mask[:, None, :].expand(batch_size, len_nun_max, num_proposal)   # (B, L, N)
        obj_mask = obj_mask.reshape(batch_size * len_nun_max, num_proposal)             # (B*L, N)

        confidence1 = confidence1 + self.obj_bias * (obj_mask - 0.5)

        # --------- 语言 embedding 分支（你原来就有） ---------
        if self.use_lang_emb:
            lang_emb_feature = lang_emb.unsqueeze(-1)                                   # (B*L, H, 1)
            confidence2 = self.lang_emb_proj(lang_emb_feature).squeeze(2)              # (B*L, N)
        else:
            confidence2 = 0.0

        confidence = confidence1 + confidence2 if self.use_lang_emb else confidence1

        data_dict["cluster_ref"] = confidence

        # --------- 可选：reg_head（保持原逻辑） ---------
        if self.use_reg_head:
            # restrict the value in [-0.05, 0.05]
            box_reg = self.reg_head(feature1_agg) * 0.1 - 0.05
            box_reg = box_reg.view(batch_size, len_nun_max, num_proposal, 6)
            data_dict['pred_center_reg'] = box_reg[..., 0:3]
            data_dict['pred_size_reg'] = box_reg[..., 3:6]

        return data_dict
