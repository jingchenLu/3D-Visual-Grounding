import torch
import torch.nn as nn
import torch.nn.functional as F
from models.transformer.attention import MultiHeadAttention
from models.transformer.mmattention import MultiModalAttention, CrossAttentionDecoderLayer
from models.transformer.utils import PositionWiseFeedForward
from models.refnet.lang_guide_relation import LangCondGeomBiasSelfAttn
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

        # ===== new args =====
        use_lang_rel_module: bool = True
        rel_k: int = 128
        rel_dropout: float = 0.3
        rel_warmup_epochs: int = 10
        rel_alpha_init: float = 0.05

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

        self.use_lang_rel_module = True  # 开关，做 ablation 用

        if self.use_lang_rel_module:
            self.lang_geom_attn = LangCondGeomBiasSelfAttn(
                hidden_size=hidden_size,
                num_heads=head,
                geom_dim=12,
                k_rel=rel_k,
                dropout=rel_dropout,
                warmup_epochs=rel_warmup_epochs,
                alpha_init=rel_alpha_init,
            )

    def forward(self, data_dict):
        """
        Args:
            xyz: (B,K,3)
            features: (B,C,K)
        Returns:
            scores: (B,num_proposal,2+3+NH*2+NS*4)
        """

        objectness_masks = data_dict['objectness_scores'].max(
            2)[1].float().unsqueeze(2)  # batch_size, num_proposals, 1
        # batch_size, num_proposals, feat_size
        features = data_dict["bbox_feature"]

        batch_size, num_proposal = features.shape[:2]
        len_nun_max = data_dict["input_ids"].shape[1]  # 最多文本描述的条数
        # objectness_masks = objectness_masks.permute(0, 2, 1).contiguous()  # batch_size, 1, num_proposals
        # 生成随机数用于数据增强决策
        data_dict["random"] = random.random()

        # copy paste
        feature0 = features.clone()
        # 仅在训练模式且随机数小于0.5时执行数据增强（将复制的物体特征中选择一部分，替换这些非物体proposal的特征）
        # 模型看到的"非物体"区域实际上包含了真实物体的特征，迫使模型学习更准确地识别物体边界和特征。
        
        if data_dict["istrain"][0] == 1 and data_dict["random"] < 0.5:
            # 将objectness mask转换为布尔类型，表示哪些proposal是物体
            obj_masks = objectness_masks.bool().squeeze(2)  # batch_size, num_proposals
            # 初始化张量记录每个样本中的物体数量
            obj_lens = torch.zeros(batch_size, dtype=torch.int).cuda()
            for i in range(batch_size):
                obj_mask = torch.where(obj_masks[i, :] == True)[0]
                obj_len = obj_mask.shape[0]
                obj_lens[i] = obj_len

            obj_masks_reshape = obj_masks.reshape(batch_size*num_proposal)
            obj_features = features.reshape(batch_size*num_proposal, -1)
            obj_mask = torch.where(obj_masks_reshape[:] == True)[0]
            total_len = obj_mask.shape[0]
            obj_features = obj_features[obj_mask, :].repeat(
                2, 1)  # total_len, hidden_size
            j = 0
            for i in range(batch_size):
                obj_mask = torch.where(obj_masks[i, :] == False)[0]
                obj_len = obj_mask.shape[0]
                j += obj_lens[i]
                if obj_len < total_len - obj_lens[i]:
                    feature0[i, obj_mask, :] = obj_features[j:j + obj_len, :]
                else:
                    feature0[i, obj_mask[:total_len - obj_lens[i]],
                             :] = obj_features[j:j + total_len - obj_lens[i], :]
        # 将视觉特征扩展维度并与语言描述数量对齐
        feature1 = feature0[:, None, :, :].repeat(1, len_nun_max, 1, 1).reshape(
            batch_size*len_nun_max, num_proposal, -1)
        # if self.training:
        #     lang_fea = data_dict["mlm_lang_fea"]
        # else:
        # 获取语言特征并去除第一个token（通常是[CLS]）
        lang_fea = data_dict["lang_fea"]
        lang_fea = lang_fea[:,1:]

        # ===== LC-GeomBias Self-Attn BEFORE cross-attn =====
        if self.use_lang_rel_module:
            corners = data_dict["pred_bbox_corner"]  # (B, N, 8, 3)
            # centers from corners
            coord_min = torch.min(corners, dim=2)[0]
            coord_max = torch.max(corners, dim=2)[0]
            centers = (coord_min + coord_max) / 2.0  # (B, N, 3)

            # repeat to (BL, ...)
            centers = centers[:, None, :, :].repeat(1, len_nun_max, 1, 1).reshape(
                batch_size * len_nun_max, num_proposal, 3
            )
            corners_rep = corners[:, None, :, :, :].repeat(1, len_nun_max, 1, 1, 1).reshape(
                batch_size * len_nun_max, num_proposal, 8, 3
            )

            obj_mask = objectness_masks.squeeze(2).bool()  # (B, N)
            obj_mask = obj_mask[:, None, :].repeat(1, len_nun_max, 1).reshape(
                batch_size * len_nun_max, num_proposal
            )  # (BL, N)

            epoch = data_dict.get("epoch", None)
            feature1, rel_attn, rel_bias = self.lang_geom_attn(
                feature1, lang_fea, centers, corners_rep,
                obj_mask=obj_mask, epoch=epoch, return_attn=True
            )

            # store for visualization
            data_dict["rel_attn"] = rel_attn       # (BL, H, N, N)
            data_dict["rel_bias"] = rel_bias       # (BL, H, N, N)
            data_dict["rel_centers"] = centers     # (BL, N, 3)
        else:
            # corners not provided -> skip safely
            data_dict["rel_attn"] = None
            data_dict["rel_bias"] = None
            data_dict["rel_centers"] = None

        # ---------------------------------------------------

        # cross-attention
        # 应用多层交叉注意力机制，将语言特征作为key和value，视觉特征作为query
        for i in range(self.depth):
            feature1 = self.grounding_cross_attn[i](
                feature1, lang_fea, lang_fea)  # (B*lang_num_max, 256, hidden)
        data_dict["cross_box_feature"] = feature1
        
        # match
        feature1_agg = feature1
        feature1_agg = feature1_agg.view(
            batch_size*len_nun_max*num_proposal, -1)
        
        # 通过match网络计算基础匹配分数
        confidence1 = self.match(feature1_agg).squeeze(1)
        # 将匹配分数重塑为 (batch_size*len_nun_max, num_proposal) 形状
        confidence1 = confidence1.view(batch_size*len_nun_max, num_proposal)

        # match by lang_emb
        if self.use_lang_emb:
            lang_emb = data_dict["lang_emb"]
            lang_num_max = lang_emb.shape[0]//batch_size
            lang_emb = lang_emb.view(batch_size, lang_num_max, -1)
            # 应用额外的语言嵌入交叉注意力
            lang_emb_feature = self.lang_emb_cross_attn(
                lang_emb, feature0, feature0)
            lang_emb_feature = lang_emb_feature.view(
                batch_size*lang_num_max, -1, 1).contiguous()
            # 通过lang_emb_proj网络计算补充匹配分数
            confidence2 = self.lang_emb_proj(lang_emb_feature).squeeze(2)

        # (batch_size*lang_num_max, num_proposal)
        confidence = confidence1+confidence2 if self.use_lang_emb else confidence1

        data_dict["cluster_ref"] = confidence

        if self.use_reg_head:
            # restrict the value in [-0.05, 0.05]
            box_reg = self.reg_head(feature1_agg)*0.1-0.05
            box_reg = box_reg.view(batch_size, len_nun_max, num_proposal, 6)
            data_dict['pred_center_reg'] = box_reg[..., 0:3]
            data_dict['pred_size_reg'] = box_reg[..., 3:6]

        return data_dict
