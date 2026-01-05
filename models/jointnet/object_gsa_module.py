import torch
import torch.nn as nn

class ObjectnessAwareGSA(nn.Module):
    """
    [New Scheme] Objectness-Aware Global Aggregation (OGA)
    
    Logic:
    1. Instead of querying noisy scene points, we aggregate the proposals themselves.
    2. Crucially, we weight them by their 'objectness_scores'.
    3. This creates a 'Global Object Context' that ignores background proposals.
    """
    def __init__(self, hidden_size=128):
        super().__init__()
        
        # 用于处理聚合后的全局特征
        self.global_mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, hidden_size)
        )
        
        # 显式的 Gate 机制，控制全局特征注入的强度
        # 相比单纯的加法，这能进一步保护原始特征
        self.context_gate = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Sigmoid()
        )

    def forward(self, data_dict):
        # features: (B, N, C) - 来自 Relation Module 的 Proposal 特征
        features = data_dict["bbox_feature"]
        
        # objectness_scores: (B, N, 2) - 来自检测头
        # 假设 index 1 是 "is_object" 的分数
        obj_logits = data_dict["objectness_scores"] 
        
        # 1. 计算权重：使用 Softmax 归一化物体分数
        # 这样高置信度的物体（椅子、桌子）权重高，背景（墙、空地）权重接近 0
        obj_scores = obj_logits[:, :, 1] # (B, N)
        # (B, N, 1)
        weights = torch.softmax(obj_scores, dim=1).unsqueeze(2)
        
        # 2. 加权聚合 (Weighted Aggregation)
        # Global Feature: (B, 1, C)
        global_feat = torch.sum(features * weights, dim=1, keepdim=True)
        
        # 3. 特征变换
        global_feat = self.global_mlp(global_feat)
        
        # 4. 广播回所有 Proposal (B, N, C)
        global_feat_expanded = global_feat.expand_as(features)
        
        # 存入字典供后续融合使用
        data_dict["gsa_feature"] = global_feat_expanded
        return data_dict