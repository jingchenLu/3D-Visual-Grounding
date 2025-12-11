import torch
import torch.nn as nn
import torch.nn.functional as F

class AdaptivePointFeatureAugmentation(nn.Module):
    def __init__(self, input_dim, output_dim, num_heads=8, context_dim=128):
        """
        :param input_dim: 输入点特征维度 (如256)
        :param output_dim: 输出特征维度 (通常与input_dim相同)
        :param num_heads: 注意力头数
        :param context_dim: 语言上下文维度 (如128)
        """
        super(AdaptivePointFeatureAugmentation, self).__init__()
        
        # 计算拼接后的总维度
        self.combined_dim = input_dim + 3 + input_dim  # features + positions + context_embeddings
        
        # 1. 将拼接后的特征投影回原始维度 (515 -> 256)
        self.projection = nn.Linear(self.combined_dim, input_dim)
        
        # 2. 注意力层 (现在处理正确维度)
        self.attention = nn.MultiheadAttention(
            embed_dim=input_dim,  # 现在使用原始维度
            num_heads=num_heads,
            batch_first=True  # 关键：指定batch在第一个维度
        )
        
        # 3. 特征精炼网络
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.GELU(),
            nn.Linear(128, output_dim)
        )
        
        # 4. 上下文投影 (128 -> 256)
        self.context_proj = nn.Linear(context_dim, input_dim)
        
        # 5. 层归一化
        self.norm = nn.LayerNorm(output_dim)
        
        # 6. 残差连接 (确保输入输出维度匹配)
        if input_dim != output_dim:
            self.residual_proj = nn.Linear(input_dim, output_dim)
        else:
            self.residual_proj = nn.Identity()

    def forward(self, features, positions, context):
        """
        Apply attention and context-based enhancement to the point features.
        
        :param features: (B, N, D) Point features.
        :param positions: (B, N, 3) Point positions (x, y, z).
        :param context: (B, C) Context information for the scene.
        
        :return: Enhanced features with context-aware attention.
        """
        batch_size, num_points, feature_dim = features.shape
        
        # 1. 投影上下文到特征空间 (B, C) -> (B, D)
        context_embeddings = self.context_proj(context)  # (B, D)
        
        # 2. 扩展上下文以匹配点数 (B, D) -> (B, N, D)
        context_embeddings = context_embeddings.unsqueeze(1).expand(-1, num_points, -1)
        
        # 3. 拼接所有特征 (B, N, D + 3 + D)
        combined_features = torch.cat([features, positions, context_embeddings], dim=-1)
        
        # 4. 投影回原始维度 (B, N, 515) -> (B, N, D)
        projected_features = self.projection(combined_features)
        
        # 5. 注意力机制 (使用batch_first=True)
        attn_output, _ = self.attention(
            projected_features,  # query
            projected_features,  # key
            projected_features,  # value
            need_weights=False
        )
        
        # 6. 残差连接 + MLP精炼
        residual = self.residual_proj(features)
        mlp_output = self.mlp(attn_output)
        
        # 7. 归一化并返回
        enhanced_features = self.norm(mlp_output + residual)
        
        return enhanced_features