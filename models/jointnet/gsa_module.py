import torch
import torch.nn as nn

from models.transformer.attention import MultiHeadAttention
from models.transformer.utils import PositionWiseFeedForward


class GlobalBiasGSA(nn.Module):
    """
    GSA (Global Scene Aggregation) for two-stage pipeline.

    Q:  data_dict["bbox_feature"]      (B, K, C)   # proposal features (after RelationModule or parallel branch)
    K/V: seed/vote scene tokens        (B, N, C_in)

    Scheme-2 (recommended): add explicit xyz positional embedding to scene_feat:
        scene_feat <- scene_feat + xyz_scale * xyz_mlp(norm_xyz)

    NOTE about legacy "global bias":
      - We KEEP parameters (global_mlp/global_scale) for checkpoint compatibility.
      - But by default we DO NOT USE them, because the current bias is constant along key dimension
        and will be cancelled by softmax (no effect).
      - If you ever want a real bias, it must vary across keys (pair-wise with scene_xyz).
    """

    def __init__(
        self,
        d_model=128,
        nhead=4,
        depth=1,
        scene_in_dim=256,
        d_ff=256,
        dropout=0.1,
        identity_map_reordering=False,
        top_scene_k=None,

        # [NEW] scheme2: xyz positional encoding for scene tokens
        use_xyz_pos=True,
        xyz_hidden=128,           # hidden size inside xyz_mlp
        xyz_ln=True,              # add LayerNorm at the end of xyz_mlp output (recommended)
        xyz_scale_init=0.1,       # start from 0 for stable finetune at epoch200; can set 0.1 if you want stronger effect

        # [KEEP for ckpt] legacy bias switch (default False)
        enable_global_bias=False,
    ):
        super().__init__()
        assert d_model % nhead == 0, "d_model must be divisible by nhead"
        self.d_model = d_model
        self.nhead = nhead
        self.depth = depth
        self.top_scene_k = top_scene_k

        self.use_xyz_pos = use_xyz_pos
        self.enable_global_bias = enable_global_bias

        # scene token channel -> d_model
        self.scene_proj = nn.Identity() if scene_in_dim == d_model else nn.Linear(scene_in_dim, d_model)

        dk = d_model // nhead
        dv = d_model // nhead

        self.cross_attn = nn.ModuleList([
            MultiHeadAttention(
                d_model=d_model, d_k=dk, d_v=dv, h=nhead,
                dropout=dropout,
                identity_map_reordering=identity_map_reordering
            )
            for _ in range(depth)
        ])

        self.ffn = nn.ModuleList([
            PositionWiseFeedForward(
                d_model=d_model, d_ff=d_ff,
                dropout=dropout,
                identity_map_reordering=identity_map_reordering
            )
            for _ in range(depth)
        ])

        # -------------------------
        # [NEW] xyz positional embedding for scene tokens
        # -------------------------
        if self.use_xyz_pos:
            mlp = [
                nn.Linear(3, xyz_hidden),
                nn.ReLU(inplace=True),
                nn.Linear(xyz_hidden, d_model),
            ]
            if xyz_ln:
                mlp.append(nn.LayerNorm(d_model))
            self.xyz_mlp = nn.Sequential(*mlp)

            # learnable strength for xyz PE (stable finetune)
            self.xyz_scale = nn.Parameter(torch.tensor(float(xyz_scale_init)))
        else:
            self.xyz_mlp = None
            self.xyz_scale = None

        # -------------------------
        # [KEEP] legacy "global bias" branch for ckpt compatibility
        # (by default not used)
        # -------------------------
        self.global_mlp = nn.ModuleList([
            nn.Sequential(
                nn.Linear(3, 32),
                nn.ReLU(inplace=True),
                nn.LayerNorm(32),
                nn.Linear(32, nhead),
            )
            for _ in range(depth)
        ])

        # learnable strength
        self.global_scale = nn.Parameter(torch.zeros(depth, nhead, 1, 1))

    @staticmethod
    def _get_bbox_centers(corners: torch.Tensor) -> torch.Tensor:
        # corners: (B, K, 8, 3) -> (B, K, 3)
        coord_min = torch.min(corners, dim=2)[0]
        coord_max = torch.max(corners, dim=2)[0]
        return (coord_min + coord_max) / 2.0

    @staticmethod
    def _get_scene_tokens(data_dict):
        """
        Prefer seed_xyz/seed_features; fallback to vote_xyz/vote_features.
        Return:
          scene_xyz:  (B, N, 3)
          scene_feat: (B, N, C_in)
        """
        if ("seed_xyz" in data_dict) and ("seed_features" in data_dict):
            scene_xyz = data_dict["seed_xyz"]  # (B, N, 3)
            sf = data_dict["seed_features"]    # (B, C, N)
            scene_feat = sf.permute(0, 2, 1).contiguous()  # (B, N, C)
            return scene_xyz, scene_feat

        scene_xyz = data_dict["vote_xyz"]
        vf = data_dict["vote_features"]
        scene_feat = vf.permute(0, 2, 1).contiguous()
        return scene_xyz, scene_feat

    @staticmethod
    def _normalize_scene_xyz(scene_xyz: torch.Tensor) -> torch.Tensor:
        """
        Normalize xyz to [-1, 1] using scene bounds (per batch).
        scene_xyz: (B, N, 3)
        return:   (B, N, 3)
        """
        scene_min = scene_xyz.min(dim=1, keepdim=True)[0]               # (B,1,3)
        scene_max = scene_xyz.max(dim=1, keepdim=True)[0]               # (B,1,3)
        denom = (scene_max - scene_min).clamp(min=1e-5)                 # (B,1,3)
        xyz_norm = (scene_xyz - scene_min) / denom                      # [0,1]
        xyz_norm = xyz_norm.clamp(0.0, 1.0) * 2.0 - 1.0                 # [-1,1]
        return xyz_norm

    def _build_global_bias(self, layer_id, centers, scene_xyz):
        """
        Legacy bias (kept for ckpt). WARNING: if expanded to all keys identically,
        it can be cancelled by softmax. Default enable_global_bias=False.
        """
        B, K, _ = centers.shape
        N = scene_xyz.shape[1]

        scene_min = scene_xyz.min(dim=1, keepdim=True)[0]           # (B,1,3)
        scene_max = scene_xyz.max(dim=1, keepdim=True)[0]           # (B,1,3)
        denom = (scene_max - scene_min).clamp(min=1e-5)             # (B,1,3)

        c_norm = (centers - scene_min) / denom                      # (B,K,3) ~ [0,1]
        c_norm = c_norm.clamp(0.0, 1.0) * 2.0 - 1.0                 # -> [-1,1]

        # (B, K, H) -> (B, H, K, 1) -> expand to N
        per_head = self.global_mlp[layer_id](c_norm).permute(0, 2, 1).unsqueeze(-1)  # (B,H,K,1)
        per_head = per_head.expand(B, self.nhead, K, N)                               # (B,H,K,N)

        bias = self.global_scale[layer_id] * per_head
        return bias

    def forward(self, data_dict: dict):
        x = data_dict["bbox_feature"]  # (B, K, C)

        scene_xyz, scene_feat = self._get_scene_tokens(data_dict)  # (B,N,3), (B,N,C_in)

        # (optional) truncate scene tokens
        if self.top_scene_k is not None and scene_xyz.shape[1] > self.top_scene_k:
            # 仍保留你之前的 slice 行为（你已验证 random topk 用处不大）
            scene_xyz = scene_xyz[:, : self.top_scene_k, :]
            scene_feat = scene_feat[:, : self.top_scene_k, :]

        # project to d_model
        scene_feat = self.scene_proj(scene_feat)  # (B, N, C)

        # [NEW] add xyz positional embedding
        if self.use_xyz_pos:
            xyz_norm = self._normalize_scene_xyz(scene_xyz)                 # (B,N,3)
            xyz_emb = self.xyz_mlp(xyz_norm)                                # (B,N,C)
            scene_feat = scene_feat + self.xyz_scale * xyz_emb

        centers = self._get_bbox_centers(data_dict["pred_bbox_corner"])  # (B, K, 3)

        for i in range(self.depth):
            if self.enable_global_bias:
                bias = self._build_global_bias(i, centers, scene_xyz)  # (B,H,K,N)
                x = self.cross_attn[i](
                    x, scene_feat, scene_feat,
                    attention_weights=bias,
                    way="add"
                )
            else:
                # default: no bias
                x = self.cross_attn[i](x, scene_feat, scene_feat)

            x = self.ffn[i](x)

        return x
