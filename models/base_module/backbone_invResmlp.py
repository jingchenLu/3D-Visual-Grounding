import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import os

from lib.pointnet2.pointnet2_modules import PointnetSAModuleVotes, PointnetFPModule

# 引入 InvResMLP 模块
sys.path.append('/home/ljc/work/3DVLP/lib/pointnet2')
from InvResMLP import InvResMLP


class Pointnet2Backbone(nn.Module):
    """
    PointNet++ backbone + InvResMLP增强模块
    """

    def __init__(self, input_feature_dim=0):
        super().__init__()
        self.input_feature_dim = input_feature_dim

        # --------- 4 SET ABSTRACTION LAYERS ---------
        self.sa1 = PointnetSAModuleVotes(
            npoint=2048,
            radius=0.2,
            nsample=64,
            mlp=[input_feature_dim, 64, 64, 128],
            use_xyz=True,
            normalize_xyz=True
        )

        self.sa2 = PointnetSAModuleVotes(
            npoint=1024,
            radius=0.4,
            nsample=32,
            mlp=[128, 128, 128, 256],
            use_xyz=True,
            normalize_xyz=True
        )

        self.sa3 = PointnetSAModuleVotes(
            npoint=512,
            radius=0.8,
            nsample=16,
            mlp=[256, 128, 128, 256],
            use_xyz=True,
            normalize_xyz=True
        )

        self.sa4 = PointnetSAModuleVotes(
            npoint=256,
            radius=1.2,
            nsample=16,
            mlp=[256, 128, 128, 256],
            use_xyz=True,
            normalize_xyz=True
        )

        # --------- InvResMLP 模块（PointNeXt风格）---------
        self.inv1 = InvResMLP(radius=0.2, nsample=32, in_channel=128, coor_dim=3, expansion=4)
        self.inv2 = InvResMLP(radius=0.4, nsample=32, in_channel=256, coor_dim=3, expansion=4)
        self.inv3 = InvResMLP(radius=0.8, nsample=16, in_channel=256, coor_dim=3, expansion=4)
        self.inv4 = InvResMLP(radius=1.2, nsample=16, in_channel=256, coor_dim=3, expansion=4)

        # --------- 2 FEATURE UPSAMPLING LAYERS ---------
        self.fp1 = PointnetFPModule(mlp=[256+256, 256, 256])
        self.fp2 = PointnetFPModule(mlp=[256+256, 256, 256])

    def _break_up_pc(self, pc):
        xyz = pc[..., :3].contiguous()
        features = pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None
        return xyz, features

    def forward(self, data_dict):
        pointcloud = data_dict["point_clouds"]
        xyz, features = self._break_up_pc(pointcloud)

        # --------- SA1 + InvResMLP ---------
        xyz, features, fps_inds = self.sa1(xyz, features)
        data_dict['sa1_inds'] = fps_inds
        data_dict['sa1_xyz'] = xyz
        data_dict['sa1_features'] = features
        xyz_t = xyz.transpose(1, 2)
        xyz_t, features = self.inv1([xyz_t, features])
        xyz = xyz_t.transpose(1, 2)
        data_dict['sa1_features'] = features

        # --------- SA2 + InvResMLP ---------
        xyz, features, fps_inds = self.sa2(xyz, features)
        data_dict['sa2_inds'] = fps_inds
        data_dict['sa2_xyz'] = xyz
        data_dict['sa2_features'] = features
        xyz_t = xyz.transpose(1, 2)
        xyz_t, features = self.inv2([xyz_t, features])
        xyz = xyz_t.transpose(1, 2)
        data_dict['sa2_features'] = features

        # --------- SA3 + InvResMLP ---------
        xyz, features, fps_inds = self.sa3(xyz, features)
        data_dict['sa3_xyz'] = xyz
        data_dict['sa3_features'] = features
        xyz_t = xyz.transpose(1, 2)
        xyz_t, features = self.inv3([xyz_t, features])
        xyz = xyz_t.transpose(1, 2)
        data_dict['sa3_features'] = features

        # --------- SA4 + InvResMLP ---------
        xyz, features, fps_inds = self.sa4(xyz, features)
        data_dict['sa4_xyz'] = xyz
        data_dict['sa4_features'] = features
        xyz_t = xyz.transpose(1, 2)
        xyz_t, features = self.inv4([xyz_t, features])
        xyz = xyz_t.transpose(1, 2)
        data_dict['sa4_features'] = features

        # --------- Feature Propagation ---------
        features = self.fp1(data_dict['sa3_xyz'], data_dict['sa4_xyz'],
                            data_dict['sa3_features'], data_dict['sa4_features'])
        features = self.fp2(data_dict['sa2_xyz'], data_dict['sa3_xyz'],
                            data_dict['sa2_features'], features)
        data_dict['fp2_features'] = features
        data_dict['fp2_xyz'] = data_dict['sa2_xyz']
        num_seed = data_dict['fp2_xyz'].shape[1]
        data_dict['fp2_inds'] = data_dict['sa1_inds'][:, 0:num_seed]

        return data_dict


if __name__ == '__main__':
    from types import SimpleNamespace
    import torch
    data_dict = {"point_clouds": torch.rand(2, 20000, 6).cuda()}
    model = Pointnet2Backbone(input_feature_dim=3).cuda()
    out = model(data_dict)
    for k, v in out.items():
        if isinstance(v, torch.Tensor):
            print(k, v.shape)
