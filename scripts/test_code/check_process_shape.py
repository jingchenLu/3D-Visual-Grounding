import os
import sys
import numpy as np
# ✅ 把项目根目录加入 PYTHONPATH
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from lib.configs.config import CONF

scene_id = "scene0000_00"   # 改成你要看的
split = "train"             # "train" 或 "val"

path = os.path.join(CONF.PATH.SCANNET_DATA, scene_id) + f"_preprocess_{split}.npy"
pc = np.load(path)

print("file:", path)
print("pc.shape =", pc.shape)          # (N, C)
print("N =", pc.shape[0], "C =", pc.shape[1])   # C 就是特征维度

# 看第 idx 个点（比如第 0 个）
idx = 0
print(f"point[{idx}] dim =", pc[idx].shape)     # (C,)
print(f"point[{idx}] first 10 vals =", pc[idx][:10])

pcl_color_path = os.path.join(CONF.PATH.SCANNET_DATA, scene_id) + f"_pcl_color_{split}.npy"
rgb = np.load(pcl_color_path)
print("rgb.shape =", rgb.shape, "rgb[0] =", rgb[0])

# # 可选：看看每一维大致范围，帮助判断哪段是 normal / multiview
# print("per-dim min (first 10 dims):", pc.min(axis=0)[:10])
# print("per-dim max (first 10 dims):", pc.max(axis=0)[:10])
