import torch

ckpt_path = "/home/ljc/work/3DVLP/outputs/exp_joint/2025-11-27_14-54-41-12realation/checkpoint.tar"
ckpt = torch.load(ckpt_path, map_location="cpu")
print("checkpoint epoch =", ckpt["epoch"])
