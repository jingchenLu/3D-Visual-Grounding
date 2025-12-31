import torch
import os

# 修改为你的 checkpoint 实际路径
CHECKPOINT_DIR = "/home/ljc/work/3DVLP/outputs/exp_joint/2025-12-27_22-05-44"
CKPT_NAME = "model_last.pth" 

def inspect_checkpoint(ckpt_path):
    if not os.path.exists(ckpt_path):
        print(f"Error: Checkpoint not found at {ckpt_path}")
        return

    print(f"Loading checkpoint from: {ckpt_path}")
    try:
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint

        print("Checkpoint loaded successfully.\n")
        print("-" * 40)
        print("INSPECTION RESULTS:")
        print("-" * 40)

        # 查找 alpha
        alpha_key = None
        for k in state_dict.keys():
            if "rel_global_fusion.alpha" in k:
                alpha_key = k
                break

        if alpha_key:
            alpha_tensor = state_dict[alpha_key]
            # 计算统计量
            alpha_mean = alpha_tensor.mean().item()
            alpha_max = alpha_tensor.max().item()
            alpha_min = alpha_tensor.min().item()
            
            print(f"\n[Alpha Parameter] (Channel-wise Fusion Weights):")
            print(f"  Key: {alpha_key}")
            print(f"  Shape: {list(alpha_tensor.shape)}")
            print(f"  Mean Value: {alpha_mean:.6f}") # 关注这个！
            print(f"  Max  Value: {alpha_max:.6f}")
            print(f"  Min  Value: {alpha_min:.6f}")
            
            # 智能诊断
            if abs(alpha_mean) < 1e-4:
                print("\n>>> DIAGNOSIS: Alpha is effectively ZERO. The GSA module was ASLEEP. 😴")
                print("    Action: You MUST change init_alpha to 0.1 or 0.2 to wake it up.")
            elif abs(alpha_mean) < 0.05:
                print("\n>>> DIAGNOSIS: Alpha is very small. GSA contribution was minimal. ⚠️")
                print("    Action: Increase learning rate for GSA branch.")
            else:
                print("\n>>> DIAGNOSIS: Alpha is active! GSA is contributing. ✅")
                if alpha_mean > 0.5:
                     print("    Note: GSA is dominating the features (Weight > 0.5).")
        else:
            print("\n[Alpha Parameter]: Not found! Did you save the correct model?")

    except Exception as e:
        print(f"Error inspecting checkpoint: {e}")

if __name__ == "__main__":
    full_path = os.path.join(CHECKPOINT_DIR, CKPT_NAME)
    inspect_checkpoint(full_path)