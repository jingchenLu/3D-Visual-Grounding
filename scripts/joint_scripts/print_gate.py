import torch
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt",
        type=str,
        required=True,
        help="path to model.pth / model_last.pth / checkpoint.tar"
    )
    args = parser.parse_args()

    # 读取文件
    ckpt = torch.load(args.ckpt, map_location="cpu")

    # 既兼容 checkpoint.tar，也兼容 model.pth
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
    else:
        state = ckpt  # 直接就是 state_dict

    # 找 gate 参数
    gate_keys = [k for k in state.keys()
                 if "patch_fusion" in k and "gate_logit" in k]

    if not gate_keys:
        print("没有找到 patch_fusion 的 gate_logit 参数（可能模块没加进这个模型？）")
        print("可用参数键示例：")
        for i, k in enumerate(list(state.keys())[:30]):
            print("  ", k)
        return

    print("\n找到的 gate 参数键：")
    for k in gate_keys:
        val = state[k]          # 这是一个标量 tensor
        if not torch.is_tensor(val):
            print(f"  {k}: 不是 tensor，值 = {val}")
            continue

        raw = val.item() if val.numel() == 1 else val
        sig = torch.sigmoid(val).item() if val.numel() == 1 else None

        print(f"  {k}: raw = {raw}")
        if sig is not None:
            print(f"        sigmoid(raw) = {sig}")

    print()

if __name__ == "__main__":
    main()
