# run_prune_backbone_only.py
import os
import sys

# ----------------- 路径与环境修复 -----------------
# 1. 获取脚本所在目录的绝对路径 (.../3DVLP/scripts/pure)
current_script_dir = os.path.dirname(os.path.abspath(__file__))

# 2. 推导项目根目录 (.../3DVLP)
#    逻辑：从 scripts/pure 向上两级
project_root = os.path.abspath(os.path.join(current_script_dir, "../../"))

# 3. 【关键】将当前工作目录强制切换到项目根目录
#    这解决了 FileNotFoundError: 'lib/configs/config_bert.json'
os.chdir(project_root)
print(f"[Info] Working directory changed to: {os.getcwd()}")

# 4. 将根目录加入 Python 搜索路径
#    这解决了 ModuleNotFoundError: No module named 'models'
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import os, json
import torch
import torch.nn as nn

from models.jointnet.jointnet import JointNet
from data.scannet.model_util_scannet import ScannetDatasetConfig
from models.base_module.backbone_module import PAModuleVotes

from lib.pruning.eval_importance import evaluate_pam_hidden_keepidx

DC = ScannetDatasetConfig()

import os
import numpy as np
import matplotlib.pyplot as plt

def _ensure_outdir(outdir):
    os.makedirs(outdir, exist_ok=True)

def _extract_conv_records(report):
    """
    兼容三种 report：
      1) report["conv_bn"][layer] ...
      2) report["layers"][layer] ...
      3) report["pam_hidden"][pam] ...  (只剪 hidden_dim 的版本)
    """
    if "conv_bn" in report:
        d = report["conv_bn"]
        recs = []
        for name, info in d.items():
            C_out = info.get("C_out", None)
            keep = info.get("keep", None)
            if C_out is None or keep is None:
                continue
            keep_ratio = info.get("keep_ratio", float(keep) / float(C_out))
            recs.append({
                "name": name,
                "C_out": int(C_out),
                "keep": int(keep),
                "keep_ratio": float(keep_ratio),
                "score_mean": info.get("score_mean", None),
                "block": info.get("block", None),
            })
        return recs

    if "layers" in report:
        d = report["layers"]
        recs = []
        for name, info in d.items():
            C_out = info.get("C_out", None)
            keep = info.get("keep", None)
            if C_out is None or keep is None:
                continue
            keep_ratio = info.get("keep_ratio", float(keep) / float(C_out))
            recs.append({
                "name": name,
                "C_out": int(C_out),
                "keep": int(keep),
                "keep_ratio": float(keep_ratio),
                "score_mean": info.get("score_mean", None),
                "block": info.get("block", None),
            })
        return recs

    # pam_hidden：把每个 PAM 当作一个“层”记录（C_out=hidden_dim）
    if "pam_hidden" in report:
        recs = []
        for name, info in report["pam_hidden"].items():
            hidden = info.get("hidden_dim", None)
            keep = info.get("keep", None)
            if hidden is None or keep is None:
                continue
            recs.append({
                "name": name,
                "C_out": int(hidden),
                "keep": int(keep),
                "keep_ratio": float(keep) / float(hidden),
                "score_mean": None,
                "block": name.split(".")[0] if "." in name else None,
            })
        return recs

    return []


def _extract_scores_for_hist(report, model=None):
    """
    返回 (proxy, full)
    proxy: 每层一个分数（用于快速画图）
    full: 直接从模型抓 backbone BN gamma 的全量分布（更真实）
    """
    # A) proxy：优先用 conv 记录的 score_mean；如果没有就用 pam_hidden 的 keep_ratio/hidden_dim proxy
    proxy = None

    conv_recs = _extract_conv_records(report)
    proxy_vals = [r["score_mean"] for r in conv_recs if r.get("score_mean", None) is not None]
    if len(proxy_vals) > 0:
        proxy = np.asarray(proxy_vals, dtype=np.float32)
    else:
        # 兼容 pam_hidden 结构：用 “keep_ratio” 或 “hidden_dim” 构造一个 proxy（至少不为空）
        if "pam_hidden" in report:
            vals = []
            for _, v in report["pam_hidden"].items():
                # 优先 keep_ratio，其次 hidden_dim
                if "keep_ratio" in v:
                    vals.append(float(v["keep_ratio"]))
                elif "hidden_dim" in v:
                    vals.append(float(v["hidden_dim"]))
            if len(vals) > 0:
                proxy = np.asarray(vals, dtype=np.float32)

    # B) full：从模型抓 backbone BN gamma 全量
    full = None
    if model is not None:
        import torch.nn as nn
        gammas = []
        for n, m in model.named_modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                if getattr(m, "weight", None) is not None:
                    if n.startswith("backbone_net."):
                        gammas.append(m.weight.detach().abs().float().cpu().numpy())
        if len(gammas) > 0:
            full = np.concatenate(gammas, axis=0)

    return proxy, full


def plot_keep_ratio_per_layer(report, outdir="prune_viz", max_layers=120):
    _ensure_outdir(outdir)
    recs = _extract_conv_records(report)
    if not recs:
        print("[viz] No conv records found.")
        return

    # 按层排序：更稳定的展示（按名字）
    recs = sorted(recs, key=lambda x: x["name"])
    if len(recs) > max_layers:
        recs = recs[:max_layers]

    keep_ratio = [r["keep_ratio"] for r in recs]
    C_out = [r["C_out"] for r in recs]
    names = [r["name"] for r in recs]

    x = np.arange(len(recs))

    plt.figure(figsize=(22, 5))
    plt.bar(x, keep_ratio)
    plt.ylim(0.0, 1.05)
    plt.ylabel("Keep ratio")
    plt.xlabel("Layer index")
    plt.title("Per-layer keep ratio (structured pruning plan)")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "keep_ratio_per_layer.png"))
    plt.close()

    plt.figure(figsize=(22, 5))
    plt.bar(x, C_out)
    plt.ylabel("C_out")
    plt.xlabel("Layer index")
    plt.title("Per-layer original output channels (C_out)")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "channels_per_layer.png"))
    plt.close()

    # 可选：保存 layer 名单，方便论文对齐
    with open(os.path.join(outdir, "layers.txt"), "w") as f:
        for i, n in enumerate(names):
            f.write(f"{i}\t{n}\tC_out={C_out[i]}\tkeep_ratio={keep_ratio[i]:.3f}\n")

    print(f"[viz] Saved keep_ratio_per_layer.png / channels_per_layer.png / layers.txt -> {outdir}")

def plot_importance_hist_and_cdf(report, model=None, outdir="prune_viz"):
    _ensure_outdir(outdir)
    proxy, full = _extract_scores_for_hist(report, model=model)

    # 1) proxy hist（每层 mean）
    if proxy is not None and len(proxy):
        plt.figure(figsize=(8, 5))
        plt.hist(proxy, bins=40)
        plt.xlabel("Importance (proxy = per-layer mean)")
        plt.ylabel("Count")
        plt.title("Importance distribution (proxy, per-layer)")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "importance_hist_proxy.png"))
        plt.close()

        # CDF
        x = np.sort(proxy)
        y = np.arange(1, len(x) + 1) / float(len(x))
        plt.figure(figsize=(8, 5))
        plt.plot(x, y)
        plt.xlabel("Importance (proxy)")
        plt.ylabel("CDF")
        plt.title("Importance CDF (proxy, per-layer)")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "importance_cdf_proxy.png"))
        plt.close()

        print(f"[viz] Saved importance_hist_proxy.png / importance_cdf_proxy.png -> {outdir}")

    # 2) full hist（BN gamma 全量，推荐）
    if full is not None and len(full):
        plt.figure(figsize=(8, 5))
        plt.hist(full, bins=80)
        plt.xlabel("|gamma| (BN weights)")
        plt.ylabel("Count")
        plt.title("BN gamma distribution (backbone)")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "bn_gamma_hist_full.png"))
        plt.close()

        x = np.sort(full)
        y = np.arange(1, len(x) + 1) / float(len(x))
        plt.figure(figsize=(8, 5))
        plt.plot(x, y)
        plt.xlabel("|gamma| (BN weights)")
        plt.ylabel("CDF")
        plt.title("BN gamma CDF (backbone)")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "bn_gamma_cdf_full.png"))
        plt.close()

        # 额外：统计“接近 0 的比例”
        thr = [0.01, 0.05, 0.1]
        stats = {t: float((full < t).mean()) for t in thr}
        with open(os.path.join(outdir, "bn_gamma_small_ratio.txt"), "w") as f:
            for t, v in stats.items():
                f.write(f"P(|gamma|<{t}) = {v:.6f}\n")
        print(f"[viz] Saved bn_gamma_hist_full.png / bn_gamma_cdf_full.png / bn_gamma_small_ratio.txt -> {outdir}")

def plot_block_summary(report, outdir="prune_viz"):
    _ensure_outdir(outdir)
    recs = _extract_conv_records(report)
    if not recs:
        print("[viz] No conv records found for block summary.")
        return

    # 如果 report 里没有 block 字段，则用 name 前缀粗分
    def infer_block(name):
        for b in ["backbone_net.sa1", "backbone_net.sa2", "backbone_net.sa3", "backbone_net.sa4", "backbone_net.fp1", "backbone_net.fp2"]:
            if name.startswith(b):
                return b
        return "other"

    by_block = {}
    for r in recs:
        b = r["block"] if r["block"] is not None else infer_block(r["name"])
        by_block.setdefault(b, []).append(r)

    blocks = sorted(by_block.keys())
    mean_scores = []
    mean_keep = []
    for b in blocks:
        xs = [x["score_mean"] for x in by_block[b] if x["score_mean"] is not None]
        kr = [x["keep_ratio"] for x in by_block[b]]
        mean_scores.append(float(np.mean(xs)) if len(xs) else np.nan)
        mean_keep.append(float(np.mean(kr)) if len(kr) else np.nan)

    x = np.arange(len(blocks))

    plt.figure(figsize=(12, 4))
    plt.bar(x, mean_keep)
    plt.xticks(x, blocks, rotation=30, ha="right")
    plt.ylim(0.0, 1.05)
    plt.ylabel("Mean keep ratio")
    plt.title("Mean keep ratio per block")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "keep_ratio_per_block.png"))
    plt.close()

    plt.figure(figsize=(12, 4))
    plt.bar(x, mean_scores)
    plt.xticks(x, blocks, rotation=30, ha="right")
    plt.ylabel("Mean importance (score_mean)")
    plt.title("Mean importance per block (proxy)")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "importance_mean_per_block.png"))
    plt.close()

    print(f"[viz] Saved keep_ratio_per_block.png / importance_mean_per_block.png -> {outdir}")



def load_args_from_info(exp_dir: str):
    with open(os.path.join(exp_dir, "info.json"), "r") as f:
        return json.load(f)

def build_model_from_train_args(train_args: dict, device: str):
    use_multiview = bool(train_args.get("use_multiview", False))
    use_normal = bool(train_args.get("use_normal", False))
    use_color = bool(train_args.get("use_color", False))
    no_height = bool(train_args.get("no_height", False))
    input_channels = int(use_multiview) * 128 + int(use_normal) * 3 + int(use_color) * 3 + int(not no_height)

    model = JointNet(
        num_class=DC.num_class,
        vocabulary=None,
        num_heading_bin=DC.num_heading_bin,
        num_size_cluster=DC.num_size_cluster,
        mean_size_arr=DC.mean_size_arr,
        input_feature_dim=input_channels,
        num_proposal=int(train_args.get("num_proposals", 256)),
        no_caption=bool(train_args.get("no_caption", False)),
        use_topdown=bool(train_args.get("use_topdown", False)),
        num_locals=int(train_args.get("num_locals", 64)),
        query_mode=train_args.get("query_mode", "corner"),
        num_graph_steps=int(train_args.get("num_graph_steps", 0)),
        use_relation=bool(train_args.get("use_relation", False)),
        use_lang_classifier=(not bool(train_args.get("no_lang_cls", False))),
        use_bidir=bool(train_args.get("use_bidir", False)),
        no_reference=bool(train_args.get("no_reference", False)),
        dataset_config=DC,
        use_distil=bool(train_args.get("use_distil", False)),
        unfreeze=int(train_args.get("unfreeze", 0)),
        use_mlm=bool(train_args.get("use_mlm", False)),
        use_con=bool(train_args.get("use_con", False)),
        use_lang_emb=bool(train_args.get("use_lang_emb", False)),
        mask_box=bool(train_args.get("mask_box", False)),
        use_pc_encoder=bool(train_args.get("use_pc_encoder", False)),
        use_match_con_loss=bool(train_args.get("use_match_con_loss", False)),
        use_reg_head=bool(train_args.get("use_reg_head", False)),
        use_kl_loss=bool(train_args.get("use_kl_loss", False)),
        use_mlcv_net=bool(train_args.get("use_mlcv_net", False)),
        use_vote_weight=bool(train_args.get("use_vote_weight", False)),
    ).to(device)
    return model

def load_checkpoint_weights(model, exp_dir: str, device: str):
    ckpt_tar = os.path.join(exp_dir, "checkpoint.tar")
    ckpt = torch.load(ckpt_tar, map_location=device)
    sd = ckpt["model_state_dict"]
    sd = { (k[7:] if k.startswith("module.") else k): v for k, v in sd.items() }
    model.load_state_dict(sd, strict=True)

def _get_submodule(root: nn.Module, name: str) -> nn.Module:
    cur = root
    for p in name.split("."):
        cur = getattr(cur, p)
    return cur

def _set_submodule(root: nn.Module, name: str, new_m: nn.Module):
    parent_name, attr = name.rsplit(".", 1)
    parent = _get_submodule(root, parent_name)
    if isinstance(parent, nn.Sequential):
        parent[int(attr)] = new_m
    else:
        setattr(parent, attr, new_m)

@torch.no_grad()
def _slice_out(conv: nn.Conv2d, out_idx: torch.Tensor):
    return conv.weight.data.index_select(0, out_idx).contiguous()

@torch.no_grad()
def _slice_in(conv: nn.Conv2d, in_idx: torch.Tensor):
    return conv.weight.data.index_select(1, in_idx).contiguous()

@torch.no_grad()
def _slice_out_in(conv: nn.Conv2d, out_idx: torch.Tensor, in_idx: torch.Tensor):
    w = conv.weight.data.index_select(0, out_idx)
    w = w.index_select(1, in_idx)
    return w.contiguous()

@torch.no_grad()
def _copy_bn2d(dst: nn.BatchNorm2d, src: nn.BatchNorm2d):
    dst.weight.data.copy_(src.weight.data)
    dst.bias.data.copy_(src.bias.data)
    dst.running_mean.data.copy_(src.running_mean.data)
    dst.running_var.data.copy_(src.running_var.data)

@torch.no_grad()
def _slice_bn2d(dst: nn.BatchNorm2d, src: nn.BatchNorm2d, idx: torch.Tensor):
    dst.weight.data.copy_(src.weight.data.index_select(0, idx))
    dst.bias.data.copy_(src.bias.data.index_select(0, idx))
    dst.running_mean.data.copy_(src.running_mean.data.index_select(0, idx))
    dst.running_var.data.copy_(src.running_var.data.index_select(0, idx))

def rebuild_pruned_pam_hidden(pam: PAModuleVotes, keep_idx: torch.Tensor) -> PAModuleVotes:
    """
    只剪 hidden_dim，保持 out_channel/full_in 不变 => voting 绝对不受影响
    """
    device = next(pam.parameters()).device
    ki = keep_idx.to(device=device, dtype=torch.long)

    # 从原模块读取关键超参（这些名字按你 PAM 实现）
    use_xyz = bool(getattr(pam, "use_xyz", True))
    full_in = int(getattr(pam, "full_in"))
    in_channel = full_in - (3 if use_xyz else 0)
    out_channel = int(pam.out_mapper[0].out_channels)  # out_mapper Conv1d out=out_channel
    new_hidden = int(ki.numel())

    new_pam = PAModuleVotes(
        npoint=int(pam.npoint),
        radius=float(pam.radius),
        nsample=int(pam.nsample),
        in_channel=in_channel,
        out_channel=out_channel,
        hidden_dim=new_hidden,
        n_aug=int(getattr(pam, "n_aug", 0)),
        use_xyz=use_xyz,
        normalize_xyz=bool(getattr(pam, "normalize_xyz", False)),
        knn_chunk_size=int(getattr(pam, "knn_chunk_size", 1024)),
        use_dist_bias=bool(getattr(pam, "use_dist_bias", True)),
        dist_sigma_scale=float(getattr(pam, "dist_sigma_scale", 1.0)),
        adaptive_aug=bool(getattr(pam, "adaptive_aug", True)),
    ).to(device)

    # ---- phi: full_in -> hidden (slice OUT) ----
    new_pam.phi[0].weight.data.copy_(_slice_out(pam.phi[0], ki))
    _slice_bn2d(new_pam.phi[1], pam.phi[1], ki)

    # ---- psi: full_in -> hidden (slice OUT) ----
    new_pam.psi[0].weight.data.copy_(_slice_out(pam.psi[0], ki))
    _slice_bn2d(new_pam.psi[1], pam.psi[1], ki)

    # ---- delta_mlp: 3->hidden (slice OUT), hidden->hidden (slice OUT&IN) ----
    new_pam.delta_mlp[0].weight.data.copy_(_slice_out(pam.delta_mlp[0], ki))
    _slice_bn2d(new_pam.delta_mlp[1], pam.delta_mlp[1], ki)

    new_pam.delta_mlp[3].weight.data.copy_(_slice_out_in(pam.delta_mlp[3], ki, ki))
    _slice_bn2d(new_pam.delta_mlp[4], pam.delta_mlp[4], ki)

    # ---- delta_to_in: hidden->full_in (slice IN only, out=full_in 不变!) ----
    new_pam.delta_to_in[0].weight.data.copy_(_slice_in(pam.delta_to_in[0], ki))
    _copy_bn2d(new_pam.delta_to_in[1], pam.delta_to_in[1])  # full_in BN 完整拷贝

    # ---- gamma: hidden->out (slice IN only), out->out 全拷贝 ----
    new_pam.gamma[0].weight.data.copy_(_slice_in(pam.gamma[0], ki))
    _copy_bn2d(new_pam.gamma[1], pam.gamma[1])
    new_pam.gamma[3].weight.data.copy_(pam.gamma[3].weight.data)
    _copy_bn2d(new_pam.gamma[4], pam.gamma[4])

    # ---- alpha/out_mapper：接口层全拷贝（out_channel 不变） ----
    for i in range(len(new_pam.alpha)):
        if isinstance(new_pam.alpha[i], nn.Conv2d):
            new_pam.alpha[i].weight.data.copy_(pam.alpha[i].weight.data)
        elif isinstance(new_pam.alpha[i], nn.BatchNorm2d):
            _copy_bn2d(new_pam.alpha[i], pam.alpha[i])

    new_pam.out_mapper[0].weight.data.copy_(pam.out_mapper[0].weight.data)
    new_pam.out_mapper[1].weight.data.copy_(pam.out_mapper[1].weight.data)
    new_pam.out_mapper[1].bias.data.copy_(pam.out_mapper[1].bias.data)
    new_pam.out_mapper[1].running_mean.data.copy_(pam.out_mapper[1].running_mean.data)
    new_pam.out_mapper[1].running_var.data.copy_(pam.out_mapper[1].running_var.data)

    return new_pam

def prune_backbone_only(model: nn.Module, report: dict):
    for pam_name, info in report["pam_hidden"].items():
        pam = _get_submodule(model, pam_name)
        assert isinstance(pam, PAModuleVotes)
        ki = torch.tensor(info["keep_idx"], device=next(pam.parameters()).device)
        new_pam = rebuild_pruned_pam_hidden(pam, ki)
        _set_submodule(model, pam_name, new_pam)

def main():
    EXP_DIR = "/home/ljc/work/3DVLP/outputs/exp_joint/2026-01-12_22-46-07"
    device = "cuda:1" if torch.cuda.is_available() else "cpu"

    args = load_args_from_info(EXP_DIR)
    model = build_model_from_train_args(args, device)
    load_checkpoint_weights(model, EXP_DIR, device)
    model.eval()

    # 只评估/剪 sa1-4 的 PAM hidden
    report = evaluate_pam_hidden_keepidx(
        model,
        keep_ratio=0.5,
        include_prefixes=("backbone_net.sa1","backbone_net.sa2","backbone_net.sa3","backbone_net.sa4"),
        min_keep_channels=8,
    )

    outdir = "prune_viz"
    plot_keep_ratio_per_layer(report, outdir=outdir)

    # 如果你愿意传 model，就能画 backbone BN gamma 的全量分布（更像论文证据）
    plot_importance_hist_and_cdf(report, model=model, outdir=outdir)

    plot_block_summary(report, outdir=outdir)


    prune_backbone_only(model, report)
    model.eval()

    torch.save({"model_state_dict": model.state_dict(), "pruning_report": report},
               "pruned_backbone_only.pth")
    print(">>> Saved pruned_backbone_only.pth (voting untouched, interface aligned)")

if __name__ == "__main__":
    main()
