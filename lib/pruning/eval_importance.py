# lib/pruning/eval_importance.py
import torch
import torch.nn as nn
from collections import OrderedDict
from typing import Dict, Tuple, Optional

from models.base_module.backbone_module import PAModuleVotes

DEFAULT_PAM_PREFIXES = (
    "backbone_net.sa1",
    "backbone_net.sa2",
    "backbone_net.sa3",
    "backbone_net.sa4",

)

def _topk_keep(scores: torch.Tensor, keep_ratio: float, min_keep: int = 8) -> torch.Tensor:
    c = scores.numel()
    k = max(min_keep, int(round(c * keep_ratio)))
    k = min(c, k)
    return torch.topk(scores, k=k, largest=True, sorted=False).indices

@torch.no_grad()
def _pam_hidden_score(pam: PAModuleVotes) -> torch.Tensor:
    """
    只用 hidden 相关 BN 的 |gamma| 作为 hidden_dim 的重要性（长度=hidden_dim）
    """
    scores = None
    cnt = 0

    def add_bn(bn: Optional[nn.Module]):
        nonlocal scores, cnt
        if bn is None or (not hasattr(bn, "weight")) or bn.weight is None:
            return
        v = bn.weight.detach().abs().float()
        scores = v if scores is None else (scores + v)
        cnt += 1

    # phi: Conv2d, BN2d, ReLU
    if isinstance(pam.phi, nn.Sequential) and len(pam.phi) >= 2:
        add_bn(pam.phi[1])
    # psi
    if isinstance(pam.psi, nn.Sequential) and len(pam.psi) >= 2:
        add_bn(pam.psi[1])
    # delta_mlp: [conv, bn, relu, conv, bn, relu]
    if isinstance(pam.delta_mlp, nn.Sequential) and len(pam.delta_mlp) >= 5:
        add_bn(pam.delta_mlp[1])
        add_bn(pam.delta_mlp[4])

    if scores is None:
        # fallback: hidden_dim 从 phi 的 out_channels 推断
        hidden_dim = pam.phi[0].out_channels
        return torch.ones(hidden_dim, device=next(pam.parameters()).device)

    return scores / max(cnt, 1)

def evaluate_pam_hidden_keepidx(
    model: nn.Module,
    keep_ratio: float = 0.75,
    include_prefixes: Tuple[str, ...] = DEFAULT_PAM_PREFIXES,
    min_keep_channels: int = 8,
) -> Dict:
    model.eval()
    report = OrderedDict()
    report["meta"] = {
        "mode": "pam_hidden_only",
        "keep_ratio": float(keep_ratio),
        "include_prefixes": list(include_prefixes),
        "min_keep_channels": int(min_keep_channels),
    }
    report["pam_hidden"] = OrderedDict()

    for name, m in model.named_modules():
        if not name.startswith(include_prefixes):
            continue
        if not isinstance(m, PAModuleVotes):
            continue

        scores = _pam_hidden_score(m)
        keep_idx = _topk_keep(scores, keep_ratio, min_keep_channels)

        report["pam_hidden"][name] = {
            "hidden_dim": int(scores.numel()),
            "keep": int(keep_idx.numel()),
            "keep_ratio": float(keep_idx.numel()) / float(scores.numel()),
            "keep_idx": keep_idx.cpu().tolist(),
        }

    return report
