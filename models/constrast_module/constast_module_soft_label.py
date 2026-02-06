import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from utils.utils_fn import *
from utils.box_util import box3d_diou_batch_tensor
from pytorch3d.ops.iou_box3d import box3d_overlap


def create_box_batch(center, size):
    """
    center: (N,3)
    size:   (N,3)
    return: (N,8,3)
    """
    extend_center = center[:, None, :].repeat(1, 8, 1)  # (N,8,3)
    unit = center.new_tensor(
        [[[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
          [-1, -1,  1], [1, -1,  1], [1, 1,  1], [-1, 1,  1]]]
    )  # (1,8,3) on same device/dtype
    extend_size = size[:, None, :] * unit / 2.0
    box_batch = extend_center + extend_size
    return box_batch.float()


def soft_target_cross_entropy(logits: torch.Tensor,
                              target: torch.Tensor,
                              dim: int = -1,
                              reduction: str = "mean"):
    """
    Standard soft-label cross-entropy:
        CE = - sum_j target_j * log softmax(logits)_j
    target is expected to be non-negative and (usually) row-normalized.
    """
    log_prob = F.log_softmax(logits, dim=dim)
    loss = -(target * log_prob).sum(dim=dim)  # (row,)
    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    else:
        return loss


def row_normalize(x: torch.Tensor, eps: float = 1e-8):
    s = x.sum(dim=-1, keepdim=True)
    return x / (s + eps)


class NCELoss(nn.Module):
    """
    Soft-target InfoNCE style loss.

    If target shape matches logits: CE(logits, target)
    If symmetric=True and target^T matches logits^T: add CE(logits^T, target^T)
    """
    def __init__(self, init_tau=0.07, clamp=4.6051):
        super().__init__()
        self.tau = torch.nn.Parameter(torch.tensor(
            [np.log(1.0 / init_tau)], dtype=torch.float32))
        self.clamp = clamp

    def forward(self, logits: torch.Tensor, target: torch.Tensor, symmetric: bool = True):
        # Optional learnable temperature (kept as your original, but commented out in forward)
        # self.tau.data = torch.clamp(self.tau.data, 0, self.clamp)
        # logits = logits * self.tau.exp()

        loss_v = soft_target_cross_entropy(logits, target, dim=-1, reduction="mean")

        if not symmetric:
            return loss_v

        # Only add symmetric term when shapes are consistent
        if logits.t().shape == target.t().shape:
            loss_t = soft_target_cross_entropy(logits.t(), target.t(), dim=-1, reduction="mean")
            return 0.5 * (loss_v + loss_t)
        else:
            return loss_v


class ContrastModule(nn.Module):
    def __init__(self, config, hidden=128):
        super().__init__()
        self.pc_proj = nn.Linear(hidden, hidden, bias=False)
        self.text_proj = nn.Linear(hidden, hidden, bias=False)
        self.pc_proj_iou = nn.Sequential(nn.Linear(hidden, hidden, bias=False))

        self.nce_loss = NCELoss()
        self.config = config

        # soft label hyperparams (you can tune)
        self.soft_gamma = 2.0          # IoU^gamma, larger -> more peaky
        self.min_iou_clip = 0.0        # clip IoU lower bound before power
        self.drop_diag = True          # avoid trivial self-pair in iou branch

    def _build_soft_label_from_ious(self, ious: torch.Tensor, gamma: float = 2.0, eps: float = 1e-8):
        """
        ious: (K,)
        return p: (K,) normalized distribution (sum=1) or None if all-zero
        """
        ious = ious.clamp(min=self.min_iou_clip)
        w = torch.pow(ious, gamma)
        s = w.sum()
        if s.item() <= eps:
            return None
        p = (w / (s + eps)).detach()
        return p

    def forward(self, data_dict):
        # Keep your original "start after 50 epochs"
        if data_dict["epoch"] < 1:
            data_dict["con_loss"] = torch.zeros(1, device=data_dict["bbox_feature"].device)
            data_dict["lang_con_loss"] = torch.zeros(1, device=data_dict["bbox_feature"].device)
            data_dict["iou_con_loss"] = torch.zeros(1, device=data_dict["bbox_feature"].device)
            return data_dict

        pred_center = data_dict['pred_center'].detach()
        pred_box_size = data_dict['pred_size'].detach()
        features = data_dict["bbox_feature"]  # (B, num_proposal, hidden)

        gt_center_list = data_dict['ref_center_label_list'].detach()
        gt_heading_class_list = data_dict['ref_heading_class_label_list'].detach()
        gt_heading_residual_list = data_dict['ref_heading_residual_label_list'].detach()
        gt_size_class_list = data_dict['ref_size_class_label_list'].detach()
        gt_size_residual_list = data_dict['ref_size_residual_label_list'].detach()

        batch_size, num_proposals = data_dict['aggregated_vote_features'].shape[:2]
        _, len_num_max = gt_center_list.shape[:2]

        lang_num = data_dict["lang_num"]
        lang_emb = data_dict["lang_emb"]
        lang_num_max = lang_emb.shape[0] // batch_size
        lang_emb = lang_emb.view(batch_size, lang_num_max, -1)

        objectness_masks = data_dict['objectness_scores'].max(2)[1].float()  # (B, num_prop)

        lang_con_loss = 0.0
        iou_con_loss = 0.0

        for i in range(batch_size):
            pred_center_batch = pred_center[i]
            pred_box_size_batch = pred_box_size[i]

            gt_box_center, gt_box_size = self.config.param2obb_batch_tensor(
                gt_center_list[i][:, 0:3],
                gt_heading_class_list[i],
                gt_heading_residual_list[i],
                gt_size_class_list[i],
                gt_size_residual_list[i]
            )

            object_index = torch.where(objectness_masks[i])[0]
            object_cnt = object_index.shape[0]
            if object_cnt <= 1:
                continue

            features_batch = features[i][object_index]  # (K, hidden)
            box_batch = create_box_batch(
                pred_center_batch[object_index], pred_box_size_batch[object_index]
            )  # (K,8,3)

            # precompute normalized proposal feats for both branches
            box_feat_norm_lang = F.normalize(self.pc_proj(features_batch), dim=-1)     # (K,h)
            box_feat_norm_iou  = F.normalize(self.pc_proj_iou(features_batch), dim=-1) # (K,h)

            # proposal-proposal sim (K,K)
            sim_iou = torch.mm(box_feat_norm_iou, box_feat_norm_iou.t())
            if self.drop_diag:
                sim_iou = sim_iou.clone()
                sim_iou.fill_diagonal_(-1e9)

            for j in range(len_num_max):
                if j >= lang_num[i]:
                    continue

                lang_emb_batch = lang_emb[i][j][None, :]  # (1,h)
                gt_box_size_batch = gt_box_size[j][None, :]
                gt_box_center_batch = gt_box_center[j][None, :]
                gt_box_batch = create_box_batch(gt_box_center_batch, gt_box_size_batch + 1e-2)  # (1,8,3)

                try:
                    _, ious = box3d_overlap(gt_box_batch, box_batch, eps=1e-7)  # (1,K)
                    ious = ious.view(-1)  # (K,)

                    # --------- build soft label p over proposals (K,) ----------
                    p = self._build_soft_label_from_ious(ious, gamma=self.soft_gamma)
                    if p is None:
                        continue

                    # ============ (A) lang_con_loss: text -> proposals ============
                    text_feat_norm = F.normalize(self.text_proj(lang_emb_batch), dim=-1)  # (1,h)
                    sim_lang = torch.mm(text_feat_norm, box_feat_norm_lang.t())          # (1,K)

                    # soft target distribution for this sentence: (1,K)
                    target_lang = p[None, :]  # already sums to 1
                    lang_con_loss += self.nce_loss(sim_lang, target_lang, symmetric=False)

                    # ============ (B) iou_con_loss: proposals -> proposals ============
                    # For each anchor row i, target distribution is p over columns,
                    # but we weight rows by p_i so only "more-positive" anchors contribute strongly.
                    target_iou = p[None, :].repeat(object_cnt, 1)            # (K,K) row-wise same distribution
                    row_weight = p[:, None]                                  # (K,1)
                    target_iou = (target_iou * row_weight)                   # (K,K), rows scaled
                    # normalize each row to sum=1 where possible
                    target_iou = row_normalize(target_iou)

                    iou_con_loss += self.nce_loss(sim_iou, target_iou, symmetric=True)

                except Exception as e:
                    print("Error:", e)

        lang_con_loss = lang_con_loss / batch_size
        iou_con_loss = iou_con_loss / batch_size

        data_dict["lang_con_loss"] = lang_con_loss
        data_dict["iou_con_loss"] = iou_con_loss
        data_dict["con_loss"] = lang_con_loss + iou_con_loss
        return data_dict
