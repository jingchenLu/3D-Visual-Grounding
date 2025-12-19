import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils_fn import *
from utils.box_util import box3d_diou_batch_tensor
from pytorch3d.ops.iou_box3d import box3d_overlap


# def create_box_batch(center, size):
#     extend_center = center[:, None, :].repeat(1, 8, 1)  # bs, 8, 3
#     unit = torch.tensor([[[-1, -1, -1], [1, -1, -1], [1, 1, -1],
#                           [-1, 1, -1], [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]]]).cuda()
#     extend_size = size[:, None, :]*unit/2  # bs, 8, 3
#     box_batch = extend_center+extend_size
#     return box_batch.float()
def create_box_batch(center, size):
    # center: (M,3), size:(M,3)
    device = center.device
    extend_center = center[:, None, :].repeat(1, 8, 1)
    unit = torch.tensor(
        [[[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
          [-1, -1,  1], [1, -1,  1], [1, 1,  1], [-1, 1,  1]]],
        device=device, dtype=center.dtype
    )
    extend_size = size[:, None, :] * unit / 2
    return (extend_center + extend_size).float()


# def SoftCrossEntropy(inputs, target):
#     log_likelihood = -F.log_softmax(inputs, dim=-1)
#     loss = torch.mean(torch.mul(log_likelihood, target))
#     return loss
def soft_ce_from_target(logits: torch.Tensor, target: torch.Tensor, dim: int = -1, eps: float = 1e-6):
    """
    logits: (..., K)
    target: same shape or broadcastable; can be mask/weights; will be normalized to a distribution.
    """
    target = target.float()
    target = target / (target.sum(dim=dim, keepdim=True) + eps)
    logp = F.log_softmax(logits, dim=dim)
    return -(target * logp).sum(dim=dim).mean()


class NCELoss(nn.Module):
    def __init__(self, init_tau=0.07, clamp_max_inv_tau=100.0):
        super().__init__()
        # log(1/tau)
        self.log_inv_tau = nn.Parameter(torch.tensor([float(torch.log(torch.tensor(1.0 / init_tau)))]))
        self.clamp_max_inv_tau = clamp_max_inv_tau

    def inv_tau(self, max_inv_tau=None):
        inv = self.log_inv_tau.exp()
        inv = torch.clamp(inv, max=self.clamp_max_inv_tau)
        if max_inv_tau is not None:
            inv = torch.clamp(inv, max=max_inv_tau)
        return inv

    def forward(self, logits, target, symmetric=False, max_inv_tau=None):
        logits = logits * self.inv_tau(max_inv_tau)
        loss = soft_ce_from_target(logits, target, dim=-1)
        if symmetric:
            loss_t = soft_ce_from_target(logits.t(), target.t(), dim=-1)
            loss = 0.5 * (loss + loss_t)
        return loss

class ContrastModule(nn.Module):
    def __init__(self, config, hidden=128):
        super().__init__()
        self.pc_proj = nn.Linear(hidden, hidden, bias=False)
        self.text_proj = nn.Linear(hidden, hidden, bias=False)
        self.nce_loss = NCELoss()
        self.config = config
        self.bce_loss = nn.BCEWithLogitsLoss()

        self.pc_proj_iou = nn.Sequential(
            nn.Linear(hidden,hidden,bias=False)
        )

    def _ramp(self, epoch, a, b):
        if epoch <= a: return 0.0
        if epoch >= b: return 1.0
        return float(epoch - a) / float(b - a)

    def _curriculum(self, epoch):
        # coarse -> fine
        p = self._ramp(epoch, 0, 80)       # 0~80 收紧
        thr = 0.05 + 0.3 * p              # 0.05 -> 0.35
        gamma = 1.0 + 2.0 * p              # 1.0 -> 3.0
        min_tau = 0.08 - 0.04 * p          # 0.08 -> 0.04
        con_w = self._ramp(epoch, 5, 50)   # 对比loss权重 warmup
        return thr, gamma, min_tau, con_w

    def forward(self, data_dict):
        # if data_dict["epoch"] < 50:
        #     data_dict["con_loss"] = torch.zeros(1)
        #     return data_dict

        epoch = int(data_dict["epoch"])
        thr, gamma, min_tau, con_w = self._curriculum(epoch)

        if con_w <= 0:
            device = data_dict["bbox_feature"].device
            z = torch.zeros((), device=device)   # 0-d scalar
            data_dict["lang_con_loss"] = z
            data_dict["iou_con_loss"]  = z
            data_dict["con_w"] = con_w
            data_dict["con_thr"] = thr
            data_dict["con_gamma"] = gamma
            data_dict["con_min_tau"] = min_tau
            # diagnostics 也顺手给标量，避免后面写TB时报 shape 不一致
            data_dict["con_pos_sim"] = z
            data_dict["con_neg_sim"] = z
            data_dict["con_gap"] = z
            data_dict["con_corr"] = z
            return data_dict
        
        pred_center = data_dict['pred_center'].detach()
        pred_box_size = data_dict['pred_size'].detach()
        features = data_dict["bbox_feature"]  # bs, num_proposal, hidden

        gt_center_list = data_dict['ref_center_label_list'].detach()  # (B,3)
        # B
        gt_heading_class_list = data_dict['ref_heading_class_label_list'].detach(
        )
        # B
        gt_heading_residual_list = data_dict['ref_heading_residual_label_list'].detach(
        )
        # B
        gt_size_class_list = data_dict['ref_size_class_label_list'].detach()
        # B,3
        gt_size_residual_list = data_dict['ref_size_residual_label_list'].detach(
        )
        # convert gt bbox parameters to bbox corners
        batch_size, num_proposals = data_dict['aggregated_vote_features'].shape[:2]
        batch_size, len_nun_max = gt_center_list.shape[:2]
        lang_num = data_dict["lang_num"]
        lang_emb = data_dict["lang_emb"]
        lang_num_max = lang_emb.shape[0]//batch_size
        lang_emb = lang_emb.view(batch_size, lang_num_max, -1)
        # objectness_masks = data_dict['objectness_scores'].max(2)[1].float()
        # ==== NEW: topK proposals by objectness prob (avoid empty) ====
        obj_logits = data_dict["objectness_scores"]              # (B, N, 2)
        obj_prob   = F.softmax(obj_logits.detach(), dim=2)[..., 1]  # (B, N) prob of "object"
        topk = min(64, obj_prob.shape[1])
        topk = max(topk, 1)
        topk_inds = torch.topk(obj_prob, k=topk, dim=1, largest=True, sorted=False).indices  # (B, topk)

        device = features.device
        lang_con_loss = torch.zeros((), device=device)
        iou_con_loss  = torch.zeros((), device=device)
        for i in range(batch_size):
            pred_center_batch = pred_center[i]
            pred_box_size_batch = pred_box_size[i]
            gt_box_center, gt_box_size = self.config.param2obb_batch_tensor(
                gt_center_list[i][:, 0:3], gt_heading_class_list[i], gt_heading_residual_list[i], gt_size_class_list[i], gt_size_residual_list[i])
            
            object_index = topk_inds[i]              # (topk,)
            features_batch = features[i][object_index]
            box_batch = create_box_batch(
                pred_center_batch[object_index],
                pred_box_size_batch[object_index]
            )
            
            for j in range(len_nun_max):
                if j < lang_num[i]:
                    # convert the bbox parameters to bbox corners
                    lang_emb_batch = lang_emb[i][j][None,:]
                    gt_box_size_batch = gt_box_size[j][None, :]
                    gt_box_center_batch = gt_box_center[j][None, :]
                    gt_box_batch = create_box_batch(
                        gt_box_center_batch, gt_box_size_batch+1e-2)
                    try:
                        # 计算 Ground Truth 和预测的 box 之间的重叠度
                        _, ious = box3d_overlap(
                            gt_box_batch, box_batch, eps=1e-7)  # 1, 256
                        ious = ious.view(-1)  # 256
                        # ---- soft target (IoU^gamma) + curriculum threshold ----
                        w = (ious.clamp(min=0.0) ** gamma) * (ious >= thr).float()
                        if w.sum() < 1e-6:
                            w.zero_()
                            w[torch.argmax(ious)] = 1.0
                        target_dist = (w / (w.sum() + 1e-6)).unsqueeze(0).detach()  # (1, N)

                        # ---- similarity (text -> proposals) ----
                        text_feat = F.normalize(self.text_proj(lang_emb_batch), dim=-1)        # (1,H)
                        box_feat  = F.normalize(self.pc_proj(features_batch), dim=-1)          # (N,H)
                        sim_lang  = torch.mm(text_feat, box_feat.t())                          # (1,N)

                        max_inv_tau = 1.0 / max(min_tau, 1e-6)
                        lang_con = self.nce_loss(sim_lang, target_dist, symmetric=False, max_inv_tau=max_inv_tau)

                        # ---- hard negative ranking (关键：专打“高相似但IoU低”的错配) ----
                        neg_mask = (ious < 0.05)
                        if neg_mask.any():
                            s_neg = sim_lang[0][neg_mask].max()
                            s_pos = (sim_lang[0] * target_dist[0]).sum()
                            margin = 0.2
                            rank_loss = F.relu(margin - s_pos + s_neg)
                            lang_con = lang_con + 0.5 * rank_loss

                        lang_con_loss += lang_con

                        # ---- proposal consistency (IoU-guided) ----
                        w_obj = (ious.clamp(min=0.0) ** gamma) * (ious >= thr).float()
                        if w_obj.sum() < 1e-6:
                            w_obj.zero_()
                            w_obj[torch.argmax(ious)] = 1.0

                        target_mat = (w_obj[:, None] * w_obj[None, :]).detach()                # (N,N)
                        target_mat = target_mat / (target_mat.sum(dim=-1, keepdim=True) + 1e-6)

                        box_feat_iou = F.normalize(self.pc_proj_iou(features_batch), dim=-1)
                        sim_iou = torch.mm(box_feat_iou, box_feat_iou.t())                      # (N,N)
                        iou_con_loss += self.nce_loss(sim_iou, target_mat, symmetric=False, max_inv_tau=max_inv_tau)

                        # ---- diagnostics for visualization ----
                        with torch.no_grad():
                            pos_sim = (sim_lang[0] * target_dist[0]).sum()
                            neg_sim = sim_lang[0][neg_mask].max() if neg_mask.any() else sim_lang[0].min()
                            gap = pos_sim - neg_sim

                            s_vec = sim_lang[0]
                            iou_vec = ious  

                            s_vec = (s_vec - s_vec.mean()) / (s_vec.std() + 1e-6)
                            iou_vec = (iou_vec - iou_vec.mean()) / (iou_vec.std() + 1e-6)
                            corr = (s_vec * iou_vec).mean()

                            # 累加到 data_dict
                            data_dict.setdefault("con_pos_sim_sum", 0.0)
                            data_dict.setdefault("con_neg_sim_sum", 0.0)
                            data_dict.setdefault("con_gap_sum", 0.0)
                            data_dict.setdefault("con_corr_sum", 0.0)
                            data_dict.setdefault("con_cnt", 0.0)

                            data_dict["con_pos_sim_sum"] += pos_sim.item()
                            data_dict["con_neg_sim_sum"] += neg_sim.item()
                            data_dict["con_gap_sum"] += gap.item()
                            data_dict["con_corr_sum"] += corr.item()
                            data_dict["con_cnt"] += 1.0
                    except Exception as e:
                        print("Error:", e)

        lang_con_loss /= batch_size
        iou_con_loss /= batch_size
        data_dict["lang_con_loss"] = lang_con_loss
        data_dict["iou_con_loss"] =iou_con_loss 
        # curriculum scalars
        data_dict["con_w"] = con_w
        data_dict["con_thr"] = thr
        data_dict["con_gamma"] = gamma
        data_dict["con_min_tau"] = min_tau

        # averaged diagnostics
        cnt = float(data_dict.get("con_cnt", 0.0))
        if cnt > 0:
            data_dict["con_pos_sim"] = torch.tensor(data_dict["con_pos_sim_sum"] / cnt, device=features.device)
            data_dict["con_neg_sim"] = torch.tensor(data_dict["con_neg_sim_sum"] / cnt, device=features.device)
            data_dict["con_gap"]     = torch.tensor(data_dict["con_gap_sum"] / cnt, device=features.device)
            data_dict["con_corr"]    = torch.tensor(data_dict["con_corr_sum"] / cnt, device=features.device)
        else:
            z = torch.zeros(1, device=features.device)
            data_dict["con_pos_sim"] = z
            data_dict["con_neg_sim"] = z
            data_dict["con_gap"] = z
            data_dict["con_corr"] = z

        return data_dict
    