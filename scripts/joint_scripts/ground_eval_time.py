import os
import sys
import json
import pickle
import argparse
import importlib
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import time

from torch.utils.data import DataLoader
from datetime import datetime
from tqdm import tqdm
from copy import deepcopy

sys.path.append(os.path.join(os.getcwd())) # HACK add the root folder
from lib.configs.config_joint import CONF
from lib.joint.dataset import ScannetReferenceDataset
# 11月27日 注释
from lib.joint.solver_3dvlp import Solver
from lib.ap_helper.ap_helper_fcos import APCalculator, parse_predictions, parse_groundtruths
from lib.loss_helper.loss_joint import get_joint_loss
from lib.joint.eval_ground import get_eval
from models.jointnet.jointnet import JointNet
from data.scannet.model_util_scannet import ScannetDatasetConfig

print('Import Done', flush=True)
SCANREFER_TRAIN = json.load(open(os.path.join(CONF.PATH.DATA, "ScanRefer_filtered_train.json")))
SCANREFER_VAL = json.load(open(os.path.join(CONF.PATH.DATA, "ScanRefer_filtered_val.json")))

def get_dataloader(args, scanrefer, scanrefer_new, all_scene_list, split, config):
    dataset = ScannetReferenceDataset(
        scanrefer=scanrefer,
        scanrefer_new=scanrefer_new,
        scanrefer_all_scene=all_scene_list, 
        split=split,
        name=args.dataset,
        num_points=args.num_points, 
        use_color=args.use_color, 
        use_height=(not args.no_height),
        use_normal=args.use_normal, 
        use_multiview=args.use_multiview,
        lang_num_max=args.lang_num_max
    )
    print("evaluate on {} samples".format(len(dataset)))
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    return dataset, dataloader

def get_model(args, DC, dataset):
    input_channels = int(args.use_multiview) * 128 + int(args.use_normal) * 3 + int(args.use_color) * 3 + int(not args.no_height)
    model = JointNet(
        num_class=DC.num_class,
        vocabulary=dataset.vocabulary,
        embeddings=dataset.glove,
        num_heading_bin=DC.num_heading_bin,
        num_size_cluster=DC.num_size_cluster,
        mean_size_arr=DC.mean_size_arr,
        input_feature_dim=input_channels,
        num_proposal=args.num_proposals,
        no_caption=True,
        use_topdown=False,
        use_con=args.use_con,
        use_lang_classifier=(not args.no_lang_cls),
        use_bidir=args.use_bidir,
        dataset_config=DC
    ).cuda()

    model_name = "model_last.pth" if args.detection else "model.pth"
    path = os.path.join(CONF.PATH.OUTPUT, args.folder, model_name)
    model.load_state_dict(torch.load(path), strict=False)
    model.eval()
    return model

def get_scannet_scene_list(split):
    scene_list = sorted([line.rstrip() for line in open(os.path.join(CONF.PATH.SCANNET_META, "scannetv2_{}.txt".format(split)))])
    return scene_list

def get_scanrefer(args):
    if args.detection:
        scene_list = get_scannet_scene_list("val")
        scanrefer = []
        for scene_id in scene_list:
            data = deepcopy(SCANREFER_TRAIN[0])
            data["scene_id"] = scene_id
            scanrefer.append(data)
    else:
        scanrefer = SCANREFER_TRAIN if args.use_train else SCANREFER_VAL
        scene_list = sorted(list(set([data["scene_id"] for data in scanrefer])))
        if args.num_scenes != -1:
            scene_list = scene_list[:args.num_scenes]

        scanrefer = [data for data in scanrefer if data["scene_id"] in scene_list]

        new_scanrefer_val = scanrefer
        scanrefer_val_new = []
        scanrefer_val_new_scene = []
        scene_id = ""
        for data in scanrefer:
            if scene_id != data["scene_id"]:
                scene_id = data["scene_id"]
                if len(scanrefer_val_new_scene) > 0:
                    scanrefer_val_new.append(scanrefer_val_new_scene)
                scanrefer_val_new_scene = []
            if len(scanrefer_val_new_scene) >= args.lang_num_max:
                scanrefer_val_new.append(scanrefer_val_new_scene)
                scanrefer_val_new_scene = []
            scanrefer_val_new_scene.append(data)
        if len(scanrefer_val_new_scene) > 0:
            scanrefer_val_new.append(scanrefer_val_new_scene)

    return scanrefer, scene_list, scanrefer_val_new

# ==========================================
#  修改后的 eval_ref 函数：只测 FPS，只跑 30 个样本
# ==========================================
def eval_ref(args):
    print("Evaluating FPS Only...")
    DC = ScannetDatasetConfig()

    print("preparing data...")
    scanrefer, scene_list, scanrefer_val_new = get_scanrefer(args)
    dataset, dataloader = get_dataloader(args, scanrefer, scanrefer_val_new, scene_list, "val", DC)

    model = get_model(args, DC, dataset)
    
    # 随机种子设置
    seed = args.seed
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)

    print("Start FPS evaluation loop...")
    
    # === 时间统计列表 ===
    time_stats = {
        'pointnet': [],
        'vote_prop': [],
        'relation': [],
        'text': [],
        'fusion': [],
        'total': []
    }
    
    # === 控制变量 ===
    TARGET_SAMPLE_COUNT = 100  # 目标有效样本数
    valid_sample_count = 0    # 当前有效样本数 (不含预热)
    warmup_steps = 5          # 预热 Batch 数
    
    target_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    for i, data in enumerate(tqdm(dataloader)):
        # 1. 数据搬运
        for key in data:
            if isinstance(data[key], torch.Tensor):
                data[key] = data[key].to(target_device)

        # 2. 推理
        with torch.no_grad():
            data["epoch"] = 0
            
            # 判断是否为预热阶段
            is_warmup = i < warmup_steps
            
            # 执行模型前向传播 (包含内部计时)
            data = model(data, is_eval=True)

            # 3. 收集时间 (仅非预热阶段)
            if not is_warmup and 'exec_times' in data:
                et = data['exec_times']
                time_stats['pointnet'].append(et['pointnet_time'])
                time_stats['vote_prop'].append(et['voting_proposal_time'])
                time_stats['relation'].append(et['relation_time'])
                time_stats['text'].append(et['text_time'])
                time_stats['fusion'].append(et['fusion_time'])
                time_stats['total'].append(et['total_time'])

                # 获取当前 Batch 的样本数量
                # 如果没有 scan_idx，回退使用 batch_size
                current_batch_size = data['scan_idx'].shape[0] if 'scan_idx' in data else args.batch_size
                valid_sample_count += current_batch_size
                
                # 4. 检查是否达到目标样本数
                if valid_sample_count >= TARGET_SAMPLE_COUNT:
                    print(f"\n[Info] Successfully processed {valid_sample_count} samples (excluding warmup). Stopping evaluation.")
                    break
        
        # 这里的 break 用于跳出 tqdm 循环
        if valid_sample_count >= TARGET_SAMPLE_COUNT:
            break

    # === 5. 打印 FPS 报告 ===
    if len(time_stats['total']) > 0:
        avg_pn = np.mean(time_stats['pointnet'])
        avg_vp = np.mean(time_stats['vote_prop'])
        avg_rel = np.mean(time_stats['relation'])
        avg_txt = np.mean(time_stats['text'])
        avg_fus = np.mean(time_stats['fusion'])
        avg_tot = np.mean(time_stats['total'])
        
        # 计算 FPS
        fps = args.batch_size / avg_tot
        
        print("\n" + "="*50)
        print(f"FPS Benchmark Report (Batch Size = {args.batch_size})")
        print(f"Processed Valid Samples: {valid_sample_count}")
        print("-" * 50)
        print(f"{'Module':<25} | {'Time (s)':<10} | {'Ratio':<10}")
        print("-" * 50)
        print(f"{'PointNet++':<25} | {avg_pn:.4f}     | {avg_pn/avg_tot*100:.1f}%")
        print(f"{'Voting & Proposal':<25} | {avg_vp:.4f}     | {avg_vp/avg_tot*100:.1f}%")
        print(f"{'Relation':<25} | {avg_rel:.4f}     | {avg_rel/avg_tot*100:.1f}%")
        print(f"{'Text Encoder':<25} | {avg_txt:.4f}     | {avg_txt/avg_tot*100:.1f}%")
        print(f"{'Fusion (Match)':<25} | {avg_fus:.4f}     | {avg_fus/avg_tot*100:.1f}%")
        print("-" * 50)
        print(f"{'Total Latency / Batch':<25} | {avg_tot:.4f} s")
        print("-" * 50)
        print(f"System FPS                : {fps:.2f}")
        print("="*50 + "\n")
    else:
        print("[Error] No valid timing data collected. Please check batch size or loop steps.")

# eval_det 函数略... 如果不需要也可以留空或者保持原样

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, help="Choose a dataset: ScanRefer or ReferIt3D", default="ScanRefer")
    parser.add_argument("--folder", type=str, help="Folder containing the model")
    parser.add_argument("--gpu", type=str, help="gpu", default="0")
    parser.add_argument("--batch_size", type=int, help="batch size", default=8)
    parser.add_argument("--lang_num_max", type=int, help="lang num max", default=32)
    parser.add_argument("--num_points", type=int, default=40000, help="Point Number [default: 40000]")
    parser.add_argument("--num_proposals", type=int, default=256, help="Proposal number [default: 256]")
    parser.add_argument("--num_scenes", type=int, default=-1, help="Number of scenes [default: -1]")
    parser.add_argument("--force", action="store_true", help="enforce the generation of results")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--repeat", type=int, default=1, help="Number of times for evaluation")
    parser.add_argument("--no_height", action="store_true", help="Do NOT use height signal in input.")
    parser.add_argument("--no_lang_cls", action="store_true", help="Do NOT use language classifier.")
    parser.add_argument("--no_nms", action="store_true", help="do NOT use non-maximum suppression for post-processing.")
    parser.add_argument("--use_color", action="store_true", help="Use RGB color in input.")
    parser.add_argument("--use_normal", action="store_true", help="Use RGB color in input.")
    parser.add_argument("--use_multiview", action="store_true", help="Use multiview images.")
    parser.add_argument("--use_bidir", action="store_true", help="Use bi-directional GRU.")
    parser.add_argument("--use_train", action="store_true", help="Use train split in evaluation.")
    parser.add_argument("--use_con", action="store_true", help="Use contrastive losses / contrast module.")
    parser.add_argument("--use_diou_loss", action="store_true", help="Use DIOU loss in grounding/detection.")
    parser.add_argument("--use_oracle", action="store_true", help="Use ground truth bounding boxes.")
    parser.add_argument("--use_cat_rand", action="store_true", help="Use randomly selected bounding boxes from correct categories as outputs.")
    parser.add_argument("--use_best", action="store_true", help="Use best bounding boxes as outputs.")
    parser.add_argument("--reference", action="store_true", help="evaluate the reference localization results")
    parser.add_argument("--detection", action="store_true", help="evaluate the object detection results")
    args = parser.parse_args()

    for _flag in ["use_reg_head","use_kl_loss","use_diou_loss","use_attr_loss",
            "use_vote_weight","use_answer","use_mlm","debug"]:
       if not hasattr(args, _flag):
           setattr(args, _flag, False)
    assert args.lang_num_max == 1, 'lang max num == 1; avoid bugs'

    if args.reference: eval_ref(args)