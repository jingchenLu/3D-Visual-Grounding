#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
将 ReferIt3D 的 nr3d.csv / sr3d.csv 转成 ScanRefer 风格的 JSON：
    NR3D_filtered_{train,val,test}.json
    SR3D_filtered_{train,val,test}.json

用法示例：
    python scripts/convert_referit3d_to_scanrefer.py \
        --data_path data/ \
        --out_dir data/ScanRefer/
"""

import os
import csv
import json
import argparse

# 直接复用 joint_det_dataset 里的文本清洗（可选）
try:
    from joint_det_dataset import Scene_graph_parse
    _HAS_SNG_PARSE = True
except Exception:
    # 如果你不想依赖 sng_parser，可以把它关掉，直接用原始 utterance
    print("[WARN] 无法导入 joint_det_dataset.Scene_graph_parse，将不进行文本解耦，只用原始句子。")
    _HAS_SNG_PARSE = False


def _load_scan_ids(dataset: str, split: str, meta_root: str):
    """
    读取 data/meta_data/{dataset}_{split}_scans.txt 里的 scan_id 列表。
    dataset: 'nr3d' or 'sr3d'
    split: 'train' / 'val' / 'test'
    """
    # 按 ReferIt3D 的写法：val 使用 *_test_scans.txt
    split_for_file = 'test' if split in ['val', 'test'] else 'train'
    meta_path = os.path.join(meta_root, f"{dataset}_{split_for_file}_scans.txt")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"找不到扫描列表文件: {meta_path}")
    with open(meta_path, "r") as f:
        scan_ids = set(eval(f.read()))
    return scan_ids, split_for_file


def load_nr3d_annos(split: str, data_path: str, meta_root: str):
    """
    仿照 joint_det_dataset.load_nr3d_annos 的逻辑：
      - train: 使用 nr3d_train_scans.txt，全部样本；
      - val / test: 使用 nr3d_test_scans.txt，且只保留 correct_guess == True。
    返回一个 anno 列表，每个元素带字段：
      scan_id, target_id, target, utterance, dataset='nr3d'
    """
    scan_ids, split_for_file = _load_scan_ids("nr3d", split, meta_root)

    csv_path = os.path.join(data_path, "ReferIt3D", "nr3d.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"找不到 nr3d.csv: {csv_path}")

    annos = []
    with open(csv_path, "r", newline='') as f:
        csv_reader = csv.reader(f)
        headers = next(csv_reader)
        headers = {header: h for h, header in enumerate(headers)}

        for line in csv_reader:
            scan_id = line[headers["scan_id"]]
            if scan_id not in scan_ids:
                continue

            # 按 BUTD-DETR 的实现：test/val 只保留 correct_guess == True
            if split_for_file == "test":
                correct_guess = str(line[headers["correct_guess"]]).lower()
                if correct_guess != "true":
                    continue

            anno = {
                "scan_id": scan_id,
                "target_id": int(line[headers["target_id"]]),
                "target": line[headers["instance_type"]],
                "utterance": line[headers["utterance"]],
                "anchor_ids": [],
                "anchors": [],
                "dataset": "nr3d",
            }
            annos.append(anno)

    # 文本解耦（可选）
    if _HAS_SNG_PARSE:
        Scene_graph_parse(annos)

    return annos


def load_sr3d_annos(split: str, data_path: str, meta_root: str, dset_name: str = "sr3d"):
    """
    仿照 joint_det_dataset.load_sr3d_annos：
      - train: sr3d_train_scans.txt
      - val/test: sr3d_test_scans.txt
      - 必须 mentions_target_class == True
    dset_name: 'sr3d' 或 'sr3d+' （默认 sr3d）
    返回字段：
      scan_id, target_id, target, utterance, dataset='sr3d' or 'sr3d+'
    """
    scan_ids, split_for_file = _load_scan_ids("sr3d", split, meta_root)

    csv_path = os.path.join(data_path, "ReferIt3D", f"{dset_name}.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"找不到 {dset_name}.csv: {csv_path}")

    annos = []
    with open(csv_path, "r", newline='') as f:
        csv_reader = csv.reader(f)
        headers = next(csv_reader)
        headers = {header: h for h, header in enumerate(headers)}

        for line in csv_reader:
            scan_id = line[headers["scan_id"]]
            if scan_id not in scan_ids:
                continue

            mentions_target_class = str(line[headers["mentions_target_class"]]).lower()
            if mentions_target_class != "true":
                continue

            anno = {
                "scan_id": scan_id,
                "target_id": int(line[headers["target_id"]]),
                "distractor_ids": eval(line[headers["distractor_ids"]]),
                "utterance": line[headers["utterance"]],
                "target": line[headers["instance_type"]],
                "anchors": eval(line[headers["anchors_types"]]),
                "anchor_ids": eval(line[headers["anchor_ids"]]),
                "dataset": dset_name,
            }
            annos.append(anno)

    if _HAS_SNG_PARSE:
        Scene_graph_parse(annos)

    return annos


def annos_to_scanrefer_json(annos, out_path):
    """
    把 ReferIt3D 的 annos 列表，转换成 ScanRefer 风格 json：
      {
        "scene_id": str,
        "object_id": int,
        "ann_id": int,
        "description": str,
        "token": [str, ...],
        "object_name": str
      }
    """
    out_list = []
    for ann_id, anno in enumerate(annos):
        desc = anno["utterance"]
        tokens = desc.split()
        item = {
            "scene_id": anno["scan_id"],
            "object_id": anno["target_id"],
            "ann_id": ann_id,
            "description": desc,
            "token": tokens,
            "object_name": anno["target"],
        }
        out_list.append(item)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_list, f)
    print(f"[OK] 写出 {out_path}，共 {len(out_list)} 条样本。")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/",
        help="数据根目录，里面包含 ReferIt3D/ 和 meta_data/ 等",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="data/ScanRefer/",
        help="输出 json 的目录，例如 data/ScanRefer/",
    )
    parser.add_argument(
        "--sr3d_name",
        type=str,
        default="sr3d",
        help="SR3D 对应的 csv 名字（sr3d 或 sr3d+），默认为 sr3d",
    )
    args = parser.parse_args()

    data_path = args.data_path
    meta_root = os.path.join("data", "meta_data")  # 和原项目保持一致
    out_dir = args.out_dir
    sr3d_name = args.sr3d_name

    splits = ["train", "val", "test"]

    # ===== NR3D =====
    for split in splits:
        print(f"\n[NR3D] 处理 split = {split} ...")
        annos = load_nr3d_annos(split, data_path, meta_root)
        out_path = os.path.join(out_dir, f"NR3D_filtered_{split}.json")
        annos_to_scanrefer_json(annos, out_path)

    # ===== SR3D / SR3D+ =====
    for split in splits:
        print(f"\n[SR3D] 处理 split = {split} (dset={sr3d_name}) ...")
        annos = load_sr3d_annos(split, data_path, meta_root, dset_name=sr3d_name)
        # 文件名里我用 SR3D，以区分 ScanRefer
        prefix = "SR3Dplus" if sr3d_name == "sr3d+" else "SR3D"
        out_path = os.path.join(out_dir, f"{prefix}_filtered_{split}.json")
        annos_to_scanrefer_json(annos, out_path)


if __name__ == "__main__":
    main()
