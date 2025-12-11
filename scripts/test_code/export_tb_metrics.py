#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import re
from typing import List, Dict

import pandas as pd
from tensorboard.backend.event_processing import event_accumulator


def load_scalars(logdir: str) -> Dict[str, pd.DataFrame]:
    """
    从 TensorBoard logdir 中读取所有 scalar，返回 {tag: DataFrame(step, value)}。
    """
    ea = event_accumulator.EventAccumulator(
        logdir,
        size_guidance={event_accumulator.SCALARS: 0},
    )
    ea.Reload()

    scalar_tags = ea.Tags().get("scalars", [])
    print(f"[INFO] 在 {logdir} 中发现 {len(scalar_tags)} 个 scalar tag：")
    for t in scalar_tags:
        print("  -", t)

    tag2df = {}
    for tag in scalar_tags:
        events = ea.Scalars(tag)
        steps = [e.step for e in events]
        values = [e.value for e in events]
        df = pd.DataFrame({"step": steps, tag: values})
        tag2df[tag] = df

    return tag2df


def select_tags(tag2df: Dict[str, pd.DataFrame],
                tags: List[str],
                pattern: str) -> List[str]:
    """
    根据明确 tag 列表或正则 pattern 过滤 tag。
    """
    all_tags = list(tag2df.keys())
    selected = []

    if tags:
        for t in tags:
            if t in tag2df:
                selected.append(t)
            else:
                print(f"[WARN] tag '{t}' 不在 TensorBoard 中，跳过。")

    if pattern:
        regex = re.compile(pattern)
        for t in all_tags:
            if regex.search(t) and t not in selected:
                selected.append(t)

    if not selected:
        print("[WARN] 没有匹配到任何 tag，请检查 --tags 或 --pattern。")
    else:
        print("[INFO] 选中的 tag：")
        for t in selected:
            print("  -", t)

    return selected


def merge_and_save(tag2df: Dict[str, pd.DataFrame],
                   selected_tags: List[str],
                   save_csv: str,
                   preview: int = 5):
    """
    把选中的 tag 合并成一个 DataFrame（按 step 对齐）并保存 CSV。
    同时在终端打印每个 tag 的前/后几行，方便查看变化。
    """
    merged = None
    for tag in selected_tags:
        df = tag2df[tag].copy()
        safe_tag = tag.replace("/", "_")
        df = df.rename(columns={tag: safe_tag})
        if merged is None:
            merged = df
        else:
            merged = pd.merge(merged, df, on="step", how="outer")

    if merged is None:
        print("[ERROR] 没有数据可保存。")
        return

    merged = merged.sort_values("step")
    merged.to_csv(save_csv, index=False)
    print(f"[INFO] 已保存 CSV 到: {save_csv}")

    # 打印每个指标前 / 后几行，方便你直接复制回来给我
    print("\n[PREVIEW] 每个指标开头和结尾几条记录：")
    for col in merged.columns:
        if col == "step":
            continue
        print(f"\n==== {col} ====")
        print("前几条：")
        print(merged[["step", col]].head(preview))
        print("后几条：")
        print(merged[["step", col]].tail(preview))


def main():
    parser = argparse.ArgumentParser(
        description="从 TensorBoard logdir 中提取指定 scalar 指标，导出 CSV 并打印预览。"
    )
    parser.add_argument("--logdir", type=str, required=True,
                        help="TensorBoard 日志目录（例如 .../tensorboard/val）")
    parser.add_argument("--save_csv", type=str, required=True,
                        help="导出指标的 CSV 路径")
    parser.add_argument("--tags", type=str, nargs="*", default=None,
                        help="要精确匹配的 tag 名称列表（可选）")
    parser.add_argument("--pattern", type=str, default=None,
                        help="按正则匹配 tag 名称（可选，如 'score/class_iou_rate_.*'）")
    parser.add_argument("--preview", type=int, default=5,
                        help="每个指标打印前/后多少条记录")
    args = parser.parse_args()

    tag2df = load_scalars(args.logdir)
    selected = select_tags(tag2df, args.tags or [], args.pattern)
    if not selected:
        return
    merge_and_save(tag2df, selected, args.save_csv, preview=args.preview)


if __name__ == "__main__":
    main()
