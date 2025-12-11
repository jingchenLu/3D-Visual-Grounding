#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
from glob import glob
from tensorboard.backend.event_processing import event_accumulator

# 你可以在这里按需增删 tag
DEFAULT_TAGS = [
    # loss
    "loss/ref_loss",
    "loss/box_loss",
    "loss/objectness_loss",
    "loss/con_loss",
    "loss/lang_loss",
    "loss/diou_loss",

    # grounding score
    "score/ref_acc",
    "score/top_iou_rate_0.25",
    "score/top_iou_rate_0.5",
    "score/pred_iou_rate_0.25",
    "score/pred_iou_rate_0.5",

    # 正负样本比例
    "score/pos_ratio",
    "score/neg_ratio",
]

def find_event_file(logdir):
    """
    在 logdir 下找到一个 event 文件（默认选最新的那个）
    """
    pattern = os.path.join(logdir, "**", "events.*")
    files = glob(pattern, recursive=True)
    if not files:
        raise FileNotFoundError(f"No event files found under {logdir}")
    # 按修改时间排序，选最新的
    files.sort(key=os.path.getmtime)
    return files[-1]

def load_scalars(event_path):
    """
    用 event_accumulator 读取所有 scalar
    """
    ea = event_accumulator.EventAccumulator(
        event_path,
        size_guidance={  # 只关心 scalar
            event_accumulator.SCALARS: 0,
        }
    )
    ea.Reload()
    scalar_tags = ea.Tags().get('scalars', [])
    data = {}
    for tag in scalar_tags:
        events = ea.Scalars(tag)
        # events: list of ScalarEvent(wall_time, step, value)
        steps = [e.step for e in events]
        vals = [e.value for e in events]
        data[tag] = (steps, vals)
    return data

def summarize_tag(tag, steps, vals):
    """
    给定一个 tag 的曲线，输出：最后值 / 最小值 / 最大值
    """
    if not steps:
        return f"  {tag}: (no data)"

    last_step, last_val = steps[-1], vals[-1]
    min_val = min(vals)
    max_val = max(vals)
    min_step = steps[vals.index(min_val)]
    max_step = steps[vals.index(max_val)]

    return (
        f"  {tag}:\n"
        f"    last   = {last_val:.6f} (step {last_step})\n"
        f"    min    = {min_val:.6f} (step {min_step})\n"
        f"    max    = {max_val:.6f} (step {max_step})"
    )

def main():
    parser = argparse.ArgumentParser(
        description="Summarize key TensorBoard scalars for a given logdir."
    )
    parser.add_argument(
        "--logdir",
        type=str,
        required=True,
        help="Path to a TensorBoard log directory (e.g., .../tensorboard/train)",
    )
    parser.add_argument(
        "--tags",
        type=str,
        nargs="*",
        default=None,
        help="Optional list of scalar tags to summarize. "
             "If not set, use a built-in default list."
    )
    args = parser.parse_args()

    tags = args.tags if args.tags else DEFAULT_TAGS

    event_path = find_event_file(args.logdir)
    print(f"[INFO] Using event file: {event_path}")

    data = load_scalars(event_path)

    print("\n[SUMMARY]")
    for tag in tags:
        if tag not in data:
            print(f"  {tag}: (not found in this run)")
            continue
        steps, vals = data[tag]
        print(summarize_tag(tag, steps, vals))

if __name__ == "__main__":
    main()
