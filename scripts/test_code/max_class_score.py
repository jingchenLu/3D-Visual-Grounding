import os
import argparse
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def load_event_accumulator(logdir):
    """
    读取某个 tensorboard 日志目录（例如 .../tensorboard/val）
    """
    ea = EventAccumulator(logdir)
    ea.Reload()
    return ea

def get_max_scalar(ea, prefix="score/class_"):
    """
    从 EventAccumulator 中筛选 tag，以 prefix 开头的所有 scalar，
    计算每个 tag 的最大值及其 step。
    """
    tags = ea.Tags().get("scalars", [])
    result = {}

    for tag in tags:
        if not tag.startswith(prefix):
            continue

        events = ea.Scalars(tag)
        if not events:
            continue

        # 找到最大值
        max_event = max(events, key=lambda e: e.value)
        result[tag] = {
            "max_value": max_event.value,
            "step": max_event.step,
            "wall_time": max_event.wall_time,
        }

    return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--logdir",
        type=str,
        required=True,
        help="tensorboard 日志目录，例如: outputs/exp_joint/2025-11-27_14-54-41/tensorboard/val",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="score/class_",
        help="要筛选的 tag 前缀，默认 score/class_",
    )
    args = parser.parse_args()

    assert os.path.isdir(args.logdir), f"Logdir not found: {args.logdir}"

    print(f"Loading tensorboard logs from: {args.logdir}")
    ea = load_event_accumulator(args.logdir)
    max_stats = get_max_scalar(ea, prefix=args.prefix)

    # 按数值从大到小排序
    sorted_items = sorted(max_stats.items(), key=lambda kv: kv[1]["max_value"], reverse=True)

    print("\n==== Max values for tags starting with '{}' ====\n".format(args.prefix))
    for tag, info in sorted_items:
        print(f"{tag:40s}  max={info['max_value']:.4f}  step={info['step']}")

if __name__ == "__main__":
    main()
