import argparse
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def load_scalars(logdir: str):
    ea = EventAccumulator(logdir, size_guidance={"scalars": 0})
    ea.Reload()
    tags = ea.Tags().get("scalars", [])
    return ea, tags

def best_of_tag(ea, tag):
    events = ea.Scalars(tag)
    if not events:
        return None
    best = max(events, key=lambda e: e.value)
    return best.step, best.value

def last_of_tag(ea, tag):
    events = ea.Scalars(tag)
    if not events:
        return None
    e = events[-1]
    return e.step, e.value

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--logdir", required=True, help="tensorboard log dir, e.g. .../tensorboard/val")
    ap.add_argument("--mode", default="best", choices=["best", "last"])
    ap.add_argument("--prefix", default="score/", help="tag prefix, default score/")
    args = ap.parse_args()

    ea, tags = load_scalars(args.logdir)

    keys = [
        "iou_rate_0.25", "iou_rate_0.5",
        "max_iou_rate_0.25", "max_iou_rate_0.5",
        "pred_iou_rate_0.25", "pred_iou_rate_0.5",
        "top_iou_rate_1", "top_iou_rate_2", "top_iou_rate_3", "top_iou_rate_4", "top_iou_rate_5",
        "ref_acc"
    ]

    print(f"Found {len(tags)} scalar tags.")
    for k in keys:
        tag = args.prefix + k
        if tag not in tags:
            print(f"[MISS] {tag}")
            continue
        step, val = best_of_tag(ea, tag) if args.mode == "best" else last_of_tag(ea, tag)
        print(f"[{args.mode.upper()}] {tag:28s}  step={step:<8d} value={val:.6f}")
