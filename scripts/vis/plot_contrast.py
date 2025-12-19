import argparse, json, os
import matplotlib.pyplot as plt

def load_scalars(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    # data[tag] = [[step, value], ...]
    out = {}
    for k, pairs in data.items():
        if not pairs:
            continue
        steps = [p[0] for p in pairs]
        vals  = [p[1] for p in pairs]
        out[k] = (steps, vals)
    return out

def plot_tags(ax, scalars, tags, title):
    for t in tags:
        if t in scalars:
            x, y = scalars[t]
            ax.plot(x, y, label=t)
    ax.set_title(title)
    ax.grid(True)
    ax.legend()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_json", required=True, help=".../tensorboard/train/all_scalars.json")
    ap.add_argument("--val_json", required=False, help=".../tensorboard/val/all_scalars.json")
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    train = load_scalars(args.train_json)
    val = load_scalars(args.val_json) if args.val_json else None

    # 1) 对比学习诊断：gap/corr/pos/neg
    fig = plt.figure(figsize=(12,8))
    ax1 = fig.add_subplot(2,1,1)
    plot_tags(ax1, train,
              ["score/con_gap", "score/con_corr", "score/con_pos_sim", "score/con_neg_sim"],
              "Contrast Diagnostics (train)")
    ax2 = fig.add_subplot(2,1,2)
    plot_tags(ax2, train,
              ["loss/con_loss", "loss/lang_con_loss", "loss/iou_con_loss"],
              "Contrast Loss Breakdown (train)")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "contrast_train.png"), dpi=200)
    plt.close(fig)

    # 2) 课程学习曲线：thr/gamma/tau/weight
    fig = plt.figure(figsize=(12,6))
    ax = fig.add_subplot(1,1,1)
    plot_tags(ax, train,
              ["score/con_w", "score/con_thr", "score/con_gamma", "score/con_min_tau"],
              "Curriculum Schedule (train)")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "contrast_curriculum.png"), dpi=200)
    plt.close(fig)

    # 3) 主指标对齐：ref_acc / IoU@0.25/0.5 vs con_gap
    fig = plt.figure(figsize=(12,8))
    ax1 = fig.add_subplot(2,1,1)
    plot_tags(ax1, train, ["score/ref_acc", "score/iou_rate_0.5", "score/iou_rate_0.25"], "Grounding Metrics (train)")
    ax2 = fig.add_subplot(2,1,2)
    plot_tags(ax2, train, ["score/con_gap", "score/con_corr"], "Alignment Evidence (train)")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "metrics_vs_contrast_train.png"), dpi=200)
    plt.close(fig)

    if val:
        fig = plt.figure(figsize=(12,6))
        ax = fig.add_subplot(1,1,1)
        plot_tags(ax, val, ["score/ref_acc", "score/iou_rate_0.5", "score/con_gap", "score/con_corr"], "Val Metrics & Contrast Evidence")
        fig.tight_layout()
        fig.savefig(os.path.join(args.out_dir, "val_metrics_vs_contrast.png"), dpi=200)
        plt.close(fig)

    print("Saved figures to:", args.out_dir)

if __name__ == "__main__":
    main()
