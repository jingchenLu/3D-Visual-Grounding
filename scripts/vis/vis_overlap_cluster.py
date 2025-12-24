import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg') # 后台绘图，防止卡顿
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
import time

def cosine_sim_matrix(a, b, eps=1e-8):
    a_n = a / (np.linalg.norm(a, axis=-1, keepdims=True) + eps)
    b_n = b / (np.linalg.norm(b, axis=-1, keepdims=True) + eps)
    return a_n @ b_n.T

def process_epoch_data(filepath, max_samples=2000):
    print(f"正在加载文件: {filepath} ...")
    data = dict(np.load(filepath, allow_pickle=True))
    feat = data['feat']
    ious = data['ious']
    
    valid_mask = ious >= 0
    feat = feat[valid_mask]
    ious = ious[valid_mask]
    
    # 排序并采样
    sort_idx = np.argsort(ious)[::-1] 
    feat = feat[sort_idx]
    ious = ious[sort_idx]
    
    if len(feat) > max_samples:
        feat = feat[:max_samples]
        ious = ious[:max_samples]
        
    return feat, ious, int(data['epoch'])

def plot_tsne_enhanced(ax, feat, ious, title):
    if feat.shape[0] < 5:
        return

    print(f"  正在计算 t-SNE ({title})...")
    # 降低 perplexity 以获得更紧凑的簇
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, init='pca', learning_rate='auto', n_jobs=-1)
    feat_2d = tsne.fit_transform(feat)
    
    # 绘图
    # 颜色映射：Spectral_r (红=高IoU, 蓝=低IoU)
    sc = ax.scatter(feat_2d[:, 0], feat_2d[:, 1], c=ious, cmap='Spectral_r', 
                    s=50, alpha=0.7, edgecolors='none')
    
    # --- 增强提示部分 ---
    # 1. 自动找到 High IoU 点的中心，画个圈或者写个字
    high_iou_idx = ious > 0.5
    if np.sum(high_iou_idx) > 0:
        center = np.mean(feat_2d[high_iou_idx], axis=0)
        # 添加文本指向高质量区域
        ax.annotate('High Quality Proposals\n(Correct Instances)', 
                    xy=(center[0], center[1]), xytext=(center[0]+5, center[1]+5),
                    arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5),
                    fontsize=10, fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.8))

    ax.set_title(title, fontsize=14, fontweight='bold')
    # 去掉坐标轴刻度，因为t-SNE坐标绝对值无意义，相对距离有意义
    ax.set_xticks([])
    ax.set_yticks([])
    
    # 添加简单的文字说明框
    desc = "Points closer = Similar Features\nRed = High IoU (Match)\nBlue = Low IoU (Noise)"
    ax.text(0.02, 0.02, desc, transform=ax.transAxes, fontsize=9,
            verticalalignment='bottom', bbox=dict(boxstyle="round", fc="white", alpha=0.8))
    
    return sc

def plot_sim_distribution_clean(ax, feat, ious, iou_thresh=0.5, title=""):
    print(f"  正在计算分布 ({title})...")
    pos_mask = ious > iou_thresh
    neg_mask = ious <= iou_thresh
    
    feat_pos = feat[pos_mask]
    feat_neg = feat[neg_mask]
    
    if len(feat_pos) < 2:
        return

    # 1. 正-正 相似度 (Intra-Class)
    sim_pos_pos = cosine_sim_matrix(feat_pos, feat_pos)
    upper_indices = np.triu_indices_from(sim_pos_pos, k=1)
    sim_pos_vals = sim_pos_pos[upper_indices]
    
    # 2. 正-负 相似度 (Inter-Class)
    sim_neg_vals = []
    if len(feat_neg) > 0:
        sim_pos_neg = cosine_sim_matrix(feat_pos, feat_neg)
        sim_neg_vals = sim_pos_neg.flatten()
        if len(sim_neg_vals) > 10000:
             sim_neg_vals = np.random.choice(sim_neg_vals, 10000, replace=False)

    # 绘图：填充曲线，更美观
    sns.kdeplot(sim_pos_vals, ax=ax, fill=True, color='#e74c3c', alpha=0.5, linewidth=2, label='Pos-Pos Similarity\n(Consistency)')
    if len(sim_neg_vals) > 0:
        sns.kdeplot(sim_neg_vals, ax=ax, fill=True, color='#3498db', alpha=0.5, linewidth=2, label='Pos-Neg Similarity\n(Separability)')
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel("Cosine Similarity", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_xlim(-0.5, 1.05)
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.legend(loc='upper left', fontsize=10)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ep50", required=True)
    parser.add_argument("--ep100", required=True)
    parser.add_argument("--out", default="contrastive_vis_final.png")
    args = parser.parse_args()

    f50, i50, ep50_num = process_epoch_data(args.ep50, max_samples=2000)
    f100, i100, ep100_num = process_epoch_data(args.ep100, max_samples=2000)

    # 创建 2x2 布局
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    plt.subplots_adjust(wspace=0.25, hspace=0.3)

    # Row 1: Epoch 50
    sc = plot_tsne_enhanced(axes[0, 0], f50, i50, f"Feature Space @ Epoch {ep50_num}\n(Early Training)")
    plot_sim_distribution_clean(axes[0, 1], f50, i50, title=f"Similarity Distribution @ Epoch {ep50_num}")

    # Row 2: Epoch 100
    sc = plot_tsne_enhanced(axes[1, 0], f100, i100, f"Feature Space @ Epoch {ep100_num}\n(Converged)")
    plot_sim_distribution_clean(axes[1, 1], f100, i100, title=f"Similarity Distribution @ Epoch {ep100_num}")

    # 公共 Colorbar
    cbar_ax = fig.add_axes([0.05, 0.05, 0.4, 0.02]) # 位置：左下角横条
    cb = fig.colorbar(sc, cax=cbar_ax, orientation='horizontal')
    cb.set_label("IoU with Ground Truth (Red=High, Blue=Low)", fontsize=12)

    plt.suptitle("Evolution of Feature Representations via Intra-Modal Contrastive Learning", fontsize=18, y=0.95)
    
    print(f"正在保存最终可视化图到 {args.out} ...")
    plt.savefig(args.out, dpi=200, bbox_inches='tight')
    print("完成！")

if __name__ == "__main__":
    main()