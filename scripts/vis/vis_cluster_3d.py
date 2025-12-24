import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from matplotlib.lines import Line2D

# 12 edges of a box, given 8 corners indexing (common order)
EDGES = [
    (0,1),(1,2),(2,3),(3,0),
    (4,5),(5,6),(6,7),(7,4),
    (0,4),(1,5),(2,6),(3,7),
]

def points_in_aabb(pc, center, size):
    c = center.reshape(1,3)
    s = size.reshape(1,3)
    lo = c - s/2
    hi = c + s/2
    mask = (pc >= lo) & (pc <= hi)
    return mask.all(axis=1)

def make_gt_corners(center, size):
    cx, cy, cz = center
    sx, sy, sz = size
    # axis-aligned corners
    x0, x1 = cx - sx/2, cx + sx/2
    y0, y1 = cy - sy/2, cy + sy/2
    z0, z1 = cz - sz/2, cz + sz/2
    corners = np.array([
        [x0,y0,z0],[x1,y0,z0],[x1,y1,z0],[x0,y1,z0],
        [x0,y0,z1],[x1,y0,z1],[x1,y1,z1],[x0,y1,z1],
    ], dtype=np.float32)
    return corners

def draw_box(ax, corners, color="C0", linewidth=1.0, alpha=1.0):
    for a, b in EDGES:
        xs = [corners[a, 0], corners[b, 0]]
        ys = [corners[a, 1], corners[b, 1]]
        zs = [corners[a, 2], corners[b, 2]]
        ax.plot(xs, ys, zs, color=color, linewidth=linewidth, alpha=alpha)

def set_equal_xyz(ax, pts):
    # pts: (M,3)
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    center = (mins + maxs) / 2
    radius = (maxs - mins).max() / 2
    ax.set_xlim(center[0]-radius, center[0]+radius)
    ax.set_ylim(center[1]-radius, center[1]+radius)
    ax.set_zlim(center[2]-radius, center[2]+radius)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True, help="dump file, e.g. ep050.npz")
    ap.add_argument("--out", default="cluster_3d.png")
    ap.add_argument("--azim", type=float, default=-60)
    ap.add_argument("--elev", type=float, default=18)
    ap.add_argument("--max_boxes", type=int, default=20, help="max cluster boxes to draw")
    args = ap.parse_args()

    d = dict(np.load(args.npz, allow_pickle=True))
    epoch = int(d["epoch"][0])
    corners = d["pred_corner"]          # (N,8,3)
    ious = d["ious"]                    # (N,)
    cidx = d["cluster_idx"]             # (M,)
    gt_center = d["gt_center"]
    gt_size = d["gt_size"]
    gt_corners = make_gt_corners(gt_center, gt_size)

    # sort cluster by IoU, draw top ones
    cidx = np.array(cidx, dtype=np.int64)
    if cidx.size > 0:
        order = np.argsort(-ious[cidx])
        cidx = cidx[order][:args.max_boxes]
    else:
        cidx = np.array([], dtype=np.int64)

    fig = plt.figure(figsize=(8,7))
    ax = fig.add_subplot(111, projection="3d")

    #  先确定 best_k（cluster 内 IoU 最大的那个）
    best_k = None
    if len(cidx) > 0:
        best_k = int(cidx[0])  # 因为已经按 IoU 降序排过了

    pts_all = [gt_corners.reshape(-1, 3)]

    # 画 “其它簇内框” 用浅蓝
    for k in cidx:
        k = int(k)
        if best_k is not None and k == best_k:
            continue
        lw = 0.8 + 2.5 * float(ious[k])
        draw_box(ax, corners[k], color="tab:blue", linewidth=lw, alpha=0.25)
        pts_all.append(corners[k].reshape(-1, 3))

    #  画 best proposal 用绿色（更粗更亮）
    if best_k is not None:
        lw = 1.6 + 3.2 * float(ious[best_k])
        draw_box(ax, corners[best_k], color="tab:green", linewidth=lw, alpha=0.95)
        pts_all.append(corners[best_k].reshape(-1, 3))

    # 画 GT 用红色
    draw_box(ax, gt_corners, color="tab:red", linewidth=3.0, alpha=1.0)

    # ✅ legend：三种颜色分别对应三类
    handles = [
        Line2D([0], [0], color="tab:red",   lw=3.0, label="GT box"),
        Line2D([0], [0], color="tab:green", lw=2.5, label="Best proposal (max IoU)"),
        Line2D([0], [0], color="tab:blue",  lw=2.0, label="Other cluster proposals"),
    ]
    ax.legend(handles=handles, loc="upper left")


    pts_all = np.concatenate(pts_all, axis=0)
    set_equal_xyz(ax, pts_all)

    ax.view_init(elev=args.elev, azim=args.azim)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    
    max_iou = float(ious[cidx].max()) if len(cidx) > 0 else 0.0
    ax.set_title(f"3D cluster @ epoch {epoch} (GT + top {len(cidx)}), maxIoU={max_iou:.3f}")

    pc = d.get("pc_xyz", None)
    if pc is not None:
        pc = np.asarray(pc)
        m = points_in_aabb(pc, gt_center, gt_size * 0.9)
        obj_pts = pc[m]
        # 如果太多点，继续下采样
        if obj_pts.shape[0] > 3000:
            obj_pts = obj_pts[np.random.choice(obj_pts.shape[0], 3000, replace=False)]
        ax.scatter(obj_pts[:,0], obj_pts[:,1], obj_pts[:,2], s=2, alpha=0.35)

    plt.tight_layout()
    plt.savefig(args.out, dpi=200)
    print("Saved:", args.out)

if __name__ == "__main__":
    main()
