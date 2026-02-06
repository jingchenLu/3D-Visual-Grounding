import argparse
import json
import os
import sys
import numpy as np
import torch
import open3d as o3d

# ====== 保证可从项目根目录导入 ======
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
if _ROOT_DIR not in sys.path:
    sys.path.append(_ROOT_DIR)

# ====== 按你项目实际路径修改这些 import（你原文件里已能跑通就别改）======
from models.jointnet.jointnet import JointNet
from data.scannet.model_util_scannet import ScannetDatasetConfig
from transformers import BertTokenizer, DistilBertTokenizer

# 训练 dataset.py 里颜色归一化： (rgb - MEAN_COLOR_RGB) / 256
MEAN_COLOR_RGB = np.array([109.8, 97.2, 83.8], dtype=np.float32)


def write_ply(verts, colors, indices, output_file):
    if colors is None:
        colors = np.zeros_like(verts)
    if indices is None:
        indices = []

    with open(output_file, "w") as f:
        f.write("ply \n")
        f.write("format ascii 1.0\n")
        f.write("element vertex {:d}\n".format(len(verts)))
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("element face {:d}\n".format(len(indices)))
        f.write("property list uchar uint vertex_indices\n")
        f.write("end_header\n")
        for vert, color in zip(verts, colors):
            f.write(
                "{:f} {:f} {:f} {:d} {:d} {:d}\n".format(
                    vert[0], vert[1], vert[2],
                    int(color[0] * 255), int(color[1] * 255), int(color[2] * 255)
                )
            )
        for ind in indices:
            f.write("3 {:d} {:d} {:d}\n".format(ind[0], ind[1], ind[2]))


def write_bbox_wireframe(bbox, mode, output_file, radius=0.003):
    import math

    def compute_length_vec3(vec3):
        return math.sqrt(vec3[0]*vec3[0] + vec3[1]*vec3[1] + vec3[2]*vec3[2])

    def rotation(axis, angle):
        rot = np.eye(4)
        c = np.cos(-angle)
        s = np.sin(-angle)
        t = 1.0 - c
        axis = axis / (compute_length_vec3(axis) + 1e-12)
        x, y, z = axis[0], axis[1], axis[2]
        rot[0, 0] = 1 + t*(x*x-1)
        rot[0, 1] = z*s + t*x*y
        rot[0, 2] = -y*s + t*x*z
        rot[1, 0] = -z*s + t*x*y
        rot[1, 1] = 1 + t*(y*y-1)
        rot[1, 2] = x*s + t*y*z
        rot[2, 0] = y*s + t*x*z
        rot[2, 1] = -x*s + t*y*z
        rot[2, 2] = 1 + t*(z*z-1)
        return rot

    def create_cylinder_mesh(radius, p0, p1, stacks=10, slices=10):
        verts = []
        indices = []
        diff = (p1 - p0).astype(np.float32)
        height = compute_length_vec3(diff)
        if height < 1e-8:
            return [], []

        for i in range(stacks + 1):
            for j in range(slices):
                theta = j * 2.0 * math.pi / slices
                pos = np.array(
                    [radius*math.cos(theta), radius*math.sin(theta), height*i/stacks]
                )
                verts.append(pos)

        for i in range(stacks):
            for j in range(slices):
                jp1 = int(math.fmod(j + 1, slices))
                indices.append(
                    np.array([(i + 1)*slices + j, i*slices + j, i*slices + jp1], dtype=np.uint32)
                )
                indices.append(
                    np.array([(i + 1)*slices + j, i*slices + jp1, (i + 1)*slices + jp1], dtype=np.uint32)
                )

        transform = np.eye(4)
        va = np.array([0, 0, 1], dtype=np.float32)
        vb = diff / (height + 1e-12)
        axis = np.cross(vb, va)
        angle = np.arccos(np.clip(np.dot(va, vb), -1, 1))

        if angle != 0:
            if compute_length_vec3(axis) == 0:
                dotx = va[0]
                if math.fabs(dotx) != 1.0:
                    axis = np.array([1, 0, 0], dtype=np.float32) - dotx * va
                else:
                    axis = np.array([0, 1, 0], dtype=np.float32) - va[1] * va
                axis = axis / (compute_length_vec3(axis) + 1e-12)
            transform = rotation(axis, -angle)

        transform[:3, 3] += p0
        verts = [np.dot(transform, np.array([v[0], v[1], v[2], 1.0])) for v in verts]
        verts = [np.array([v[0], v[1], v[2]]) / v[3] for v in verts]
        return verts, indices

    def get_bbox_corners(bbox):
        centers, lengths = bbox[:3], bbox[3:6]
        xmin, xmax = centers[0] - lengths[0]/2, centers[0] + lengths[0]/2
        ymin, ymax = centers[1] - lengths[1]/2, centers[1] + lengths[1]/2
        zmin, zmax = centers[2] - lengths[2]/2, centers[2] + lengths[2]/2
        corners = np.array([
            [xmax, ymax, zmax],
            [xmax, ymax, zmin],
            [xmin, ymax, zmin],
            [xmin, ymax, zmax],
            [xmax, ymin, zmax],
            [xmax, ymin, zmin],
            [xmin, ymin, zmin],
            [xmin, ymin, zmax],
        ], dtype=np.float32)
        return corners

    def get_bbox_edges(box_min, box_max):
        v = [
            np.array([box_min[0], box_min[1], box_min[2]]),
            np.array([box_max[0], box_min[1], box_min[2]]),
            np.array([box_max[0], box_max[1], box_min[2]]),
            np.array([box_min[0], box_max[1], box_min[2]]),
            np.array([box_min[0], box_min[1], box_max[2]]),
            np.array([box_max[0], box_min[1], box_max[2]]),
            np.array([box_max[0], box_max[1], box_max[2]]),
            np.array([box_min[0], box_max[1], box_max[2]]),
        ]
        edges = [
            (v[0], v[1]), (v[1], v[2]), (v[2], v[3]), (v[3], v[0]),
            (v[4], v[5]), (v[5], v[6]), (v[6], v[7]), (v[7], v[4]),
            (v[0], v[4]), (v[1], v[5]), (v[2], v[6]), (v[3], v[7]),
        ]
        return edges

    palette = {
        0: [0, 255, 0],
        1: [0, 0, 255],
    }
    chosen = palette[int(mode)]

    corners = get_bbox_corners(np.asarray(bbox, dtype=np.float32))
    box_min = np.min(corners, axis=0)
    box_max = np.max(corners, axis=0)

    verts, indices, colors = [], [], []
    for p0, p1 in get_bbox_edges(box_min, box_max):
        cyl_verts, cyl_ind = create_cylinder_mesh(radius, p0, p1)
        base = len(verts)
        cyl_colors = [[c/255.0 for c in chosen] for _ in cyl_verts]
        cyl_ind = [x + base for x in cyl_ind]
        verts.extend(cyl_verts)
        indices.extend(cyl_ind)
        colors.extend(cyl_colors)

    write_ply(verts, colors, indices, output_file)


def export_topk_ply(proposals, out_dir, prefix, radius):
    if not proposals:
        return []
    os.makedirs(out_dir, exist_ok=True)
    saved = []
    for item in proposals:
        rank = int(item.get("rank", 0))
        pidx = int(item.get("proposal_idx", rank))
        if "ref_score_softmax" in item:
            score = float(item.get("ref_score_softmax", 0.0))
        else:
            score = float(item.get("objectness_prob", 0.0))

        cx, cy, cz = item["center_xyz_m"]
        lx, ly, lz = item["size_xyz_m"]
        r = float(item.get("heading_rad", 0.0))
        obb = np.array([cx, cy, cz, lx, ly, lz, r], dtype=np.float32)

        bbox_name = f"{prefix}_rank{rank:02d}_idx{pidx:03d}_score{score:.4f}.ply"
        bbox_out = os.path.join(out_dir, bbox_name)
        write_bbox_wireframe(obb, mode=1, output_file=bbox_out, radius=radius)
        saved.append(bbox_out)
    return saved


def random_sampling(pc: np.ndarray, num_points: int) -> np.ndarray:
    n = pc.shape[0]
    if n >= num_points:
        idx = np.random.choice(n, num_points, replace=False)
    else:
        idx = np.random.choice(n, num_points, replace=True)
    return pc[idx]


def load_xyzrgb_from_npy(npy_path: str):
    try:
        pc = np.load(npy_path, allow_pickle=False).astype(np.float32)
    except ValueError:
        pc = np.load(npy_path, allow_pickle=True).astype(np.float32)

    if pc.ndim != 2 or pc.shape[1] < 6:
        raise ValueError(f"Expect (N,6+) npy, got {pc.shape}")
    xyz = pc[:, :3].astype(np.float32)
    rgb = pc[:, 3:6].astype(np.float32)
    if rgb.max() <= 1.0:
        rgb = rgb * 255.0
    rgb = np.clip(rgb, 0.0, 255.0)
    return xyz, rgb


def load_xyzrgb_from_ply(ply_path: str):
    pcd = o3d.io.read_point_cloud(ply_path)
    if pcd.is_empty():
        raise ValueError(f"Empty point cloud: {ply_path}")
    xyz = np.asarray(pcd.points).astype(np.float32)
    if len(pcd.colors) == 0:
        raise ValueError("PLY 中没有颜色信息（colors 为空）")
    rgb = np.asarray(pcd.colors).astype(np.float32)
    if rgb.max() <= 1.0:
        rgb = rgb * 255.0
    rgb = np.clip(rgb, 0.0, 255.0)
    return xyz, rgb


def load_xyzrgb(path: str):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        return load_xyzrgb_from_npy(path)
    if ext == ".ply":
        return load_xyzrgb_from_ply(path)
    raise ValueError(f"Unsupported file type: {ext}. Use .npy or .ply")


def estimate_normals_open3d(
    xyz: np.ndarray,
    normal_radius: float = 0.03,
    normal_max_nn: int = 30,
    orient_consistent_k: int = 50,
):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz.astype(np.float64))

    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=float(normal_radius), max_nn=int(normal_max_nn)
        )
    )

    if orient_consistent_k and orient_consistent_k > 0:
        try:
            pcd.orient_normals_consistent_tangent_plane(int(orient_consistent_k))
        except Exception:
            pass

    normals = np.asarray(pcd.normals).astype(np.float32)
    n = np.linalg.norm(normals, axis=1, keepdims=True) + 1e-6
    normals = normals / n
    return normals


def build_point_clouds_tensor(
    xyz: np.ndarray,
    rgb_255: np.ndarray,
    normals: np.ndarray,
    use_color: bool = True,
    use_normal: bool = True,
    use_height: bool = False,
):
    feats = []

    if use_color:
        rgb_norm = (rgb_255.astype(np.float32) - MEAN_COLOR_RGB) / 256.0
        feats.append(rgb_norm)

    if use_normal:
        feats.append(normals.astype(np.float32))

    if len(feats) > 0:
        feat_all = np.concatenate(feats, axis=1).astype(np.float32)
        pc = np.concatenate([xyz.astype(np.float32), feat_all], axis=1).astype(np.float32)
    else:
        pc = xyz.astype(np.float32)

    if use_height:
        floor_height = np.percentile(xyz[:, 2], 0.99)
        height = (xyz[:, 2] - floor_height).reshape(-1, 1).astype(np.float32)
        pc = np.concatenate([pc, height], axis=1)

    return torch.from_numpy(pc[None, ...])  # (1, N, C)


def tokenize_query(query: str, use_distil: bool, max_len: int):
    if use_distil:
        tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    else:
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    tok = tokenizer(
        query,
        padding="max_length",
        truncation=True,
        max_length=max_len,
        return_tensors="pt",
    )

    input_ids = tok["input_ids"].unsqueeze(0)          # (1,1,L)
    attn_mask = tok["attention_mask"].unsqueeze(0)     # (1,1,L)
    token_type_ids = tok.get("token_type_ids", None)
    if token_type_ids is not None:
        token_type_ids = token_type_ids.unsqueeze(0)   # (1,1,L)

    return input_ids, attn_mask, token_type_ids


def fill_dataset_like_fields(data_dict: dict, device: torch.device):
    data_dict["istrain"] = torch.zeros(1, dtype=torch.long, device=device)
    data_dict["lang_num"] = torch.ones(1, dtype=torch.long, device=device)
    data_dict["dataset_idx"] = torch.zeros(1, dtype=torch.long, device=device)
    data_dict["scan_idx"] = torch.zeros(1, dtype=torch.long, device=device)
    return data_dict


def describe_position(center_xyz: np.ndarray, scene_xyz: np.ndarray):
    mn = scene_xyz.min(axis=0)
    mx = scene_xyz.max(axis=0)
    mid = (mn + mx) / 2.0
    ext = (mx - mn) + 1e-6
    rel = (center_xyz - mid) / ext

    def tag(v, name):
        if v < -0.15:
            return f"{name}偏小"
        if v > 0.15:
            return f"{name}偏大"
        return f"{name}居中"

    return f"相对场景范围：X {tag(rel[0],'X')}，Y {tag(rel[1],'Y')}，Z {tag(rel[2],'Z')}（坐标系为输入点云坐标）"


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npy", required=True, help="输入点云 .npy 或 .ply")
    ap.add_argument("--ckpt", required=True, help="训练好的 checkpoint（.pth 或 checkpoint.tar）")
    ap.add_argument("--query", required=True, help="英文查询语句，例如: 'the mug on the desk'")
    ap.add_argument("--num_points", type=int, default=40000)
    ap.add_argument("--num_proposals", type=int, default=256)

    # 输出控制
    ap.add_argument("--topk", type=int, default=8, help="输出 topK proposals（按 cluster_ref softmax 排序）")
    ap.add_argument("--out_json", type=str, default="", help="可选：把结果保存成 json 文件")
    ap.add_argument("--out_dir", type=str, default="", help="可选：输出 topk 的 ply/json 到目录")
    ap.add_argument("--ply_radius", type=float, default=0.003, help="PLY 线框粗细(米)")
    ap.add_argument("--save_scene_ply", action="store_true", help="若设置，将场景点云也写出为 scene_points.ply")

    # normals
    ap.add_argument("--normal_radius", type=float, default=0.03)
    ap.add_argument("--normal_max_nn", type=int, default=30)
    ap.add_argument("--orient_k", type=int, default=50)

    # tokenizer
    ap.add_argument("--use_distil", action="store_true")
    ap.add_argument("--bert_max_len", type=int, default=32)

    # model input flags (align with ground_eval)
    ap.add_argument("--use_color", action="store_true")
    ap.add_argument("--use_normal", action="store_true")
    ap.add_argument("--use_multiview", action="store_true")
    ap.add_argument("--no_height", action="store_true")
    ap.add_argument("--use_bidir", action="store_true")
    ap.add_argument("--use_con", action="store_true")
    ap.add_argument("--no_lang_cls", action="store_true")

    # 可选规整
    ap.add_argument("--voxel", type=float, default=0.0, help=">0 时 voxel downsample（米）")

    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ===== 1) 读点云 =====
    xyz, rgb = load_xyzrgb(args.npy)

    # voxel downsample（可选）
    if args.voxel and args.voxel > 0:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz.astype(np.float64))
        pcd.colors = o3d.utility.Vector3dVector((rgb / 255.0).astype(np.float64))
        pcd = pcd.voxel_down_sample(float(args.voxel))
        xyz = np.asarray(pcd.points).astype(np.float32)
        rgb = (np.asarray(pcd.colors).astype(np.float32) * 255.0)

    # ===== 2) normals（如需）=====
    if args.use_normal:
        normals = estimate_normals_open3d(
            xyz,
            normal_radius=args.normal_radius,
            normal_max_nn=args.normal_max_nn,
            orient_consistent_k=args.orient_k,
        )
    else:
        normals = np.zeros_like(xyz, dtype=np.float32)

    # ===== 3) 固定点数采样 =====
    pc_tmp = np.concatenate([xyz, rgb, normals], axis=1)
    pc_tmp = random_sampling(pc_tmp, args.num_points)
    xyz = pc_tmp[:, :3]
    rgb = pc_tmp[:, 3:6]
    normals = pc_tmp[:, 6:9]

    # ===== 4) 构造 data_dict =====
    if args.use_multiview:
        raise ValueError("use_multiview=True 但本脚本未提供多视角特征输入，请设为 False")

    data_dict = {}
    data_dict["point_clouds"] = build_point_clouds_tensor(
        xyz,
        rgb,
        normals,
        use_color=args.use_color,
        use_normal=args.use_normal,
        use_height=(not args.no_height),
    ).to(device)

    input_ids, attn_mask, token_type_ids = tokenize_query(
        args.query, use_distil=args.use_distil, max_len=args.bert_max_len
    )
    data_dict["input_ids"] = input_ids.to(device)
    data_dict["bert_attention_mask"] = attn_mask.to(device)
    if token_type_ids is not None:
        data_dict["token_type_ids"] = token_type_ids.to(device)

    data_dict = fill_dataset_like_fields(data_dict, device)
    data_dict["epoch"] = 0

    # ===== 5) 模型 =====
    DC = ScannetDatasetConfig()
    input_channels = (
        int(args.use_multiview) * 128
        + int(args.use_normal) * 3
        + int(args.use_color) * 3
        + int(not args.no_height)
    )
    model = JointNet(
        num_class=DC.num_class,
        vocabulary=None,
        embeddings=None,
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
        dataset_config=DC,
        use_distil=args.use_distil,
    ).to(device)

    ckpt = torch.load(args.ckpt, map_location="cpu")
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(state, strict=False)
    print("[load ckpt] missing:", missing)
    print("[load ckpt] unexpected:", unexpected)
    model.eval()

    # ===== 6) 推理 =====
    out = model(data_dict, use_tf=True, is_eval=True)
    if "cluster_ref" not in out:
        raise KeyError("模型输出里没有 cluster_ref；请确认 match/reference 分支在推理时开启。")

    # 参考 ground_visualize：对 cluster_ref 做 softmax 作为可比较分数
    # ===== 6.1) ref top-k =====
    ref_logits = out["cluster_ref"][0]                 # (P,)
    ref_prob = torch.softmax(ref_logits, dim=0)        # (P,)
    P = ref_prob.numel()
    topk = min(int(args.topk), int(P))
    topk_idx = torch.topk(ref_prob, k=topk, largest=True, sorted=True).indices
    topk_idx = topk_idx.detach().cpu().numpy().tolist()
    best_idx = int(topk_idx[0])

    # ===== 6.2) objectness top-k（若有）=====
    obj_prob = None
    topk_obj_idx = []
    if "objectness_scores" in out:
        obj_logits = out["objectness_scores"][0]  # (P,2)
        obj_prob = torch.softmax(obj_logits, dim=-1)[:, 1]  # (P,)
        topk_obj = min(int(args.topk), int(obj_prob.numel()))
        topk_obj_idx = torch.topk(obj_prob, k=topk_obj, largest=True, sorted=True).indices
        topk_obj_idx = topk_obj_idx.detach().cpu().numpy().tolist()
    else:
        print("[warn] 模型输出里没有 objectness_scores，跳过 proposal topk 输出")

    # ===== 6.3) proposals by ref =====
    proposals = []
    for rank, idx in enumerate(topk_idx):
        center = out["pred_center"][0, idx].detach().cpu().numpy()
        size = out["pred_size"][0, idx].detach().cpu().numpy()
        heading = out["pred_heading"][0, idx].detach().cpu().numpy()
        corner = out["pred_bbox_corner"][0, idx].detach().cpu().numpy()

        item = {
            "rank": rank,
            "proposal_idx": int(idx),
            "ref_score_softmax": float(ref_prob[idx].item()),
            "ref_score_logit": float(ref_logits[idx].item()),
            "center_xyz_m": center.tolist(),
            "size_xyz_m": size.tolist(),
            "heading_rad": float(heading),
            "bbox_corners_8x3": corner.tolist(),
        }
        if obj_prob is not None:
            item["objectness_prob"] = float(obj_prob[idx].item())
        proposals.append(item)

    # ===== 6.4) proposals by objectness =====
    proposals_by_objectness = []
    for rank, idx in enumerate(topk_obj_idx):
        center = out["pred_center"][0, idx].detach().cpu().numpy()
        size = out["pred_size"][0, idx].detach().cpu().numpy()
        heading = out["pred_heading"][0, idx].detach().cpu().numpy()
        corner = out["pred_bbox_corner"][0, idx].detach().cpu().numpy()

        proposals_by_objectness.append({
            "rank": rank,
            "proposal_idx": int(idx),
            "objectness_prob": float(obj_prob[idx].item()),
            "ref_score_softmax": float(ref_prob[idx].item()),
            "ref_score_logit": float(ref_logits[idx].item()),
            "center_xyz_m": center.tolist(),
            "size_xyz_m": size.tolist(),
            "heading_rad": float(heading),
            "bbox_corners_8x3": corner.tolist(),
        })

    # ===== 6.5) result =====
    result = {
        "query": args.query,
        "best_proposal_idx": best_idx,
        "topk": topk,
        "proposals": proposals,
        "proposals_by_objectness": proposals_by_objectness,
        "best_position_description": describe_position(
            out["pred_center"][0, best_idx].detach().cpu().numpy(), xyz
        ),
    }

    print(json.dumps(result, indent=2, ensure_ascii=False))

    # ===== 6.6) save json =====
    if args.out_json:
        out_dir = os.path.dirname(args.out_json)
        if out_dir:  # 关键：避免 args.out_json="pred.json" 时 dirname="" 导致 makedirs 报错
            os.makedirs(out_dir, exist_ok=True)
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"[save] wrote: {os.path.abspath(args.out_json)}")

    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)

        if args.save_scene_ply:
            scene_out = os.path.join(args.out_dir, "scene_points.ply")
            write_ply(xyz.tolist(), (rgb / 255.0).tolist(), [], scene_out)
            print(f"[save] wrote: {scene_out}")

        cluster_dir = os.path.join(args.out_dir, "cluster_topk")
        proposal_dir = os.path.join(args.out_dir, "proposal_topk")

        saved_cluster = export_topk_ply(proposals, cluster_dir, "cluster", args.ply_radius)
        if saved_cluster:
            print(f"[save] wrote {len(saved_cluster)} files to: {cluster_dir}")

        saved_proposal = export_topk_ply(proposals_by_objectness, proposal_dir, "proposal", args.ply_radius)
        if saved_proposal:
            print(f"[save] wrote {len(saved_proposal)} files to: {proposal_dir}")

        # 分开写两份 json，便于直接被其他脚本消费
        cluster_json = os.path.join(args.out_dir, "cluster_topk.json")
        with open(cluster_json, "w", encoding="utf-8") as f:
            json.dump({"query": args.query, "topk": topk, "proposals": proposals}, f, indent=2, ensure_ascii=False)
        print(f"[save] wrote: {cluster_json}")

        proposal_json = os.path.join(args.out_dir, "proposal_topk.json")
        with open(proposal_json, "w", encoding="utf-8") as f:
            json.dump(
                {"query": args.query, "topk": len(proposals_by_objectness), "proposals": proposals_by_objectness},
                f,
                indent=2,
                ensure_ascii=False,
            )
        print(f"[save] wrote: {proposal_json}")

        result_json = os.path.join(args.out_dir, "result.json")
        with open(result_json, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"[save] wrote: {result_json}")

if __name__ == "__main__":
    main()