import os
import sys
import json
import h5py
from tqdm import tqdm
import numpy as np

sys.path.append(os.path.join(os.getcwd()))  # HACK add the root folder
from lib.configs.config import CONF
MEAN_COLOR_RGB = np.array([109.8, 97.2, 83.8])


def get_scannet_scene_list(split):
    scene_list = sorted([line.rstrip() for line in open(os.path.join(
        CONF.PATH.SCANNET_META, "scannetv2_{}.txt".format(split)))])
    return scene_list


def work(split, use_color=False, use_normal=True, use_multiview=True):
    multiview_data = h5py.File(CONF.MULTIVIEW, "r", libver="latest")
    scene_list = get_scannet_scene_list(split)

    dump_data = []

    # for scene_id in tqdm(scene_list[:10]):
    for scene_id in tqdm(scene_list):
        # load scene data
        mesh_vertices = np.load(os.path.join(
            CONF.PATH.SCANNET_DATA, scene_id)+"_aligned_vert.npy")

        # -------------------------- 新增：2. 先构造输出文件路径，判断是否已存在 --------------------------
        preprocess_output_path = os.path.join(CONF.PATH.SCANNET_DATA, scene_id) + f"_preprocess_{split}.npy"
        pcl_color_output_path = os.path.join(CONF.PATH.SCANNET_DATA, scene_id) + f"_pcl_color_{split}.npy"
        
        # 若两个输出文件都已存在，直接跳过后续处理和保存
        if os.path.exists(preprocess_output_path) and os.path.exists(pcl_color_output_path):
            print(f"提示：场景 {scene_id} 的输出文件已存在，跳过保存")
            continue  # 跳过当前场景的保存逻辑，进入下一个循环

        # use color
        if not use_color:
            if not use_color:
                point_cloud = mesh_vertices[:, 0:3]  # do not use color for now
                pcl_color = mesh_vertices[:, 3:6]
            else:
                point_cloud = mesh_vertices[:, 0:6]
                point_cloud[:, 3:6] = (
                    point_cloud[:, 3:6]-MEAN_COLOR_RGB)/256.0
                pcl_color = point_cloud[:, 3:6]

            if use_normal:
                normals = mesh_vertices[:, 6:9]
                point_cloud = np.concatenate([point_cloud, normals], 1)

            if use_multiview:
                multiview = multiview_data[scene_id]
                point_cloud = np.concatenate([point_cloud, multiview], 1)
            
            dump_data.append(point_cloud.tolist())

        # -------------------------- 原代码：文件保存逻辑（保留不变，仅在输出文件不存在时执行） --------------------------
        np.save(preprocess_output_path, point_cloud)
        np.save(pcl_color_output_path, pcl_color)
    with open("dump.json", "w") as f:
        json.dump(dump_data, f)

if __name__ == "__main__":
    print("preprocess train dataset")
    work('train', use_color=False, use_normal=True, use_multiview=True)
    print("preprocess val dataset")
    work('val', use_color=False, use_normal=True, use_multiview=True)
