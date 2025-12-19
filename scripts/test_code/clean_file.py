import os
import shutil

# ================= 配置区域 =================

# 目标路径
SCANS_TEST_DIR = '/home/ljc/work/3DVLP/data/scannet/scans_test'

# 安全开关：True = 只打印不删除 (预演模式)； False = 真正删除文件
DRY_RUN = False 

# 只保留这两种后缀
KEEP_SUFFIXES = [
    '_vh_clean_2.ply',  # 保留 clean_2 点云
    '.txt'              # 保留 meta 信息
]

# ===========================================

def clean_scannet_test():
    if not os.path.exists(SCANS_TEST_DIR):
        print(f"错误: 目录不存在 - {SCANS_TEST_DIR}")
        return

    print(f"正在扫描目录: {SCANS_TEST_DIR}")
    print(f"保留规则: 仅保留以 {KEEP_SUFFIXES} 结尾的文件")
    print(f"当前模式: {'[预演模式 - 不会删除文件]' if DRY_RUN else '[执行模式 - 将物理删除文件]'}")
    print("-" * 50)

    deleted_count = 0
    kept_count = 0

    # 遍历 scans_test 下的所有 scene 文件夹
    # 使用 sorted 确保顺序一致
    scenes = sorted(os.listdir(SCANS_TEST_DIR))
    
    for scene_id in scenes:
        scene_path = os.path.join(SCANS_TEST_DIR, scene_id)
        
        # 确保只处理目录
        if os.path.isdir(scene_path):
            
            # 遍历 scene 文件夹内的文件
            for filename in os.listdir(scene_path):
                file_path = os.path.join(scene_path, filename)
                
                # 检查是否匹配保留后缀
                should_keep = False
                for suffix in KEEP_SUFFIXES:
                    if filename.endswith(suffix):
                        should_keep = True
                        break
                
                if should_keep:
                    kept_count += 1
                else:
                    # 不在保留列表里，删除
                    if DRY_RUN:
                        print(f"[预演删除] {scene_id}/{filename}")
                    else:
                        try:
                            if os.path.isfile(file_path) or os.path.islink(file_path):
                                os.remove(file_path)
                                print(f"[已删除] {scene_id}/{filename}")
                            elif os.path.isdir(file_path):
                                shutil.rmtree(file_path)
                                print(f"[已删除目录] {scene_id}/{filename}")
                        except Exception as e:
                            print(f"[删除失败] {file_path}: {e}")
                    
                    deleted_count += 1

    print("-" * 50)
    print(f"扫描完成。")
    print(f"保留文件数: {kept_count}")
    if DRY_RUN:
        print(f"待删除文件数: {deleted_count} (请将代码中 DRY_RUN 改为 False 以执行删除)")
    else:
        print(f"已删除文件数: {deleted_count}")

if __name__ == "__main__":
    clean_scannet_test()