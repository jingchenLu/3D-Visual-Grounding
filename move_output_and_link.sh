#!/bin/bash

# 1. 定义源目录和目标目录
SRC_DIR="/home/ljc/work/3DVLP/outputs/exp_joint"
DEST_DIR="/sdb/ljc/output-3dvlp"

# 2. 确保目标父目录存在 (如果不存在则创建)
if [ ! -d "$DEST_DIR" ]; then
    echo "创建目标目录: $DEST_DIR"
    mkdir -p "$DEST_DIR"
fi

# 3. 进入源目录
cd "$SRC_DIR" || { echo "无法进入目录 $SRC_DIR"; exit 1; }

echo "当前工作目录: $(pwd)"
echo "----------------------------------------"

# 4. 遍历当前目录下的所有文件夹
for folder in *; do
    # 只处理文件夹，忽略文件
    if [ -d "$folder" ]; then
        
        # 5. 检查是否在保留列表中 (使用 case 语句匹配)
        case "$folder" in
            "2025-11-07_10-54-31" | \
            "2025-11-27_14-54-41-12realation" | \
            "2026-01-08_00-26-44")
                echo "[保留] 跳过文件夹: $folder"
                ;;
            *)
                # 6. 执行移动和软链操作
                # 检查目标位置是否已经存在同名文件夹，防止覆盖
                if [ -d "$DEST_DIR/$folder" ]; then
                    echo "[警告] 目标位置已存在 $folder，跳过移动。"
                else
                    echo "[处理] 正在迁移: $folder ..."
                    
                    # A. 移动文件夹到新位置
                    mv "$folder" "$DEST_DIR/"
                    
                    # B. 在原位置创建指向新位置的软链接
                    # ln -s 目标实际路径 软链接名称
                    ln -s "$DEST_DIR/$folder" "$folder"
                    
                    echo "       -> 已建立软链接"
                fi
                ;;
        esac
    fi
done

echo "----------------------------------------"
echo "所有操作已完成。"