import os
import glob
from tqdm import tqdm

def convert_deep_pcb_to_yolo(label_dir, img_width=640, img_height=640):
    """
    1. 将坐标从 绝对像素值 转为 0-1 归一化值。
    2. 将类别 ID 从 1-6 转为 0-5。
    """
    
    # 获取所有 txt 文件
    label_files = glob.glob(os.path.join(label_dir, "*.txt"))
    print(f"正在处理 {label_dir} 下的 {len(label_files)} 个文件...")

    for file_path in tqdm(label_files):
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        new_lines = []
        is_modified = False # 标记文件是否需要修改

        for line in lines:
            parts = list(map(float, line.strip().split()))
            
            if len(parts) < 5:
                continue

            # ==========================================
            # 关键区域：请确认你的原数据列顺序！
            # 情况 A: x_min y_min x_max y_max class_id (DeepPCB 原始格式常见)
            x_min, y_min, x_max, y_max = parts[0], parts[1], parts[2], parts[3]
            original_id = int(parts[4])
            
            # 情况 B: class_id x_min y_min x_max y_max (如果 ID 在第一列，解开下面注释，注释掉上面)
            # original_id = int(parts[0])
            # x_min, y_min, x_max, y_max = parts[1], parts[2], parts[3], parts[4]
            # ==========================================

            # --- 核心修正 1: 类别 ID 减 1 ---
            # 将 1-6 映射到 0-5
            new_id = original_id - 1
            
            # 安全检查：防止已经是 0-5 的数据被减成负数
            if new_id < 0:
                print(f"警告：在文件 {os.path.basename(file_path)} 发现 ID 为 {original_id}，修正后为负数，强制归零。")
                new_id = 0

            # --- 核心修正 2: 坐标归一化 ---
            # 计算中心点和宽高
            w = x_max - x_min
            h = y_max - y_min
            x_center = x_min + w / 2
            y_center = y_min + h / 2

            # 归一化
            x_center /= img_width
            y_center /= img_height
            w /= img_width
            h /= img_height

            # 越界截断（防止坐标稍微超出 1.0）
            x_center = min(max(x_center, 0.0), 1.0)
            y_center = min(max(y_center, 0.0), 1.0)
            w = min(max(w, 0.0), 1.0)
            h = min(max(h, 0.0), 1.0)

            # 生成 YOLO 格式: class_id x_center y_center w h
            new_lines.append(f"{new_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")
            is_modified = True

        # 覆写文件
        if is_modified:
            with open(file_path, 'w') as f:
                f.writelines(new_lines)

# --- 执行配置 ---
# 请修改为你的实际路径
train_path = "/root/autodl-tmp/DeepPCB/DeepPCB/labels/test"
# val_path = "/root/autodl-tmp/DeepPCB/DeepPCB/labels/val" 

# 先清除缓存，否则 YOLO 会读取旧的 .cache 文件！
os.system(f"rm {train_path}.cache 2>/dev/null") 
os.system(f"rm {os.path.dirname(train_path)}/*.cache 2>/dev/null")

print("开始转换 test 集...")
convert_deep_pcb_to_yolo(train_path)

# print("开始转换 Val 集...")
# convert_deep_pcb_to_yolo(val_path)