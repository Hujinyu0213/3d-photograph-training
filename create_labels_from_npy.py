"""
从 NPY 文件创建标签文件
适配新的数据导出格式（完整点云 + NPY格式地标点）
"""
import os
import sys
import io
# 设置UTF-8编码以支持中文输出
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import numpy as np
import pandas as pd
from tqdm import tqdm

# =========================================================
# 配置
# =========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# 选项1: 从项目目录读取（推荐）
EXPORT_ROOT = os.path.join(BASE_DIR, 'data', 'pointcloud')

# 选项2: 从网络路径读取（如果数据在网络路径，取消下面的注释）
# EXPORT_ROOT = r"\\uz\data\Admin\mka\results\hou-and-hu\vs\pointcloud"

OUTPUT_LABELS_FILE = os.path.join(BASE_DIR, 'labels.csv')

# 目标地标点数量（9个解剖学地标点）
NUM_TARGET_POINTS = 9
OUTPUT_DIM = NUM_TARGET_POINTS * 3  # 27维

# =========================================================
# 核心功能：从NPY文件提取标签
# =========================================================
def extract_labels_from_npy():
    """
    从每个项目的 target_landmarks.npy 文件中提取标签
    返回: (N, 27) 的numpy数组，N为样本数
    """
    print(f"--- 从NPY文件提取 {NUM_TARGET_POINTS} 个地标点标签 ---")
    print(f"注意: 如果文件包含21个点，将自动提取后9个解剖学地标点")
    print(f"搜索目录: {EXPORT_ROOT}")
    
    if not os.path.exists(EXPORT_ROOT):
        raise FileNotFoundError(f"❌ 导出根目录不存在: {EXPORT_ROOT}")
    
    # 获取所有项目文件夹
    project_dirs = [d for d in os.listdir(EXPORT_ROOT) 
                    if os.path.isdir(os.path.join(EXPORT_ROOT, d))]
    project_dirs.sort()
    
    if len(project_dirs) == 0:
        raise RuntimeError(f"❌ 在 {EXPORT_ROOT} 中未找到任何项目文件夹")
    
    print(f"找到 {len(project_dirs)} 个项目文件夹")
    
    all_label_vectors = []
    valid_projects = []
    skipped_projects = []
    
    for project_name in tqdm(project_dirs, desc="提取标签"):
        project_dir = os.path.join(EXPORT_ROOT, project_name)
        # 优先查找 target_landmarks.npy，如果没有则查找 nose_landmarks.npy（兼容旧格式）
        target_landmarks_file = os.path.join(project_dir, "target_landmarks.npy")
        if not os.path.exists(target_landmarks_file):
            target_landmarks_file = os.path.join(project_dir, "nose_landmarks.npy")
        
        if not os.path.exists(target_landmarks_file):
            skipped_projects.append(project_name)
            print(f"[!] 跳过 {project_name}: 未找到 target_landmarks.npy 或 nose_landmarks.npy")
            continue
        
        try:
            # 加载地标点坐标
            landmarks = np.load(target_landmarks_file)
            
            # 处理不同的数据格式
            if landmarks.shape == (21, 3):
                # 如果包含21个点（12个编号型 + 9个解剖学），提取后9个解剖学地标点
                # 根据之前的分析：前12个是编号型（037-048），后9个是解剖学地标点
                target_landmarks = landmarks[-9:]  # 取最后9个点（索引12-20，即后9个）
            elif landmarks.shape == (NUM_TARGET_POINTS, 3):
                # 如果已经是9个点，直接使用
                target_landmarks = landmarks
            else:
                print(f"[!] 跳过 {project_name}: 地标点形状不正确 {landmarks.shape}，期望 (9, 3) 或 (21, 3)")
                skipped_projects.append(project_name)
                continue
            
            # 确保是9个点
            if target_landmarks.shape != (NUM_TARGET_POINTS, 3):
                print(f"[!] 跳过 {project_name}: 提取后形状不正确 {target_landmarks.shape}，期望 ({NUM_TARGET_POINTS}, 3)")
                skipped_projects.append(project_name)
                continue
            
            # 扁平化为 (27,) 向量
            label_vector = target_landmarks.flatten().astype(np.float32)
            all_label_vectors.append(label_vector)
            valid_projects.append(project_name)
            
        except Exception as e:
            print(f"[X] 处理 {project_name} 时出错: {e}")
            skipped_projects.append(project_name)
    
    if len(all_label_vectors) == 0:
        raise RuntimeError("未能提取任何有效标签！")
    
    labels_matrix = np.array(all_label_vectors, dtype=np.float32)
    
    print(f"\n[OK] 成功提取 {len(valid_projects)} 个样本的标签")
    if skipped_projects:
        print(f"[!] 跳过了 {len(skipped_projects)} 个样本")
    
    return labels_matrix, valid_projects

# =========================================================
# 主程序
# =========================================================
if __name__ == "__main__":
    try:
        labels_matrix, valid_projects = extract_labels_from_npy()
        
        # 保存为CSV（无表头，无索引）
        labels_df = pd.DataFrame(labels_matrix)
        labels_df.to_csv(OUTPUT_LABELS_FILE, header=False, index=False)
        
        print(f"\n标签文件已成功创建: {OUTPUT_LABELS_FILE}")
        print(f"   文件形状: {labels_matrix.shape[0]} 行 (样本数) × {labels_matrix.shape[1]} 列 (坐标数)")
        print(f"   有效样本: {len(valid_projects)} 个")
        
        # 保存项目名称列表（用于后续数据加载）
        projects_file = os.path.join(BASE_DIR, 'valid_projects.txt')
        with open(projects_file, 'w', encoding='utf-8') as f:
            for proj in valid_projects:
                f.write(proj + '\n')
        print(f"   项目列表: {projects_file}")
        
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()
