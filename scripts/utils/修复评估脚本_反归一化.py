"""
修复评估脚本：检查反归一化过程
"""
import os
import sys
import io
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EXPORT_ROOT = os.path.join(BASE_DIR, 'data', 'pointcloud')
LABELS_FILE = os.path.join(BASE_DIR, 'labels.csv')
PROJECTS_LIST_FILE = os.path.join(BASE_DIR, 'valid_projects.txt')

import numpy as np
import pandas as pd

print("="*70)
print("检查反归一化过程")
print("="*70)

# 检查几个样本的归一化尺度
with open(PROJECTS_LIST_FILE, 'r', encoding='utf-8') as f:
    project_names = [line.strip() for line in f if line.strip()]

labels_df = pd.read_csv(LABELS_FILE, header=None)
all_labels_np = labels_df.values.astype(np.float32)

print(f"\n检查前5个样本的归一化尺度...")

scales = []
for i in range(min(5, len(project_names))):
    project_name = project_names[i]
    project_dir = os.path.join(EXPORT_ROOT, project_name)
    pointcloud_file = os.path.join(project_dir, "pointcloud_full.npy")
    
    if os.path.exists(pointcloud_file):
        pointcloud = np.load(pointcloud_file).astype(np.float32)
        current_label = all_labels_np[i].reshape(9, 3)
        
        # 使用地标点质心
        label_centroid = np.mean(current_label, axis=0)
        centered_pointcloud = pointcloud - label_centroid
        centered_label = current_label - label_centroid
        
        # 计算尺度
        scale = np.std(centered_pointcloud)
        scales.append(scale)
        
        print(f"\n样本 {i+1} ({project_name}):")
        print(f"   点云原始范围: [{pointcloud.min():.2f}, {pointcloud.max():.2f}]")
        print(f"   地标点范围: [{current_label.min():.2f}, {current_label.max():.2f}]")
        print(f"   地标点质心: [{label_centroid[0]:.2f}, {label_centroid[1]:.2f}, {label_centroid[2]:.2f}]")
        print(f"   归一化尺度: {scale:.2f}")
        print(f"   归一化后点云范围: [{centered_pointcloud.min()/scale:.2f}, {centered_pointcloud.max()/scale:.2f}]")
        print(f"   归一化后地标点范围: [{centered_label.min()/scale:.2f}, {centered_label.max()/scale:.2f}]")

if scales:
    print(f"\n平均归一化尺度: {np.mean(scales):.2f}")
    print(f"尺度范围: {np.min(scales):.2f} - {np.max(scales):.2f}")

print("\n" + "="*70)
print("反归一化验证")
print("="*70)

# 验证反归一化公式
print("\n假设归一化后的预测误差为0.0081（基于验证损失0.000066）")
normalized_rmse = np.sqrt(0.000066)
print(f"归一化后的RMSE: {normalized_rmse:.6f}")

if scales:
    avg_scale = np.mean(scales)
    actual_rmse = normalized_rmse * avg_scale
    print(f"使用平均尺度 {avg_scale:.2f} 反归一化:")
    print(f"   实际RMSE = {normalized_rmse:.6f} × {avg_scale:.2f} = {actual_rmse:.2f}")
    
    print(f"\n但测试集实际RMSE是10.06mm")
    print(f"差异: {10.06 / actual_rmse:.1f}倍")
    print(f"\n可能原因:")
    print(f"   1. 测试集使用的尺度不同")
    print(f"   2. 反归一化过程有误")
    print(f"   3. 测试集划分不同导致")

print("="*70)
