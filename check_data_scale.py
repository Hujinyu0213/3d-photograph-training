"""
检查原始数据的坐标范围和单位
"""
import sys
import io
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import os
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EXPORT_ROOT = os.path.join(BASE_DIR, 'data', 'pointcloud')
LABELS_FILE = os.path.join(BASE_DIR, 'labels.csv')
PROJECTS_LIST_FILE = os.path.join(BASE_DIR, 'valid_projects.txt')

print("="*70)
print("原始数据坐标范围和单位检查")
print("="*70)

# 读取标签文件
print("\n1. 检查标签文件 (labels.csv)...")
if os.path.exists(LABELS_FILE):
    labels_df = pd.read_csv(LABELS_FILE, header=None)
    labels_np = labels_df.values.astype(np.float32)
    
    print(f"   标签文件形状: {labels_np.shape}")
    print(f"   样本数: {labels_np.shape[0]}")
    print(f"   坐标数: {labels_np.shape[1]} (应该是27 = 9个点 × 3维)")
    
    # 重塑为 (N, 9, 3)
    labels_reshaped = labels_np.reshape(-1, 9, 3)
    
    print(f"\n   坐标统计 (所有样本的所有坐标):")
    print(f"     最小值: {labels_np.min():.2f}")
    print(f"     最大值: {labels_np.max():.2f}")
    print(f"     平均值: {labels_np.mean():.2f}")
    print(f"     标准差: {labels_np.std():.2f}")
    print(f"     中位数: {np.median(labels_np):.2f}")
    
    print(f"\n   每个地标点的坐标范围 (9个点):")
    for i in range(9):
        point_coords = labels_reshaped[:, i, :]  # (N, 3)
        print(f"     点 {i+1}: X范围 [{point_coords[:, 0].min():.2f}, {point_coords[:, 0].max():.2f}], "
              f"Y范围 [{point_coords[:, 1].min():.2f}, {point_coords[:, 1].max():.2f}], "
              f"Z范围 [{point_coords[:, 2].min():.2f}, {point_coords[:, 2].max():.2f}]")
    
    # 计算每个点的3D距离范围
    print(f"\n   每个地标点的3D位置范围 (相对于原点的距离):")
    for i in range(9):
        point_coords = labels_reshaped[:, i, :]  # (N, 3)
        distances = np.sqrt(np.sum(point_coords**2, axis=1))
        print(f"     点 {i+1}: 距离范围 [{distances.min():.2f}, {distances.max():.2f}], "
              f"平均距离: {distances.mean():.2f}")
    
    # 检查单位（基于数值大小推断）
    max_val = abs(labels_np).max()
    print(f"\n   单位推断:")
    if max_val < 10:
        print(f"     可能是: 厘米 (cm) 或 分米 (dm)")
    elif max_val < 100:
        print(f"     可能是: 厘米 (cm) 或 分米 (dm)")
    elif max_val < 1000:
        print(f"     可能是: 毫米 (mm) 或 厘米 (cm)")
    else:
        print(f"     可能是: 毫米 (mm)")
    print(f"     最大绝对值: {max_val:.2f}")
    
else:
    print(f"   ❌ 标签文件不存在: {LABELS_FILE}")

# 检查原始NPY文件
print("\n2. 检查原始NPY文件 (target_landmarks.npy 或 nose_landmarks.npy)...")
if os.path.exists(PROJECTS_LIST_FILE):
    with open(PROJECTS_LIST_FILE, 'r', encoding='utf-8') as f:
        project_names = [line.strip() for line in f if line.strip()]
    
    print(f"   找到 {len(project_names)} 个项目")
    
    # 检查前几个项目的原始数据
    sample_count = min(5, len(project_names))
    all_original_coords = []
    
    for i, project_name in enumerate(project_names[:sample_count]):
        project_dir = os.path.join(EXPORT_ROOT, project_name)
        
        # 尝试找到地标点文件
        target_file = os.path.join(project_dir, "target_landmarks.npy")
        nose_file = os.path.join(project_dir, "nose_landmarks.npy")
        
        if os.path.exists(target_file):
            landmarks = np.load(target_file)
            all_original_coords.append(landmarks)
            print(f"   {project_name}: target_landmarks.npy, 形状: {landmarks.shape}")
        elif os.path.exists(nose_file):
            landmarks = np.load(nose_file)
            all_original_coords.append(landmarks)
            print(f"   {project_name}: nose_landmarks.npy, 形状: {landmarks.shape}")
        else:
            print(f"   {project_name}: 未找到地标点文件")
    
    if all_original_coords:
        # 合并所有坐标
        all_coords = np.concatenate(all_original_coords, axis=0)
        print(f"\n   原始NPY文件坐标统计 (前{sample_count}个样本):")
        print(f"     最小值: {all_coords.min():.2f}")
        print(f"     最大值: {all_coords.max():.2f}")
        print(f"     平均值: {all_coords.mean():.2f}")
        print(f"     标准差: {all_coords.std():.2f}")

# 检查点云数据
print("\n3. 检查点云数据范围...")
if os.path.exists(PROJECTS_LIST_FILE):
    with open(PROJECTS_LIST_FILE, 'r', encoding='utf-8') as f:
        project_names = [line.strip() for line in f if line.strip()]
    
    sample_count = min(3, len(project_names))
    all_pointcloud_coords = []
    
    for i, project_name in enumerate(project_names[:sample_count]):
        project_dir = os.path.join(EXPORT_ROOT, project_name)
        pointcloud_file = os.path.join(project_dir, "pointcloud_full.npy")
        
        if os.path.exists(pointcloud_file):
            pointcloud = np.load(pointcloud_file)
            all_pointcloud_coords.append(pointcloud)
    
    if all_pointcloud_coords:
        all_pc = np.concatenate(all_pointcloud_coords, axis=0)
        print(f"   点云坐标统计 (前{sample_count}个样本):")
        print(f"     最小值: {all_pc.min():.2f}")
        print(f"     最大值: {all_pc.max():.2f}")
        print(f"     平均值: {all_pc.mean():.2f}")
        print(f"     标准差: {all_pc.std():.2f}")

# 分析误差
print("\n" + "="*70)
print("误差分析")
print("="*70)

if os.path.exists(LABELS_FILE):
    labels_np = pd.read_csv(LABELS_FILE, header=None).values.astype(np.float32)
    labels_reshaped = labels_np.reshape(-1, 9, 3)
    
    # 计算每个样本内9个点之间的平均距离
    print("\n4. 计算地标点之间的典型距离...")
    inter_point_distances = []
    for i in range(len(labels_reshaped)):
        points = labels_reshaped[i]  # (9, 3)
        # 计算所有点对之间的距离
        for j in range(9):
            for k in range(j+1, 9):
                dist = np.linalg.norm(points[j] - points[k])
                inter_point_distances.append(dist)
    
    inter_point_distances = np.array(inter_point_distances)
    print(f"   地标点之间的平均距离: {inter_point_distances.mean():.2f}")
    print(f"   地标点之间的中位数距离: {np.median(inter_point_distances):.2f}")
    print(f"   地标点之间的最小距离: {inter_point_distances.min():.2f}")
    print(f"   地标点之间的最大距离: {inter_point_distances.max():.2f}")
    
    # 当前测试集RMSE
    test_rmse = np.sqrt(244681)  # 从K折结果
    print(f"\n5. 当前模型误差分析:")
    print(f"   测试集RMSE: {test_rmse:.2f}")
    print(f"   相对于地标点平均距离: {test_rmse / inter_point_distances.mean() * 100:.1f}%")
    print(f"   目标误差: 2mm")
    print(f"   当前误差是目标的: {test_rmse / 2:.1f}倍")
    
    # 单位推断
    if test_rmse > 100:
        print(f"\n   ⚠️  如果目标是2mm，当前误差 {test_rmse:.2f} 说明:")
        print(f"      - 如果坐标单位是mm: 误差是 {test_rmse:.2f}mm (需要改进 {test_rmse/2:.1f}倍)")
        print(f"      - 如果坐标单位是cm: 误差是 {test_rmse/10:.2f}mm (接近目标)")
        print(f"      - 如果坐标单位是m: 误差是 {test_rmse*1000:.2f}mm (远大于目标)")

print("\n" + "="*70)
print("建议")
print("="*70)
print("\n根据数据检查结果，请查看上面的统计信息来确定:")
print("1. 坐标的实际单位 (mm/cm/m)")
print("2. 地标点的典型距离范围")
print("3. 当前误差相对于数据规模的比例")
print("="*70)
