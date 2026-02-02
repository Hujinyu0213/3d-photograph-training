"""
检查标签文件是否已正确创建
"""
import os
import sys
import io
# 设置UTF-8编码
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

print("="*60)
print("检查标签文件")
print("="*60)

# 检查1: labels.csv
labels_file = os.path.join(BASE_DIR, 'labels.csv')
print("\n检查1: labels.csv")
if os.path.exists(labels_file):
    print(f"   [OK] 文件存在")
    df = pd.read_csv(labels_file, header=None)
    print(f"   行数（样本数）: {len(df)}")
    print(f"   列数（坐标数）: {len(df.columns)}")
    print(f"   预期: 100行 x 27列")
    if len(df) == 100 and len(df.columns) == 27:
        print(f"   [OK] 文件格式正确！")
    else:
        print(f"   [!] 文件格式可能有问题")
else:
    print(f"   [X] 文件不存在")

# 检查2: valid_projects.txt
projects_file = os.path.join(BASE_DIR, 'valid_projects.txt')
print("\n检查2: valid_projects.txt")
if os.path.exists(projects_file):
    print(f"   [OK] 文件存在")
    with open(projects_file, 'r', encoding='utf-8') as f:
        projects = [line.strip() for line in f if line.strip()]
    print(f"   项目数量: {len(projects)} 个")
    if len(projects) == 100:
        print(f"   [OK] 项目数量正确！")
    else:
        print(f"   [!] 项目数量: {len(projects)}，预期100")
else:
    print(f"   [X] 文件不存在")

# 总结
print("\n" + "="*60)
print("检查总结")
print("="*60)

if os.path.exists(labels_file) and os.path.exists(projects_file):
    df = pd.read_csv(labels_file, header=None)
    if len(df) == 100 and len(df.columns) == 27:
        print("[OK] 所有文件已正确创建！")
        print("\n可以开始训练了！")
        print("\n运行命令:")
        print("   python main_script_full_pointcloud.py")
    else:
        print("[!] 文件格式可能有问题，请检查")
else:
    print("[X] 文件缺失，需要运行: python create_labels_from_npy.py")

print("="*60)
