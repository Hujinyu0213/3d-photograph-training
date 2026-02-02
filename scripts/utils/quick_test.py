"""
快速测试GPU环境
"""
import sys
import io
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import torch
import os

print("="*60)
print("快速环境检查")
print("="*60)

print(f"\nPython版本: {sys.version.split()[0]}")
print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    print("\n[OK] GPU环境配置正确！")
else:
    print("\n[!] CUDA不可用")

print("\n检查数据文件...")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
labels_file = os.path.join(BASE_DIR, 'labels.csv')
projects_file = os.path.join(BASE_DIR, 'valid_projects.txt')
data_dir = os.path.join(BASE_DIR, 'data', 'pointcloud')

if os.path.exists(labels_file):
    print(f"[OK] labels.csv 存在")
else:
    print(f"[X] labels.csv 不存在")

if os.path.exists(projects_file):
    print(f"[OK] valid_projects.txt 存在")
else:
    print(f"[X] valid_projects.txt 不存在")

if os.path.exists(data_dir):
    print(f"[OK] 数据目录存在: {data_dir}")
else:
    print(f"[X] 数据目录不存在: {data_dir}")

print("\n" + "="*60)
print("如果所有检查都通过，可以开始训练！")
print("="*60)
