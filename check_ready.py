"""
训练前检查脚本
检查所有准备工作是否完成
"""
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 设置UTF-8编码
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

print("="*60)
print("训练前检查")
print("="*60)

# 检查1: 标签文件
labels_file = os.path.join(BASE_DIR, 'labels.csv')
valid_projects_file = os.path.join(BASE_DIR, 'valid_projects.txt')

print("\n📋 检查1: 标签文件")
if os.path.exists(labels_file):
    print(f"   [OK] labels.csv 存在")
    # 检查文件大小
    size = os.path.getsize(labels_file)
    print(f"      文件大小: {size:,} 字节")
else:
    print(f"   [X] labels.csv 不存在")
    print(f"      需要运行: python create_labels_from_npy.py")

if os.path.exists(valid_projects_file):
    print(f"   [OK] valid_projects.txt 存在")
    with open(valid_projects_file, 'r', encoding='utf-8') as f:
        count = len([line for line in f if line.strip()])
    print(f"      项目数量: {count} 个")
else:
    print(f"   [X] valid_projects.txt 不存在")
    print(f"      需要运行: python create_labels_from_npy.py")

# 检查2: 数据目录
data_dir = os.path.join(BASE_DIR, 'data', 'pointcloud')
print("\n📋 检查2: 数据目录")
if os.path.exists(data_dir):
    print(f"   [OK] 数据目录存在: {data_dir}")
    # 检查项目文件夹数量
    project_dirs = [d for d in os.listdir(data_dir) 
                    if os.path.isdir(os.path.join(data_dir, d))]
    print(f"      项目文件夹数量: {len(project_dirs)} 个")
    
    # 检查几个样本文件
    sample_count = 0
    for project_name in project_dirs[:5]:  # 只检查前5个
        npy_file = os.path.join(data_dir, project_name, 'pointcloud_full.npy')
        if os.path.exists(npy_file):
            sample_count += 1
    
    if sample_count > 0:
        print(f"      [OK] 找到点云文件（检查了前5个，{sample_count}个有效）")
    else:
        print(f"      [!]  未找到点云文件")
else:
    print(f"   [X] 数据目录不存在: {data_dir}")

# 检查3: Python依赖
print("\n📋 检查3: Python依赖包")
try:
    import torch
    print(f"   [OK] PyTorch: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"   [OK] CUDA可用: {torch.cuda.get_device_name(0)}")
        print(f"      GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    else:
        print(f"   [!]  CUDA不可用（将使用CPU，训练会很慢）")
except ImportError:
    print(f"   [X] PyTorch 未安装")
    print(f"      需要安装: pip install torch")

try:
    import numpy
    print(f"   [OK] NumPy: {numpy.__version__}")
except ImportError:
    print(f"   [X] NumPy 未安装")
    print(f"      需要安装: pip install numpy")

try:
    import pandas
    print(f"   [OK] Pandas: {pandas.__version__}")
except ImportError:
    print(f"   [X] Pandas 未安装")
    print(f"      需要安装: pip install pandas")

try:
    import sklearn
    print(f"   [OK] Scikit-learn: {sklearn.__version__}")
except ImportError:
    print(f"   [X] Scikit-learn 未安装（K折交叉验证需要）")
    print(f"      需要安装: pip install scikit-learn")

try:
    import tqdm
    print(f"   [OK] tqdm: {tqdm.__version__}")
except ImportError:
    print(f"   [X] tqdm 未安装")
    print(f"      需要安装: pip install tqdm")

# 检查4: 训练脚本
print("\n📋 检查4: 训练脚本")
scripts = [
    'main_script_full_pointcloud.py',
    'main_script_kfold.py',
    'create_labels_from_npy.py',
    'pointnet_utils.py'
]

for script in scripts:
    script_path = os.path.join(BASE_DIR, script)
    if os.path.exists(script_path):
        print(f"   [OK] {script}")
    else:
        print(f"   [X] {script} 不存在")

# 总结
print("\n" + "="*60)
print("📊 检查总结")
print("="*60)

all_ready = (
    os.path.exists(labels_file) and 
    os.path.exists(valid_projects_file) and
    os.path.exists(data_dir)
)

if all_ready:
    print("[OK] 所有准备工作已完成！")
    print("\n可以开始训练了！")
    print("\n推荐命令:")
    print("   python main_script_full_pointcloud.py  (快速训练)")
    print("   或")
    print("   python main_script_kfold.py  (K折交叉验证)")
else:
    print("[!] 还有准备工作未完成")
    print("\n下一步:")
    if not os.path.exists(labels_file):
        print("   1. 运行: python create_labels_from_npy.py")
    if not os.path.exists(data_dir):
        print("   2. 检查数据目录是否存在")

print("="*60)
