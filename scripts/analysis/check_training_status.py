"""
检查训练脚本状态
"""
import sys
import io
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import os
import subprocess
import time

print("="*60)
print("训练状态检查")
print("="*60)

# 检查1: Python进程
print("\n检查1: 查找Python训练进程...")
try:
    result = subprocess.run(['tasklist', '/FI', 'IMAGENAME eq python.exe'], 
                          capture_output=True, text=True, encoding='gbk', errors='ignore')
    if 'python.exe' in result.stdout:
        print("  [OK] 发现Python进程正在运行")
        print("  进程信息:")
        lines = result.stdout.split('\n')
        for line in lines:
            if 'python.exe' in line:
                print(f"    {line.strip()}")
    else:
        print("  [!] 未发现Python进程")
        print("  可能原因:")
        print("    1. 训练脚本还未启动")
        print("    2. 训练脚本已结束")
        print("    3. 训练脚本出错退出")
except Exception as e:
    print(f"  [!] 检查进程时出错: {e}")

# 检查2: 模型文件
print("\n检查2: 检查模型文件...")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
best_model = os.path.join(BASE_DIR, 'pointnet_regression_model_full_best.pth')
final_model = os.path.join(BASE_DIR, 'pointnet_regression_model_full.pth')

if os.path.exists(best_model):
    size = os.path.getsize(best_model) / (1024*1024)
    mtime = os.path.getmtime(best_model)
    print(f"  [OK] 最佳模型文件存在: {best_model}")
    print(f"    大小: {size:.2f} MB")
    print(f"    修改时间: {time.ctime(mtime)}")
else:
    print(f"  [!] 最佳模型文件不存在（训练可能还未开始或未完成）")

if os.path.exists(final_model):
    size = os.path.getsize(final_model) / (1024*1024)
    mtime = os.path.getmtime(final_model)
    print(f"  [OK] 最终模型文件存在: {final_model}")
    print(f"    大小: {size:.2f} MB")
    print(f"    修改时间: {time.ctime(mtime)}")
else:
    print(f"  [!] 最终模型文件不存在（训练可能还未完成）")

# 检查3: 训练历史
print("\n检查3: 检查训练历史...")
history_file = os.path.join(BASE_DIR, 'training_history_full.json')
if os.path.exists(history_file):
    size = os.path.getsize(history_file)
    mtime = os.path.getmtime(history_file)
    print(f"  [OK] 训练历史文件存在: {history_file}")
    print(f"    大小: {size} bytes")
    print(f"    修改时间: {time.ctime(mtime)}")
else:
    print(f"  [!] 训练历史文件不存在")

print("\n" + "="*60)
print("建议:")
print("  1. 如果看到Python进程，训练可能正在进行（数据加载阶段）")
print("  2. 如果没看到进程，训练可能已结束或出错")
print("  3. 检查训练脚本的输出窗口，查看是否有错误信息")
print("  4. 如果训练脚本窗口已关闭，重新运行训练脚本")
print("="*60)
