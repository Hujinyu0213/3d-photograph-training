"""
检查GPU和PyTorch CUDA支持
"""
import sys
import io
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import torch
import sys

print("="*60)
print("GPU和PyTorch检查")
print("="*60)

print(f"\nPython版本: {sys.version}")
print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA版本: {torch.version.cuda}")
    print(f"GPU数量: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"    内存: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
    print("\n[OK] GPU可用！可以加速训练！")
else:
    print("\n[!] CUDA不可用")
    print("可能原因:")
    print("  1. Python 3.13可能还没有完整的CUDA支持")
    print("  2. 需要安装CUDA版本的PyTorch")
    print("\n建议:")
    print("  方案A: 使用conda安装GPU版本")
    print("    conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia")
    print("  方案B: 使用Python 3.11或3.12（更稳定的CUDA支持）")
    print("  方案C: 暂时使用CPU训练（会很慢）")

print("="*60)
