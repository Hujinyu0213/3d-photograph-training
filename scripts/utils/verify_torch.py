"""
验证PyTorch安装
"""
import sys
import io
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

try:
    import torch
    print("="*60)
    print("PyTorch安装验证")
    print("="*60)
    print(f"[OK] PyTorch版本: {torch.__version__}")
    
    if torch.cuda.is_available():
        print(f"[OK] CUDA可用: {torch.cuda.get_device_name(0)}")
        print(f"   GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        print(f"   CUDA版本: {torch.version.cuda}")
        print("\n[OK] 可以使用GPU加速训练！")
    else:
        print("[!] CUDA不可用（将使用CPU训练）")
        print("   注意: CPU训练会很慢，建议使用GPU")
    
    print("\n" + "="*60)
    print("[OK] PyTorch安装成功！可以开始训练了！")
    print("="*60)
    print("\n运行命令:")
    print("   python main_script_full_pointcloud.py  (快速训练)")
    print("   或")
    print("   python main_script_kfold.py  (K折交叉验证)")
    
except ImportError:
    print("[X] PyTorch未安装")
    print("请运行: pip install torch torchvision torchaudio")
except Exception as e:
    print(f"[!] 检查出错: {e}")
