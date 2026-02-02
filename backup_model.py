"""
模型备份工具
用于保存当前训练好的模型，避免被新训练覆盖
"""
import os
import shutil
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'pointnet_regression_model.pth')

def backup_model(backup_name=None):
    """
    备份当前模型
    
    参数:
        backup_name: 备份文件名（可选），如果不提供，会自动生成带时间戳的名称
    """
    if not os.path.exists(MODEL_PATH):
        print(f"❌ 错误: 找不到模型文件 {MODEL_PATH}")
        print("请先训练模型！")
        return False
    
    # 创建备份目录
    backup_dir = os.path.join(BASE_DIR, 'model_backups')
    os.makedirs(backup_dir, exist_ok=True)
    
    # 生成备份文件名
    if backup_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"pointnet_model_{timestamp}.pth"
    elif not backup_name.endswith('.pth'):
        backup_name = backup_name + '.pth'
    
    backup_path = os.path.join(backup_dir, backup_name)
    
    # 复制模型文件
    try:
        shutil.copy2(MODEL_PATH, backup_path)
        file_size = os.path.getsize(backup_path) / (1024 * 1024)  # MB
        print(f"✅ 模型已备份成功！")
        print(f"   原文件: {MODEL_PATH}")
        print(f"   备份文件: {backup_path}")
        print(f"   文件大小: {file_size:.2f} MB")
        return True
    except Exception as e:
        print(f"❌ 备份失败: {e}")
        return False

def list_backups():
    """列出所有备份的模型"""
    backup_dir = os.path.join(BASE_DIR, 'model_backups')
    if not os.path.exists(backup_dir):
        print("📁 备份目录不存在，还没有备份过模型")
        return
    
    backups = [f for f in os.listdir(backup_dir) if f.endswith('.pth')]
    if not backups:
        print("📁 备份目录为空")
        return
    
    print(f"\n📦 找到 {len(backups)} 个备份模型:")
    print("-" * 60)
    for i, backup in enumerate(sorted(backups), 1):
        backup_path = os.path.join(backup_dir, backup)
        file_size = os.path.getsize(backup_path) / (1024 * 1024)  # MB
        mod_time = datetime.fromtimestamp(os.path.getmtime(backup_path))
        print(f"{i}. {backup}")
        print(f"   大小: {file_size:.2f} MB")
        print(f"   时间: {mod_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print()

def restore_model(backup_name):
    """从备份恢复模型"""
    backup_dir = os.path.join(BASE_DIR, 'model_backups')
    backup_path = os.path.join(backup_dir, backup_name)
    
    if not os.path.exists(backup_path):
        print(f"❌ 错误: 找不到备份文件 {backup_path}")
        list_backups()
        return False
    
    try:
        shutil.copy2(backup_path, MODEL_PATH)
        print(f"✅ 模型已从备份恢复！")
        print(f"   备份文件: {backup_path}")
        print(f"   恢复位置: {MODEL_PATH}")
        return True
    except Exception as e:
        print(f"❌ 恢复失败: {e}")
        return False

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='模型备份管理工具')
    parser.add_argument('--backup', type=str, nargs='?', const='auto', 
                       help='备份当前模型（可指定名称，不指定则自动生成）')
    parser.add_argument('--list', action='store_true', help='列出所有备份')
    parser.add_argument('--restore', type=str, help='从备份恢复模型（指定备份文件名）')
    
    args = parser.parse_args()
    
    if args.backup:
        if args.backup == 'auto':
            backup_model()
        else:
            backup_model(args.backup)
    elif args.list:
        list_backups()
    elif args.restore:
        restore_model(args.restore)
    else:
        # 默认行为：自动备份
        print("💾 自动备份当前模型...")
        backup_model()
        print("\n💡 提示:")
        print("   - 查看所有备份: python backup_model.py --list")
        print("   - 恢复某个备份: python backup_model.py --restore <文件名>")
        print("   - 指定名称备份: python backup_model.py --backup <名称>")

