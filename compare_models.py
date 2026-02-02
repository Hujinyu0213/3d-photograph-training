"""
模型比较工具
用于比较两个不同训练版本的模型性能
"""
import os
import sys

# 添加当前目录到 Python 路径，确保能找到 pointnet_utils 模块
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 设置 matplotlib 使用支持中文的字体，如果失败则使用英文
try:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
except:
    # 如果中文字体不可用，使用英文标签
    pass

from pointnet_utils import PointNetEncoder
import torch.nn as nn

from check_result import PointNetRegressor, DATA_DIR, LABELS_FILE, OUTPUT_DIR

def load_model(model_path, device='cpu'):
    """加载模型"""
    if not os.path.exists(model_path):
        print(f"❌ 错误: 找不到模型文件 {model_path}")
        return None
    
    model = PointNetRegressor(output_dim=27).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def evaluate_model(model, device='cpu', sample_indices=None):
    """评估模型性能"""
    if sample_indices is None:
        sample_indices = list(range(100))  # 评估所有样本
    
    csv_files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith('.csv')])
    
    try:
        labels_df = pd.read_csv(LABELS_FILE, header=None)
        all_labels = labels_df.values.astype(np.float32)
    except Exception as e:
        print(f"❌ 无法加载标签文件: {e}")
        return None
    
    all_rmse = []
    all_mae = []
    
    for idx in sample_indices:
        if idx >= len(csv_files):
            continue
            
        try:
            filename = csv_files[idx]
            df = pd.read_csv(os.path.join(DATA_DIR, filename))
            raw_points = df[['x', 'y', 'z']].values[:9].astype(np.float32)
            
            # 预处理
            centroid = np.mean(raw_points, axis=0)
            centered_points = raw_points - centroid
            
            # 预测
            input_tensor = torch.from_numpy(centered_points).unsqueeze(0).transpose(2, 1).to(device)
            with torch.no_grad():
                pred_centered, _ = model(input_tensor)
            
            # 后处理
            pred_centered_np = pred_centered.numpy().reshape(9, 3)
            pred_final = pred_centered_np + centroid
            
            # 计算误差
            gt_np = all_labels[idx].reshape(9, 3)
            rmse = np.sqrt(np.mean((pred_final - gt_np) ** 2))
            mae = np.mean(np.abs(pred_final - gt_np))
            
            all_rmse.append(rmse)
            all_mae.append(mae)
            
        except Exception as e:
            continue
    
    if not all_rmse:
        return None
    
    return {
        'rmse_mean': np.mean(all_rmse),
        'rmse_std': np.std(all_rmse),
        'rmse_min': np.min(all_rmse),
        'rmse_max': np.max(all_rmse),
        'mae_mean': np.mean(all_mae),
        'mae_std': np.std(all_mae),
        'all_rmse': all_rmse,
        'all_mae': all_mae
    }

def compare_models(model1_path, model2_path, model1_name="模型1", model2_name="模型2"):
    """比较两个模型"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("🖥️  使用设备: {}".format(device))
    
    print("=" * 60)
    print("📊 模型比较工具")
    print("=" * 60)
    
    # 加载模型
    print(f"\n📂 加载 {model1_name}...")
    model1 = load_model(model1_path, device)
    if model1 is None:
        return
    
    print(f"📂 加载 {model2_name}...")
    model2 = load_model(model2_path, device)
    if model2 is None:
        return
    
    # 评估模型（使用所有样本或前50个样本进行快速比较）
    print(f"\n🔍 评估模型性能（使用所有样本）...")
    results1 = evaluate_model(model1, device)
    results2 = evaluate_model(model2, device)
    
    if results1 is None or results2 is None:
        print("❌ 评估失败")
        return
    
    # 打印比较结果
    print("\n" + "=" * 60)
    print("📈 性能比较结果")
    print("=" * 60)
    
    print(f"\n{model1_name}:")
    print(f"  RMSE: {results1['rmse_mean']:.4f} ± {results1['rmse_std']:.4f} mm")
    print(f"  RMSE 范围: {results1['rmse_min']:.4f} - {results1['rmse_max']:.4f} mm")
    print(f"  MAE: {results1['mae_mean']:.4f} ± {results1['mae_std']:.4f} mm")
    
    print(f"\n{model2_name}:")
    print(f"  RMSE: {results2['rmse_mean']:.4f} ± {results2['rmse_std']:.4f} mm")
    print(f"  RMSE 范围: {results2['rmse_min']:.4f} - {results2['rmse_max']:.4f} mm")
    print(f"  MAE: {results2['mae_mean']:.4f} ± {results2['mae_std']:.4f} mm")
    
    # 比较
    print(f"\n{'=' * 60}")
    print("🏆 比较结果")
    print("=" * 60)
    
    rmse_diff = results2['rmse_mean'] - results1['rmse_mean']
    mae_diff = results2['mae_mean'] - results1['mae_mean']
    
    if rmse_diff < 0:
        print(f"✅ {model2_name} 的 RMSE 更好（低 {abs(rmse_diff):.4f} mm）")
    elif rmse_diff > 0:
        print(f"✅ {model1_name} 的 RMSE 更好（低 {rmse_diff:.4f} mm）")
    else:
        print("🤝 两个模型的 RMSE 相同")
    
    if mae_diff < 0:
        print(f"✅ {model2_name} 的 MAE 更好（低 {abs(mae_diff):.4f} mm）")
    elif mae_diff > 0:
        print(f"✅ {model1_name} 的 MAE 更好（低 {mae_diff:.4f} mm）")
    else:
        print("🤝 两个模型的 MAE 相同")
    
    # 绘制比较图表
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # RMSE 比较
    bp1 = axes[0].boxplot([results1['all_rmse'], results2['all_rmse']], 
                         labels=[model1_name, model2_name], patch_artist=True)
    colors1 = ['lightblue', 'lightcoral']
    for patch, color in zip(bp1['boxes'], colors1):
        patch.set_facecolor(color)
    axes[0].set_ylabel('RMSE (mm)', fontsize=11)
    axes[0].set_title('RMSE Comparison', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # MAE 比较
    bp2 = axes[1].boxplot([results1['all_mae'], results2['all_mae']], 
                         labels=[model1_name, model2_name], patch_artist=True)
    colors2 = ['lightblue', 'lightcoral']
    for patch, color in zip(bp2['boxes'], colors2):
        patch.set_facecolor(color)
    axes[1].set_ylabel('MAE (mm)', fontsize=11)
    axes[1].set_title('MAE Comparison', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    # 使用英文文件名避免编码问题，文件名：model_comparison.png
    comparison_path = os.path.join(OUTPUT_DIR, 'model_comparison.png')
    plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n📊 比较图表已保存到: {comparison_path}")
    print(f"   文件名: model_comparison.png (模型性能比较图)")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='比较两个模型')
    parser.add_argument('model1', help='第一个模型路径')
    parser.add_argument('model2', help='第二个模型路径')
    parser.add_argument('--name1', default='模型1', help='第一个模型名称')
    parser.add_argument('--name2', default='模型2', help='第二个模型名称')
    
    args = parser.parse_args()
    
    compare_models(args.model1, args.model2, args.name1, args.name2)

