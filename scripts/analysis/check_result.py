import os
import sys

# 添加当前目录到 Python 路径，确保能找到 pointnet_utils 模块
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端，避免显示警告
import matplotlib.pyplot as plt
import pandas as pd
import torch.nn as nn
from pointnet_utils import PointNetEncoder

class PointNetRegressor(nn.Module):
    def __init__(self, output_dim=27):
        super(PointNetRegressor, self).__init__()
        self.feat = PointNetEncoder(global_feat=True, feature_transform=True, channel=3)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, output_dim) 
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(p=0.4)
        self.relu = nn.ReLU()

    def forward(self, x):
        x, trans, trans_feat = self.feat(x)
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.relu(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x) 
        return x, trans_feat

# 配置（BASE_DIR 已在上面定义）
DATA_DIR = os.path.join(BASE_DIR, 'data') 
LABELS_FILE = os.path.join(BASE_DIR, 'labels.csv')
DEFAULT_MODEL_PATH = os.path.join(BASE_DIR, 'pointnet_regression_model.pth')  # 默认模型路径
OUTPUT_DIR = os.path.join(BASE_DIR, 'results')  # 保存图片的目录

def visualize_result(evaluate_all=False, model_path=None):
    """
    评估模型性能
    
    参数:
        evaluate_all: 如果为True，评估所有样本；如果为False，只评估样本1, 11, 51
        model_path: 模型文件路径，如果为None则使用默认路径
    """
    # 确定使用的模型路径
    if model_path is None:
        model_path = DEFAULT_MODEL_PATH
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  使用设备: {device}")
    model = PointNetRegressor(output_dim=27).to(device)
    
    if not os.path.exists(model_path):
        print(f"❌ 错误: 找不到模型文件 {model_path}")
        print("请先训练模型或检查模型路径！")
        return
    
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"📂 加载模型: {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    csv_files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith('.csv')])
    
    try:
        if not os.path.exists(LABELS_FILE):
            print(f"❌ 错误: 找不到文件 {LABELS_FILE}")
            print("请先运行 create_lable.py 创建标签文件")
            return
        
        labels_df = pd.read_csv(LABELS_FILE, header=None)
        all_labels = labels_df.values.astype(np.float32)
        print(f"✅ 成功加载 labels.csv，共 {len(all_labels)} 个样本")
    except Exception as e:
        print(f"❌ 无法加载 labels.csv: {e}")
        print(f"文件路径: {LABELS_FILE}")
        print("请检查文件格式是否正确，或重新运行 create_lable.py")
        return

    # 地标点名称（用于显示）
    landmark_names = ['Glabella', 'Nasion', 'Rhinion', 'Nasal Tip', 'Subnasale', 
                      'Alare (R)', 'Alare (L)', 'Zygion (R)', 'Zygion (L)']
    
    # 存储所有样本的误差统计
    all_rmse = []
    all_mae = []
    all_point_errors = []  # 每个点的误差
    all_axis_errors = {'x': [], 'y': [], 'z': []}
    
    # 选择要评估的样本
    if evaluate_all:
        sample_indices = list(range(len(csv_files)))
        print(f"📊 评估所有 {len(sample_indices)} 个样本...")
    else:
        sample_indices = [1, 11, 51]
        print(f"📊 评估样本: {sample_indices}")
    
    # 检查样本
    for idx in sample_indices:
        filename = csv_files[idx]
        print(f"\n{'='*60}")
        print(f"--- Sample {idx}: {filename} ---")
        print(f"{'='*60}")
        
        try:
            # 1. 原始数据
            df = pd.read_csv(os.path.join(DATA_DIR, filename))
            raw_points = df[['x', 'y', 'z']].values[:9].astype(np.float32)
            
            # 2. 预处理：计算质心并去中心化
            centroid = np.mean(raw_points, axis=0)
            centered_points = raw_points - centroid
            
            # 3. 预测
            input_tensor = torch.from_numpy(centered_points).unsqueeze(0).transpose(2, 1).to(device)
            with torch.no_grad():
                pred_centered, _ = model(input_tensor)
            
            # 4. 后处理：还原位置
            pred_centered_np = pred_centered.numpy().reshape(9, 3)
            pred_final = pred_centered_np + centroid
            
            # 5. 计算详细误差
            gt_np = all_labels[idx].reshape(9, 3)
            
            # 整体误差
            rmse = np.sqrt(np.mean((pred_final - gt_np) ** 2))
            mae = np.mean(np.abs(pred_final - gt_np))  # 平均绝对误差
            all_rmse.append(rmse)
            all_mae.append(mae)
            
            # 每个点的3D欧氏距离误差
            point_errors = np.sqrt(np.sum((pred_final - gt_np) ** 2, axis=1))
            all_point_errors.append(point_errors)
            
            # 每个坐标轴的误差
            axis_errors = np.abs(pred_final - gt_np)  # (9, 3)
            x_error = np.mean(axis_errors[:, 0])
            y_error = np.mean(axis_errors[:, 1])
            z_error = np.mean(axis_errors[:, 2])
            all_axis_errors['x'].append(x_error)
            all_axis_errors['y'].append(y_error)
            all_axis_errors['z'].append(z_error)
            
            print(f"\n✅ 整体误差 (单位: 毫米 mm):")
            print(f"   RMSE (均方根误差): {rmse:.4f} mm")
            print(f"   MAE (平均绝对误差): {mae:.4f} mm")
            print(f"\n   各坐标轴平均误差 (单位: 毫米 mm):")
            print(f"   X轴: {x_error:.4f} mm")
            print(f"   Y轴: {y_error:.4f} mm")
            print(f"   Z轴: {z_error:.4f} mm")
            print(f"\n   每个地标点的3D距离误差 (单位: 毫米 mm):")
            for i in range(9):
                print(f"   {landmark_names[i]:15s}: {point_errors[i]:.4f} mm")
            
            # 显示每个点的详细坐标误差
            print(f"\n   每个点的详细坐标误差 (单位: 毫米 mm, 格式: X, Y, Z):")
            for i in range(9):
                x_err, y_err, z_err = axis_errors[i]
                print(f"   {landmark_names[i]:15s}: X={x_err:.4f} mm, Y={y_err:.4f} mm, Z={z_err:.4f} mm")

            # 6. 绘图 (只对部分样本生成可视化，或评估所有时只生成前几个)
            if not evaluate_all or idx < 5:  # 只生成前5个样本的可视化
                fig = plt.figure(figsize=(8, 6))
                ax = fig.add_subplot(111, projection='3d')
                
                # Ground Truth (Green)
                ax.scatter(gt_np[:,0], gt_np[:,1], gt_np[:,2], c='g', s=50, label='Ground Truth')
                
                # Prediction (Red)
                ax.scatter(pred_final[:,0], pred_final[:,1], pred_final[:,2], c='r', marker='^', s=50, label='Prediction')
                
                for i in range(9):
                    ax.plot([gt_np[i,0], pred_final[i,0]], [gt_np[i,1], pred_final[i,1]], [gt_np[i,2], pred_final[i,2]], 'gray', linestyle='--')
                
                ax.set_title(f'Sample {idx} - RMSE: {rmse:.2f}')
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_zlabel('Z')
                plt.legend()
                
            # 保存图片而不是显示
            # 使用英文文件名避免编码问题，文件名格式：sample_索引_结果.png
            output_path = os.path.join(OUTPUT_DIR, f'sample_{idx:03d}_result.png')
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()  # 关闭图形以释放内存
            print(f"   图片已保存到: {output_path}")
            print(f"   文件名: sample_{idx:03d}_result.png (样本 {idx} 的预测结果)")
            
        except Exception as e:
            print(f"Skipping sample {idx}: {e}")
    
    # 计算所有样本的统计信息
    if all_rmse:
        print(f"\n{'='*60}")
        print("📊 所有测试样本的统计信息")
        print(f"{'='*60}")
        print(f"\n整体误差统计 (单位: 毫米 mm):")
        print(f"   RMSE - 平均值: {np.mean(all_rmse):.4f} mm, 标准差: {np.std(all_rmse):.4f} mm")
        print(f"   RMSE - 最小值: {np.min(all_rmse):.4f} mm, 最大值: {np.max(all_rmse):.4f} mm")
        print(f"   MAE - 平均值: {np.mean(all_mae):.4f} mm, 标准差: {np.std(all_mae):.4f} mm")
        print(f"   MAE - 最小值: {np.min(all_mae):.4f} mm, 最大值: {np.max(all_mae):.4f} mm")
        
        print(f"\n各坐标轴误差统计 (单位: 毫米 mm):")
        for axis in ['x', 'y', 'z']:
            errors = all_axis_errors[axis]
            print(f"   {axis.upper()}轴 - 平均值: {np.mean(errors):.4f} mm, 标准差: {np.std(errors):.4f} mm")
            print(f"   {axis.upper()}轴 - 最小值: {np.min(errors):.4f} mm, 最大值: {np.max(errors):.4f} mm")
        
        # 计算每个地标点的平均误差
        print(f"\n每个地标点的平均3D距离误差 (单位: 毫米 mm):")
        point_errors_array = np.array(all_point_errors)  # (n_samples, 9)
        for i in range(9):
            avg_error = np.mean(point_errors_array[:, i])
            std_error = np.std(point_errors_array[:, i])
            print(f"   {landmark_names[i]:15s}: {avg_error:.4f} ± {std_error:.4f} mm")
        
        # 保存详细误差报告到CSV
        report_path = os.path.join(OUTPUT_DIR, 'error_report.csv')
        report_data = {
            'Sample': sample_indices[:len(all_rmse)],
            'RMSE': all_rmse,
            'MAE': all_mae,
            'X_Error': all_axis_errors['x'],
            'Y_Error': all_axis_errors['y'],
            'Z_Error': all_axis_errors['z']
        }
        # 添加每个点的误差
        for i in range(9):
            report_data[f'{landmark_names[i]}_Error'] = [all_point_errors[j][i] for j in range(len(all_point_errors))]
        
        report_df = pd.DataFrame(report_data)
        report_df.to_csv(report_path, index=False)
        print(f"\n✅ 详细误差报告已保存到: {report_path}")
        print(f"   报告包含 {len(report_df)} 个样本的详细误差信息")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='评估 PointNet 回归模型')
    parser.add_argument('--all', action='store_true', help='评估所有样本（默认只评估样本1, 11, 51）')
    parser.add_argument('--model', type=str, default=None, 
                       help='模型文件路径（默认: pointnet_regression_model.pth）')
    args = parser.parse_args()
    
    visualize_result(evaluate_all=args.all, model_path=args.model)