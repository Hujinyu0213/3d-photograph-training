"""
在K折交叉验证的测试集上评估单次训练模型
使用与K折模型相同的测试集（10个样本，90/10划分，随机种子42）
确保公平对比
"""
import os
import sys
import io
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# 添加当前目录到 Python 路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split
from tqdm import tqdm
from pointnet_utils import PointNetEncoder, feature_transform_reguliarzer
import json

# =========================================================
# 配置
# =========================================================
EXPORT_ROOT = os.path.join(BASE_DIR, 'data', 'pointcloud')
LABELS_FILE = os.path.join(BASE_DIR, 'labels.csv')
PROJECTS_LIST_FILE = os.path.join(BASE_DIR, 'valid_projects.txt')

NUM_TARGET_POINTS = 9
OUTPUT_DIM = NUM_TARGET_POINTS * 3  # 27维

# 模型文件（单次训练模型）
MODEL_PATH = os.path.join(BASE_DIR, 'pointnet_regression_model_full_best.pth')

# GPU 配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🖥️  使用设备: {device}")
if torch.cuda.is_available():
    print(f"   GPU名称: {torch.cuda.get_device_name(0)}")

# =========================================================
# 模型定义（与训练时相同）
# =========================================================
class PointNetRegressor(nn.Module):
    def __init__(self, output_dim=27, dropout_rate=0.3):
        super(PointNetRegressor, self).__init__()
        self.feat = PointNetEncoder(global_feat=True, feature_transform=True, channel=3)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, output_dim)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.relu = nn.ReLU()

    def forward(self, x):
        x, trans, trans_feat = self.feat(x)
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.relu(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)
        return x, trans_feat

# =========================================================
# 数据加载（与训练时相同）
# =========================================================
def load_data():
    """加载完整点云和标签，使用与训练时相同的预处理"""
    print("--- 正在加载完整点云和标签 ---")
    
    if not os.path.exists(PROJECTS_LIST_FILE):
        raise FileNotFoundError(f"❌ 项目列表文件不存在: {PROJECTS_LIST_FILE}")
    
    with open(PROJECTS_LIST_FILE, 'r', encoding='utf-8') as f:
        project_names = [line.strip() for line in f if line.strip()]
    
    if not os.path.exists(LABELS_FILE):
        raise FileNotFoundError(f"❌ 标签文件不存在: {LABELS_FILE}")
    
    labels_df = pd.read_csv(LABELS_FILE, header=None)
    all_labels_np = labels_df.values.astype(np.float32)
    
    if len(project_names) != len(all_labels_np):
        min_len = min(len(project_names), len(all_labels_np))
        project_names = project_names[:min_len]
        all_labels_np = all_labels_np[:min_len]
    
    valid_features = []
    valid_labels = []
    point_counts = []
    scales_list = []  # 保存每个样本的归一化尺度
    label_centroids_list = []  # 保存每个样本的地标点质心
    
    print(f"加载 {len(project_names)} 个样本的点云...")
    
    for i, project_name in enumerate(tqdm(project_names, desc="加载点云")):
        project_dir = os.path.join(EXPORT_ROOT, project_name)
        pointcloud_file = os.path.join(project_dir, "pointcloud_full.npy")
        
        if not os.path.exists(pointcloud_file):
            continue
        
        try:
            pointcloud = np.load(pointcloud_file).astype(np.float32)
            
            if len(pointcloud) == 0:
                continue
            
            # 使用与训练时相同的预处理
            current_label = all_labels_np[i].reshape(NUM_TARGET_POINTS, 3)
            label_centroid = np.mean(current_label, axis=0)
            centered_pointcloud = pointcloud - label_centroid
            centered_label = current_label - label_centroid
            
            # 归一化
            scale = np.std(centered_pointcloud)
            if scale > 1e-6:
                centered_pointcloud = centered_pointcloud / scale
                centered_label = centered_label / scale
            else:
                scale = 1.0
            
            scales_list.append(scale)
            label_centroids_list.append(label_centroid)
            
            # 转置为 (3, N) 格式
            centered_pointcloud_T = centered_pointcloud.T
            
            valid_features.append(centered_pointcloud_T)
            valid_labels.append(centered_label.flatten())
            point_counts.append(len(pointcloud))
            
        except Exception as e:
            continue
    
    if not valid_features:
        raise RuntimeError("❌ 未能加载任何有效数据！")
    
    print(f"\n✅ 成功加载 {len(valid_features)} 个样本")
    
    # 统一采样到固定点数
    MAX_POINTS = 8192
    print(f"统一采样到 {MAX_POINTS} 个点...")
    
    processed_features = []
    for feat in valid_features:
        num_points = feat.shape[1]
        if num_points >= MAX_POINTS:
            indices = np.random.choice(num_points, MAX_POINTS, replace=False)
            sampled_feat = feat[:, indices]
        else:
            indices = np.random.choice(num_points, MAX_POINTS, replace=True)
            sampled_feat = feat[:, indices]
        processed_features.append(sampled_feat)
    
    X_np = np.array(processed_features, dtype=np.float32)
    Y_np = np.array(valid_labels, dtype=np.float32)
    
    print(f"   最终数据形状: X={X_np.shape}, Y={Y_np.shape}")
    
    return torch.from_numpy(X_np), torch.from_numpy(Y_np), scales_list, label_centroids_list

# =========================================================
# 评估函数
# =========================================================
def evaluate_single_model_on_kfold_testset():
    """在K折交叉验证的测试集上评估单次训练模型"""
    print("\n" + "="*70)
    print("在K折测试集上评估单次训练模型（公平对比）")
    print("="*70)
    
    # 加载数据
    X, Y, scales_list, label_centroids_list = load_data()
    
    # 划分测试集（与K折训练时相同的比例和随机种子）
    TEST_RATIO = 0.1  # 10%作为测试集（与K折训练时相同）
    dataset = TensorDataset(X, Y)
    test_size = int(TEST_RATIO * len(dataset))
    train_val_size = len(dataset) - test_size
    
    # 使用与K折训练时相同的随机种子（42）
    train_val_dataset, test_dataset = random_split(
        dataset, [train_val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
    
    print(f"\n📊 数据集划分（与K折训练时相同）:")
    print(f"   测试集: {len(test_dataset)} 个样本 (10%)")
    print(f"   训练+验证集: {len(train_val_dataset)} 个样本 (90%)")
    print(f"   ⚠️  注意：这是K折模型的测试集，用于公平对比")
    
    # 加载模型
    print(f"\n📦 加载单次训练模型: {MODEL_PATH}")
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"❌ 模型文件不存在: {MODEL_PATH}")
    
    model = PointNetRegressor(output_dim=OUTPUT_DIM, dropout_rate=0.3).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print("✅ 模型加载成功")
    
    # 评估
    print(f"\n🔍 开始评估...")
    all_predictions = []
    all_targets = []
    all_scales = []
    all_centroids = []
    
    # 预先获取测试集的所有索引
    if hasattr(test_dataset.indices, 'tolist'):
        test_indices_list = test_dataset.indices.tolist()
    else:
        test_indices_list = list(test_dataset.indices)
    
    with torch.no_grad():
        sample_idx = 0  # 当前处理的样本索引（在测试集中的位置）
        for batch_idx, (data, target) in enumerate(tqdm(test_loader, desc="评估")):
            data, target = data.to(device), target.to(device)
            pred, _ = model(data)
            
            # 转换为numpy
            pred_np = pred.cpu().numpy()
            target_np = target.cpu().numpy()
            batch_size = len(pred_np)
            
            all_predictions.append(pred_np)
            all_targets.append(target_np)
            
            # 获取对应的尺度和质心
            for i in range(batch_size):
                if sample_idx < len(test_indices_list):
                    orig_idx = test_indices_list[sample_idx]
                    all_scales.append(scales_list[orig_idx])
                    all_centroids.append(label_centroids_list[orig_idx])
                    sample_idx += 1
    
    # 合并所有预测和标签
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    print(f"\n✅ 评估完成")
    print(f"   测试样本数: {len(all_predictions)}")
    
    # 反归一化（恢复到原始坐标单位）
    print(f"\n🔄 反归一化预测结果...")
    predictions_denorm = []
    targets_denorm = []
    
    for i in range(len(all_predictions)):
        scale = all_scales[i]
        centroid = all_centroids[i]
        pred_norm = all_predictions[i].reshape(9, 3)
        target_norm = all_targets[i].reshape(9, 3)
        
        # 反归一化：先乘以尺度，再加回质心
        pred_denorm = pred_norm * scale + centroid
        target_denorm = target_norm * scale + centroid
        
        predictions_denorm.append(pred_denorm)
        targets_denorm.append(target_denorm)
    
    predictions_denorm = np.array(predictions_denorm)  # (N, 9, 3)
    targets_denorm = np.array(targets_denorm)  # (N, 9, 3)
    
    # 计算误差
    print(f"\n📊 计算误差统计...")
    errors = predictions_denorm - targets_denorm  # (N, 9, 3)
    
    # 每个坐标的误差
    coord_errors = errors.reshape(-1, 27)  # (N, 27)
    
    # 每个点的3D误差
    point_errors_3d = np.linalg.norm(errors, axis=2)  # (N, 9)
    
    # 每个坐标的RMSE和MAE
    coord_rmse = np.sqrt(np.mean(coord_errors**2, axis=0))  # (27,)
    coord_mae = np.mean(np.abs(coord_errors), axis=0)  # (27,)
    
    # 每个点的RMSE和MAE（3D距离）
    point_rmse_3d = np.sqrt(np.mean(point_errors_3d**2, axis=0))  # (9,)
    point_mae_3d = np.mean(point_errors_3d, axis=0)  # (9,)
    
    # 每个坐标的误差（按X, Y, Z分别）
    errors_xyz = errors  # (N, 9, 3)
    rmse_x = np.sqrt(np.mean(errors_xyz[:, :, 0]**2, axis=0))  # (9,)
    rmse_y = np.sqrt(np.mean(errors_xyz[:, :, 1]**2, axis=0))  # (9,)
    rmse_z = np.sqrt(np.mean(errors_xyz[:, :, 2]**2, axis=0))  # (9,)
    mae_x = np.mean(np.abs(errors_xyz[:, :, 0]), axis=0)  # (9,)
    mae_y = np.mean(np.abs(errors_xyz[:, :, 1]), axis=0)  # (9,)
    mae_z = np.mean(np.abs(errors_xyz[:, :, 2]), axis=0)  # (9,)
    
    # 打印详细分析
    print("\n" + "="*70)
    print("详细分析：所有9个地标点的性能")
    print("="*70)
    
    # 总体统计
    overall_rmse = np.sqrt(np.mean(coord_errors**2))
    overall_mae = np.mean(np.abs(coord_errors))
    overall_rmse_3d = np.sqrt(np.mean(point_errors_3d**2))
    overall_mae_3d = np.mean(point_errors_3d)
    
    # 计算平均精度（基于欧氏距离，使用不同阈值）
    thresholds = [1.0, 2.0, 5.0, 10.0]  # mm
    mean_precision = {}
    for threshold in thresholds:
        precision_per_point = []
        for point_idx in range(9):
            errors_point = point_errors_3d[:, point_idx]
            precision = np.sum(errors_point < threshold) / len(errors_point) * 100
            precision_per_point.append(precision)
        mean_precision[threshold] = np.mean(precision_per_point)
    
    print(f"\n📊 总体性能:")
    print(f"   所有坐标的RMSE: {overall_rmse:.4f}")
    print(f"   所有坐标的MAE: {overall_mae:.4f}")
    print(f"   所有点的3D RMSE: {overall_rmse_3d:.4f}")
    print(f"   所有点的3D MAE: {overall_mae_3d:.4f}")
    print(f"\n📊 平均精度（基于欧氏距离）:")
    for threshold in thresholds:
        print(f"   平均精度 @ {threshold}mm: {mean_precision[threshold]:.2f}%")
    
    # 保存结果
    results = {
        'overall': {
            'rmse_all_coords': float(overall_rmse),
            'mae_all_coords': float(overall_mae),
            'rmse_3d_all_points': float(overall_rmse_3d),
            'mae_3d_all_points': float(overall_mae_3d),
            'mean_precision_1mm': float(mean_precision[1.0]),
            'mean_precision_2mm': float(mean_precision[2.0]),
            'mean_precision_5mm': float(mean_precision[5.0]),
            'mean_precision_10mm': float(mean_precision[10.0])
        },
        'points': []
    }
    
    # 地标点名称
    landmark_names = ['Glabella', 'Nasion', 'Rhinion', 'Nasal Tip', 'Subnasale', 
                      'Alare (R)', 'Alare (L)', 'Zygion (R)', 'Zygion (L)']
    
    for point_idx in range(9):
        errors_point = point_errors_3d[:, point_idx]
        point_results = {
            'point_id': point_idx + 1,
            'landmark_name': landmark_names[point_idx],
            'rmse_3d': float(point_rmse_3d[point_idx]),
            'mae_3d': float(point_mae_3d[point_idx]),
            'rmse_x': float(rmse_x[point_idx]),
            'rmse_y': float(rmse_y[point_idx]),
            'rmse_z': float(rmse_z[point_idx]),
            'mae_x': float(mae_x[point_idx]),
            'mae_y': float(mae_y[point_idx]),
            'mae_z': float(mae_z[point_idx]),
            'max_error': float(np.max(point_errors_3d[:, point_idx])),
            'min_error': float(np.min(point_errors_3d[:, point_idx])),
            'median_error': float(np.median(point_errors_3d[:, point_idx])),
            'std_error': float(np.std(point_errors_3d[:, point_idx])),
            'precision_1mm': float(np.sum(errors_point < 1) / len(errors_point) * 100),
            'precision_2mm': float(np.sum(errors_point < 2) / len(errors_point) * 100),
            'precision_5mm': float(np.sum(errors_point < 5) / len(errors_point) * 100),
            'precision_10mm': float(np.sum(errors_point < 10) / len(errors_point) * 100)
        }
        results['points'].append(point_results)
    
    results_file = os.path.join(BASE_DIR, 'single_model_on_kfold_testset_results.json')
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ 结果已保存: {results_file}")
    
    # 总结
    print(f"\n" + "="*70)
    print("总结")
    print("="*70)
    
    points_under_2mm = np.sum(point_rmse_3d < 2)
    print(f"\n达到2mm目标的点数: {points_under_2mm}/9 ({points_under_2mm/9*100:.1f}%)")
    
    avg_rmse = np.mean(point_rmse_3d)
    print(f"平均3D RMSE: {avg_rmse:.4f} mm")
    
    print(f"\n📊 平均精度（基于欧氏距离）:")
    print(f"   平均精度 @ 1mm: {mean_precision[1.0]:.2f}%")
    print(f"   平均精度 @ 2mm: {mean_precision[2.0]:.2f}%")
    print(f"   平均精度 @ 5mm: {mean_precision[5.0]:.2f}%")
    print(f"   平均精度 @ 10mm: {mean_precision[10.0]:.2f}%")
    
    if avg_rmse < 2:
        print(f"✅ 总体性能优秀！平均误差小于2mm")
    elif avg_rmse < 5:
        print(f"⚠️  总体性能良好，但需要进一步改进")
    else:
        print(f"❌ 需要显著改进")
    
    if mean_precision[2.0] >= 80:
        print(f"✅ 平均精度优秀！2mm精度达到{mean_precision[2.0]:.1f}%")
    elif mean_precision[2.0] >= 50:
        print(f"⚠️  平均精度良好，2mm精度为{mean_precision[2.0]:.1f}%，需要改进")
    else:
        print(f"❌ 平均精度较低，2mm精度仅为{mean_precision[2.0]:.1f}%，需要显著改进")
    
    print("="*70)
    
    return results

if __name__ == "__main__":
    evaluate_single_model_on_kfold_testset()
