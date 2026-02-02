"""
评估增强+FPS模型在测试集上的性能
并与K折模型对比
"""
import os
import sys
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# 路径设置
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(BASE_DIR))
UTILS_DIR = os.path.join(ROOT_DIR, "scripts", "utils")
for p in (ROOT_DIR, UTILS_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

import importlib.util
_pn_path = os.path.join(UTILS_DIR, "pointnet_utils.py")
spec = importlib.util.spec_from_file_location("pointnet_utils", _pn_path)
pointnet_utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pointnet_utils)
PointNetEncoder = pointnet_utils.PointNetEncoder

EXPORT_ROOT = os.path.join(ROOT_DIR, 'data', 'pointcloud')
LABELS_FILE = os.path.join(ROOT_DIR, 'results', 'labels.csv')
PROJECTS_LIST_FILE = os.path.join(ROOT_DIR, 'results', 'valid_projects.txt')

NUM_TARGET_POINTS = 9
OUTPUT_DIM = NUM_TARGET_POINTS * 3
MAX_POINTS = 8192

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 模型定义
class PointNetRegressor(nn.Module):
    def __init__(self, output_dim=27, dropout_rate=0.3):
        super().__init__()
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

# FPS
def farthest_point_sampling(points_np, n_samples):
    N = points_np.shape[0]
    if N <= n_samples:
        idx = np.random.choice(N, n_samples, replace=True)
        return points_np[idx]
    farthest_pts = np.zeros((n_samples,), dtype=np.int64)
    distances = np.full((N,), np.inf)
    farthest = np.random.randint(0, N)
    for i in range(n_samples):
        farthest_pts[i] = farthest
        centroid = points_np[farthest][None, :]
        dist = np.sum((points_np - centroid) ** 2, axis=1)
        distances = np.minimum(distances, dist)
        farthest = np.argmax(distances)
    return points_np[farthest_pts]

# 加载数据
def load_test_data():
    with open(PROJECTS_LIST_FILE, 'r', encoding='utf-8') as f:
        project_names = [ln.strip() for ln in f if ln.strip()]
    labels_df = pd.read_csv(LABELS_FILE, header=None)
    labels_np = labels_df.values.astype(np.float32)
    
    # 使用与训练相同的划分（后20%作为测试集）
    test_size = max(1, int(0.2 * len(project_names)))
    test_indices = list(range(len(project_names) - test_size, len(project_names)))
    
    feats = []
    labels = []
    names = []
    scales = []
    
    for idx in test_indices:
        name = project_names[idx]
        pc_path = os.path.join(EXPORT_ROOT, name, "pointcloud_full.npy")
        if not os.path.exists(pc_path):
            continue
        pc = np.load(pc_path).astype(np.float32)
        if pc.shape[0] == 0:
            continue
        
        label = labels_np[idx].reshape(NUM_TARGET_POINTS, 3)
        label_centroid = np.mean(label, axis=0)
        pc_centered = pc - label_centroid
        label_centered = label - label_centroid
        
        scale = np.std(pc_centered)
        if scale > 1e-6:
            pc_centered /= scale
            label_centered /= scale
        
        pc_sampled = farthest_point_sampling(pc_centered, MAX_POINTS)
        feats.append(pc_sampled.T)
        labels.append(label_centered.flatten())
        names.append(name)
        scales.append(scale if scale > 1e-6 else 1.0)
    
    X = torch.from_numpy(np.stack(feats, axis=0)).float()
    Y = torch.from_numpy(np.stack(labels, axis=0)).float()
    scales = np.array(scales, dtype=np.float32)
    return X, Y, names, scales

# 评估函数
def evaluate_model(model_path, model_name):
    print(f"\n{'='*60}")
    print(f"评估模型: {model_name}")
    print(f"模型路径: {model_path}")
    print(f"{'='*60}")
    
    X, Y, names, scales = load_test_data()
    print(f"测试集样本数: {len(X)}")
    
    model = PointNetRegressor(output_dim=OUTPUT_DIM, dropout_rate=0.3).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    predictions = []
    with torch.no_grad():
        for i in range(len(X)):
            data = X[i:i+1].to(device)
            pred, _ = model(data)
            predictions.append(pred.cpu().numpy()[0])
    
    predictions = np.array(predictions)          # normalized
    targets = Y.numpy()                          # normalized

    # 反归一化到原始单位（假设原始单位为毫米）
    scales_expanded = scales[:, None, None]      # (N,1,1)
    preds_mm = predictions.reshape(len(X), NUM_TARGET_POINTS, 3) * scales_expanded
    targets_mm = targets.reshape(len(X), NUM_TARGET_POINTS, 3) * scales_expanded

    # 计算指标（毫米）
    errors_mm = preds_mm - targets_mm
    mse_mm = np.mean(errors_mm ** 2)
    rmse_mm = np.sqrt(mse_mm)
    mae_mm = np.mean(np.abs(errors_mm))

    # 每个点的3D误差（毫米）
    point_errors = []
    for i in range(NUM_TARGET_POINTS):
        pred_pt = preds_mm[:, i, :]
        true_pt = targets_mm[:, i, :]
        dist_3d = np.sqrt(np.sum((pred_pt - true_pt)**2, axis=1))
        point_errors.append({
            'point_id': i+1,
            'rmse_3d_mm': float(np.sqrt(np.mean(dist_3d**2))),
            'mae_3d_mm': float(np.mean(dist_3d)),
            'max_mm': float(np.max(dist_3d)),
            'min_mm': float(np.min(dist_3d))
        })

    results = {
        'model_name': model_name,
        'test_samples': len(X),
        'overall': {
            'mse_mm': float(mse_mm),
            'rmse_mm': float(rmse_mm),
            'mae_mm': float(mae_mm)
        },
        'per_point': point_errors
    }
    
    return results

# 主函数
def main():
    print("="*60)
    print("🧪 增强+FPS模型 vs K折模型测试集对比评估")
    print("="*60)
    
    # 评估新模型
    aug_fps_model = os.path.join(ROOT_DIR, "models", "pointnet_regression_model_full_aug_fps_best.pth")
    aug_results = evaluate_model(aug_fps_model, "数据增强+FPS模型")
    
    # 评估K折+增强+FPS模型
    kfold_aug_fps_model = os.path.join(ROOT_DIR, "models", "pointnet_regression_model_kfold_aug_fps_best.pth")
    if os.path.exists(kfold_aug_fps_model):
        kfold_results = evaluate_model(kfold_aug_fps_model, "K折+增强+FPS模型")
    else:
        kfold_results = None
        print("\n⚠️  K折+增强+FPS模型文件不存在，跳过对比")
    
    # 打印对比结果
    print(f"\n{'='*60}")
    print("📊 测试集性能对比 (毫米)")
    print(f"{'='*60}")
    
    print(f"\n整体指标对比 (毫米):")
    print(f"{'指标':<20} {'增强+FPS':>15} {'K折模型':>15} {'改进':>10}")
    print("-"*60)
    
    metrics = ['mse_mm', 'rmse_mm', 'mae_mm']
    metric_names = {'mse_mm': 'MSE(mm^2)', 'rmse_mm': 'RMSE(mm)', 'mae_mm': 'MAE(mm)'}
    
    for metric in metrics:
        aug_val = aug_results['overall'][metric]
        if kfold_results:
            kfold_val = kfold_results['overall'][metric]
            improvement = ((kfold_val - aug_val) / kfold_val) * 100
            print(f"{metric_names[metric]:<20} {aug_val:>15.6f} {kfold_val:>15.6f} {improvement:>9.2f}%")
        else:
            print(f"{metric_names[metric]:<20} {aug_val:>15.6f} {'N/A':>15} {'N/A':>10}")
    
    # 每个地标点对比
    landmark_names = ['Glabella', 'Nasion', 'Rhinion', 'Nasal Tip', 'Subnasale',
                      'Alare (R)', 'Alare (L)', 'Zygion (R)', 'Zygion (L)']
    
    print(f"\n每个地标点3D误差对比 (RMSE, mm):")
    print(f"{'地标点':<15} {'增强+FPS':>12} {'K折模型':>12} {'改进':>10}")
    print("-"*60)
    
    for i in range(NUM_TARGET_POINTS):
        aug_rmse = aug_results['per_point'][i]['rmse_3d_mm']
        name = landmark_names[i] if i < len(landmark_names) else f"Point {i+1}"
        if kfold_results:
            kfold_rmse = kfold_results['per_point'][i]['rmse_3d_mm']
            improvement = ((kfold_rmse - aug_rmse) / kfold_rmse) * 100
            print(f"{name:<15} {aug_rmse:>12.6f} {kfold_rmse:>12.6f} {improvement:>9.2f}%")
        else:
            print(f"{name:<15} {aug_rmse:>12.6f} {'N/A':>12} {'N/A':>10}")
    
    # 保存结果
    output = {
        'aug_fps_model': aug_results,
        'kfold_model': kfold_results
    }
    
    output_path = os.path.join(ROOT_DIR, "results", "test_comparison_aug_fps_vs_kfold_aug_fps.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"\n📁 详细结果已保存: {output_path}")
    
    # 结论
    print(f"\n{'='*60}")
    print("💡 结论与建议")
    print(f"{'='*60}")
    
    if kfold_results:
        overall_improvement = ((kfold_results['overall']['rmse_mm'] - aug_results['overall']['rmse_mm']) / 
                              kfold_results['overall']['rmse_mm']) * 100
        
        if overall_improvement > 10:
            print(f"✅ 增强+FPS模型显著优于K折模型 (RMSE改进 {overall_improvement:.2f}%)")
            print(f"   建议: 采用增强+FPS模型作为最终模型")
        elif overall_improvement > 0:
            print(f"✅ 增强+FPS模型略优于K折模型 (RMSE改进 {overall_improvement:.2f}%)")
            print(f"   建议: 可以使用增强+FPS模型，或结合两者优点")
        else:
            print(f"⚠️  K折模型在测试集上表现更好")
            print(f"   建议: 检查增强参数是否过强，或增加训练样本")
    
    print(f"\n下一步:")
    print(f"  1. 如果增强+FPS效果好，可以用它做K折交叉验证")
    print(f"  2. 尝试 PointNet++ 或其他架构")
    print(f"  3. 收集更多训练数据")
    print(f"  4. 调整增强参数以获得更好的泛化")
    
    print("="*60)

if __name__ == "__main__":
    main()
