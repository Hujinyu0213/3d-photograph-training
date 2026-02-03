"""
评估调优后的模型并与所有历史模型对比
"""
import os
import sys
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

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

def load_test_data():
    with open(PROJECTS_LIST_FILE, 'r', encoding='utf-8') as f:
        project_names = [ln.strip() for ln in f if ln.strip()]
    labels_df = pd.read_csv(LABELS_FILE, header=None)
    labels_np = labels_df.values.astype(np.float32)
    
    test_size = max(1, int(0.2 * len(project_names)))
    test_indices = list(range(len(project_names) - test_size, len(project_names)))
    
    feats, labels, names, scales = [], [], [], []
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

def evaluate_model(model_path, model_name, X, Y, scales):
    print(f"\n{'='*80}")
    print(f"评估模型: {model_name}")
    print(f"{'='*80}")
    
    if not os.path.exists(model_path):
        print(f"⚠️  模型文件不存在: {model_path}")
        return None
    
    model = PointNetRegressor(output_dim=OUTPUT_DIM, dropout_rate=0.3).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
    model.eval()
    
    predictions = []
    with torch.no_grad():
        for i in range(len(X)):
            data = X[i:i+1].to(device)
            pred, _ = model(data)
            predictions.append(pred.cpu().numpy()[0])
    
    predictions = np.array(predictions)
    targets = Y.numpy()
    
    scales_expanded = scales[:, None, None]
    preds_mm = predictions.reshape(len(X), NUM_TARGET_POINTS, 3) * scales_expanded
    targets_mm = targets.reshape(len(X), NUM_TARGET_POINTS, 3) * scales_expanded
    
    errors_mm = preds_mm - targets_mm
    mse_mm = np.mean(errors_mm ** 2)
    rmse_mm = np.sqrt(mse_mm)
    mae_mm = np.mean(np.abs(errors_mm))
    
    point_errors = []
    for i in range(NUM_TARGET_POINTS):
        pred_pt = preds_mm[:, i, :]
        true_pt = targets_mm[:, i, :]
        dist_3d = np.sqrt(np.sum((pred_pt - true_pt)**2, axis=1))
        point_errors.append({
            'point_id': i+1,
            'rmse_3d_mm': float(np.sqrt(np.mean(dist_3d**2))),
            'mae_3d_mm': float(np.mean(dist_3d))
        })
    
    results = {
        'model_name': model_name,
        'model_path': model_path,
        'overall': {
            'mse_mm': float(mse_mm),
            'rmse_mm': float(rmse_mm),
            'mae_mm': float(mae_mm)
        },
        'per_point': point_errors
    }
    
    print(f"  RMSE: {rmse_mm:.4f} mm")
    print(f"  MAE:  {mae_mm:.4f} mm")
    
    return results

def main():
    print("="*80)
    print("🧪 超参数调优模型 vs 所有历史模型 - 测试集对比")
    print("="*80)
    
    X, Y, names, scales = load_test_data()
    print(f"\n测试集样本数: {len(X)}")
    
    # 定义所有要比较的模型
    models = {
        '超参数调优K折模型': 'pointnet_regression_model_tuned_kfold_best.pth',
        'K折+增强+FPS': 'pointnet_regression_model_kfold_aug_fps_best.pth',
        '单次训练+增强+FPS': 'pointnet_regression_model_full_aug_fps_best.pth',
        '旧K折模型（无增强）': 'pointnet_regression_model_kfold_best.pth'
    }
    
    all_results = {}
    
    for model_name, model_file in models.items():
        model_path = os.path.join(ROOT_DIR, "models", model_file)
        result = evaluate_model(model_path, model_name, X, Y, scales)
        if result:
            all_results[model_name] = result
    
    # 打印对比表格
    print(f"\n{'='*80}")
    print("📊 测试集性能对比 (毫米)")
    print(f"{'='*80}")
    
    print(f"\n{'模型':<30} {'RMSE(mm)':>15} {'MAE(mm)':>15} {'vs 旧K折':>15}")
    print("-"*80)
    
    baseline_rmse = None
    if '旧K折模型（无增强）' in all_results:
        baseline_rmse = all_results['旧K折模型（无增强）']['overall']['rmse_mm']
    
    for model_name, result in all_results.items():
        rmse = result['overall']['rmse_mm']
        mae = result['overall']['mae_mm']
        
        if baseline_rmse and baseline_rmse > 0:
            improvement = ((baseline_rmse - rmse) / baseline_rmse) * 100
            imp_str = f"{improvement:+.2f}%"
        else:
            imp_str = "N/A"
        
        print(f"{model_name:<30} {rmse:>15.4f} {mae:>15.4f} {imp_str:>15}")
    
    # 地标点详细对比
    landmark_names = ['Glabella', 'Nasion', 'Rhinion', 'Nasal Tip', 'Subnasale',
                      'Alare (R)', 'Alare (L)', 'Zygion (R)', 'Zygion (L)']
    
    print(f"\n{'='*80}")
    print("📍 每个地标点 RMSE 对比 (mm)")
    print(f"{'='*80}")
    
    print(f"\n{'地标点':<15}", end="")
    for model_name in all_results.keys():
        short_name = model_name[:12]
        print(f"{short_name:>15}", end="")
    print()
    print("-"*80)
    
    for i in range(NUM_TARGET_POINTS):
        name = landmark_names[i] if i < len(landmark_names) else f"Point {i+1}"
        print(f"{name:<15}", end="")
        for result in all_results.values():
            rmse = result['per_point'][i]['rmse_3d_mm']
            print(f"{rmse:>15.4f}", end="")
        print()
    
    # 保存结果
    output_path = os.path.join(ROOT_DIR, "results", "test_evaluations", "test_comparison_all_models_with_tuned.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n📁 详细结果已保存: {output_path}")
    
    # 结论
    print(f"\n{'='*80}")
    print("💡 结论")
    print(f"{'='*80}")
    
    if '超参数调优K折模型' in all_results:
        tuned_rmse = all_results['超参数调优K折模型']['overall']['rmse_mm']
        print(f"\n🎯 超参数调优模型测试集 RMSE: {tuned_rmse:.4f} mm")
        
        if baseline_rmse:
            improvement = ((baseline_rmse - tuned_rmse) / baseline_rmse) * 100
            print(f"   相比旧K折模型改进: {improvement:.2f}%")
        
        if tuned_rmse < 10:
            print("\n✅ 已达到 10mm 以内精度！")
        elif tuned_rmse < 12:
            print("\n⚠️  接近目标，建议继续优化或增加训练数据")
        else:
            print("\n⚠️  距离 2mm 目标仍有差距，建议:")
            print("   1. 收集更多训练样本")
            print("   2. 尝试 PointNet++ 架构")
            print("   3. 分析高误差地标点特征")
    
    print("="*80)

if __name__ == "__main__":
    main()
