"""
K折交叉验证超参数调优
测试不同的超参数组合，找到最佳配置
"""
import os
import sys
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import KFold
from tqdm import tqdm
from itertools import product
import time

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
feature_transform_reguliarzer = pointnet_utils.feature_transform_reguliarzer

EXPORT_ROOT = os.path.join(ROOT_DIR, 'data', 'pointcloud')
LABELS_FILE = os.path.join(ROOT_DIR, 'results', 'labels.csv')
PROJECTS_LIST_FILE = os.path.join(ROOT_DIR, 'results', 'valid_projects.txt')

NUM_TARGET_POINTS = 9
OUTPUT_DIM = NUM_TARGET_POINTS * 3
MAX_POINTS = 8192
K_FOLDS = 5
RANDOM_SEED = 42
NUM_EPOCHS = 80  # 更短轮次用于快速调参

# 调参策略
USE_RANDOM_SEARCH = True          # True: 随机采样; False: 全量网格
MAX_RANDOM_TRIALS = 20            # 随机采样的组合数

# 超参数搜索空间
PARAM_GRID = {
    # 学习率聚焦在已知表现好的 1e-3 附近
    'learning_rate': [0.0007, 0.001, 0.0015],
    # 批大小压缩到 8/12，兼顾显存与稳定性
    'batch_size': [8, 12],
    # Dropout 中等范围
    'dropout_rate': [0.25, 0.35],
    # 学习率衰减步长适配 80/120 轮
    'lr_decay_step': [80, 120],
    # 衰减幅度两档
    'lr_decay_gamma': [0.5, 0.7],
    # 特征变换正则权重两档
    'feature_transform_weight': [0.0005, 0.001]
}

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

def augment(pc, label):
    theta = np.random.uniform(-np.pi/12, np.pi/12)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    rot = np.array([[cos_t, -sin_t, 0], [sin_t, cos_t, 0], [0, 0, 1]], dtype=np.float32)
    pc = pc @ rot.T
    label_r = label.reshape(NUM_TARGET_POINTS, 3) @ rot.T
    
    scale = np.random.uniform(0.95, 1.05)
    pc *= scale
    label_r *= scale
    
    shift = np.random.uniform(-0.02, 0.02, size=(1, 3))
    pc += shift
    label_r += shift
    
    jitter = np.random.normal(0, 0.005, size=pc.shape).astype(np.float32)
    pc += jitter
    
    return pc, label_r.flatten()

def load_data():
    with open(PROJECTS_LIST_FILE, 'r', encoding='utf-8') as f:
        project_names = [ln.strip() for ln in f if ln.strip()]
    labels_df = pd.read_csv(LABELS_FILE, header=None)
    labels_np = labels_df.values.astype(np.float32)
    
    feats, labels = [], []
    for i, name in enumerate(project_names):
        pc_path = os.path.join(EXPORT_ROOT, name, "pointcloud_full.npy")
        if not os.path.exists(pc_path):
            continue
        pc = np.load(pc_path).astype(np.float32)
        if pc.shape[0] == 0:
            continue
        
        label = labels_np[i].reshape(NUM_TARGET_POINTS, 3)
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
    
    X = np.stack(feats, axis=0)
    Y = np.stack(labels, axis=0)
    return X, Y

def train_kfold_with_params(X, Y, params, trial_num):
    """使用给定的超参数进行K折训练"""
    lr = params['learning_rate']
    bs = params['batch_size']
    dropout = params['dropout_rate']
    decay_step = params['lr_decay_step']
    decay_gamma = params['lr_decay_gamma']
    ft_weight = params['feature_transform_weight']
    
    print(f"\n{'='*70}")
    print(f"🧪 Trial {trial_num}: LR={lr}, BS={bs}, Dropout={dropout}, DecayStep={decay_step}, Gamma={decay_gamma}, FT={ft_weight}")
    print(f"{'='*70}")
    
    kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    fold_losses = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X), start=1):
        X_train, X_val = X[train_idx], X[val_idx]
        Y_train, Y_val = Y[train_idx], Y[val_idx]
        
        model = PointNetRegressor(output_dim=OUTPUT_DIM, dropout_rate=dropout).to(device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=decay_step, gamma=decay_gamma)
        
        X_train_t = torch.from_numpy(X_train).float()
        Y_train_t = torch.from_numpy(Y_train).float()
        X_val_t = torch.from_numpy(X_val).float().to(device)
        Y_val_t = torch.from_numpy(Y_val).float().to(device)
        
        train_dataset = TensorDataset(X_train_t, Y_train_t)
        train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
        
        best_val_loss = float('inf')
        
        for epoch in range(NUM_EPOCHS):
            model.train()
            train_loss = 0.0
            for batch_pc, batch_lbl in train_loader:
                batch_aug_pc = []
                batch_aug_lbl = []
                for pc_np, lbl_np in zip(batch_pc.numpy(), batch_lbl.numpy()):
                    pc_aug, lbl_aug = augment(pc_np.T, lbl_np)
                    batch_aug_pc.append(pc_aug.T)
                    batch_aug_lbl.append(lbl_aug)
                
                batch_aug_pc = torch.from_numpy(np.stack(batch_aug_pc)).float().to(device)
                batch_aug_lbl = torch.from_numpy(np.stack(batch_aug_lbl)).float().to(device)
                
                optimizer.zero_grad()
                pred, trans_feat = model(batch_aug_pc)
                loss = criterion(pred, batch_aug_lbl)
                loss += ft_weight * feature_transform_reguliarzer(trans_feat)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            scheduler.step()
            
            model.eval()
            with torch.no_grad():
                val_pred, val_trans_feat = model(X_val_t)
                val_loss = criterion(val_pred, Y_val_t).item()
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
        
        fold_losses.append(best_val_loss)
        print(f"  Fold {fold_idx}: Best Val Loss = {best_val_loss:.8f}")
    
    mean_loss = np.mean(fold_losses)
    std_loss = np.std(fold_losses)
    print(f"  Mean Val Loss: {mean_loss:.8f} ± {std_loss:.8f}")
    
    return mean_loss, std_loss, fold_losses

def main():
    print("="*70)
    print("🔍 K折交叉验证超参数调优")
    print("="*70)
    print(f"Device: {device}")
    if device.type != 'cuda':
        print("⚠️  当前未使用 GPU，调参会很慢，建议切换到 GPU 环境后再运行。")
    print(f"K-Folds: {K_FOLDS}")
    print(f"Epochs per fold: {NUM_EPOCHS}")
    
    print(f"\n超参数搜索空间:")
    for key, values in PARAM_GRID.items():
        print(f"  {key}: {values}")
    
    # 计算总组合数
    total_combinations = 1
    for values in PARAM_GRID.values():
        total_combinations *= len(values)
    print(f"\n总共 {total_combinations} 种组合")
    
    # 加载数据
    print("\n加载数据...")
    X, Y = load_data()
    print(f"样本数: {len(X)}")
    
    # 生成超参数组合（随机采样或全量）
    keys = list(PARAM_GRID.keys())
    values = list(PARAM_GRID.values())
    all_combinations = [dict(zip(keys, v)) for v in product(*values)]

    if USE_RANDOM_SEARCH:
        rng = np.random.default_rng(RANDOM_SEED)
        sample_size = min(MAX_RANDOM_TRIALS, len(all_combinations))
        sampled_idx = rng.choice(len(all_combinations), size=sample_size, replace=False)
        param_combinations = [all_combinations[i] for i in sampled_idx]
        print(f"\n随机采样 {sample_size} / {len(all_combinations)} 个组合")
    else:
        param_combinations = all_combinations
        print(f"\n全量网格，共 {len(param_combinations)} 个组合")
    
    # 记录所有结果
    results = []
    best_mean_loss = float('inf')
    best_params = None
    
    start_time = time.time()
    
    for trial_num, params in enumerate(param_combinations, start=1):
        try:
            mean_loss, std_loss, fold_losses = train_kfold_with_params(X, Y, params, trial_num)
            
            result = {
                'trial': trial_num,
                'params': params,
                'mean_val_loss': float(mean_loss),
                'std_val_loss': float(std_loss),
                'fold_losses': [float(l) for l in fold_losses]
            }
            results.append(result)
            
            if mean_loss < best_mean_loss:
                best_mean_loss = mean_loss
                best_params = params
                print(f"\n✨ 新的最佳结果! Mean Loss = {mean_loss:.8f}")
        
        except Exception as e:
            print(f"❌ Trial {trial_num} 失败: {str(e)}")
            continue
    
    elapsed = time.time() - start_time
    
    print("\n" + "="*70)
    print("🏆 超参数调优完成")
    print("="*70)
    print(f"总耗时: {elapsed/60:.1f} 分钟")
    print(f"\n最佳超参数配置:")
    for key, value in best_params.items():
        print(f"  {key}: {value}")
    print(f"\n最佳平均验证损失: {best_mean_loss:.8f}")
    
    # 保存结果
    output = {
        'search_space': PARAM_GRID,
        'total_trials': len(results),
        'best_params': best_params,
        'best_mean_val_loss': float(best_mean_loss),
        'all_results': results,
        'elapsed_time_minutes': float(elapsed/60)
    }
    
    output_path = os.path.join(ROOT_DIR, "results", "hyperparameter_tuning_results.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"\n📁 详细结果已保存: {output_path}")
    
    # 显示前5个最佳配置
    sorted_results = sorted(results, key=lambda x: x['mean_val_loss'])
    print("\n📊 Top 5 最佳配置:")
    print("-"*70)
    for i, res in enumerate(sorted_results[:5], 1):
        print(f"\n{i}. Mean Loss = {res['mean_val_loss']:.8f}")
        for key, value in res['params'].items():
            print(f"   {key}: {value}")

if __name__ == "__main__":
    main()
