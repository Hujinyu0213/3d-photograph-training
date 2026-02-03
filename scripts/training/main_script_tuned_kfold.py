"""
使用超参数调优找到的最佳配置重新训练完整模型
基于最佳超参数: LR=0.0015, BS=8, Dropout=0.35, DecayStep=120, Gamma=0.7, FT=0.001
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

# 🎯 最佳超参数（来自调优结果）
LEARNING_RATE = 0.0015
BATCH_SIZE = 8
DROPOUT_RATE = 0.35
LR_DECAY_STEP = 120
LR_DECAY_GAMMA = 0.7
FEATURE_TRANSFORM_WEIGHT = 0.001
NUM_EPOCHS = 300  # 增加至300轮以充分训练

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🖥️  Device: {device}")

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

def main():
    print("="*80)
    print("🚀 使用最佳超参数训练 K 折交叉验证模型")
    print("="*80)
    print(f"Learning Rate: {LEARNING_RATE}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Dropout: {DROPOUT_RATE}")
    print(f"LR Decay Step: {LR_DECAY_STEP}")
    print(f"LR Decay Gamma: {LR_DECAY_GAMMA}")
    print(f"Feature Transform Weight: {FEATURE_TRANSFORM_WEIGHT}")
    print(f"Epochs per fold: {NUM_EPOCHS}")
    print("="*80)
    
    X, Y = load_data()
    print(f"\n📦 加载数据: {len(X)} 样本")
    
    kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    fold_results = []
    all_histories = []
    
    best_overall_loss = float('inf')
    best_fold_idx = -1
    
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X), start=1):
        print(f"\n{'='*80}")
        print(f"📊 Fold {fold_idx}/{K_FOLDS}")
        print(f"{'='*80}")
        
        X_train, X_val = X[train_idx], X[val_idx]
        Y_train, Y_val = Y[train_idx], Y[val_idx]
        
        print(f"训练集: {len(X_train)} 样本, 验证集: {len(X_val)} 样本")
        
        model = PointNetRegressor(output_dim=OUTPUT_DIM, dropout_rate=DROPOUT_RATE).to(device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=LR_DECAY_STEP, gamma=LR_DECAY_GAMMA)
        
        X_train_t = torch.from_numpy(X_train).float()
        Y_train_t = torch.from_numpy(Y_train).float()
        X_val_t = torch.from_numpy(X_val).float().to(device)
        Y_val_t = torch.from_numpy(Y_val).float().to(device)
        
        train_dataset = TensorDataset(X_train_t, Y_train_t)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        
        history = {'epoch': [], 'train_loss': [], 'val_loss': [], 'lr': []}
        best_val_loss = float('inf')
        best_model_state = None
        
        for epoch in range(1, NUM_EPOCHS + 1):
            model.train()
            train_loss_accum = 0.0
            batch_count = 0
            
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
                loss += FEATURE_TRANSFORM_WEIGHT * feature_transform_reguliarzer(trans_feat)
                loss.backward()
                optimizer.step()
                
                train_loss_accum += loss.item()
                batch_count += 1
            
            avg_train_loss = train_loss_accum / batch_count
            
            model.eval()
            with torch.no_grad():
                val_pred, val_trans_feat = model(X_val_t)
                val_loss = criterion(val_pred, Y_val_t).item()
            
            current_lr = optimizer.param_groups[0]['lr']
            scheduler.step()
            
            history['epoch'].append(epoch)
            history['train_loss'].append(float(avg_train_loss))
            history['val_loss'].append(float(val_loss))
            history['lr'].append(float(current_lr))
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict().copy()
            
            if epoch % 20 == 0 or epoch == NUM_EPOCHS:
                print(f"  Epoch {epoch}/{NUM_EPOCHS} - Train: {avg_train_loss:.8f}, Val: {val_loss:.8f}, LR: {current_lr:.6f}, Best: {best_val_loss:.8f}")
        
        # 保存当前折的最佳模型
        fold_model_path = os.path.join(ROOT_DIR, "models", f"pointnet_regression_model_tuned_kfold_fold{fold_idx}_best.pth")
        torch.save(best_model_state, fold_model_path)
        print(f"\n💾 Fold {fold_idx} 最佳模型已保存: {fold_model_path}")
        print(f"   最佳验证损失: {best_val_loss:.8f}")
        
        fold_results.append({
            'fold': fold_idx,
            'best_val_loss': float(best_val_loss),
            'train_size': len(X_train),
            'val_size': len(X_val)
        })
        all_histories.append(history)
        
        if best_val_loss < best_overall_loss:
            best_overall_loss = best_val_loss
            best_fold_idx = fold_idx
    
    # 保存整体最佳模型
    best_fold_model = os.path.join(ROOT_DIR, "models", f"pointnet_regression_model_tuned_kfold_fold{best_fold_idx}_best.pth")
    best_overall_model = os.path.join(ROOT_DIR, "models", "pointnet_regression_model_tuned_kfold_best.pth")
    
    import shutil
    shutil.copy(best_fold_model, best_overall_model)
    
    # 统计信息
    val_losses = [f['best_val_loss'] for f in fold_results]
    stats = {
        'mean_best_val_loss': float(np.mean(val_losses)),
        'std_best_val_loss': float(np.std(val_losses)),
        'best_fold': int(best_fold_idx),
        'best_fold_loss': float(best_overall_loss)
    }
    
    output = {
        'hyperparameters': {
            'learning_rate': LEARNING_RATE,
            'batch_size': BATCH_SIZE,
            'dropout_rate': DROPOUT_RATE,
            'lr_decay_step': LR_DECAY_STEP,
            'lr_decay_gamma': LR_DECAY_GAMMA,
            'feature_transform_weight': FEATURE_TRANSFORM_WEIGHT,
            'num_epochs': NUM_EPOCHS
        },
        'k_folds': K_FOLDS,
        'fold_results': fold_results,
        'statistics': stats,
        'training_histories': all_histories
    }
    
    history_path = os.path.join(ROOT_DIR, "results", "training_histories", "training_history_tuned_kfold.json")
    with open(history_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print("\n" + "="*80)
    print("🏆 训练完成！")
    print("="*80)
    print(f"\n✨ 最佳折: Fold {best_fold_idx}")
    print(f"   最佳验证损失: {best_overall_loss:.8f}")
    print(f"\n📊 K 折统计:")
    print(f"   平均验证损失: {stats['mean_best_val_loss']:.8f} ± {stats['std_best_val_loss']:.8f}")
    print(f"\n💾 最佳模型已保存: {best_overall_model}")
    print(f"📁 训练历史已保存: {history_path}")
    print("="*80)

if __name__ == "__main__":
    main()
