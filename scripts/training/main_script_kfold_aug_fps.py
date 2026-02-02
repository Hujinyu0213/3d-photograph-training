"""
PointNet 回归训练（K折 + 数据增强 + FPS）
- FPS 统一采样到 8192 点
- 在线增强（旋转/缩放/平移/抖动）仅训练阶段
- 5 折交叉验证，保存每折最佳模型 + 汇总
- 路径：
  点云 data/pointcloud/<project>/pointcloud_full.npy
  标签 results/labels.csv
  项目列表 results/valid_projects.txt
- 输出：
  models/pointnet_regression_model_kfold_aug_fps_fold{i}_best.pth
  models/pointnet_regression_model_kfold_aug_fps_best.pth
  results/training_history_kfold_aug_fps.json
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
feature_transform_reguliarzer = pointnet_utils.feature_transform_reguliarzer

# 配置
EXPORT_ROOT = os.path.join(ROOT_DIR, 'data', 'pointcloud')
LABELS_FILE = os.path.join(ROOT_DIR, 'results', 'labels.csv')
PROJECTS_LIST_FILE = os.path.join(ROOT_DIR, 'results', 'valid_projects.txt')

NUM_TARGET_POINTS = 9
OUTPUT_DIM = NUM_TARGET_POINTS * 3
MAX_POINTS = 8192

BATCH_SIZE = 8
NUM_EPOCHS = 250
LEARNING_RATE = 0.001
LR_DECAY_STEP = 120
LR_DECAY_GAMMA = 0.5
DROPOUT_RATE = 0.3
FEATURE_TRANSFORM_WEIGHT = 0.001
K_FOLDS = 5
RANDOM_SEED = 42

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🖥️  device: {device}")

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

# 数据增强（仅训练批次）
def augment(points_np):
    pts = points_np.copy()
    theta = np.random.uniform(-np.pi/12, np.pi/12)  # ±15°
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s, 0],[s, c, 0],[0,0,1]], dtype=np.float32)
    pts = R @ pts
    scale = np.random.uniform(0.95, 1.05)
    pts *= scale
    trans = np.random.uniform(-0.02, 0.02, size=(3,1)).astype(np.float32)
    pts += trans
    noise = np.random.normal(0, 0.005, size=pts.shape).astype(np.float32)
    pts += noise
    return pts

# 加载与预处理
def load_data():
    with open(PROJECTS_LIST_FILE, 'r', encoding='utf-8') as f:
        project_names = [ln.strip() for ln in f if ln.strip()]
    labels_df = pd.read_csv(LABELS_FILE, header=None)
    labels_np = labels_df.values.astype(np.float32)
    if len(project_names) != len(labels_np):
        m = min(len(project_names), len(labels_np))
        project_names = project_names[:m]
        labels_np = labels_np[:m]

    feats, labels, counts = [], [], []
    for i, name in enumerate(tqdm(project_names, desc="load pointcloud")):
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
        counts.append(pc.shape[0])

    X = torch.from_numpy(np.stack(feats, axis=0)).float()
    Y = torch.from_numpy(np.stack(labels, axis=0)).float()
    print(f"loaded {len(X)} samples; raw points min/max/avg: {min(counts)}/{max(counts)}/{np.mean(counts):.0f}")
    print(f"tensor shapes: X={X.shape}, Y={Y.shape}")
    return X, Y

# 单折训练
def train_fold(X, Y, train_idx, val_idx, fold_num):
    train_ds = TensorDataset(X[train_idx], Y[train_idx])
    val_ds = TensorDataset(X[val_idx], Y[val_idx])
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

    model = PointNetRegressor(output_dim=OUTPUT_DIM, dropout_rate=DROPOUT_RATE).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=LR_DECAY_STEP, gamma=LR_DECAY_GAMMA)

    best_val = float('inf')
    history = {'epoch': [], 'train_loss': [], 'val_loss': []}

    for epoch in range(NUM_EPOCHS):
        model.train()
        total = 0
        n = 0
        for data, target in train_loader:
            data_aug = []
            for d in data.numpy():
                data_aug.append(augment(d).astype(np.float32))
            data_aug = torch.from_numpy(np.stack(data_aug, axis=0)).to(device)
            target = target.to(device)
            optimizer.zero_grad()
            pred, trans_feat = model(data_aug)
            loss = criterion(pred, target)
            if trans_feat is not None:
                loss += feature_transform_reguliarzer(trans_feat) * FEATURE_TRANSFORM_WEIGHT
            loss.backward()
            optimizer.step()
            total += loss.item()
            n += 1
        train_loss = total / max(1, n)

        model.eval()
        val_total = 0
        val_n = 0
        with torch.no_grad():
            for data, target in val_loader:
                data = data.to(device)
                target = target.to(device)
                pred, trans_feat = model(data)
                loss = criterion(pred, target)
                if trans_feat is not None:
                    loss += feature_transform_reguliarzer(trans_feat) * FEATURE_TRANSFORM_WEIGHT
                val_total += loss.item()
                val_n += 1
        val_loss = val_total / max(1, val_n)

        scheduler.step()

        history['epoch'].append(epoch + 1)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), os.path.join(ROOT_DIR, "models", f"pointnet_regression_model_kfold_aug_fps_fold{fold_num}_best.pth"))

        if (epoch + 1) % 25 == 0 or epoch == 0:
            print(f"Fold {fold_num} | Epoch {epoch+1:3d}/{NUM_EPOCHS} | train {train_loss:.6f} | val {val_loss:.6f} | best {best_val:.6f}")

    torch.save(model.state_dict(), os.path.join(ROOT_DIR, "models", f"pointnet_regression_model_kfold_aug_fps_fold{fold_num}_final.pth"))
    return best_val, history


def main():
    X, Y = load_data()
    kfold = KFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_SEED)

    fold_results = []
    histories = []

    for fold_num, (train_idx, val_idx) in enumerate(kfold.split(X), 1):
        best_val, history = train_fold(X, Y, train_idx, val_idx, fold_num)
        fold_results.append({
            'fold': fold_num,
            'best_val_loss': float(best_val),
            'train_size': len(train_idx),
            'val_size': len(val_idx)
        })
        histories.append(history)

    best_losses = [f['best_val_loss'] for f in fold_results]
    best_fold = int(np.argmin(best_losses) + 1)
    mean_loss = float(np.mean(best_losses))
    std_loss = float(np.std(best_losses))

    import shutil
    src = os.path.join(ROOT_DIR, "models", f"pointnet_regression_model_kfold_aug_fps_fold{best_fold}_best.pth")
    dst = os.path.join(ROOT_DIR, "models", "pointnet_regression_model_kfold_aug_fps_best.pth")
    shutil.copy(src, dst)

    summary = {
        'k_folds': K_FOLDS,
        'fold_results': fold_results,
        'statistics': {
            'mean_best_val_loss': mean_loss,
            'std_best_val_loss': std_loss,
            'best_fold': best_fold,
            'best_fold_loss': float(np.min(best_losses))
        },
        'training_histories': histories
    }

    out_path = os.path.join(ROOT_DIR, "results", "training_history_kfold_aug_fps.json")
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n" + "="*60)
    print("📦 K折训练完成")
    print("平均最佳验证损失:", mean_loss)
    print("最佳折:", best_fold, "loss:", np.min(best_losses))
    print("最佳模型:", dst)
    print("训练历史:", out_path)
    print("="*60)


if __name__ == "__main__":
    main()
