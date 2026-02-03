"""
PointNet++ 回归训练（K折 + 数据增强 + FPS）
- 使用 PointNet2RegMSG (MSG) 输出 27 维
- FPS 采样 8192 点；仅训练阶段做旋转/缩放/平移/抖动
- 5 折交叉验证，保存每折最佳 + 整体最佳
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
MODELS_DIR = os.path.join(ROOT_DIR, "models")
for p in (ROOT_DIR, UTILS_DIR, MODELS_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from pointnet2_reg import PointNet2RegMSG

EXPORT_ROOT = os.path.join(ROOT_DIR, 'data', 'pointcloud')
LABELS_FILE = os.path.join(ROOT_DIR, 'results', 'labels.csv')
PROJECTS_LIST_FILE = os.path.join(ROOT_DIR, 'results', 'valid_projects.txt')

NUM_TARGET_POINTS = 9
OUTPUT_DIM = NUM_TARGET_POINTS * 3
MAX_POINTS = 8192
K_FOLDS = 5
RANDOM_SEED = 42

# 超参（温和版）
BATCH_SIZE = 8
NUM_EPOCHS = 220
LEARNING_RATE = 0.001
LR_DECAY_STEP = 80
LR_DECAY_GAMMA = 0.7
DROPOUT_RATE = 0.35

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🖥️  device: {device}")


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
        feats.append(pc_sampled.T)          # (3, N)
        labels.append(label_centered.flatten())

    X = np.stack(feats, axis=0)
    Y = np.stack(labels, axis=0)
    return X, Y


def train_kfold():
    X, Y = load_data()
    print(f"样本数: {len(X)}")

    # 10/90 split: 10% test, 90% for k-fold training
    n_samples = len(X)
    n_test = max(1, int(n_samples * 0.1))
    n_train = n_samples - n_test
    
    test_indices = np.random.RandomState(RANDOM_SEED).choice(n_samples, size=n_test, replace=False)
    train_indices = np.setdiff1d(np.arange(n_samples), test_indices)
    
    X_train_full = X[train_indices]
    Y_train_full = Y[train_indices]
    X_test = X[test_indices]
    Y_test = Y[test_indices]
    
    print(f"训练集: {len(X_train_full)}, 测试集: {len(X_test)}")

    kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    fold_results = []
    all_histories = []

    best_overall_loss = float('inf')
    best_fold_idx = -1

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_train_full), start=1):
        print(f"\n===== Fold {fold_idx}/{K_FOLDS} =====")
        X_train, X_val = X_train_full[train_idx], X_train_full[val_idx]
        Y_train, Y_val = Y_train_full[train_idx], Y_train_full[val_idx]

        model = PointNet2RegMSG(output_dim=OUTPUT_DIM, normal_channel=False, dropout=DROPOUT_RATE).to(device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=LR_DECAY_STEP, gamma=LR_DECAY_GAMMA)

        X_train_t = torch.from_numpy(X_train).float()
        Y_train_t = torch.from_numpy(Y_train).float()
        X_val_t = torch.from_numpy(X_val).float().to(device)
        Y_val_t = torch.from_numpy(Y_val).float().to(device)

        train_dataset = TensorDataset(X_train_t, Y_train_t)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

        best_val_loss = float('inf')
        best_state = None
        history = {'epoch': [], 'train_loss': [], 'val_loss': [], 'lr': []}

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
                pred = model(batch_aug_pc)
                loss = criterion(pred, batch_aug_lbl)
                loss.backward()
                optimizer.step()

                train_loss_accum += loss.item()
                batch_count += 1

            scheduler.step()
            avg_train_loss = train_loss_accum / batch_count

            model.eval()
            with torch.no_grad():
                val_pred = model(X_val_t)
                val_loss = criterion(val_pred, Y_val_t).item()

            history['epoch'].append(epoch)
            history['train_loss'].append(float(avg_train_loss))
            history['val_loss'].append(float(val_loss))
            history['lr'].append(float(optimizer.param_groups[0]['lr']))

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = model.state_dict().copy()

            if epoch % 20 == 0 or epoch == NUM_EPOCHS:
                print(f"Epoch {epoch}/{NUM_EPOCHS} - Train {avg_train_loss:.6f}  Val {val_loss:.6f}  Best {best_val_loss:.6f}")

        # 保存本折最佳
        fold_model_path = os.path.join(ROOT_DIR, "models", f"pointnet2_regression_kfold_fold{fold_idx}_best.pth")
        torch.save(best_state, fold_model_path)
        print(f"Fold {fold_idx} Best Val Loss: {best_val_loss:.6f}  -> {fold_model_path}")

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

    # 汇总与复制最佳模型
    best_fold_model = os.path.join(ROOT_DIR, "models", f"pointnet2_regression_kfold_fold{best_fold_idx}_best.pth")
    best_overall_model = os.path.join(ROOT_DIR, "models", "pointnet2_regression_kfold_best.pth")
    import shutil
    shutil.copy(best_fold_model, best_overall_model)

    # 在测试集上评估最佳模型
    print(f"\n===== 在测试集上评估 =====")
    best_model = PointNet2RegMSG(output_dim=OUTPUT_DIM, normal_channel=False, dropout=DROPOUT_RATE).to(device)
    best_model.load_state_dict(torch.load(best_overall_model))
    best_model.eval()
    
    X_test_t = torch.from_numpy(X_test).float().to(device)
    Y_test_t = torch.from_numpy(Y_test).float().to(device)
    criterion = nn.MSELoss()
    
    with torch.no_grad():
        test_pred = best_model(X_test_t)
        test_loss = criterion(test_pred, Y_test_t).item()
    
    # 计算L2距离
    test_pred_np = test_pred.cpu().numpy()
    Y_test_np = Y_test_t.cpu().numpy()
    
    # 重塑为 (num_samples, 9, 3) - 9个地标，每个3维
    test_pred_reshaped = test_pred_np.reshape(-1, NUM_TARGET_POINTS, 3)
    Y_test_reshaped = Y_test_np.reshape(-1, NUM_TARGET_POINTS, 3)
    
    # 计算每个样本的每个地标的L2距离
    l2_distances = np.linalg.norm(test_pred_reshaped - Y_test_reshaped, axis=2)  # (num_samples, 9)
    
    l2_mean = np.mean(l2_distances)
    l2_std = np.std(l2_distances)
    l2_per_landmark = np.mean(l2_distances, axis=0)  # 每个地标的平均L2距离
    
    print(f"测试集损失: {test_loss:.6f}")
    print(f"L2距离 (平均): {l2_mean:.6f} ± {l2_std:.6f}")
    print(f"各地标L2距离: {l2_per_landmark}")

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
            'num_epochs': NUM_EPOCHS
        },
        'k_folds': K_FOLDS,
        'fold_results': fold_results,
        'statistics': stats,
        'training_histories': all_histories,
        'test_loss': float(test_loss),
        'test_set_size': int(len(X_test)),
        'test_l2_distance': {
            'mean': float(l2_mean),
            'std': float(l2_std),
            'per_landmark': l2_per_landmark.tolist()
        }
    }

    history_path = os.path.join(ROOT_DIR, "results", "training_histories", "training_history_pointnet2_kfold.json")
    with open(history_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print("\n==== 训练完成 ==== ")
    print(f"最佳折: Fold {best_fold_idx}, 最佳验证损失: {best_overall_loss:.6f}")
    print(f"平均验证损失: {stats['mean_best_val_loss']:.6f} ± {stats['std_best_val_loss']:.6f}")
    print(f"测试集损失: {test_loss:.6f}")
    print(f"最佳模型: {best_overall_model}")
    print(f"训练历史: {history_path}")


if __name__ == "__main__":
    train_kfold()
