"""
PointNet++ K-Fold Training with Deduplication (Fixes Density Issues)
- Uses Best Hyperparameters found in grid search
- Adds 'clean_point_cloud' step to merge near-duplicate points (voxel/epsilon filtering)
- Then samples to 8192 points using FPS
- 5-Fold Cross Validation
"""
import os
import sys
import json
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
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
from pointnet2_ops.pointnet2_utils import furthest_point_sample

EXPORT_ROOT = os.path.join(ROOT_DIR, 'data', 'pointcloud')
LABELS_FILE = os.path.join(ROOT_DIR, 'results', 'labels.csv')
PROJECTS_LIST_FILE = os.path.join(ROOT_DIR, 'results', 'valid_projects.txt')

NUM_TARGET_POINTS = 9
OUTPUT_DIM = NUM_TARGET_POINTS * 3
MAX_POINTS = 8192
K_FOLDS = 5
RANDOM_SEED = 42

# 🏆 Best Hyperparameters from Grid Search
BATCH_SIZE = 8
NUM_EPOCHS = 180  # Aligned with best result training duration
LEARNING_RATE = 0.001
LR_DECAY_STEP = 80
LR_DECAY_GAMMA = 0.7
DROPOUT_RATE = 0.4
WEIGHT_DECAY = 0.0
GEO_LAMBDA = 0.0
LOSS_TYPE = 'smoothl1'
SA1_RADII = [0.1, 0.2, 0.4]
SA2_RADII = [0.2, 0.4, 0.8]

# 🧹 Deduplication Parameter
# ANALYSIS: Epsilon=85.0 -> 2600 pts (Too small). Epsilon=2.5 -> 17700 pts (Too big).
# Interpolating to target ~7k-10k pts. 
ENABLE_DEDUPLICATION = True
DEDUP_EPSILON = 35.0  

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🖥️  device: {device}")


def clean_point_cloud(points, epsilon=1.0):
    """
    Deduplicates points using voxel grid merging.
    Points within the same epsilon-sized voxel are averaged.
    Uses np.unique + np.add.at for robustness.
    """
    if points.shape[0] == 0:
        return points
    
    # Quantize to grid indices
    quant = np.floor(points / epsilon).astype(np.int64)
    
    # Identify unique voxels and inverse mapping
    _, inv = np.unique(quant, axis=0, return_inverse=True)
    num_unique = _.shape[0]
    
    # Sum points in each voxel
    sums = np.zeros((num_unique, 3), dtype=np.float32)
    np.add.at(sums, inv, points)
    
    # Count points in each voxel
    counts = np.bincount(inv).astype(np.float32)
    
    # Compute means
    means = sums / counts[:, None]
    return means



def farthest_point_sampling(points_np, n_samples):
    """Use PointNet2's GPU-accelerated FPS if available."""
    if device.type == 'cuda':
        points_t = torch.from_numpy(points_np).float().unsqueeze(0).to(device)
        # Check if we have enough points, if not, repeat or just take all?
        # Usually FPS requires N >= n_samples.
        # If N < n_samples, we need to upsample.
        if points_np.shape[0] < n_samples:
            # Simple random upsampling with replacement
            indices = np.random.choice(points_np.shape[0], n_samples, replace=True)
            return points_np[indices]
            
        try:
            idx = furthest_point_sample(points_t, n_samples)  # (1, n_samples)
            return points_np[idx[0].cpu().numpy()]
        except Exception as e:
            print(f"FPS CUDA Warning: {e}. Fallback to CPU/Random.")
    
    # CPU Fallback
    if points_np.shape[0] < n_samples:
        indices = np.random.choice(points_np.shape[0], n_samples, replace=True)
        return points_np[indices]
        
    # Pythonic FPS is slow, use random for fallback or naive implementation
    # For speed, strictly falling back to random choice here if CUDA fails
    indices = np.random.choice(points_np.shape[0], n_samples, replace=False)
    return points_np[indices]


def augment_batch(batch_pc, batch_lbl):
    """Vectorized augmentation on torch tensors."""
    B, _, N = batch_pc.shape
    device_local = batch_pc.device

    theta = (torch.rand(B, 1, 1, device=device_local) * (2 * np.pi / 12) - (np.pi / 12))
    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)
    rot = torch.zeros(B, 3, 3, device=device_local)
    rot[:, 0, 0] = cos_t.flatten()
    rot[:, 0, 1] = -sin_t.flatten()
    rot[:, 1, 0] = sin_t.flatten()
    rot[:, 1, 1] = cos_t.flatten()
    rot[:, 2, 2] = 1.0

    pc_t = batch_pc.transpose(1, 2)
    pc_rot = torch.bmm(pc_t, rot)

    lbl = batch_lbl.view(B, NUM_TARGET_POINTS, 3)
    lbl_rot = torch.bmm(lbl, rot)

    scale = torch.rand(B, 1, 1, device=device_local) * 0.1 + 0.95
    pc_rot = pc_rot * scale
    lbl_rot = lbl_rot * scale

    shift = (torch.rand(B, 1, 3, device=device_local) * 0.04) - 0.02
    pc_rot = pc_rot + shift
    lbl_rot = lbl_rot + shift

    jitter = torch.randn(B, N, 3, device=device_local) * 0.005
    pc_rot = pc_rot + jitter

    pc_out = pc_rot.transpose(1, 2)
    lbl_out = lbl_rot.view(B, -1)
    return pc_out, lbl_out


def compute_loss(pred, target):
    B = pred.shape[0]
    pred_pts = pred.view(B, NUM_TARGET_POINTS, 3)
    tgt_pts = target.view(B, NUM_TARGET_POINTS, 3)
    
    if LOSS_TYPE == 'smoothl1':
        loss = F.smooth_l1_loss(pred_pts, tgt_pts)
    else:
        loss = F.mse_loss(pred_pts, tgt_pts)
        
    return loss


def load_data():
    print(f"⏳ Loading data...")
    with open(PROJECTS_LIST_FILE, 'r', encoding='utf-8') as f:
        project_names = [ln.strip() for ln in f if ln.strip()]
    labels_df = pd.read_csv(LABELS_FILE, header=None)
    labels_np = labels_df.values.astype(np.float32)

    feats, labels = [], []
    cleanup_stats = []
    ratios = []
    bbox_diagonals = []
    norm_factors = []

    for i, name in enumerate(tqdm(project_names, desc="Processing")):
        pc_path = os.path.join(EXPORT_ROOT, name, "pointcloud_full.npy")
        if not os.path.exists(pc_path):
            continue
        pc = np.load(pc_path).astype(np.float32)
        if pc.shape[0] == 0:
            continue

        # Check scale (bounding box diagonal) to verify units
        bb_min = np.min(pc, axis=0)
        bb_max = np.max(pc, axis=0)
        diagonal = np.linalg.norm(bb_max - bb_min)
        bbox_diagonals.append(diagonal)
            
        # --- 1. Deduplication (New Step) ---
        original_count = pc.shape[0]
        if ENABLE_DEDUPLICATION:
            pc_clean = clean_point_cloud(pc, epsilon=DEDUP_EPSILON)
        else:
            pc_clean = pc
            
        clean_count = pc_clean.shape[0]
        
        ratio = clean_count / original_count
        cleanup_stats.append(clean_count)
        ratios.append(ratio)
        
        # --- 2. Normalization (Point Cloud Centroid - Inference Compatible) ---
        # Center on point cloud itself (no ground truth needed)
        label = labels_np[i].reshape(NUM_TARGET_POINTS, 3)
        
        pc_centroid = np.mean(pc_clean, axis=0)
        pc_centered = pc_clean - pc_centroid
        label_centered = label - pc_centroid
        
        # Scale by bounding sphere radius (max distance from centroid)
        max_dist = np.max(np.linalg.norm(pc_centered, axis=1))
        if max_dist < 1e-6:
            max_dist = 1.0  # Avoid division by zero
        
        pc_centered /= max_dist
        label_centered /= max_dist
        
        norm_factors.append(max_dist)

        # --- 3. Sampling ---
        pc_sampled = farthest_point_sampling(pc_centered, MAX_POINTS)
        feats.append(pc_sampled.T)
        labels.append(label_centered.flatten())

    print(f"\n===== 📊 Data Analysis (Deduplication={'ON' if ENABLE_DEDUPLICATION else 'OFF'}) =====")
    if ENABLE_DEDUPLICATION:
        print(f"   Epsilon: {DEDUP_EPSILON}")
    
    print(f"1. Unit Scale Check (Bounding Box Diagonal):")
    print(f"   Mean: {np.mean(bbox_diagonals):.1f}, Min: {np.min(bbox_diagonals):.1f}, Max: {np.max(bbox_diagonals):.1f}")
    if np.mean(bbox_diagonals) < 10.0:
        print("   ⚠️ WARNING: Scale is very small (<10). Data might be in METERS, not mm!")
        print("   If so, DEDUP_EPSILON=1.5 is huge. Set epsilon approx 0.001-0.002.")
    elif np.mean(bbox_diagonals) > 1000.0:
        print("   ⚠️ WARNING: Scale is very large (>1000). Check units.")
    else:
        print("   ✅ Scale looks like mm (typical face: 150-250mm).")

    print(f"2. Cleanup Stats:")
    print(f"   Mean Points: {np.mean(cleanup_stats):.1f} (Original ~{original_count if 'original_count' in locals() else 'Unknown'})")
    print(f"   Range: [{np.min(cleanup_stats)}, {np.max(cleanup_stats)}]")
    print(f"   Retention Ratio (Clean/Original):")
    print(f"   Mean: {np.mean(ratios):.1%}")
    print(f"   Min: {np.min(ratios):.1%} - Max: {np.max(ratios):.1%}")
    
    if ENABLE_DEDUPLICATION:
        if np.mean(ratios) > 0.95:
            print("   ⚠️ WARNING: Less than 5% points removed. Epsilon might be too SMALL (under-smoothing).")
        elif np.mean(ratios) < 0.2:
            print("   ⚠️ WARNING: More than 80% points removed. Epsilon might be too LARGE (over-smoothing).")

    X = np.stack(feats, axis=0)
    Y = np.stack(labels, axis=0)
    Scales = np.array(norm_factors, dtype=np.float32)
    return X, Y, Scales


def train_kfold():
    X, Y, Scales = load_data()
    print(f"样本数 Total Samples: {len(X)}")

    # 10/90 split
    n_samples = len(X)
    n_test = max(1, int(n_samples * 0.1))
    
    test_indices = np.random.RandomState(RANDOM_SEED).choice(n_samples, size=n_test, replace=False)
    train_indices = np.setdiff1d(np.arange(n_samples), test_indices)
    
    X_train_full = X[train_indices]
    Y_train_full = Y[train_indices]
    Scales_train_full = Scales[train_indices]

    X_test = X[test_indices]
    Y_test = Y[test_indices]
    Scales_test = Scales[test_indices]
    
    print(f"Train/Val Full: {len(X_train_full)}, Test Held-out: {len(X_test)}")

    kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    fold_results = []
    
    best_overall_loss = float('inf')
    best_fold_idx = -1

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_train_full), start=1):
        print(f"\n===== Fold {fold_idx}/{K_FOLDS} =====")
        X_train, X_val = X_train_full[train_idx], X_train_full[val_idx]
        Y_train, Y_val = Y_train_full[train_idx], Y_train_full[val_idx]
        Scales_train, Scales_val = Scales_train_full[train_idx], Scales_train_full[val_idx]

        # Use Best Hyperparameters in Model
        model = PointNet2RegMSG(
            output_dim=OUTPUT_DIM, 
            normal_channel=False, 
            dropout=DROPOUT_RATE,
            sa1_radii=SA1_RADII,
            sa2_radii=SA2_RADII
        ).to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=LR_DECAY_STEP, gamma=LR_DECAY_GAMMA)

        X_train_t = torch.from_numpy(X_train).float()
        Y_train_t = torch.from_numpy(Y_train).float()
        X_val_t = torch.from_numpy(X_val).float().to(device)
        Y_val_t = torch.from_numpy(Y_val).float().to(device)

        train_dataset = TensorDataset(X_train_t, Y_train_t)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

        best_val_loss = float('inf')
        best_val_l2 = float('inf')  # 🏆 Metric for selection (in mm)
        best_state = None

        for epoch in range(1, NUM_EPOCHS + 1):
            model.train()
            train_loss_accum = 0.0
            
            for batch_pc, batch_lbl in train_loader:
                batch_aug_pc, batch_aug_lbl = augment_batch(batch_pc, batch_lbl)
                batch_aug_pc = batch_aug_pc.to(device)
                batch_aug_lbl = batch_aug_lbl.to(device)

                optimizer.zero_grad()
                pred = model(batch_aug_pc)
                loss = compute_loss(pred, batch_aug_lbl)
                loss.backward()
                optimizer.step()

                train_loss_accum += loss.item()

            scheduler.step()
            avg_train_loss = train_loss_accum / len(train_loader)

            model.eval()
            with torch.no_grad():
                val_pred = model(X_val_t)
                val_loss = compute_loss(val_pred, Y_val_t).item()
                
                # Compute L2 mean for selection
                val_pred_np = val_pred.cpu().numpy().reshape(-1, NUM_TARGET_POINTS, 3)
                Y_val_np = Y_val_t.cpu().numpy().reshape(-1, NUM_TARGET_POINTS, 3)
                
                # Calculate normalized distance
                l2_dists_norm = np.linalg.norm(val_pred_np - Y_val_np, axis=2)
                
                # Denormalize to mm => distance * scale
                l2_dists_mm = l2_dists_norm * Scales_val[:, np.newaxis]
                
                val_l2_mean_mm = np.mean(l2_dists_mm)

            # Select based on L2 Mean (Validation Metric in mm)
            if val_l2_mean_mm < best_val_l2:
                best_val_l2 = val_l2_mean_mm
                best_val_loss = val_loss
                best_state = copy.deepcopy(model.state_dict())

            if epoch % 20 == 0 or epoch == NUM_EPOCHS:
                print(f"Epoch {epoch} - Train {avg_train_loss:.6f}  Val Loss {val_loss:.6f}  Val L2(mm) {val_l2_mean_mm:.4f}  Best L2(mm) {best_val_l2:.4f}")

        # Save Best Fold Model
        fold_model_path = os.path.join(ROOT_DIR, "models", f"pointnet2_dedup_fold{fold_idx}_best.pth")
        torch.save(best_state, fold_model_path)
        print(f"Fold {fold_idx} Best Val L2 (mm): {best_val_l2:.4f}")

        fold_results.append(best_val_l2)
        if best_val_l2 < best_overall_loss:
            best_overall_loss = best_val_l2  # Now tracks best L2
            best_fold_idx = fold_idx

    print(f"\n🎉 Best Fold: {best_fold_idx} (Val L2 mm: {best_overall_loss:.4f})")
    
    # --- Final Training on ALL 90% Training Data ---
    print(f"\n🚀 Retraining Final Model on Full 90% Dataset (Train+Val)...")
    
    final_model = PointNet2RegMSG(
        output_dim=OUTPUT_DIM, 
        normal_channel=False, 
        dropout=DROPOUT_RATE,
        sa1_radii=SA1_RADII,
        sa2_radii=SA2_RADII
    ).to(device)
    
    optimizer = torch.optim.Adam(final_model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=LR_DECAY_STEP, gamma=LR_DECAY_GAMMA)
    
    # Use Full Training Set (X_train_full)
    X_train_full_t = torch.from_numpy(X_train_full).float()
    Y_train_full_t = torch.from_numpy(Y_train_full).float()
    train_full_ds = TensorDataset(X_train_full_t, Y_train_full_t)
    train_full_loader = DataLoader(train_full_ds, batch_size=BATCH_SIZE, shuffle=True)
    
    for epoch in range(1, NUM_EPOCHS + 1):
        final_model.train()
        train_loss_accum = 0.0
        
        for batch_pc, batch_lbl in train_full_loader:
            batch_aug_pc, batch_aug_lbl = augment_batch(batch_pc, batch_lbl)
            batch_aug_pc = batch_aug_pc.to(device)
            batch_aug_lbl = batch_aug_lbl.to(device)

            optimizer.zero_grad()
            pred = final_model(batch_aug_pc)
            loss = compute_loss(pred, batch_aug_lbl)
            loss.backward()
            optimizer.step()
            train_loss_accum += loss.item()
            
        scheduler.step()
        
        if epoch % 20 == 0 or epoch == NUM_EPOCHS:
             print(f"Final Train Epoch {epoch}/{NUM_EPOCHS} - Loss: {train_loss_accum/len(train_full_loader):.6f}")

    final_model_path = os.path.join(ROOT_DIR, "models", "pointnet2_dedup_final_best.pth")
    torch.save(final_model.state_dict(), final_model_path)
    print(f"💾 Final Model Saved: {final_model_path}")

    # --- Evaluate Final Model on Held-out Test Set ---
    final_model.eval()
    X_test_t = torch.from_numpy(X_test).float().to(device)
    Y_test_t = torch.from_numpy(Y_test).float().to(device)
    
    with torch.no_grad():
        test_pred = final_model(X_test_t)
        test_loss = compute_loss(test_pred, Y_test_t).item()
    
    test_pred_np = test_pred.cpu().numpy().reshape(-1, NUM_TARGET_POINTS, 3)
    Y_test_np = Y_test_t.cpu().numpy().reshape(-1, NUM_TARGET_POINTS, 3)
    
    # Denormalize Test Results
    l2_dists_norm = np.linalg.norm(test_pred_np - Y_test_np, axis=2)
    l2_dists_mm = l2_dists_norm * Scales_test[:, np.newaxis]
    
    l2_per_landmark = np.mean(l2_dists_mm, axis=0) # Mean across samples
    mean_l2_mm = np.mean(l2_dists_mm)
    std_l2_mm = np.std(l2_dists_mm)

    print(f"\n===== Test Set Evaluation (on Final Model) =====")
    print(f"Test Loss ({LOSS_TYPE}): {test_loss:.6f}")
    print(f"L2 Mean Error: {mean_l2_mm:.4f} mm ± {std_l2_mm:.4f} mm")
    print("Per-Landmark L2 (mm):")
    for i, err in enumerate(l2_per_landmark):
        print(f"  P{i}: {err:.4f}")
    print(f"Result Saved to: {final_model_path}")



if __name__ == "__main__":
    train_kfold()
