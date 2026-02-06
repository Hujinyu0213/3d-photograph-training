"""
Coarse-to-Fine PointNet++ Training
Stage 1: Global PointNet++ predicts coarse landmarks on full face point cloud.
Stage 2: Local PointNet++ refines each landmark by regressing residuals on cropped patches.
- Mirrors the existing k-fold routine, adds patch-based residual refinement.
- Uses GT-centered crops with random jitter during training to make the refiner robust to coarse errors.
"""
import os
import sys
import copy
import json
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, Dataset
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

EXPORT_ROOT = os.path.join(ROOT_DIR, "data", "pointcloud")
LABELS_FILE = os.path.join(ROOT_DIR, "results", "labels.csv")
PROJECTS_LIST_FILE = os.path.join(ROOT_DIR, "results", "valid_projects.txt")

NUM_TARGET_POINTS = 9
OUTPUT_DIM = NUM_TARGET_POINTS * 3
MAX_POINTS = 8192
PATCH_POINTS = 512
K_FOLDS = 5
RANDOM_SEED = 42

# Stage 1 (global)
BATCH_SIZE_STAGE1 = 8
NUM_EPOCHS_STAGE1 = 200
LR_STAGE1 = 0.001
LR_DECAY_STEP_STAGE1 = 80
LR_DECAY_GAMMA_STAGE1 = 0.7
DROPOUT_STAGE1 = 0.4
SA1_RADII_STAGE1 = [0.1, 0.2, 0.4]
SA2_RADII_STAGE1 = [0.2, 0.4, 0.8]

# Stage 2 (local refiner)
BATCH_SIZE_STAGE2 = 32  # per-patch batch; each sample contributes 9 patches
NUM_EPOCHS_STAGE2 = 140
LR_STAGE2 = 0.001
LR_DECAY_STEP_STAGE2 = 60
LR_DECAY_GAMMA_STAGE2 = 0.7
DROPOUT_STAGE2 = 0.3
SA1_RADII_STAGE2 = [0.05, 0.1, 0.2]
SA2_RADII_STAGE2 = [0.1, 0.2, 0.4]
PATCH_RADIUS = 0.25  # normalized crop radius
CENTER_JITTER = 0.05  # perturb coarse center during Stage 2 training

LOSS_TYPE = "smoothl1"
WEIGHT_DECAY = 0.0
ENABLE_DEDUPLICATION = False
DEDUP_EPSILON = 35.0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")


# ---------- Utils ----------
def clean_point_cloud(points, epsilon=1.0):
    if points.shape[0] == 0:
        return points
    quant = np.floor(points / epsilon).astype(np.int64)
    uniq, inv = np.unique(quant, axis=0, return_inverse=True)
    sums = np.zeros((uniq.shape[0], 3), dtype=np.float32)
    np.add.at(sums, inv, points)
    counts = np.bincount(inv).astype(np.float32)
    means = sums / counts[:, None]
    return means


def farthest_point_sampling(points_np, n_samples):
    if device.type == "cuda":
        pts = torch.from_numpy(points_np).float().unsqueeze(0).to(device)
        if points_np.shape[0] < n_samples:
            idx = np.random.choice(points_np.shape[0], n_samples, replace=True)
            return points_np[idx]
        try:
            idx = furthest_point_sample(pts, n_samples)
            return points_np[idx[0].cpu().numpy()]
        except Exception as e:
            print(f"FPS CUDA Warning: {e}. Fallback to random.")
    if points_np.shape[0] < n_samples:
        idx = np.random.choice(points_np.shape[0], n_samples, replace=True)
        return points_np[idx]
    idx = np.random.choice(points_np.shape[0], n_samples, replace=False)
    return points_np[idx]


def augment_batch(batch_pc, batch_lbl):
    B, _, N = batch_pc.shape
    theta = (torch.rand(B, 1, 1, device=batch_pc.device) * (2 * np.pi / 12) - (np.pi / 12))
    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)
    rot = torch.zeros(B, 3, 3, device=batch_pc.device)
    rot[:, 0, 0] = cos_t.flatten()
    rot[:, 0, 1] = -sin_t.flatten()
    rot[:, 1, 0] = sin_t.flatten()
    rot[:, 1, 1] = cos_t.flatten()
    rot[:, 2, 2] = 1.0

    pc_t = batch_pc.transpose(1, 2)
    pc_rot = torch.bmm(pc_t, rot)

    lbl = batch_lbl.view(B, NUM_TARGET_POINTS, 3)
    lbl_rot = torch.bmm(lbl, rot)

    scale = torch.rand(B, 1, 1, device=batch_pc.device) * 0.1 + 0.95
    pc_rot = pc_rot * scale
    lbl_rot = lbl_rot * scale

    shift = (torch.rand(B, 1, 3, device=batch_pc.device) * 0.04) - 0.02
    pc_rot = pc_rot + shift
    lbl_rot = lbl_rot + shift

    jitter = torch.randn(B, N, 3, device=batch_pc.device) * 0.005
    pc_rot = pc_rot + jitter

    pc_out = pc_rot.transpose(1, 2)
    lbl_out = lbl_rot.view(B, -1)
    return pc_out, lbl_out


def compute_loss(pred, target):
    pred_pts = pred.view(-1, NUM_TARGET_POINTS, 3)
    tgt_pts = target.view(-1, NUM_TARGET_POINTS, 3)
    if LOSS_TYPE == "smoothl1":
        return F.smooth_l1_loss(pred_pts, tgt_pts)
    return F.mse_loss(pred_pts, tgt_pts)


# ---------- Data ----------
def load_data():
    with open(PROJECTS_LIST_FILE, "r", encoding="utf-8") as f:
        project_names = [ln.strip() for ln in f if ln.strip()]
    labels_df = pd.read_csv(LABELS_FILE, header=None)
    labels_np = labels_df.values.astype(np.float32)

    feats, labels, scales = [], [], []
    for i, name in enumerate(tqdm(project_names, desc="load")):
        pc_path = os.path.join(EXPORT_ROOT, name, "pointcloud_full.npy")
        if not os.path.exists(pc_path):
            continue
        pc = np.load(pc_path).astype(np.float32)
        if pc.shape[0] == 0:
            continue
        if ENABLE_DEDUPLICATION:
            pc = clean_point_cloud(pc, epsilon=DEDUP_EPSILON)
        label = labels_np[i].reshape(NUM_TARGET_POINTS, 3)
        pc_centroid = np.mean(pc, axis=0)
        pc_centered = pc - pc_centroid
        label_centered = label - pc_centroid
        scale_val = np.std(pc_centered)
        scale_val = 1.0 if scale_val < 1e-6 else scale_val
        pc_centered /= scale_val
        label_centered /= scale_val
        pc_sampled = farthest_point_sampling(pc_centered, MAX_POINTS)
        feats.append(pc_sampled.T)  # (3, N)
        labels.append(label_centered.flatten())
        scales.append(scale_val)
    X = np.stack(feats, axis=0)
    Y = np.stack(labels, axis=0)
    Scales = np.array(scales, dtype=np.float32)
    return X, Y, Scales


def predict_coarse(model_stage1, X_np, batch_size=12):
    model_stage1.eval()
    preds = []
    loader = DataLoader(torch.from_numpy(X_np).float(), batch_size=batch_size)
    with torch.no_grad():
        for pc in loader:
            coarse = model_stage1(pc.to(device)).cpu().numpy().reshape(-1, NUM_TARGET_POINTS, 3)
            preds.append(coarse)
    return np.concatenate(preds, axis=0)


def compute_error_stats(coarse, gt, scales):
    tgt = gt.reshape(-1, NUM_TARGET_POINTS, 3)
    err_norm = np.linalg.norm(coarse - tgt, axis=2)  # normalized space
    err_mm = err_norm * scales[:, None]
    pct = [50, 80, 90, 95]
    stats_norm = {f"p{p}": float(np.percentile(err_norm, p)) for p in pct}
    stats_mm = {f"p{p}": float(np.percentile(err_mm, p)) for p in pct}
    return stats_norm, stats_mm


class Stage2PatchDataset(Dataset):
    """On-the-fly patch cropping with random jitter to avoid overfitting."""

    def __init__(self, X, Y, coarse_preds, mode="train", radius=PATCH_RADIUS, jitter_std=CENTER_JITTER, n_points=PATCH_POINTS, mix_prob=0.5):
        self.X = X
        self.Y = Y
        self.coarse = coarse_preds
        self.mode = mode
        self.radius = radius
        self.jitter_std = jitter_std
        self.n_points = n_points
        self.mix_prob = mix_prob
        self.total = len(X) * NUM_TARGET_POINTS

    def __len__(self):
        return self.total

    def __getitem__(self, idx):
        sample_idx = idx // NUM_TARGET_POINTS
        lm_idx = idx % NUM_TARGET_POINTS
        pc = self.X[sample_idx].T  # (N,3)
        gt = self.Y[sample_idx].reshape(NUM_TARGET_POINTS, 3)[lm_idx]
        coarse = self.coarse[sample_idx, lm_idx]

        use_gt = False
        if self.mode == "train":
            use_gt = np.random.rand() < self.mix_prob
        center_base = gt if use_gt else coarse
        jitter = np.random.normal(scale=self.jitter_std, size=3) if (self.mode == "train" and self.jitter_std > 0) else 0.0
        center = center_base + jitter

        dists = np.linalg.norm(pc - center, axis=1)
        inside = np.where(dists < self.radius)[0]
        if inside.size == 0:
            inside = np.argsort(dists)[: self.n_points]
        if inside.size < self.n_points:
            repeat_idx = np.random.choice(inside, self.n_points - inside.size, replace=True)
            inside = np.concatenate([inside, repeat_idx])
        else:
            inside = np.random.choice(inside, self.n_points, replace=False)
        patch = pc[inside] - center
        residual = gt - center
        return patch.T.astype(np.float32), residual.astype(np.float32)


# ---------- Training ----------
def train_stage1_kfold(X, Y, Scales):
    n_samples = len(X)
    n_test = max(1, int(n_samples * 0.1))
    rng = np.random.RandomState(RANDOM_SEED)
    test_idx = rng.choice(n_samples, size=n_test, replace=False)
    train_idx = np.setdiff1d(np.arange(n_samples), test_idx)
    X_train_full, Y_train_full, Scales_train_full = X[train_idx], Y[train_idx], Scales[train_idx]
    X_test, Y_test, Scales_test = X[test_idx], Y[test_idx], Scales[test_idx]

    kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    best_overall_l2 = float("inf")
    best_state = None
    for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train_full), start=1):
        print(f"Fold {fold}/{K_FOLDS}")
        X_tr, X_val = X_train_full[tr_idx], X_train_full[val_idx]
        Y_tr, Y_val = Y_train_full[tr_idx], Y_train_full[val_idx]
        Scales_val = Scales_train_full[val_idx]

        model = PointNet2RegMSG(
            output_dim=OUTPUT_DIM,
            normal_channel=False,
            dropout=DROPOUT_STAGE1,
            sa1_radii=SA1_RADII_STAGE1,
            sa2_radii=SA2_RADII_STAGE1,
        ).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=LR_STAGE1, weight_decay=WEIGHT_DECAY)
        sched = torch.optim.lr_scheduler.StepLR(opt, step_size=LR_DECAY_STEP_STAGE1, gamma=LR_DECAY_GAMMA_STAGE1)

        tr_ds = TensorDataset(torch.from_numpy(X_tr).float(), torch.from_numpy(Y_tr).float())
        tr_loader = DataLoader(tr_ds, batch_size=BATCH_SIZE_STAGE1, shuffle=True)
        X_val_t = torch.from_numpy(X_val).float().to(device)
        Y_val_t = torch.from_numpy(Y_val).float().to(device)

        best_fold_l2 = float("inf")
        best_fold_state = None
        for epoch in range(1, NUM_EPOCHS_STAGE1 + 1):
            model.train()
            loss_accum = 0.0
            for pc, lbl in tr_loader:
                pc_aug, lbl_aug = augment_batch(pc, lbl)
                pc_aug = pc_aug.to(device)
                lbl_aug = lbl_aug.to(device)
                opt.zero_grad()
                pred = model(pc_aug)
                loss = compute_loss(pred, lbl_aug)
                loss.backward()
                opt.step()
                loss_accum += loss.item()
            sched.step()
            if epoch % 20 == 0 or epoch == NUM_EPOCHS_STAGE1:
                model.eval()
                with torch.no_grad():
                    val_pred = model(X_val_t)
                    val_loss = compute_loss(val_pred, Y_val_t).item()
                    val_np = val_pred.cpu().numpy().reshape(-1, NUM_TARGET_POINTS, 3)
                    Y_val_np = Y_val_t.cpu().numpy().reshape(-1, NUM_TARGET_POINTS, 3)
                    l2_norm = np.linalg.norm(val_np - Y_val_np, axis=2)
                    l2_mm = l2_norm * Scales_val[:, None]
                    val_l2 = np.mean(l2_mm)
                print(f"Epoch {epoch} fold {fold}: train {loss_accum/len(tr_loader):.6f} val {val_loss:.6f} l2_mm {val_l2:.4f} best {best_fold_l2:.4f}")
                if val_l2 < best_fold_l2:
                    best_fold_l2 = val_l2
                    best_fold_state = copy.deepcopy(model.state_dict())
        torch.save(best_fold_state, os.path.join(MODELS_DIR, f"pointnet2_stage1_fold{fold}_best.pth"))
        if best_fold_l2 < best_overall_l2:
            best_overall_l2 = best_fold_l2
            best_state = copy.deepcopy(best_fold_state)

    # retrain on full train set using best fold weights as init
    model_final = PointNet2RegMSG(
        output_dim=OUTPUT_DIM,
        normal_channel=False,
        dropout=DROPOUT_STAGE1,
        sa1_radii=SA1_RADII_STAGE1,
        sa2_radii=SA2_RADII_STAGE1,
    ).to(device)
    if best_state is not None:
        model_final.load_state_dict(best_state)
    opt = torch.optim.Adam(model_final.parameters(), lr=LR_STAGE1, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=LR_DECAY_STEP_STAGE1, gamma=LR_DECAY_GAMMA_STAGE1)
    tr_full_ds = TensorDataset(torch.from_numpy(X_train_full).float(), torch.from_numpy(Y_train_full).float())
    tr_full_loader = DataLoader(tr_full_ds, batch_size=BATCH_SIZE_STAGE1, shuffle=True)
    for epoch in range(1, NUM_EPOCHS_STAGE1 + 1):
        model_final.train()
        loss_accum = 0.0
        for pc, lbl in tr_full_loader:
            pc_aug, lbl_aug = augment_batch(pc, lbl)
            pc_aug = pc_aug.to(device)
            lbl_aug = lbl_aug.to(device)
            opt.zero_grad()
            pred = model_final(pc_aug)
            loss = compute_loss(pred, lbl_aug)
            loss.backward()
            opt.step()
            loss_accum += loss.item()
        sched.step()
        if epoch % 30 == 0 or epoch == NUM_EPOCHS_STAGE1:
            print(f"Stage1 final epoch {epoch}: loss {loss_accum/len(tr_full_loader):.6f}")
    stage1_path = os.path.join(MODELS_DIR, "pointnet2_stage1_final.pth")
    torch.save(model_final.state_dict(), stage1_path)
    print(f"saved Stage1: {stage1_path}")
    return model_final, (X_train_full, Y_train_full, Scales_train_full), (X_test, Y_test, Scales_test)


def train_stage2(model_stage1, X_train_full, Y_train_full, Scales_train_full, mix_prob=0.5):
    n_samples = len(X_train_full)
    n_val = max(1, int(n_samples * 0.1))
    rng = np.random.RandomState(RANDOM_SEED)
    val_idx = rng.choice(n_samples, size=n_val, replace=False)
    train_idx = np.setdiff1d(np.arange(n_samples), val_idx)

    X_tr, Y_tr, Scales_tr = X_train_full[train_idx], Y_train_full[train_idx], Scales_train_full[train_idx]
    X_val, Y_val, Scales_val = X_train_full[val_idx], Y_train_full[val_idx], Scales_train_full[val_idx]

    coarse_tr = predict_coarse(model_stage1, X_tr)
    stats_norm, stats_mm = compute_error_stats(coarse_tr, Y_tr, Scales_tr)
    jitter_std = max(CENTER_JITTER, stats_norm["p80"])
    radius_use = max(PATCH_RADIUS, stats_norm["p95"] * 1.2)
    print(f"Stage2 tuning from Stage1 errors (normalized): {stats_norm}")
    print(f"Stage2 tuning from Stage1 errors (mm): {stats_mm}")
    print(f"Stage2 jitter_std set to {jitter_std:.4f}, radius set to {radius_use:.4f} (normalized units)")

    tr_ds = Stage2PatchDataset(
        X_tr,
        Y_tr,
        coarse_preds=coarse_tr,
        mode="train",
        radius=radius_use,
        jitter_std=jitter_std,
        n_points=PATCH_POINTS,
        mix_prob=mix_prob,
    )
    tr_loader = DataLoader(tr_ds, batch_size=BATCH_SIZE_STAGE2, shuffle=True)

    model = PointNet2RegMSG(
        output_dim=3,
        normal_channel=False,
        dropout=DROPOUT_STAGE2,
        sa1_radii=SA1_RADII_STAGE2,
        sa2_radii=SA2_RADII_STAGE2,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR_STAGE2, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=LR_DECAY_STEP_STAGE2, gamma=LR_DECAY_GAMMA_STAGE2)

    best_refined_mm = float("inf")
    best_state = None

    for epoch in range(1, NUM_EPOCHS_STAGE2 + 1):
        model.train()
        loss_accum = 0.0
        center_dists_sample = None  # capture one batch of center-vs-GT distances for B4 sanity
        for patch, res in tr_loader:
            patch = patch.to(device)
            res = res.to(device)
            opt.zero_grad()
            pred = model(patch)
            loss = F.smooth_l1_loss(pred, res) if LOSS_TYPE == "smoothl1" else F.mse_loss(pred, res)
            loss.backward()
            opt.step()
            loss_accum += loss.item()
            if center_dists_sample is None:
                center_dists_sample = torch.norm(res, dim=1).detach().cpu().numpy()
        sched.step()

        if epoch % 10 == 0 or epoch == NUM_EPOCHS_STAGE2:
            metrics = evaluate_pipeline(model_stage1, model, X_val, Y_val, Scales_val, radius=radius_use)
            val_refined = metrics["refined_mean_mm"]
            if center_dists_sample is not None:
                dist_stats = {
                    "p50": float(np.percentile(center_dists_sample, 50)),
                    "p90": float(np.percentile(center_dists_sample, 90)),
                    "max": float(np.max(center_dists_sample)),
                }
                print(f"Stage2 epoch {epoch}: center-vs-GT dist (normed) p50 {dist_stats['p50']:.4f} p90 {dist_stats['p90']:.4f} max {dist_stats['max']:.4f}")
            print(
                f"Stage2 epoch {epoch}: train_loss {loss_accum/len(tr_loader):.6f} "
                f"val_refined_mm {val_refined:.4f} best {best_refined_mm:.4f}"
            )
            if val_refined < best_refined_mm:
                best_refined_mm = val_refined
                best_state = copy.deepcopy(model.state_dict())

    if best_state is not None:
        model.load_state_dict(best_state)
    stage2_path = os.path.join(MODELS_DIR, "pointnet2_stage2_refiner_best.pth")
    torch.save(model.state_dict(), stage2_path)
    print(f"saved Stage2 best: {stage2_path} (val refined_mean_mm={best_refined_mm:.4f})")
    return model, radius_use


# ---------- Inference pipeline ----------
def crop_patch(pc_np, center, radius=PATCH_RADIUS, n_points=PATCH_POINTS):
    dists = np.linalg.norm(pc_np - center, axis=1)
    inside = np.where(dists < radius)[0]
    if inside.size == 0:
        inside = np.argsort(dists)[:n_points]
    if inside.size < n_points:
        repeat_idx = np.random.choice(inside, n_points - inside.size, replace=True)
        inside = np.concatenate([inside, repeat_idx])
    else:
        inside = np.random.choice(inside, n_points, replace=False)
    patch = pc_np[inside] - center
    return patch.T


def refine_batch(model_stage1, model_stage2, X_batch_np, radius=PATCH_RADIUS):
    model_stage1.eval()
    model_stage2.eval()
    with torch.no_grad():
        pc_t = torch.from_numpy(X_batch_np).float().to(device)
        coarse = model_stage1(pc_t).cpu().numpy().reshape(-1, NUM_TARGET_POINTS, 3)
    refined = []
    for i in range(X_batch_np.shape[0]):
        pc_np = X_batch_np[i].T
        coarse_i = coarse[i]
        patches = []
        centers = []
        for j in range(NUM_TARGET_POINTS):
            patch = crop_patch(pc_np, coarse_i[j], radius=radius)
            patches.append(patch)
            centers.append(coarse_i[j])
        patches_t = torch.from_numpy(np.stack(patches, axis=0)).float().to(device)
        with torch.no_grad():
            res = model_stage2(patches_t).cpu().numpy().reshape(-1, 3)
        refined_pts = coarse_i + res
        refined.append(refined_pts)
    return np.stack(refined, axis=0), coarse


# ---------- Evaluation ----------
def evaluate_pipeline(model_stage1, model_stage2, X, Y, Scales, radius=PATCH_RADIUS):
    refined, coarse = refine_batch(model_stage1, model_stage2, X, radius=radius)
    tgt = Y.reshape(-1, NUM_TARGET_POINTS, 3)
    l2_coarse = np.linalg.norm(coarse - tgt, axis=2) * Scales[:, None]
    l2_refined = np.linalg.norm(refined - tgt, axis=2) * Scales[:, None]
    return {
        "coarse_mean_mm": float(np.mean(l2_coarse)),
        "refined_mean_mm": float(np.mean(l2_refined)),
        "coarse_per_landmark_mm": np.mean(l2_coarse, axis=0).tolist(),
        "refined_per_landmark_mm": np.mean(l2_refined, axis=0).tolist(),
    }


def main():
    X, Y, Scales = load_data()
    model_stage1, train_split, test_split = train_stage1_kfold(X, Y, Scales)
    X_train_full, Y_train_full, Scales_train_full = train_split
    X_test, Y_test, Scales_test = test_split

    # train Stage2 on full train+val portion (same 90%)
    model_stage2, radius_use = train_stage2(model_stage1, X_train_full, Y_train_full, Scales_train_full)

    # Evaluate on held-out test set
    metrics = evaluate_pipeline(model_stage1, model_stage2, X_test, Y_test, Scales_test, radius=radius_use)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
