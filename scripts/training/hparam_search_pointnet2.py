"""
Hyperparameter search + final training for PointNet++ landmark regression.
Features:
- 10% held-out test, rest for train/val search
- Grid search over lr, dropout, weight_decay, loss type (MSE or SmoothL1)
- Weighted landmark loss + geometry regularization (pairwise distance consistency)
- Uses pointnet2_ops FPS and vectorized torch augmentations
- Retrains best config on train+val, evaluates on test
"""
import os
import sys
import json
import copy
import logging
from datetime import datetime
import itertools
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(BASE_DIR))
UTILS_DIR = os.path.join(ROOT_DIR, "scripts", "utils")
MODELS_DIR = os.path.join(ROOT_DIR, "models")
for p in (ROOT_DIR, UTILS_DIR, MODELS_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from pointnet2_reg import PointNet2RegMSG
from pointnet2_ops.pointnet2_utils import furthest_point_sample

# Data files
EXPORT_ROOT = os.path.join(ROOT_DIR, 'data', 'pointcloud')
LABELS_FILE = os.path.join(ROOT_DIR, 'results', 'labels.csv')
PROJECTS_LIST_FILE = os.path.join(ROOT_DIR, 'results', 'valid_projects.txt')

NUM_TARGET_POINTS = 9
OUTPUT_DIM = NUM_TARGET_POINTS * 3
MAX_POINTS = 8192
RANDOM_SEED = 42

# Search space
SEARCH_EPOCHS = 80
FINAL_EPOCHS = 160
BATCH_SIZE = 8
LR_DECAY_STEP = 60
LR_DECAY_GAMMA = 0.7

LR_LIST = [1e-3, 5e-4]
DROPOUT_LIST = [0.3, 0.4]
WEIGHT_DECAY_LIST = [0.0, 1e-4]
LOSS_TYPES = ["mse", "smoothl1"]
GEO_LAMBDAS = [0.0, 0.01, 0.05, 0.1]

# SA radii search (nsamples kept constant)
SA1_RADII_LIST = [
    [0.1, 0.2, 0.4],
    [0.05, 0.1, 0.2],
]
SA2_RADII_LIST = [
    [0.2, 0.4, 0.8],
    [0.1, 0.2, 0.4],
]

# Landmark weights (can be tuned; default uniform)
LANDMARK_WEIGHTS = [1.0] * NUM_TARGET_POINTS

# Logging
LOG_DIR = os.path.join(ROOT_DIR, 'results', 'logs')
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, f"hparam_search_pointnet2_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler(LOG_FILE, encoding='utf-8'), logging.StreamHandler(sys.stdout)],
)


def set_seed(seed=RANDOM_SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def farthest_point_sampling(points_np, n_samples, device):
    """FPS with CUDA path; fallback to random choice on CPU."""
    if device.type == 'cuda':
        points_t = torch.from_numpy(points_np).float().unsqueeze(0).to(device)
        idx = furthest_point_sample(points_t, n_samples)  # (1, n_samples)
        return points_np[idx[0].cpu().numpy()]
    # CPU fallback: uniform random subset to avoid pointnet2_ops CUDA requirement
    if points_np.shape[0] <= n_samples:
        return points_np
    choice = np.random.choice(points_np.shape[0], size=n_samples, replace=False)
    return points_np[choice]


def augment_batch(batch_pc, batch_lbl):
    """Vectorized augmentation on torch tensors (runs on whatever device tensors already on)."""
    B, _, N = batch_pc.shape
    device_local = batch_pc.device

    theta = (torch.rand(B, 1, 1, device=device_local) * (2 * np.pi / 12) - (np.pi / 12))
    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)
    rot = torch.zeros(B, 3, 3, device=device_local)
    rot[:, 0, 0] = cos_t.squeeze(-1)
    rot[:, 0, 1] = -sin_t.squeeze(-1)
    rot[:, 1, 0] = sin_t.squeeze(-1)
    rot[:, 1, 1] = cos_t.squeeze(-1)
    rot[:, 2, 2] = 1.0

    pc_t = batch_pc.transpose(1, 2)  # (B, N, 3)
    pc_rot = torch.bmm(pc_t, rot)    # (B, N, 3)

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

    pc_out = pc_rot.transpose(1, 2)  # back to (B, 3, N)
    lbl_out = lbl_rot.view(B, -1)
    return pc_out, lbl_out


def load_data(device):
    with open(PROJECTS_LIST_FILE, 'r', encoding='utf-8') as f:
        project_names = [ln.strip() for ln in f if ln.strip()]
    labels_np = np.loadtxt(LABELS_FILE, delimiter=',').astype(np.float32)

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

        pc_sampled = farthest_point_sampling(pc_centered, MAX_POINTS, device)
        feats.append(pc_sampled.T)          # (3, N)
        labels.append(label_centered.flatten())

    X = np.stack(feats, axis=0)
    Y = np.stack(labels, axis=0)
    return X, Y


def make_loaders(X_train, Y_train, X_val, Y_val, batch_size):
    train_ds = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(Y_train).float())
    val_ds = None
    if X_val is not None and Y_val is not None:
        val_ds = TensorDataset(torch.from_numpy(X_val).float(), torch.from_numpy(Y_val).float())
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = None if val_ds is None else DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def compute_loss(pred, target, loss_type, landmark_weights, geo_lambda):
    # pred/target: (B, 27)
    B = pred.shape[0]
    pred_pts = pred.view(B, NUM_TARGET_POINTS, 3)
    tgt_pts = target.view(B, NUM_TARGET_POINTS, 3)

    if loss_type == "smoothl1":
        base = F.smooth_l1_loss(pred_pts, tgt_pts, reduction='none')
    else:
        base = F.mse_loss(pred_pts, tgt_pts, reduction='none')

    lw = torch.tensor(landmark_weights, device=pred.device).view(1, NUM_TARGET_POINTS, 1)
    base = (base * lw).mean()

    if geo_lambda > 0:
        dist_pred = torch.cdist(pred_pts, pred_pts)
        dist_true = torch.cdist(tgt_pts, tgt_pts)
        geo_loss = F.l1_loss(dist_pred, dist_true)
        return base + geo_lambda * geo_loss, base.item(), geo_loss.item()
    else:
        return base, base.item(), 0.0


def train_one_run(config, train_loader, val_loader, device, epochs, landmark_weights, use_val_selection=True):
    model = PointNet2RegMSG(
        output_dim=OUTPUT_DIM,
        normal_channel=False,
        dropout=config['dropout'],
        sa1_radii=config['sa1_radii'],
        sa2_radii=config['sa2_radii'],
    ).to(device)
    criterion_type = config['loss_type']
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=LR_DECAY_STEP, gamma=LR_DECAY_GAMMA)

    best_metric = float('inf')  # val L2 mean for selection
    best_val_loss = None
    best_state = None
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_accum = 0.0
        batches = 0
        for batch_pc, batch_lbl in train_loader:
            batch_pc = batch_pc.to(device)
            batch_lbl = batch_lbl.to(device)
            batch_aug_pc, batch_aug_lbl = augment_batch(batch_pc, batch_lbl)

            optimizer.zero_grad()
            pred = model(batch_aug_pc)
            loss, base_loss_val, geo_loss_val = compute_loss(pred, batch_aug_lbl, criterion_type, landmark_weights, config['geo_lambda'])
            loss.backward()
            optimizer.step()

            train_loss_accum += loss.item()
            batches += 1

        scheduler.step()
        avg_train = train_loss_accum / max(1, batches)

        avg_val = None
        avg_val_l2 = None
        if val_loader is not None:
            model.eval()
            with torch.no_grad():
                val_loss_accum = 0.0
                val_l2_accum = 0.0
                val_count = 0
                for v_pc, v_lbl in val_loader:
                    v_pc = v_pc.to(device)
                    v_lbl = v_lbl.to(device)
                    v_pred = model(v_pc)
                    v_loss, _, _ = compute_loss(v_pred, v_lbl, criterion_type, landmark_weights, config['geo_lambda'])
                    val_loss_accum += v_loss.item()

                    v_pred_np = v_pred.detach().cpu().numpy().reshape(-1, NUM_TARGET_POINTS, 3)
                    v_lbl_np = v_lbl.detach().cpu().numpy().reshape(-1, NUM_TARGET_POINTS, 3)
                    val_l2 = np.linalg.norm(v_pred_np - v_lbl_np, axis=2)
                    val_l2_accum += val_l2.sum()
                    val_count += val_l2.size

                avg_val = val_loss_accum / max(1, len(val_loader))
                avg_val_l2 = val_l2_accum / max(1, val_count)

        history.append({'epoch': epoch, 'train_loss': avg_train, 'val_loss': avg_val, 'val_l2_mean': avg_val_l2, 'lr': optimizer.param_groups[0]['lr']})

        if val_loader is not None and use_val_selection:
            if avg_val_l2 is not None and avg_val_l2 < best_metric:
                best_metric = avg_val_l2
                best_val_loss = avg_val
                best_state = copy.deepcopy(model.state_dict())
        else:
            # For train+val final run, keep last epoch
            best_metric = avg_train
            best_state = copy.deepcopy(model.state_dict())

        if epoch % 20 == 0 or epoch == epochs:
            logging.info("Epoch %d/%d - train %.6f val %s val_l2 %s (best metric %.6f)",
                         epoch, epochs, avg_train,
                         f"{avg_val:.6f}" if avg_val is not None else "-",
                         f"{avg_val_l2:.6f}" if avg_val_l2 is not None else "-",
                         best_metric)

    return best_metric, best_state, history, best_val_loss


def evaluate(model_state, config, X_test, Y_test, device, landmark_weights):
    model = PointNet2RegMSG(
        output_dim=OUTPUT_DIM,
        normal_channel=False,
        dropout=config['dropout'],
        sa1_radii=config['sa1_radii'],
        sa2_radii=config['sa2_radii'],
    ).to(device)
    model.load_state_dict(model_state)
    model.eval()
    with torch.no_grad():
        X_t = torch.from_numpy(X_test).float().to(device)
        Y_t = torch.from_numpy(Y_test).float().to(device)
        pred = model(X_t)
        loss, base_loss_val, geo_loss_val = compute_loss(pred, Y_t, config['loss_type'], landmark_weights, config['geo_lambda'])
        # L2 per-landmark distances
        pred_np = pred.cpu().numpy().reshape(-1, NUM_TARGET_POINTS, 3)
        tgt_np = Y_test.reshape(-1, NUM_TARGET_POINTS, 3)
        l2 = np.linalg.norm(pred_np - tgt_np, axis=2)
        return {
            'loss': float(loss.item()),
            'base_loss': float(base_loss_val),
            'geo_loss': float(geo_loss_val),
            'l2_mean': float(l2.mean()),
            'l2_std': float(l2.std()),
            'l2_per_landmark': l2.mean(axis=0).tolist(),
        }


def main():
    set_seed()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info("device: %s", device)

    X, Y = load_data(device)
    n_samples = len(X)
    n_test = max(1, int(n_samples * 0.1))
    X_temp, X_test, Y_temp, Y_test = train_test_split(X, Y, test_size=n_test, random_state=RANDOM_SEED, shuffle=True)
    X_train, X_val, Y_train, Y_val = train_test_split(X_temp, Y_temp, test_size=0.2, random_state=RANDOM_SEED, shuffle=True)

    logging.info("train %d, val %d, test %d", len(X_train), len(X_val), len(X_test))

    landmark_weights = LANDMARK_WEIGHTS

    search_results = []
    best_overall = None

    grid = itertools.product(LR_LIST, DROPOUT_LIST, WEIGHT_DECAY_LIST, LOSS_TYPES, GEO_LAMBDAS, SA1_RADII_LIST, SA2_RADII_LIST)
    for lr, dropout, wd, loss_type, geo_lambda, sa1_r, sa2_r in grid:
        config = {
            'lr': lr,
            'dropout': dropout,
            'weight_decay': wd,
            'loss_type': loss_type,
            'geo_lambda': geo_lambda,
            'sa1_radii': sa1_r,
            'sa2_radii': sa2_r,
        }
        logging.info("Config: %s", config)

        train_loader, val_loader = make_loaders(X_train, Y_train, X_val, Y_val, BATCH_SIZE)
        val_metric, state, hist, best_val_loss = train_one_run(
            config, train_loader, val_loader, device, SEARCH_EPOCHS, landmark_weights, use_val_selection=True
        )
        result = {'config': config, 'val_l2_mean': val_metric, 'val_loss_at_best': best_val_loss, 'history': hist}
        search_results.append(result)

        if best_overall is None or val_metric < best_overall['val_l2_mean']:
            best_overall = {'config': config, 'val_l2_mean': val_metric, 'val_loss_at_best': best_val_loss, 'state': state}

    logging.info("Best config: %s with val_l2_mean %.6f", best_overall['config'], best_overall['val_l2_mean'])

    # Retrain on train+val with best config
    best_config = best_overall['config']
    logging.info("Retraining best config on train+val for %d epochs (no val selection)", FINAL_EPOCHS)
    train_loader_full, _ = make_loaders(X_temp, Y_temp, None, None, BATCH_SIZE)
    final_metric, final_state, final_hist, _ = train_one_run(
        best_config, train_loader_full, None, device, FINAL_EPOCHS, landmark_weights, use_val_selection=False
    )

    # Evaluate on test
    test_metrics = evaluate(final_state, best_config, X_test, Y_test, device, landmark_weights)
    logging.info("Test metrics: %s", test_metrics)

    # Save model
    os.makedirs(MODELS_DIR, exist_ok=True)
    best_model_path = os.path.join(MODELS_DIR, "pointnet2_regression_hparam_best.pth")
    torch.save(final_state, best_model_path)

    # Save summary
    summary = {
        'best_config': best_config,
        'best_val_l2_mean': best_overall['val_l2_mean'],
        'best_val_loss_at_best_l2': best_overall['val_loss_at_best'],
        'search_results': [{k: (v if k != 'history' else None) for k, v in r.items()} for r in search_results],
        'test_metrics': test_metrics,
        'log_file': LOG_FILE,
        'model_path': best_model_path,
    }
    os.makedirs(os.path.join(ROOT_DIR, "results", "training_histories"), exist_ok=True)
    summary_path = os.path.join(ROOT_DIR, "results", "training_histories", "hparam_search_pointnet2.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logging.info("Done. Best model saved to %s", best_model_path)
    logging.info("Summary saved to %s", summary_path)


if __name__ == "__main__":
    main()
