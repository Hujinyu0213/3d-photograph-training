"""
PointNet regression training with data augmentation + FPS sampling.
Changes vs. main_script_full_pointcloud:
- Adds on-the-fly augmentations (rot z, scale, translate, jitter).
- Uses farthest point sampling (FPS) to select a stable 8192-point subset.
- Uses updated data paths (labels/results) consistent with reorganized project.
- Keeps explicit train/val split (configurable VAL_RATIO).
"""
import os
import sys
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split
from tqdm import tqdm

# ensure utils module is importable after repo re-org
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# project root = two levels up (PointFeatureProject)
ROOT_DIR = os.path.dirname(os.path.dirname(BASE_DIR))
UTILS_DIR = os.path.join(ROOT_DIR, "scripts", "utils")
for p in (ROOT_DIR, UTILS_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

# robust import even if package init files are missing
import importlib.util
_pn_path = os.path.join(UTILS_DIR, "pointnet_utils.py")
if not os.path.exists(_pn_path):
    raise FileNotFoundError(f"pointnet_utils.py not found at {_pn_path}")
spec = importlib.util.spec_from_file_location("pointnet_utils", _pn_path)
pointnet_utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pointnet_utils)
PointNetEncoder = pointnet_utils.PointNetEncoder
feature_transform_reguliarzer = pointnet_utils.feature_transform_reguliarzer

# =========================================================
# Paths
# =========================================================
EXPORT_ROOT = os.path.join(ROOT_DIR, 'data', 'pointcloud')
LABELS_FILE = os.path.join(ROOT_DIR, 'results', 'labels.csv')
PROJECTS_LIST_FILE = os.path.join(ROOT_DIR, 'results', 'valid_projects.txt')

NUM_TARGET_POINTS = 9
OUTPUT_DIM = NUM_TARGET_POINTS * 3

# =========================================================
# Training config
# =========================================================
BATCH_SIZE = 8
NUM_EPOCHS = 400
LEARNING_RATE = 0.001
LR_DECAY_STEP = 120
LR_DECAY_GAMMA = 0.5
DROPOUT_RATE = 0.3
FEATURE_TRANSFORM_WEIGHT = 0.001
MAX_POINTS = 8192
TRAIN_RATIO = 0.8
VAL_RATIO = 0.2  # overrides TRAIN_RATIO if both set; keep val set explicitly

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🖥️  device: {device}")
if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name(0)}")

# =========================================================
# Model
# =========================================================
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

# =========================================================
# FPS sampler (numpy)
# =========================================================
def farthest_point_sampling(points_np, n_samples):
    """points_np: (N,3) -> return (n_samples,3) unique points via FPS."""
    N = points_np.shape[0]
    if N <= n_samples:
        # pad by repeating
        idx = np.random.choice(N, n_samples, replace=True)
        return points_np[idx]
    # initialize
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

# =========================================================
# Augmentations
# =========================================================
def augment(points_np):
    """points_np: (3,N) -> augmented (3,N)"""
    pts = points_np.copy()
    # rot around z
    theta = np.random.uniform(-np.pi/12, np.pi/12)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s, 0],[s, c, 0],[0,0,1]], dtype=np.float32)
    pts = R @ pts
    # scale
    scale = np.random.uniform(0.95, 1.05)
    pts *= scale
    # translate small
    trans = np.random.uniform(-0.02, 0.02, size=(3,1)).astype(np.float32)
    pts += trans
    # jitter
    noise = np.random.normal(0, 0.005, size=pts.shape).astype(np.float32)
    pts += noise
    return pts

# =========================================================
# Data loading
# =========================================================
def load_data():
    if not os.path.exists(PROJECTS_LIST_FILE):
        raise FileNotFoundError(f"missing project list: {PROJECTS_LIST_FILE}")
    if not os.path.exists(LABELS_FILE):
        raise FileNotFoundError(f"missing labels: {LABELS_FILE}")

    with open(PROJECTS_LIST_FILE, 'r', encoding='utf-8') as f:
        project_names = [ln.strip() for ln in f if ln.strip()]
    labels_df = pd.read_csv(LABELS_FILE, header=None)
    labels_np = labels_df.values.astype(np.float32)
    if len(project_names) != len(labels_np):
        m = min(len(project_names), len(labels_np))
        project_names = project_names[:m]
        labels_np = labels_np[:m]

    feats = []
    labels = []
    counts = []

    for i, name in enumerate(tqdm(project_names, desc="load pointcloud")):
        pc_path = os.path.join(EXPORT_ROOT, name, "pointcloud_full.npy")
        if not os.path.exists(pc_path):
            continue
        pc = np.load(pc_path).astype(np.float32)  # (N,3)
        if pc.shape[0] == 0:
            continue
        label = labels_np[i].reshape(NUM_TARGET_POINTS, 3)
        # center using landmark centroid
        label_centroid = np.mean(label, axis=0)
        pc_centered = pc - label_centroid
        label_centered = label - label_centroid
        # scale by std
        scale = np.std(pc_centered)
        if scale > 1e-6:
            pc_centered /= scale
            label_centered /= scale
        # FPS to fixed size
        pc_sampled = farthest_point_sampling(pc_centered, MAX_POINTS)  # (MAX_POINTS,3)
        pc_T = pc_sampled.T  # (3,MAX_POINTS)
        feats.append(pc_T)
        labels.append(label_centered.flatten())
        counts.append(pc.shape[0])

    if not feats:
        raise RuntimeError("no valid samples loaded")
    X = torch.from_numpy(np.stack(feats, axis=0)).float()
    Y = torch.from_numpy(np.stack(labels, axis=0)).float()
    print(f"loaded {len(X)} samples; min/max/avg points raw: {min(counts)}/{max(counts)}/{np.mean(counts):.0f}")
    print(f"tensor shapes: X={X.shape}, Y={Y.shape}")
    return X, Y

# =========================================================
# Training
# =========================================================
def train():
    X, Y = load_data()
    dataset = TensorDataset(X, Y)
    val_size = max(1, int(VAL_RATIO * len(dataset)))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

    model = PointNetRegressor(output_dim=OUTPUT_DIM, dropout_rate=DROPOUT_RATE).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=LR_DECAY_STEP, gamma=LR_DECAY_GAMMA)

    best_val = float('inf')
    history = {"epoch": [], "train_loss": [], "val_loss": []}

    for epoch in range(NUM_EPOCHS):
        model.train()
        total = 0
        n = 0
        for data, target in train_loader:
            # apply augmentation per batch
            data_aug = []
            for d in data.numpy():  # d: (3,MAX_POINTS)
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

        history["epoch"].append(epoch + 1)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), os.path.join(ROOT_DIR, "models", "pointnet_regression_model_full_aug_fps_best.pth"))

        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:4d}/{NUM_EPOCHS} | train {train_loss:.6f} | val {val_loss:.6f} | best {best_val:.6f}")

    torch.save(model.state_dict(), os.path.join(ROOT_DIR, "models", "pointnet_regression_model_full_aug_fps_final.pth"))
    with open(os.path.join(ROOT_DIR, "results", "training_history_full_aug_fps.json"), "w") as f:
        json.dump(history, f, indent=2)
    print("done. best val=", best_val)


if __name__ == "__main__":
    train()
