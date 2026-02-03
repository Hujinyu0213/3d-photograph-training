"""
================================================================================
PointNet++ Regression Model Evaluation Script
================================================================================
Evaluate a trained PointNet++ model and compute L2 distance errors for each 
landmark in three different units:

  1. Normalized space (after dataset std normalization)
  2. Original unit (meters or millimeters, depending on input data)
  3. Millimeters (assumes original unit is meters; ignore if already in mm)

Usage:
    python scripts/evaluation/eval_pointnet2_l2.py --model <path_to_model.pth>

Example:
    python scripts/evaluation/eval_pointnet2_l2.py --model models/pointnet2_regression_kfold_best.pth

Output:
    - Console: Summary statistics for all landmarks
    - JSON file: Detailed metrics saved to results/evaluation/eval_pointnet2_l2.json
================================================================================
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, random_split


# ============================================================================
#  Path Configuration
# ============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(BASE_DIR))
UTILS_DIR = os.path.join(ROOT_DIR, "scripts", "utils")
MODELS_DIR = os.path.join(ROOT_DIR, "models")

for p in (ROOT_DIR, UTILS_DIR, MODELS_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from pointnet2_reg import PointNet2RegMSG
from pointnet2_ops.pointnet2_utils import furthest_point_sample


# ============================================================================
#  Constants
# ============================================================================
EXPORT_ROOT = os.path.join(ROOT_DIR, 'data', 'pointcloud')
LABELS_FILE = os.path.join(ROOT_DIR, 'results', 'labels.csv')
PROJECTS_LIST_FILE = os.path.join(ROOT_DIR, 'results', 'valid_projects.txt')

NUM_TARGET_POINTS = 9
OUTPUT_DIM = NUM_TARGET_POINTS * 3
MAX_POINTS = 8192
BATCH_SIZE = 16

# Landmark names (9 landmarks)
LANDMARK_NAMES = [
    'Landmark 1',
    'Landmark 2',
    'Landmark 3',
    'Landmark 4',
    'Landmark 5',
    'Landmark 6',
    'Landmark 7',
    'Landmark 8',
    'Landmark 9',
]

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ============================================================================
#  Helper Functions
# ============================================================================

def farthest_point_sampling(points_np, n_samples):
    """
    Perform Farthest Point Sampling (FPS) on point cloud.
    
    Uses GPU-accelerated version when CUDA is available, falls back to
    CPU numpy implementation otherwise.
    
    Args:
        points_np: Point cloud array of shape (N, 3)
        n_samples: Number of points to sample
        
    Returns:
        Sampled points array of shape (n_samples, 3)
    """
    if torch.cuda.is_available():
        # GPU version
        points_t = torch.from_numpy(points_np).float().unsqueeze(0).to(DEVICE)
        idx = furthest_point_sample(points_t, n_samples)
        return points_np[idx[0].cpu().numpy()]
    
    # CPU fallback: numpy FPS implementation
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


def load_data():
    """
    Load and preprocess all point cloud data and labels.
    
    Returns:
        X: Point cloud features, shape (num_samples, 3, MAX_POINTS)
        Y: Landmark labels (normalized), shape (num_samples, 27)
        S: Scale factors for denormalization, shape (num_samples,)
    """
    print("Loading project list and labels...")
    
    with open(PROJECTS_LIST_FILE, 'r', encoding='utf-8') as f:
        project_names = [ln.strip() for ln in f if ln.strip()]
    
    labels_np = np.loadtxt(LABELS_FILE, delimiter=',').astype(np.float32)

    feats, labels, scales = [], [], []
    
    print(f"Processing {len(project_names)} projects...")
    
    for i, name in enumerate(project_names):
        pc_path = os.path.join(EXPORT_ROOT, name, "pointcloud_full.npy")
        
        if not os.path.exists(pc_path):
            continue
            
        pc = np.load(pc_path).astype(np.float32)
        if pc.shape[0] == 0:
            continue

        # Center by landmark centroid
        label = labels_np[i].reshape(NUM_TARGET_POINTS, 3)
        label_centroid = np.mean(label, axis=0)
        pc_centered = pc - label_centroid
        label_centered = label - label_centroid

        # Normalize by std
        scale = np.std(pc_centered)
        if scale > 1e-6:
            pc_centered /= scale
            label_centered /= scale
        else:
            scale = 1.0

        # Sample points using FPS
        pc_sampled = farthest_point_sampling(pc_centered, MAX_POINTS)
        
        feats.append(pc_sampled.T)  # Shape: (3, N)
        labels.append(label_centered.flatten())
        scales.append(scale)

    X = np.stack(feats, axis=0)
    Y = np.stack(labels, axis=0)
    S = np.array(scales, dtype=np.float32)
    
    print(f"Loaded {len(X)} samples successfully.\n")
    
    return X, Y, S


# ============================================================================
#  Main Evaluation Function
# ============================================================================

def evaluate(model_path):
    """
    Evaluate the trained model on the 10% held-out test set.
    Uses the SAME test split as k-fold training (np.random.RandomState(42)).
    
    Args:
        model_path: Path to the .pth model weights file
    """
    print("=" * 80)
    print("POINTNET++ LANDMARK REGRESSION - MODEL EVALUATION")
    print("=" * 80)
    print(f"Model: {model_path}")
    print(f"Device: {DEVICE}")
    print("=" * 80)
    print()
    
    # Load all data
    X, Y, S = load_data()
    
    # Use EXACT same split as k-fold training script
    # (10% test using np.random.RandomState(42))
    print("Using same 10% test split as k-fold training...")
    n_samples = len(X)
    n_test = max(1, int(n_samples * 0.1))
    
    test_indices = np.random.RandomState(42).choice(n_samples, size=n_test, replace=False)
    train_indices = np.setdiff1d(np.arange(n_samples), test_indices)
    
    X_test = X[test_indices]
    Y_test = Y[test_indices]
    S_test = S[test_indices]
    
    print(f"Test set: {len(X_test)} samples (10%)")
    print(f"Train set: {len(train_indices)} samples (90%)")
    print()
    
    # Load model
    print("Loading model...")
    model = PointNet2RegMSG(output_dim=OUTPUT_DIM, normal_channel=False).to(DEVICE)
    state = torch.load(model_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    print("Model loaded successfully.\n")
    
    # Run inference on test set only
    print("Running inference on test set...")
    X_test_t = torch.from_numpy(X_test).float().to(DEVICE)
    Y_test_t = torch.from_numpy(Y_test).float().to(DEVICE)
    
    with torch.no_grad():
        pred = model(X_test_t)
    
    pred_np = pred.cpu().numpy()
    print("Inference complete.\n")
    
    # Compute metrics
    print("Computing metrics...")
    pred_pts = pred_np.reshape(-1, NUM_TARGET_POINTS, 3)
    tgt_pts = Y_test.reshape(-1, NUM_TARGET_POINTS, 3)

    # L2 distances in normalized space
    l2_norm = np.linalg.norm(pred_pts - tgt_pts, axis=2)
    
    # Denormalize to original unit
    l2_orig = l2_norm * S_test[:, None]
    
    # Convert to millimeters (assuming original unit is meters)
    l2_mm = l2_orig * 1000.0

    # Build metrics dictionary
    metrics = {
        'model_path': model_path,
        'device': str(DEVICE),
        'num_samples': len(X_test),
        'num_landmarks': NUM_TARGET_POINTS,
        'landmark_names': LANDMARK_NAMES,
        'test_split_method': '10% using np.random.RandomState(42) - matches k-fold training',
        
        'l2_normalized': {
            'mean': float(l2_norm.mean()),
            'std': float(l2_norm.std()),
            'per_landmark_mean': l2_norm.mean(axis=0).tolist(),
            'per_landmark_names': LANDMARK_NAMES,
        },
        
        'l2_original_unit': {
            'mean': float(l2_orig.mean()),
            'std': float(l2_orig.std()),
            'per_landmark_mean': l2_orig.mean(axis=0).tolist(),
            'per_landmark_names': LANDMARK_NAMES,
        },
        
        'l2_millimeters_if_input_is_meters': {
            'mean': float(l2_mm.mean()),
            'std': float(l2_mm.std()),
            'per_landmark_mean': l2_mm.mean(axis=0).tolist(),
            'per_landmark_names': LANDMARK_NAMES,
        },
    }
    
    # Print results to console
    print_results(metrics)
    
    # Save to JSON
    save_results(metrics)
    
    return metrics


def print_results(metrics):
    """Print evaluation results to console in a readable format."""
    print()
    print("=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)
    print()
    
    print(f"Number of samples: {metrics['num_samples']}")
    print(f"Number of landmarks: {metrics['num_landmarks']}")
    print()
    
    print("-" * 80)
    print("L2 DISTANCE - NORMALIZED SPACE")
    print("-" * 80)
    print(f"  Mean: {metrics['l2_normalized']['mean']:.6f}")
    print(f"  Std:  {metrics['l2_normalized']['std']:.6f}")
    print(f"  Per-landmark:")
    for i, val in enumerate(metrics['l2_normalized']['per_landmark_mean']):
        print(f"    {LANDMARK_NAMES[i]:15s}: {val:.6f}")
    print()
    
    print("-" * 80)
    print("L2 DISTANCE - ORIGINAL UNIT (meters or mm, depends on input)")
    print("-" * 80)
    print(f"  Mean: {metrics['l2_original_unit']['mean']:.6f}")
    print(f"  Std:  {metrics['l2_original_unit']['std']:.6f}")
    print(f"  Per-landmark:")
    for i, val in enumerate(metrics['l2_original_unit']['per_landmark_mean']):
        print(f"    {LANDMARK_NAMES[i]:15s}: {val:.6f}")
    print()
    
    print("-" * 80)
    print("L2 DISTANCE - MILLIMETERS (assuming input is in meters)")
    print("-" * 80)
    print(f"  Mean: {metrics['l2_millimeters_if_input_is_meters']['mean']:.2f} mm")
    print(f"  Std:  {metrics['l2_millimeters_if_input_is_meters']['std']:.2f} mm")
    print(f"  Per-landmark:")
    for i, val in enumerate(metrics['l2_millimeters_if_input_is_meters']['per_landmark_mean']):
        print(f"    {LANDMARK_NAMES[i]:15s}: {val:.2f} mm")
    print()
    print("=" * 80)
    print()


def format_landmark_list(values):
    """Format a list of landmark values for compact display."""
    return "[" + ", ".join(f"{v:.6f}" for v in values) + "]"


def save_results(metrics):
    """Save evaluation metrics to JSON file."""
    out_dir = os.path.join(ROOT_DIR, "results", "evaluation")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "eval_pointnet2_l2.json")
    
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    
    print(f"Results saved to: {out_path}")
    print()


# ============================================================================
#  Command Line Interface
# ============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Evaluate PointNet++ landmark regression model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/evaluation/eval_pointnet2_l2.py --model models/pointnet2_regression_kfold_best.pth
  python scripts/evaluation/eval_pointnet2_l2.py --model models/pointnet2_regression_hparam_best.pth
        """
    )
    
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to the trained model weights (.pth file)'
    )
    
    return parser.parse_args()


# ============================================================================
#  Main Entry Point
# ============================================================================

if __name__ == "__main__":
    args = parse_args()
    evaluate(args.model)
