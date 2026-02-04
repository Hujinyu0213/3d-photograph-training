# 3D Photograph Training - PointNet Facial Landmark Prediction

PointNet-based facial landmark prediction from 3D point clouds.

## Project Structure

```
├── data/                       # Point cloud data
│   └── pointcloud/            # Raw point cloud files (.npy)
├── scripts/                    # All Python scripts
│   ├── training/              # Training scripts
│   │   ├── main_script_full_pointcloud.py
│   │   └── main_script_kfold.py
│   ├── evaluation/            # Model evaluation scripts
│   │   ├── evaluate_*.py
│   │   └── compare_*.py
│   ├── analysis/              # Analysis and checking scripts
│   │   ├── analyze_*.py
│   │   └── check_*.py
│   └── utils/                 # Utility functions
│       ├── pointnet_utils.py
│       └── create_labels_from_npy.py
├── models/                     # Trained models (.pth)
├── results/                    # Training results and logs
│   ├── *.json                 # Training histories
│   ├── labels.csv             # Ground truth labels
│   └── valid_projects.txt     # Valid project list
├── docs/                       # Documentation
│   ├── reports/               # Analysis reports
│   └── *.md                   # Training guides (Chinese)
└── README.md

```

## Quick Start

### 1. Training
- **Full training**: `python scripts/training/main_script_full_pointcloud.py`
- **K-fold CV**: `python scripts/training/main_script_kfold.py`

### 2. Evaluation
- Evaluate models in `scripts/evaluation/`

### 3. Requirements
- PyTorch
- NumPy
- pandas
- scikit-learn
- tqdm

## Features

- PointNet-based regression for 9 facial landmarks (27 coordinates)
- K-fold cross-validation (K=5)
- Automatic data normalization and alignment
- Point cloud sampling to 8192 points
- Early stopping and learning rate scheduling
- Comprehensive evaluation metrics

## Landmark Mapping

The model predicts 9 facial anatomical landmarks (3D coordinates each):

| Index | Landmark Name | Chinese | Description |
|-------|---------------|---------|-------------|
| 1 | Glabella | 眉间 | Between eyebrows |
| 2 | Nasion | 鼻根点 | Bridge of nose |
| 3 | Rhinion | 鼻背点 | Nasal dorsum |
| 4 | Nasal Tip | 鼻尖 | Tip of nose |
| 5 | Subnasale | 鼻下点 | Below nose |
| 6 | Alare (R) | 右鼻翼点 | Right nose wing |
| 7 | Alare (L) | 左鼻翼点 | Left nose wing |
| 8 | Zygion (R) | 右颧骨点 | Right cheekbone |
| 9 | Zygion (L) | 左颧骨点 | Left cheekbone |

**Output format**: 27 values = 9 landmarks × 3 coordinates (x, y, z)
