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
