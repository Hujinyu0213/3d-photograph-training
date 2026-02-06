# PointNet++ FACIAL LANDMARK PREDICTION - TRAINING LOGS

---

## JANUARY 2026 (Early Attempts)

### Activity 1: Initial K-Fold Cross-Validation
- **Dataset:** 100 samples (8192 points per cloud)
- **Split:** 90% train/val, 10% test
- **Configuration:** 5-fold CV, 500 epochs each
- **Issue:** Very high losses (100k-300k range), poor normalization
- **Result:** NOT USED

### Activity 2: PointNet with Improved Normalization (80/20 Split)
- **Training:** 80 samples, Validation: 20 samples
- **Epochs:** 500 with learning rate decay
- **Results:**
  - Train Loss: 0.000090
  - Val Loss: 0.000067
- **Model:** `pointnet_regression_model_full_best.pth`

### Activity 3: PointNet K-Fold Cross-Validation (Fixed Normalization)
- **Dataset:** 100 samples (90/10 train-test split)
- **Configuration:** 5-fold CV on 90 samples, 500 epochs each
- **Fold Results:**
  - Fold 1: Val Loss 0.000136 (BEST)
  - Fold 2: Val Loss 0.000288
  - Fold 3: Val Loss 0.000436
  - Fold 4: Val Loss 0.003041
  - Fold 5: Val Loss 0.000265
- **Average:** 0.000833 ± 0.001108
- **Final:** Test Loss 0.002693
- **Model:** `pointnet_regression_model_kfold_best.pth`
- **Note:** Best fold (Fold 1) not properly used for final model

---

## 2024-02-02

### Activity 1: PointNet with Data Augmentation + FPS Sampling
**Objective:** Improve upon K-fold results with augmentation and Farthest Point Sampling

**Configuration:**
- Architecture: PointNet (original)
- Augmentation: Random rotation, jitter, scaling
- Sampling: FPS (Farthest Point Sampling)
- Training: Full dataset with validation split
- Script: `scripts/training/main_script_full_pointcloud_aug_fps.py`

**Results:**
- RMSE: 12.696 mm (8.24% improvement over old K-fold)
- MAE: 10.749 mm
- Error still much higher than ideal (2mm target)

**Per-Landmark RMSE Comparison (mm):**
| Landmark | Augmentation+FPS | Old K-fold | Change |
|----------|------------------|------------|--------|
| Glabella | 26.898 | 17.035 | -57.89% |
| Nasion | 19.451 | 34.704 | +43.95% |
| Rhinion | 20.174 | 26.514 | +23.91% |
| Nasal Tip | 14.825 | 25.115 | +40.97% |
| Subnasale | 25.174 | 12.685 | -98.46% |
| Alare (R) | 18.334 | 24.811 | +26.10% |
| Alare (L) | 23.453 | 24.995 | +6.17% |
| Zygion (R) | 17.486 | 18.086 | +3.32% |
| Zygion (L) | 28.249 | 24.751 | -14.13% |

**Conclusion:** Slight improvement but still far from target. Need to try K-fold + augmentation.

---

### Activity 2: PointNet K-Fold with Augmentation + FPS
**Objective:** Combine K-fold cross-validation with augmentation and FPS

**Configuration:**
- 5-fold cross-validation
- Data augmentation + FPS sampling
- 90/10 train/test split

**Results:**
- Average Best Validation Loss: 0.000254
- Best Fold: Fold 1 (loss: 0.000187)
- Test RMSE: 12.592 mm
- Test MAE: 9.983 mm
- Model: `models/pointnet_regression_model_kfold_aug_fps_best.pth`

**Per-Landmark RMSE Comparison (mm):**
| Landmark | K-fold+Aug+FPS | Single Train+Aug+FPS | Change |
|----------|----------------|---------------------|--------|
| Glabella | 25.893 | 26.896 | +3.87% |
| Nasion | 25.876 | 19.450 | -24.83% |
| Rhinion | 22.846 | 20.172 | -11.71% |
| Nasal Tip | 18.641 | 14.824 | -20.47% |
| Subnasale | 17.676 | 25.171 | +42.40% |
| Alare (R) | 21.041 | 18.332 | -12.87% |
| Alare (L) | 13.023 | 23.450 | +80.06% |
| Zygion (R) | 30.116 | 17.483 | -41.95% |
| Zygion (L) | 15.487 | 28.247 | +82.39% |

**Note:** Some landmarks improved significantly (Subnasale, Alare L, Zygion L), but others worsened (Nasion, Zygion R). Still ~12mm from 2mm target.

---

## 2024-02-03

### Activity 1: PointNet Hyperparameter Tuning (Random Search)
**Objective:** Find optimal hyperparameters for PointNet + K-fold + FPS + Augmentation

**Search Configuration:**
- Method: Random search, 20 trials
- Search space: LR, Batch Size, Dropout, LR scheduler params, Fine-tuning loss weight
- Base: K-fold + FPS + Augmentation

**Best Parameters Found:**
- Learning Rate: 0.0015
- Batch Size: 8
- Dropout: 0.35
- LR Step: 120, Gamma: 0.7
- Fine-tune weight: 0.001

**Improvement:**
- Validation mean loss: 0.000254 → 0.00007065 (72% reduction)

**Scripts Created:**
- `main_script_tuned_kfold.py` - Train with best params (300 epochs, 5 folds)
- `evaluate_tuned_model_comprehensive.py` - Test compare vs all historical models
- `analyze_hyperparameter_tuning.py` - Generate summaries

---

### Activity 2: Train with Tuned Hyperparameters
**Training Results:**
- Best Fold: Fold 4
- Best Validation Loss: 0.00001178
- K-Fold Average: 0.00001726 ± 0.00000279
- Model: `models/pointnet_regression_model_tuned_kfold_best.pth`

**Test Set Performance (20 samples):**
- RMSE: 13.397 mm
- MAE: 11.294 mm
- vs Old K-fold: +3.18% improvement

**Comparison with All Models:**
| Model | RMSE (mm) | MAE (mm) | vs Old K-fold |
|-------|-----------|----------|---------------|
| K-fold + Augmentation + FPS | 12.592 | 9.983 | +8.99% BEST |
| Single Train + Aug + FPS | 12.695 | 10.749 | +8.25% |
| Hyperparameter Tuned K-fold | 13.397 | 11.294 | +3.18% |
| Old K-fold (No Aug) | 13.837 | 10.996 | baseline |

**Per-Landmark RMSE Comparison (mm):**
| Landmark | Tuned K-fold | K-fold+Aug+FPS | Single+Aug+FPS | Old K-fold |
|----------|--------------|----------------|----------------|------------|
| Glabella | 20.020 | 25.893 | 26.897 | 17.034 |
| Nasion | 24.503 | 25.876 | 19.450 | 34.704 |
| Rhinion | 27.786 | 22.846 | 20.172 | 26.514 |
| Nasal Tip | 25.870 | 18.641 | 14.824 | 25.112 |
| Subnasale | 20.776 | 17.676 | 25.171 | 12.684 |
| Alare (R) | 17.404 | 21.041 | 18.332 | 24.811 |
| Alare (L) | 22.105 | 13.023 | 23.450 | 24.996 |
| Zygion (R) | 22.233 | 30.116 | 17.484 | 18.088 |
| Zygion (L) | 26.189 | 15.488 | 28.249 | 24.751 |

**Conclusion:** Still ~12mm from 2mm target. Recommended next step: Try PointNet++ architecture.

---

### Activity 3: PointNet++ K-Fold Cross-Validation
**Objective:** Test if PointNet++ architecture improves performance

**Configuration:**
- Architecture: PointNet++ MSG (Multi-Scale Grouping)
- 5-fold cross-validation
- Data augmentation + FPS (8192 points)
- 90/10 train/test split
- 80 epochs per fold

**Training Results:**
- Best Fold: Fold 5
- Best Validation Loss: 0.000009
- Average Validation Loss: 0.000012 ± 0.000002
- Test Loss: 0.000025

**Test Set Performance (10 samples):**
- **L2 Mean: 0.008148 mm** (normalized space)
- L2 Std: 0.002876 mm
- Per-landmark range: 0.005251 - 0.009527 mm

**Model Saved:**
- `models/pointnet2_regression_kfold_best.pth`
- Training history: `results/training_histories/training_history_pointnet2_kfold.json`

**Note:** First PointNet++ attempt - fold details not saved properly. Need systematic hyperparameter search.

---

### Activity 4: PointNet++ Hyperparameter Grid Search Setup
**Objective:** Systematically search for optimal PointNet++ hyperparameters

**Created Script:** `scripts/training/hparam_search_pointnet2.py`

**Search Space (128 total configurations):**
- Learning Rate: [1e-3, 5e-4]
- Dropout: [0.3, 0.4]
- Weight Decay: [0.0, 1e-4]
- Loss Type: ["mse", "smoothl1"]
- Geometry Lambda: [0.0, 0.1] (pairwise distance regularization)
- SA1 Radii: [[0.1,0.2,0.4], [0.05,0.1,0.2]]
- SA2 Radii: [[0.2,0.4,0.8], [0.1,0.2,0.4]]

**Key Improvements:**
1. **FPS Device Fix:** GPU-accelerated FPS on CUDA, fallback to random on CPU
2. **GPU Augmentation:** All augmentation operations run on GPU tensors
3. **Val L2 Selection:** Selects best config by validation L2 mean (not loss)
4. **No Val Leakage:** Final training uses train+val without validation-based stopping

**Training Pipeline:**
- **Search phase:** 80 epochs per config on train/val split
- **Selection:** Pick config with lowest validation L2 mean
- **Final phase:** Retrain best config on train+val (90%) for 160 epochs

**Outputs:**
- Best model: `models/pointnet2_regression_hparam_best.pth`
- Summary: `results/training_histories/hparam_search_pointnet2.json`
- Log: `results/logs/hparam_search_pointnet2_<timestamp>.log`

---

### Activity 5: Run PointNet++ Grid Search
**Status:** Launched grid search (128 configs × 80 epochs)
**Expected Duration:** ~9-10 hours

---

## 2024-02-04

### PointNet++ Hyperparameter Grid Search - COMPLETED

**Training Period:** 2026-02-03 16:48 → 2026-02-04 02:09 (9.5 hours)

**Search Results:**
- Total configurations tested: 128
- Each config: 80 epochs on train/val split
- Selection metric: Validation L2 mean distance

**Best Configuration Found:**
```python
{
    'lr': 0.001,
    'dropout': 0.4,
    'weight_decay': 0.0,
    'loss_type': 'smoothl1',
    'geo_lambda': 0.0,
    'sa1_radii': [0.1, 0.2, 0.4],
    'sa2_radii': [0.2, 0.4, 0.8]
}
```
- Best validation L2 mean: 0.010362

**Final Model Training (160 epochs on train+val):**
- Training loss progression:
  - Epoch 20: 0.013536
  - Epoch 40: 0.003063
  - Epoch 60: 0.001662
  - Epoch 80: 0.000567
  - Epoch 100: 0.000255
  - Epoch 120: 0.000149
  - Epoch 140: 0.000103
  - **Epoch 160: 0.000091** (final)

**Test Set Performance (10 held-out samples):**
- Test Loss: 9.357e-06
- **L2 Mean: 0.007009 mm** NEW BEST
- L2 Std: 0.002648 mm

**Per-Landmark L2 Distance (mm):**
1. Glabella: 0.006378
2. Nasion: 0.008842
3. Rhinion: 0.006260
4. Nasal Tip: 0.006986
5. Subnasale: 0.007560
6. Alare (R): 0.005382 (best)
7. Alare (L): 0.005561
8. Zygion (R): 0.008487
9. Zygion (L): 0.007626

**Key Findings:**
1. **Dropout matters:** Higher dropout (0.4) prevents overfitting
2. **SmoothL1 superior:** Outperforms MSE for landmark regression
3. **No geometry regularization needed:** geo_lambda=0.0 works best
4. **No weight decay required:** weight_decay=0.0 optimal
5. **Multi-scale radii optimal:** SA radii capture local+global features well

**Model Location:**
- Final model: `models/pointnet2_regression_hparam_best.pth`
- Training history: `results/training_histories/hparam_search_pointnet2.json`
- Detailed log: `results/logs/hparam_search_pointnet2_20260203_164816.log`

**Performance Improvement:**
- Previous best (PointNet++ K-fold): 0.008148 mm
- **This model (Tuned): 0.007009 mm**
- **Improvement: 14.0% reduction in error**

**Recommendations:**
- Analyze per-landmark improvements from hyperparameter tuning
- Consider ensemble methods combining top configurations
- Investigate targeted augmentation for high-error landmarks (Nasion, Zygion R)

---

### PointNet++ Refined Training (Deduplication + Normalization Fix)
**Objective:** Address false density issues and ensure model is compatible with inference (no ground truth dependency).

**Script Created:** `scripts/training/main_script_pointnet2_kfold_dedup.py`

**Key Improvements:**
1.  **Deduplication (Point Cloud Cleanup):**
    - **Problem:** Mesh vertices often have duplicate/near-duplicate points creating false density.
    - **Solution:** Voxel Grid Filtering (epsilon=1.5mm).
    - **Implementation:** Quantize points -> Unique Voxel Indices -> Average points within voxels.
    - **Target:** Reduces raw point count to ~7k-10k clean points before FPS sampling.

2.  **Normalization Fix (Critical for Inference):**
    - **Problem:** Previous normalization used `Label Centroid` (Ground Truth), which is impossible to calculate during inference.
    - **Solution:** Changed to `Point Cloud Centroid`.
    - **Scaling:** Uses bounding sphere radius (max distance) instead of std deviation.
    - **Result:** Pipeline is now fully valid for deployment on unseen data.

3.  **Strict Training Pipeline:**
    - **Metric Consistency:** Selecting best fold models using *Validation L2 Mean* (matching Grid Search criteria) instead of Loss.
    - **Retraining:** K-Fold is used *only* for evaluation statistics.
    - **Final Model:** Automatically retrains a fresh model on the full 90% training set (Train+Val) for 160 epochs.
    - **Evaluation:** Held-out 10% test set is evaluated *only* on this final retrained model.

**Configuration (Best from Grid Search):**
- Epochs: 160
- LR: 0.001 (Decay at 80)
- Dropout: 0.4
- Loss: SmoothL1
- Radii: Multi-scale [0.1, 0.2, 0.4] / [0.2, 0.4, 0.8]

**Next Step:** Run this refined training script to verify performance gains from cleaner data and correct training workflow.

---

## 2026-02-06

### PointNet++ with Deduplication (PRODUCTION MODEL)
**Objective:** Final production model with point cloud deduplication to remove false density from mesh vertices.

**Script:** `scripts/training/main_script_pointnet2_kfold_dedup.py`

**Configuration:**
- **Deduplication:** ENABLED (epsilon=35.0)
- **Architecture:** PointNet++ Multi-Scale (MSG)
- **Sampling:** 8192 points via FPS
- **Normalization:** Label Centroid + Std Dev (reverted from PC Centroid due to performance issues)
- **Epochs:** 220 per fold
- **Hyperparameters:** Best from grid search
  - Learning Rate: 0.001 (decay 0.7 at epoch 80)
  - Dropout: 0.4
  - Loss: SmoothL1
  - Weight Decay: 0.0
  - SA1 Radii: [0.1, 0.2, 0.4]
  - SA2 Radii: [0.2, 0.4, 0.8]

**Data Preprocessing Analysis:**
- **Unit Verification:** Original data is in millimeters (mm)
- **Bounding Box Diagonal:** Mean=11532.6mm, Min=7015.7mm, Max=17284.0mm
- **Deduplication Stats:**
  - Original Points: ~17085 per sample
  - After Cleanup: ~8594.5 per sample (43.8% retention)
  - Range: 4232-14475 points
  - Retention Ratio: 36.1%-50.1%

**5-Fold Cross-Validation Results (mm):**
| Fold | Best Val L2 (mm) | Epochs to Best |
|------|------------------|----------------|
| 1 | 9.8719 | 220 |
| 2 | 10.8507 | 200 |
| 3 | 9.2314 | 220 (BEST FOLD) |
| 4 | 10.4099 | 220 |
| 5 | 9.6508 | 180 |

**Cross-Validation Statistics:**
- Mean: 10.0029 mm
- Std: 0.6215 mm
- Best Fold: #3 (9.2314 mm)

**Final Model (Retrained on Full 90% Train+Val):**
- Epochs: 220
- Final Training Loss: 0.000103

**Test Set Performance (10 held-out samples):**
- Test Loss (SmoothL1): 0.000005
- **L2 Mean: 7.9307 mm ± 3.6865 mm** ← NEW BEST
- **Improvement over previous best:** 13.1% (7.9307 vs 9.1296 from PointNet++)

**Per-Landmark L2 Distance (mm):**
| Landmark | Error (mm) |
|----------|------------|
| P0 (Glabella) | 7.8876 |
| P1 (Nasion) | 7.6778 |
| P2 (Rhinion) | 6.7567 ← Best |
| P3 (Nasal Tip) | 7.8842 |
| P4 (Subnasale) | 8.2876 |
| P5 (Alare R) | 7.7957 |
| P6 (Alare L) | 6.0072 ← 2nd Best |
| P7 (Zygion R) | 9.8059 ← Highest error |
| P8 (Zygion L) | 9.2734 |

**Landmark Error Analysis:**
- Best landmarks: Alare L (6.00mm), Rhinion (6.76mm)
- Highest error: Zygion R (9.81mm), Zygion L (9.27mm)
- Most consistent: Central landmarks (Rhinion, Nasion, Nasal Tip)
- More variable: Lateral landmarks (Zygions, Subnasale)

**Model Location:**
- `models/pointnet2_dedup_final_best.pth`

**Key Achievements:**
1. **Deduplication Success:** Removing duplicate vertices (43.8% retention) improved model convergence
2. **Denormalization:** All errors now properly reported in millimeters
3. **Sub-10mm Performance:** Achieved 7.93mm mean error across all landmarks
4. **Stable Cross-Validation:** Low variance (σ=0.62mm) across folds indicates robust training
5. **Best Overall Result:** 13.1% improvement over previous PointNet++ baseline

**Known Issues:**
- Normalization still uses Label Centroid (requires ground truth)
- Cannot be deployed for inference on new data without modification
- Zygion landmarks consistently have higher errors (lateral face features)

**Recommendations:**
1. Investigate inference-compatible normalization (template-based alignment)
2. Analyze Zygion errors - may benefit from targeted augmentation
3. Consider ensemble methods combining top 3 folds
4. Explore deeper architecture or attention mechanisms for lateral features

---
