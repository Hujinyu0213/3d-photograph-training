"""
PointNet 回归训练脚本（K折交叉验证版本）
使用完整点云作为输入，预测9个地标点坐标
采用K折交叉验证提供更可靠的模型评估
"""
import os
import sys
import io
# 设置UTF-8编码以支持中文输出
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# 添加当前目录到 Python 路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split
from sklearn.model_selection import KFold
from tqdm import tqdm
import json
from pointnet_utils import PointNetEncoder, feature_transform_reguliarzer

# =========================================================
# 配置
# =========================================================
EXPORT_ROOT = os.path.join(BASE_DIR, 'data', 'pointcloud')
LABELS_FILE = os.path.join(BASE_DIR, 'labels.csv')
PROJECTS_LIST_FILE = os.path.join(BASE_DIR, 'valid_projects.txt')

NUM_TARGET_POINTS = 9
OUTPUT_DIM = NUM_TARGET_POINTS * 3  # 27维

# GPU 配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🖥️  使用设备: {device}")
if torch.cuda.is_available():
    print(f"   GPU名称: {torch.cuda.get_device_name(0)}")
    print(f"   GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

# =========================================================
# 训练参数配置
# =========================================================
BATCH_SIZE = 8               # 点云较大（平均19345点），使用较小批次以避免内存不足
NUM_EPOCHS = 500
LEARNING_RATE = 0.001
LR_DECAY_STEP = 150
LR_DECAY_GAMMA = 0.5
DROPOUT_RATE = 0.3
FEATURE_TRANSFORM_WEIGHT = 0.001

# K折交叉验证配置
K_FOLDS = 5                  # 5折交叉验证
RANDOM_SEED = 42            # 随机种子，确保可重复

# 测试集划分配置
TEST_RATIO = 0.1            # 10%作为测试集（独立评估用）

# 模型保存配置
MODEL_NAME_PREFIX = 'pointnet_regression_model_kfold'
BEST_MODEL_NAME = 'pointnet_regression_model_kfold_best.pth'

# =========================================================
# 模型定义
# =========================================================
class PointNetRegressor(nn.Module):
    def __init__(self, output_dim=27, dropout_rate=0.3):
        super(PointNetRegressor, self).__init__()
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
# 数据加载（完整点云版本）
# =========================================================
def load_data():
    """
    加载完整点云和标签
    返回: X (torch.Tensor), Y (torch.Tensor)
    """
    print("--- 正在加载完整点云和标签 ---")
    
    # 读取项目列表
    if not os.path.exists(PROJECTS_LIST_FILE):
        raise FileNotFoundError(f"❌ 项目列表文件不存在: {PROJECTS_LIST_FILE}\n请先运行 create_labels_from_npy.py")
    
    with open(PROJECTS_LIST_FILE, 'r', encoding='utf-8') as f:
        project_names = [line.strip() for line in f if line.strip()]
    
    # 读取标签文件
    if not os.path.exists(LABELS_FILE):
        raise FileNotFoundError(f"❌ 标签文件不存在: {LABELS_FILE}\n请先运行 create_labels_from_npy.py")
    
    labels_df = pd.read_csv(LABELS_FILE, header=None)
    all_labels_np = labels_df.values.astype(np.float32)
    
    if len(project_names) != len(all_labels_np):
        print(f"⚠️  警告: 项目数量 ({len(project_names)}) 与标签数量 ({len(all_labels_np)}) 不匹配")
        min_len = min(len(project_names), len(all_labels_np))
        project_names = project_names[:min_len]
        all_labels_np = all_labels_np[:min_len]
    
    valid_features = []
    valid_labels = []
    point_counts = []
    
    print(f"加载 {len(project_names)} 个样本的点云...")
    
    for i, project_name in enumerate(tqdm(project_names, desc="加载点云")):
        project_dir = os.path.join(EXPORT_ROOT, project_name)
        pointcloud_file = os.path.join(project_dir, "pointcloud_full.npy")
        
        if not os.path.exists(pointcloud_file):
            print(f"⚠️  跳过 {project_name}: 未找到 pointcloud_full.npy")
            continue
        
        try:
            # 加载完整点云
            pointcloud = np.load(pointcloud_file).astype(np.float32)  # shape: (N, 3)
            
            if len(pointcloud) == 0:
                print(f"⚠️  跳过 {project_name}: 点云为空")
                continue
            
            # 先获取标签
            current_label = all_labels_np[i].reshape(NUM_TARGET_POINTS, 3)
            
            # 使用地标点质心作为参考点（而不是点云质心）
            # 这样可以确保点云和地标点使用相同的参考坐标系
            label_centroid = np.mean(current_label, axis=0)
            
            # 点云和标签都相对于地标点质心
            centered_pointcloud = pointcloud - label_centroid
            centered_label = current_label - label_centroid
            
            # 归一化：使用点云的标准差进行缩放
            # 这样可以统一不同样本的尺度
            scale = np.std(centered_pointcloud)
            if scale > 1e-6:  # 避免除零
                centered_pointcloud = centered_pointcloud / scale
                centered_label = centered_label / scale
            
            # 转置为 (3, N) 格式，适配 PointNet
            centered_pointcloud_T = centered_pointcloud.T  # (3, N)
            
            valid_features.append(centered_pointcloud_T)
            valid_labels.append(centered_label.flatten())
            point_counts.append(len(pointcloud))
            
        except Exception as e:
            print(f"❌ 处理 {project_name} 时出错: {e}")
            continue
    
    if not valid_features:
        raise RuntimeError("❌ 未能加载任何有效数据！")
    
    print(f"\n✅ 成功加载 {len(valid_features)} 个样本")
    print(f"   点云数量统计:")
    print(f"     最小: {min(point_counts)} 个点")
    print(f"     最大: {max(point_counts)} 个点")
    print(f"     平均: {np.mean(point_counts):.0f} 个点")
    
    # 统一采样到固定点数
    MAX_POINTS = 8192
    print(f"\n统一采样到 {MAX_POINTS} 个点...")
    
    processed_features = []
    for feat in valid_features:
        num_points = feat.shape[1]
        if num_points >= MAX_POINTS:
            # 随机采样
            indices = np.random.choice(num_points, MAX_POINTS, replace=False)
            sampled_feat = feat[:, indices]
        else:
            # 重复采样（有放回）
            indices = np.random.choice(num_points, MAX_POINTS, replace=True)
            sampled_feat = feat[:, indices]
        processed_features.append(sampled_feat)
    
    # 转换为numpy数组并转置为 (N, 3, MAX_POINTS)
    X_np = np.array(processed_features, dtype=np.float32)  # (N, 3, MAX_POINTS)
    Y_np = np.array(valid_labels, dtype=np.float32)  # (N, 27)
    
    print(f"   最终数据形状: X={X_np.shape}, Y={Y_np.shape}")
    
    return torch.from_numpy(X_np), torch.from_numpy(Y_np)

# =========================================================
# 单折训练函数
# =========================================================
def train_fold(X, Y, train_indices, val_indices, fold_num):
    """
    训练单个折
    
    参数:
        X: 特征数据
        Y: 标签数据
        train_indices: 训练集索引
        val_indices: 验证集索引
        fold_num: 折数（1-K）
    
    返回:
        best_val_loss: 最佳验证损失
        training_history: 训练历史
    """
    # 创建数据集
    train_dataset = TensorDataset(X[train_indices], Y[train_indices])
    val_dataset = TensorDataset(X[val_indices], Y[val_indices])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
    
    print(f"\n{'='*60}")
    print(f"📊 折 {fold_num}/{K_FOLDS}")
    print(f"{'='*60}")
    print(f"训练集: {len(train_indices)} 个样本")
    print(f"验证集: {len(val_indices)} 个样本")
    
    # 创建模型
    model = PointNetRegressor(output_dim=OUTPUT_DIM, dropout_rate=DROPOUT_RATE).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=LR_DECAY_STEP, gamma=LR_DECAY_GAMMA
    )
    
    best_val_loss = float('inf')
    training_history = {'train_loss': [], 'val_loss': [], 'epoch': []}
    
    for epoch in range(NUM_EPOCHS):
        # 训练阶段
        model.train()
        total_train_loss = 0
        train_count = 0
        
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            pred, trans_feat = model(data)
            loss = criterion(pred, target)
            if trans_feat is not None:
                loss += feature_transform_reguliarzer(trans_feat) * FEATURE_TRANSFORM_WEIGHT
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
            train_count += 1
        
        # 验证阶段
        model.eval()
        total_val_loss = 0
        val_count = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                pred, trans_feat = model(data)
                loss = criterion(pred, target)
                if trans_feat is not None:
                    loss += feature_transform_reguliarzer(trans_feat) * FEATURE_TRANSFORM_WEIGHT
                total_val_loss += loss.item()
                val_count += 1
        
        avg_train_loss = total_train_loss / train_count
        avg_val_loss = total_val_loss / val_count if val_count > 0 else 0
        
        training_history['train_loss'].append(avg_train_loss)
        training_history['val_loss'].append(avg_val_loss)
        training_history['epoch'].append(epoch + 1)
        
        # 更新学习率
        scheduler.step()
        
        # 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            fold_model_path = os.path.join(BASE_DIR, f'{MODEL_NAME_PREFIX}_fold{fold_num}_best.pth')
            torch.save(model.state_dict(), fold_model_path)
        
        # 打印进度
        if (epoch+1) % 50 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:4d}/{NUM_EPOCHS} | "
                  f"Train Loss: {avg_train_loss:.6f} | "
                  f"Val Loss: {avg_val_loss:.6f} | "
                  f"Best Val: {best_val_loss:.6f}")
    
    # 保存最终模型
    fold_final_path = os.path.join(BASE_DIR, f'{MODEL_NAME_PREFIX}_fold{fold_num}_final.pth')
    torch.save(model.state_dict(), fold_final_path)
    
    print(f"\n✅ 折 {fold_num} 训练完成")
    print(f"   最佳验证损失: {best_val_loss:.6f}")
    print(f"   最佳模型: {fold_model_path}")
    print(f"   最终模型: {fold_final_path}")
    
    return best_val_loss, training_history

# =========================================================
# K折交叉验证主函数
# =========================================================
def train_kfold():
    """
    K折交叉验证训练（包含独立测试集）
    """
    # 加载数据
    X, Y = load_data()
    if X is None:
        return
    
    print(f"\n{'='*60}")
    print(f"🔄 K折交叉验证训练 (K={K_FOLDS}) + 独立测试集")
    print(f"{'='*60}")
    print(f"总样本数: {len(X)}")
    
    # =========================================================
    # 第一步：划分测试集（10%）
    # =========================================================
    from torch.utils.data import random_split
    
    dataset = TensorDataset(X, Y)
    test_size = int(TEST_RATIO * len(dataset))
    train_val_size = len(dataset) - test_size
    
    # 划分测试集和训练+验证集
    train_val_dataset, test_dataset = random_split(
        dataset, [train_val_size, test_size], 
        generator=torch.Generator().manual_seed(RANDOM_SEED)
    )
    
    # 提取训练+验证集的数据
    train_val_indices = train_val_dataset.indices
    test_indices = test_dataset.indices
    
    train_val_X = X[train_val_indices]
    train_val_Y = Y[train_val_indices]
    test_X = X[test_indices]
    test_Y = Y[test_indices]
    
    print(f"\n📊 数据划分:")
    print(f"   测试集: {len(test_indices)} 个样本 ({TEST_RATIO*100:.0f}%)")
    print(f"   训练+验证集: {len(train_val_indices)} 个样本 ({(1-TEST_RATIO)*100:.0f}%)")
    print(f"   每折验证集大小: 约 {len(train_val_indices) // K_FOLDS} 个样本")
    print(f"   每折训练集大小: 约 {len(train_val_indices) - len(train_val_indices) // K_FOLDS} 个样本")
    
    # =========================================================
    # 第二步：在训练+验证集上做K折交叉验证
    # =========================================================
    # 创建K折划分（在90%的数据上）
    kfold = KFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    
    # 存储所有折的结果
    fold_results = []
    all_training_histories = []
    
    # 训练每一折（在训练+验证集上）
    # KFold需要numpy数组，使用索引数组
    train_val_indices_array = np.array(train_val_indices)
    
    for fold_num, (fold_train_idx, fold_val_idx) in enumerate(kfold.split(train_val_indices_array), 1):
        # 将折内的索引映射回原始索引
        train_indices = train_val_indices_array[fold_train_idx].tolist()
        val_indices = train_val_indices_array[fold_val_idx].tolist()
        
        best_val_loss, training_history = train_fold(
            X, Y, train_indices, val_indices, fold_num
        )
        fold_results.append({
            'fold': fold_num,
            'best_val_loss': best_val_loss,
            'train_size': len(train_indices),
            'val_size': len(val_indices)
        })
        all_training_histories.append(training_history)
    
    # 计算统计信息
    val_losses = [r['best_val_loss'] for r in fold_results]
    mean_val_loss = np.mean(val_losses)
    std_val_loss = np.std(val_losses)
    min_val_loss = np.min(val_losses)
    best_fold = np.argmin(val_losses) + 1
    
    # 打印总结
    print(f"\n{'='*60}")
    print(f"📊 K折交叉验证结果总结")
    print(f"{'='*60}")
    for result in fold_results:
        print(f"折 {result['fold']}: 最佳验证损失 = {result['best_val_loss']:.6f}")
    print(f"\n统计信息:")
    print(f"  平均验证损失: {mean_val_loss:.6f} ± {std_val_loss:.6f}")
    print(f"  最小验证损失: {min_val_loss:.6f} (折 {best_fold})")
    print(f"  标准差: {std_val_loss:.6f}")
    
    # 选择最佳折的模型作为最终模型
    best_model_source = os.path.join(BASE_DIR, f'{MODEL_NAME_PREFIX}_fold{best_fold}_best.pth')
    best_model_dest = os.path.join(BASE_DIR, BEST_MODEL_NAME)
    
    import shutil
    shutil.copy(best_model_source, best_model_dest)
    
    print(f"\n✅ 最佳模型已复制:")
    print(f"   来源: 折 {best_fold} 的最佳模型")
    print(f"   目标: {best_model_dest}")
    
    # 保存所有折的训练历史
    kfold_history = {
        'k_folds': K_FOLDS,
        'fold_results': fold_results,
        'statistics': {
            'mean_val_loss': float(mean_val_loss),
            'std_val_loss': float(std_val_loss),
            'min_val_loss': float(min_val_loss),
            'best_fold': int(best_fold)
        },
        'training_histories': all_training_histories
    }
    
    history_path = os.path.join(BASE_DIR, 'training_history_kfold.json')
    with open(history_path, 'w') as f:
        json.dump(kfold_history, f, indent=2)
    
    print(f"   训练历史: {history_path}")
    
    # =========================================================
    # 第三步：用所有训练+验证数据重新训练最终模型（带验证集和早停）
    # =========================================================
    print(f"\n{'='*60}")
    print(f"🔄 用所有训练+验证数据重新训练最终模型（带验证集和早停）")
    print(f"{'='*60}")
    print(f"总数据: {len(train_val_indices)} 个样本（所有90%的数据）")
    print(f"目的: 充分利用所有数据，同时避免过拟合")
    
    # 从90%的数据中再分出10%作为验证集（用于早停）
    # 这样最终训练集是80%，验证集是10%，测试集是10%
    final_val_ratio = 0.1  # 从90%中分出10%作为验证集
    final_train_val_dataset = TensorDataset(train_val_X, train_val_Y)
    final_train_size = int((1 - final_val_ratio) * len(final_train_val_dataset))
    final_val_size = len(final_train_val_dataset) - final_train_size
    
    # 划分最终训练集和验证集（使用不同的随机种子，确保与测试集划分不同）
    final_train_dataset, final_val_dataset = random_split(
        final_train_val_dataset, [final_train_size, final_val_size],
        generator=torch.Generator().manual_seed(RANDOM_SEED + 100)  # 使用不同的随机种子
    )
    
    final_train_loader = DataLoader(
        final_train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True
    )
    final_val_loader = DataLoader(
        final_val_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False
    )
    
    print(f"   最终训练集: {len(final_train_dataset)} 个样本 ({final_train_size/len(train_val_indices)*100:.1f}%)")
    print(f"   最终验证集: {len(final_val_dataset)} 个样本 ({final_val_size/len(train_val_indices)*100:.1f}%)")
    print(f"   测试集: {len(test_indices)} 个样本 (10%)")
    
    # 创建新的模型（从头训练）
    final_model = PointNetRegressor(output_dim=OUTPUT_DIM, dropout_rate=DROPOUT_RATE).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(final_model.parameters(), lr=LEARNING_RATE)
    
    # 学习率调度器（使用与K折相同的配置）
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=LR_DECAY_STEP, gamma=LR_DECAY_GAMMA
    )
    
    # 早停配置
    PATIENCE = 50  # 如果验证损失在50轮内没有改善，则早停
    best_val_loss = float('inf')
    patience_counter = 0
    best_epoch = 0
    
    print(f"\n开始训练最终模型（最多{NUM_EPOCHS}轮，早停耐心={PATIENCE}）...")
    final_model.train()
    final_training_history = {'train_loss': [], 'val_loss': [], 'epoch': []}
    
    for epoch in range(NUM_EPOCHS):
        # 训练阶段
        final_model.train()
        total_train_loss = 0
        train_count = 0
        
        for data, target in final_train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            pred, trans_feat = final_model(data)
            loss = criterion(pred, target)
            if trans_feat is not None:
                loss += feature_transform_reguliarzer(trans_feat) * FEATURE_TRANSFORM_WEIGHT
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
            train_count += 1
        
        avg_train_loss = total_train_loss / train_count
        
        # 验证阶段
        final_model.eval()
        total_val_loss = 0
        val_count = 0
        
        with torch.no_grad():
            for data, target in final_val_loader:
                data, target = data.to(device), target.to(device)
                pred, trans_feat = final_model(data)
                loss = criterion(pred, target)
                if trans_feat is not None:
                    loss += feature_transform_reguliarzer(trans_feat) * FEATURE_TRANSFORM_WEIGHT
                total_val_loss += loss.item()
                val_count += 1
        
        avg_val_loss = total_val_loss / val_count if val_count > 0 else float('inf')
        
        final_training_history['train_loss'].append(avg_train_loss)
        final_training_history['val_loss'].append(avg_val_loss)
        final_training_history['epoch'].append(epoch + 1)
        
        scheduler.step()
        
        # 检查是否是最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch + 1
            patience_counter = 0
            # 保存最佳模型
            best_model_state = final_model.state_dict().copy()
        else:
            patience_counter += 1
        
        # 打印进度
        if (epoch+1) % 50 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:4d}/{NUM_EPOCHS} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} | Best Val: {best_val_loss:.6f} (Epoch {best_epoch})")
        
        # 早停检查
        if patience_counter >= PATIENCE:
            print(f"\n⚠️  早停触发！验证损失在{PATIENCE}轮内没有改善")
            print(f"   最佳验证损失: {best_val_loss:.6f} (Epoch {best_epoch})")
            break
    
    # 加载最佳模型
    final_model.load_state_dict(best_model_state)
    
    # 保存最终模型（最佳模型）
    final_model_path = os.path.join(BASE_DIR, BEST_MODEL_NAME)
    torch.save(final_model.state_dict(), final_model_path)
    
    print(f"\n✅ 最终模型训练完成！")
    print(f"   最佳验证损失: {best_val_loss:.6f} (Epoch {best_epoch})")
    print(f"   最终训练损失: {final_training_history['train_loss'][best_epoch-1]:.6f}")
    print(f"   模型已保存: {final_model_path}")
    print(f"   使用数据: {len(final_train_dataset)} 个样本训练，{len(final_val_dataset)} 个样本验证")
    
    # =========================================================
    # 第四步：在测试集上评估最终模型
    # =========================================================
    print(f"\n{'='*60}")
    print(f"🧪 在测试集上评估最终模型")
    print(f"{'='*60}")
    
    # 使用最终训练的模型
    final_model.eval()
    
    # 在测试集上评估
    test_dataset_final = TensorDataset(test_X, test_Y)
    test_loader = DataLoader(test_dataset_final, batch_size=BATCH_SIZE, shuffle=False)
    
    criterion = nn.MSELoss()
    total_test_loss = 0
    test_count = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            pred, trans_feat = final_model(data)
            loss = criterion(pred, target)
            if trans_feat is not None:
                loss += feature_transform_reguliarzer(trans_feat) * FEATURE_TRANSFORM_WEIGHT
            total_test_loss += loss.item()
            test_count += 1
    
    avg_test_loss = total_test_loss / test_count if test_count > 0 else 0
    
    print(f"测试集大小: {len(test_indices)} 个样本")
    print(f"测试集损失: {avg_test_loss:.6f}")
    
    # 更新训练历史，添加最终模型和测试集结果
    kfold_history['final_model'] = {
        'training_history': final_training_history,
        'train_size': len(final_train_dataset),
        'val_size': len(final_val_dataset),
        'best_epoch': best_epoch,
        'best_val_loss': float(best_val_loss),
        'final_train_loss': float(final_training_history['train_loss'][best_epoch-1]),
        'early_stopped': patience_counter >= PATIENCE
    }
    kfold_history['test_loss'] = float(avg_test_loss)
    kfold_history['test_size'] = len(test_indices)
    
    # 重新保存训练历史（包含最终模型和测试集结果）
    with open(history_path, 'w') as f:
        json.dump(kfold_history, f, indent=2)
    
    print(f"\n🎉 K折交叉验证 + 最终模型训练完成！")
    print(f"   K折交叉验证: 训练了 {K_FOLDS} 个模型用于选择最佳配置")
    print(f"   平均验证损失: {mean_val_loss:.6f} ± {std_val_loss:.6f}")
    print(f"   最佳折: 折 {best_fold}，验证损失: {min_val_loss:.6f}")
    print(f"   最终模型: 用 {len(final_train_dataset)} 个样本训练，{len(final_val_dataset)} 个样本验证")
    print(f"   最佳验证损失: {best_val_loss:.6f} (Epoch {best_epoch})")
    if patience_counter >= PATIENCE:
        print(f"   ⚠️  早停触发（耐心={PATIENCE}）")
    print(f"   ⭐ 测试集损失: {avg_test_loss:.6f} (独立评估，无偏)")
    print(f"\n📁 最终模型文件: {final_model_path}")

if __name__ == "__main__":
    train_kfold()
