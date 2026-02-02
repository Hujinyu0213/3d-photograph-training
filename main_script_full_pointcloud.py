"""
PointNet 回归训练脚本（完整点云版本）
使用完整点云作为输入，预测9个地标点坐标
"""
import os
import sys

# 添加当前目录到 Python 路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split
from tqdm import tqdm
from pointnet_utils import PointNetEncoder, feature_transform_reguliarzer

# =========================================================
# 配置
# =========================================================
# 选项1: 从项目目录读取（推荐）
EXPORT_ROOT = os.path.join(BASE_DIR, 'data', 'pointcloud')

# 选项2: 从网络路径读取（如果数据在网络路径，取消下面的注释）
# EXPORT_ROOT = r"\\uz\data\Admin\mka\results\hou-and-hu\vs\pointcloud"

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

# 训练/验证集划分
TRAIN_RATIO = 0.8

# 模型保存配置
MODEL_NAME = 'pointnet_regression_model_full.pth'
BEST_MODEL_NAME = 'pointnet_regression_model_full_best.pth'

# =========================================================
# 模型定义
# =========================================================
class PointNetRegressor(nn.Module):
    def __init__(self, output_dim=27, dropout_rate=0.3):
        super(PointNetRegressor, self).__init__()
        # PointNetEncoder 可以处理任意数量的点
        self.feat = PointNetEncoder(global_feat=True, feature_transform=True, channel=3)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, output_dim)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x shape: (batch_size, 3, num_points)
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
            # PointNet 期望输入格式: (batch_size, 3, num_points)
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
    
    # 注意：由于每个样本的点数不同，我们需要统一处理
    # 方法1: 使用最大点数，不足的用0填充（不推荐，浪费内存）
    # 方法2: 使用固定点数，随机采样或截取（推荐）
    # 方法3: 使用动态批处理（复杂）
    
    # 这里使用方法2：统一采样到固定点数
    # 根据点云统计：平均19345点，最小11564点，最大29182点
    # 使用8192点可以保留更多细节，同时保持计算效率
    MAX_POINTS = 8192  # 优化：从2048增加到8192以保留更多点云细节
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
# 训练函数
# =========================================================
def train():
    X, Y = load_data()
    if X is None:
        return
    
    # 划分训练集和验证集
    dataset = TensorDataset(X, Y)
    train_size = int(TRAIN_RATIO * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
    
    print(f"\n📊 数据集划分: 训练集 {train_size} 个样本, 验证集 {val_size} 个样本")
    
    model = PointNetRegressor(output_dim=OUTPUT_DIM, dropout_rate=DROPOUT_RATE).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=LR_DECAY_STEP, gamma=LR_DECAY_GAMMA
    )
    
    print(f"\n--- 开始训练 {NUM_EPOCHS} 轮 ---")
    print(f"📋 训练配置:")
    print(f"   批次大小: {BATCH_SIZE}")
    print(f"   初始学习率: {LEARNING_RATE}")
    print(f"   学习率衰减: 每 {LR_DECAY_STEP} 轮 × {LR_DECAY_GAMMA}")
    print(f"   Dropout: {DROPOUT_RATE}")
    print(f"   特征变换正则化权重: {FEATURE_TRANSFORM_WEIGHT}")
    
    model.train()
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
        current_lr = optimizer.param_groups[0]['lr']
        
        # 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_path = os.path.join(BASE_DIR, BEST_MODEL_NAME)
            torch.save(model.state_dict(), best_model_path)
        
        # 打印进度
        if (epoch+1) % 25 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:4d}/{NUM_EPOCHS} | "
                  f"Train Loss: {avg_train_loss:.6f} | "
                  f"Val Loss: {avg_val_loss:.6f} | "
                  f"LR: {current_lr:.6f} | "
                  f"Best Val: {best_val_loss:.6f}")
    
    # 保存最终模型
    model_path = os.path.join(BASE_DIR, MODEL_NAME)
    torch.save(model.state_dict(), model_path)
    
    print(f"\n🎉 训练完成！")
    print(f"   最终模型: {model_path}")
    print(f"   最佳模型: {os.path.join(BASE_DIR, BEST_MODEL_NAME)} (验证损失: {best_val_loss:.6f})")
    
    # 保存训练历史
    import json
    history_path = os.path.join(BASE_DIR, 'training_history_full.json')
    with open(history_path, 'w') as f:
        json.dump(training_history, f, indent=2)
    print(f"   训练历史: {history_path}")

if __name__ == "__main__":
    train()
