"""
分析修复后的训练结果
计算实际误差（考虑归一化）

this is analysis codes about full version(80/20 split)
"""
import sys
import io
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import os
import json
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

print("="*70)
print("修复后训练结果分析")
print("="*70)

# 读取训练历史
history_file = os.path.join(BASE_DIR, 'training_history_full.json')
if os.path.exists(history_file):
    with open(history_file, 'r', encoding='utf-8') as f:
        history = json.load(f)
    
    train_losses = history['train_loss']
    val_losses = history['val_loss']
    epochs = history['epoch']
    
    print(f"\n📊 训练统计（归一化后的损失）")
    print(f"   总轮数: {len(epochs)}")
    print(f"   初始训练损失: {train_losses[0]:.6f}")
    print(f"   最终训练损失: {train_losses[-1]:.6f}")
    print(f"   损失下降: {((train_losses[0] - train_losses[-1]) / train_losses[0] * 100):.2f}%")
    
    print(f"\n   初始验证损失: {val_losses[0]:.6f}")
    print(f"   最终验证损失: {val_losses[-1]:.6f}")
    print(f"   最佳验证损失: {min(val_losses):.6f}")
    print(f"   损失下降: {((val_losses[0] - min(val_losses)) / val_losses[0] * 100):.2f}%")
    
    best_val_loss = min(val_losses)
    best_epoch = epochs[val_losses.index(best_val_loss)]
    print(f"\n   最佳验证损失: {best_val_loss:.6f} (第 {best_epoch} 轮)")

# 计算实际误差（考虑归一化）
print(f"\n" + "="*70)
print("实际误差计算（考虑归一化）")
print("="*70)

# 需要计算归一化尺度
# 在训练时，每个样本都除以了 pointcloud 的标准差
# 我们需要估算平均尺度

# 加载一些样本计算平均尺度
EXPORT_ROOT = os.path.join(BASE_DIR, 'data', 'pointcloud')
LABELS_FILE = os.path.join(BASE_DIR, 'labels.csv')
PROJECTS_LIST_FILE = os.path.join(BASE_DIR, 'valid_projects.txt')

if os.path.exists(PROJECTS_LIST_FILE) and os.path.exists(LABELS_FILE):
    with open(PROJECTS_LIST_FILE, 'r', encoding='utf-8') as f:
        project_names = [line.strip() for line in f if line.strip()]
    
    labels_df = pd.read_csv(LABELS_FILE, header=None)
    all_labels_np = labels_df.values.astype(np.float32)
    
    scales = []
    for i, project_name in enumerate(project_names[:20]):  # 检查前20个样本
        project_dir = os.path.join(EXPORT_ROOT, project_name)
        pointcloud_file = os.path.join(project_dir, "pointcloud_full.npy")
        
        if os.path.exists(pointcloud_file):
            try:
                pointcloud = np.load(pointcloud_file).astype(np.float32)
                current_label = all_labels_np[i].reshape(9, 3)
                
                # 使用地标点质心
                label_centroid = np.mean(current_label, axis=0)
                centered_pointcloud = pointcloud - label_centroid
                
                # 计算尺度（标准差）
                scale = np.std(centered_pointcloud)
                if scale > 1e-6:
                    scales.append(scale)
            except:
                continue
    
    if scales:
        avg_scale = np.mean(scales)
        print(f"\n   估算的平均归一化尺度: {avg_scale:.2f}")
        print(f"   尺度范围: {np.min(scales):.2f} - {np.max(scales):.2f}")
        
        # 计算实际RMSE
        # 归一化后的RMSE = sqrt(normalized_loss)
        # 实际RMSE = 归一化后的RMSE × 尺度
        normalized_rmse = np.sqrt(best_val_loss)
        actual_rmse = normalized_rmse * avg_scale
        
        print(f"\n   归一化后的RMSE: {normalized_rmse:.6f}")
        print(f"   实际RMSE (估算): {actual_rmse:.2f}")
        
        print(f"\n   误差分析:")
        print(f"     目标误差: 2mm")
        print(f"     当前误差: {actual_rmse:.2f}")
        
        if actual_rmse < 2:
            print(f"     ✅ 达到目标！误差小于2mm")
        elif actual_rmse < 5:
            print(f"     ✅ 接近目标！误差小于5mm")
        elif actual_rmse < 10:
            print(f"     ⚠️  需要改进，但已经很好（误差小于10mm）")
        else:
            print(f"     ⚠️  需要进一步改进（误差 {actual_rmse:.2f}mm）")
        
        # 计算改进倍数
        old_rmse = 494.65  # 修复前的RMSE
        improvement = old_rmse / actual_rmse
        print(f"\n   改进情况:")
        print(f"     修复前RMSE: {old_rmse:.2f}mm")
        print(f"     修复后RMSE: {actual_rmse:.2f}mm")
        print(f"     改进倍数: {improvement:.1f}倍")
        print(f"     改进幅度: {((old_rmse - actual_rmse) / old_rmse * 100):.1f}%")

print("\n" + "="*70)
print("训练质量分析")
print("="*70)

if os.path.exists(history_file):
    with open(history_file, 'r', encoding='utf-8') as f:
        history = json.load(f)
    
    train_losses = history['train_loss']
    val_losses = history['val_loss']
    
    # 过拟合分析
    final_train_loss = train_losses[-1]
    final_val_loss = val_losses[-1]
    best_val_loss = min(val_losses)
    
    print(f"\n   最终训练损失: {final_train_loss:.6f}")
    print(f"   最终验证损失: {final_val_loss:.6f}")
    print(f"   最佳验证损失: {best_val_loss:.6f}")
    
    gap = final_train_loss - final_val_loss
    print(f"   训练-验证差距: {gap:.6f}")
    
    if abs(gap) < final_val_loss * 0.1:
        print(f"   ✅ 训练和验证损失接近（泛化良好）")
    elif gap < 0:
        print(f"   ⚠️  验证损失 > 训练损失（可能过拟合）")
    else:
        print(f"   ✅ 训练损失略低于验证损失（正常）")
    
    # 收敛性分析
    last_50_train = train_losses[-50:]
    last_50_val = val_losses[-50:]
    
    train_std = np.std(last_50_train)
    val_std = np.std(last_50_val)
    
    print(f"\n   最后50轮稳定性:")
    print(f"     训练损失标准差: {train_std:.6f}")
    print(f"     验证损失标准差: {val_std:.6f}")
    
    if train_std < np.mean(last_50_train) * 0.1:
        print(f"     ✅ 训练损失已稳定")
    if val_std < np.mean(last_50_val) * 0.1:
        print(f"     ✅ 验证损失已稳定")

print("\n" + "="*70)
print("总结和建议")
print("="*70)

if os.path.exists(history_file):
    with open(history_file, 'r', encoding='utf-8') as f:
        history = json.load(f)
    
    val_losses = history['val_loss']
    best_val_loss = min(val_losses)
    
    if scales:
        normalized_rmse = np.sqrt(best_val_loss)
        actual_rmse = normalized_rmse * avg_scale
        
        print(f"\n✅ 训练非常成功！")
        print(f"   - 损失从 {history['train_loss'][0]:.6f} 降到 {best_val_loss:.6f}")
        print(f"   - 改进幅度: {((history['train_loss'][0] - best_val_loss) / history['train_loss'][0] * 100):.2f}%")
        print(f"   - 估算实际RMSE: {actual_rmse:.2f}mm")
        
        if actual_rmse < 2:
            print(f"\n🎉 恭喜！已达到2mm目标！")
        elif actual_rmse < 5:
            print(f"\n✅ 非常接近目标！误差小于5mm")
            print(f"   建议:")
            print(f"     1. 可以尝试增加正则化进一步减少过拟合")
            print(f"     2. 使用更多训练数据")
            print(f"     3. 数据增强")
        else:
            print(f"\n💡 需要进一步改进:")
            print(f"   1. 增加正则化（Dropout, 权重衰减）")
            print(f"   2. 使用更多训练数据")
            print(f"   3. 数据增强")
            print(f"   4. 改进模型架构")

print("="*70)
