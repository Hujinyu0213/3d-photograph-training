"""
数据增强+FPS训练结果分析脚本
分析 main_script_full_pointcloud_aug_fps.py 的训练结果
"""
import json
import os
import numpy as np

# matplotlib optional for plotting
try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
    matplotlib.rcParams['axes.unicode_minus'] = False
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# 读取训练历史
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(BASE_DIR))
history_path = os.path.join(ROOT_DIR, "results", "training_history_full_aug_fps.json")

with open(history_path, 'r') as f:
    history = json.load(f)

epochs = history['epoch']
train_loss = history['train_loss']
val_loss = history['val_loss']

# 统计分析
print("=" * 60)
print("📊 数据增强+FPS训练结果分析")
print("=" * 60)
print(f"\n训练配置:")
print(f"  总轮数: {len(epochs)}")
print(f"  采样方式: FPS (最远点采样) 到 8192 点")
print(f"  数据增强: 旋转(±15°) + 缩放(±5%) + 平移 + 抖动(σ=0.005)")
print(f"  验证集比例: 20%")

print(f"\n训练结果:")
print(f"  最终训练损失: {train_loss[-1]:.8f}")
print(f"  最终验证损失: {val_loss[-1]:.8f}")
print(f"  最佳验证损失: {min(val_loss):.8f} (epoch {epochs[np.argmin(val_loss)]})")
print(f"  最差验证损失: {max(val_loss):.8f} (epoch {epochs[np.argmax(val_loss)]})")

# 与之前模型对比（如果存在）
kfold_history_path = os.path.join(ROOT_DIR, "results", "training_history_kfold.json")
if os.path.exists(kfold_history_path):
    with open(kfold_history_path, 'r') as f:
        kfold = json.load(f)
    kfold_best_val = kfold['final_model']['best_val_loss']
    kfold_test_loss = kfold['test_loss']
    
    print(f"\n📈 与K折模型对比:")
    print(f"  K折最终模型验证损失: {kfold_best_val:.8f}")
    print(f"  K折最终模型测试损失: {kfold_test_loss:.8f}")
    print(f"  新模型最佳验证损失: {min(val_loss):.8f}")
    
    improvement = ((kfold_best_val - min(val_loss)) / kfold_best_val) * 100
    if improvement > 0:
        print(f"  ✅ 改进: {improvement:.2f}% (验证损失降低)")
    else:
        print(f"  ⚠️  变化: {improvement:.2f}% (验证损失上升)")

# 训练稳定性分析
val_loss_std = np.std(val_loss)
val_loss_last_50 = val_loss[-50:]
val_loss_last_50_std = np.std(val_loss_last_50)

print(f"\n🔍 训练稳定性:")
print(f"  验证损失标准差(全程): {val_loss_std:.8f}")
print(f"  验证损失标准差(最后50轮): {val_loss_last_50_std:.8f}")
if val_loss_last_50_std < val_loss_std * 0.5:
    print(f"  ✅ 训练后期稳定，收敛良好")
else:
    print(f"  ⚠️  训练后期仍有波动，可能需要更多轮或调整学习率")

# 过拟合检测
train_val_gap = train_loss[-1] - val_loss[-1]
print(f"\n🎯 过拟合检测:")
print(f"  训练-验证损失差: {train_val_gap:.8f}")
if abs(train_val_gap) < 0.0001:
    print(f"  ✅ 训练集和验证集损失接近，未过拟合")
elif train_val_gap < -0.0002:
    print(f"  ⚠️  验证损失高于训练损失，可能欠拟合或数据增强过强")
else:
    print(f"  ⚠️  训练损失明显低于验证损失，存在轻微过拟合")

# 绘制训练曲线
if HAS_MATPLOTLIB:
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, label='训练损失', alpha=0.8)
    plt.plot(epochs, val_loss, label='验证损失', alpha=0.8)
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('训练和验证损失曲线')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    # 只画最后100轮，看收敛情况
    start_idx = max(0, len(epochs) - 100)
    plt.plot(epochs[start_idx:], train_loss[start_idx:], label='训练损失', alpha=0.8)
    plt.plot(epochs[start_idx:], val_loss[start_idx:], label='验证损失', alpha=0.8)
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('最后100轮损失曲线（收敛细节）')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = os.path.join(ROOT_DIR, "results", "training_analysis_aug_fps.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n📁 训练曲线已保存: {output_path}")
else:
    print(f"\n⚠️  matplotlib 未安装，跳过绘图（仅显示文本分析）")
    print(f"   安装方式: pip install matplotlib")

# 建议
print(f"\n💡 改进建议:")
if min(val_loss) < 0.0002:
    print(f"  ✅ 验证损失已经很低 (<0.0002)，模型表现良好")
    print(f"  - 可以在独立测试集上评估真实性能")
    print(f"  - 考虑增加样本数量或尝试 PointNet++")
else:
    print(f"  - 验证损失仍有下降空间，可以:")
    print(f"    1. 增加训练轮数或调整学习率衰减")
    print(f"    2. 调整数据增强强度（当前可能过强或过弱）")
    print(f"    3. 尝试不同的采样点数（4096/6144/8192）")

if val_loss_last_50_std > 0.00005:
    print(f"  - 训练后期仍有波动，建议:")
    print(f"    1. 降低学习率或使用 Cosine Annealing")
    print(f"    2. 增加 batch size（如果显存允许）")
    print(f"    3. 添加早停机制（patience=30-50）")

print(f"\n下一步:")
print(f"  1. 运行评估脚本在测试集上验证模型:")
print(f"     python scripts/evaluation/evaluate_model_testset.py")
print(f"  2. 与之前的模型做详细对比")
print(f"  3. 如果效果好，可以尝试 K 折交叉验证版本（加增强+FPS）")

print("=" * 60)
