"""
检查单次训练模型的训练历史，确认是否使用了修复后的归一化
"""
import json
import sys
import io
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

print("="*70)
print("检查单次训练模型的归一化状态")
print("="*70)

# 读取训练历史
with open('training_history_full.json', 'r', encoding='utf-8') as f:
    history = json.load(f)

print("\n📊 训练历史分析:")
print("-"*70)

# 检查初始损失
if 'train_loss' in history and len(history['train_loss']) > 0:
    initial_loss = history['train_loss'][0]
    final_loss = history['train_loss'][-1]
    
    print(f"初始训练损失: {initial_loss:.6f}")
    print(f"最终训练损失: {final_loss:.6f}")
    print(f"损失下降: {(1 - final_loss/initial_loss)*100:.2f}%")
    
    # 判断是否使用了修复后的归一化
    # 修复后的归一化：初始损失应该在0.2-0.3左右（基于之前的训练结果）
    # 修复前的归一化：初始损失可能在数百万（基于之前的训练结果）
    
    print("\n🔍 归一化状态判断:")
    print("-"*70)
    
    if initial_loss < 1.0:
        print("✅ **使用了修复后的归一化**")
        print(f"   - 初始损失: {initial_loss:.6f} (很小，说明数据已归一化)")
        print(f"   - 这与修复后的训练结果一致（初始损失约0.236）")
    elif initial_loss > 1000000:
        print("❌ **使用了修复前的归一化**")
        print(f"   - 初始损失: {initial_loss:.6f} (非常大，说明数据未正确归一化)")
        print(f"   - 这与修复前的训练结果一致（初始损失约5,000,000）")
    else:
        print("⚠️  **无法确定**")
        print(f"   - 初始损失: {initial_loss:.6f}")
        print(f"   - 需要进一步检查")
    
    # 检查验证损失
    if 'val_loss' in history and len(history['val_loss']) > 0:
        best_val_loss = min(history['val_loss'])
        final_val_loss = history['val_loss'][-1]
        
        print(f"\n验证损失:")
        print(f"   最佳验证损失: {best_val_loss:.6f}")
        print(f"   最终验证损失: {final_val_loss:.6f}")
        
        # 与修复后的K折模型对比
        print(f"\n与K折模型对比:")
        print(f"   - 单次训练模型最佳验证损失: {best_val_loss:.6f}")
        print(f"   - K折模型最佳验证损失: 0.000136 (折1)")
        print(f"   - K折模型平均验证损失: 0.000833")
        
        if best_val_loss < 0.001:
            print(f"   ✅ 损失在同一数量级，都使用了修复后的归一化")
        else:
            print(f"   ⚠️  损失不在同一数量级，可能归一化不一致")

# 检查训练轮数
if 'train_loss' in history:
    num_epochs = len(history['train_loss'])
    print(f"\n训练轮数: {num_epochs}")

# 检查是否有其他信息
print(f"\n训练历史中的其他信息:")
for key in history.keys():
    if key not in ['train_loss', 'val_loss', 'epoch']:
        print(f"   - {key}: {history[key]}")

print("\n" + "="*70)
print("结论:")
print("="*70)

if 'train_loss' in history and len(history['train_loss']) > 0:
    initial_loss = history['train_loss'][0]
    if initial_loss < 1.0:
        print("✅ 单次训练模型**使用了修复后的归一化**")
        print("   - 这意味着两个模型使用了相同的预处理")
        print("   - 性能差异主要来自：")
        print("     1. 测试集不同（最重要）")
        print("     2. 训练数据量不同（80 vs 90样本）")
        print("     3. 模型训练策略不同")
    else:
        print("❌ 单次训练模型**可能使用了修复前的归一化**")
        print("   - 需要重新训练单次模型，使用修复后的归一化")
        print("   - 然后才能在相同测试集上公平对比")
else:
    print("⚠️  无法从训练历史中确定归一化状态")

print("="*70)
