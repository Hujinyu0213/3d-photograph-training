"""
分析超参数调优结果并与历史模型对比
"""
import os
import json
import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS_DIR = os.path.join(ROOT_DIR, "results")

# 加载超参数调优结果
tuning_file = os.path.join(RESULTS_DIR, "hyperparameter_tuning_results.json")
with open(tuning_file, 'r', encoding='utf-8') as f:
    tuning_data = json.load(f)

print("="*80)
print("🔍 超参数调优结果分析")
print("="*80)

# 最佳配置
best_params = tuning_data['best_params']
best_loss = tuning_data['best_mean_val_loss']

print(f"\n✨ 最佳超参数配置 (来自 {tuning_data['total_trials']} 次试验):")
print("-"*80)
for key, value in best_params.items():
    print(f"  {key:30} = {value}")
print(f"\n  {'最佳平均验证损失':30} = {best_loss:.8f}")

# Top 5 配置
print("\n📊 Top 5 最佳配置:")
print("-"*80)
sorted_results = sorted(tuning_data['all_results'], key=lambda x: x['mean_val_loss'])

for i, res in enumerate(sorted_results[:5], 1):
    print(f"\n{i}. Mean Val Loss = {res['mean_val_loss']:.8f} (±{res['std_val_loss']:.8f})")
    for key, value in res['params'].items():
        print(f"   {key}: {value}")

# 加载历史训练记录进行对比
print("\n" + "="*80)
print("📈 与历史模型对比")
print("="*80)

history_files = {
    'K折+增强+FPS (旧配置)': 'training_histories/training_history_kfold_aug_fps.json',
    '单次训练+增强+FPS': 'training_histories/training_history_full_aug_fps.json',
    '旧K折模型（无增强）': 'training_histories/training_history_kfold.json'
}

comparison_data = []

for name, filepath in history_files.items():
    full_path = os.path.join(RESULTS_DIR, filepath)
    if not os.path.exists(full_path):
        continue
    
    with open(full_path, 'r', encoding='utf-8') as f:
        history = json.load(f)
    
    if 'statistics' in history:  # K-fold 格式
        stats = history['statistics']
        mean_loss = stats.get('mean_best_val_loss', stats.get('mean_val_loss', 0))
        best_fold_loss = stats.get('best_fold_loss', 0)
        comparison_data.append({
            'name': name,
            'type': 'K-fold',
            'mean_val_loss': mean_loss,
            'best_val_loss': best_fold_loss
        })
    elif 'best_val_loss' in history:  # 单次训练格式
        best_loss_hist = history['best_val_loss']
        comparison_data.append({
            'name': name,
            'type': 'Single',
            'best_val_loss': best_loss_hist
        })

# 添加超参数调优结果
comparison_data.append({
    'name': '超参数调优最佳配置',
    'type': 'K-fold (Tuned)',
    'mean_val_loss': best_loss,
    'best_val_loss': min([res['mean_val_loss'] for res in tuning_data['all_results']])
})

print(f"\n{'模型':<30} {'类型':<20} {'最佳验证损失':<20} {'改进幅度'}")
print("-"*100)

# 找到基准（旧K折+增强+FPS）
baseline = None
for item in comparison_data:
    if 'K折+增强+FPS' in item['name']:
        baseline = item['mean_val_loss'] if 'mean_val_loss' in item else item.get('best_val_loss', 0)
        break

for item in comparison_data:
    name = item['name']
    model_type = item['type']
    
    if 'mean_val_loss' in item:
        loss = item['mean_val_loss']
    else:
        loss = item['best_val_loss']
    
    if baseline and baseline > 0:
        improvement = ((baseline - loss) / baseline) * 100
        improvement_str = f"{improvement:+.2f}%"
    else:
        improvement_str = "N/A"
    
    print(f"{name:<30} {model_type:<20} {loss:<20.8f} {improvement_str}")

print("\n" + "="*80)
print("💡 结论")
print("="*80)

if baseline:
    improvement = ((baseline - best_loss) / baseline) * 100
    print(f"\n✅ 超参数调优使 K 折验证损失改进了 {improvement:.2f}%")
    print(f"   从 {baseline:.8f} → {best_loss:.8f}")
else:
    print(f"\n✅ 超参数调优找到最佳配置，验证损失: {best_loss:.8f}")

print(f"\n🎯 关键改进:")
print(f"   - 学习率调整至 {best_params['learning_rate']}")
print(f"   - Dropout 增加至 {best_params['dropout_rate']} (防止过拟合)")
print(f"   - 学习率衰减步长 {best_params['lr_decay_step']} epochs")
print(f"   - Batch size = {best_params['batch_size']}")

print(f"\n📝 建议:")
print(f"   1. 使用最佳超参数重新训练完整模型（更多 epochs，如 250-300）")
print(f"   2. 在测试集上评估性能，验证泛化能力")
print(f"   3. 如果效果显著，可作为最终模型")

print("\n" + "="*80)
