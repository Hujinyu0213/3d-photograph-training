"""
对比单次训练模型和K折模型的性能
"""
import json
import os
import sys
import io
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# 读取两个模型的评估结果
single_model_results = json.load(open('test_evaluation_results.json', 'r', encoding='utf-8'))
kfold_model_results = json.load(open('kfold_test_evaluation_results.json', 'r', encoding='utf-8'))

print("="*70)
print("模型性能对比分析")
print("="*70)

print("\n📊 总体性能对比:")
print("-"*70)
print(f"{'指标':<25} {'单次训练模型':<20} {'K折模型':<20} {'差异':<15}")
print("-"*70)

single_rmse_3d = single_model_results['overall']['rmse_3d_all_points']
kfold_rmse_3d = kfold_model_results['overall']['rmse_3d_all_points']
diff_rmse = kfold_rmse_3d - single_rmse_3d
diff_rmse_pct = (diff_rmse / single_rmse_3d) * 100

single_mae_3d = single_model_results['overall']['mae_3d_all_points']
kfold_mae_3d = kfold_model_results['overall']['mae_3d_all_points']
diff_mae = kfold_mae_3d - single_mae_3d

print(f"{'3D RMSE (mm)':<25} {single_rmse_3d:<20.4f} {kfold_rmse_3d:<20.4f} {diff_rmse:+.4f} ({diff_rmse_pct:+.1f}%)")
print(f"{'3D MAE (mm)':<25} {single_mae_3d:<20.4f} {kfold_mae_3d:<20.4f} {diff_mae:+.4f}")

single_prec_2mm = single_model_results['overall']['mean_precision_2mm']
kfold_prec_2mm = kfold_model_results['overall']['mean_precision_2mm']
print(f"{'精度 @ 2mm (%)':<25} {single_prec_2mm:<20.2f} {kfold_prec_2mm:<20.2f} {kfold_prec_2mm-single_prec_2mm:+.2f}%")

single_prec_5mm = single_model_results['overall']['mean_precision_5mm']
kfold_prec_5mm = kfold_model_results['overall']['mean_precision_5mm']
print(f"{'精度 @ 5mm (%)':<25} {single_prec_5mm:<20.2f} {kfold_prec_5mm:<20.2f} {kfold_prec_5mm-single_prec_5mm:+.2f}%")

single_prec_10mm = single_model_results['overall']['mean_precision_10mm']
kfold_prec_10mm = kfold_model_results['overall']['mean_precision_10mm']
print(f"{'精度 @ 10mm (%)':<25} {single_prec_10mm:<20.2f} {kfold_prec_10mm:<20.2f} {kfold_prec_10mm-single_prec_10mm:+.2f}%")

print("\n" + "="*70)
print("关键发现:")
print("="*70)

if kfold_rmse_3d > single_rmse_3d:
    print(f"❌ K折模型的3D RMSE比单次训练模型高 {diff_rmse:.2f}mm ({diff_rmse_pct:.1f}%)")
    print(f"   - 单次训练模型: {single_rmse_3d:.2f}mm")
    print(f"   - K折模型: {kfold_rmse_3d:.2f}mm")
else:
    print(f"✅ K折模型的3D RMSE比单次训练模型低 {abs(diff_rmse):.2f}mm")

print(f"\n📊 精度对比:")
print(f"   - 2mm精度: 单次训练 {single_prec_2mm:.2f}% vs K折 {kfold_prec_2mm:.2f}%")
print(f"   - 5mm精度: 单次训练 {single_prec_5mm:.2f}% vs K折 {kfold_prec_5mm:.2f}%")
print(f"   - 10mm精度: 单次训练 {single_prec_10mm:.2f}% vs K折 {kfold_prec_10mm:.2f}%")

print("\n" + "="*70)
print("可能的原因分析:")
print("="*70)

print("\n1. ⚠️⚠️⚠️ **测试集不同** (最重要)")
print("   - 单次训练模型: 使用80/20划分，测试集20个样本")
print("   - K折模型: 使用90/10划分，测试集10个样本")
print("   - 这两个测试集是完全不同的样本！")
print("   - K折的测试集可能包含更难的样本")

print("\n2. ⚠️⚠️ **模型训练数据不同**")
print("   - 单次训练模型: 使用80个样本训练")
print("   - K折模型: 使用90个样本训练（最终模型）")
print("   - 虽然K折模型用了更多数据，但可能过拟合了")

print("\n3. ⚠️ **归一化修复的影响**")
print("   - K折模型是在修复归一化后训练的")
print("   - 单次训练模型可能是在修复前训练的")
print("   - 需要确认单次训练模型是否也使用了修复后的归一化")

print("\n4. ⚠️ **最终模型训练策略**")
print("   - K折的最终模型是在所有90%数据上重新训练的")
print("   - 没有使用验证集来早停或选择最佳模型")
print("   - 可能训练过度或使用了次优的超参数")

print("\n" + "="*70)
print("建议:")
print("="*70)

print("\n1. ⭐⭐⭐ **在相同的测试集上评估两个模型**")
print("   - 使用相同的测试集（例如K折的10个样本）")
print("   - 重新评估单次训练模型，确保公平对比")

print("\n2. ⭐⭐ **检查单次训练模型是否使用了修复后的归一化**")
print("   - 如果单次训练模型是在修复前训练的，需要重新训练")

print("\n3. ⭐ **分析K折模型的训练历史**")
print("   - 检查最终模型是否过拟合")
print("   - 考虑使用验证集进行早停")

print("\n4. ⭐ **对比两个模型的训练损失**")
print("   - 检查归一化损失是否可比")
print("   - 确认两个模型都使用了相同的预处理")
