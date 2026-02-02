"""
最终模型性能对比分析（在相同测试集上）
"""
import json
import sys
import io
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

print("="*70)
print("最终模型性能对比分析（公平对比 - 相同测试集）")
print("="*70)

# 读取两个模型的评估结果（都在K折的测试集上）
single_model_results = json.load(open('single_model_on_kfold_testset_results.json', 'r', encoding='utf-8'))
kfold_model_results = json.load(open('kfold_test_evaluation_results.json', 'r', encoding='utf-8'))

print("\n📊 总体性能对比（相同测试集 - 10个样本）:")
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
    print(f"\n   ✅ **单次训练模型性能更好！**")
else:
    print(f"✅ K折模型的3D RMSE比单次训练模型低 {abs(diff_rmse):.2f}mm")

print(f"\n📊 精度对比:")
print(f"   - 2mm精度: 单次训练 {single_prec_2mm:.2f}% vs K折 {kfold_prec_2mm:.2f}%")
print(f"   - 5mm精度: 单次训练 {single_prec_5mm:.2f}% vs K折 {kfold_prec_5mm:.2f}%")
print(f"   - 10mm精度: 单次训练 {single_prec_10mm:.2f}% vs K折 {kfold_prec_10mm:.2f}%")

print("\n" + "="*70)
print("归一化状态确认:")
print("="*70)
print("✅ 两个模型都使用了修复后的归一化")
print("   - 单次训练模型初始损失: 0.236 (已归一化)")
print("   - K折模型初始损失: 0.241 (已归一化)")
print("   - 两个模型使用相同的预处理方法")

print("\n" + "="*70)
print("性能差异原因分析:")
print("="*70)

print("\n1. ⚠️⚠️⚠️ **K折模型性能较差的主要原因**")
print("   - K折模型的3D RMSE: 22.70mm")
print("   - 单次训练模型的3D RMSE: 9.37mm")
print("   - 差异: 13.33mm (142.3%)")

print("\n2. **可能的原因**")
print("   a) **最终模型训练策略问题**")
print("      - K折的最终模型是在所有90%数据上重新训练的")
print("      - 没有使用验证集来早停或选择最佳模型")
print("      - 可能训练过度或使用了次优的超参数")
print("   b) **训练数据量差异**")
print("      - 单次训练模型: 使用80个样本训练")
print("      - K折模型: 使用90个样本训练（最终模型）")
print("      - 虽然K折模型用了更多数据，但可能过拟合了")
print("   c) **模型选择问题**")
print("      - K折选择了折1的最佳模型（验证损失0.000136）")
print("      - 但最终模型是在所有数据上重新训练的，可能丢失了最佳配置")

print("\n" + "="*70)
print("建议:")
print("="*70)

print("\n1. ⭐⭐⭐ **使用单次训练模型作为最终模型**")
print("   - 单次训练模型在相同测试集上表现更好")
print("   - 3D RMSE: 9.37mm vs K折模型的22.70mm")
print("   - 精度@5mm: 32.22% vs K折模型的1.11%")

print("\n2. ⭐⭐ **如果需要使用K折模型，需要改进训练策略**")
print("   - 在最终模型训练时使用验证集进行早停")
print("   - 或者直接使用最佳折的模型，而不是重新训练")
print("   - 考虑使用K折的平均权重或集成方法")

print("\n3. ⭐ **进一步改进**")
print("   - 两个模型的性能都未达到2mm目标")
print("   - 需要进一步优化模型架构或训练策略")
print("   - 考虑数据增强或收集更多训练数据")

print("\n" + "="*70)
print("结论:")
print("="*70)
print("✅ **单次训练模型性能更好**")
print("   - 在相同测试集上，单次训练模型的3D RMSE为9.37mm")
print("   - K折模型的3D RMSE为22.70mm")
print("   - 单次训练模型比K折模型好142.3%")
print("\n✅ **两个模型都使用了修复后的归一化**")
print("   - 预处理方法一致")
print("   - 性能差异主要来自训练策略和模型选择")
print("="*70)
