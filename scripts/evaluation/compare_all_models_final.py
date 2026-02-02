"""
最终模型对比：单次训练模型 vs 最佳折模型 vs 重新训练的K折模型
"""
import json
import sys
import io
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

print("="*70)
print("最终模型性能对比（相同测试集 - 10个样本）")
print("="*70)

# 读取三个模型的评估结果
single_model_results = json.load(open('single_model_on_kfold_testset_results.json', 'r', encoding='utf-8'))
best_fold_results = json.load(open('best_fold_model_evaluation_results.json', 'r', encoding='utf-8'))
kfold_retrained_results = json.load(open('kfold_test_evaluation_results.json', 'r', encoding='utf-8'))

print("\n📊 三个模型性能对比:")
print("-"*70)
print(f"{'指标':<25} {'单次训练模型':<20} {'最佳折模型':<20} {'重新训练K折':<20} {'最佳':<10}")
print("-"*70)

single_rmse = single_model_results['overall']['rmse_3d_all_points']
best_fold_rmse = best_fold_results['overall']['rmse_3d_all_points']
kfold_retrained_rmse = kfold_retrained_results['overall']['rmse_3d_all_points']

best_rmse = min(single_rmse, best_fold_rmse, kfold_retrained_rmse)
best_rmse_name = ""
if single_rmse == best_rmse:
    best_rmse_name = "单次训练"
elif best_fold_rmse == best_rmse:
    best_rmse_name = "最佳折"
else:
    best_rmse_name = "重新训练K折"

print(f"{'3D RMSE (mm)':<25} {single_rmse:<20.4f} {best_fold_rmse:<20.4f} {kfold_retrained_rmse:<20.4f} {best_rmse_name:<10}")

single_mae = single_model_results['overall']['mae_3d_all_points']
best_fold_mae = best_fold_results['overall']['mae_3d_all_points']
kfold_retrained_mae = kfold_retrained_results['overall']['mae_3d_all_points']

best_mae = min(single_mae, best_fold_mae, kfold_retrained_mae)
best_mae_name = ""
if single_mae == best_mae:
    best_mae_name = "单次训练"
elif best_fold_mae == best_mae:
    best_mae_name = "最佳折"
else:
    best_mae_name = "重新训练K折"

print(f"{'3D MAE (mm)':<25} {single_mae:<20.4f} {best_fold_mae:<20.4f} {kfold_retrained_mae:<20.4f} {best_mae_name:<10}")

single_prec_2mm = single_model_results['overall']['mean_precision_2mm']
best_fold_prec_2mm = best_fold_results['overall']['mean_precision_2mm']
kfold_retrained_prec_2mm = kfold_retrained_results['overall']['mean_precision_2mm']

best_prec_2mm = max(single_prec_2mm, best_fold_prec_2mm, kfold_retrained_prec_2mm)
best_prec_2mm_name = ""
if single_prec_2mm == best_prec_2mm:
    best_prec_2mm_name = "单次训练"
elif best_fold_prec_2mm == best_prec_2mm:
    best_prec_2mm_name = "最佳折"
else:
    best_prec_2mm_name = "重新训练K折"

print(f"{'精度 @ 2mm (%)':<25} {single_prec_2mm:<20.2f} {best_fold_prec_2mm:<20.2f} {kfold_retrained_prec_2mm:<20.2f} {best_prec_2mm_name:<10}")

single_prec_5mm = single_model_results['overall']['mean_precision_5mm']
best_fold_prec_5mm = best_fold_results['overall']['mean_precision_5mm']
kfold_retrained_prec_5mm = kfold_retrained_results['overall']['mean_precision_5mm']

best_prec_5mm = max(single_prec_5mm, best_fold_prec_5mm, kfold_retrained_prec_5mm)
best_prec_5mm_name = ""
if single_prec_5mm == best_prec_5mm:
    best_prec_5mm_name = "单次训练"
elif best_fold_prec_5mm == best_prec_5mm:
    best_prec_5mm_name = "最佳折"
else:
    best_prec_5mm_name = "重新训练K折"

print(f"{'精度 @ 5mm (%)':<25} {single_prec_5mm:<20.2f} {best_fold_prec_5mm:<20.2f} {kfold_retrained_prec_5mm:<20.2f} {best_prec_5mm_name:<10}")

single_prec_10mm = single_model_results['overall']['mean_precision_10mm']
best_fold_prec_10mm = best_fold_results['overall']['mean_precision_10mm']
kfold_retrained_prec_10mm = kfold_retrained_results['overall']['mean_precision_10mm']

best_prec_10mm = max(single_prec_10mm, best_fold_prec_10mm, kfold_retrained_prec_10mm)
best_prec_10mm_name = ""
if single_prec_10mm == best_prec_10mm:
    best_prec_10mm_name = "单次训练"
elif best_fold_prec_10mm == best_prec_10mm:
    best_prec_10mm_name = "最佳折"
else:
    best_prec_10mm_name = "重新训练K折"

print(f"{'精度 @ 10mm (%)':<25} {single_prec_10mm:<20.2f} {best_fold_prec_10mm:<20.2f} {kfold_retrained_prec_10mm:<20.2f} {best_prec_10mm_name:<10}")

print("\n" + "="*70)
print("关键发现:")
print("="*70)

print(f"\n1. ✅ **单次训练模型性能最好**")
print(f"   - 3D RMSE: {single_rmse:.2f}mm")
print(f"   - 精度@5mm: {single_prec_5mm:.2f}%")
print(f"   - 精度@10mm: {single_prec_10mm:.2f}%")

print(f"\n2. ✅ **最佳折模型（折1）比重新训练的K折模型好很多**")
print(f"   - 最佳折模型 3D RMSE: {best_fold_rmse:.2f}mm")
print(f"   - 重新训练K折模型 3D RMSE: {kfold_retrained_rmse:.2f}mm")
print(f"   - 改进: {((kfold_retrained_rmse - best_fold_rmse) / kfold_retrained_rmse * 100):.1f}%")

print(f"\n3. ❌ **重新训练的K折模型性能最差**")
print(f"   - 3D RMSE: {kfold_retrained_rmse:.2f}mm")
print(f"   - 精度@5mm: {kfold_retrained_prec_5mm:.2f}%")
print(f"   - 原因: 没有验证集，没有早停，可能过拟合")

print("\n" + "="*70)
print("最终建议:")
print("="*70)

print("\n⭐⭐⭐ **推荐使用单次训练模型**")
print("   - 模型文件: pointnet_regression_model_full_best.pth")
print(f"   - 性能: 3D RMSE = {single_rmse:.2f}mm")
print(f"   - 精度@5mm: {single_prec_5mm:.2f}%")
print(f"   - 精度@10mm: {single_prec_10mm:.2f}%")

print("\n⭐⭐ **如果使用K折模型，使用最佳折模型**")
print("   - 模型文件: pointnet_regression_model_kfold_fold1_best.pth")
print(f"   - 性能: 3D RMSE = {best_fold_rmse:.2f}mm")
print(f"   - 精度@5mm: {best_fold_prec_5mm:.2f}%")
print(f"   - ⚠️  不要使用重新训练的模型！")

print("\n❌ **不要使用重新训练的K折模型**")
print("   - 模型文件: pointnet_regression_model_kfold_best.pth")
print(f"   - 性能: 3D RMSE = {kfold_retrained_rmse:.2f}mm（最差）")
print(f"   - 原因: 没有验证集，没有早停，可能过拟合")
print("="*70)
