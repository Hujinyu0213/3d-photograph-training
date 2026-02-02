"""
误差报告比较工具
用于比较两个 error_report.csv 文件的差异
"""
import os
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 设置 matplotlib 使用支持中文的字体，如果失败则使用英文
try:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
except:
    # 如果中文字体不可用，使用英文标签
    pass

# 添加当前目录到 Python 路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

OUTPUT_DIR = os.path.join(BASE_DIR, 'results')

def compare_error_reports(report1_path, report2_path, name1="报告1", name2="报告2"):
    """
    比较两个误差报告
    
    参数:
        report1_path: 第一个报告文件路径
        report2_path: 第二个报告文件路径
        name1: 第一个报告的名称
        name2: 第二个报告的名称
    """
    print("=" * 60)
    print("📊 误差报告比较工具")
    print("=" * 60)
    
    # 读取报告
    try:
        df1 = pd.read_csv(report1_path)
        print(f"\n✅ 成功加载 {name1}: {report1_path}")
        print(f"   样本数: {len(df1)}")
    except Exception as e:
        print(f"❌ 无法加载 {name1}: {e}")
        return
    
    try:
        df2 = pd.read_csv(report2_path)
        print(f"✅ 成功加载 {name2}: {report2_path}")
        print(f"   样本数: {len(df2)}")
    except Exception as e:
        print(f"❌ 无法加载 {name2}: {e}")
        return
    
    # 检查列是否一致
    if list(df1.columns) != list(df2.columns):
        print("⚠️  警告: 两个报告的列不完全一致")
        common_cols = set(df1.columns) & set(df2.columns)
        print(f"   共同列: {len(common_cols)} 个")
        df1 = df1[list(common_cols)]
        df2 = df2[list(common_cols)]
    
    # 确保样本数一致
    min_samples = min(len(df1), len(df2))
    df1 = df1.iloc[:min_samples]
    df2 = df2.iloc[:min_samples]
    print(f"\n📊 比较 {min_samples} 个样本")
    
    # 比较主要指标
    print("\n" + "=" * 60)
    print("📈 主要指标比较")
    print("=" * 60)
    
    metrics = ['RMSE', 'MAE', 'X_Error', 'Y_Error', 'Z_Error']
    available_metrics = [m for m in metrics if m in df1.columns]
    
    comparison_results = {}
    
    for metric in available_metrics:
        val1 = df1[metric].values
        val2 = df2[metric].values
        
        mean1 = np.mean(val1)
        mean2 = np.mean(val2)
        std1 = np.std(val1)
        std2 = np.std(val2)
        min1 = np.min(val1)
        min2 = np.min(val2)
        max1 = np.max(val1)
        max2 = np.max(val2)
        
        diff = mean2 - mean1
        improvement = (mean1 - mean2) / mean1 * 100 if mean1 > 0 else 0
        
        comparison_results[metric] = {
            'mean1': mean1, 'mean2': mean2, 'diff': diff, 'improvement': improvement
        }
        
        print(f"\n{metric}:")
        print(f"  {name1:15s}: {mean1:.4f} ± {std1:.4f} (范围: {min1:.4f} - {max1:.4f})")
        print(f"  {name2:15s}: {mean2:.4f} ± {std2:.4f} (范围: {min2:.4f} - {max2:.4f})")
        
        if diff < 0:
            print(f"  ✅ {name2} 更好（低 {abs(diff):.4f}，改善 {abs(improvement):.2f}%）")
        elif diff > 0:
            print(f"  ✅ {name1} 更好（低 {diff:.4f}，改善 {improvement:.2f}%）")
        else:
            print(f"  🤝 两个报告相同")
    
    # 比较每个地标点的误差
    landmark_cols = [col for col in df1.columns if col.endswith('_Error') and col not in metrics]
    
    if landmark_cols:
        print("\n" + "=" * 60)
        print("📍 各地标点误差比较")
        print("=" * 60)
        
        landmark_comparison = []
        for col in landmark_cols:
            landmark_name = col.replace('_Error', '')
            mean1 = np.mean(df1[col].values)
            mean2 = np.mean(df2[col].values)
            diff = mean2 - mean1
            improvement = (mean1 - mean2) / mean1 * 100 if mean1 > 0 else 0
            
            landmark_comparison.append({
                'landmark': landmark_name,
                'mean1': mean1,
                'mean2': mean2,
                'diff': diff,
                'improvement': improvement
            })
            
            status = "✅" if diff < 0 else "⚠️" if diff > 0 else "🤝"
            print(f"{status} {landmark_name:20s}: {name1}={mean1:.4f}, {name2}={mean2:.4f}, "
                  f"差异={diff:+.4f} ({improvement:+.2f}%)")
    
    # 绘制比较图表
    print("\n" + "=" * 60)
    print("📊 生成比较图表...")
    print("=" * 60)
    
    # 创建图表
    n_metrics = len(available_metrics)
    if n_metrics > 0:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        for idx, metric in enumerate(available_metrics[:4]):  # 最多显示4个指标
            ax = axes[idx]
            
            data1 = df1[metric].values
            data2 = df2[metric].values
            
            bp = ax.boxplot([data1, data2], labels=[name1, name2], patch_artist=True)
            
            # 设置颜色
            colors = ['lightblue', 'lightcoral']
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
            
            ax.set_ylabel(f'{metric} (mm)', fontsize=10)
            ax.set_title(f'{metric} Comparison', fontsize=11)
            ax.grid(True, alpha=0.3)
        
        # 隐藏多余的子图
        for idx in range(len(available_metrics), 4):
            axes[idx].axis('off')
        
        plt.tight_layout()
        
        # 保存图表
        comparison_path = os.path.join(OUTPUT_DIR, 'error_report_comparison.png')
        plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✅ 比较图表已保存到: {comparison_path}")
    
    # 保存详细比较报告
    comparison_df = pd.DataFrame(comparison_results).T
    comparison_df.columns = [f'{name1}_Mean', f'{name2}_Mean', 'Difference', 'Improvement_%']
    comparison_df.index.name = 'Metric'
    
    comparison_csv_path = os.path.join(OUTPUT_DIR, 'error_report_comparison.csv')
    comparison_df.to_csv(comparison_csv_path)
    print(f"✅ 详细比较报告已保存到: {comparison_csv_path}")
    
    # 总结
    print("\n" + "=" * 60)
    print("📋 总结")
    print("=" * 60)
    
    if 'RMSE' in comparison_results:
        rmse_improvement = comparison_results['RMSE']['improvement']
        if rmse_improvement > 0:
            print(f"✅ {name2} 的 RMSE 比 {name1} 改善了 {rmse_improvement:.2f}%")
        elif rmse_improvement < 0:
            print(f"⚠️  {name2} 的 RMSE 比 {name1} 差了 {abs(rmse_improvement):.2f}%")
        else:
            print(f"🤝 两个报告的 RMSE 相同")
    
    if 'MAE' in comparison_results:
        mae_improvement = comparison_results['MAE']['improvement']
        if mae_improvement > 0:
            print(f"✅ {name2} 的 MAE 比 {name1} 改善了 {mae_improvement:.2f}%")
        elif mae_improvement < 0:
            print(f"⚠️  {name2} 的 MAE 比 {name1} 差了 {abs(mae_improvement):.2f}%")
        else:
            print(f"🤝 两个报告的 MAE 相同")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='比较两个误差报告')
    parser.add_argument('report1', help='第一个误差报告文件路径')
    parser.add_argument('report2', help='第二个误差报告文件路径')
    parser.add_argument('--name1', default='报告1', help='第一个报告的名称')
    parser.add_argument('--name2', default='报告2', help='第二个报告的名称')
    
    args = parser.parse_args()
    
    compare_error_reports(args.report1, args.report2, args.name1, args.name2)

