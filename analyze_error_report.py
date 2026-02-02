"""
误差报告分析工具
详细分析 error_report.csv，识别哪些地标点预测精确，哪些不精确
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

# 地标点名称
LANDMARK_NAMES = ['Glabella', 'Nasion', 'Rhinion', 'Nasal Tip', 'Subnasale', 
                  'Alare (R)', 'Alare (L)', 'Zygion (R)', 'Zygion (L)']

def analyze_error_report(report_path, output_name="分析结果"):
    """
    分析误差报告
    
    参数:
        report_path: 误差报告文件路径
        output_name: 输出名称（用于保存文件）
    """
    print("=" * 70)
    print("📊 误差报告详细分析")
    print("=" * 70)
    
    # 读取报告
    try:
        df = pd.read_csv(report_path)
        print(f"\n✅ 成功加载报告: {report_path}")
        print(f"   样本数: {len(df)}")
        print(f"   列数: {len(df.columns)}")
    except Exception as e:
        print(f"❌ 无法加载报告: {e}")
        return
    
    # 整体统计
    print("\n" + "=" * 70)
    print("📈 整体性能统计")
    print("=" * 70)
    
    if 'RMSE' in df.columns:
        rmse_mean = df['RMSE'].mean()
        rmse_std = df['RMSE'].std()
        rmse_min = df['RMSE'].min()
        rmse_max = df['RMSE'].max()
        rmse_median = df['RMSE'].median()
        
        print(f"\nRMSE (均方根误差):")
        print(f"  平均值: {rmse_mean:.4f} mm")
        print(f"  中位数: {rmse_median:.4f} mm")
        print(f"  标准差: {rmse_std:.4f} mm")
        print(f"  范围: {rmse_min:.4f} - {rmse_max:.4f} mm")
        
        # 性能评级
        if rmse_mean < 2:
            grade = "优秀 ⭐⭐⭐"
        elif rmse_mean < 5:
            grade = "良好 ⭐⭐"
        elif rmse_mean < 10:
            grade = "可接受 ⭐"
        else:
            grade = "需要改进 ⚠️"
        print(f"  性能评级: {grade}")
    
    if 'MAE' in df.columns:
        mae_mean = df['MAE'].mean()
        mae_std = df['MAE'].std()
        mae_min = df['MAE'].min()
        mae_max = df['MAE'].max()
        
        print(f"\nMAE (平均绝对误差):")
        print(f"  平均值: {mae_mean:.4f} mm")
        print(f"  标准差: {mae_std:.4f} mm")
        print(f"  范围: {mae_min:.4f} - {mae_max:.4f} mm")
    
    # 各坐标轴误差分析
    print("\n" + "=" * 70)
    print("📐 各坐标轴误差分析")
    print("=" * 70)
    
    axis_errors = {}
    for axis in ['X_Error', 'Y_Error', 'Z_Error']:
        if axis in df.columns:
            mean_err = df[axis].mean()
            std_err = df[axis].std()
            axis_errors[axis] = {'mean': mean_err, 'std': std_err}
            
            axis_name = axis.replace('_Error', '')
            print(f"\n{axis_name} 轴误差:")
            print(f"  平均误差: {mean_err:.4f} mm")
            print(f"  标准差: {std_err:.4f} mm")
            
            if mean_err == min([axis_errors[k]['mean'] for k in axis_errors.keys()]):
                print(f"  ✅ {axis_name} 轴误差最小（最精确）")
            elif mean_err == max([axis_errors[k]['mean'] for k in axis_errors.keys()]):
                print(f"  ⚠️  {axis_name} 轴误差最大（需要改进）")
    
    # 各地标点误差分析
    print("\n" + "=" * 70)
    print("📍 各地标点误差详细分析")
    print("=" * 70)
    
    landmark_errors = []
    landmark_cols = [col for col in df.columns if col.endswith('_Error') and 
                     any(name in col for name in LANDMARK_NAMES)]
    
    for col in landmark_cols:
        landmark_name = col.replace('_Error', '')
        errors = df[col].values
        
        mean_err = np.mean(errors)
        std_err = np.std(errors)
        min_err = np.min(errors)
        max_err = np.max(errors)
        median_err = np.median(errors)
        
        landmark_errors.append({
            'name': landmark_name,
            'mean': mean_err,
            'std': std_err,
            'min': min_err,
            'max': max_err,
            'median': median_err
        })
    
    # 按平均误差排序
    landmark_errors.sort(key=lambda x: x['mean'])
    
    print("\n地标点精度排名（从最精确到最不精确）:")
    print("-" * 70)
    
    for idx, landmark in enumerate(landmark_errors, 1):
        name = landmark['name']
        mean = landmark['mean']
        std = landmark['std']
        median = landmark['median']
        
        # 精度评级
        if mean < 2:
            grade = "优秀 ⭐⭐⭐"
            status = "✅"
        elif mean < 5:
            grade = "良好 ⭐⭐"
            status = "✅"
        elif mean < 10:
            grade = "可接受 ⭐"
            status = "⚠️"
        else:
            grade = "需要改进 ⚠️"
            status = "❌"
        
        print(f"{idx:2d}. {status} {name:20s}: 平均={mean:6.4f} mm, "
              f"中位数={median:6.4f} mm, 标准差={std:6.4f} mm [{grade}]")
    
    # 最精确和最不精确的地标点
    print("\n" + "=" * 70)
    print("🏆 关键发现")
    print("=" * 70)
    
    if landmark_errors:
        best = landmark_errors[0]
        worst = landmark_errors[-1]
        
        print(f"\n✅ 最精确的地标点: {best['name']}")
        print(f"   平均误差: {best['mean']:.4f} mm")
        print(f"   中位数误差: {best['median']:.4f} mm")
        print(f"   标准差: {best['std']:.4f} mm")
        
        print(f"\n❌ 最不精确的地标点: {worst['name']}")
        print(f"   平均误差: {worst['mean']:.4f} mm")
        print(f"   中位数误差: {worst['median']:.4f} mm")
        print(f"   标准差: {worst['std']:.4f} mm")
        
        improvement_needed = worst['mean'] - best['mean']
        print(f"\n📊 精度差异: {improvement_needed:.4f} mm")
        print(f"   最不精确的点比最精确的点误差大 {improvement_needed:.4f} mm")
    
    # 生成可视化图表
    print("\n" + "=" * 70)
    print("📊 生成分析图表...")
    print("=" * 70)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 图1: 各地标点误差箱线图
    if landmark_errors:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. 各地标点误差箱线图
        ax1 = axes[0, 0]
        names = [l['name'] for l in landmark_errors]
        error_data = []
        error_labels = []
        for name in names:
            col_name = f"{name}_Error"
            if col_name in df.columns:
                error_data.append(df[col_name].values)
                error_labels.append(name)
        
        if error_data:
            bp = ax1.boxplot(error_data, labels=error_labels, patch_artist=True, vert=True)
            colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(bp['boxes'])))
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
            ax1.set_ylabel('Error (mm)', fontsize=10)
            ax1.set_title('Landmark Error Distribution (Boxplot)', fontsize=11)
            ax1.tick_params(axis='x', rotation=45, labelsize=8)
            ax1.grid(True, alpha=0.3)
        
        # 2. 各地标点平均误差柱状图
        ax2 = axes[0, 1]
        means = [l['mean'] for l in landmark_errors]
        colors_bar = ['green' if m < 5 else 'orange' if m < 10 else 'red' for m in means]
        bars = ax2.barh(error_labels, means, color=colors_bar)
        ax2.set_xlabel('Mean Error (mm)', fontsize=10)
        ax2.set_title('Mean Error by Landmark', fontsize=11)
        ax2.grid(True, alpha=0.3, axis='x')
        
        # 添加数值标签
        for i, (bar, mean) in enumerate(zip(bars, means)):
            ax2.text(mean, i, f' {mean:.2f}', va='center', fontsize=8)
        
        # 3. 整体RMSE分布
        ax3 = axes[1, 0]
        if 'RMSE' in df.columns:
            ax3.hist(df['RMSE'].values, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
            mean_val = df['RMSE'].mean()
            median_val = df['RMSE'].median()
            ax3.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}')
            ax3.axvline(median_val, color='green', linestyle='--', linewidth=2, label=f'Median: {median_val:.2f}')
            ax3.set_xlabel('RMSE (mm)', fontsize=10)
            ax3.set_ylabel('Number of Samples', fontsize=10)
            ax3.set_title('RMSE Distribution Histogram', fontsize=11)
            ax3.legend(fontsize=9)
            ax3.grid(True, alpha=0.3)
        
        # 4. 各坐标轴误差比较
        ax4 = axes[1, 1]
        if all(col in df.columns for col in ['X_Error', 'Y_Error', 'Z_Error']):
            axis_data = [df['X_Error'].values, df['Y_Error'].values, df['Z_Error'].values]
            bp4 = ax4.boxplot(axis_data, labels=['X-Axis', 'Y-Axis', 'Z-Axis'], patch_artist=True)
            colors_axis = ['lightblue', 'lightgreen', 'lightcoral']
            for patch, color in zip(bp4['boxes'], colors_axis):
                patch.set_facecolor(color)
            ax4.set_ylabel('Error (mm)', fontsize=10)
            ax4.set_title('Error Comparison by Axis', fontsize=11)
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        chart_path = os.path.join(OUTPUT_DIR, f'{output_name}_analysis.png')
        plt.savefig(chart_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✅ 分析图表已保存到: {chart_path}")
    
    # 保存详细分析报告
    analysis_data = {
        'Landmark': [l['name'] for l in landmark_errors],
        'Mean_Error_mm': [l['mean'] for l in landmark_errors],
        'Median_Error_mm': [l['median'] for l in landmark_errors],
        'Std_Error_mm': [l['std'] for l in landmark_errors],
        'Min_Error_mm': [l['min'] for l in landmark_errors],
        'Max_Error_mm': [l['max'] for l in landmark_errors],
        'Rank': list(range(1, len(landmark_errors) + 1))
    }
    
    analysis_df = pd.DataFrame(analysis_data)
    analysis_csv_path = os.path.join(OUTPUT_DIR, f'{output_name}_detailed_analysis.csv')
    analysis_df.to_csv(analysis_csv_path, index=False)
    print(f"✅ 详细分析报告已保存到: {analysis_csv_path}")
    
    # 总结
    print("\n" + "=" * 70)
    print("📋 分析总结")
    print("=" * 70)
    
    if landmark_errors:
        excellent = [l for l in landmark_errors if l['mean'] < 2]
        good = [l for l in landmark_errors if 2 <= l['mean'] < 5]
        acceptable = [l for l in landmark_errors if 5 <= l['mean'] < 10]
        poor = [l for l in landmark_errors if l['mean'] >= 10]
        
        print(f"\n精度分布:")
        print(f"  优秀 (误差 < 2mm): {len(excellent)} 个地标点")
        if excellent:
            print(f"    - {', '.join([l['name'] for l in excellent])}")
        
        print(f"  良好 (误差 2-5mm): {len(good)} 个地标点")
        if good:
            print(f"    - {', '.join([l['name'] for l in good])}")
        
        print(f"  可接受 (误差 5-10mm): {len(acceptable)} 个地标点")
        if acceptable:
            print(f"    - {', '.join([l['name'] for l in acceptable])}")
        
        print(f"  需要改进 (误差 ≥ 10mm): {len(poor)} 个地标点")
        if poor:
            print(f"    - {', '.join([l['name'] for l in poor])}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='分析误差报告')
    parser.add_argument('report', help='误差报告文件路径')
    parser.add_argument('--name', default='分析结果', help='输出文件名称')
    
    args = parser.parse_args()
    
    analyze_error_report(args.report, args.name)

