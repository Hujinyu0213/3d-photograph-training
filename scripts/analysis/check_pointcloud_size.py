"""
检查点云大小统计信息
用于确定最佳的MAX_POINTS设置
"""
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'pointcloud')

if not os.path.exists(DATA_DIR):
    print(f"❌ 数据目录不存在: {DATA_DIR}")
    exit(1)

point_counts = []
valid_projects = []

print("正在检查点云大小...")
for project_name in os.listdir(DATA_DIR):
    project_dir = os.path.join(DATA_DIR, project_name)
    if not os.path.isdir(project_dir):
        continue
    
    npy_file = os.path.join(project_dir, "pointcloud_full.npy")
    if os.path.exists(npy_file):
        try:
            pc = np.load(npy_file)
            point_counts.append(len(pc))
            valid_projects.append(project_name)
        except Exception as e:
            print(f"⚠️  跳过 {project_name}: {e}")

if not point_counts:
    print("❌ 未找到任何点云文件")
    exit(1)

print(f"\n{'='*60}")
print(f"点云数量统计（共 {len(point_counts)} 个样本）")
print(f"{'='*60}")
print(f"  最小: {min(point_counts):,} 个点")
print(f"  最大: {max(point_counts):,} 个点")
print(f"  平均: {np.mean(point_counts):.0f} 个点")
print(f"  中位数: {np.median(point_counts):.0f} 个点")
print(f"  标准差: {np.std(point_counts):.0f} 个点")

# 统计分布
print(f"\n点云大小分布:")
ranges = [
    (0, 1000, "< 1000点"),
    (1000, 2048, "1000-2048点"),
    (2048, 5000, "2048-5000点"),
    (5000, 10000, "5000-10000点"),
    (10000, 20000, "10000-20000点"),
    (20000, float('inf'), "> 20000点")
]

for min_val, max_val, label in ranges:
    count = sum(1 for c in point_counts if min_val <= c < max_val)
    percentage = count / len(point_counts) * 100
    print(f"  {label:20s}: {count:3d} 个样本 ({percentage:5.1f}%)")

# 推荐设置
print(f"\n{'='*60}")
print("推荐设置:")
print(f"{'='*60}")

avg_points = np.mean(point_counts)
median_points = np.median(point_counts)

if avg_points < 1000:
    print("✅ 推荐: MAX_POINTS = 使用所有点（不采样）")
    print("   原因: 点数较少，不需要采样")
elif avg_points < 5000:
    print("✅ 推荐: MAX_POINTS = 2048（当前设置）")
    print("   原因: 点云大小适中，2048点是PointNet标准配置")
elif avg_points < 15000:
    print("⚠️  推荐: MAX_POINTS = 4096")
    print("   原因: 点云较大，建议增加采样点数以保留更多细节")
    print("   注意: 可能需要减少 BATCH_SIZE 到 8")
else:
    print("⚠️  推荐: MAX_POINTS = 8192")
    print("   原因: 点云很大，需要更多点保留细节")
    print("   注意: 需要减少 BATCH_SIZE 到 4-8，并确保GPU内存充足")

print(f"\n当前设置: MAX_POINTS = 2048")
if avg_points < 5000:
    print("✅ 当前设置适合您的数据！")
else:
    print("⚠️  建议根据上述推荐调整 MAX_POINTS")
