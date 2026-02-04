import trimatic
import csv
import os
import numpy as np

# =========================================================
# 配置
# =========================================================
EXPORT_ROOT = r"\\uz\data\Admin\mka\results\hou-and-hu\vs\pointcloud"

# ① 编号型 landmark（只看前缀，顺序即 label 顺序）
LANDMARK_PREFIXES = [
    "037", "038", "039", "040", "041", "042",
    "043", "044", "045", "046", "047", "048"
]

# ② 解剖学 landmark（名字固定）
ANATOMICAL_LANDMARKS = [
    "Glabella", "Nasion", "Rhinion", "Nasal Tip", "Subnasale",
    "Alare (R)", "Alare (L)", "Zygion (R)", "Zygion (L)"
]

# mesh Part 的稳定标识
MESH_KEYWORD = "_Tricorder_object"

# =========================================================
# 工具函数
# =========================================================
def find_point_by_prefix(prefix):
    """查找 037_xxxx 这类 landmark"""
    for p in trimatic.get_points():
        if p.name.startswith(prefix + "_"):
            return p
    return None

def find_point_by_name(name):
    """查找 Glabella / Nasion 等"""
    return trimatic.find_point(name)

# =========================================================
# 项目名
# =========================================================
proj_path = trimatic.get_project_filename()
project_name = os.path.splitext(os.path.basename(proj_path))[0]
print("Processing project:", project_name)

export_dir = os.path.join(EXPORT_ROOT, project_name)
os.makedirs(export_dir, exist_ok=True)

# =========================================================
# 1️⃣ 自动查找 mesh Part
# =========================================================
mesh_part = None
for part in trimatic.get_parts():
    if MESH_KEYWORD in part.name:
        mesh_part = part
        break

if mesh_part is None:
    raise RuntimeError(f"❌ 未找到包含 '{MESH_KEYWORD}' 的 mesh Part")

print("Using mesh Part:", mesh_part.name)

# =========================================================
# 2️⃣ 提取点云（triangle → point cloud）
# =========================================================
triangles = mesh_part.get_triangles()
if len(triangles) == 0:
    raise RuntimeError("❌ mesh Part 中没有 triangles")

points = []

for tri in triangles:              # tri = ((x1,y1,z1),(x2,y2,z2),(x3,y3,z3))
    for p in tri:                  # p = (x, y, z)
        points.append([p[0], p[1], p[2]])

pointcloud = np.array(points)
print("Raw point cloud shape:", pointcloud.shape)

# 去重（推荐）
pointcloud = np.unique(pointcloud, axis=0)
print("Unique point cloud shape:", pointcloud.shape)

np.save(os.path.join(export_dir, "pointcloud_full.npy"), pointcloud)

# =========================================================
# 3️⃣ 导出 landmarks（两类，顺序固定）
# =========================================================
landmarks = []

# --- 3.1 编号型 landmarks（037–048） ---
for prefix in LANDMARK_PREFIXES:
    pt = find_point_by_prefix(prefix)
    if pt is None:
        raise RuntimeError(f"❌ 未找到 landmark: {prefix}_xxxx")

    x, y, z = pt.coordinates
    print(f"Landmark {pt.name}: {x}, {y}, {z}")
    landmarks.append([project_name, prefix, x, y, z])

# --- 3.2 解剖学 landmarks ---
for name in ANATOMICAL_LANDMARKS:
    pt = find_point_by_name(name)
    if pt is None:
        raise RuntimeError(f"❌ 未找到 landmark: {name}")

    x, y, z = pt.coordinates
    print(f"Landmark {name}: {x}, {y}, {z}")
    landmarks.append([project_name, name, x, y, z])

print("Total landmarks:", len(landmarks))  # 应为 21

# =========================================================
# 保存 CSV（检查用）
# =========================================================
csv_path = os.path.join(export_dir, "nose_landmarks.csv")
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["project", "landmark", "x", "y", "z"])
    writer.writerows(landmarks)

# =========================================================
# 保存 NPY（PointNet label）
# =========================================================
coords = np.array([[x, y, z] for (_, _, x, y, z) in landmarks])
np.save(os.path.join(export_dir, "nose_landmarks.npy"), coords)

# =========================================================
# 完成
# =========================================================
print("Saved files to:", export_dir)
print(" - pointcloud_full.npy")
print(" - nose_landmarks.npy")
print(" - nose_landmarks.csv")
print("✅ Export finished successfully")
