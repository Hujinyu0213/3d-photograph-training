import os
import numpy as np
import random
import glob

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EXPORT_ROOT = os.path.join(ROOT_DIR, "data", "pointcloud")
PROJECTS_LIST_FILE = os.path.join(ROOT_DIR, "results", "valid_projects.txt")
LABELS_FILE = os.path.join(ROOT_DIR, "results", "labels.csv")

with open(PROJECTS_LIST_FILE, "r", encoding="utf-8") as f:
    projects = [ln.strip() for ln in f if ln.strip()]

if not projects:
    raise RuntimeError("No valid projects found in valid_projects.txt")

name = random.choice(projects)
project_index = projects.index(name)
pc_path = os.path.join(EXPORT_ROOT, name, "pointcloud_full.npy")
if not os.path.exists(pc_path):
    raise FileNotFoundError(f"Missing pointcloud: {pc_path}")

pc = np.load(pc_path).astype(np.float32)
if pc.shape[0] == 0:
    raise RuntimeError(f"Empty point cloud for project: {name}")

landmark_npy = None
landmark_npy_candidates = glob.glob(os.path.join(EXPORT_ROOT, name, "*landmarks*.npy"))
if landmark_npy_candidates:
    landmark_npy = landmark_npy_candidates[0]

labels = np.loadtxt(LABELS_FILE, delimiter=",", dtype=np.float32)
if labels.ndim == 1:
    labels = labels.reshape(1, -1)
if project_index >= labels.shape[0]:
    raise RuntimeError(f"labels.csv has {labels.shape[0]} rows, but project index is {project_index}")
lm = labels[project_index].reshape(-1, 3)

bb_min = pc.min(axis=0)
bb_max = pc.max(axis=0)
size = bb_max - bb_min
width = size[0]
height = size[1]
depth = size[2]
diagonal = np.linalg.norm(size)

lm_min = lm.min(axis=0)
lm_max = lm.max(axis=0)
lm_size = lm_max - lm_min
lm_diag = np.linalg.norm(lm_size)

if landmark_npy:
    lm_npy = np.load(landmark_npy).astype(np.float32)
    lm_npy = lm_npy.reshape(-1, 3)
    lm_npy_min = lm_npy.min(axis=0)
    lm_npy_max = lm_npy.max(axis=0)
    lm_npy_size = lm_npy_max - lm_npy_min
    lm_npy_diag = np.linalg.norm(lm_npy_size)

print(f"Project: {name}")
print(f"Point count: {pc.shape[0]}")
print(f"BBox min: {bb_min}")
print(f"BBox max: {bb_max}")
print(f"BBox size (width, height, depth): {size}")
print(f"Width (x): {width}")
print(f"Height (y): {height}")
print(f"Depth (z): {depth}")
print(f"Diagonal length: {diagonal}")

print("\nLandmarks (from labels.csv):")
print(f"Landmark count: {lm.shape[0]}")
print(f"LM min: {lm_min}")
print(f"LM max: {lm_max}")
print(f"LM size (width, height, depth): {lm_size}")
print(f"LM diagonal length: {lm_diag}")

if landmark_npy:
    print("\nLandmarks (from .npy file):")
    print(f"Landmark file: {os.path.basename(landmark_npy)}")
    print(f"Landmark count: {lm_npy.shape[0]}")
    print(f"LM npy min: {lm_npy_min}")
    print(f"LM npy max: {lm_npy_max}")
    print(f"LM npy size (width, height, depth): {lm_npy_size}")
    print(f"LM npy diagonal length: {lm_npy_diag}")
