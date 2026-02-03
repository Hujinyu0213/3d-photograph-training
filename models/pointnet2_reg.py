import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

# Prefer the packaged pointnet2_ops modules; fall back to any legacy utils path if present.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
UTILS_DIR = os.path.join(ROOT_DIR, "scripts", "utils")
if UTILS_DIR not in sys.path:
    sys.path.insert(0, UTILS_DIR)

try:
    from pointnet2_ops.pointnet2_modules import (
        PointnetSAModuleMSG as PointNetSetAbstractionMsg,
        PointnetSAModule as PointNetSetAbstraction,
    )
except ImportError:
    # Fallback: if packaged import fails, keep previous search path behavior
    from pointnet2_utils import PointNetSetAbstractionMsg, PointNetSetAbstraction

class PointNet2RegMSG(nn.Module):
    """
    PointNet++ MSG 回归头，输出 27 维（9 个地标 * 3）。
    输入: (B, 3, N) 或 (B, 6, N) 如果包含法线
    """
    def __init__(
        self,
        output_dim=27,
        normal_channel=True,
        dropout=0.35,
        sa1_radii=None,
        sa2_radii=None,
        sa1_nsamples=None,
        sa2_nsamples=None,
    ):
        super().__init__()
        in_channel = 3 if normal_channel else 0
        self.normal_channel = normal_channel
        sa1_radii = sa1_radii or [0.1, 0.2, 0.4]
        sa2_radii = sa2_radii or [0.2, 0.4, 0.8]
        sa1_nsamples = sa1_nsamples or [16, 32, 128]
        sa2_nsamples = sa2_nsamples or [32, 64, 128]
        # PointnetSAModuleMSG expects mlps with input channel as first element; when use_xyz=True it adds 3 internally.
        self.sa1 = PointNetSetAbstractionMsg(
            npoint=512,
            radii=sa1_radii,
            nsamples=sa1_nsamples,
            mlps=[
                [in_channel, 32, 32, 64],
                [in_channel, 64, 64, 128],
                [in_channel, 64, 96, 128],
            ],
            use_xyz=True,
        )
        # After sa1, feature dims = 64 + 128 + 128 = 320
        self.sa2 = PointNetSetAbstractionMsg(
            npoint=128,
            radii=sa2_radii,
            nsamples=sa2_nsamples,
            mlps=[
                [320, 64, 64, 128],
                [320, 128, 128, 256],
                [320, 128, 128, 256],
            ],
            use_xyz=True,
        )
        # After sa2, feature dims = 128 + 256 + 256 = 640
        self.sa3 = PointNetSetAbstraction(
            npoint=None,
            radius=None,
            nsample=None,
            mlp=[640, 256, 512, 1024],
            use_xyz=True,
        )
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(256, output_dim)

    def forward(self, xyz):
        # xyz: (B, C, N)
        B, _, _ = xyz.shape
        if self.normal_channel and xyz.shape[1] >= 6:
            norm = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
        else:
            norm = None
        # pointnet2_ops expects xyz as (B, N, 3) and features as (B, C, N)
        l1_xyz, l1_points = self.sa1(
            xyz.transpose(1, 2), norm.transpose(1, 2) if norm is not None else None
        )
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        x = l3_points.view(B, 1024)
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        return x
