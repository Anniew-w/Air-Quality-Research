# models_pgn.py
import torch, torch.nn as nn, torch.nn.functional as F

class PGN(nn.Module):
    def __init__(self, lookback, in_features, horizon=1, hidden=64):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Flatten(),                      # (B,L,F)->(B,L*F)
            nn.Linear(lookback*in_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, horizon)
        )
        self.wind_idx = None  # 训练时外部告诉它风速列的下标（用于物理正则）

    def forward(self, x):  # (B,L,F)
        return self.backbone(x)

    def physics_penalty(self, x, y_hat, feat_names, lam=1e-3):
        pen = 0.0
        # 1) 非负性约束
        pen += F.relu(-y_hat).mean()
        # 2) 风速负相关（示例）：最后一步的风速越大，预测应越低
        if "Avg_10m_Wind_Speed" in feat_names:
            idx = feat_names.index("Avg_10m_Wind_Speed")
            wind_last = x[:, -1, idx]   # (B,)
            pen += torch.cov(torch.stack([wind_last, y_hat.squeeze()], dim=0))[0,1].clamp(min=0)  # 惩罚正相关
        return lam * pen