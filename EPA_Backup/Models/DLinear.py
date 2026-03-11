# models_dlinear.py
import torch, torch.nn as nn

class DLinear(nn.Module):
    def __init__(self, lookback, in_features, horizon=1):
        super().__init__()
        # 展平后走一层线性： (L*F) -> horizon
        self.proj = nn.Linear(lookback*in_features, horizon)

    def forward(self, x):  # x: (B, L, F)
        B, L, F = x.shape
        return self.proj(x.reshape(B, L*F))