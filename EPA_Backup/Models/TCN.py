# models_tcn.py
import torch, torch.nn as nn

class Chomp1d(nn.Module):
    def __init__(self, chomp_size): super().__init__(); self.chomp_size=chomp_size
    def forward(self, x): return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, k, d, dropout):
        super().__init__()
        pad = (k-1) * d
        self.net = nn.Sequential(
            nn.Conv1d(n_inputs, n_outputs, k, padding=pad, dilation=d),
            Chomp1d(pad), nn.ReLU(),
            nn.Conv1d(n_outputs, n_outputs, k, padding=pad, dilation=d),
            Chomp1d(pad), nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs!=n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TCN(nn.Module):
    def __init__(self, in_features, channels=(32,32,32), k=3, dropout=0.1, horizon=1):
        super().__init__()
        layers, d = [], 1
        c_prev = in_features
        for c in channels:
            layers += [TemporalBlock(c_prev, c, k, d, dropout)]
            c_prev = c; d *= 2
        self.tcn = nn.Sequential(*layers)
        self.head = nn.Linear(c_prev, horizon)

    def forward(self, x):           # x:(B,L,F) -> 需要转成 (B,F,L)
        x = x.transpose(1,2)        # (B,F,L)
        y = self.tcn(x)             # (B,C,L)
        last = y[:, :, -1]          # 取最后一步
        return self.head(last)      # (B,horizon)