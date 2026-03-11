import torch
import torch.nn as nn
import torch.nn.functional as F


class Chomp1d(nn.Module):
    """
    去掉 1D 卷积右边多出来的 padding，保证因果性（未来不能看到）。
    """
    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        # x: [B, C, L + padding]
        if self.chomp_size == 0:
            return x
        return x[..., :-self.chomp_size]


class GatedTemporalBlock(nn.Module):
    """
    一个带 gating 的时间卷积残差块：
      - 两条卷积分支：filter 分支 + gate 分支
      - out = tanh(filter) * sigmoid(gate)
      - 残差连接保证梯度流稳定
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float
    ):
        super().__init__()

        padding = (kernel_size - 1) * dilation

        # filter 分支
        self.conv_f = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
        )
        self.chomp_f = Chomp1d(padding)

        # gate 分支
        self.conv_g = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
        )
        self.chomp_g = Chomp1d(padding)

        self.dropout = nn.Dropout(dropout)

        # 如果 in/out 维度不一致，用 1x1 做 residual 对齐
        self.downsample = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else None
        )
        self.final_relu = nn.ReLU()

        self._init_weights()

    def _init_weights(self):
        for m in [self.conv_f, self.conv_g]:
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        if self.downsample is not None:
            nn.init.kaiming_normal_(self.downsample.weight, nonlinearity="relu")
            if self.downsample.bias is not None:
                nn.init.zeros_(self.downsample.bias)

    def forward(self, x):
        # x: [B, C_in, L]
        f = self.conv_f(x)
        f = self.chomp_f(f)

        g = self.conv_g(x)
        g = self.chomp_g(g)

        # gated activation
        out = torch.tanh(f) * torch.sigmoid(g)
        out = self.dropout(out)

        # 残差连接
        res = x if self.downsample is None else self.downsample(x)
        out = out + res
        out = self.final_relu(out)
        return out  # [B, C_out, L]


class PGN(nn.Module):
    """
    PGN: 一组堆叠的 GatedTemporalBlock，最后用全连接预测多步 horizon。
    
    在 pgn_running.py 中的用法：
        model = PGN(
            in_features=X.shape[2],
            channels=(64, 64, 64, 64, 64, 64),
            kernel_size=3,
            dropout=0.3,
            horizon=HORIZON
        )

    输入:
        x: [batch_size, lookback, in_features]
    输出:
        y_hat: [batch_size, horizon]
    """
    def __init__(
        self,
        in_features: int,
        channels=(64, 64, 64, 64, 64, 64),
        kernel_size: int = 3,
        dropout: float = 0.3,
        horizon: int = 14,
    ):
        super().__init__()

        layers = []
        num_levels = len(channels)
        prev_channels = in_features

        for i in range(num_levels):
            out_channels = channels[i]
            dilation = 2 ** i  # 1, 2, 4, 8, ...
            block = GatedTemporalBlock(
                in_channels=prev_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                dilation=dilation,
                dropout=dropout,
            )
            layers.append(block)
            prev_channels = out_channels

        self.network = nn.Sequential(*layers)

        # 最后用最后一个时间步的特征预测 horizon 维输出
        self.fc = nn.Linear(prev_channels, horizon)

    def forward(self, x):
        """
        x: [B, T, F]
        return: [B, horizon]
        """
        # Conv1d 期望输入 [B, C, L]，这里把特征维当 C，时间维当 L
        # [B, T, F] -> [B, F, T]
        x = x.permute(0, 2, 1)

        out = self.network(x)  # [B, C_last, T]

        # 取最后一个时间步（因果卷积，最后一步聚合了所有历史）
        last_step = out[:, :, -1]  # [B, C_last]

        y_hat = self.fc(last_step)  # [B, horizon]
        return y_hat