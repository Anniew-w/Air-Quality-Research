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


class TemporalBlock(nn.Module):
    """
    TCN 里的基本残差块：
      Conv1d (dilated, causal) -> ReLU -> Dropout
      Conv1d (dilated, causal) -> ReLU -> Dropout
      + 残差连接
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

        # 为了实现因果卷积：padding = (kernel_size - 1) * dilation
        padding = (kernel_size - 1) * dilation

        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
        )
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
        )
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(dropout)

        # 如果 in/out 维度不一样，用 1x1 卷积把 residual 对齐
        self.downsample = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else None
        )
        self.final_relu = nn.ReLU()

        # Kaiming 初始化稍微好一点
        self._init_weights()

    def _init_weights(self):
        for m in [self.conv1, self.conv2]:
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        if self.downsample is not None:
            nn.init.kaiming_normal_(self.downsample.weight, nonlinearity="relu")
            if self.downsample.bias is not None:
                nn.init.zeros_(self.downsample.bias)

    def forward(self, x):
        # x: [B, C_in, L]
        out = self.conv1(x)
        out = self.chomp1(out)
        out = self.relu1(out)
        out = self.drop1(out)

        out = self.conv2(out)
        out = self.chomp2(out)
        out = self.relu2(out)
        out = self.drop2(out)

        # 残差
        res = x if self.downsample is None else self.downsample(x)
        out = out + res
        out = self.final_relu(out)
        return out  # [B, C_out, L]


class TCN(nn.Module):
    """
    时序卷积网络，用在 tcn_running.py 里：
      - 输入:  X shape = [B, T, in_features]
      - 输出:  y_pred shape = [B, horizon]
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

        # 堆多层 dilated causal conv
        for i in range(num_levels):
            out_channels = channels[i]
            dilation = 2 ** i  # 1, 2, 4, 8, ...
            block = TemporalBlock(
                in_channels=prev_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                dilation=dilation,
                dropout=dropout,
            )
            layers.append(block)
            prev_channels = out_channels

        self.network = nn.Sequential(*layers)

        # 最后把最后一个时间步的特征 -> horizon 维的输出（多步预测）
        self.fc = nn.Linear(prev_channels, horizon)

    def forward(self, x):
        """
        x: [B, T, in_features]
        return: [B, horizon]
        """
        # Conv1d 的输入是 [B, C, L]，这里把特征维当 channel，时间维当长度
        # x: [B, T, F] -> [B, F, T]
        x = x.permute(0, 2, 1)

        out = self.network(x)  # [B, C_last, T]

        # 取最后一个时间步的表征（因为是因果卷积，最后一步包含过去全部信息）
        last_step = out[:, :, -1]  # [B, C_last]

        # 映射到 horizon 维
        y_hat = self.fc(last_step)  # [B, horizon]
        return y_hat