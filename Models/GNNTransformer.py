import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SimpleGCNLayer(nn.Module):
    """
    简单 GCN 层：
    X' = Â X W
    这里 Â 是加入自环后的归一化邻接矩阵。
    """
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x, adj):
        """
        x: [B, N, F]
        adj: [N, N] 未归一化邻接矩阵
        """
        B, N, F = x.shape

        # 构造 Â = D^{-1/2} (A + I) D^{-1/2}
        device = x.device
        A = adj.to(device)
        I = torch.eye(N, device=device)
        A_hat = A + I
        deg = A_hat.sum(dim=-1)  # [N]
        deg_inv_sqrt = torch.pow(deg + 1e-8, -0.5)
        D_inv_sqrt = torch.diag(deg_inv_sqrt)
        A_norm = D_inv_sqrt @ A_hat @ D_inv_sqrt  # [N, N]

        # [B, N, F] -> [B, N, out_dim]
        x = self.lin(x)          # [B, N, out_dim]
        x = torch.matmul(A_norm, x)  # [N, N] x [B, N, out_dim] -> [B, N, out_dim]
        return x


class PositionalEncoding(nn.Module):
    """
    标准 Transformer 位置编码（支持 batch_first=True）。
    """
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)  # [T, D]
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, T, D]
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        x: [B, T, D]
        """
        T = x.size(1)
        x = x + self.pe[:, :T, :]
        return x


class GNNTemporalTransformer(nn.Module):
    """
    简单 GNN + Transformer baseline.

    输入:
        x: [B, T, N, F]   (batch, lookback, num_nodes, node_features)
    输出:
        y_hat: [B, horizon]   (预测一个目标序列，例如主站点 PM2.5)

    对于单站点 N=1 的情况：
        就是 1 个节点的 GNN（退化成 MLP），
        但结构已经支持未来扩展多站点。
    """
    def __init__(
        self,
        num_nodes: int,
        in_features: int,
        gnn_hidden_dim: int = 64,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        horizon: int = 14,
        dropout: float = 0.1,
        adj_matrix: torch.Tensor = None,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.in_features = in_features
        self.horizon = horizon

        # 如果没提供邻接矩阵，默认用全连接图（包括自环），单站点时就是 [1,1]=1
        if adj_matrix is None:
            adj_matrix = torch.ones(num_nodes, num_nodes, dtype=torch.float32)
        self.register_buffer("adj", adj_matrix)

        # GNN 部分：两层简单 GCN
        self.gcn1 = SimpleGCNLayer(in_features, gnn_hidden_dim)
        self.gcn2 = SimpleGCNLayer(gnn_hidden_dim, gnn_hidden_dim)

        # 时间 Transformer 部分
        self.input_proj = nn.Linear(gnn_hidden_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,  # 用 [B, T, D]
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        # 输出层：用最后一个 time step 的表示预测 horizon 步
        self.fc_out = nn.Linear(d_model, horizon)

    def forward(self, x):
        """
        x: [B, T, N, F]
        """
        B, T, N, Fdim = x.shape          # ✅ 改名
        assert N == self.num_nodes
        assert Fdim == self.in_features

        # 合并 time 维，逐时间步做 GNN：
        # [B, T, N, F] -> [B*T, N, F]
        x_bt = x.reshape(B * T, N, Fdim)

        # 两层 GCN
        h = F.relu(self.gcn1(x_bt, self.adj))
        h = F.relu(self.gcn2(h, self.adj))   # [B*T, N, gnn_hidden_dim]

        # 节点平均池化
        h = h.mean(dim=1)                    # [B*T, H]

        # reshape 回 [B, T, H]
        h = h.reshape(B, T, -1)

        # Transformer + 输出
        h = self.input_proj(h)               # [B, T, d_model]
        h = self.pos_encoder(h)
        h_enc = self.transformer(h)          # [B, T, d_model]
        last = h_enc[:, -1, :]
        y_hat = self.fc_out(last)            # [B, horizon]
        return y_hat