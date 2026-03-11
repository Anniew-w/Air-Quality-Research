# Flower_Client.py
import argparse, os, numpy as np, torch, torch.nn as nn
import flwr as fl
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from Models.data_preprocessing import load_and_preprocess_data
from Models.TCN import TCN
from Models.DLinear import DLinear
from Models.PGN import PGN

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

def make_model(name, lookback, in_features, horizon):
    name = name.lower()
    if name == "tcn":
        return TCN(in_features=in_features, channels=(64, 64, 64), k=3, dropout=0.2, horizon=horizon).to(DEVICE)
    if name == "dlinear":
        # 改进的DLinear
        class ImprovedDLinear(nn.Module):
            def __init__(self, lookback, in_features, horizon=1):
                super().__init__()
                self.network = nn.Sequential(
                    nn.Linear(lookback * in_features, 128),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.Linear(64, horizon)
                )
            def forward(self, x):
                B, L, F = x.shape
                return self.network(x.reshape(B, -1))
        return ImprovedDLinear(lookback, in_features, horizon).to(DEVICE)
    if name == "pgn":
        return PGN(lookback=lookback, in_features=in_features, horizon=horizon).to(DEVICE)
    raise ValueError("unknown model")

class EPAClient(fl.client.NumPyClient):
    def __init__(self, location_id, model_name, lookback=14, horizon=1):
        try:
            self.location_id = location_id  # 添加这行！
            
            # 使用新的预处理函数
            X, y, feats = load_and_preprocess_data(
                location_id, 
                mapping_csv="epa_to_bom.csv",
                epa_dir="clients_epa_daily", 
                bom_dir="Cleaned_BOM",
                target_col="SBPM25", 
                lookback=lookback, 
                horizon=horizon
            )
            
            if len(X) == 0:
                raise ValueError("No data available after preprocessing.")
            
            # 时间序列划分（不要shuffle！）
            n = len(X)
            split_idx = int(n * 0.8)
            self.Xtr, self.ytr = X[:split_idx], y[:split_idx]
            self.Xte, self.yte = X[split_idx:], y[split_idx:]
            
            # 使用RobustScaler处理异常值
            self.scaler_x = RobustScaler()
            self.scaler_y = StandardScaler()
            
            # 缩放特征
            self.Xtr = self.scaler_x.fit_transform(self.Xtr.reshape(-1, X.shape[-1])).reshape(self.Xtr.shape)
            self.Xte = self.scaler_x.transform(self.Xte.reshape(-1, X.shape[-1])).reshape(self.Xte.shape)
            
            # 缩放目标
            self.ytr = self.scaler_y.fit_transform(self.ytr.reshape(-1, 1)).flatten()
            self.yte = self.scaler_y.transform(self.yte.reshape(-1, 1)).flatten()
            
            self.feat_names = feats
            in_features = self.Xtr.shape[-1]
            self.lookback = self.Xtr.shape[1] if len(self.Xtr.shape) > 1 else 1
            
            self.model = make_model(model_name, self.lookback, in_features, horizon)
            self.opt = torch.optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-5)
            self.crit = nn.MSELoss()
            self.model_name = model_name.lower()
            
            print(f"Client {location_id} initialized: {len(self.Xtr)} train, {len(self.Xte)} test samples")
            
        except Exception as e:
            print(f"Failed to initialize client for location {location_id}: {e}")
            raise

    def _prep(self, X):
        """准备数据"""
        if len(X.shape) == 2:
            X = X.reshape(X.shape[0], 1, X.shape[1])
        return torch.tensor(X, dtype=torch.float32, device=DEVICE)

    def get_parameters(self, config):
        """获取模型参数"""
        return [p.detach().cpu().numpy() for p in self.model.state_dict().values()]

    def set_parameters(self, params):
        """设置模型参数"""
        sd = self.model.state_dict()
        for k, v in zip(sd.keys(), params):
            if k in sd and sd[k].shape == v.shape:
                sd[k] = torch.tensor(v)
        self.model.load_state_dict(sd)

    def fit(self, parameters, config):
        """本地训练"""
        self.set_parameters(parameters)
        self.model.train()
        
        X = self._prep(self.Xtr)
        y = torch.tensor(self.ytr, dtype=torch.float32, device=DEVICE)
        
        # 多轮本地训练
        local_epochs = config.get("local_epochs", 3)
        
        for epoch in range(local_epochs):
            self.opt.zero_grad()
            yhat = self.model(X)
            
            # 确保输出非负（PM2.5不能为负）
            yhat = torch.relu(yhat)
            
            loss = self.crit(yhat.squeeze(), y)
            
            # PGN 物理正则
            if self.model_name == "pgn":
                loss = loss + self.model.physics_penalty(X, yhat, self.feat_names, lam=1e-3)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)  # 梯度裁剪
            self.opt.step()
        
        new_params = self.get_parameters(config)
        return new_params, len(self.Xtr), {"train_loss": float(loss.item())}

    def evaluate(self, parameters, config):
        """本地评估"""
        self.set_parameters(parameters)
        self.model.eval()
        
        X = self._prep(self.Xte)
        y = torch.tensor(self.yte, dtype=torch.float32, device=DEVICE)
        
        with torch.no_grad():
            yhat = self.model(X)
            yhat = torch.relu(yhat)  # 确保非负
            
            # 反标准化得到原始尺度的预测
            yhat_original = self.scaler_y.inverse_transform(
                yhat.cpu().numpy().reshape(-1, 1)
            ).flatten()
            y_original = self.scaler_y.inverse_transform(
                y.cpu().numpy().reshape(-1, 1)
            ).flatten()
            
            mse = mean_squared_error(y_original, yhat_original)
            mae = mean_absolute_error(y_original, yhat_original)
            r2 = r2_score(y_original, yhat_original)
        
        return float(mse), len(self.Xte), {
            "mse": float(mse), 
            "mae": float(mae), 
            "r2": float(r2),
            "location_id": self.location_id  # 现在这个属性存在了
        }

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--location_id", required=True, help="EPA location ID")
    ap.add_argument("--server", default="0.0.0.0:8080", help="Server address")
    ap.add_argument("--model", default="dlinear", choices=["tcn","dlinear","pgn"], help="Model type")
    ap.add_argument("--lookback", type=int, default=14, help="Lookback window size")
    args = ap.parse_args()

    client = EPAClient(args.location_id, args.model, lookback=args.lookback, horizon=1)
    
    # 使用新的API调用方式
    fl.client.start_client(
        server_address=args.server,
        client=client.to_client(),
    )