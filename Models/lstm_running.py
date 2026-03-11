import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pathlib import Path


# ========== 参数 ==========
LOOKBACK = 14
HORIZON = 1
EPOCHS = 100
LR = 3e-4
TEST_SIZE = 0.2
BATCH_SIZE = 64

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs("results", exist_ok=True)

DEFAULT_DATA_PATH = (
    Path(__file__).resolve().parents[1]
    / "Finalized_Dataset"
    / "Melbourne_CBD_10239_merged.csv"
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train the LSTM model on one station CSV.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH, help="Path to the station CSV file.")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Number of training epochs.")
    parser.add_argument("--name", type=str, default=None, help="Optional run name for saved plots.")
    return parser.parse_args()


# =========== 滑动窗口 ==========
def create_sliding_windows(features, target, lookback, horizon, dates=None):
    """FreEformer-style windowing: drop windows containing NaNs, ignore date gaps."""
    X, y = [], []
    end_dates = []
    n = len(features)
    win_len = lookback + horizon
    if n < win_len:
        return np.array([]), np.array([]), pd.Series(dtype='datetime64[ns]')

    if dates is not None:
        dates = pd.to_datetime(dates)
        if not isinstance(dates, pd.Series):
            dates = pd.Series(dates)

    valid_row = ~np.isnan(features).any(axis=1)
    valid_row &= ~np.isnan(target)
    valid_window = np.convolve(valid_row.astype(int), np.ones(win_len, dtype=int), mode='valid') == win_len
    start_indices = np.where(valid_window)[0]

    for idx in start_indices:
        X.append(features[idx: idx + lookback])
        y.append(target[idx + lookback: idx + lookback + horizon])
        if dates is not None:
            end_dates.append(dates.iloc[idx + win_len - 1])

    X = np.array(X)
    y = np.array(y)
    end_dates = pd.to_datetime(end_dates) if len(end_dates) > 0 else pd.Series(dtype='datetime64[ns]')
    return X, y, end_dates



# ========== 训练单个站点 ==========
def train_one_station(location_name, X, y, win_dates):
    print(f"\n🚀 Training station {location_name} | X={X.shape}, y={y.shape}")

    # win_dates 已与 X/y 等长（每个窗口的末尾日期）
    X_train, X_test, y_train, y_test, d_train, d_test = train_test_split(
        X, y, win_dates, test_size=TEST_SIZE, shuffle=False
    )

    # 标准化特征
    mean = X_train.reshape(-1, X_train.shape[-1]).mean(axis=0)
    std = X_train.reshape(-1, X_train.shape[-1]).std(axis=0) + 1e-8
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    # 标准化目标（与 TCN 一致）
    y_mean = y_train.mean()
    y_std = y_train.std() + 1e-8
    y_train = (y_train - y_mean) / y_std
    y_test = (y_test - y_mean) / y_std

    # ========== 打印输入输出分布 ==========
    print(f"📊 X_train mean={X_train.mean():.4f}, std={X_train.std():.4f}")
    print(f"📊 y_train (std) mean={y_train.mean():.4f}, std={y_train.std():.4f}")

    # 转 Tensor
    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(DEVICE)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).to(DEVICE)
    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
    y_test_t = torch.tensor(y_test, dtype=torch.float32).to(DEVICE)

    # ========== LSTM MODEL ==========
    class LSTMModel(nn.Module):
        def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout=0.3):
            super(LSTMModel, self).__init__()
            self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout)
            self.fc = nn.Linear(hidden_dim, output_dim)

        def forward(self, x):
            out, _ = self.lstm(x)
            out = self.fc(out[:, -1, :])
            return out

    model = LSTMModel(
        input_dim=X.shape[2],
        hidden_dim=256,
        num_layers=4,
        output_dim=HORIZON,
        dropout=0.2
    ).to(DEVICE)

    criterion = nn.L1Loss()  # 替换 nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    patience = 10
    best_mae = float('inf')
    patience_counter = 0

    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # ========== 训练 ==========
    train_losses = []
    val_maes = []
    val_mses = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            X_batch = torch.nan_to_num(X_batch, nan=0.0)
            mask = (~torch.isnan(X_batch)).float()
            X_batch = X_batch * mask
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * X_batch.size(0)
        epoch_loss /= len(train_loader.dataset)

        # 验证
        model.eval()
        with torch.no_grad():
            X_test_t_masked = torch.nan_to_num(X_test_t, nan=0.0)
            mask = (~torch.isnan(X_test_t)).float()
            X_test_t_masked = X_test_t_masked * mask
            y_pred_val = model(X_test_t_masked).cpu().numpy()
            y_test_denorm = (y_test * y_std) + y_mean
            y_pred_denorm = (y_pred_val * y_std) + y_mean
            val_mae = mean_absolute_error(y_test.reshape(-1), y_pred_val.reshape(-1))
            val_mse = mean_squared_error(y_test_denorm[:, 0], y_pred_denorm[:, 0])
        train_losses.append(epoch_loss)
        val_maes.append(val_mae)
        val_mses.append(val_mse)

        scheduler.step(val_mae)

        if epoch % 5 == 0 or epoch == 1:
            print(f"[{location_name}] Epoch {epoch:02d} | TrainLoss={epoch_loss:.4f} | ValMAE={val_mae:.4f}")

        # Early stopping check
        if val_mae < best_mae:
            best_mae = val_mae
            patience_counter = 0
            best_model_wts = model.state_dict()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"⏸️ Early stopping triggered at epoch {epoch}")
                model.load_state_dict(best_model_wts)
                break

    # ========== 绘图 ==========
    # 1) 训练曲线
    plt.figure(figsize=(12, 4))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_maes, label='Val MAE')
    plt.title(f"LSTM Training History ({location_name})")
    plt.legend()
    plt.tight_layout()

    # 2) 测试集预测（PM2.5 单目标，但为多步预测 horizon，所以选择一个步或取均值）
    model.eval()
    with torch.no_grad():
        X_test_t_masked = torch.nan_to_num(X_test_t, nan=0.0)
        mask = (~torch.isnan(X_test_t)).float()
        X_test_t_masked = X_test_t_masked * mask
        y_pred_test = model(X_test_t_masked).cpu().numpy()

    # 反标准化（还原真实 PM2.5 数值）
    y_test_denorm = y_test * y_std + y_mean
    y_pred_test_denorm = y_pred_test * y_std + y_mean

    # 只看第 1 个预测步（更清晰）；如需第 k 步，将 step_idx 改为 k-1
    step_idx = 0
    plt.figure(figsize=(12, 4))
    plt.plot(d_test, y_test_denorm[:, step_idx], label=f'True step {step_idx+1}')
    plt.plot(d_test, y_pred_test_denorm[:, step_idx], label=f'Predicted step {step_idx+1}')
    plt.title(f"LSTM Prediction - PM2.5 (step {step_idx+1}) - {location_name}")
    plt.legend()
    plt.tight_layout()

    # 保存图像
    save_hist = f"results/{location_name}_history.png"
    save_pred = f"results/{location_name}_prediction_step{step_idx+1}.png"
    plt.savefig(save_pred, dpi=200)  # 保存最后一张（预测）
    plt.close('all')
    print(f"✅ Saved training history and prediction: {save_hist}, {save_pred}")

    # ========== 额外图表：验证 MAE 曲线 + 多步预测 MAE ==========
    # 1) 每个 epoch 的验证 MAE 曲线
    plt.figure(figsize=(8, 4))
    plt.plot(val_maes, marker='o', color='orange')
    plt.title(f"LSTM Validation MAE over Epochs ({location_name})")
    plt.xlabel("Epoch")
    plt.ylabel("MAE")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"results/{location_name}_ValMAE_curve.png", dpi=200)

    # 1b) 每个 epoch 的验证 MSE 曲线
    plt.figure(figsize=(8, 4))
    plt.plot(val_mses, marker='o', color='green')
    plt.title(f"LSTM Validation MSE over Epochs ({location_name})")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"results/{location_name}_ValMSE_curve.png", dpi=200)

    # 2) 验证集 MAE 表格（与曲线一致：每个 epoch）
    table_data = [[f"{i+1}", f"{val_maes[i]:.4f}"] for i in range(len(val_maes))]
    plt.figure(figsize=(6, 0.6 + 0.4 * len(val_maes)))
    plt.axis("off")
    table = plt.table(
        cellText=table_data,
        colLabels=["Epoch", "Val MAE"],
        cellLoc="center",
        loc="center"
    )
    table.scale(1, 1.2)
    plt.title(f"LSTM Validation MAE Table ({location_name})")
    plt.tight_layout()
    plt.savefig(f"results/{location_name}_ValMAE_table.png", dpi=200)
    plt.close('all')

    # 3) Val MSE 表格（与曲线一致：每个 epoch）
    table_data = [[f"{i+1}", f"{val_mses[i]:.4f}"] for i in range(len(val_mses))]
    plt.figure(figsize=(6, 0.6 + 0.4 * len(val_mses)))
    plt.axis("off")
    table = plt.table(
        cellText=table_data,
        colLabels=["Epoch", "Val MSE"],
        cellLoc="center",
        loc="center"
    )
    table.scale(1, 1.2)
    plt.title(f"LSTM Validation MSE Table ({location_name})")
    plt.tight_layout()
    plt.savefig(f"results/LSTM_{location_name}_ValMSE_table.png", dpi=200)
    plt.close('all')

    print(f"✅ Saved extra MAE plots: results/{location_name}_ValMAE_curve.png, results/{location_name}_StepMAE_bar.png")


# ========== 主程序 ==========
if __name__ == "__main__":
    args = parse_args()
    EPOCHS = args.epochs

    df = pd.read_csv(args.data)

    # ==== 日期分段 ====
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)

    diffs = df['Date'].diff().dt.days.fillna(1)
    valid_segments = []
    start = 0
    for i in range(1, len(df)):
        if diffs.iloc[i] > 3:  # gap 大于 3 天 -> 分段（不插值）
            valid_segments.append(df.iloc[start:i].copy())
            start = i
    valid_segments.append(df.iloc[start:].copy())

    # ==== 小于等于 3 天的 gap 才线性插值 ====
    df_filled = []
    for seg in valid_segments:
        seg = seg.drop_duplicates(subset='Date').sort_values('Date')
        diffs_seg = seg['Date'].diff().dt.days.fillna(1)
        if diffs_seg.max() <= 3:
            full_dates = pd.date_range(seg['Date'].min(), seg['Date'].max(), freq='D')
            seg = seg.set_index('Date').reindex(full_dates)
            seg.index.name = 'Date'
            seg = seg.interpolate(limit=3, limit_direction='both')
            df_filled.append(seg.reset_index())
        else:
            df_filled.append(seg)
    df = pd.concat(df_filled, ignore_index=True)

    # 自动识别目标列
    target_col = [c for c in df.columns if "pm25" in c.lower()]
    if not target_col:
        raise ValueError("未找到 PM2.5 相关列！")
    target_col = target_col[0]

    # 拆分特征与目标
    features_cols = [c for c in df.columns if c not in ['Date', 'location_id', 'location_name', 'station']]
    features = df[features_cols].values.astype(np.float32)
    target = df[target_col].values.astype(np.float32)
    dates = pd.to_datetime(df['Date'])  # 保留为 Series

    # 创建滑动窗口（同时返回与样本等长的末尾日期）
    X, y, end_dates = create_sliding_windows(features, target, LOOKBACK, HORIZON, dates=dates)

    # 训练模型（使用与 X/y 对齐的 end_dates）
    run_name = args.name or args.data.stem
    train_one_station(run_name, X, y, end_dates)
