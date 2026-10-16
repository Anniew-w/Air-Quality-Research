import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import json
import os
import random
import pandas as pd
from PGN import PGN  # 👈 假设你有一个 PGN 类，按需要调整
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pathlib import Path


# ========== 参数 ==========
LOOKBACK = 14
HORIZON = 14
EPOCHS = 20
LR = 3e-4
TEST_SIZE = 0.2
BATCH_SIZE = 64

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
os.makedirs("results", exist_ok=True)

DEFAULT_DATA_PATH = (
    Path(__file__).resolve().parents[1]
    / "Finalized_Dataset"
    / "Melbourne_CBD_10239_merged.csv"
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train the PGN model on one station CSV.")
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
    y = np.array(y).squeeze()
    end_dates = pd.to_datetime(end_dates) if len(end_dates) > 0 else pd.Series(dtype='datetime64[ns]')
    return X, y, end_dates


# ========== 训练单个站点 ==========
def train_one_station(location_name, X, y, win_dates):
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    print(f"\n🚀 Training station {location_name} | X={X.shape}, y={y.shape}")

    # win_dates 已与 X/y 等长（每个窗口的末尾日期）
    n = len(X)
    n_train = int(n * 0.8)
    n_val   = int(n * 0.1)
    X_train, y_train, d_train = X[:n_train],              y[:n_train],              win_dates[:n_train]
    X_val,   y_val,   d_val   = X[n_train:n_train+n_val], y[n_train:n_train+n_val], win_dates[n_train:n_train+n_val]
    X_test,  y_test,  d_test  = X[n_train+n_val:],        y[n_train+n_val:],        win_dates[n_train+n_val:]

    # 标准化特征（按训练集统计）
    mean = X_train.reshape(-1, X_train.shape[-1]).mean(axis=0)
    std = X_train.reshape(-1, X_train.shape[-1]).std(axis=0) + 1e-8
    X_train = (X_train - mean) / std
    X_val   = (X_val   - mean) / std
    X_test  = (X_test  - mean) / std

    # 标准化目标
    y_mean = y_train.mean()
    y_std = y_train.std() + 1e-8
    y_train = (y_train - y_mean) / y_std
    y_val   = (y_val   - y_mean) / y_std
    y_test  = (y_test  - y_mean) / y_std

    # 转 Tensor
    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(DEVICE)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).to(DEVICE)
    X_val_t   = torch.tensor(X_val,   dtype=torch.float32).to(DEVICE)
    X_test_t  = torch.tensor(X_test,  dtype=torch.float32).to(DEVICE)
    y_test_t  = torch.tensor(y_test,  dtype=torch.float32).to(DEVICE)

    # ========== 初始化 PGN 模型 ==========
    # 这里假设 PGN 的 __init__ 和 TCN 类似：
    # PGN(in_features, channels, kernel_size, dropout, horizon)
    # 如果你的 PGN 定义不一样，只改这一段的参数即可。
    model = PGN(
        in_features=X.shape[2],
        channels=(64, 64, 64, 64, 64, 64),
        kernel_size=3,
        dropout=0.3,
        horizon=HORIZON
    ).to(DEVICE)

    criterion = nn.L1Loss()  # 用 MAE 作为训练损失
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

        # 验证（在验证集上，PM2.5 空间）
        model.eval()
        with torch.no_grad():
            X_val_t_masked = torch.nan_to_num(X_val_t, nan=0.0)
            mask = (~torch.isnan(X_val_t)).float()
            X_val_t_masked = X_val_t_masked * mask
            y_pred_val = model(X_val_t_masked).cpu().numpy()
            y_val_denorm  = (y_val  * y_std) + y_mean
            y_pred_denorm = (y_pred_val * y_std) + y_mean
            val_mae = mean_absolute_error(y_val_denorm.reshape(-1), y_pred_denorm.reshape(-1))
            val_mse = mean_squared_error(y_val_denorm[:, 0], y_pred_denorm[:, 0])
        train_losses.append(epoch_loss)
        val_maes.append(val_mae)
        val_mses.append(val_mse)

        scheduler.step(val_mae)

        if epoch % 5 == 0 or epoch == 1:
            print(f"[{location_name}] Epoch {epoch:02d} | TrainLoss={epoch_loss:.4f} | ValMAE={val_mae:.4f}")

        # Early stopping
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
    save_hist = f"results/PGN_{location_name}_history.png"
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    ax1.plot(train_losses, color='steelblue', label='Train Loss')
    ax1.set_ylabel("Train Loss (normalized)")
    ax1.set_title(f"PGN Training History ({location_name})")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax2.plot(val_maes, color='orange', label='Val MAE (PM2.5)')
    ax2.set_ylabel("Val MAE (μg/m³)")
    ax2.set_xlabel("Epoch")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_hist, dpi=200)
    plt.close()

    # 2) 测试集预测
    model.eval()
    with torch.no_grad():
        X_test_t_masked = torch.nan_to_num(X_test_t, nan=0.0)
        mask = (~torch.isnan(X_test_t)).float()
        X_test_t_masked = X_test_t_masked * mask
        y_pred_test = model(X_test_t_masked).cpu().numpy()

    # 反标准化（还原真实 PM2.5 数值）
    y_pred_test = (y_pred_test * y_std) + y_mean
    y_test_real = (y_test * y_std) + y_mean

    # 只看第 1 个预测步（如需第 k 步，将 step_idx 改为 k-1）
    step_idx = 0
    plt.figure(figsize=(12, 4))
    plt.plot(d_test, y_test_real[:, step_idx], label=f'True step {step_idx+1}')
    plt.plot(d_test, y_pred_test[:, step_idx], label=f'Predicted step {step_idx+1}')
    plt.title(f"PGN Prediction - PM2.5 (step {step_idx+1}) - {location_name}")
    plt.legend()
    plt.tight_layout()

    # 保存图像
    save_pred = f"results/PGN_{location_name}_prediction_step{step_idx+1}.png"
    plt.savefig(save_pred, dpi=200)
    plt.close('all')
    print(f"✅ Saved training history and prediction: {save_hist}, {save_pred}")

    # ========== 额外图表：验证 MAE 曲线 + 多步预测 MAE ==========
    # 1) 每个 epoch 的验证 MAE 曲线
    plt.figure(figsize=(8, 4))
    plt.plot(val_maes, marker='o', color='orange')
    plt.title(f"Validation MAE over Epochs (PGN - {location_name})")
    plt.xlabel("Epoch")
    plt.ylabel("MAE (standardised)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"results/PGN_{location_name}_ValMAE_curve.png", dpi=200)

    # 1b) 每个 epoch 的验证 MSE 曲线
    plt.figure(figsize=(8, 4))
    plt.plot(val_mses, marker='o', color='green')
    plt.title(f"Validation MSE over Epochs (PGN - {location_name})")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"results/PGN_{location_name}_ValMSE_curve.png", dpi=200)

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
    plt.title(f"PGN Validation MAE Table ({location_name})")
    plt.tight_layout()
    plt.savefig(f"results/PGN_{location_name}_ValMAE_table.png", dpi=200)
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
    plt.title(f"PGN Validation MSE Table ({location_name})")
    plt.tight_layout()
    plt.savefig(f"results/PGN_{location_name}_ValMSE_table.png", dpi=200)
    plt.close('all')

    print(f"✅ Saved extra MAE plots: results/PGN_{location_name}_ValMAE_curve.png, results/{location_name}_PGN_StepMAE_bar.png")

    # ========== 最终测试指标（PM2.5 原始空间）==========
    y_true_flat = y_test_real.reshape(-1)
    y_pred_flat = y_pred_test.reshape(-1)
    test_mae  = mean_absolute_error(y_true_flat, y_pred_flat)
    test_rmse = float(np.sqrt(mean_squared_error(y_true_flat, y_pred_flat)))
    test_r2   = r2_score(y_true_flat, y_pred_flat)
    step_maes = [float(mean_absolute_error(y_test_real[:, i], y_pred_test[:, i])) for i in range(HORIZON)]

    print(f"\n{'='*50}")
    print(f"[{location_name}] Final Test Results (PM2.5 space, horizon={HORIZON})")
    print(f"  MAE:  {test_mae:.4f}")
    print(f"  RMSE: {test_rmse:.4f}")
    print(f"  R2:   {test_r2:.4f}")
    print(f"  Step MAEs: {[f'{m:.3f}' for m in step_maes]}")
    print(f"{'='*50}\n")

    metrics = {
        "model": "PGN",
        "location": location_name,
        "lookback": LOOKBACK,
        "horizon": HORIZON,
        "test_mae": round(float(test_mae), 4),
        "test_rmse": round(test_rmse, 4),
        "test_r2": round(float(test_r2), 4),
        "step_maes": [round(m, 4) for m in step_maes],
    }
    with open(f"results/PGN_{location_name}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"✅ Saved metrics to results/PGN_{location_name}_metrics.json")


# ========== 主程序 ==========
if __name__ == "__main__":
    # 这里直接复用你 tcn_running 里的数据路径，按需改成 Melbourne / Traralgon 都行
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

    # 拆分特征与目标（这里保留 target 列在 features 里还是排除，可以按你现在 TCN 的逻辑一致）
    # Common feature set present (low-NaN) across all 11 EPA stations; using a fixed
    # set instead of "all non-id columns" keeps the feature space identical across
    # stations and avoids columns like pan_evaporation that are ~100% NaN at most sites.
    COMMON_FEATURE_COLS = ['evapotranspiration', 'rain', 'max_temperature', 'min_temperature',
                           'max_rel_humidity', 'min_rel_humidity', 'avg_10m_wind_speed', 'solar_radiation']
    features_cols = [c for c in COMMON_FEATURE_COLS if c in df.columns]
    features = df[features_cols].values.astype(np.float32)
    target = df[target_col].values.astype(np.float32)
    dates = pd.to_datetime(df['Date'])  # 保留为 Series

    # 创建滑动窗口（同时返回与样本等长的末尾日期）
    X, y, end_dates = create_sliding_windows(features, target, LOOKBACK, HORIZON, dates=dates)

    # 训练模型（使用与 X/y 对齐的 end_dates）
    run_name = args.name or args.data.stem
    train_one_station(run_name, X, y, end_dates)
