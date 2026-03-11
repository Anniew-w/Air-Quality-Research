import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch.utils.data import DataLoader, TensorDataset

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from ts_benchmark.baselines.time_series_library.models.DLinear import DLinear


def create_sliding_windows(features, target, lookback, horizon, dates=None):
    X, y = [], []
    end_dates = []
    n = len(features)
    win_len = lookback + horizon
    if n < win_len:
        return np.array([]), np.array([]), pd.Series(dtype="datetime64[ns]")

    if dates is not None:
        dates = pd.to_datetime(dates)
        if not isinstance(dates, pd.Series):
            dates = pd.Series(dates)

    valid_row = ~np.isnan(features).any(axis=1)
    valid_row &= ~np.isnan(target)
    valid_window = (
        np.convolve(valid_row.astype(int), np.ones(win_len, dtype=int), mode="valid") == win_len
    )
    start_indices = np.where(valid_window)[0]

    for idx in start_indices:
        X.append(features[idx: idx + lookback])
        y.append(target[idx + lookback: idx + lookback + horizon])
        if dates is not None:
            end_dates.append(dates.iloc[idx + win_len - 1])

    X = np.array(X)
    y = np.array(y)
    end_dates = pd.to_datetime(end_dates) if len(end_dates) > 0 else pd.Series(dtype="datetime64[ns]")
    return X, y, end_dates


def fill_gaps_tcn_style(df, date_col, gap_days=3):
    df = df.drop_duplicates(subset=date_col).sort_values(date_col)
    diffs = df[date_col].diff().dt.days.fillna(1)
    segments = []
    start = 0
    for i in range(1, len(df)):
        if diffs.iloc[i] > gap_days:
            segments.append(df.iloc[start:i].copy())
            start = i
    segments.append(df.iloc[start:].copy())

    filled = []
    for seg in segments:
        seg = seg.drop_duplicates(subset=date_col).sort_values(date_col)
        diffs_seg = seg[date_col].diff().dt.days.fillna(1)
        if diffs_seg.max() <= gap_days:
            full_dates = pd.date_range(seg[date_col].min(), seg[date_col].max(), freq="D")
            seg = seg.set_index(date_col).reindex(full_dates)
            seg.index.name = date_col
            seg = seg.interpolate(limit=gap_days, limit_direction="both")
            seg = seg.reset_index()
        filled.append(seg)
    return pd.concat(filled, ignore_index=True)


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--target", type=str, default="pm25")
    parser.add_argument("--seq_len", type=int, default=56)
    parser.add_argument("--pred_len", type=int, default=7)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--plot_dir", type=str, default="results")
    args = parser.parse_args()

    os.makedirs(args.plot_dir, exist_ok=True)

    df = pd.read_csv(args.data_path)
    date_col = "Date" if "Date" in df.columns else "date"
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)
    df = fill_gaps_tcn_style(df, date_col, gap_days=3)

    if args.target not in df.columns:
        raise ValueError(f"Target column '{args.target}' not found in data.")

    target = df[args.target].values.astype(np.float32)
    features = df[[args.target]].values.astype(np.float32)
    dates = pd.to_datetime(df[date_col])

    X, y, end_dates = create_sliding_windows(
        features, target, args.seq_len, args.pred_len, dates=dates
    )
    if y.ndim == 2:
        y = y[:, :, None]

    test_size = 0.2
    split_idx = int(len(X) * (1 - test_size))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    d_train, d_test = end_dates[:split_idx], end_dates[split_idx:]

    X_mean = X_train.reshape(-1, X_train.shape[-1]).mean(axis=0)
    X_std = X_train.reshape(-1, X_train.shape[-1]).std(axis=0) + 1e-8
    X_train = (X_train - X_mean) / X_std
    X_test = (X_test - X_mean) / X_std

    y_mean = y_train.reshape(-1).mean()
    y_std = y_train.reshape(-1).std() + 1e-8
    y_train = (y_train - y_mean) / y_std
    y_test = (y_test - y_mean) / y_std

    device = get_device()
    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_t = torch.tensor(y_test, dtype=torch.float32).to(device)

    class Config:
        pass

    cfg = Config()
    cfg.task_name = "short_term_forecast"
    cfg.seq_len = args.seq_len
    cfg.pred_len = args.pred_len
    cfg.enc_in = 1
    cfg.moving_avg = 25
    cfg.dropout = 0.0

    model = DLinear(cfg).to(device)
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    train_loader = DataLoader(
        TensorDataset(X_train_t, y_train_t),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )

    val_maes = []
    val_mses = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            preds = model(xb, None, None, None)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            preds = model(X_test_t, None, None, None).cpu().numpy()
        preds_denorm = preds * y_std + y_mean
        y_test_denorm = y_test * y_std + y_mean
        val_mae = mean_absolute_error(y_test.reshape(-1), preds.reshape(-1))
        val_mse = mean_squared_error(y_test_denorm[:, 0, 0], preds_denorm[:, 0, 0])
        val_maes.append(val_mae)
        val_mses.append(val_mse)

    # Prediction plot (step 1)
    with torch.no_grad():
        preds = model(X_test_t, None, None, None).cpu().numpy()
    preds = preds * y_std + y_mean
    y_test_plot = y_test * y_std + y_mean

    import matplotlib.pyplot as plt
    step_idx = 0
    plt.figure(figsize=(12, 4))
    plt.plot(d_test, y_test_plot[:, step_idx, 0], label="True step 1")
    plt.plot(d_test, preds[:, step_idx, 0], label="Predicted step 1")
    plt.title(f"DLinear PM2.5 Prediction (step 1)")
    plt.legend()
    plt.tight_layout()
    pred_path = os.path.join(
        args.plot_dir,
        "DLinear_Melbourne_CBD_10239_merged_prediction_step1.png",
    )
    plt.savefig(pred_path, dpi=200)
    plt.close()

    # MAE curve
    plt.figure(figsize=(8, 4))
    plt.plot(val_maes, marker="o")
    plt.title("DLinear Validation MAE over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("MAE")
    plt.tight_layout()
    mae_path = os.path.join(
        args.plot_dir,
        "DLinear_Melbourne_CBD_10239_merged_ValMAE_curve.png",
    )
    plt.savefig(mae_path, dpi=200)
    plt.close()

    # Val MAE table (per epoch)
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
    plt.title("DLinear Validation MAE Table")
    plt.tight_layout()
    mae_table_path = os.path.join(
        args.plot_dir,
        "DLinear_Melbourne_CBD_10239_merged_ValMAE_table.png",
    )
    plt.savefig(mae_table_path, dpi=200)
    plt.close()

    # Val MSE curve (validation set, step 1, real scale)
    plt.figure(figsize=(8, 4))
    plt.plot(val_mses, marker="o", color="green")
    plt.title("DLinear Validation MSE over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.tight_layout()
    mse_curve_path = os.path.join(
        args.plot_dir,
        "DLinear_Melbourne_CBD_10239_merged_ValMSE_curve.png",
    )
    plt.savefig(mse_curve_path, dpi=200)
    plt.close()

    # Val MSE table (per epoch)
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
    plt.title("DLinear Validation MSE Table")
    plt.tight_layout()
    mse_path = os.path.join(
        args.plot_dir,
        "DLinear_Melbourne_CBD_10239_merged_ValMSE_table.png",
    )
    plt.savefig(mse_path, dpi=200)
    plt.close()

    print("Saved:", pred_path)
    print("Saved:", mae_path)
    print("Saved:", mae_table_path)
    print("Saved:", mse_curve_path)
    print("Saved:", mse_path)


if __name__ == "__main__":
    main()
