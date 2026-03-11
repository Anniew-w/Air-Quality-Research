import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import os

from Models.data_utils import build_all_datasets
from Models.TCN import TCN

# ========== 参数 ==========
LOOKBACK = 14
HORIZON = 1
EPOCHS = 30
LR = 1e-3
TEST_SIZE = 0.2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs("results", exist_ok=True)


# ========== 训练单个站点 ==========
def train_one_station(location_id, X, y, y_dates):
    print(f"\n🚀 Training station {location_id} | X={X.shape}, y={y.shape}")

    X_train, X_test, y_train, y_test, d_train, d_test = train_test_split(
        X, y, y_dates, test_size=TEST_SIZE, shuffle=False
    )

    # 标准化
    mean = X_train.reshape(-1, X_train.shape[-1]).mean(axis=0)
    std = X_train.reshape(-1, X_train.shape[-1]).std(axis=0) + 1e-8
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    # 转 Tensor
    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(DEVICE)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).to(DEVICE)
    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
    y_test_t = torch.tensor(y_test, dtype=torch.float32).to(DEVICE)

    # 初始化模型
    model = TCN(
        in_features=X.shape[2],
        channels=(64, 64, 64),
        kernel_size=3,
        dropout=0.2,
        horizon=HORIZON
    ).to(DEVICE)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # ========== 训练 ==========
    train_losses = []
    val_maes = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        optimizer.zero_grad()
        pred = model(X_train_t)
        loss = criterion(pred, y_train_t)
        loss.backward()
        optimizer.step()

        # 验证
        model.eval()
        with torch.no_grad():
            y_pred_val = model(X_test_t).cpu().numpy()
            val_mae = mean_absolute_error(y_test, y_pred_val)
        train_losses.append(loss.item())
        val_maes.append(val_mae)

        if epoch % 5 == 0 or epoch == 1:
            print(f"[{location_id}] Epoch {epoch:02d} | TrainLoss={loss.item():.4f} | ValMAE={val_mae:.4f}")

    # ========== 绘图 ==========
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_maes, label='Val MAE')
    plt.title(f"Training History ({location_id})")
    plt.legend()

    model.eval()
    with torch.no_grad():
        y_pred_test = model(X_test_t).cpu().numpy()

    plt.subplot(1, 2, 2)
    plt.plot(d_test, y_test, label="True")
    plt.plot(d_test, y_pred_test, label="Predicted")
    plt.title(f"TCN Prediction - Station {location_id}")
    plt.legend()
    plt.tight_layout()

    plt.savefig(f"results/tcn_station_{location_id}.png", dpi=200)
    plt.close()
    print(f"✅ Saved results/tcn_station_{location_id}.png")


# ========== 主程序 ==========
if __name__ == "__main__":
    datasets = build_all_datasets(
        mapping_csv="epa_to_bom.csv",
        epa_dir="clients_epa_daily",
        bom_dir="Cleaned_BOM",
        target_col="SBPM25",
        lookback=LOOKBACK,
        horizon=HORIZON
    )

    # 只跑第一个站点看看效果
    for loc_id, (X, y, feats, y_dates) in datasets.items():
        train_one_station(loc_id, X, y, y_dates)
        break