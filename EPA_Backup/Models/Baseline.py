import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from data_preprocessing import preprocess_all_datasets

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# ===== 定义一个简单的模型 (比如 DLinear) =====
class SimpleDLinear(nn.Module):
    def __init__(self, lookback, in_features, horizon=1):
        super().__init__()
        self.fc = nn.Linear(lookback*in_features, horizon)
    def forward(self, x):
        B, L, F = x.shape
        return self.fc(x.reshape(B, L*F))

def plot_training_history(train_losses, val_losses, val_maes, location_id, model_name="DLinear"):
    """绘制训练历史图表"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # 损失曲线
    epochs = range(1, len(train_losses) + 1)
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title(f'{model_name} - Training History\nLocation {location_id}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # MAE曲线
    ax2.plot(epochs, val_maes, 'g-', label='Validation MAE', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('MAE')
    ax2.set_title(f'{model_name} - Validation MAE\nLocation {location_id}')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'results/{model_name}_training_{location_id}.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_predictions_vs_actual(y_true, y_pred, location_id, model_name="DLinear"):
    """绘制预测值 vs 真实值"""
    plt.figure(figsize=(10, 8))
    
    # 散点图
    plt.scatter(y_true, y_pred, alpha=0.6, s=50, color='blue')
    
    # 完美预测线
    min_val = min(min(y_true), min(y_pred))
    max_val = max(max(y_true), max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    # 计算指标
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    plt.xlabel('True Values (SBPM25)')
    plt.ylabel('Predictions (SBPM25)')
    plt.title(f'{model_name} - Predictions vs True Values\nLocation {location_id}\n'
              f'MAE: {mae:.3f}, RMSE: {rmse:.3f}, R²: {r2:.3f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig(f'results/{model_name}_predictions_{location_id}.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_time_series_comparison(y_true, y_pred, location_id, model_name="DLinear"):
    """绘制时间序列对比图"""
    plt.figure(figsize=(12, 6))
    
    # 创建时间索引
    time_index = range(len(y_true))
    
    plt.plot(time_index, y_true, 'b-', label='True Values', linewidth=2, alpha=0.8)
    plt.plot(time_index, y_pred, 'r-', label='Predictions', linewidth=1.5, alpha=0.7)
    
    plt.xlabel('Time Index')
    plt.ylabel('SBPM25')
    plt.title(f'{model_name} - Time Series Comparison\nLocation {location_id}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig(f'results/{model_name}_timeseries_{location_id}.png', dpi=300, bbox_inches='tight')
    plt.close()

# =================================
# 主程序
# =================================
if __name__ == "__main__":
    # 创建结果目录
    import os
    os.makedirs('results', exist_ok=True)
    
    # 使用新的预处理函数
    datasets = preprocess_all_datasets(
        epa_dir="clients_epa_daily",
        bom_dir="Cleaned_BOM", 
        mapping_csv="epa_to_bom.csv",
        target_col="SBPM25", 
        lookback=14, 
        horizon=1
    )

    all_results = {}
    
    for location_id, (X, y, feats) in datasets.items():
        print(f"Location {location_id}: X shape: {X.shape}, y shape: {y.shape}")
        
        # 检查数据形状
        if len(X.shape) == 2:
            X = X.reshape(X.shape[0], 1, X.shape[1])
            print(f"Reshaped X to: {X.shape}")
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        
        # 转换为tensor
        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32)
        X_test  = torch.tensor(X_test, dtype=torch.float32)
        y_test  = torch.tensor(y_test, dtype=torch.float32)

        lookback = X_train.shape[1]
        in_features = X_train.shape[2]
        
        print(f"Lookback: {lookback}, In_features: {in_features}")

        model = SimpleDLinear(lookback, in_features, horizon=1)
        optimizer = optim.Adam(model.parameters(), lr=0.0005)
        criterion = nn.MSELoss()

        # 存储训练历史
        train_losses = []
        val_losses = []
        val_maes = []
        final_predictions = None

        # ===== 训练 =====
    best_mae = float('inf')
    best_predictions = None
    patience = 20
    patience_counter = 0

    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, verbose=True
    )

    for epoch in range(200):
        model.train()
        optimizer.zero_grad()
        y_pred = model(X_train)
        
        loss = criterion(y_pred.squeeze(), y_train)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(X_test)
            val_loss = criterion(val_pred.squeeze(), y_test).item()
            val_mae = mean_absolute_error(y_test.numpy(), val_pred.squeeze().detach().numpy())
        
        train_losses.append(loss.item())
        val_losses.append(val_loss)
        val_maes.append(val_mae)
        
        # 更新学习率
        scheduler.step(val_loss)
        
        # 保存最佳结果
        if val_mae < best_mae:
            best_mae = val_mae
            best_predictions = val_pred.squeeze().detach().numpy()
            patience_counter = 0
            print(f"🎯 New best MAE: {best_mae:.4f}")
        else:
            patience_counter += 1
        
        print(f"[{location_id}] Epoch {epoch+1}: TrainLoss={loss.item():.4f}, ValLoss={val_loss:.4f}, ValMAE={val_mae:.4f}, Patience: {patience_counter}/{patience}")
        
        # 早停检查
        if patience_counter >= patience:
            print(f"🛑 Early stopping at epoch {epoch+1}")
            break

    # 使用最佳预测结果绘制图表
    y_test_np = y_test.numpy()
    plot_training_history(train_losses, val_losses, val_maes, location_id, "Baseline_DLinear")
    plot_predictions_vs_actual(y_test_np, best_predictions, location_id, "Baseline_DLinear")
    plot_time_series_comparison(y_test_np, best_predictions, location_id, "Baseline_DLinear")

    # 保存结果
    all_results[location_id] = {
        'final_train_loss': train_losses[-1],
        'final_val_loss': val_losses[-1],
        'final_val_mae': best_mae,  # 使用最佳MAE而不是最后一代的
        'best_epoch': len(train_losses) - patience_counter,  # 最佳epoch
        'features_count': in_features
    }
    # 绘制所有站点的比较图
    plt.figure(figsize=(12, 6))
    locations = list(all_results.keys())
    maes = [all_results[loc]['final_val_mae'] for loc in locations]
    
    plt.bar(locations, maes, alpha=0.7, color='skyblue')
    plt.xlabel('Location ID')
    plt.ylabel('Final MAE')
    plt.title('Baseline DLinear - Final MAE across Locations')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # 在柱子上添加数值标签
    for i, v in enumerate(maes):
        plt.text(i, v + 0.1, f'{v:.2f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('results/Baseline_DLinear_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("所有图表已保存到 results/ 目录！")