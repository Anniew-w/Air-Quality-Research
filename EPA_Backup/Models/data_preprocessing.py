# data_preprocessing.py
import pandas as pd
import numpy as np
import os
import glob
from datetime import datetime
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_data(location_id, mapping_csv="epa_to_bom.csv", 
                           epa_dir="clients_epa_daily", bom_dir="Cleaned_BOM",
                           target_col="SBPM25", lookback=14, horizon=1, use_cols=None):
    """
    加载并预处理单个站点的数据
    """
    # 读映射表
    mapping = pd.read_csv(mapping_csv)
    location_id_str = str(location_id)
    row = mapping[mapping["location_id"].astype(str) == location_id_str]
    if row.empty:
        raise ValueError(f"Location ID {location_id} not found in mapping file.")
    row = row.iloc[0]

    # 加载 EPA 数据
    epa_path = os.path.join(epa_dir, location_id_str, f"epa_daily_{location_id_str}.csv")
    if not os.path.exists(epa_path):
        raise FileNotFoundError(f"EPA file not found: {epa_path}")
    
    epa = pd.read_csv(epa_path)
    epa.rename(columns={"date": "Date"}, inplace=True)
    epa["Date"] = pd.to_datetime(epa["Date"], errors="coerce")
    epa = epa.dropna(subset=["Date"])
    
    if target_col not in epa.columns:
        raise ValueError(f"Target column '{target_col}' not found in EPA data for {location_id}")

    # 加载 BOM 数据
    bom_name = row["bom_name"].replace(" ", "_").lower()
    bom_folder = os.path.join(bom_dir, bom_name)
    if not os.path.exists(bom_folder):
        raise FileNotFoundError(f"BOM folder not found: {bom_folder}")
    
    search_pattern = os.path.join(bom_folder, f"{bom_name}-*_cleaned.csv")
    matches = glob.glob(search_pattern)
    if not matches:
        raise FileNotFoundError(f"No cleaned BOM files found for {bom_name} in {bom_folder}")
    
    bom_path = max(matches, key=os.path.getmtime)
    bom = pd.read_csv(bom_path)
    bom.rename(columns={"Station Name": "Station"}, inplace=True)
    bom["Date"] = pd.to_datetime(bom["Date"], errors="coerce", dayfirst=True)
    bom = bom.dropna(subset=["Date"])

    # 合并数据
    df = epa.merge(bom, on="Date", how="left", suffixes=('', '_bom'))
    df = df.loc[:, ~df.columns.duplicated()]

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found after merge for {location_id}")

    # 处理缺失值 - 更严格的处理
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # 首先删除全为 NaN 的列
    cols_to_drop = []
    for col in numeric_cols:
        if df[col].isnull().all():
            cols_to_drop.append(col)
            print(f"Dropping column {col} because it's all NaN")
    
    df = df.drop(columns=cols_to_drop)
    numeric_cols = [col for col in numeric_cols if col not in cols_to_drop]
    
    # 用前后值填充
    df[numeric_cols] = df[numeric_cols].ffill().bfill()
    
    # 如果还有缺失，用均值填充
    for col in numeric_cols:
        if df[col].isnull().any():
            col_mean = df[col].mean()
            if pd.isna(col_mean):  # 如果均值也是 NaN，用0填充
                df[col].fillna(0, inplace=True)
                print(f"Filled NaN in {col} with 0 (mean was NaN)")
            else:
                df[col].fillna(col_mean, inplace=True)
                print(f"Filled NaN in {col} with mean: {col_mean}")

    # 选择特征
    if use_cols is not None:
        available_cols = [col for col in use_cols if col in df.columns]
        X = df[available_cols].values
    else:
        non_feature_cols = [target_col, "Date", "location_id", "Station"]
        feature_cols = [col for col in df.columns if col not in non_feature_cols]
        X = df[feature_cols].values

    y = df[target_col].values
    feature_names = df.columns.tolist()

    # 最终检查 NaN
    if np.isnan(X).any():
        # 如果还有 NaN，用0填充
        X = np.nan_to_num(X, nan=0.0)
        print(f"Final cleanup: filled {np.isnan(X).sum()} remaining NaN values in X with 0")
    
    if np.isnan(y).any():
        y_mean = np.nanmean(y)
        y = np.nan_to_num(y, nan=y_mean if not np.isnan(y_mean) else 0.0)
        print(f"Final cleanup: filled {np.isnan(y).sum()} remaining NaN values in y")

    return X, y, feature_names

def preprocess_all_datasets(**kwargs):
    """
    预处理所有站点的数据
    """
    # 设置默认值
    epa_dir = kwargs.get('epa_dir', 'clients_epa_daily')
    bom_dir = kwargs.get('bom_dir', 'Cleaned_BOM')
    mapping_csv = kwargs.get('mapping_csv', 'epa_to_bom.csv')
    target_col = kwargs.get('target_col', 'SBPM25')
    lookback = kwargs.get('lookback', 14)
    horizon = kwargs.get('horizon', 1)
    use_cols = kwargs.get('use_cols', None)
    
    datasets = {}
    for location_id in os.listdir(epa_dir):
        location_path = os.path.join(epa_dir, location_id)
        if os.path.isdir(location_path):
            try:
                X, y, feats = load_and_preprocess_data(
                    location_id, 
                    mapping_csv=mapping_csv,
                    epa_dir=epa_dir,
                    bom_dir=bom_dir,
                    target_col=target_col,
                    lookback=lookback,
                    horizon=horizon,
                    use_cols=use_cols
                )
                if len(X) > 0:
                    datasets[location_id] = (X, y, feats)
                    print(f"Successfully preprocessed location {location_id}")
            except Exception as e:
                print(f"Failed to preprocess location_id {location_id}: {e}")
    return datasets

# 在 data_preprocessing.py 文件末尾添加这个函数
def create_sequences(X, y, lookback=14, horizon=1):
    """
    创建时间序列数据
    X: 特征数据 (n_samples, n_features)
    y: 目标数据 (n_samples,)
    lookback: 时间步长
    horizon: 预测步长
    """
    X_seq, y_seq = [], []
    for i in range(len(X) - lookback - horizon + 1):
        X_seq.append(X[i:i+lookback])
        y_seq.append(y[i+lookback:i+lookback+horizon])
    return np.array(X_seq), np.array(y_seq)