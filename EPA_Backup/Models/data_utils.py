import pandas as pd
import os
import glob
from datetime import datetime
import numpy as np 


def build_dataset(location_id, mapping_csv="epa_to_bom.csv", epa_dir="clients_epa_daily", bom_dir="Cleaned_BOM", 
                  target_col="SBPM25", lookback=14, horizon=1, use_cols=None):
    # 读映射表
    mapping = pd.read_csv(mapping_csv)
    # 确保 location_id 类型一致（可能是字符串或整数）
    location_id_str = str(location_id)
    row = mapping[mapping["location_id"].astype(str) == location_id_str]
    if row.empty:
        raise ValueError(f"Location ID {location_id} not found in mapping file.")
    row = row.iloc[0]

    # 找到 EPA 文件路径
    epa_path = os.path.join(epa_dir, location_id_str, f"epa_daily_{location_id_str}.csv")
    if not os.path.exists(epa_path):
        raise FileNotFoundError(f"EPA file not found: {epa_path}")
    epa = pd.read_csv(epa_path)
    epa.rename(columns={"date": "Date"}, inplace=True)
    epa["Date"] = pd.to_datetime(epa["Date"], errors="coerce")
    # 去除无效日期
    epa = epa.dropna(subset=["Date"])
    # 确保有目标列
    if target_col not in epa.columns:
        raise ValueError(f"Target column '{target_col}' not found in EPA data for {location_id}")

    # 找到对应 BOM 文件路径
    bom_name = row["bom_name"].replace(" ", "_").lower()
    bom_folder = os.path.join(bom_dir, bom_name)
    if not os.path.exists(bom_folder):
        raise FileNotFoundError(f"BOM folder not found: {bom_folder}")
    search_pattern = os.path.join(bom_folder, f"{bom_name}-*_cleaned.csv")
    matches = glob.glob(search_pattern)
    if not matches:
        raise FileNotFoundError(f"No cleaned BOM files found for {bom_name} in {bom_folder}")
    # 取最新的一个文件（按修改时间）
    bom_path = max(matches, key=os.path.getmtime)
    bom = pd.read_csv(bom_path)
    bom.rename(columns={"Station Name": "Station"}, inplace=True)
    bom["Date"] = pd.to_datetime(bom["Date"], errors="coerce", dayfirst=True)
    bom = bom.dropna(subset=["Date"])

    # 合并数据，保留所有日期（左连接以 EPA 为准）
    df = epa.merge(bom, on="Date", how="left", suffixes=('', '_bom'))
    # 删除可能重复的列
    df = df.loc[:, ~df.columns.duplicated()]

    # 检查目标列是否存在
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found after merge for {location_id}")

    # 处理缺失值：用前后值填充数值列，但注意目标列可能本身有缺失
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    df[numeric_cols] = df[numeric_cols].ffill().bfill()

    # 选择特征
    if use_cols is not None:
        # 确保 use_cols 中的列都存在
        available_cols = [col for col in use_cols if col in df.columns]
        X = df[available_cols].values
    else:
        # 移除非特征列
        non_feature_cols = [target_col, "Date", "location_id", "Station"]
        feature_cols = [col for col in df.columns if col not in non_feature_cols]
        X = df[feature_cols].values

    y = df[target_col].values
    feature_names = df.columns.tolist()

    return X, y, feature_names

def build_all_datasets(mapping_csv="epa_to_bom.csv", epa_dir="clients_epa_daily", bom_dir="Cleaned_BOM",
                       target_col="SBPM25", lookback=14, horizon=1, use_cols=None):
    """
    批量生成所有站点的数据。返回一个字典，键为 location_id, 值为 (X, y, feats)。
    """
    datasets = {}
    # 遍历 epa_dir 下所有子文件夹（每个为 location_id）
    for location_id in os.listdir(epa_dir):
        location_path = os.path.join(epa_dir, location_id)
        if os.path.isdir(location_path):
            try:
                X, y, feats = build_dataset(location_id, mapping_csv, epa_dir, bom_dir,
                                            target_col, lookback, horizon, use_cols)
                datasets[location_id] = (X, y, feats)
            except Exception as e:
                print(f"Failed to build dataset for location_id {location_id}: {e}")
    return datasets