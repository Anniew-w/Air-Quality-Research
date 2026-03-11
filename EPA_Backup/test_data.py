# test_data.py
from Models.data_utils import build_dataset

try:
    X, y, feats = build_dataset("10377")  # 用你有的站点ID
    print(f"X shape: {X.shape}, y shape: {y.shape}")
    print(f"Features: {feats}")
except Exception as e:
    print(f"Error: {e}")