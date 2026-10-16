#!/bin/bash
# AFFIRM — unified comparison run for air quality PM2.5 forecasting
# Horizon=14, Lookback=32 (patch-aligned), L1 loss, PM2.5 space output via inverse_transform

STATION=${1:-"Melbourne_CBD_10239_merged.csv"}
ROOT_PATH="/Users/wangshihan/Air Quality Research/Finalized_Dataset"

python Affirm_Forecasting.py \
  --data custom \
  --root_path "$ROOT_PATH" \
  --data_path "$STATION" \
  --features S \
  --target pm25 \
  --freq d \
  --seq_len 32 \
  --label_len 0 \
  --pred_len 14 \
  --emb_dim 128 \
  --depth 2 \
  --patch_size 32 \
  --mask_ratio 0.4 \
  --batch_size 8 \
  --train_epochs 30 \
  --train_ratio 0.8 \
  --val_ratio 0.1 \
  --window_split True \
  --gap_interp True \
  --gap_days 3 \
  --interp_limit 3 \
  --interp_freq D \
  --AFFB True \
  --adaptive_filter True \
  --ICB False \
  --Mamba True \
  --load_from_pretrained False \
  --save_mae_curve True \
  --save_test_trend_plot True \
  --enc_in 1
