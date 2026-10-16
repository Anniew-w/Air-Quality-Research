#!/bin/bash
# FreEformer — unified comparison run for air quality PM2.5 forecasting
# Horizon=14, Lookback=14, L1 loss (loss_mode=L1), PM2.5 space output (inverse=1)
# Train/Val/Test = 80/10/10

STATION=${1:-"Melbourne_CBD_10239_merged.csv"}
ROOT_PATH="/Users/wangshihan/Air Quality Research/Finalized_Dataset"

python run.py \
  --is_training 1 \
  --model FrePatchTST3_fre_patch \
  --model_id "${STATION%.csv}" \
  --data custom \
  --root_path "$ROOT_PATH" \
  --data_path "$STATION" \
  --features S \
  --target pm25 \
  --freq d \
  --seq_len 14 \
  --pred_len 14 \
  --label_len 0 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --d_model 512 \
  --n_heads 8 \
  --e_layers 2 \
  --d_layers 1 \
  --d_ff 2048 \
  --dropout 0.1 \
  --fc_dropout 0.3 \
  --head_dropout 0.3 \
  --patch_len 16 \
  --stride 8 \
  --batch_size 8 \
  --learning_rate 1e-4 \
  --loss_mode L1 \
  --train_epochs 30 \
  --patience 10 \
  --train_ratio 0.8 \
  --val_ratio 0.1 \
  --air_dataset 1 \
  --gap_days 3 \
  --interp_limit 3 \
  --interp_freq D \
  --inverse 1 \
  --use_revin 1 \
  --fix_seed 1 \
  --itr 1 \
  --des test
