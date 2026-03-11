#!/usr/bin/env bash

set -euo pipefail

DATA_FILE="${1:-Melbourne_CBD_10239_merged.csv}"
MODEL_ID="${2:-${DATA_FILE%.csv}}"
EPOCHS="${3:-5}"
GPU_ID="${4:-0}"

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
FREEFORMER_DIR="$ROOT_DIR/FreEformer"
DATA_DIR="$ROOT_DIR/Finalized_Dataset"

cd "$FREEFORMER_DIR"

python -u run.py \
  --is_training 1 \
  --model_id "$MODEL_ID" \
  --model FrePatchTST3_fre_patch \
  --data custom \
  --root_path "$DATA_DIR" \
  --data_path "$DATA_FILE" \
  --features S \
  --target pm25 \
  --freq d \
  --air_dataset 1 \
  --window_split 1 \
  --train_ratio 0.8 \
  --val_ratio 0.0 \
  --seq_len 14 \
  --label_len 0 \
  --pred_len 14 \
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
  --temp_patch_len 16 \
  --temp_patch_len2 16 \
  --temp_stride 8 \
  --temp_stride2 8 \
  --Patch_layer_num 2 \
  --Patch_layer_num2 1 \
  --PatchTST_hier 1 \
  --Patch_CI 1 \
  --git_multi_stage 4 \
  --second_e_layers 1 \
  --decoder_cat_num 2 \
  --attn_enhance 1 \
  --attn_softmax_flag 1 \
  --attn_lookback 2 \
  --mapping_fun softmax_learn \
  --loss MSE \
  --loss_mode L1 \
  --learning_rate 3e-5 \
  --batch_size 8 \
  --test_batch_size 1 \
  --train_epochs "$EPOCHS" \
  --patience 5 \
  --itr 1 \
  --fix_seed 1 \
  --use_gpu 1 \
  --gpu "$GPU_ID" \
  --num_workers 4 \
  --des runpod \
  --use_revin 1 \
  --use_norm 1 \
  --inverse 0
