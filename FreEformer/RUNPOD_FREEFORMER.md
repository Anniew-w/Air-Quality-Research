# FreEformer on RunPod

## What to run

For your air-quality CSV files, use:

```bash
bash FreEformer/scripts/runpod_air_quality_freformer.sh Melbourne_CBD_10239_merged.csv melbourne_freformer 5 0
```

Arguments:

- arg1: CSV file name inside `Finalized_Dataset/`
- arg2: model id
- arg3: training epochs
- arg4: GPU id

## Recommended RunPod setup

- Template: `PyTorch`
- GPU: 1 x NVIDIA GPU
- Disk: 30GB or more

## Install

From the repo root on RunPod:

```bash
python -m pip install -U pip
pip install -r FreEformer/requirements.txt
```

## Run

From the repo root:

```bash
bash FreEformer/scripts/runpod_air_quality_freformer.sh Melbourne_CBD_10239_merged.csv melbourne_freformer 5 0
```

Another station:

```bash
bash FreEformer/scripts/runpod_air_quality_freformer.sh Traralgon_10011_merged.csv traralgon_freformer 10 0
```

## Output

FreEformer writes outputs under:

- `FreEformer/results/`
- `FreEformer/checkpoints/`
- `FreEformer/best_results/`

## Notes

- The script uses the same air-quality settings you already used locally: `data=custom`, `features=S`, `target=pm25`, `seq_len=14`, `pred_len=14`, `air_dataset=1`, `window_split=1`.
- It now runs with a lower default learning rate `1e-5` and enables `inverse=1`, so validation/test metrics are reported on the de-normalized PM2.5 scale.
- It uses the repository-relative data directory, so it does not depend on your local macOS absolute path.
- `num_workers` is set to `4` for safer RunPod behavior. You can raise it later if needed.
- Run it from the repository root so the output folders stay under `FreEformer/`.

## Direct raw command

If you do not want the helper script:

```bash
cd FreEformer
python -u run.py \
  --is_training 1 \
  --model_id melbourne_freformer \
  --model FrePatchTST3_fre_patch \
  --data custom \
  --root_path ../Finalized_Dataset \
  --data_path Melbourne_CBD_10239_merged.csv \
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
  --learning_rate 1e-5 \
  --batch_size 8 \
  --test_batch_size 1 \
  --train_epochs 5 \
  --patience 5 \
  --itr 1 \
  --fix_seed 1 \
  --use_gpu 1 \
  --gpu 0 \
  --num_workers 4 \
  --des runpod \
  --use_revin 1 \
  --use_norm 1 \
  --inverse 1
```
