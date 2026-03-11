if [ ! -d "./logs/freformer/ablation/attn" ]; then
    mkdir -p ./logs/freformer/ablation/attn
fi

#export CUDA_VISIBLE_DEVICES=0

model_name=FrePatchTST3_attn_ablation

seq_lens=(96 96 96 96)
pred_lens=(12 24 48 96)
train_ratios=(1.0 1.0 1.0 1.0)

d_models=(512 512 512 512)
embed_size=(16 16 16 16)
cuda_ids1=(1 1 1 1)

attn_enhance=1
attn_softmax_flag=0
attn_weight_plus=0
attn_outside_softmax=0

attn_enhance=1
attn_softmax_flag=(0 1)
attn_weight_plus=(0 1)
attn_outside_softmax=(0 1)


for ((i = 3; i < 4; i++))
do
  for ((k0 = 1; k0 < 2; k0++))
  do
    for ((k1 = 1; k1 < 2; k1++))
    do
      for ((k2 = 1; k2 < 2; k2++))
      do

        if [[ $k0 -eq 0 && $k1 -eq 1 && $k2 -eq 1 ]]; then
            continue
        fi
        if [[ $k0 -eq 1 && $k1 -eq 0 && $k2 -eq 0 ]]; then
            continue
        fi
        if [[ $k0 -eq 0 && $k1 -eq 0 && $k2 -eq 0 ]]; then
            continue
        fi
        if [[ $i -eq 0 &&  $k0 -eq 0 && $k1 -eq 0 && $k2 -eq 1 ]]; then
            continue
        fi
        if [[ $i -eq 0 &&  $k0 -eq 0 && $k1 -eq 1 && $k2 -eq 0 ]]; then
            continue
        fi
        if [[ $i -eq 0 &&  $k0 -eq 1 && $k1 -eq 0 && $k2 -eq 1 ]]; then
            continue
        fi

        seq_len=${seq_lens[i]}
        pred_len=${pred_lens[i]}

        export CUDA_VISIBLE_DEVICES=${cuda_ids1[i]}

        python -u run.py \
          --is_training 1 \
          --root_path ./dataset/PEMS/ \
          --data_path PEMS07.npz \
          --model_id PEMS07_freformer_${seq_len}_${pred_len}_e${attn_enhance}_s${attn_softmax_flag[k0]}\
_p${attn_weight_plus[k1]}_o${attn_outside_softmax[k2]}_attn_ablation \
          --model $model_name \
          --data PEMS \
          --features M \
          --seq_len ${seq_len} \
          --pred_len ${pred_len} \
          --enc_in 883 \
          --dec_in 883 \
          --c_out 883 \
          --des 'Exp' \
          --embed_size ${embed_size[i]} \
          --d_model ${d_models[i]} \
          --d_ff ${d_models[i]} \
          --batch_size 32 \
          --learning_rate 5e-4 \
          --itr 1 \
          --e_layers 3 \
          --lossfun_alpha 0.5 \
          --test_batch_size 16 \
          --test_mode 0 \
          --CKA_flag 0 \
          --fix_seed 1 \
          --resume_training 0 \
          --resume_epoch 0 \
          --save_every_epoch 0 \
          --use_revin 1 \
          --use_norm 1 \
          --send_mail 0 \
          --save_pdf 0 \
          --train_epochs 40 \
          --patience 5 \
          --lradj cosine \
          --loss_mode L1 \
          --train_ratio 1.0 \
          --dropout 0.0 \
          --plot_mat_flag 0 \
          --attn_enhance ${attn_enhance} \
          --attn_softmax_flag ${attn_softmax_flag[k0]} \
          --attn_weight_plus ${attn_weight_plus[k1]} \
          --attn_outside_softmax ${attn_outside_softmax[k2]} \
          2>&1 | tee -a logs/freformer/ablation/attn/Freformer_attn_abl_PEMS07_${seq_len}_${pred_len}_e${attn_enhance}\
_s${attn_softmax_flag[k0]}_p${attn_weight_plus[k1]}_o${attn_outside_softmax[k2]}.log

      done
    done
  done
done
