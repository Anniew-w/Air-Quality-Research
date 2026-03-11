if [ ! -d "./logs/freformer/ablation/attn" ]; then
    mkdir -p ./logs/freformer/ablation/attn
fi

#export CUDA_VISIBLE_DEVICES=0

model_name=FrePatchTST3_attn_ablation

pred_lens=(24 36 48 60)
seq_lens=(36 36 36 36)

#pred_lens=(3 6 9 12)
#seq_lens=(12 12 12 12)


train_ratios=(1.0 1.0 1.0 1.0)

d_models=(256 256 256 256)
embed_size=(16 16 16 16)
cuda_ids1=(1 1 1 1)
epochs=(100 50 50 50)
lradj=(type3 type3 type3 type3)
layers=(1 2 2 2)
batch_sizes=(32 16 32 32)
resumes=(0 0 0 0)

attn_enhance=(0 1)
attn_softmax_flag=(0 1)
attn_weight_plus=(0 1)
attn_outside_softmax=(0 1)


for ((i = 0; i < 4; i++))
do
  for ((k0 = 0; k0 < 2; k0++))
  do
    for ((k1 = 0; k1 < 2; k1++))
    do
      for ((k2 = 0; k2 < 2; k2++))
      do
        for ((k3 = 0; k3 < 2; k3++))
        do

          if [[ $k0 -eq 0 && $(($k1 + $k2 + $k3)) -gt 0 ]]; then
            continue
          fi

          if [[ $k0 -eq 1 && $k1 -eq 0 && $k2 -eq 1 && $k3 -eq 1 ]]; then
              continue
          fi

          seq_len=${seq_lens[i]}
          pred_len=${pred_lens[i]}

          export CUDA_VISIBLE_DEVICES=${cuda_ids1[i]}

          python -u run.py \
            --is_training 1 \
            --root_path ./dataset/wiki/ \
            --data_path wiki_mini.csv \
            --model_id Wiki_freformer_${seq_len}_${pred_len}_e${attn_enhance[k0]}\
_s${attn_softmax_flag[k1]}_p${attn_weight_plus[k2]}_o${attn_outside_softmax[k3]} \
            --model $model_name \
            --data custom \
            --features M \
            --seq_len ${seq_len} \
            --pred_len ${pred_len} \
            --enc_in 99 \
            --dec_in 99 \
            --c_out 99 \
            --des 'Exp' \
            --embed_size ${embed_size[i]} \
            --d_model ${d_models[i]} \
            --d_ff ${d_models[i]} \
            --batch_size ${batch_sizes[i]} \
            --learning_rate 1e-4 \
            --itr 1 \
            --e_layers ${layers[i]} \
            --lossfun_alpha 0.5 \
            --test_batch_size 16 \
            --test_mode 0 \
            --CKA_flag 0 \
            --fix_seed 1 \
            --resume_training ${resumes[i]} \
            --resume_epoch 0 \
            --save_every_epoch 0 \
            --use_revin 1 \
            --use_norm 1 \
            --send_mail 0 \
            --save_pdf 0 \
            --train_epochs ${epochs[i]} \
            --patience 5 \
            --lradj ${lradj[i]} \
            --loss_mode L1 \
            --train_ratio 1.0 \
            --dropout 0.0 \
            --plot_mat_flag 0 \
            --attn_enhance ${attn_enhance[k0]} \
            --attn_softmax_flag ${attn_softmax_flag[k1]} \
            --attn_weight_plus ${attn_weight_plus[k2]} \
            --attn_outside_softmax ${attn_outside_softmax[k3]} \
            2>&1 | tee -a logs/freformer/ablation/attn/Freformer_Wiki_${seq_len}_${pred_len}_${lradj[i]}_e${attn_enhance[k0]}\
_s${attn_softmax_flag[k1]}_p${attn_weight_plus[k2]}_o${attn_outside_softmax[k3]}.log

        done
      done
    done
  done
done


