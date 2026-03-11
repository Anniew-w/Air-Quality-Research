if [ ! -d "./logs/freformer/ablation/linear_or_attn" ]; then
    mkdir -p ./logs/freformer/ablation/linear_or_attn
fi

#export CUDA_VISIBLE_DEVICES=0

model_name=FrePatchTST3_linear_cha

seq_lens=(96 96 96 96)
pred_lens=(96 192 336 720)
train_ratios=(1.0 1.0 1.0 1.0)

d_models=(512 512 512 512)
embed_size=(16 16 16 16)
cuda_ids1=(1 1 1 1)
epochs=(30 50 30 30)

for ((i = 0; i < 4; i++))
do

    seq_len=${seq_lens[i]}
    pred_len=${pred_lens[i]}
    train_ratio=${train_ratios[i]}

    export CUDA_VISIBLE_DEVICES=${cuda_ids1[i]}

    python -u run.py \
      --is_training 1 \
      --root_path ./dataset/electricity/ \
      --data_path electricity.csv \
      --model_id ECL_freformer_${seq_len}_${pred_len}_linear_cha \
      --model $model_name \
      --data custom \
      --features M \
      --seq_len ${seq_len} \
      --pred_len ${pred_len} \
      --enc_in 321 \
      --dec_in 321 \
      --c_out 321 \
      --des 'Exp' \
      --embed_size ${embed_size[i]} \
      --d_model ${d_models[i]} \
      --d_ff ${d_models[i]} \
      --batch_size 32 \
      --learning_rate 5e-4 \
      --itr 1 \
      --e_layers 3 \
      --lossfun_alpha 0.5 \
      --test_batch_size 64 \
      --test_mode 0 \
      --fix_seed 1 \
      --resume_training 0 \
      --save_every_epoch 0 \
      --use_revin 1 \
      --use_norm 1 \
      --send_mail 0 \
      --save_pdf 0 \
      --train_epochs ${epochs[i]} \
      --patience 5 \
      --lradj cosine \
      --loss_mode L1 \
      --train_ratio $train_ratio \
      --dropout 0.0 \
      --plot_mat_flag 0 \
      --linear_attention 0 \
      2>&1 | tee -a logs/freformer/ablation/linear_or_attn/Freformer_ECL_${seq_len}_${pred_len}_linear_cha.log
done




