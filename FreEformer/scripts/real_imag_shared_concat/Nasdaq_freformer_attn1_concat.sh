if [ ! -d "./logs/freformer/ablation/attn" ]; then
    mkdir -p ./logs/freformer/ablation/attn
fi

model_name=FrePatchTST3_attn_concat

pred_lens=(3 6 9 12)
seq_lens=(12 12 12 12)


train_ratios=(1.0 1.0 1.0 1.0)

d_models=(512 512 512 512)
embed_size=(16 16 16 16)
cuda_ids1=(1 1 1 1)
epochs=(50 50 50 50)
lradj=(type3 type3 type3 type3)

attn_enhance=1
attn_softmax_flag=0
attn_weight_plus=1
attn_outside_softmax=1


for ((i = 0; i < 4; i++))
do

    seq_len=${seq_lens[i]}
    pred_len=${pred_lens[i]}

    export CUDA_VISIBLE_DEVICES=${cuda_ids1[i]}

    python -u run.py \
      --is_training 1 \
      --root_path ./dataset/nasdaq/ \
      --data_path nasdaq.csv \
      --model_id Nasdaq_freformer_${seq_len}_${pred_len}_e${attn_enhance}_s${attn_softmax_flag}\
_p${attn_weight_plus}_o${attn_outside_softmax}_${lradj[i]}_attn_concat \
      --model $model_name \
      --data custom \
      --features M \
      --seq_len ${seq_len} \
      --pred_len ${pred_len} \
      --enc_in 12 \
      --dec_in 12 \
      --c_out 12 \
      --des 'Exp' \
      --embed_size ${embed_size[i]} \
      --d_model ${d_models[i]} \
      --d_ff ${d_models[i]} \
      --batch_size 4 \
      --learning_rate 1e-4 \
      --itr 1 \
      --e_layers 3 \
      --lossfun_alpha 0.5 \
      --test_batch_size 16 \
      --test_mode 0 \
      --CKA_flag 0 \
      --fix_seed 1 \
      --resume_training 0 \
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
      --attn_enhance ${attn_enhance} \
      --attn_softmax_flag ${attn_softmax_flag} \
      --attn_weight_plus ${attn_weight_plus} \
      --attn_outside_softmax ${attn_outside_softmax} \
      2>&1 | tee -a logs/freformer/ablation/attn/Freformer_Nasdaq_${seq_len}_${pred_len}_${lradj[i]}_e${attn_enhance}\
_s${attn_softmax_flag}_p${attn_weight_plus}_o${attn_outside_softmax}_concat.log

done


