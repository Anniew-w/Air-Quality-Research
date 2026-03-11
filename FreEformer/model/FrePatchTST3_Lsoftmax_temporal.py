import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from layers.RevIN import RevIN
from layers.Transformer_EncDec import Encoder_ori, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer, FullAttention_SF, LinearAttention, \
    FlashAttention, FullAttention_ablation


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in  # channels
        self.seq_len = configs.seq_len
        self.hidden_size = self.d_model = configs.d_model  # hidden_size
        self.d_ff = configs.d_ff  # d_ff
        self.time_branch = configs.time_branch  # hidden_size

        self.patch_len = configs.temp_patch_len
        self.stride = configs.temp_stride

        # self.channel_independence = configs.channel_independence
        self.embed_size = configs.embed_size  # embed_size
        self.embeddings = nn.Parameter(torch.randn(1, self.embed_size))
        self.embeddings2 = nn.Parameter(torch.randn(1, self.embed_size))
        self.embeddings_time = nn.Parameter(torch.randn(1, self.embed_size))

        self.fc = nn.Sequential(
            nn.Linear(self.seq_len * self.embed_size, self.d_ff),
            nn.GELU(),
            nn.Linear(self.d_ff, self.pred_len)
        )

        # for final input and output
        self.revin_layer = RevIN(self.enc_in, affine=True)
        self.dropout = nn.Dropout(configs.dropout)

        # ablation: real and imag parts share the attention

        self.encoder = Encoder_ori(
            [
                EncoderLayer(
                    AttentionLayer(
                        # FullAttention_ablation(False, configs.factor, attention_dropout=configs.dropout,
                        #                        output_attention=configs.output_attention, token_num=configs.enc_in,
                        #                        SF_mode=configs.attn_enhance, softmax_flag=configs.attn_softmax_flag,
                        #                        weight_plus=configs.attn_weight_plus,
                        #                        outside_softmax=configs.attn_outside_softmax,
                        #                        plot_mat_flag=configs.plot_mat_flag and _ == configs.e_layers - 1,
                        #                        save_folder=f'./attn_results/{configs.plot_mat_label}_{self.seq_len}'
                        #                                    f'_{self.pred_len}_011_last_layer/real'),
                        LinearAttention(mask_flag=False, attention_dropout=configs.dropout,
                                        output_attention=configs.output_attention, token_num=self.enc_in,
                                        mapping_fun='softmax_learn',
                                        plot_mat_flag=configs.plot_mat_flag,
                                        save_folder='./attn_results/FLatten/linear'),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for _ in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
            one_output=True,
            CKA_flag=configs.CKA_flag
        )
        self.temp_trans = nn.Sequential(
            nn.Linear(self.seq_len * self.embed_size, self.d_model),
            self.encoder,
            nn.Linear(self.d_model, self.seq_len * self.embed_size)
        )

    # dimension extension
    def tokenEmb(self, x, embeddings):
        if self.embed_size <= 1:
            return x.transpose(-1, -2).unsqueeze(-1)
        # x: [B, T, N] --> [B, N, T]
        x = x.transpose(-1, -2)
        x = x.unsqueeze(-1)
        # B*N*T*1 x 1*D = B*N*T*D
        return x * embeddings

    def Fre_Temp(self, x):
        # [B, N, T, D]
        B, N, T, D = x.shape
        assert T == self.seq_len
        # [B, N, D, T]
        x = x.transpose(-1, -2)

        x = self.temp_trans(x.flatten(-2)).reshape(B, N, D, T)

        # [B, N, T, D]
        x = x.transpose(-1, -2)
        return x

    def forward(self, x, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        # x: [Batch, Input length, Channel]
        B, T, N = x.shape

        # revin norm
        x = self.revin_layer(x, mode='norm')

        # embedding x: [B, N, T, D]
        x = self.tokenEmb(x, self.embeddings)
        # [B, N, T, D]
        x = self.Fre_Temp(x) + x

        # linear
        # [B, N, T*D] --> [B, N, dim] --> [B, N, tau] --> [B, tau, N]
        out = self.fc(x.flatten(-2)).transpose(-1, -2)

        # dropout
        out = self.dropout(out)

        # revin denorm
        out = self.revin_layer(out, mode='denorm')

        return out
