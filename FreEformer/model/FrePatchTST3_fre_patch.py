import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from layers.RevIN import RevIN
from layers.Transformer_EncDec import Encoder_ori, EncoderLayer
from layers.SelfAttention_Family import (FullAttention, AttentionLayer, FullAttention_SF, LinearAttention,
                                         FlashAttention, FullAttention_ablation)
from layers.Embed import PatchEmbedding


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

        self.valid_fre_points = int((self.seq_len + 1) / 2 + 0.5)

        self.fc = nn.Sequential(
            nn.Linear(self.seq_len * self.embed_size, self.d_ff),
            nn.GELU(),
            nn.Linear(self.d_ff, self.pred_len)
        )

        # for final input and output
        self.revin_layer = RevIN(self.enc_in, affine=True)
        self.dropout = nn.Dropout(configs.dropout)

        #
        self.patch_embedding_real = PatchEmbedding(
            self.d_model, patch_len=self.patch_len, stride=self.stride, padding=self.stride,
            dropout=configs.dropout, one_output=True)
        self.patch_embedding_imag = PatchEmbedding(
            self.d_model, patch_len=self.patch_len, stride=self.stride, padding=self.stride,
            dropout=configs.dropout, one_output=True)

        self.token_num = int((self.valid_fre_points - self.patch_len) / self.stride + 2)
        self.flatten_dim = self.d_model * self.token_num

        # ablation: real and imag parts share the attention
        self.encoder_fre_real = Encoder_ori(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention_ablation(False, configs.factor, attention_dropout=configs.dropout,
                                               output_attention=configs.output_attention, token_num=self.token_num,
                                               SF_mode=configs.attn_enhance, softmax_flag=configs.attn_softmax_flag,
                                               weight_plus=configs.attn_weight_plus,
                                               outside_softmax=configs.attn_outside_softmax,
                                               plot_mat_flag=configs.plot_mat_flag and _ == configs.e_layers - 1,
                                               plot_grad_flag=configs.plot_grad_flag and _ == configs.e_layers - 1,
                                               save_folder=f'./attn_results/{configs.plot_mat_label}_{self.seq_len}'
                                                           f'_{self.pred_len}_last_layer/real'),
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
        # [B*N, E, fre] --> [B*N*E, tokens, D] --> [B*N*E, fre]
        self.patch_real = nn.Sequential(self.patch_embedding_real,
                                        self.encoder_fre_real,
                                        nn.Flatten(start_dim=-2),
                                        nn.Linear(self.flatten_dim, self.valid_fre_points))

        self.encoder_fre_imag = Encoder_ori(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention_ablation(False, configs.factor, attention_dropout=configs.dropout,
                                               output_attention=configs.output_attention, token_num=self.token_num,
                                               SF_mode=configs.attn_enhance, softmax_flag=configs.attn_softmax_flag,
                                               weight_plus=configs.attn_weight_plus,
                                               outside_softmax=configs.attn_outside_softmax,
                                               plot_mat_flag=configs.plot_mat_flag and _ == configs.e_layers - 1,
                                               plot_grad_flag=configs.plot_grad_flag and _ == configs.e_layers - 1,
                                               save_folder=f'./attn_results/{configs.plot_mat_label}_{self.seq_len}'
                                                           f'_{self.pred_len}_last_layer/imag'),
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
        self.patch_imag = nn.Sequential(self.patch_embedding_imag,
                                        self.encoder_fre_imag,
                                        nn.Flatten(start_dim=-2),
                                        nn.Linear(self.flatten_dim, self.valid_fre_points))

    # dimension extension
    def tokenEmb(self, x, embeddings):
        if self.embed_size <= 1:
            return x.transpose(-1, -2).unsqueeze(-1)
        # x: [B, T, N] --> [B, N, T]
        x = x.transpose(-1, -2)
        x = x.unsqueeze(-1)
        # B*N*T*1 x 1*D = B*N*T*D
        return x * embeddings

    def Fre_Trans(self, x):
        # [B, N, T, D]
        B, N, T, D = x.shape
        assert T == self.seq_len
        # [B, N, D, T]
        x = x.transpose(-1, -2)

        # fft
        # [B, N, D, fre_points]
        x_fre = torch.fft.rfft(x, dim=-1, norm='ortho')  # FFT on L dimension
        # [B, N, D, fre_points]
        assert x_fre.shape[-1] == self.valid_fre_points

        # [B*N, D, fre]
        x_fre = x_fre.flatten(0, 1)

        y_real, y_imag = x_fre.real, x_fre.imag

        # ########## transformer ####
        y_real = self.patch_real(y_real).reshape(B, N, D, self.valid_fre_points)
        y_imag = self.patch_imag(y_imag).reshape(B, N, D, self.valid_fre_points)
        y = torch.complex(y_real, y_imag)

        # [B, N, D, T]; automatically neglect the imag part of freq 0
        x = torch.fft.irfft(y, n=T, dim=-1, norm='ortho')

        # [B, N, T, D]
        x = x.transpose(-1, -2)
        return x

    def forward(self, x, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        # x: [Batch, Input length, Channel]
        B, T, N = x.shape

        # revin norm
        x = self.revin_layer(x, mode='norm')
        x_ori = x

        # ###########  frequency (high-level) part ##########
        # input fre fine-tuning
        # [B, T, N]
        # embedding x: [B, N, T, D]
        x = self.tokenEmb(x_ori, self.embeddings)
        # [B, N, T, D]
        x = self.Fre_Trans(x) + x

        # linear
        # [B, N, T*D] --> [B, N, dim] --> [B, N, tau] --> [B, tau, N]
        out_fre = self.fc(x.flatten(-2)).transpose(-1, -2)

        # ###########  summarize both parts  ###########
        out = out_fre

        # dropout
        out = self.dropout(out)

        # revin denorm
        out = self.revin_layer(out, mode='denorm')

        return out
