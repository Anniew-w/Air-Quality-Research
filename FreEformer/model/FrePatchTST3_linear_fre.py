import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from layers.RevIN import RevIN


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.pre_len = configs.pred_len
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
        self.valid_fre_points2 = int((self.pre_len + 1) / 2 + 0.5)

        # input fre fine-tuning
        self.fre_linear_real = nn.Sequential(
            nn.Linear(self.valid_fre_points, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, self.valid_fre_points)
        )
        self.fre_linear_imag = nn.Sequential(
            nn.Linear(self.valid_fre_points, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, self.valid_fre_points)
        )

        self.fc = nn.Sequential(
            nn.Linear(self.seq_len * self.embed_size, self.d_ff),
            nn.GELU(),
            nn.Linear(self.d_ff, self.pre_len)
        )

        # for final input and output
        self.revin_layer = RevIN(self.enc_in, affine=True)
        self.dropout = nn.Dropout(configs.dropout)

    # dimension extension
    def tokenEmb(self, x, embeddings):
        if self.embed_size <= 1:
            return x.transpose(-1, -2).unsqueeze(-1)
        # x: [B, T, N] --> [B, N, T]
        x = x.transpose(-1, -2)
        x = x.unsqueeze(-1)
        # B*N*T*1 x 1*D = B*N*T*D
        return x * embeddings

    # frequency temporal learner
    def Fre_Linear(self, x):
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

        # ########## real Linear: use real image instead; seems that not sharing linear works better ####
        y_real = self.fre_linear_real(x_fre.real)
        y_imag = self.fre_linear_imag(x_fre.imag)
        y = torch.complex(y_real, y_imag)

        # [B, N, D, T]; automatically neglect the imag part of freq 0
        x = torch.fft.irfft(y, n=T, dim=-1, norm='ortho')

        # [B, N, T, D]
        x = x.transpose(-1, -2)
        return x

    def forward(self, x, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        # x: [Batch, Input length, Channel]
        # revin norm
        x = self.revin_layer(x, mode='norm')
        x_ori = x

        # ###########  frequency (high-level) part ##########
        # input fre fine-tuning
        # [B, T, N]
        # embedding x: [B, N, T, D]
        x = self.tokenEmb(x_ori, self.embeddings)
        # [B, N, T, D]
        x = self.Fre_Linear(x) + x

        # linear
        # [B, N, T*D] --> [B, N, dim] --> [B, N, tau] --> [B, tau, N]
        out_fre = self.fc(x.flatten(-2)).transpose(-1, -2)

        out = out_fre

        # dropout
        out = self.dropout(out)

        # revin denorm
        out = self.revin_layer(out, mode='denorm')

        return out
