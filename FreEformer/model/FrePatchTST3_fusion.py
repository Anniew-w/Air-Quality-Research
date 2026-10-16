"""Time + Frequency fusion backbone (strong baseline).

Runs BOTH the frequency branch (FFT -> real/imag Transformer encoders -> iFFT,
identical to FrePatchTST3_fre_patch) and the time branch (temporal Transformer,
identical to FrePatchTST3_temporal) on the same embedded input, then fuses the
two predictions with a learnable scalar gate:

    out = g * out_freq + (1 - g) * out_time,   g = sigmoid(gate_logit)

The gate is a single interpretable parameter (read it to see how much frequency
vs time the model relies on). Parameter names keep the two branches separable
(`patch_*` / `encoder_fre_*` = frequency; `temp_trans` / `encoder` = time),
which also makes this the natural backbone for spectral-split federated
aggregation later.
"""
import torch
import torch.nn as nn

from layers.RevIN import RevIN
from layers.Transformer_EncDec import Encoder_ori, EncoderLayer
from layers.SelfAttention_Family import AttentionLayer, FullAttention_ablation
from layers.Embed import PatchEmbedding


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.seq_len = configs.seq_len
        self.d_model = configs.d_model
        self.d_ff = configs.d_ff
        self.patch_len = configs.temp_patch_len
        self.stride = configs.temp_stride
        self.embed_size = configs.embed_size

        self.embeddings = nn.Parameter(torch.randn(1, self.embed_size))
        self.valid_fre_points = int((self.seq_len + 1) / 2 + 0.5)

        self.revin_layer = RevIN(self.enc_in, affine=True)
        self.dropout = nn.Dropout(configs.dropout)

        # separate output heads for the two branches, fused by a learnable gate
        self.fc_fre = nn.Sequential(
            nn.Linear(self.seq_len * self.embed_size, self.d_ff), nn.GELU(),
            nn.Linear(self.d_ff, self.pred_len))
        self.fc_time = nn.Sequential(
            nn.Linear(self.seq_len * self.embed_size, self.d_ff), nn.GELU(),
            nn.Linear(self.d_ff, self.pred_len))
        self.gate_logit = nn.Parameter(torch.zeros(1))  # sigmoid(0)=0.5 -> equal mix at init

        # ---------- frequency branch (mirrors FrePatchTST3_fre_patch) ----------
        self.patch_embedding_real = PatchEmbedding(
            self.d_model, patch_len=self.patch_len, stride=self.stride, padding=self.stride,
            dropout=configs.dropout, one_output=True)
        self.patch_embedding_imag = PatchEmbedding(
            self.d_model, patch_len=self.patch_len, stride=self.stride, padding=self.stride,
            dropout=configs.dropout, one_output=True)
        self.token_num = int((self.valid_fre_points - self.patch_len) / self.stride + 2)
        self.flatten_dim = self.d_model * self.token_num

        def _fre_encoder(tag):
            return Encoder_ori(
                [EncoderLayer(
                    AttentionLayer(
                        FullAttention_ablation(False, configs.factor, attention_dropout=configs.dropout,
                                               output_attention=configs.output_attention, token_num=self.token_num,
                                               SF_mode=configs.attn_enhance, softmax_flag=configs.attn_softmax_flag,
                                               weight_plus=configs.attn_weight_plus,
                                               outside_softmax=configs.attn_outside_softmax,
                                               plot_mat_flag=0, plot_grad_flag=0,
                                               save_folder=f'./attn_results/fusion_{tag}'),
                        configs.d_model, configs.n_heads),
                    configs.d_model, configs.d_ff, dropout=configs.dropout, activation=configs.activation)
                 for _ in range(configs.e_layers)],
                norm_layer=torch.nn.LayerNorm(configs.d_model), one_output=True, CKA_flag=configs.CKA_flag)

        self.encoder_fre_real = _fre_encoder("real")
        self.patch_real = nn.Sequential(self.patch_embedding_real, self.encoder_fre_real,
                                        nn.Flatten(start_dim=-2),
                                        nn.Linear(self.flatten_dim, self.valid_fre_points))
        self.encoder_fre_imag = _fre_encoder("imag")
        self.patch_imag = nn.Sequential(self.patch_embedding_imag, self.encoder_fre_imag,
                                        nn.Flatten(start_dim=-2),
                                        nn.Linear(self.flatten_dim, self.valid_fre_points))

        # ---------- time branch (mirrors FrePatchTST3_temporal) ----------
        self.encoder = Encoder_ori(
            [EncoderLayer(
                AttentionLayer(
                    FullAttention_ablation(False, configs.factor, attention_dropout=configs.dropout,
                                           output_attention=configs.output_attention, token_num=configs.enc_in,
                                           SF_mode=configs.attn_enhance, softmax_flag=configs.attn_softmax_flag,
                                           weight_plus=configs.attn_weight_plus,
                                           outside_softmax=configs.attn_outside_softmax,
                                           plot_mat_flag=0, save_folder='./attn_results/fusion_time'),
                    configs.d_model, configs.n_heads),
                configs.d_model, configs.d_ff, dropout=configs.dropout, activation=configs.activation)
             for _ in range(configs.e_layers)],
            norm_layer=torch.nn.LayerNorm(configs.d_model), one_output=True, CKA_flag=configs.CKA_flag)
        self.temp_trans = nn.Sequential(
            nn.Linear(self.seq_len * self.embed_size, self.d_model),
            self.encoder,
            nn.Linear(self.d_model, self.seq_len * self.embed_size))

    def tokenEmb(self, x, embeddings):
        if self.embed_size <= 1:
            return x.transpose(-1, -2).unsqueeze(-1)
        x = x.transpose(-1, -2).unsqueeze(-1)
        return x * embeddings

    def Fre_Trans(self, x):
        B, N, T, D = x.shape
        x = x.transpose(-1, -2)                                  # [B,N,D,T]
        x_fre = torch.fft.rfft(x, dim=-1, norm='ortho')         # [B,N,D,fre]
        x_fre = x_fre.flatten(0, 1)
        y_real = self.patch_real(x_fre.real).reshape(B, N, D, self.valid_fre_points)
        y_imag = self.patch_imag(x_fre.imag).reshape(B, N, D, self.valid_fre_points)
        y = torch.complex(y_real, y_imag)
        x = torch.fft.irfft(y, n=T, dim=-1, norm='ortho')       # [B,N,D,T]
        return x.transpose(-1, -2)                               # [B,N,T,D]

    def Fre_Temp(self, x):
        B, N, T, D = x.shape
        x = x.transpose(-1, -2)                                  # [B,N,D,T]
        x = self.temp_trans(x.flatten(-2)).reshape(B, N, D, T)
        return x.transpose(-1, -2)                               # [B,N,T,D]

    def forward(self, x, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        B, T, N = x.shape
        x = self.revin_layer(x, mode='norm')
        x_emb = self.tokenEmb(x, self.embeddings)               # [B,N,T,D]

        z_fre = self.Fre_Trans(x_emb) + x_emb
        z_time = self.Fre_Temp(x_emb) + x_emb

        out_fre = self.fc_fre(z_fre.flatten(-2)).transpose(-1, -2)   # [B,pred_len,N]
        out_time = self.fc_time(z_time.flatten(-2)).transpose(-1, -2)

        g = torch.sigmoid(self.gate_logit)
        out = g * out_fre + (1.0 - g) * out_time

        out = self.dropout(out)
        out = self.revin_layer(out, mode='denorm')
        return out
