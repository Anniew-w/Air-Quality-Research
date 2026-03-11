import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from layers.RevIN import RevIN
from layers.Transformer_EncDec import Encoder_ori, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer, FullAttention_SF, LinearAttention, \
    FlashAttention, FullAttention_ablation, ProbAttention, ReformerLayer, FlowAttention


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
        self.valid_fre_points2 = int((self.pred_len + 1) / 2 + 0.5)

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

        # output fre fine-tuning
        self.fre_linear_real_out = nn.Sequential(
            nn.Linear(self.valid_fre_points2, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, self.valid_fre_points2)
        )
        self.fre_linear_imag_out = nn.Sequential(
            nn.Linear(self.valid_fre_points2, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, self.valid_fre_points2)
        )

        self.fre_embed_2_dim = nn.Linear(self.seq_len * self.embed_size, self.d_model)
        self.input_2_dim = nn.Linear(self.seq_len, self.d_model)

        self.fc = nn.Sequential(
            nn.Linear(self.seq_len * self.embed_size, self.d_ff),
            nn.GELU(),
            nn.Linear(self.d_ff, self.pred_len)
        )

        self.fc_out = nn.Sequential(
            nn.Linear(self.pred_len * self.embed_size, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, self.pred_len)
        )

        # for final input and output
        self.revin_layer = RevIN(self.enc_in, affine=True)
        self.dropout = nn.Dropout(configs.dropout)

        # #############  transformer related  #########
        # patching
        # self.patch_embedding = PatchEmbedding(
        #     self.d_model, patch_len=self.patch_len, stride=self.stride, padding=self.stride, dropout=configs.dropout)
        # transformers
        if self.time_branch:
            if configs.linear_attention:
                self.encoder = Encoder_ori(
                    [
                        EncoderLayer(
                            AttentionLayer(
                                LinearAttention(mask_flag=False, attention_dropout=configs.dropout,
                                                output_attention=configs.output_attention, token_num=self.enc_in,
                                                mapping_fun='softmax_learn_v2',
                                                plot_mat_flag=configs.plot_mat_flag,
                                                save_folder='./attn_results/freformer_ECL_all_linear/linear'),
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
            else:
                self.encoder = Encoder_ori(
                    [
                        EncoderLayer(
                            AttentionLayer(
                                FullAttention_SF(False, configs.factor, attention_dropout=configs.dropout,
                                                 output_attention=configs.output_attention, token_num=configs.enc_in,
                                                 SF_mode=False, contri_flag=False, ij_mat_para=0),
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
            # time iTransformer
            self.fc_itrans = nn.Sequential(
                nn.Linear(self.seq_len, self.d_model),
                self.encoder,
                nn.Linear(self.d_model, self.pred_len)
            )

        # ablation: real and imag parts share the attention

        self.encoder_fre_real = Encoder_ori(
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
                        # LinearAttention(mask_flag=False, attention_dropout=configs.dropout,
                        #                 output_attention=configs.output_attention, token_num=self.enc_in,
                        #                 mapping_fun='softmax_learn',
                        #                 plot_mat_flag=configs.plot_mat_flag,
                        #                 save_folder='./attn_results/FLatten/linear'),
                        # ProbAttention(mask_flag=False, attention_dropout=configs.dropout,
                        #               output_attention=configs.output_attention),
                        # ReformerLayer(attention=None, d_model=configs.d_model, n_heads=configs.n_heads),
                        FlowAttention(attention_dropout=configs.dropout),
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
        self.fre_trans_real = nn.Sequential(
            nn.Linear(self.valid_fre_points * self.embed_size, self.d_model),
            self.encoder_fre_real,
            nn.Linear(self.d_model, self.valid_fre_points * self.embed_size)
        )
        self.encoder_fre_imag = Encoder_ori(
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
                        #                                    f'_{self.pred_len}_011_last_layer/imag'),
                        # LinearAttention(mask_flag=False, attention_dropout=configs.dropout,
                        #                 output_attention=configs.output_attention, token_num=self.enc_in,
                        #                 mapping_fun='softmax_learn',
                        #                 plot_mat_flag=configs.plot_mat_flag,
                        #                 save_folder='./attn_results/FLatten/linear'),
                        # ProbAttention(mask_flag=False, attention_dropout=configs.dropout,
                        #               output_attention=configs.output_attention),
                        # ReformerLayer(attention=None, d_model=configs.d_model, n_heads=configs.n_heads),
                        FlowAttention(attention_dropout=configs.dropout),
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
        # self.encoder_fre_imag = Encoder_ori(
        #     [
        #         EncoderLayer(
        #             self.attn_layers[_],
        #             configs.d_model,
        #             configs.d_ff,
        #             dropout=configs.dropout,
        #             activation=configs.activation
        #         ) for _ in range(configs.e_layers)
        #     ],
        #     norm_layer=torch.nn.LayerNorm(configs.d_model),
        #     one_output=True
        # )
        self.fre_trans_imag = nn.Sequential(
            nn.Linear(self.valid_fre_points * self.embed_size, self.d_model),
            self.encoder_fre_imag,
            nn.Linear(self.d_model, self.valid_fre_points * self.embed_size)
        )

        self.time_fre_weights = nn.Parameter(torch.ones(2))

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

        # frequency fine-tuning: not helpful
        # y_real = F.softplus(self.w_fre_real) * x_fre.real + self.b_fre_real
        # y_imag = F.softplus(self.w_fre_imag) * x_fre.imag + self.b_fre_imag

        y_real, y_imag = x_fre.real, x_fre.imag

        # ########## transformer ####
        y_real = self.fre_trans_real(y_real.flatten(-2)).reshape(B, N, D, self.valid_fre_points)
        y_imag = self.fre_trans_imag(y_imag.flatten(-2)).reshape(B, N, D, self.valid_fre_points)
        y = torch.complex(y_real, y_imag)

        # ablations: real and imag share the whole transformer
        # y_real = self.fre_trans_real(y_real.flatten(-2)).reshape(B, N, D, self.valid_fre_points)
        # y_imag = self.fre_trans_real(y_imag.flatten(-2)).reshape(B, N, D, self.valid_fre_points)
        # y = torch.complex(y_real, y_imag)

        # real and imag process together
        # y_real_imag = torch.cat([y_real.flatten(-2), y_imag.flatten(-2)], dim=-1)
        # y_real_imag = (self.fre_trans_real_imag(y_real_imag).reshape(B, N, D, 2, self.valid_fre_points)
        #                .transpose(-1, -2).contiguous())
        # y = torch.view_as_complex(y_real_imag)

        # [B, N, D, T]; automatically neglect the imag part of freq 0
        x = torch.fft.irfft(y, n=T, dim=-1, norm='ortho')

        # [B, N, T, D]
        x = x.transpose(-1, -2)
        return x

    def Fre_Linear_out(self, x):
        # [B, N, T, D]
        B, N, T, D = x.shape
        assert T == self.pred_len
        # [B, N, D, T]
        x = x.transpose(-1, -2)

        # fft
        # [B, N, D, fre_points]
        x_fre = torch.fft.rfft(x, dim=-1, norm='ortho')  # FFT on L dimension
        # [B, N, D, fre_points]
        assert x_fre.shape[-1] == self.valid_fre_points2

        # ########## real Linear: use real image instead; seems that not sharing linear works better ####
        y_real = self.fre_linear_real_out(x_fre.real)
        y_imag = self.fre_linear_imag_out(x_fre.imag)
        y = torch.complex(y_real, y_imag)

        # [B, N, D, T]; automatically neglect the imag part of freq 0
        x = torch.fft.irfft(y, n=T, dim=-1, norm='ortho')

        # [B, N, T, D]
        x = x.transpose(-1, -2)
        return x

    def Fre_Trans_out(self, x):
        # [B, N, T, D]
        B, N, T, D = x.shape
        assert T == self.pred_len
        # [B, N, D, T]
        x = x.transpose(-1, -2)

        # fft
        # [B, N, D, fre_points]
        x_fre = torch.fft.rfft(x, dim=-1, norm='ortho')  # FFT on L dimension
        # [B, N, D, fre_points]
        assert x_fre.shape[-1] == self.valid_fre_points2

        # ########## real trans: use real image instead; seems that not sharing linear works better ####
        y_real = self.fre_trans_real_out(x_fre.real.flatten(-2)).reshape(B, N, D, self.valid_fre_points2)
        y_imag = self.fre_trans_imag_out(x_fre.imag.flatten(-2)).reshape(B, N, D, self.valid_fre_points2)
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

        # ###########  temporal (low-level) part ##########
        # [B,N,tau] --> [B,tau,N]
        if self.time_branch:
            out_time = self.fc_itrans(x.transpose(-1, -2)).transpose(-1, -2)

        # ###########  frequency (high-level) part ##########
        # input fre fine-tuning
        # [B, T, N]
        # embedding x: [B, N, T, D]
        x = self.tokenEmb(x_ori, self.embeddings)
        # [B, N, T, D]
        x = self.Fre_Trans(x) + x
        # x = self.Fre_Linear(x) + x

        # linear
        # [B, N, T*D] --> [B, N, dim] --> [B, N, tau] --> [B, tau, N]
        out_fre = self.fc(x.flatten(-2)).transpose(-1, -2)

        # ###########  summarize both parts  ###########
        # weights = F.sigmoid(self.time_fre_weights[0])
        # out = weights * out_time + out_fre
        if self.time_branch:
            out = out_time + out_fre
        else:
            out = out_fre

        # dropout
        out = self.dropout(out)

        # revin denorm
        out = self.revin_layer(out, mode='denorm')

        return out
