import argparse
import datetime
import os

import lightning as L
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from einops import rearrange
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint, TQDMProgressBar
from torchmetrics.regression import MeanSquaredError, MeanAbsoluteError
from timm.models.layers import DropPath
from timm.models.layers import trunc_normal_

from data_factory import data_provider
from utils import save_copy_of_files, random_masking_3D, str2bool
from affnet.layers import ConvLayer, get_normalization_layer, get_activation_fn
# from mamba_ssm import Mamba
try:
    from mamba_simple import Mamba
except Exception as exc:
    Mamba = None
    _MAMBA_IMPORT_ERROR = exc
from low_denoise import LowFrequencyDenoiser
import torch.nn.functional as F

class ICB(L.LightningModule):
    def __init__(self, in_features, hidden_features, drop=0.):
        super().__init__()
        self.conv1 = nn.Conv1d(in_features, hidden_features, 1)
        self.conv2 = nn.Conv1d(in_features, hidden_features, 3, 1, padding=1)
        self.conv3 = nn.Conv1d(hidden_features, in_features, 1)
        self.drop = nn.Dropout(drop)
        self.act = nn.GELU()

    def forward(self, x):
        x = x.transpose(1, 2)
        x1 = self.conv1(x)
        x1_1 = self.act(x1)
        x1_2 = self.drop(x1_1)

        x2 = self.conv2(x)
        x2_1 = self.act(x2)
        x2_2 = self.drop(x2_1)

        out1 = x1 * x2_2
        out2 = x2 * x1_2

        x = self.conv3(out1 + out2)
        x = x.transpose(1, 2)
        return x
    
class IDMamba_Block(L.LightningModule):
    def __init__(self, configs, dim, drop=0.):
        super().__init__()
        if Mamba is None:
            raise RuntimeError("Mamba is not available. Set --Mamba False to disable it.")
        self.configs = configs
        self.act = nn.SiLU()
        self.drop = nn.Dropout(drop)
        self.conv = nn.Conv1d(dim, dim, 1)
        self.mamba = Mamba(dim, d_state=self.configs.d_state, d_conv_1=self.configs.d_conv_1, d_conv_2=self.configs.d_conv_2, expand=self.configs.e_fact)
        
    def forward(self, x):
        x_act = self.act(x)
        x1, x2 = self.mamba(x)
        x1_1 = self.act(x1)
        x1_2 = self.drop(x1_1)

        x2_1 = self.act(x2)
        x2_2 = self.drop(x2_1)

        out1 = x1 * x2_2 * x_act
        out2 = x2 * x1_2 * x_act

        x = out1 + out2
        x = x.transpose(2, 1)
        x = self.conv(x)
        x = x.transpose(1, 2)
        return x
        
def complex_relu(x):
    real = F.relu(x.real)
    imag = F.relu(x.imag)
    return torch.complex(real, imag)


class LearnableFilterLayer(nn.Module):
    def __init__(self, dim):
        super(LearnableFilterLayer, self).__init__()
        self.complex_weight_1 = nn.Parameter(torch.randn(dim, 2, dtype=torch.float32) * 0.02)
        self.complex_weight_2 = nn.Parameter(torch.randn(dim, 2, dtype=torch.float32) * 0.02)
        trunc_normal_(self.complex_weight_1, std=.02)
        trunc_normal_(self.complex_weight_2, std=.02)

    def forward(self, x_fft):
        weight_1 = torch.view_as_complex(self.complex_weight_1)
        weight_2 = torch.view_as_complex(self.complex_weight_2)
        x_weighted = x_fft * weight_1
        x_weighted = complex_relu(x_weighted)
        x_weighted = x_weighted * weight_2
        return x_weighted
    
class Adaptive_Fourier_Filter_Block(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.learnable_filter_layer_1 = LearnableFilterLayer(dim)
        self.learnable_filter_layer_2 = LearnableFilterLayer(dim)
        self.learnable_filter_layer_3 = LearnableFilterLayer(dim)
        
        self.threshold_param = nn.Parameter(torch.rand(1) * 0.5)
        self.low_pass_cut_freq_param = nn.Parameter(dim // 2 - torch.rand(1) * 0.5)#用于确定低通滤波的截至频率，维度大小的一半减去一个小的随机值
        self.high_pass_cut_freq_param = nn.Parameter(dim // 4 - torch.rand(1) * 0.5)#高通滤波的截至频率，维度大小的四分之一减去一个小的随机值
    
    def create_adaptive_high_freq_mask(self, x_fft):
        B, _, _ = x_fft.shape

        # Calculate energy in the frequency domain
        energy = torch.abs(x_fft).pow(2).sum(dim=-1)

        # Flatten energy across H and W dimensions and then compute median
        flat_energy = energy.view(B, -1)  # Flattening H and W into a single dimension
        median_energy = flat_energy.median(dim=1, keepdim=True)[0]  # Compute median
        median_energy = median_energy.view(B, 1)  # Reshape to match the original dimensions

        # Normalize energy
        normalized_energy = energy / (median_energy + 1e-6)

        threshold = torch.quantile(normalized_energy, self.threshold_param_high)
        dominant_frequencies = normalized_energy > threshold

        # Initialize adaptive mask
        adaptive_mask = torch.zeros_like(x_fft, device=x_fft.device)
        adaptive_mask[dominant_frequencies] = 1

        return adaptive_mask
        
    def adaptive_freq_pass(self, x_fft, flag="high"): #对频率进行mask，flag指定是应用高通还是低通
        B, H, W_half = x_fft.shape  # W_half is the reduced dimension for real FFT
        W = (W_half - 1) * 2  # Calculate the full width assuming the input was real
        
        # Generate the non-negative frequency values along one dimension
        freq = torch.fft.rfftfreq(W, d=1/W).to(x_fft.device)
        
        if flag == "high": #根据flag创建mask
            freq_mask = torch.abs(freq) >= self.high_pass_cut_freq_param.to(x_fft.device)#允许高于 high_pass_cut_freq_param 的频率
        else:
            freq_mask = torch.abs(freq) <= self.low_pass_cut_freq_param.to(x_fft.device)#允许低于 low_pass_cut_freq_param 的频率
        return x_fft * freq_mask#将此mask应用于x_fft, 以选择性地保留某些频率
        
    def forward(self, x_in):
        B, N, C = x_in.shape
        dtype = x_in.dtype
        x = x_in.to(torch.float32)
         
        # Apply FFT along the time dimension
        x_fft = torch.fft.rfft(x, dim=1, norm='ortho')

        if args.adaptive_filter:
            # freq_mask = self.create_adaptive_high_freq_mask(x_fft)
            x_low_pass = self.adaptive_freq_pass(x_fft, flag="low")#低通
            
            x_high_pass = self.adaptive_freq_pass(x_fft, flag="high")#高通

        x_weighted = self.learnable_filter_layer_1(x_fft) + self.learnable_filter_layer_2(x_low_pass) + self.learnable_filter_layer_3(x_high_pass)
        #x_weighted = self.learnable_filter_layer_1(x_fft) + self.learnable_filter_layer_2(x_low_pass)#只要低通，相当于只去掉高频噪声
        # Apply Inverse FFT
        x = torch.fft.irfft(x_weighted, n=N, dim=1, norm='ortho')

        x = x.to(dtype)
        x = x.view(B, N, C)  # Reshape back to original shape

        return x


class Affirm_layer(L.LightningModule):
    def __init__(self, dim, mlp_ratio=3., drop=0., bias=True, drop_path=0., norm_layer=nn.LayerNorm, configs=0):
        super().__init__()
        self.configs = configs
        self.affb = Adaptive_Fourier_Filter_Block(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        if args.Mamba and Mamba is not None:
            self.idmamba = IDMamba_Block(configs, dim)
        else:
            self.idmamba = nn.Identity()

    def forward(self, x_in):
        if args.Mamba and args.AFFB:
            x = self.norm1(x_in)
            x = self.affb(x)
            x = self.norm2(x)
            x = self.idmamba(x)
            x = x_in + self.drop_path(x)
        elif args.Mamba:
            x = self.norm1(x_in)
            x = self.idmamba(x)
            x = x_in + self.drop_path(x)
        elif args.AFFB:
            x = x_in + self.drop_path(self.affb(self.norm1(x_in)))
        return x


class Affirm(nn.Module):

    def __init__(self,configs):
        super(Affirm, self).__init__()
        self.configs = configs
        self.patch_size = args.patch_size
        self.stride = self.patch_size // 2
        num_patches = int((args.seq_len - self.patch_size) / self.stride + 1)
        # Layers/Networks
        self.input_layer = nn.Linear(self.patch_size, args.emb_dim)
        
        dpr = [x.item() for x in torch.linspace(0, args.dropout, args.depth)]  # stochastic depth decay rule

        self.affirm_blocks = nn.ModuleList([
            Affirm_layer(dim=args.emb_dim, drop=args.dropout, drop_path=dpr[i],configs=self.configs)
            for i in range(args.depth)]
        )
        print(f"self.affirm_blocks:{self.affirm_blocks}")
        # Parameters/Embeddings
        # self.conv3 = nn.Conv1d(hidden_features, in_features, 1)
        self.out_layer = nn.Linear(args.emb_dim * num_patches, args.pred_len)

    def pretrain(self, x_in):
        x = rearrange(x_in, 'b l m -> b m l')
        x_patched = x.unfold(dimension=-1, size=self.patch_size, step=self.stride)
        x_patched = rearrange(x_patched, 'b m n p -> (b m) n p')

        xb_mask, _, self.mask, _ = random_masking_3D(x_patched, mask_ratio=args.mask_ratio)
        self.mask = self.mask.bool()  # mask: [bs x num_patch]
        xb_mask = self.input_layer(xb_mask)

        for affirm_block in self.affirm_blocks:
            xb_mask = affirm_block(xb_mask)

        return xb_mask, self.input_layer(x_patched)


    def forward(self, x):
        B, L, M = x.shape

        means = x.mean(1, keepdim=True).detach()
        x = x - means
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        x /= stdev

        x = rearrange(x, 'b l m -> b m l')
        x = x.unfold(dimension=-1, size=self.patch_size, step=self.stride)
        x = rearrange(x, 'b m n p -> (b m) n p')
        x = self.input_layer(x)

        for affirm_block in self.affirm_blocks:
            x = affirm_block(x)

        outputs = self.out_layer(x.reshape(B * M, -1))
        outputs = rearrange(outputs, '(b m) l -> b l m', b=B)

        outputs = outputs * stdev
        outputs = outputs + means

        return outputs


class model_pretraining(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.save_hyperparameters()
        self.model = Affirm(configs = args)
        print(f"self.model:{self.model}")

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.model.parameters(), lr=1e-4, weight_decay=1e-6)
        return optimizer

    def _calculate_loss(self, batch, mode="train"):
        batch_x, batch_y, _, _ = batch
        _, _, C = batch_x.shape
        batch_x = batch_x.float().to(self.device)

        preds, target = self.model.pretrain(batch_x)

        loss = (preds - target) ** 2
        loss = loss.mean(dim=-1)
        loss = (loss * self.model.mask).sum() / self.model.mask.sum()

        # Logging for both step and epoch
        self.log(f"{mode}_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._calculate_loss(batch, mode="train")
        return loss

    def validation_step(self, batch, batch_idx):
        self._calculate_loss(batch, mode="val")

    def test_step(self, batch, batch_idx):
        self._calculate_loss(batch, mode="test")


class model_training(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.save_hyperparameters()
        self.model = Affirm(configs=args)
        print(f"self.model:{self.model}")
        self.criterion = nn.MSELoss()
        self.mse = MeanSquaredError()
        self.mae = MeanAbsoluteError()
        self.preds = []
        self.trues = []
       
    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.model.parameters(), lr=1e-4, weight_decay=1e-6)
        scheduler = {
            'scheduler': optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2,
                                                              verbose=True),
            'monitor': 'val_mse',
            'interval': 'epoch',
            'frequency': 1
        }
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    def _calculate_loss(self, batch, mode="train"):
        batch_x, batch_y, _, _ = batch
        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float().to(self.device)

        outputs = self.model(batch_x)
        outputs = outputs[:, -args.pred_len:, :]
        batch_y = batch_y[:, -args.pred_len:, :].to(self.device)
        loss = self.criterion(outputs, batch_y)

        pred = outputs.detach().contiguous().cpu()
        true = batch_y.detach().contiguous().cpu()

        mse = self.mse(pred, true)
        mae = self.mae(pred, true)

        # Logging for both step and epoch
        self.log(f"{mode}_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log(f"{mode}_mse", mse, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log(f"{mode}_mae", mae, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss, pred, true

    def training_step(self, batch, batch_idx):
        loss, _, _ = self._calculate_loss(batch, mode="train")
        return loss

    def validation_step(self, batch, batch_idx):
        self._calculate_loss(batch, mode="val")

    def test_step(self, batch, batch_idx):
        loss, preds, trues = self._calculate_loss(batch, mode="test")
        self.preds.append(preds)
        self.trues.append(trues)
        return {'test_loss': loss, 'pred': preds, 'true': trues}

    def on_train_batch_end(self, outputs, batch, batch_idx):
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)

    def on_test_epoch_end(self):
        preds = torch.cat(self.preds)
        trues = torch.cat(self.trues)

        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])

        preds = preds.contiguous()
        trues = trues.contiguous()
        mse = self.mse(preds, trues)
        mae = self.mae(preds, trues)
        print(f"{mae, mse}")


def save_prediction_plot(preds, trues, save_path, channel=0, max_points=400, inv_transform=None, title=None):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skipping prediction plot.")
        return

    if preds.ndim != 3 or trues.ndim != 3:
        raise ValueError("preds and trues must be [batch, time, channels]")

    preds_np = preds.cpu().numpy()
    trues_np = trues.cpu().numpy()

    if inv_transform is not None:
        try:
            preds_flat = preds_np.reshape(-1, preds_np.shape[-1])
            trues_flat = trues_np.reshape(-1, trues_np.shape[-1])
            preds_np = inv_transform(preds_flat).reshape(preds_np.shape)
            trues_np = inv_transform(trues_flat).reshape(trues_np.shape)
        except Exception as exc:
            print(f"Inverse transform failed; plotting normalized scale. Error: {exc}")

    channel = int(channel)
    channel = max(0, min(channel, preds_np.shape[-1] - 1))

    pred_series = preds_np[0, :, channel]
    true_series = trues_np[0, :, channel]

    if max_points is not None and max_points > 0:
        pred_series = pred_series[:max_points]
        true_series = true_series[:max_points]

    plt.figure(figsize=(10, 4))
    plt.plot(true_series, label="True")
    plt.plot(pred_series, label="Pred")
    plt.title(title or f"Prediction vs True (channel {channel})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def save_global_prediction_plot(model, loader, save_path, channel=0, inv_transform=None, max_points=None, frac=1.0):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skipping prediction plot.")
        return

    preds_all = []
    trues_all = []
    model.eval()
    model_device = next(model.parameters()).device
    with torch.no_grad():
        for batch_x, batch_y, _, _ in loader:
            batch_x = batch_x.float().to(model_device)
            batch_y = batch_y[:, -args.pred_len:, :].float().to(model_device)
            outputs = model(batch_x)
            outputs = outputs[:, -args.pred_len:, :]
            preds_all.append(outputs.detach().cpu())
            trues_all.append(batch_y.detach().cpu())

    preds = torch.cat(preds_all, dim=0)
    trues = torch.cat(trues_all, dim=0)

    preds_np = preds.numpy()
    trues_np = trues.numpy()

    if inv_transform is not None:
        try:
            preds_flat = preds_np.reshape(-1, preds_np.shape[-1])
            trues_flat = trues_np.reshape(-1, trues_np.shape[-1])
            preds_np = inv_transform(preds_flat).reshape(preds_np.shape)
            trues_np = inv_transform(trues_flat).reshape(trues_np.shape)
        except Exception as exc:
            print(f"Inverse transform failed; plotting normalized scale. Error: {exc}")

    channel = int(channel)
    channel = max(0, min(channel, preds_np.shape[-1] - 1))

    pred_series = preds_np[:, :, channel].reshape(-1)
    true_series = trues_np[:, :, channel].reshape(-1)

    if frac is not None and frac > 0 and frac < 1.0:
        start = int(len(pred_series) * (1.0 - frac))
        pred_series = pred_series[start:]
        true_series = true_series[start:]

    if max_points is not None and max_points > 0:
        pred_series = pred_series[:max_points]
        true_series = true_series[:max_points]

    plt.figure(figsize=(12, 4))
    plt.plot(true_series, label="True")
    plt.plot(pred_series, label="Pred")
    plt.title(f"Global Prediction vs True (channel {channel})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def save_test_trend_plot(model, dataset, loader, save_path, channel=0, inv_transform=None, max_points=None):
    if not hasattr(dataset, "data_dates"):
        print("Dataset has no data_dates; skipping test trend plot.")
        return
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skipping prediction plot.")
        return

    preds_all = []
    trues_all = []
    date_ends = []
    model.eval()
    model_device = next(model.parameters()).device

    idx = 0
    map_len = None
    if hasattr(dataset, "_index_map") and dataset._index_map:
        map_len = len(dataset._index_map)
    with torch.no_grad():
        for batch_x, batch_y, _, _ in loader:
            bsz = batch_x.size(0)
            batch_x = batch_x.float().to(model_device)
            batch_y = batch_y[:, -args.pred_len:, :].float().to(model_device)
            outputs = model(batch_x)
            outputs = outputs[:, -args.pred_len:, :]

            preds_all.append(outputs.mean(dim=1).detach().cpu().numpy())
            trues_all.append(batch_y.mean(dim=1).detach().cpu().numpy())

            for j in range(bsz):
                if map_len is not None:
                    if idx + j >= map_len:
                        continue
                    start_idx = dataset._index_map[idx + j]
                else:
                    start_idx = idx + j
                end_idx = start_idx + args.seq_len + args.pred_len - 1
                if end_idx < len(dataset.data_dates):
                    date_ends.append(dataset.data_dates.iloc[end_idx])
            idx += bsz

    if not preds_all:
        print("No predictions for test trend plot.")
        return

    preds = np.concatenate(preds_all, axis=0)
    trues = np.concatenate(trues_all, axis=0)
    n = min(len(date_ends), preds.shape[0], trues.shape[0])
    preds = preds[:n]
    trues = trues[:n]
    date_ends = date_ends[:n]

    if inv_transform is not None:
        try:
            preds = inv_transform(preds)
            trues = inv_transform(trues)
        except Exception as exc:
            print(f"Inverse transform failed; plotting normalized scale. Error: {exc}")

    channel = int(channel)
    channel = max(0, min(channel, preds.shape[-1] - 1))
    pred_series = preds[:, channel]
    true_series = trues[:, channel]

    if max_points is not None and max_points > 0:
        pred_series = pred_series[-max_points:]
        true_series = true_series[-max_points:]
        date_ends = date_ends[-max_points:]

    plt.figure(figsize=(12, 4))
    plt.plot(date_ends, true_series, label="True")
    plt.plot(date_ends, pred_series, label="Pred (mean over horizon)")
    plt.title(f"Test Trend (mean over pred_len, channel {channel})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def pretrain_model():
    PRETRAIN_MAX_EPOCHS = args.pretrain_epochs
    trainer = L.Trainer(
        default_root_dir=CHECKPOINT_PATH,
        accelerator="auto",
        devices=1,
        num_sanity_val_steps=0,
        max_epochs=PRETRAIN_MAX_EPOCHS,
        callbacks=[
            pretrain_checkpoint_callback,
            LearningRateMonitor("epoch"),
            TQDMProgressBar(refresh_rate=500)
        ],
    )
    trainer.logger._log_graph = False  # If True, we plot the computation graph in tensorboard
    trainer.logger._default_hp_metric = None  # Optional logging argument that we don't need

    L.seed_everything(args.seed)  # To be reproducible
    model = model_pretraining()
    trainer.fit(model, train_loader, val_loader)

    return model, pretrain_checkpoint_callback.best_model_path


def train_model(pretrained_model_path):
    trainer = L.Trainer(
        default_root_dir=CHECKPOINT_PATH,
        accelerator="auto",
        num_sanity_val_steps=0,
        devices=1,
        max_epochs=args.train_epochs,
        callbacks=[
            checkpoint_callback,
            LearningRateMonitor("epoch"),
            TQDMProgressBar(refresh_rate=500)
        ],
    )
    trainer.logger._log_graph = False  # If True, we plot the computation graph in tensorboard
    trainer.logger._default_hp_metric = None  # Optional logging argument that we don't need

    L.seed_everything(args.seed)  # To be reproducible
    if args.load_from_pretrained:
        model = model_training.load_from_checkpoint(pretrained_model_path)
    else:
        model = model_training()
    trainer.fit(model, train_loader, val_loader)

    # Load the best checkpoint after training
    model = model_training.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
    # Test best model on validation and test set
    val_result = trainer.test(model, dataloaders=val_loader, verbose=False)
    test_result = trainer.test(model, dataloaders=test_loader, verbose=False)
    mse_result = {"test": test_result[0]["test_mse"], "val": val_result[0]["test_mse"]}
    mae_result = {"test": test_result[0]["test_mae"], "val": val_result[0]["test_mae"]}

    if args.save_pred_plot:
        model.eval()
        batch_x, batch_y, _, _ = next(iter(test_loader))
        model_device = next(model.parameters()).device
        batch_x = batch_x.float().to(model_device)
        batch_y = batch_y[:, -args.pred_len:, :].float().to(model_device)
        with torch.no_grad():
            outputs = model(batch_x)
            outputs = outputs[:, -args.pred_len:, :]
        plot_dir = args.plot_dir if args.plot_dir else CHECKPOINT_PATH
        os.makedirs(plot_dir, exist_ok=True)
        plot_path = os.path.join(
            plot_dir,
            f"pred_vs_true_{os.path.basename(args.data_path).split('.')[0]}.png"
        )
        save_prediction_plot(
            outputs.cpu(),
            batch_y.cpu(),
            plot_path,
            channel=args.plot_channel,
            max_points=args.plot_max_points,
            inv_transform=getattr(test_data, "inverse_transform", None)
        )
        if args.save_global_plot:
            global_plot_path = os.path.join(
                plot_dir,
                f"pred_vs_true_global_{os.path.basename(args.data_path).split('.')[0]}.png"
            )
            save_global_prediction_plot(
                model,
                test_loader,
                global_plot_path,
                channel=args.plot_channel,
                inv_transform=getattr(test_data, "inverse_transform", None),
                max_points=args.global_plot_max_points,
                frac=args.global_plot_frac
            )
        if args.save_test_trend_plot:
            trend_plot_path = os.path.join(
                plot_dir,
                f"pred_vs_true_test_trend_{os.path.basename(args.data_path).split('.')[0]}.png"
            )
            save_test_trend_plot(
                model,
                test_data,
                test_loader,
                trend_plot_path,
                channel=args.plot_channel,
                inv_transform=getattr(test_data, "inverse_transform", None),
                max_points=args.test_trend_max_points
            )

    return model, mse_result, mae_result


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Data args...
    parser.add_argument('--data', type=str, default='custom', help='dataset type')
    parser.add_argument('--root_path', type=str, default='/home/huhuajin/Affirm/datasets/ETT-small',
                        help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')

    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')

    # forecasting lengths
    parser.add_argument('--seq_len', type=int, default=256, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')
    parser.add_argument('--n1',type=int,default=512,help='First Embedded representation')
    parser.add_argument('--n2',type=int,default=128,help='Second Embedded representation')

    # optimization
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--pretrain_epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=8, help='batch size of train input data')
    parser.add_argument('--seed', type=int, default=51)

    # model
    parser.add_argument('--emb_dim', type=int, default=128, help='dimension of model')
    parser.add_argument('--depth', type=int, default=2, help='num of layers')
    parser.add_argument('--dropout', type=float, default=0.7, help='dropout value')
    parser.add_argument('--patch_size', type=int, default=32, help='size of patches')
    parser.add_argument('--mask_ratio', type=float, default=0.4)

    # Affirm components:
    parser.add_argument('--load_from_pretrained', type=str2bool, default=False, help='False: without pretraining')
    parser.add_argument('--ICB', type=str2bool, default=False)
    parser.add_argument('--Mamba', type=str2bool, default=True)
    parser.add_argument('--AFFB', type=str2bool, default=True)
    parser.add_argument('--adaptive_filter', type=str2bool, default=True)
    parser.add_argument('--gap_interp', type=str2bool, default=True, help='interpolate gaps within threshold')
    parser.add_argument('--gap_days', type=int, default=3, help='max gap days to keep within a segment')
    parser.add_argument('--interp_limit', type=int, default=3, help='max consecutive gaps to interpolate')
    parser.add_argument('--interp_freq', type=str, default='', help='freq for interpolation reindex (e.g., D, H)')
    parser.add_argument('--save_pred_plot', type=str2bool, default=True, help='save prediction plot')
    parser.add_argument('--plot_channel', type=int, default=0, help='channel index to plot')
    parser.add_argument('--plot_max_points', type=int, default=400, help='max points to plot')
    parser.add_argument('--plot_dir', type=str, default='', help='override output dir for plots')
    parser.add_argument('--save_global_plot', type=str2bool, default=True, help='save global test plot')
    parser.add_argument('--global_plot_max_points', type=int, default=0, help='max points in global plot (0 = all)')
    parser.add_argument('--global_plot_frac', type=float, default=1.0, help='fraction of global test plot to keep (e.g., 0.2 for last 20%)')
    parser.add_argument('--save_test_trend_plot', type=str2bool, default=True, help='save test trend plot aligned to dates')
    parser.add_argument('--test_trend_max_points', type=int, default=0, help='max points in test trend plot (0 = all)')
    
    # METHOD
    parser.add_argument('--ch_ind', type=int, default=0, help='Channel Independence; True 1 False 0')
    parser.add_argument('--residual', type=int, default=1, help='Residual Connection; True 1 False 0')
    parser.add_argument('--d_state', type=int, default=16, help='d_state parameter of Mamba')
    parser.add_argument('--d_conv_1', type=int, default=2, help='d_conv parameter of Mamba')
    parser.add_argument('--d_conv_2', type=int, default=4, help='d_conv parameter of Mamba')
    parser.add_argument('--e_fact', type=int, default=1, help='expand factor parameter of Mamba')
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size') #Use this hyperparameter as the number of channels
    # parser.add_argument('--dropout', type=float, default=0.0, help='dropout')
    # parser.add_argument('--embed', type=str, default='timeF',
                        # help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--do_predict', action='store_true',default=False,help='whether to predict unseen future data')

    args = parser.parse_args()

    if args.Mamba and Mamba is None:
        print(f"Mamba import failed ({_MAMBA_IMPORT_ERROR}); disabling Mamba blocks.")
        args.Mamba = False

    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    # load from checkpoint
    run_description = f"{args.data_path.split('.')[0]}_emb{args.emb_dim}_d{args.depth}_ps{args.patch_size}"
    run_description += f"_pl{args.pred_len}_bs{args.batch_size}_mr{args.mask_ratio}"
    run_description += f"_AFFB_{args.AFFB}_AF_{args.adaptive_filter}_ICB_{args.ICB}_preTr_{args.load_from_pretrained}"
    run_description += f"_{datetime.datetime.now().strftime('%H_%M')}"
    print(f"========== {run_description} ===========")

    CHECKPOINT_PATH = f"lightning_logs/{run_description}"
    pretrain_checkpoint_callback = ModelCheckpoint(
        dirpath=CHECKPOINT_PATH,
        save_top_k=1,
        filename='pretrain-{epoch}',
        monitor='val_loss',
        mode='min'
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=CHECKPOINT_PATH,
        save_top_k=1,
        monitor='val_mse',
        mode='min'
    )

    # Save a copy of this file and configs file as a backup
    save_copy_of_files(checkpoint_callback)

    # Ensure that all operations are deterministic on GPU (if used) for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # load datasets ...
    train_data, train_loader = data_provider(args, flag='train')
    vali_data, val_loader = data_provider(args, flag='val')
    test_data, test_loader = data_provider(args, flag='test')
    print("Dataset loaded ...")

    if args.load_from_pretrained:
        pretrained_model, best_model_path = pretrain_model()
    else:
        best_model_path = ''


    model, mse_result, mae_result = train_model(best_model_path)
    print("MSE results", mse_result)
    print("MAE  results", mae_result)

    # Save results into an Excel sheet ...
    df = pd.DataFrame({
        'MSE': mse_result,
        'MAE': mae_result
    })
    df.to_excel(os.path.join(CHECKPOINT_PATH, f"results_{datetime.datetime.now().strftime('%H_%M')}.xlsx"))

    # Append results into a text file ...
    os.makedirs("textOutput", exist_ok=True)
    f = open(f"textOutput/AFFIRM_{os.path.basename(args.data_path)}.txt", 'a')
    f.write(run_description + "  \n")
    f.write('MSE:{}, MAE:{}'.format(mse_result, mae_result))
    f.write('\n')
    f.write('\n')
    f.close()
