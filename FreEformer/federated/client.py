"""A federated client = one EPA station: its local data + a model instance.

Reuses FreEformer's data_provider (per-station Dataset_Custom) and model variants
directly. Replicates the core train/eval logic of experiments/exp_forecast.py so
we can drive one local epoch at a time and exchange weights for FedAvg.
"""
import copy
import importlib

import numpy as np
import torch
import torch.nn as nn
from torch import optim

from sklearn.metrics import r2_score

from data_provider.data_factory import data_provider
from utils.metrics import metric


def build_model(backbone, config):
    """Instantiate model.<backbone>.Model(config)."""
    module = importlib.import_module(f"model.{backbone}")
    return module.Model(config).float()


class Client:
    def __init__(self, station, backbone, config, device):
        self.station = station
        self.backbone = backbone
        self.device = device
        # per-station config (only data_path differs)
        self.config = copy.copy(config)
        self.config.data_path = f"{station}{getattr(config, 'file_suffix', '_merged.csv')}"

        self.train_set, self.train_loader = data_provider(self.config, flag="train")
        self.test_set, self.test_loader = data_provider(self.config, flag="test")
        # subsample TRAIN windows: consecutive hourly windows overlap ~167/168 and are
        # highly redundant; keeping every `window_stride`-th start cuts compute with
        # negligible information loss. Test set is kept full for accurate metrics.
        ws = int(getattr(config, "window_stride", 1))
        if ws > 1 and getattr(self.train_set, "_index_map", None) is not None:
            self.train_set._index_map = self.train_set._index_map[::ws]
            if hasattr(self.train_set, "window_end_dates"):
                self.train_set.window_end_dates = self.train_set.window_end_dates[::ws]
        self.n_train = len(self.train_set)
        self.n_test = len(self.test_set)

        # Model is kept on CPU and moved to the GPU only while THIS client is training
        # or being evaluated, then moved back. With many clients this keeps peak GPU
        # memory at one model instead of all of them (avoids OOM on 16/24GB cards).
        self.model = build_model(backbone, self.config)
        self.criterion = nn.L1Loss()  # == WeightedL1Loss(alpha=0, 'L1'); windows are NaN-free

    def _free_gpu(self):
        self.model.to("cpu")
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    # ---- weight helpers ----
    def get_state(self):
        return {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}

    def set_state(self, state_dict, strict=True):
        self.model.load_state_dict(state_dict, strict=strict)

    def set_shared_state(self, shared_state):
        """Load only the keys present in shared_state, keep the rest (e.g. local RevIN
        for FedBN, or the whole time branch for Fed-Spectral)."""
        own = self.model.state_dict()
        own.update({k: v for k, v in shared_state.items() if k in own})
        self.model.load_state_dict(own, strict=True)

    def set_state_freqband(self, global_state, band_keys, cutoff):
        """Fed-FreqBand distribution: adopt the global (averaged) weights everywhere,
        but for each frequency-output layer in `band_keys` keep this client's OWN
        high-frequency rows [cutoff:] (station-specific), taking only the low-frequency
        rows [:cutoff] from the global model."""
        own = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
        new = {k: v.clone() for k, v in global_state.items()}
        for bk in band_keys:
            if bk in new and bk in own:
                new[bk][cutoff:] = own[bk][cutoff:]  # keep local high-freq rows
        self.model.load_state_dict(new, strict=True)

    # ---- training ----
    def _forward_slice(self, batch_x, batch_x_mark, batch_y, batch_y_mark):
        out = self.model(batch_x, batch_x_mark, batch_y, batch_y_mark, mask=None)
        if isinstance(out, tuple):
            out = out[0]
        f_dim = -1 if self.config.features == "MS" else 0
        out = out[:, -self.config.pred_len:, f_dim:]
        tgt = batch_y[:, -self.config.pred_len:, f_dim:]
        return out, tgt

    def local_train(self, n_local_epochs, global_state=None, mu=0.0):
        """One or more local epochs. For FedProx, pass the current global_state and
        mu>0 to add the proximal term (mu/2)*||w - w_global||^2 to each step's loss,
        which keeps the local model from drifting too far from the global one."""
        self.model.to(self.device)
        self.model.train()
        opt = optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        prox_ref = None
        if global_state is not None and mu > 0:
            prox_ref = {n: global_state[n].to(self.device)
                        for n, _ in self.model.named_parameters() if n in global_state}
        for _ in range(n_local_epochs):
            for batch_x, batch_y, batch_x_mark, batch_y_mark in self.train_loader:
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                opt.zero_grad()
                out, tgt = self._forward_slice(batch_x, batch_x_mark, batch_y, batch_y_mark)
                loss = self.criterion(out, tgt)
                if prox_ref is not None:
                    prox = sum(((p - prox_ref[n]) ** 2).sum()
                               for n, p in self.model.named_parameters() if n in prox_ref)
                    loss = loss + (mu / 2.0) * prox
                loss.backward()
                if getattr(self.config, "grad_clip", 0):
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_norm)
                opt.step()
        state = self.get_state()
        self._free_gpu()       # move model back to CPU so the next client has room
        return state, self.n_train

    # ---- evaluation (PM2.5 space, matches Table 1) ----
    @torch.no_grad()
    def evaluate(self):
        self.model.to(self.device)
        self.model.eval()
        preds, trues = [], []
        for batch_x, batch_y, batch_x_mark, batch_y_mark in self.test_loader:
            batch_x = batch_x.float().to(self.device)
            batch_y = batch_y.float().to(self.device)
            batch_x_mark = batch_x_mark.float().to(self.device)
            batch_y_mark = batch_y_mark.float().to(self.device)
            out, tgt = self._forward_slice(batch_x, batch_x_mark, batch_y, batch_y_mark)
            preds.append(out.detach().cpu().numpy())
            trues.append(tgt.detach().cpu().numpy())

        self._free_gpu()       # done with the GPU for this client's eval
        preds = np.concatenate(preds, axis=0)  # [N, pred_len, 1]
        trues = np.concatenate(trues, axis=0)

        # inverse-transform the single target channel back to PM2.5 space
        shape = preds.shape
        preds = self.test_set.inverse_transform(preds.reshape(-1, shape[-1])).reshape(shape)
        trues = self.test_set.inverse_transform(trues.reshape(-1, shape[-1])).reshape(shape)

        mae, mse, rmse, mape, mspe, r2, pear, mase = metric(preds, trues)
        # standard global R^2 (variance of the whole test set), the publishable metric.
        # metric()'s r2 is a per-window R^2 that is pathologically harsh for short/smooth
        # windows; we keep it as test_r2_perwindow for reference.
        r2_global = r2_score(trues.reshape(-1), preds.reshape(-1))
        step_maes = [float(np.mean(np.abs(preds[:, i, :] - trues[:, i, :])))
                     for i in range(preds.shape[1])]
        return {
            "test_mae": round(float(mae), 4),
            "test_rmse": round(float(rmse), 4),
            "test_r2": round(float(r2_global), 4),
            "test_r2_perwindow": round(float(r2), 4),
            "step_maes": [round(m, 4) for m in step_maes],
        }
