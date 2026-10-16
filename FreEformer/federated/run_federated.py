"""Federated-learning orchestrator for PM2.5 forecasting across 11 EPA stations.

Modes (share one backbone + one evaluation protocol so they are directly comparable):
  centralized  -- pool all stations' train windows, train one model (upper bound)
  independent  -- train a separate model per station (lower bound)
  fedavg       -- FedAvg over 11 station-clients
  fedavg_revin -- FedAvg but RevIN affine params stay local per client (FedBN-style)

Backbones: FrePatchTST3_fre_patch (frequency) or FrePatchTST3_temporal (time).
Small backbone by default; pass --full_size for the paper-scale GPU rerun.

Run from the FreEformer/ directory, e.g.:
  python federated/run_federated.py --mode fedavg --backbone FrePatchTST3_fre_patch --rounds 30
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import ConcatDataset, DataLoader

HERE = Path(__file__).resolve().parent
FREEFORMER_DIR = HERE.parent
sys.path.insert(0, str(FREEFORMER_DIR))  # so `model`, `data_provider`, `utils` import

from federated.fed_config import build_config, STATIONS, STATIONS_HOURLY, HOURLY_OVERRIDES
from federated.client import Client, build_model
from federated.aggregation import (fedavg, fedavg_personalized, FUSION_TIME_KEYS,
                                   band_keys_of, band_cutoff)

RESULTS_DIR = HERE / "results"
BACKBONE_SHORT = {"FrePatchTST3_fre_patch": "fre", "FrePatchTST3_temporal": "tmp",
                  "FrePatchTST3_mamba": "mamba", "FrePatchTST3_fusion": "fusion"}
MODE_SHORT = {"centralized": "central", "independent": "indep",
              "fedavg": "fedavg", "fedprox": "fedprox", "fedavg_revin": "fedavgRevIN",
              "fed_spectral": "fedSpectral", "fed_freqband": "fedFreqBand"}
FED_MODES = ("fedavg", "fedprox", "fedavg_revin", "fed_spectral", "fed_freqband")


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_all(seed=2023):
    torch.manual_seed(seed)
    np.random.seed(seed)


def train_on_loader(model, loader, config, device, n_epochs):
    """One optimizer across n_epochs on an arbitrary loader (centralized/independent)."""
    model.train()
    criterion = nn.L1Loss()
    opt = optim.Adam(model.parameters(), lr=config.learning_rate)
    f_dim = -1 if config.features == "MS" else 0
    for _ in range(n_epochs):
        for batch_x, batch_y, batch_x_mark, batch_y_mark in loader:
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            batch_x_mark = batch_x_mark.float().to(device)
            batch_y_mark = batch_y_mark.float().to(device)
            opt.zero_grad()
            out = model(batch_x, batch_x_mark, batch_y, batch_y_mark, mask=None)
            if isinstance(out, tuple):
                out = out[0]
            out = out[:, -config.pred_len:, f_dim:]
            tgt = batch_y[:, -config.pred_len:, f_dim:]
            loss = criterion(out, tgt)
            loss.backward()
            if getattr(config, "grad_clip", 0):
                nn.utils.clip_grad_norm_(model.parameters(), config.max_norm)
            opt.step()
    return model


def save_metrics(out_dir, model_name, loc, seed, seq_len, pred_len, metrics):
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model_name,
        "location": loc,
        "seed": seed,
        "lookback": seq_len,
        "horizon": pred_len,
        **metrics,
    }
    out = out_dir / f"{model_name}_{loc}_metrics.json"
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)
    return out


def parse_args():
    p = argparse.ArgumentParser(description="Federated PM2.5 forecasting harness.")
    p.add_argument("--mode", required=True,
                   choices=["centralized", "independent", "fedavg", "fedprox",
                            "fedavg_revin", "fed_spectral", "fed_freqband"])
    p.add_argument("--band_ratio", type=float, default=0.25,
                   help="Fed-FreqBand: fraction of low-freq rows shared globally (rest kept local)")
    p.add_argument("--mu", type=float, default=0.01,
                   help="FedProx proximal coefficient (mu/2 * ||w - w_global||^2)")
    p.add_argument("--eval_every", type=int, default=0,
                   help="evaluate the global model every N rounds to record convergence "
                        "(0 = only at the end)")
    p.add_argument("--backbone", default="FrePatchTST3_fre_patch",
                   choices=["FrePatchTST3_fre_patch", "FrePatchTST3_temporal",
                            "FrePatchTST3_mamba", "FrePatchTST3_fusion"])
    p.add_argument("--rounds", type=int, default=30, help="FL communication rounds")
    p.add_argument("--local_epochs", type=int, default=1, help="local epochs per round")
    p.add_argument("--epochs", type=int, default=30,
                   help="epochs for centralized / independent training")
    p.add_argument("--full_size", action="store_true", help="paper-scale backbone (GPU)")
    p.add_argument("--seed", type=int, default=2023, help="random seed (results go to results/seeds/s<seed>/)")
    p.add_argument("--pred_len", type=int, default=14,
                   help="forecast horizon; !=14 writes to results/horizon<pred_len>/ to avoid clobbering the main study")
    p.add_argument("--seq_len", type=int, default=14, help="lookback length (168 for hourly)")
    p.add_argument("--granularity", choices=["daily", "hourly"], default="daily",
                   help="daily=11 VIC stations (MS); hourly=43 QLD stations (univariate pm25)")
    # GPU / throughput overrides (default None = keep fed_config values, so existing
    # local runs are unaffected)
    p.add_argument("--batch_size", type=int, default=None, help="override batch size")
    p.add_argument("--num_workers", type=int, default=None, help="override dataloader workers")
    p.add_argument("--window_stride", type=int, default=None,
                   help="keep every N-th training window (hourly: 24 cuts ~24x redundant compute)")
    p.add_argument("--root_path", type=str, default=None, help="override dataset root path")
    p.add_argument("--skip_existing", action="store_true",
                   help="skip if all 11 station metrics for this (mode,backbone,seed) exist")
    return p.parse_args()


def main():
    args = parse_args()
    seed_all(args.seed)
    device = get_device()
    overrides = {}
    if args.batch_size is not None:
        overrides["batch_size"] = args.batch_size
    if args.num_workers is not None:
        overrides["num_workers"] = args.num_workers
    if args.window_stride is not None:
        overrides["window_stride"] = args.window_stride
    if args.root_path is not None:
        overrides["root_path"] = args.root_path
    overrides["pred_len"] = args.pred_len
    overrides["seq_len"] = args.seq_len

    if args.granularity == "hourly":
        stations = STATIONS_HOURLY
        # hourly task defaults (CLI --batch_size/--root_path still win if given)
        for k, v in HOURLY_OVERRIDES.items():
            overrides.setdefault(k, v)
    else:
        stations = STATIONS
    loc_suffix = "_merged" if args.granularity == "daily" else ""

    config = build_config(full_size=args.full_size, **overrides)
    bb_short = BACKBONE_SHORT[args.backbone]
    model_name = f"Fed-{MODE_SHORT[args.mode]}-{bb_short}"
    # route results so different tasks never clobber each other
    if args.granularity == "hourly":
        base_dir = RESULTS_DIR / "hourly" / f"sl{args.seq_len}_pl{args.pred_len}"
    elif args.pred_len == 14:
        base_dir = RESULTS_DIR
    else:
        base_dir = RESULTS_DIR / f"horizon{args.pred_len}"
    out_dir = base_dir / "seeds" / f"s{args.seed}"

    # resumability: skip if all 11 stations already done for this config
    if args.skip_existing:
        done = all((out_dir / f"{model_name}_{st}{loc_suffix}_metrics.json").exists()
                   for st in stations)
        if done:
            print(f"[{model_name}] seed={args.seed} already complete in {out_dir}, skipping.",
                  flush=True)
            return

    print(f"[{model_name}] device={device} backbone={args.backbone} seed={args.seed} "
          f"d_model={config.d_model} e_layers={config.e_layers} bs={config.batch_size} "
          f"workers={config.num_workers} -> {out_dir}", flush=True)

    t0 = time.time()
    # Every mode needs per-station data + eval, so build all clients up front.
    clients = []
    for st in stations:
        c = Client(st, args.backbone, config, device)
        clients.append(c)
        print(f"  client {st}: n_train={c.n_train} n_test={c.n_test}", flush=True)

    if args.mode == "centralized":
        pooled = ConcatDataset([c.train_set for c in clients])
        loader = DataLoader(pooled, batch_size=config.batch_size, shuffle=True,
                            num_workers=config.num_workers, drop_last=False)
        model = build_model(args.backbone, config).to(device)
        print(f"[centralized] pooled train windows={len(pooled)}, training {args.epochs} epochs",
              flush=True)
        train_on_loader(model, loader, config, device, args.epochs)
        state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        for c in clients:
            c.set_state(state)
            m = c.evaluate()
            save_metrics(out_dir, model_name, f"{c.station}{loc_suffix}", args.seed, args.seq_len, args.pred_len, m)
            print(f"  eval {c.station}: MAE={m['test_mae']} R2={m['test_r2']}", flush=True)

    elif args.mode == "independent":
        for c in clients:
            c.local_train(args.epochs)
            m = c.evaluate()
            save_metrics(out_dir, model_name, f"{c.station}{loc_suffix}", args.seed, args.seq_len, args.pred_len, m)
            print(f"  {c.station}: MAE={m['test_mae']} R2={m['test_r2']}", flush=True)

    elif args.mode in FED_MODES:
        init_state = {k: v.detach().cpu().clone()
                      for k, v in build_model(args.backbone, config).state_dict().items()}
        global_state = init_state

        # per-mode aggregate rule + client-load rule ("distribute")
        if args.mode == "fedavg_revin":
            exclude = ("revin_layer.",)
        elif args.mode == "fed_spectral":
            exclude = FUSION_TIME_KEYS
        else:
            exclude = None  # fedavg / fed_freqband average everything

        vfp = int((config.seq_len + 1) / 2 + 0.5)          # rfft frequency points
        band_keys = band_keys_of(init_state, vfp) if args.mode == "fed_freqband" else []
        cutoff = band_cutoff(vfp, args.band_ratio) if args.mode == "fed_freqband" else None
        if args.mode == "fed_freqband":
            print(f"  Fed-FreqBand: {len(band_keys)} band params, vfp={vfp}, "
                  f"low-freq rows [:{cutoff}] shared, [{cutoff}:] local", flush=True)

        def distribute(c):
            if args.mode == "fed_freqband":
                c.set_state_freqband(global_state, band_keys, cutoff)
            elif exclude is not None:                       # spectral / revin: keep local keys
                c.set_shared_state(global_state)
            else:                                           # fedavg: full replace
                c.set_state(global_state)

        def aggregate(states, counts):
            if exclude is not None:
                return fedavg_personalized(states, counts, exclude)
            return fedavg(states, counts)                   # fedavg / fedprox / fed_freqband

        def avg_test_mae():
            maes = []
            for c in clients:
                distribute(c)
                maes.append(c.evaluate()["test_mae"])
            return float(np.mean(maes))

        is_prox = args.mode == "fedprox"
        curve = []          # (round, avg_test_mae, elapsed_s) when --eval_every>0
        round_times = []
        for rnd in range(1, args.rounds + 1):
            r0 = time.time()
            states, counts = [], []
            for c in clients:
                distribute(c)
                if is_prox:
                    st, n = c.local_train(args.local_epochs, global_state=global_state, mu=args.mu)
                else:
                    st, n = c.local_train(args.local_epochs)
                states.append(st)
                counts.append(n)
            global_state = aggregate(states, counts)
            round_times.append(time.time() - r0)
            if args.eval_every and (rnd % args.eval_every == 0 or rnd == args.rounds):
                mae = avg_test_mae()
                curve.append((rnd, round(mae, 4), round(time.time() - t0, 1)))
                print(f"  round {rnd}/{args.rounds} | avg test MAE={mae:.4f} "
                      f"| {time.time()-t0:.0f}s elapsed", flush=True)
            elif rnd == 1 or rnd % 5 == 0 or rnd == args.rounds:
                print(f"  round {rnd}/{args.rounds} aggregated "
                      f"({time.time()-t0:.0f}s elapsed)", flush=True)

        # convergence summary (round within 1% of the best avg MAE = "converged")
        conv_round = None
        if curve:
            best = min(m for _, m, _ in curve)
            conv_round = next((r for r, m, _ in curve if m <= best * 1.01), curve[-1][0])
            conv_json = {
                "model": model_name, "seed": args.seed,
                "rounds": args.rounds, "avg_round_sec": round(float(np.mean(round_times)), 1),
                "total_min": round((time.time() - t0) / 60, 1),
                "converged_round": conv_round, "best_avg_mae": round(best, 4),
                "curve_round_mae_sec": curve,
            }
            (out_dir).mkdir(parents=True, exist_ok=True)
            with open(out_dir / f"{model_name}_convergence.json", "w") as f:
                json.dump(conv_json, f, indent=2)
            print(f"  >>> converged ~round {conv_round} (within 1% of best MAE {best:.4f}); "
                  f"avg {np.mean(round_times):.1f}s/round", flush=True)

        # final per-station eval: each client uses global-shared weights + its own local parts
        for c in clients:
            distribute(c)
            m = c.evaluate()
            save_metrics(out_dir, model_name, f"{c.station}{loc_suffix}", args.seed, args.seq_len, args.pred_len, m)
            print(f"  eval {c.station}: MAE={m['test_mae']} R2={m['test_r2']}", flush=True)

    total_min = (time.time() - t0) / 60
    print(f"[{model_name}] done in {total_min:.1f} min ({total_min/60:.2f} h)", flush=True)


if __name__ == "__main__":
    main()
