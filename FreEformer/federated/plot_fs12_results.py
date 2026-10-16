"""Aggregate + plot the FS12 hourly federated results into paper-ready figures/tables.

Reads results/hourly/sl<seq>_pl<pred>/seeds/s<seed>/:
  - Fed-<mode>-<bb>_convergence.json   (per-round avg test MAE, converged_round, timing)
  - Fed-<mode>-<bb>_<station>_metrics.json  (final per-station MAE/RMSE/R2)

Produces, under results/hourly/sl<seq>_pl<pred>/paper/:
  - convergence_curves.png       : avg test MAE vs round, one line per experiment
  - fs12_convergence_timing.md   : converged_round / avg_round_sec / total_h / best MAE
  - fs12_accuracy.md + .csv      : 13-station MAE/RMSE/R2 per experiment (+ station mean)
  - fs12_per_station_mae.md      : per-station MAE across the 4 experiments (wide)

Only the experiments whose files exist are included, so it runs fine before all
four finish (re-run it as more results arrive).

  python federated/plot_fs12_results.py --seq_len 168 --pred_len 24 --seed 2023
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results" / "hourly"

# model_name = Fed-<mode>-<bb>; give each a readable label + stable plot style
LABEL = {
    ("fedavg", "fre"):    "Transformer + FedAvg",
    ("fedprox", "fre"):   "Transformer + FedProx",
    ("fedavg", "mamba"):  "Mamba + FedAvg",
    ("fedprox", "mamba"): "Mamba + FedProx",
}
STYLE = {  # color by backbone, linestyle by aggregator
    ("fedavg", "fre"):    dict(color="#1f77b4", ls="-"),
    ("fedprox", "fre"):   dict(color="#1f77b4", ls="--"),
    ("fedavg", "mamba"):  dict(color="#d62728", ls="-"),
    ("fedprox", "mamba"): dict(color="#d62728", ls="--"),
}
ORDER = [("fedavg", "fre"), ("fedprox", "fre"), ("fedavg", "mamba"), ("fedprox", "mamba")]


def station_label(loc):
    return loc.replace("qld_", "").replace("_", " ")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq_len", type=int, default=168)
    ap.add_argument("--pred_len", type=int, default=24)
    ap.add_argument("--seed", type=int, default=2023)
    args = ap.parse_args()

    seed_dir = RESULTS / f"sl{args.seq_len}_pl{args.pred_len}" / "seeds" / f"s{args.seed}"
    out_dir = RESULTS / f"sl{args.seq_len}_pl{args.pred_len}" / "paper"
    out_dir.mkdir(parents=True, exist_ok=True)
    if not seed_dir.exists():
        print(f"No results dir: {seed_dir}")
        return

    # ---- 1. convergence curves + timing summary (from *_convergence.json) ----
    timing_rows = []
    plt.figure(figsize=(7, 4.5))
    for mode, bb in ORDER:
        cj = seed_dir / f"Fed-{mode}-{bb}_convergence.json"
        if not cj.exists():
            continue
        d = json.load(open(cj))
        curve = d.get("curve_round_mae_sec", [])
        if curve:
            rounds = [c[0] for c in curve]
            maes = [c[1] for c in curve]
            plt.plot(rounds, maes, marker="o", ms=4, label=LABEL[(mode, bb)],
                     **STYLE[(mode, bb)])
        timing_rows.append({
            "Experiment": LABEL[(mode, bb)],
            "Converged round": d.get("converged_round"),
            "Best avg MAE": d.get("best_avg_mae"),
            "Avg sec/round": d.get("avg_round_sec"),
            "Total (h)": round(d.get("total_min", 0) / 60, 2),
            "Rounds": d.get("rounds"),
        })
    plt.xlabel("Communication round")
    plt.ylabel("Avg test MAE (PM2.5)")
    plt.title(f"FS12 hourly — convergence (seq={args.seq_len}, pred={args.pred_len})")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    fig_path = out_dir / "convergence_curves.png"
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"[fig] {fig_path}")

    if timing_rows:
        tdf = pd.DataFrame(timing_rows)
        (out_dir / "fs12_convergence_timing.md").write_text(
            "# FS12 convergence & timing\n\n" + tdf.to_markdown(index=False) + "\n")
        print("[tbl] fs12_convergence_timing.md\n")
        print(tdf.to_string(index=False))

    # ---- 2. per-station accuracy (from *_<station>_metrics.json) ----
    recs = []
    for mode, bb in ORDER:
        for mf in sorted(seed_dir.glob(f"Fed-{mode}-{bb}_qld_*_metrics.json")):
            d = json.load(open(mf))
            recs.append({
                "Experiment": LABEL[(mode, bb)], "mode": mode, "bb": bb,
                "station": station_label(d["location"]),
                "MAE": d["test_mae"], "RMSE": d["test_rmse"], "R2": d["test_r2"],
            })
    if not recs:
        print("\nNo per-station metrics yet (experiments still running).")
        return
    df = pd.DataFrame(recs)
    df.to_csv(out_dir / "fs12_accuracy.csv", index=False)

    # averaged over 13 stations, per experiment
    avg = (df.groupby("Experiment")[["MAE", "RMSE", "R2"]].mean().round(4)
           .reindex([LABEL[k] for k in ORDER if LABEL[k] in df["Experiment"].values]))
    (out_dir / "fs12_accuracy.md").write_text(
        "# FS12 accuracy — 13-station average\n\n" + avg.to_markdown() + "\n")
    print("\n[tbl] fs12_accuracy.md (13-station average)\n")
    print(avg.to_string())

    # per-station MAE, wide (station x experiment)
    wide = df.pivot_table(index="station", columns="Experiment", values="MAE")
    wide = wide[[LABEL[k] for k in ORDER if LABEL[k] in wide.columns]]
    (out_dir / "fs12_per_station_mae.md").write_text(
        "# FS12 per-station MAE\n\n" + wide.round(4).to_markdown() + "\n")
    print(f"\n[tbl] fs12_per_station_mae.md ({len(wide)} stations)")
    print(f"\nAll outputs in {out_dir}")


if __name__ == "__main__":
    main()
