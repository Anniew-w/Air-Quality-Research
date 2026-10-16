"""Aggregate multi-seed federated results into comparison tables with mean +/- std.

Reads federated/results/seeds/s<seed>/Fed-*_metrics.json (one subdir per seed).
Produces, under federated/results/paper_tables/:
  - fed_summary_{fre,tmp}.md   : per-mode avg MAE/RMSE/R2 as mean+/-std over seeds
  - fed_per_station_mae_{fre,tmp}.md : per-station MAE (mean over seeds) for the 4 modes
  - fed_freq_vs_time.md        : FedAvg frequency vs time, paired over seeds
  - fed_full.csv               : long form (seed, mode, backbone, location, mae, rmse, r2)

Comparability note: within each seed all modes/backbones share initialisation and
data splits, so per-seed differences are paired; std across seeds = run-to-run stability.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

import argparse

RESULTS_DIR = Path(__file__).resolve().parent / "results"
# overridable via --seeds_dir (e.g. results/hourly/sl168_pl24/seeds for the hourly study)
SEEDS_DIR = RESULTS_DIR / "seeds"
OUT_DIR = RESULTS_DIR / "paper_tables"

MODE_ORDER = ["central", "indep", "fedavg", "fedavgRevIN", "fedSpectral", "fedFreqBand"]
MODE_LABEL = {"central": "Centralized (upper)", "indep": "Independent (lower)",
              "fedavg": "FedAvg", "fedavgRevIN": "FedAvg+RevIN",
              "fedSpectral": "Fed-Spectral", "fedFreqBand": "Fed-FreqBand"}
BB_LABEL = {"fre": "Frequency", "tmp": "Time", "fusion": "Fusion", "mamba": "Freq+Mamba"}


def station_label(loc):
    stem = loc.replace("_merged", "")
    parts = stem.split("_")
    sid = parts[-1] if parts[-1].isdigit() else ""
    name = " ".join(parts[:-1] if sid else parts)
    return f"{name} ({sid})" if sid else name


def load_long():
    recs = []
    for f in sorted(SEEDS_DIR.glob("s*/Fed-*_metrics.json")):
        d = json.load(open(f))
        _, mode, bb = d["model"].split("-")
        seed = d.get("seed", int(f.parent.name[1:]))
        recs.append({"seed": seed, "mode": mode, "backbone": bb,
                     "location": d["location"], "station": station_label(d["location"]),
                     "mae": d["test_mae"], "rmse": d["test_rmse"], "r2": d["test_r2"]})
    return pd.DataFrame(recs)


def summary_md(df, backbone):
    """Per mode: station-mean per seed, then mean+/-std over seeds."""
    sub = df[df.backbone == backbone]
    modes = [m for m in MODE_ORDER if m in sub["mode"].unique()]
    n_seeds = sub["seed"].nunique()
    lines = [f"| Mode | Avg MAE | Avg RMSE | Avg R2 |  (mean±std over {n_seeds} seeds) |",
             "|---|---|---|---|---|"]
    means = {}
    for m in modes:
        s = sub[sub["mode"] == m]
        # station-mean per seed -> distribution over seeds
        per_seed = s.groupby("seed")[["mae", "rmse", "r2"]].mean()
        means[m] = per_seed["mae"].mean()
        lines.append(
            f"| {MODE_LABEL[m]} "
            f"| {per_seed['mae'].mean():.3f} ± {per_seed['mae'].std(ddof=0):.3f} "
            f"| {per_seed['rmse'].mean():.3f} ± {per_seed['rmse'].std(ddof=0):.3f} "
            f"| {per_seed['r2'].mean():.3f} ± {per_seed['r2'].std(ddof=0):.3f} |  |")
    best = min(means.values())
    # mark best mode
    out = []
    for line in lines:
        for m in modes:
            if MODE_LABEL[m] in line and abs(means[m] - best) < 1e-9:
                line = line.replace(MODE_LABEL[m], f"**{MODE_LABEL[m]}**", 1)
        out.append(line)
    return "\n".join(out)


def per_station_md(df, backbone):
    """Per-station MAE averaged over seeds, for the 4 modes."""
    sub = df[df.backbone == backbone]
    modes = [m for m in MODE_ORDER if m in sub["mode"].unique()]
    piv = sub.pivot_table(index="station", columns="mode", values="mae",
                          aggfunc="mean").reindex(columns=modes)
    cols = [MODE_LABEL[m] for m in modes]
    lines = ["| Station | " + " | ".join(cols) + " |", "|" + "---|" * (len(cols) + 1)]
    for station, row in piv.iterrows():
        best = row.min()
        cells = [(f"**{v:.3f}**" if pd.notna(v) and abs(v - best) < 1e-9 else f"{v:.3f}")
                 for v in row]
        lines.append(f"| {station} | " + " | ".join(cells) + " |")
    avg = piv.mean(axis=0)
    best = avg.min()
    cells = [(f"**{v:.3f}**" if abs(v - best) < 1e-9 else f"{v:.3f}") for v in avg]
    lines.append("| **Average** | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def freq_vs_time_md(df):
    """Paired frequency-vs-time comparison under FedAvg, over seeds."""
    sub = df[df["mode"] == "fedavg"]
    if sub["backbone"].nunique() < 2:
        return None
    # per-seed station-mean MAE for each backbone
    per_seed = sub.groupby(["seed", "backbone"])["mae"].mean().unstack("backbone")
    per_seed = per_seed[[b for b in ["fre", "tmp"] if b in per_seed.columns]]
    diff = per_seed["tmp"] - per_seed["fre"]  # >0 means frequency better
    n = len(per_seed)
    wins = int((diff > 0).sum())

    lines = ["### FedAvg: Frequency vs Time backbone (station-mean MAE per seed)\n",
             "| Seed | Frequency | Time | Time−Freq (>0 ⇒ Freq better) |",
             "|---|---|---|---|"]
    for seed, row in per_seed.iterrows():
        lines.append(f"| {seed} | {row['fre']:.3f} | {row['tmp']:.3f} | {row['tmp']-row['fre']:+.3f} |")
    lines.append(f"| **mean±std** | {per_seed['fre'].mean():.3f}±{per_seed['fre'].std(ddof=0):.3f} "
                 f"| {per_seed['tmp'].mean():.3f}±{per_seed['tmp'].std(ddof=0):.3f} "
                 f"| {diff.mean():+.3f}±{diff.std(ddof=0):.3f} |")
    lines.append(f"\n**Frequency wins {wins}/{n} seeds** (station-mean MAE). "
                 f"Mean advantage {diff.mean():+.3f} μg/m³.")

    # also per-station win rate across seeds
    ps = sub.pivot_table(index=["station", "seed"], columns="backbone", values="mae")
    if {"fre", "tmp"}.issubset(ps.columns):
        st_wins = (ps["fre"] < ps["tmp"]).groupby("station").mean()  # fraction of seeds freq wins
        lines.append("\n### Per-station: fraction of seeds where Frequency beats Time\n")
        lines.append("| Station | Freq-win rate |")
        lines.append("|---|---|")
        for st, w in st_wins.items():
            lines.append(f"| {st} | {w:.0%} |")
    return "\n".join(lines)


def main():
    global SEEDS_DIR, OUT_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds_dir", default=None,
                    help="path to a seeds/ dir (e.g. results/hourly/sl168_pl24/seeds); "
                         "default = results/seeds")
    a = ap.parse_args()
    if a.seeds_dir:
        SEEDS_DIR = Path(a.seeds_dir)
        OUT_DIR = SEEDS_DIR.parent / "paper_tables"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_long()
    if df.empty:
        print(f"No results under {SEEDS_DIR}")
        return
    df.sort_values(["backbone", "mode", "station", "seed"]).to_csv(
        OUT_DIR / "fed_full.csv", index=False)
    seeds = sorted(df["seed"].unique())
    print(f"Loaded {len(df)} records | seeds={seeds} "
          f"modes={sorted(df['mode'].unique())} backbones={sorted(df['backbone'].unique())}\n")

    for bb in df["backbone"].unique():
        s = summary_md(df, bb)
        ps = per_station_md(df, bb)
        (OUT_DIR / f"fed_summary_{bb}.md").write_text(
            f"### Federated comparison — {BB_LABEL.get(bb, bb)} backbone\n\n{s}\n")
        (OUT_DIR / f"fed_per_station_mae_{bb}.md").write_text(
            f"### Per-station MAE (mean over seeds) — {BB_LABEL.get(bb, bb)} backbone\n\n{ps}\n")
        print(f"=== {BB_LABEL.get(bb, bb)} backbone ===\n{s}\n")

    fvt = freq_vs_time_md(df)
    if fvt:
        (OUT_DIR / "fed_freq_vs_time.md").write_text(fvt + "\n")
        print(fvt, "\n")
    print(f"Wrote tables to {OUT_DIR}")


if __name__ == "__main__":
    main()
