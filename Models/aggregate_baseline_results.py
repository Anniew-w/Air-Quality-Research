"""Aggregate all Models/results/*_metrics.json into a station x model comparison table.

Usage:
    python aggregate_baseline_results.py
"""
import json
from pathlib import Path

import pandas as pd

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def load_records():
    records = []
    for path in sorted(RESULTS_DIR.glob("*_metrics.json")):
        with open(path) as f:
            data = json.load(f)
        records.append({
            "model": data["model"],
            "location": data["location"],
            "test_mae": data["test_mae"],
            "test_rmse": data["test_rmse"],
            "test_r2": data["test_r2"],
        })
    return pd.DataFrame(records)


def main():
    df = load_records()
    if df.empty:
        print(f"No *_metrics.json files found under {RESULTS_DIR}")
        return

    mae_table = df.pivot(index="location", columns="model", values="test_mae")
    rmse_table = df.pivot(index="location", columns="model", values="test_rmse")

    # rank models within each station by MAE (1 = best), then average across stations
    rank_table = mae_table.rank(axis=1, method="average")
    avg_rank = rank_table.mean(axis=0).sort_values()
    avg_mae = mae_table.mean(axis=0).sort_values()
    avg_rmse = rmse_table.mean(axis=0).reindex(avg_mae.index)

    summary = pd.DataFrame({
        "avg_mae": avg_mae.round(4),
        "avg_rmse": avg_rmse.round(4),
        "avg_rank": avg_rank.reindex(avg_mae.index).round(2),
        "n_stations": mae_table.notna().sum().reindex(avg_mae.index),
    })

    out_path = RESULTS_DIR / "baseline_summary.csv"
    summary.to_csv(out_path)
    mae_table.to_csv(RESULTS_DIR / "baseline_mae_by_station.csv")

    print(f"Loaded {len(df)} records ({df['model'].nunique()} models x {df['location'].nunique()} stations).\n")
    print("Per-model summary (sorted by avg MAE, lower is better):\n")
    print(summary.to_markdown())
    print(f"\nSaved: {out_path}")
    print(f"Saved: {RESULTS_DIR / 'baseline_mae_by_station.csv'}")


if __name__ == "__main__":
    main()
