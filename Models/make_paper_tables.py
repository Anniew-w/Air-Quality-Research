"""Build publication-ready comparison tables from Models/results/*_metrics.json.

Produces, under Models/results/paper_tables/:
  - table1_per_station_mae.tex / .md    (station x model MAE, best per row bold)
  - table1_per_station_rmse.tex / .md   (station x model RMSE)
  - table1_summary.tex / .md            (avg MAE / RMSE / R2 / avg-rank per model)
  - table1_full.csv                     (long-form: location, model, mae, rmse, r2)

Usage:
    python make_paper_tables.py
"""
import json
import glob
from pathlib import Path

import pandas as pd

RESULTS_DIR = Path(__file__).resolve().parent / "results"
OUT_DIR = RESULTS_DIR / "paper_tables"

# Column order = display order in tables. Ordered lightweight -> the two authors' models last.
MODELS = ["LSTM", "DLinear", "TCN", "GNNTransformer", "PGN", "AFFIRM"]
# Pretty labels for the paper (edit here to rename columns).
MODEL_LABELS = {
    "LSTM": "LSTM",
    "DLinear": "DLinear",
    "TCN": "TCN",
    "GNNTransformer": "GNN-Transformer",
    "PGN": "PGN",
    "AFFIRM": "AFFIRM",
}


def station_label(loc: str) -> str:
    """'Melbourne_CBD_10239_merged' -> 'Melbourne CBD (10239)'."""
    stem = loc.replace("_merged", "")
    parts = stem.split("_")
    site_id = parts[-1] if parts[-1].isdigit() else ""
    name = " ".join(parts[:-1] if site_id else parts)
    return f"{name} ({site_id})" if site_id else name


def load_long():
    recs = []
    for f in sorted(RESULTS_DIR.glob("*_metrics.json")):
        d = json.load(open(f))
        if d["model"] not in MODELS:
            continue
        recs.append({
            "location": d["location"],
            "station": station_label(d["location"]),
            "model": d["model"],
            "mae": d["test_mae"],
            "rmse": d["test_rmse"],
            "r2": d["test_r2"],
        })
    return pd.DataFrame(recs)


def pivot(df, metric):
    p = df.pivot(index="station", columns="model", values=metric)
    return p.reindex(columns=MODELS)


def md_table(pivot_df, lower_is_better=True, decimals=3):
    """Markdown table with best value per row bolded, plus an Average row."""
    cols = [MODEL_LABELS[m] for m in pivot_df.columns]
    header = "| Station | " + " | ".join(cols) + " |"
    sep = "|" + "---|" * (len(cols) + 1)
    lines = [header, sep]
    for station, row in pivot_df.iterrows():
        best = row.min() if lower_is_better else row.max()
        cells = []
        for v in row:
            s = f"{v:.{decimals}f}"
            if pd.notna(v) and abs(v - best) < 1e-9:
                s = f"**{s}**"
            cells.append(s)
        lines.append(f"| {station} | " + " | ".join(cells) + " |")
    # average row
    avg = pivot_df.mean(axis=0)
    best = avg.min() if lower_is_better else avg.max()
    cells = []
    for v in avg:
        s = f"{v:.{decimals}f}"
        if abs(v - best) < 1e-9:
            s = f"**{s}**"
        cells.append(s)
    lines.append(f"| **Average** | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def latex_table(pivot_df, metric_name, lower_is_better=True, decimals=3):
    cols = pivot_df.columns
    ncol = len(cols)
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(rf"\caption{{Per-station test {metric_name} for 14-day PM$_{{2.5}}$ "
                 rf"forecasting (lookback 14, horizon 14). Best per station in \textbf{{bold}}; "
                 rf"lower is better.}}")
    lines.append(rf"\label{{tab:per-station-{metric_name.lower()}}}")
    lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.append(r"\begin{tabular}{l" + "c" * ncol + "}")
    lines.append(r"\toprule")
    header = "Station & " + " & ".join(MODEL_LABELS[m] for m in cols) + r" \\"
    lines.append(header)
    lines.append(r"\midrule")
    for station, row in pivot_df.iterrows():
        best = row.min() if lower_is_better else row.max()
        cells = []
        for v in row:
            s = f"{v:.{decimals}f}"
            if pd.notna(v) and abs(v - best) < 1e-9:
                s = rf"\textbf{{{s}}}"
            cells.append(s)
        lines.append(f"{station} & " + " & ".join(cells) + r" \\")
    lines.append(r"\midrule")
    avg = pivot_df.mean(axis=0)
    best = avg.min() if lower_is_better else avg.max()
    cells = []
    for v in avg:
        s = f"{v:.{decimals}f}"
        if abs(v - best) < 1e-9:
            s = rf"\textbf{{{s}}}"
        cells.append(s)
    lines.append(r"Average & " + " & ".join(cells) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def summary_frame(df):
    mae = pivot(df, "mae")
    rmse = pivot(df, "rmse")
    r2 = pivot(df, "r2")
    rank = mae.rank(axis=1, method="average")  # per-station MAE rank, 1 = best
    summ = pd.DataFrame({
        "Avg MAE": mae.mean(axis=0),
        "Avg RMSE": rmse.mean(axis=0),
        "Avg R2": r2.mean(axis=0),
        "Avg Rank (MAE)": rank.mean(axis=0),
    }).reindex(MODELS)
    summ.index = [MODEL_LABELS[m] for m in summ.index]
    return summ.sort_values("Avg MAE")


def summary_md(summ):
    lines = ["| Model | Avg MAE | Avg RMSE | Avg R2 | Avg Rank (MAE) |",
             "|---|---|---|---|---|"]
    best_mae = summ["Avg MAE"].min()
    for model, row in summ.iterrows():
        name = f"**{model}**" if abs(row["Avg MAE"] - best_mae) < 1e-9 else model
        lines.append(f"| {name} | {row['Avg MAE']:.3f} | {row['Avg RMSE']:.3f} | "
                     f"{row['Avg R2']:.3f} | {row['Avg Rank (MAE)']:.2f} |")
    return "\n".join(lines)


def summary_latex(summ):
    lines = [r"\begin{table}[t]", r"\centering",
             r"\caption{Overall baseline comparison across 11 EPA stations "
             r"(14-day PM$_{2.5}$ forecasting). Averaged over stations; best Avg MAE in \textbf{bold}.}",
             r"\label{tab:baseline-summary}",
             r"\begin{tabular}{lcccc}", r"\toprule",
             r"Model & Avg MAE & Avg RMSE & Avg R$^2$ & Avg Rank (MAE) \\", r"\midrule"]
    best_mae = summ["Avg MAE"].min()
    for model, row in summ.iterrows():
        name = rf"\textbf{{{model}}}" if abs(row["Avg MAE"] - best_mae) < 1e-9 else model
        lines.append(f"{name} & {row['Avg MAE']:.3f} & {row['Avg RMSE']:.3f} & "
                     f"{row['Avg R2']:.3f} & {row['Avg Rank (MAE)']:.2f} " + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def main():
    OUT_DIR.mkdir(exist_ok=True)
    df = load_long()
    n_models = df["model"].nunique()
    n_stations = df["location"].nunique()
    print(f"Loaded {len(df)} records ({n_models} models x {n_stations} stations).")

    df.sort_values(["station", "model"]).to_csv(OUT_DIR / "table1_full.csv", index=False)

    mae_p = pivot(df, "mae")
    rmse_p = pivot(df, "rmse")
    summ = summary_frame(df)

    (OUT_DIR / "table1_per_station_mae.md").write_text(
        "### Per-station test MAE (best per station in bold)\n\n" + md_table(mae_p) + "\n")
    (OUT_DIR / "table1_per_station_rmse.md").write_text(
        "### Per-station test RMSE (best per station in bold)\n\n" + md_table(rmse_p) + "\n")
    (OUT_DIR / "table1_summary.md").write_text(
        "### Overall comparison (averaged over 11 stations)\n\n" + summary_md(summ) + "\n")

    (OUT_DIR / "table1_per_station_mae.tex").write_text(latex_table(mae_p, "MAE") + "\n")
    (OUT_DIR / "table1_per_station_rmse.tex").write_text(latex_table(rmse_p, "RMSE") + "\n")
    (OUT_DIR / "table1_summary.tex").write_text(summary_latex(summ) + "\n")

    print(f"Wrote tables to {OUT_DIR}\n")
    print("=" * 70)
    print(summary_md(summ))
    print("\n")
    print(md_table(mae_p))


if __name__ == "__main__":
    main()
