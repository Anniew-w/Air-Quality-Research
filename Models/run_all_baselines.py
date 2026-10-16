"""Run all 6 baseline models across all 11 EPA station datasets.

Usage:
    python run_all_baselines.py                # full run, each script's default epochs
    python run_all_baselines.py --quick         # smoke test, 2 epochs per run
    python run_all_baselines.py --overwrite     # re-run combos that already have metrics.json
"""
import argparse
import subprocess
import sys
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parent
RESULTS_DIR = MODELS_DIR / "results"
LOGS_DIR = RESULTS_DIR / "logs"
DATASET_DIR = MODELS_DIR.parent / "Finalized_Dataset"

STATIONS = [
    "Alphington_10001",
    "Box_Hill_10042",
    "Brighton_10007",
    "Dandenong_10022",
    "Geelong_South_10107",
    "Macleod_10377",
    "Melbourne_CBD_10239",
    "Melton_10169",
    "Morwell_East_10017",
    "Morwell_South_10217",
    "Traralgon_10011",
]

# (script filename, metrics-file model-name prefix)
SCRIPTS = [
    ("lstm_running.py", "LSTM"),
    ("dlinear_running.py", "DLinear"),
    ("tcn_running.py", "TCN"),
    ("modernTCN_running.py", "ModernTCN"),
    ("gnn_transformer_running.py", "GNNTransformer"),
    ("pgn_running.py", "PGN"),
]


def parse_args():
    parser = argparse.ArgumentParser(description="Run all baseline scripts across all stations.")
    parser.add_argument("--quick", action="store_true", help="Smoke test with 2 epochs per run.")
    parser.add_argument("--overwrite", action="store_true", help="Re-run combos with existing metrics.json.")
    return parser.parse_args()


def main():
    args = parse_args()
    RESULTS_DIR.mkdir(exist_ok=True)
    LOGS_DIR.mkdir(exist_ok=True)

    combos = [(script, model_prefix, station) for script, model_prefix in SCRIPTS for station in STATIONS]
    total = len(combos)
    skipped, succeeded, failed = [], [], []

    for i, (script, model_prefix, station) in enumerate(combos, start=1):
        data_path = DATASET_DIR / f"{station}_merged.csv"
        metrics_path = RESULTS_DIR / f"{model_prefix}_{station}_merged_metrics.json"
        tag = f"{model_prefix}/{station}"

        if not data_path.exists():
            print(f"[{i}/{total}] SKIP {tag}: data file not found at {data_path}")
            skipped.append(tag)
            continue

        if metrics_path.exists() and not args.overwrite:
            print(f"[{i}/{total}] SKIP {tag}: metrics already exist ({metrics_path.name})")
            skipped.append(tag)
            continue

        cmd = [sys.executable, str(MODELS_DIR / script), "--data", str(data_path), "--name", data_path.stem]
        if args.quick:
            cmd += ["--epochs", "2"]

        print(f"[{i}/{total}] RUN {tag} -> {' '.join(cmd)}")
        log_path = LOGS_DIR / f"{model_prefix}_{station}.log"
        with open(log_path, "w") as log_file:
            result = subprocess.run(cmd, cwd=MODELS_DIR, stdout=log_file, stderr=subprocess.STDOUT)

        if result.returncode == 0:
            succeeded.append(tag)
        else:
            print(f"[{i}/{total}] FAILED {tag} (exit {result.returncode}), see {log_path}")
            failed.append((tag, str(log_path)))

    print("\n" + "=" * 60)
    print(f"Done. {len(succeeded)} succeeded, {len(skipped)} skipped, {len(failed)} failed (of {total}).")
    if failed:
        print("Failed combos:")
        for tag, log_path in failed:
            print(f"  - {tag}: {log_path}")


if __name__ == "__main__":
    main()
