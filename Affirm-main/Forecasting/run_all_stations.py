"""Train AFFIRM across all 11 EPA stations for the Table 1 comparison.

Reuses the common trimmed feature set in ../../Finalized_Dataset_common/ (8 weather
covariates + pm25 target) so the input space matches the other baselines. Each run
writes results/AFFIRM_{station}_merged_metrics.json directly (already matching the
naming convention Models/aggregate_baseline_results.py expects), so this script just
copies it into Models/results/.

Usage:
    python run_all_stations.py [--overwrite]
"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

AFFIRM_DIR = Path(__file__).resolve().parent
DATA_DIR = AFFIRM_DIR.parent.parent / "Finalized_Dataset_common"
DEST_DIR = AFFIRM_DIR.parent.parent / "Models" / "results"
LOG_DIR = AFFIRM_DIR / "results" / "logs"

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


def parse_args():
    parser = argparse.ArgumentParser(description="Run AFFIRM across all 11 stations.")
    parser.add_argument("--overwrite", action="store_true", help="Re-run stations with existing metrics.")
    return parser.parse_args()


def main():
    args = parse_args()
    DEST_DIR.mkdir(exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    succeeded, failed = [], []

    for i, station in enumerate(STATIONS, start=1):
        loc_name = f"{station}_merged"
        dest_metrics = DEST_DIR / f"AFFIRM_{loc_name}_metrics.json"
        tag = f"AFFIRM/{station}"

        if dest_metrics.exists() and not args.overwrite:
            print(f"[{i}/{len(STATIONS)}] SKIP {tag}: {dest_metrics.name} already exists")
            continue

        cmd = [
            sys.executable, "Affirm_Forecasting.py",
            "--data", "custom",
            "--root_path", str(DATA_DIR) + "/",
            "--data_path", f"{loc_name}.csv",
            "--features", "MS", "--target", "pm25", "--enc_in", "9",
            "--seq_len", "14", "--label_len", "0", "--pred_len", "14",
            "--patch_size", "14",
            "--gap_interp", "True", "--window_split", "True",
            "--gap_days", "3", "--interp_limit", "3", "--interp_freq", "D",
            "--train_ratio", "0.8", "--val_ratio", "0.1",
            "--batch_size", "8",
        ]

        print(f"[{i}/{len(STATIONS)}] RUN {tag} -> {' '.join(cmd)}")
        log_path = LOG_DIR / f"AFFIRM_{station}.log"
        src_metrics = AFFIRM_DIR / "results" / f"AFFIRM_{loc_name}_metrics.json"
        src_metrics.unlink(missing_ok=True)  # so a stale file can't be mistaken for a fresh success
        with open(log_path, "w") as log_file:
            result = subprocess.run(cmd, cwd=AFFIRM_DIR, stdout=log_file, stderr=subprocess.STDOUT)

        if result.returncode != 0 or not src_metrics.exists():
            print(f"[{i}/{len(STATIONS)}] FAILED {tag} (exit {result.returncode}), see {log_path}")
            failed.append((tag, str(log_path)))
            continue

        shutil.copy(src_metrics, dest_metrics)
        print(f"[{i}/{len(STATIONS)}] OK {tag}: copied {src_metrics} -> {dest_metrics}")
        succeeded.append(tag)

    print("\n" + "=" * 60)
    print(f"Done. {len(succeeded)} succeeded, {len(failed)} failed (of {len(STATIONS)}).")
    if failed:
        print("Failed stations:")
        for tag, log_path in failed:
            print(f"  - {tag}: {log_path}")


if __name__ == "__main__":
    main()
