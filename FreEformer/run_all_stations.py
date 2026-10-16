"""Train FrePatchTST3_fre_patch across all 11 EPA stations for the Table 1 comparison.

Reuses the common trimmed feature set in ../Finalized_Dataset_common/ (8 weather
covariates + pm25 target) so the input space matches the other 5 baselines in
Models/. Copies each run's metrics.json into Models/results/ under the same
naming convention the aggregator script expects.

Usage:
    python run_all_stations.py [--overwrite]
"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

FREEFORMER_DIR = Path(__file__).resolve().parent
DATA_DIR = FREEFORMER_DIR.parent / "Finalized_Dataset_common"
DEST_DIR = FREEFORMER_DIR.parent / "Models" / "results"
LOG_DIR = FREEFORMER_DIR / "results" / "logs"

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
    parser = argparse.ArgumentParser(description="Run FreEformer across all 11 stations.")
    parser.add_argument("--overwrite", action="store_true", help="Re-run stations with existing metrics.")
    return parser.parse_args()


def main():
    args = parse_args()
    DEST_DIR.mkdir(exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    succeeded, failed = [], []

    for i, station in enumerate(STATIONS, start=1):
        model_id = f"{station}_merged"
        dest_metrics = DEST_DIR / f"FreEformer_{model_id}_metrics.json"
        tag = f"FreEformer/{station}"

        if dest_metrics.exists() and not args.overwrite:
            print(f"[{i}/{len(STATIONS)}] SKIP {tag}: {dest_metrics.name} already exists")
            continue

        cmd = [
            sys.executable, "run.py",
            "--is_training", "1",
            "--model_id", model_id,
            "--model", "FrePatchTST3_fre_patch",
            "--data", "custom",
            "--root_path", str(DATA_DIR) + "/",
            "--data_path", f"{station}_merged.csv",
            "--features", "MS", "--target", "pm25",
            "--enc_in", "9", "--dec_in", "9", "--c_out", "1",
            "--seq_len", "14", "--pred_len", "14",
            "--air_dataset", "1", "--window_split", "1", "--train_ratio", "0.8",
            "--train_epochs", "30", "--batch_size", "8", "--patience", "10",
            "--des", "table1", "--use_gpu", "1",
        ]

        print(f"[{i}/{len(STATIONS)}] RUN {tag} -> {' '.join(cmd)}")
        log_path = LOG_DIR / f"FreEformer_{station}.log"
        with open(log_path, "w") as log_file:
            result = subprocess.run(cmd, cwd=FREEFORMER_DIR, stdout=log_file, stderr=subprocess.STDOUT)

        if result.returncode != 0:
            print(f"[{i}/{len(STATIONS)}] FAILED {tag} (exit {result.returncode}), see {log_path}")
            failed.append((tag, str(log_path)))
            continue

        matches = sorted(FREEFORMER_DIR.glob(f"results/MSE_*_forecasting_{model_id}_*"),
                          key=lambda p: p.stat().st_mtime)
        if not matches:
            print(f"[{i}/{len(STATIONS)}] FAILED {tag}: no results folder found, see {log_path}")
            failed.append((tag, str(log_path)))
            continue

        src_metrics = matches[-1] / "metrics.json"
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
