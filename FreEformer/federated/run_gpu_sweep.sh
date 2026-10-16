#!/usr/bin/env bash
# Stage-2 full-size federated sweep for a CUDA GPU pod.
#
# Runs every (seed x backbone x mode) at paper scale (d_model=512), resumable
# via --skip_existing so the job can be re-launched after a preemption.
#
# Usage (from the FreEformer/ directory):
#   bash federated/run_gpu_sweep.sh
# Override defaults with env vars:
#   PY=python3.10 SEEDS="2023 2024 2025" BS=32 WORKERS=8 EPOCHS=50 ROUNDS=50 \
#     bash federated/run_gpu_sweep.sh
#
# Prereqs on the pod (see README_gpu.md):
#   - pip install -r requirements.txt   (timm, reformer_pytorch, einops, ...)
#   - Finalized_Dataset_common/ present at ../Finalized_Dataset_common/ (repo root),
#     or point --root_path via ROOT env var below.
set -u

PY="${PY:-python}"
SEEDS="${SEEDS:-2023 2024 2025}"
BACKBONES="${BACKBONES:-FrePatchTST3_fre_patch FrePatchTST3_temporal}"
MODES="${MODES:-centralized independent fedavg fedavg_revin}"
BS="${BS:-32}"            # GPU can afford a bigger batch than the local bs=8
WORKERS="${WORKERS:-8}"
EPOCHS="${EPOCHS:-30}"    # centralized / independent
ROUNDS="${ROUNDS:-30}"    # fedavg / fedavg_revin
LOCAL_EPOCHS="${LOCAL_EPOCHS:-1}"
ROOT="${ROOT:-../Finalized_Dataset_common/}"

mkdir -p federated/results/logs
echo "PY=$PY SEEDS=[$SEEDS] BS=$BS WORKERS=$WORKERS EPOCHS=$EPOCHS ROUNDS=$ROUNDS ROOT=$ROOT"

for SEED in $SEEDS; do
  for BB in $BACKBONES; do
    bb=$([ "$BB" = "FrePatchTST3_fre_patch" ] && echo fre || echo tmp)
    for mode in $MODES; do
      log="federated/results/logs/${mode}_${bb}_s${SEED}_full.log"
      echo "########## seed=$SEED bb=$bb mode=$mode ##########"
      common="--mode $mode --backbone $BB --seed $SEED --full_size \
              --batch_size $BS --num_workers $WORKERS --root_path $ROOT --skip_existing"
      if [ "$mode" = "centralized" ] || [ "$mode" = "independent" ]; then
        $PY federated/run_federated.py $common --epochs $EPOCHS > "$log" 2>&1
      else
        $PY federated/run_federated.py $common --rounds $ROUNDS --local_epochs $LOCAL_EPOCHS > "$log" 2>&1
      fi
      echo "  -> exit $? ; $(grep -E 'done in|skipping' "$log" | tail -1)"
    done
  done
  echo "===== SEED $SEED COMPLETE ====="
done
echo "ALL DONE. Aggregate with: $PY federated/aggregate_federated.py"
