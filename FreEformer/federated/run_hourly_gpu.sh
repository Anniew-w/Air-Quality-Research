#!/usr/bin/env bash
# Hourly full-size federated sweep for a CUDA GPU pod (RunPod etc.).
#
# Univariate pm25, 43 QLD hourly clients, lookback=168h -> horizon=24h, paper-scale
# backbone (--full_size). Resumable via --skip_existing. Run from FreEformer/.
#
#   bash federated/run_hourly_gpu.sh
#
# Controlled-variable design:
#   Cohort A (backbone comparison): {fre_patch, temporal, fusion} x {centralized, independent, fedavg}
#   Cohort B (aggregator comparison, fusion): fusion x {fedavg_revin, fed_spectral, fed_freqband}
# Override with env vars: PY, SEEDS, BS, WORKERS, EPOCHS, ROUNDS, SEQ, PRED, BAND_RATIO.
set -u

PY="${PY:-python}"
SEEDS="${SEEDS:-2023}"
BS="${BS:-128}"; WORKERS="${WORKERS:-8}"; WSTRIDE="${WSTRIDE:-12}"
EPOCHS="${EPOCHS:-20}"; ROUNDS="${ROUNDS:-20}"; LOCAL_EPOCHS="${LOCAL_EPOCHS:-1}"
SEQ="${SEQ:-168}"; PRED="${PRED:-24}"; BAND_RATIO="${BAND_RATIO:-0.25}"

# window_stride=24 subsamples the highly-redundant overlapping hourly windows (~24x less
# compute, negligible info loss); big batch uses the free VRAM. Together these make the
# full-size hourly sweep tractable on a budget.
COMMON="--granularity hourly --seq_len $SEQ --pred_len $PRED --full_size \
        --batch_size $BS --num_workers $WORKERS --window_stride $WSTRIDE --skip_existing"
mkdir -p federated/results/logs
echo "hourly sweep | SEEDS=[$SEEDS] BS=$BS EPOCHS=$EPOCHS ROUNDS=$ROUNDS seq=$SEQ pred=$PRED"

run () {  # $1=mode $2=backbone $3=extra-args
  local mode=$1 bb=$2 extra=${3:-}
  local bbs; bbs=$(echo "$bb" | sed 's/FrePatchTST3_//')
  local log="federated/results/logs/hourly_${mode}_${bbs}_s${SEED}.log"
  echo "  [$SEED] $mode / $bbs"
  $PY federated/run_federated.py --mode "$mode" --backbone "$bb" --seed "$SEED" \
      $COMMON $extra > "$log" 2>&1
  echo "    -> exit $? ; $(grep -E 'done in|skipping' "$log" | tail -1)"
}

for SEED in $SEEDS; do
  echo "===== SEED $SEED ====="
  # Cohort A: backbone comparison (frequency vs time vs fusion)
  for bb in FrePatchTST3_fre_patch FrePatchTST3_temporal FrePatchTST3_fusion; do
    run centralized "$bb" "--epochs $EPOCHS"
    run independent "$bb" "--epochs $EPOCHS"
    run fedavg      "$bb" "--rounds $ROUNDS --local_epochs $LOCAL_EPOCHS"
  done
  # Cohort B: novel aggregators on the fusion backbone
  run fedavg_revin  FrePatchTST3_fusion "--rounds $ROUNDS --local_epochs $LOCAL_EPOCHS"
  run fed_spectral  FrePatchTST3_fusion "--rounds $ROUNDS --local_epochs $LOCAL_EPOCHS"
  run fed_freqband  FrePatchTST3_fusion "--rounds $ROUNDS --local_epochs $LOCAL_EPOCHS --band_ratio $BAND_RATIO"
  echo "===== SEED $SEED DONE ====="
done
echo "ALL DONE. Aggregate: $PY federated/aggregate_federated.py"
