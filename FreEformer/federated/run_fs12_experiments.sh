#!/usr/bin/env bash
# FS12 hourly experiments: {Transformer, Mamba} x {FedAvg, FedProx}, 5-year data (2021-2025),
# 13 QLD clients, 12 covariates + pm25. Records convergence (avg test MAE per round ->
# converged_round) and wall-clock time per experiment. Run from FreEformer/ on a CUDA pod
# (Mamba needs mamba_ssm / CUDA).
#
#   bash federated/run_fs12_experiments.sh
#
# "Transformer" = FrePatchTST3_fre_patch (frequency + Transformer);
# "Mamba"       = FrePatchTST3_mamba     (frequency + Mamba).
set -u

PY="${PY:-python}"
SEED="${SEED:-2023}"
BS="${BS:-128}"; WORKERS="${WORKERS:-8}"; WSTRIDE="${WSTRIDE:-12}"
ROUNDS="${ROUNDS:-30}"; LOCAL_EPOCHS="${LOCAL_EPOCHS:-1}"
SEQ="${SEQ:-168}"; PRED="${PRED:-24}"; MU="${MU:-0.01}"; EVAL_EVERY="${EVAL_EVERY:-1}"

COMMON="--granularity hourly --seq_len $SEQ --pred_len $PRED --full_size \
        --batch_size $BS --num_workers $WORKERS --window_stride $WSTRIDE \
        --rounds $ROUNDS --local_epochs $LOCAL_EPOCHS --seed $SEED \
        --eval_every $EVAL_EVERY --skip_existing"
mkdir -p federated/results/logs
echo "FS12 experiments | ROUNDS=$ROUNDS eval_every=$EVAL_EVERY mu=$MU stride=$WSTRIDE bs=$BS"

run () {  # $1=mode $2=backbone $3=short
  local log="federated/results/logs/fs12_${1}_${3}_s${SEED}.log"
  echo "===== $1 / $3 ====="
  local t0=$(date +%s)
  if [ "$1" = "fedprox" ]; then
    $PY federated/run_federated.py --mode "$1" --backbone "$2" $COMMON --mu "$MU" > "$log" 2>&1
  else
    $PY federated/run_federated.py --mode "$1" --backbone "$2" $COMMON > "$log" 2>&1
  fi
  local dt=$(( $(date +%s) - t0 ))
  echo "  exit $? | wall $((dt/60))m$((dt%60))s | $(grep -E 'converged ~round|done in' "$log" | tail -2 | tr '\n' ' ')"
}

for bb_pair in "FrePatchTST3_fre_patch transformer" "FrePatchTST3_mamba mamba"; do
  set -- $bb_pair; BB=$1; SHORT=$2
  run fedavg  "$BB" "$SHORT"
  run fedprox "$BB" "$SHORT"
done
echo ""
echo "ALL DONE. Convergence JSONs: federated/results/hourly/sl${SEQ}_pl${PRED}/seeds/s${SEED}/*_convergence.json"
echo "Per-experiment wall time is printed above; also in each log's 'done in ... h' line."
