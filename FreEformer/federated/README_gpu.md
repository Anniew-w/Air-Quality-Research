# Stage-2: full-size federated sweep on a GPU pod

The harness auto-selects CUDA (`get_device()` prefers `cuda:0`), so no code change
is needed to use a GPU. Stage 2 reruns everything at **paper scale**
(`--full_size` → d_model=512, d_ff=2048, e_layers=2) to amplify effect sizes and
lift the absolute numbers vs the small local backbone.

## 1. Environment

```bash
cd FreEformer
pip install -r requirements.txt        # timm, reformer_pytorch, einops, fvcore, openpyxl, ...
python -c "import torch; print(torch.cuda.is_available())"   # must print True
```

## 2. Data

The clients read `../Finalized_Dataset_common/` relative to `FreEformer/` (11 files
`<Station>_merged.csv`, columns: Date + 8 weather covariates + pm25). Make sure that
folder sits next to `FreEformer/` on the pod, or pass a different path via the
`ROOT` env var (below) / `--root_path`.

## 3. Run the sweep

```bash
# from FreEformer/
bash federated/run_gpu_sweep.sh
```

Defaults: seeds `2023 2024 2025`, both backbones, all 4 modes, `--full_size`,
batch_size 32, 8 workers, 30 epochs/rounds. It is **resumable** — `--skip_existing`
skips any (mode,backbone,seed) whose 11 station files already exist, so you can
re-launch after a preemption. Override anything via env vars:

```bash
PY=python SEEDS="2023 2024 2025 2026 2027" BS=64 WORKERS=16 \
  EPOCHS=50 ROUNDS=50 bash federated/run_gpu_sweep.sh
```

Per-config logs land in `federated/results/logs/*_full.log`; metrics in
`federated/results/seeds/s<seed>/`.

## 4. Aggregate

```bash
python federated/aggregate_federated.py
```

Writes mean±std tables and the paired frequency-vs-time comparison to
`federated/results/paper_tables/`.

## Notes / knobs

- **Batch size** changes results (gradient noise), so keep it fixed across the whole
  stage-2 study (the sweep uses one `BS` for all configs). It differs from the local
  `bs=8` runs, so treat stage-2 as its own self-consistent study, not directly
  comparable cell-by-cell to the small-backbone numbers.
- To bring the **local seeds (s2023–s2026, small backbone)** onto the pod for a
  combined view, copy `federated/results/seeds/` over; but note they used a different
  backbone size, so report them separately from the full-size runs.
- Single-GPU only (uses `cuda:0`). Multi-GPU / multi-node is out of scope for this
  simulation-style FL study.
