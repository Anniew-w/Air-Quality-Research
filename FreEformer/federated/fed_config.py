"""Minimal config builder for the federated-learning harness.

Produces a SimpleNamespace holding exactly the attributes that the FreEformer
model variants (FrePatchTST3_fre_patch / _temporal) and data_provider read.
Small backbone by default so FL simulation is tractable on a Mac (MPS); pass
full_size=True to switch to the paper-scale backbone for a GPU-pod rerun.
"""
from types import SimpleNamespace

# 11 EPA stations = federated clients (data in ../Finalized_Dataset_common/).
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

# 13 QLD hourly clients that have all 12 FS12 covariates + pm25 at >=70% coverage
# (2021-2025, ~5 years). Data in ../Finalized_Dataset_hourly_common/<station>.csv, columns:
#   date, dbt, pm10, swd, sws, sigma60, relative_humidity, barometric_pressure_hpa,
#   wind_speed_std_dev, no2, no, nox, rain, pm25   (target last -> MS mode, enc_in=13).
STATIONS_HOURLY = [
    "qld_cannon_hill", "qld_clinton", "qld_deception_bay", "qld_flinders_view",
    "qld_mutdapilly", "qld_north_maclean", "qld_north_ward", "qld_rocklea",
    "qld_south_gladstone", "qld_southport", "qld_springwood", "qld_targinie",
    "qld_woolloongabba",
]

# Overrides applied on top of build_config() to switch the harness to the hourly
# multivariate task (seq_len/pred_len still come from CLI so they stay tunable).
HOURLY_OVERRIDES = dict(
    root_path="../Finalized_Dataset_hourly_common/",
    file_suffix=".csv",
    features="MS", target="pm25", enc_in=13, dec_in=13, c_out=1,   # 12 covariates + pm25
    freq="h", interp_freq="h",
    batch_size=128,       # moderate: enough gradient steps/epoch even with subsampled windows
    test_batch_size=256,  # eval was bs=1 (very slow at test windows/station)
)


def build_config(full_size=False, **overrides):
    """Return a SimpleNamespace with every field the model + data layer need."""
    cfg = dict(
        # --- task / data ---
        task_name="forecasting",
        data="custom",
        root_path="../Finalized_Dataset_common/",
        data_path="Melbourne_CBD_10239_merged.csv",  # overwritten per-station
        file_suffix="_merged.csv",   # per-station filename suffix (".csv" for hourly)
        features="MS",
        target="pm25",
        embed="timeF",   # timeenc=1 path (model ignores time marks anyway; avoids a
                          # pandas-version bug in the timeenc=0 branch of data_loader.py)
        freq="d",
        # --- window / horizon ---
        seq_len=14,
        label_len=0,
        pred_len=14,
        enc_in=9,        # 8 weather covariates + pm25 target
        dec_in=9,
        c_out=1,
        # --- air-quality gap handling (matches Table 1) ---
        air_dataset=1,
        window_split=1,
        train_ratio=0.8,
        val_ratio=0.0,
        gap_days=3,
        interp_limit=3,
        interp_freq="D",
        # --- optimisation ---
        batch_size=8,
        test_batch_size=1,
        num_workers=0,
        window_stride=1,   # keep every N-th training window (>1 subsamples redundant hourly windows)
        learning_rate=1e-4,
        loss_mode="L1",
        lossfun_alpha=0.0,
        grad_clip=0,
        max_norm=1e6,
        # --- backbone size (small by default) ---
        d_model=128,
        d_ff=256,
        n_heads=8,
        e_layers=1,
        dropout=0.1,
        activation="gelu",
        # --- FrePatchTST3-specific knobs ---
        embed_size=8,
        time_branch=0,
        d_conv=4,      # Mamba depthwise-conv width (FrePatchTST3_mamba only)
        expand=2,      # Mamba block expansion factor (FrePatchTST3_mamba only)
        temp_patch_len=16,
        temp_stride=8,
        factor=1,
        output_attention=False,
        attn_enhance=1,
        attn_softmax_flag=1,
        attn_weight_plus=0,
        attn_outside_softmax=0,
        # --- plotting/debug flags (read as attrs, values inert) ---
        plot_mat_flag=0,
        plot_grad_flag=0,
        plot_mat_label="dataset",
        CKA_flag=0,
    )
    if full_size:
        cfg.update(d_model=512, d_ff=2048, e_layers=2)
    cfg.update(overrides)
    return SimpleNamespace(**cfg)
