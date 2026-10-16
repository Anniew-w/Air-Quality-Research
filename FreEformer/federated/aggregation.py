"""Federated aggregation strategies.

Baselines:
  fedavg               -- sample-weighted mean of every parameter.
  fedavg_personalized  -- mean of every parameter EXCEPT keys matching any of
                          `exclude_substrs` (those stay local per client).
                          RevIN personalization (FedBN-style) = default.

Novel (this project), motivated by the empirical finding that frequency
parameters are station-invariant while time parameters are station-specific:
  Fed-Spectral  -- share the frequency branch globally, keep the time branch
                   local. Realised as fedavg_personalized with the fusion
                   backbone's time-branch key substrings (FUSION_TIME_KEYS).
  Fed-FreqBand  -- aggregate everything, but for the frequency-output layers
                   (rows indexed by frequency point) share only the LOW-frequency
                   rows globally and keep the HIGH-frequency rows local. Helpers:
                   `band_keys_of`, `band_cutoff`; the per-client keep-high-freq
                   merge lives in Client.set_state_freqband.
"""
import torch

# Fusion backbone: substrings identifying the TIME branch (kept local by Fed-Spectral).
# 'encoder.' has the trailing dot so it does NOT match the frequency encoders
# 'encoder_fre_real'/'encoder_fre_imag'. The shared time-encoder tensor appears under
# both 'encoder.*' and 'temp_trans.1.*'; both are matched here so it stays fully local.
FUSION_TIME_KEYS = ("encoder.", "temp_trans.", "fc_time.")


def _weighted_average(states, sample_counts, keys):
    total = float(sum(sample_counts))
    out = {}
    for k in keys:
        acc = None
        for st, n in zip(states, sample_counts):
            term = st[k].float() * (n / total)
            acc = term if acc is None else acc + term
        out[k] = acc.to(states[0][k].dtype)
    return out


def fedavg(states, sample_counts):
    """Weighted average of all keys. `states` = list of CPU state_dicts."""
    return _weighted_average(states, sample_counts, list(states[0].keys()))


def fedavg_personalized(states, sample_counts, exclude_substrs=("revin_layer.",)):
    """Average only keys that contain none of `exclude_substrs`; excluded keys are
    omitted from the returned dict so each client keeps its own (personalized) values."""
    if isinstance(exclude_substrs, str):
        exclude_substrs = (exclude_substrs,)
    shared_keys = [k for k in states[0].keys()
                   if not any(sub in k for sub in exclude_substrs)]
    return _weighted_average(states, sample_counts, shared_keys)


def band_keys_of(state, freq_points):
    """Keys whose dim-0 equals the number of rfft frequency points (the frequency-
    output linear weight/bias, e.g. patch_real.3.weight / patch_imag.3.bias)."""
    return [k for k, v in state.items() if v.ndim >= 1 and v.shape[0] == freq_points]


def band_cutoff(freq_points, share_ratio):
    """Row index splitting low-freq (shared) from high-freq (local). Rows [0:cutoff]
    are the lowest frequencies (incl. DC / the diurnal cycle) shared globally."""
    return max(1, int(round(freq_points * share_ratio)))
