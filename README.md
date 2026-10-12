# Spectral U-Net for enhancing the spatial resolution of Antarctic magnetic anomaly maps

Code and trained weights accompanying the manuscript *Spectral U-Net for enhancing the
spatial resolution of Antarctic magnetic anomaly maps* (Geophysical Journal International,
manuscript GJI-S-26-0536).

## Contents

- `train/train_ablations_spatial.py` — trains the full spectral U-Net and the four ablation
  variants under the spatial-block split (about 7,040 training, 1,056 validation and 504
  held-out patches).
- `eval/eval_ablation_mae.py` — per-patch RMSE and MAE for the full model and the ablations.
- `eval/eval_freq_methods_spatial.py`, `eval/eval_wiener.py`, `eval/eval_tikhonov.py`,
  `eval/eval_spectral_extrapolation.py` — frequency-domain baselines.
- `eval/eval_equivalent_source.py` — equivalent-source baseline.
- `eval/eval_mae_rmse_per_patch.py` — spatial-domain comparison (box plots and residual
  fields).
- `eval/eval_latitude_table.py` — per-latitude metrics along the 40.5°E meridian.
- `eval/reconstruct_all_methods_nogk.py` — reconstruction of the four held-out regions.
- `eval/boundary_depth_analysis.py`, `eval/synthetic_gap_blocks.py` — the two-layer proxy
  validation of gap filling.
- `weights/globalnorm_best.pth` — trained weights of the full model (global amplitude
  normalisation, about 4.0 million parameters).

## Data

The ADMAP-2B and ADMAP-2S magnetic anomaly grids are available from PANGAEA:
https://doi.org/10.1594/PANGAEA.965433

## Environment

Python 3 with PyTorch, NumPy, SciPy, rasterio and matplotlib; see `requirements.txt`.
A CUDA-capable GPU is used for training and inference.

## Note on paths

The scripts use absolute paths from the original working environment, set through the `DATA`
variable near the top of each file. Adjust this path to your local copy of the ADMAP grids
and the generated intermediate files before running.
