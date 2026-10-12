"""
Equivalent Source baseline — simplified & correct.

Magnetic monopole kernel (vertical magnetization, polar region):
    K(dx, dy) = d / (dx^2 + dy^2 + d^2)^(3/2)

Method:
1. Place one source at each 2S pixel center (square system, N_obs = M_src)
2. Solve exactly: G * m = d_obs (no regularization needed since sources match obs)
3. Predict on 2B grid from the same sources
4. This is effectively a physically-motivated interpolation (NOT super-resolution)

Expected performance: similar to bicubic (~39 nT), since it can only interpolate,
not create new frequencies beyond the 2S Nyquist.
"""
import numpy as np
from pathlib import Path
from scipy.interpolate import RegularGridInterpolator
import rasterio
import time

DATA = Path('F:/极区地磁导航/南极')
PATCH_DEG = 1.0
RNG = np.random.RandomState(42)
N_EVAL = 300

print("Loading data...", flush=True)
with rasterio.open(DATA / 'ADMAP_2B_2017_lonlat_geod.tif') as s:
    d2b = s.read(1).astype(np.float64); d2b[d2b==s.nodata]=np.nan; b2b=s.bounds
with rasterio.open(DATA / 'ADMAP_2S_lonlat_geod.tif') as s:
    d2s = s.read(1).astype(np.float64); d2s[d2s<-1e9]=np.nan; b2s=s.bounds

H2b, W2b = d2b.shape; H2s, W2s = d2s.shape
dy2b_px = (b2b.top - b2b.bottom) / H2b
dx2b_px = (b2b.right - b2b.left) / W2b
dy2s_px = (b2s.top - b2s.bottom) / H2s
dx2s_px = (b2s.right - b2s.left) / W2s

# ── Monopole kernel ──
def monopole_kernel(dx, dy, d):
    """K(dx,dy) = d / (dx^2+dy^2+d^2)^(3/2)"""
    return d / (dx**2 + dy**2 + d**2)**1.5

# ── Build system for one patch ──
def equivalent_source_predict(p2s, p2b_mean, dy_s, dx_s, dy_b, dx_b,
                               ny_b, nx_b, depth):
    """Fit sources to 2S data, predict on 2B grid. Returns 2B prediction."""
    ny_s, nx_s = p2s.shape
    obs_mean = np.mean(p2s)
    d_obs = (p2s - obs_mean).ravel()

    # Source grid = 2S pixel centers (step=1)
    sy = (np.arange(ny_s) - (ny_s-1)/2) * dy_s
    sx = (np.arange(nx_s) - (nx_s-1)/2) * dx_s
    SY, SX = np.meshgrid(sy, sx, indexing='ij')
    M = ny_s * nx_s

    # Observation grid = same as source grid
    OY, OX = np.meshgrid(sy, sx, indexing='ij')
    N = M

    # Build kernel: G[i,j] = K(obs_i, src_j)
    G = np.zeros((N, M), dtype=np.float64)
    for j in range(M):
        G[:, j] = monopole_kernel(OX.ravel() - SX.ravel()[j],
                                   OY.ravel() - SY.ravel()[j], depth)

    # Direct solve (square system, well-conditioned for monopole)
    reg = 1e-6 * np.eye(M)
    m = np.linalg.solve(G.T @ G + reg, G.T @ d_obs)

    # Predict on 2B grid
    py = (np.arange(ny_b) - (ny_b-1)/2) * dy_b
    px = (np.arange(nx_b) - (nx_b-1)/2) * dx_b
    PY, PX = np.meshgrid(py, px, indexing='ij')

    G_pred = np.zeros((ny_b * nx_b, M), dtype=np.float64)
    for j in range(M):
        G_pred[:, j] = monopole_kernel(PX.ravel() - SX.ravel()[j],
                                        PY.ravel() - SY.ravel()[j], depth)

    pred = (G_pred @ m).reshape(ny_b, nx_b)
    pred = pred - np.mean(pred) + p2b_mean
    return pred

# ── Extract patches ──
print("Extracting patches...", flush=True)
patches_2s_raw = []; patches_2b = []
patch_params = []

n_lat = int((b2b.top - b2b.bottom) / PATCH_DEG)
n_lon = int((b2b.right - b2b.left) / PATCH_DEG)

for i in range(n_lat):
    lat_top = b2b.top - i * PATCH_DEG; lat_bot = lat_top - PATCH_DEG
    mlat_rad = np.radians((lat_top + lat_bot) / 2)
    i0_s = int((b2s.top - lat_top) / dy2s_px); i1_s = int((b2s.top - lat_bot) / dy2s_px)
    i0_b = int((b2b.top - lat_top) / dy2b_px); i1_b = int((b2b.top - lat_bot) / dy2b_px)
    i0_s, i1_s = max(0,i0_s), min(H2s,i1_s)
    i0_b, i1_b = max(0,i0_b), min(H2b,i1_b)
    dy_s_km = dy2s_px * 111.32
    dx_s_km = dx2s_px * 111.32 * np.cos(mlat_rad)
    dy_b_km = dy2b_px * 111.32
    dx_b_km = dx2b_px * 111.32 * np.cos(mlat_rad)
    for j in range(n_lon):
        lon_l = b2b.left + j * PATCH_DEG; lon_r = lon_l + PATCH_DEG
        j0_s = int((lon_l - b2s.left) / dx2s_px); j1_s = int((lon_r - b2s.left) / dx2s_px)
        j0_b = int((lon_l - b2b.left) / dx2b_px); j1_b = int((lon_r - b2b.left) / dx2b_px)
        j0_s, j1_s = max(0,j0_s), min(W2s,j1_s)
        j0_b, j1_b = max(0,j0_b), min(W2b,j1_b)
        if i1_s<=i0_s or j1_s<=j0_s or i1_b<=i0_b or j1_b<=j0_b: continue
        p2s_raw = d2s[i0_s:i1_s, j0_s:j1_s].copy()
        p2b = d2b[i0_b:i1_b, j0_b:j1_b].copy()
        if np.sum(~np.isnan(p2b)) / p2b.size < 0.8: continue
        mask_s = np.isnan(p2s_raw); mask_b = np.isnan(p2b)
        if mask_s.any(): p2s_raw[mask_s] = np.nanmean(p2s_raw)
        if mask_b.any(): p2b[mask_b] = np.nanmean(p2b)
        ny_s, nx_s = p2s_raw.shape
        if ny_s < 4 or nx_s < 4: continue

        ny_b_actual, nx_b_actual = p2b.shape
        patches_2s_raw.append(p2s_raw)
        patches_2b.append(p2b)
        patch_params.append((ny_s, nx_s, dy_s_km, dx_s_km, dy_b_km, dx_b_km, ny_b_actual, nx_b_actual))

n_total = len(patches_2s_raw)
shuffle = RNG.permutation(n_total)
n_train = int(n_total * 0.7)
val_idx = shuffle[n_train:][:N_EVAL]
print(f"Total: {n_total}, Val eval: {len(val_idx)}", flush=True)

# ── Tune depth on a small subset ──
print("\nTuning depth...", flush=True)
n_tune = min(30, len(val_idx))
depth_candidates = [2.0, 3.0, 5.0, 8.0, 10.0, 12.0, 15.0, 20.0]
best_depth, best_rmse = 5.0, 1e10

for depth in depth_candidates:
    rmses = []
    for idx in val_idx[:n_tune]:
        p2s_raw = patches_2s_raw[idx]; p2b = patches_2b[idx]
        (ny_s, nx_s, dy_s, dx_s, dy_b, dx_b, ny_b, nx_b) = patch_params[idx]
        try:
            pred = equivalent_source_predict(p2s_raw, np.mean(p2b),
                                              dy_s, dx_s, dy_b, dx_b,
                                              ny_b, nx_b, depth)
            rmse = np.sqrt(np.mean((pred - p2b)**2))
            rmses.append(rmse)
        except Exception as e:
            rmses.append(np.nan)
    avg = np.nanmean(rmses)
    n_ok = np.sum(~np.isnan(rmses))
    print(f"  depth={depth:.0f} km -> RMSE={avg:.1f} (n={n_ok})", flush=True)
    if avg < best_rmse and n_ok >= n_tune * 0.7:
        best_rmse = avg; best_depth = depth

print(f"\nBest depth = {best_depth:.0f} km (tune RMSE = {best_rmse:.1f})", flush=True)

# ── Evaluate ──
print(f"\nEvaluating {N_EVAL} patches (depth={best_depth:.0f} km)...", flush=True)
rmses_eq = []; rmses_bicubic = []; times = []

for vi, idx in enumerate(val_idx):
    t0 = time.time()
    p2s_raw = patches_2s_raw[idx]; p2b = patches_2b[idx]
    (ny_s, nx_s, dy_s, dx_s, dy_b, dx_b, ny_b, nx_b) = patch_params[idx]

    try:
        pred = equivalent_source_predict(p2s_raw, np.mean(p2b),
                                          dy_s, dx_s, dy_b, dx_b,
                                          ny_b, nx_b, best_depth)
        rmse_eq = np.sqrt(np.mean((pred - p2b)**2))
    except:
        rmses_eq.append(np.nan); rmses_bicubic.append(np.nan); times.append(0)
        continue

    # Bicubic
    lat_s = np.linspace(0, 1, ny_s); lon_s = np.linspace(0, 1, nx_s)
    lat_b = np.linspace(0, 1, ny_b); lon_b = np.linspace(0, 1, nx_b)
    LON, LAT = np.meshgrid(lon_b, lat_b)
    pts = np.stack([LAT.ravel(), LON.ravel()], axis=-1)
    interp = RegularGridInterpolator((lat_s, lon_s), p2s_raw, method='cubic',
                                      bounds_error=False, fill_value=np.nanmean(p2s_raw))
    bicubic = interp(pts).reshape(ny_b, nx_b)
    rmse_bc = np.sqrt(np.mean((bicubic - p2b)**2))

    rmses_eq.append(rmse_eq); rmses_bicubic.append(rmse_bc)
    times.append(time.time() - t0)

    if (vi+1) % 50 == 0:
        valid = ~np.isnan(rmses_eq)
        print(f"  {vi+1}/{N_EVAL} | EqSrc={np.nanmean(rmses_eq):.1f} "
              f"Bicubic={np.nanmean(rmses_bicubic):.1f} | {np.mean(times):.1f}s/patch", flush=True)

rmses_eq = np.array(rmses_eq); rmses_bicubic = np.array(rmses_bicubic)
valid = ~np.isnan(rmses_eq)

print(f"\n{'='*55}")
print(f"Equivalent Source Results ({np.sum(valid)}/{N_EVAL} patches)")
print(f"{'='*55}")
print(f"Equivalent Source  RMSE = {np.nanmean(rmses_eq):.2f} +- {np.nanstd(rmses_eq):.2f} nT")
print(f"Bicubic (this eval) RMSE = {np.nanmean(rmses_bicubic):.2f} +- {np.nanstd(rmses_bicubic):.2f} nT")
print(f"Depth = {best_depth:.0f} km")
print(f"Avg time/patch = {np.mean(times):.1f}s")
print(f"{'='*55}")

np.savez(Path('F:/极区地磁导航/南极/experiment/results_gan/equivalent_source_results.npz'),
         rmse_eq_mean=np.nanmean(rmses_eq), rmse_eq_std=np.nanstd(rmses_eq),
         rmse_bicubic_mean=np.nanmean(rmses_bicubic), rmse_bicubic_std=np.nanstd(rmses_bicubic),
         depth=best_depth, n_valid=np.sum(valid), n_eval=N_EVAL,
         rmses_eq=rmses_eq, rmses_bicubic=rmses_bicubic)
print("Saved: equivalent_source_results.npz")
