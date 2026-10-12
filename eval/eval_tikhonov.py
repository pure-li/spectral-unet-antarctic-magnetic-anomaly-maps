"""
Tikhonov regularized deconvolution baseline.

Tikhonov:  F_est = F_2S * H / (|H|^2 + lambda)
Wiener:   F_est = F_2S * H / (|H|^2 + K_reg(k))   <- frequency-dependent

lambda is a single constant — simpler but less flexible than Wiener.
"""
import numpy as np
from pathlib import Path
from scipy.interpolate import RegularGridInterpolator
import rasterio

DATA = Path('F:/极区地磁导航/南极')
PATCH_DEG = 1.0
RNG = np.random.RandomState(42)

print("Loading data...", flush=True)
gain_data = np.load(Path('F:/极区地磁导航/南极/experiment/results_wavenumber/spectral_gain.npz'))
k_gain = gain_data['k']; G_gain = gain_data['G_smooth']

with rasterio.open(DATA / 'ADMAP_2B_2017_lonlat_geod.tif') as s:
    d2b = s.read(1).astype(np.float64); d2b[d2b==s.nodata]=np.nan; b2b=s.bounds
with rasterio.open(DATA / 'ADMAP_2S_lonlat_geod.tif') as s:
    d2s = s.read(1).astype(np.float64); d2s[d2s<-1e9]=np.nan; b2s=s.bounds

H2b, W2b = d2b.shape; H2s, W2s = d2s.shape
dy2b_px = (b2b.top - b2b.bottom) / H2b
dx2b_px = (b2b.right - b2b.left) / W2b
dy2s_px = (b2s.top - b2s.bottom) / H2s
dx2s_px = (b2s.right - b2s.left) / W2s

n_lat = int((b2b.top - b2b.bottom) / PATCH_DEG)
n_lon = int((b2b.right - b2b.left) / PATCH_DEG)

print("Extracting patches...", flush=True)

F2S_list = []; F2B_list = []
spatial_truth = []; patch_meta = []

for i in range(n_lat):
    lat_top = b2b.top - i * PATCH_DEG; lat_bot = lat_top - PATCH_DEG
    mlat_rad = np.radians((lat_top + lat_bot) / 2)
    i0_s = int((b2s.top - lat_top) / dy2s_px); i1_s = int((b2s.top - lat_bot) / dy2s_px)
    i0_b = int((b2b.top - lat_top) / dy2b_px); i1_b = int((b2b.top - lat_bot) / dy2b_px)
    i0_s, i1_s = max(0,i0_s), min(H2s,i1_s)
    i0_b, i1_b = max(0,i0_b), min(H2b,i1_b)
    dy_s_km = dy2s_px * 111.32; dx_s_km = dx2s_px * 111.32 * np.cos(mlat_rad)
    for j in range(n_lon):
        lon_l = b2b.left + j * PATCH_DEG; lon_r = lon_l + PATCH_DEG
        j0_s = int((lon_l - b2s.left) / dx2s_px); j1_s = int((lon_r - b2s.left) / dx2s_px)
        j0_b = int((lon_l - b2b.left) / dx2b_px); j1_b = int((lon_r - b2b.left) / dx2b_px)
        j0_s, j1_s = max(0,j0_s), min(W2s,j1_s)
        j0_b, j1_b = max(0,j0_b), min(W2b,j1_b)
        if i1_s<=i0_s or j1_s<=j0_s or i1_b<=i0_b or j1_b<=j0_b: continue
        p2s = d2s[i0_s:i1_s, j0_s:j1_s].copy()
        p2b = d2b[i0_b:i1_b, j0_b:j1_b].copy()
        if np.sum(~np.isnan(p2b)) / p2b.size < 0.8: continue
        mask_s = np.isnan(p2s); mask_b = np.isnan(p2b)
        if mask_s.any(): p2s[mask_s] = np.nanmean(p2s)
        if mask_b.any(): p2b[mask_b] = np.nanmean(p2b)
        ny_s, nx_s = p2s.shape; ny_b, nx_b = p2b.shape
        if ny_s < 4 or nx_s < 4: continue

        a_s = p2s - np.mean(p2s); F_s = np.fft.rfft2(a_s)
        ky_s = np.fft.fftfreq(ny_s, d=dy_s_km); kx_s = np.fft.rfftfreq(nx_s, d=dx_s_km)
        KY_s, KX_s = np.meshgrid(ky_s, kx_s, indexing='ij')
        K_s = np.sqrt(KX_s**2 + KY_s**2)
        G2d = np.interp(K_s.ravel(), k_gain, G_gain, left=G_gain[0], right=G_gain[-1])
        G2d = G2d.reshape(K_s.shape); G2d = np.maximum(G2d, 0.1)
        F_s_corr = F_s * np.sqrt(G2d)
        p2s_corrected = np.fft.irfft2(F_s_corr, s=(ny_s, nx_s)) + np.mean(p2s)

        lat_s_arr = np.linspace(lat_top, lat_bot, ny_s)
        lon_s_arr = np.linspace(lon_l, lon_r, nx_s)
        lat_b_arr = np.linspace(lat_top, lat_bot, ny_b)
        lon_b_arr = np.linspace(lon_l, lon_r, nx_b)
        LON, LAT = np.meshgrid(lon_b_arr, lat_b_arr)
        pts = np.stack([LAT.ravel(), LON.ravel()], axis=-1)
        interp = RegularGridInterpolator(
            (lat_s_arr, lon_s_arr), p2s_corrected, method='cubic',
            bounds_error=False, fill_value=np.nanmean(p2s_corrected))
        p2s_on_b = interp(pts).reshape(ny_b, nx_b)

        a_s_on_b = p2s_on_b - np.mean(p2s_on_b)
        a_b = p2b - np.mean(p2b)
        F2S_list.append(np.fft.rfft2(a_s_on_b))
        F2B_list.append(np.fft.rfft2(a_b))
        spatial_truth.append(p2b.copy())
        patch_meta.append((float(np.mean(p2b)), ny_b, nx_b))

n_total = len(F2S_list)
shuffle = RNG.permutation(n_total)
n_train = int(n_total * 0.7)
train_idx = shuffle[:n_train]; val_idx = shuffle[n_train:]
print(f"Total: {n_total}, Train: {n_train}, Val: {len(val_idx)}", flush=True)

# ── Estimate H(k) from training data (same as Wiener) ──
print("Estimating radial |H(k)|...", flush=True)
n_kbins = 80
all_k = []; all_H = []
for idx in train_idx:
    F_in = F2S_list[idx]; F_tgt = F2B_list[idx]
    _, ny_b, nx_b = patch_meta[idx]
    ky = np.fft.fftfreq(ny_b); kx = np.fft.rfftfreq(nx_b)
    KY, KX = np.meshgrid(ky, kx, indexing='ij')
    K = np.sqrt(KX**2 + KY**2)
    eps = 1e-8 * max(np.abs(F_tgt).max(), 1.0)
    H = np.abs(F_in) / (np.abs(F_tgt) + eps)
    all_k.append(K.ravel()); all_H.append(np.clip(H.ravel(), 0, 10))

all_k = np.concatenate(all_k); all_H = np.concatenate(all_H)
kmax = all_k.max(); kmin = all_k[all_k > 0].min()
kbins = np.logspace(np.log10(kmin), np.log10(kmax), n_kbins + 1)
k_cen = np.sqrt(kbins[:-1] * kbins[1:])
H_radial = np.zeros(n_kbins); cnt = np.zeros(n_kbins)
dig = np.digitize(all_k, kbins) - 1
valid = (dig >= 0) & (dig < n_kbins)
np.add.at(H_radial, dig[valid], all_H[valid])
np.add.at(cnt, dig[valid], 1)
mask_h = cnt > 0
H_radial[mask_h] /= cnt[mask_h]
for ib in range(n_kbins):
    if cnt[ib] == 0:
        nearby = np.where(mask_h)[0]
        nn = nearby[np.argmin(np.abs(nearby - ib))]
        H_radial[ib] = H_radial[nn]

# ── Tune lambda (constant regularization) ──
print("Tuning lambda...", flush=True)
lam_candidates = np.logspace(-4, 1, 20)
best_lam, best_rmse = 0.1, 1e10
n_tune = min(150, len(val_idx))

for lam in lam_candidates:
    rmses = []
    for vi in val_idx[:n_tune]:
        F_in = F2S_list[vi]; mean_b, ny_b, nx_b = patch_meta[vi]
        truth = spatial_truth[vi]
        ky = np.fft.fftfreq(ny_b); kx = np.fft.rfftfreq(nx_b)
        KY, KX = np.meshgrid(ky, kx, indexing='ij')
        K = np.sqrt(KX**2 + KY**2)
        H_2d = np.interp(K.ravel(), k_cen, H_radial).reshape(K.shape)
        H2 = H_2d**2
        # Tikhonov: constant lambda, not frequency-dependent
        gain = H_2d / (H2 + lam)
        F_est = F_in * gain
        spat_est = np.fft.irfft2(F_est, s=(ny_b, nx_b)) + mean_b
        rmses.append(np.sqrt(np.mean((spat_est - truth)**2)))
    avg_rmse = np.mean(rmses)
    if avg_rmse < best_rmse:
        best_rmse = avg_rmse; best_lam = lam
    print(f"  lambda={lam:.4f} → RMSE={avg_rmse:.2f}", flush=True)

print(f"\nBest lambda = {best_lam:.4f} (tune RMSE = {best_rmse:.2f})", flush=True)

# ── Evaluate on full validation set ──
print("Evaluating on full validation set...", flush=True)
rmses = []
for vi in val_idx:
    F_in = F2S_list[vi]; mean_b, ny_b, nx_b = patch_meta[vi]
    truth = spatial_truth[vi]
    ky = np.fft.fftfreq(ny_b); kx = np.fft.rfftfreq(nx_b)
    KY, KX = np.meshgrid(ky, kx, indexing='ij')
    K = np.sqrt(KX**2 + KY**2)
    H_2d = np.interp(K.ravel(), k_cen, H_radial).reshape(K.shape)
    gain = H_2d / (H_2d**2 + best_lam)
    F_est = F_in * gain
    spat_est = np.fft.irfft2(F_est, s=(ny_b, nx_b)) + mean_b
    rmses.append(np.sqrt(np.mean((spat_est - truth)**2)))

rmses = np.array(rmses)
print(f"\n{'='*55}")
print(f"Tikhonov Deconvolution Results ({len(val_idx)} patches)")
print(f"{'='*55}")
print(f"RMSE = {np.mean(rmses):.2f} +- {np.std(rmses):.2f} nT")
print(f"Bicubic baseline = 39.30 nT")
print(f"Improvement = {(39.30 - np.mean(rmses))/39.30*100:.1f}%")
print(f"Best lambda = {best_lam:.4f}")
print(f"Wiener (comparison)  = 34.84 nT (11.4%)")
print(f"{'='*55}")

np.savez(Path('F:/极区地磁导航/南极/experiment/results_gan/tikhonov_results.npz'),
         rmse_mean=np.mean(rmses), rmse_std=np.std(rmses),
         best_lambda=best_lam,
         k_cen=k_cen, H_radial=H_radial)
print("Saved: tikhonov_results.npz")
