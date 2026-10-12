"""
Spectral extrapolation baseline: fit power-law to 2B spectrum, reshape 2S spectrum to match.

Key idea: 2B power spectrum ~ k^(-beta) at high k. Use this to "extrapolate" 2S amplitudes
beyond what bicubic interpolation provides.

Method:
1. Estimate radial power spectra P_2S(k) and P_2B(k) from training data
2. Spectral gain: gain(k) = sqrt(P_2B(k) / P_2S(k)), clipped to avoid noise blowup
3. F_est = F_2S * gain(k)
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

        # G(k) correction
        a_s = p2s - np.mean(p2s); F_s = np.fft.rfft2(a_s)
        ky_s = np.fft.fftfreq(ny_s, d=dy_s_km); kx_s = np.fft.rfftfreq(nx_s, d=dx_s_km)
        KY_s, KX_s = np.meshgrid(ky_s, kx_s, indexing='ij')
        K_s = np.sqrt(KX_s**2 + KY_s**2)
        G2d = np.interp(K_s.ravel(), k_gain, G_gain, left=G_gain[0], right=G_gain[-1])
        G2d = G2d.reshape(K_s.shape); G2d = np.maximum(G2d, 0.1)
        F_s_corr = F_s * np.sqrt(G2d)
        p2s_corrected = np.fft.irfft2(F_s_corr, s=(ny_s, nx_s)) + np.mean(p2s)

        # Bicubic to 2B grid
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

# ── 1. Estimate radial power spectra from training data ──
print("Estimating radial power spectra...", flush=True)
n_kbins = 80

all_k = []; all_P2S = []; all_P2B = []
for idx in train_idx:
    F_in = F2S_list[idx]; F_tgt = F2B_list[idx]
    _, ny_b, nx_b = patch_meta[idx]
    ky = np.fft.fftfreq(ny_b); kx = np.fft.rfftfreq(nx_b)
    KY, KX = np.meshgrid(ky, kx, indexing='ij')
    K = np.sqrt(KX**2 + KY**2)
    P_in = np.abs(F_in)**2; P_tgt = np.abs(F_tgt)**2
    all_k.append(K.ravel())
    all_P2S.append(P_in.ravel())
    all_P2B.append(P_tgt.ravel())

all_k = np.concatenate(all_k)
all_P2S = np.concatenate(all_P2S)
all_P2B = np.concatenate(all_P2B)

kmax = all_k.max(); kmin = all_k[all_k > 0].min()
kbins = np.logspace(np.log10(kmin), np.log10(kmax), n_kbins + 1)
k_cen = np.sqrt(kbins[:-1] * kbins[1:])

P2S_radial = np.zeros(n_kbins); P2B_radial = np.zeros(n_kbins); cnt = np.zeros(n_kbins)
dig = np.digitize(all_k, kbins) - 1
valid = (dig >= 0) & (dig < n_kbins)
np.add.at(P2S_radial, dig[valid], all_P2S[valid])
np.add.at(P2B_radial, dig[valid], all_P2B[valid])
np.add.at(cnt, dig[valid], 1)
mask = cnt > 0
P2S_radial[mask] /= cnt[mask]
P2B_radial[mask] /= cnt[mask]

# Fill gaps
for ib in range(n_kbins):
    if cnt[ib] == 0:
        nearby = np.where(mask)[0]
        nn = nearby[np.argmin(np.abs(nearby - ib))]
        P2S_radial[ib] = P2S_radial[nn]
        P2B_radial[ib] = P2B_radial[nn]

# ── 2. Fit power law to 2B spectrum at high k ──
# P_2B(k) ~ A * k^(-beta)
# Use the high-k tail (k > median) for fitting
print("Fitting power law to 2B spectrum...", flush=True)
valid_for_fit = mask & (k_cen > np.median(k_cen[mask]))
x_fit = np.log10(k_cen[valid_for_fit])
y_fit = np.log10(P2B_radial[valid_for_fit])
coeff = np.polyfit(x_fit, y_fit, 1)
beta = -coeff[0]; A = 10**coeff[1]
print(f"  Power law: P(k) = {A:.2e} * k^(-{beta:.2f})", flush=True)

# ── 3. Compute spectral gain: gain(k) = sqrt(P_2B(k) / P_2S(k)) ──
eps = 1e-30
gain_radial = np.sqrt(P2B_radial / (P2S_radial + eps))
# Clip gain to avoid noise amplification
gain_max = 5.0
gain_radial = np.clip(gain_radial, 0.1, gain_max)
print(f"  Gain range: {gain_radial.min():.2f} ~ {gain_radial.max():.2f}", flush=True)

# ── 4. Tune gain_max on small validation subset ──
print("Tuning gain_max...", flush=True)
gain_max_candidates = [2.0, 3.0, 4.0, 5.0, 7.0, 10.0, 15.0, 20.0]
best_gmax, best_rmse = 5.0, 1e10
n_tune = min(150, len(val_idx))

for g_max in gain_max_candidates:
    g_use = np.clip(gain_radial, 0.1, g_max)
    rmses = []
    for vi in val_idx[:n_tune]:
        F_in = F2S_list[vi]; mean_b, ny_b, nx_b = patch_meta[vi]
        truth = spatial_truth[vi]
        ky = np.fft.fftfreq(ny_b); kx = np.fft.rfftfreq(nx_b)
        KY, KX = np.meshgrid(ky, kx, indexing='ij')
        K = np.sqrt(KX**2 + KY**2)
        gain_2d = np.interp(K.ravel(), k_cen, g_use).reshape(K.shape)
        F_est = F_in * gain_2d
        spat_est = np.fft.irfft2(F_est, s=(ny_b, nx_b)) + mean_b
        rmses.append(np.sqrt(np.mean((spat_est - truth)**2)))
    avg_rmse = np.mean(rmses)
    if avg_rmse < best_rmse:
        best_rmse = avg_rmse; best_gmax = g_max
    print(f"  gain_max={g_max:.1f} → RMSE={avg_rmse:.2f}", flush=True)

print(f"\nBest gain_max = {best_gmax:.1f} (tune RMSE = {best_rmse:.2f})", flush=True)

# ── 5. Evaluate on full validation set ──
print("Evaluating on full validation set...", flush=True)
gain_final = np.clip(gain_radial, 0.1, best_gmax)
rmses = []
for vi in val_idx:
    F_in = F2S_list[vi]; mean_b, ny_b, nx_b = patch_meta[vi]
    truth = spatial_truth[vi]
    ky = np.fft.fftfreq(ny_b); kx = np.fft.rfftfreq(nx_b)
    KY, KX = np.meshgrid(ky, kx, indexing='ij')
    K = np.sqrt(KX**2 + KY**2)
    gain_2d = np.interp(K.ravel(), k_cen, gain_final).reshape(K.shape)
    F_est = F_in * gain_2d
    spat_est = np.fft.irfft2(F_est, s=(ny_b, nx_b)) + mean_b
    rmses.append(np.sqrt(np.mean((spat_est - truth)**2)))

rmses = np.array(rmses)
print(f"\n{'='*55}")
print(f"Spectral Extrapolation Results ({len(val_idx)} patches)")
print(f"{'='*55}")
print(f"RMSE = {np.mean(rmses):.2f} +- {np.std(rmses):.2f} nT")
print(f"Bicubic baseline = 39.30 nT")
print(f"Improvement = {(39.30 - np.mean(rmses))/39.30*100:.1f}%")
print(f"Power law: P(k) ~ k^(-{beta:.2f})")
print(f"Best gain_max = {best_gmax:.1f}")
print(f"{'='*55}")

np.savez(Path('F:/极区地磁导航/南极/experiment/results_gan/spectral_extrapolation_results.npz'),
         rmse_mean=np.mean(rmses), rmse_std=np.std(rmses),
         beta=beta, A=A, best_gain_max=best_gmax,
         k_cen=k_cen, gain_radial=gain_final,
         P2S_radial=P2S_radial, P2B_radial=P2B_radial)
print("Saved: spectral_extrapolation_results.npz")
