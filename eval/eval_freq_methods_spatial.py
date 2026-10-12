# -*- coding: utf-8 -*-
"""频域对比方法在空间分块划分下的重建评估（RMSE + MAE）
================================================================
把三种经典频域方法（Wiener / Tikhonov / 谱外推）按新的空间分块划分重跑：
  - train   : 非 hold-out、非验证扇区的所有 patch（估计参数用）
  - val     : 经度 0–45°E 扇区（调参 + 评估）
  - hold-out: 4 个特殊区域，完全不用

与旧代码（eval_wiener/tikhonov/spectral_extrapolation.py）的区别：
  1. 随机 70/30 划分 → 空间分块划分
  2. 只算 RMSE → RMSE + MAE
  3. 双三次基线在 val 上重新计算（旧代码硬编码 39.30 nT）

三种方法都先做 G(k) 预校正再上采样（与原代码一致，USE_GK=True）。
"""
import json
import time
import numpy as np
from pathlib import Path
from scipy.interpolate import RegularGridInterpolator
import rasterio

DATA = Path('F:/极区地磁导航/南极')
OUT = Path('F:/极区地磁导航/南极/experiment/freq_comparison_spatial')
OUT.mkdir(parents=True, exist_ok=True)

PATCH_DEG = 1.0
USE_GK = True   # 频域基线保留 G(k) 预校正（留档 gk 版）

HOLDOUT_REGIONS = [
    ("TAM", -82.0, -76.0, 158.0, 172.0),
    ("AP",  -70.0, -63.0, -70.0, -55.0),
    ("EAC", -78.0, -68.0,  65.0,  85.0),
    ("WL",  -73.0, -64.0, 110.0, 135.0),
]
VAL_LON = (0.0, 45.0)


def classify_patch(lat_c, lon_c):
    for _, lat_min, lat_max, lon_min, lon_max in HOLDOUT_REGIONS:
        if lat_min <= lat_c <= lat_max and lon_min <= lon_c <= lon_max:
            return 'holdout'
    if VAL_LON[0] <= lon_c <= VAL_LON[1]:
        return 'val'
    return 'train'


def radial_average(all_k, all_vals, n_kbins=80):
    """把 (k, value) 散点按对数等分 bin 求径向均值，空 bin 用最近有效值填充。"""
    all_k = np.concatenate(all_k)
    all_vals = np.concatenate(all_vals)
    kmax = all_k.max()
    kmin = all_k[all_k > 0].min()
    kbins = np.logspace(np.log10(kmin), np.log10(kmax), n_kbins + 1)
    k_cen = np.sqrt(kbins[:-1] * kbins[1:])
    radial = np.zeros(n_kbins)
    cnt = np.zeros(n_kbins)
    dig = np.digitize(all_k, kbins) - 1
    valid = (dig >= 0) & (dig < n_kbins)
    np.add.at(radial, dig[valid], all_vals[valid])
    np.add.at(cnt, dig[valid], 1)
    mask = cnt > 0
    radial[mask] /= cnt[mask]
    for ib in range(n_kbins):
        if cnt[ib] == 0:
            nearby = np.where(mask)[0]
            nn = nearby[np.argmin(np.abs(nearby - ib))]
            radial[ib] = radial[nn]
    return k_cen, radial


# ═══════════════════════════════════════════════════════════════
# 1. 加载数据 + 提取 patch（空间分块）
# ═══════════════════════════════════════════════════════════════
print('Loading data + G(k)...', flush=True)
gain_data = np.load(DATA / 'experiment/results_wavenumber/spectral_gain.npz')
k_gain = gain_data['k']
G_gain = gain_data['G_smooth']

with rasterio.open(DATA / 'ADMAP_2B_2017_lonlat_geod.tif') as s:
    d2b = s.read(1).astype(np.float64); d2b[d2b == s.nodata] = np.nan; b2b = s.bounds
with rasterio.open(DATA / 'ADMAP_2S_lonlat_geod.tif') as s:
    d2s = s.read(1).astype(np.float64); d2s[d2s < -1e9] = np.nan; b2s = s.bounds

H2b, W2b = d2b.shape
H2s, W2s = d2s.shape
dy2b_px = (b2b.top - b2b.bottom) / H2b
dx2b_px = (b2b.right - b2b.left) / W2b
dy2s_px = (b2s.top - b2s.bottom) / H2s
dx2s_px = (b2s.right - b2s.left) / W2s

n_lat = int((b2b.top - b2b.bottom) / PATCH_DEG)
n_lon = int((b2b.right - b2b.left) / PATCH_DEG)

print('Extracting patches...', flush=True)
t0 = time.time()

spec_gain_list = []   # rfft2(G(k) 校正后上采样场)  —— 复谱
spec_raw_list = []    # rfft2(原始双三次上采样场)  —— 复谱
spec_tgt_list = []    # rfft2(2B 真值)
spatial_truth = []    # 2B 空间域真值 p2b
patch_meta = []       # (mean_b, ny_b, nx_b)
bic_rmse = []
bic_mae = []
split_id = []
label2id = {'train': 0, 'val': 1, 'holdout': 2}

for i in range(n_lat):
    lat_top = b2b.top - i * PATCH_DEG
    lat_bot = lat_top - PATCH_DEG
    lat_c = (lat_top + lat_bot) / 2.0
    mlat_rad = np.radians(lat_c)
    i0_s = int((b2s.top - lat_top) / dy2s_px); i1_s = int((b2s.top - lat_bot) / dy2s_px)
    i0_b = int((b2b.top - lat_top) / dy2b_px); i1_b = int((b2b.top - lat_bot) / dy2b_px)
    i0_s, i1_s = max(0, i0_s), min(H2s, i1_s)
    i0_b, i1_b = max(0, i0_b), min(H2b, i1_b)
    dy_s_km = dy2s_px * 111.32
    dx_s_km = dx2s_px * 111.32 * np.cos(mlat_rad)

    for j in range(n_lon):
        lon_l = b2b.left + j * PATCH_DEG
        lon_r = lon_l + PATCH_DEG
        lon_c = (lon_l + lon_r) / 2.0
        j0_s = int((lon_l - b2s.left) / dx2s_px); j1_s = int((lon_r - b2s.left) / dx2s_px)
        j0_b = int((lon_l - b2b.left) / dx2b_px); j1_b = int((lon_r - b2b.left) / dx2b_px)
        j0_s, j1_s = max(0, j0_s), min(W2s, j1_s)
        j0_b, j1_b = max(0, j0_b), min(W2b, j1_b)
        if i1_s <= i0_s or j1_s <= j0_s or i1_b <= i0_b or j1_b <= j0_b:
            continue

        p2s = d2s[i0_s:i1_s, j0_s:j1_s].copy()
        p2b = d2b[i0_b:i1_b, j0_b:j1_b].copy()
        if np.sum(~np.isnan(p2b)) / p2b.size < 0.8:
            continue
        mask_s = np.isnan(p2s); mask_b = np.isnan(p2b)
        if mask_s.any(): p2s[mask_s] = np.nanmean(p2s)
        if mask_b.any(): p2b[mask_b] = np.nanmean(p2b)

        ny_s, nx_s = p2s.shape; ny_b, nx_b = p2b.shape
        if ny_s < 4 or nx_s < 4:
            continue

        # G(k) 校正
        a_s = p2s - np.mean(p2s)
        F_s = np.fft.rfft2(a_s)
        ky_s = np.fft.fftfreq(ny_s, d=dy_s_km)
        kx_s = np.fft.rfftfreq(nx_s, d=dx_s_km)
        KY_s, KX_s = np.meshgrid(ky_s, kx_s, indexing='ij')
        K_s = np.sqrt(KX_s ** 2 + KY_s ** 2)
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

        # 原始双三次（基线）
        interp_raw = RegularGridInterpolator(
            (lat_s_arr, lon_s_arr), p2s, method='cubic',
            bounds_error=False, fill_value=np.nanmean(p2s))
        s_on_b = interp_raw(pts).reshape(ny_b, nx_b)

        # G(k) 校正后上采样
        interp_gain = RegularGridInterpolator(
            (lat_s_arr, lon_s_arr), p2s_corrected, method='cubic',
            bounds_error=False, fill_value=np.nanmean(p2s_corrected))
        p2s_gain_on_b = interp_gain(pts).reshape(ny_b, nx_b)

        spec_raw_list.append(np.fft.rfft2(s_on_b - np.mean(s_on_b)))
        spec_gain_list.append(np.fft.rfft2(p2s_gain_on_b - np.mean(p2s_gain_on_b)))
        spec_tgt_list.append(np.fft.rfft2(p2b - np.mean(p2b)))
        spatial_truth.append(p2b.copy())
        patch_meta.append((float(np.mean(p2b)), ny_b, nx_b))
        bic_rmse.append(float(np.sqrt(np.mean((s_on_b - p2b) ** 2))))
        bic_mae.append(float(np.mean(np.abs(s_on_b - p2b))))
        split_id.append(label2id[classify_patch(lat_c, lon_c)])

train_idx = [k for k, s in enumerate(split_id) if s == 0]
val_idx = [k for k, s in enumerate(split_id) if s == 1]
holdout_idx = [k for k, s in enumerate(split_id) if s == 2]
print(f'  Valid: {len(split_id)}, train={len(train_idx)}, val={len(val_idx)}, '
      f'holdout={len(holdout_idx)}, time={(time.time() - t0) / 60:.1f}min', flush=True)

F2S = spec_gain_list if USE_GK else spec_raw_list
F2B = spec_tgt_list

bic_rmse_val = float(np.mean([bic_rmse[k] for k in val_idx]))
bic_mae_val = float(np.mean([bic_mae[k] for k in val_idx]))
print(f'  Bicubic baseline (val): RMSE={bic_rmse_val:.2f}, MAE={bic_mae_val:.2f} nT', flush=True)


def spatial_metrics(F_est, mean_b, truth, ny_b, nx_b):
    spat_est = np.fft.irfft2(F_est, s=(ny_b, nx_b)) + mean_b
    rmse = float(np.sqrt(np.mean((spat_est - truth) ** 2)))
    mae = float(np.mean(np.abs(spat_est - truth)))
    return rmse, mae


# ═══════════════════════════════════════════════════════════════
# 2. 估计 H(k) = |F_2S|/|F_2B|（Wiener / Tikhonov 共用）
# ═══════════════════════════════════════════════════════════════
print('\nEstimating radial |H(k)| from train...', flush=True)
all_k = []
all_H = []
for idx in train_idx:
    F_in = F2S[idx]; F_tgt = F2B[idx]
    _, ny_b, nx_b = patch_meta[idx]
    ky = np.fft.fftfreq(ny_b); kx = np.fft.rfftfreq(nx_b)
    KY, KX = np.meshgrid(ky, kx, indexing='ij')
    K = np.sqrt(KX ** 2 + KY ** 2)
    eps = 1e-8 * max(np.abs(F_tgt).max(), 1.0)
    H = np.abs(F_in) / (np.abs(F_tgt) + eps)
    all_k.append(K.ravel())
    all_H.append(np.clip(H.ravel(), 0, 10))
k_cen_H, H_radial = radial_average(all_k, all_H)
print(f'  H(k) range: {H_radial.min():.3f} ~ {H_radial.max():.3f}', flush=True)


def eval_with_gain(gain_2d, idx_list, n_tune=None):
    """用给定增益场在 idx_list（默认前 n_tune 个 val）上评估平均 RMSE。"""
    use = idx_list if n_tune is None else idx_list[:n_tune]
    rmses = []
    for vi in use:
        F_in = F2S[vi]; mean_b, ny_b, nx_b = patch_meta[vi]
        truth = spatial_truth[vi]
        F_est = F_in * gain_2d(vi, ny_b, nx_b)
        rmse, _ = spatial_metrics(F_est, mean_b, truth, ny_b, nx_b)
        rmses.append(rmse)
    return float(np.mean(rmses))


# ═══════════════════════════════════════════════════════════════
# 3. Wiener：F_est = F_2S * H/(|H|^2 + K_reg)
# ═══════════════════════════════════════════════════════════════
print('\n[Wiener] Tuning K_reg...', flush=True)
K_candidates = np.logspace(-5, 1, 25)
best_K, best_rmse = 0.01, 1e10
n_tune = min(150, len(val_idx))


def gain_wiener(K_reg):
    def _g(vi, ny_b, nx_b):
        ky = np.fft.fftfreq(ny_b); kx = np.fft.rfftfreq(nx_b)
        KY, KX = np.meshgrid(ky, kx, indexing='ij')
        K = np.sqrt(KX ** 2 + KY ** 2)
        H_2d = np.interp(K.ravel(), k_cen_H, H_radial).reshape(K.shape)
        return H_2d / (H_2d ** 2 + K_reg)
    return _g


for K_reg in K_candidates:
    avg_rmse = eval_with_gain(gain_wiener(K_reg), val_idx, n_tune)
    if avg_rmse < best_rmse:
        best_rmse = avg_rmse; best_K = K_reg
    print(f'  K={K_reg:.6f} -> RMSE={avg_rmse:.2f}', flush=True)

print(f'  Best K_reg={best_K:.6f}', flush=True)
wiener_rmses = []; wiener_maes = []
for vi in val_idx:
    mean_b, ny_b, nx_b = patch_meta[vi]
    F_est = F2S[vi] * gain_wiener(best_K)(vi, ny_b, nx_b)
    r, m = spatial_metrics(F_est, mean_b, spatial_truth[vi], ny_b, nx_b)
    wiener_rmses.append(r); wiener_maes.append(m)

# ═══════════════════════════════════════════════════════════════
# 4. Tikhonov：F_est = F_2S * H/(|H|^2 + lambda)
# ═══════════════════════════════════════════════════════════════
print('\n[Tikhonov] Tuning lambda...', flush=True)
lam_candidates = np.logspace(-4, 1, 20)
best_lam, best_rmse_t = 0.1, 1e10


def gain_tikhonov(lam):
    def _g(vi, ny_b, nx_b):
        ky = np.fft.fftfreq(ny_b); kx = np.fft.rfftfreq(nx_b)
        KY, KX = np.meshgrid(ky, kx, indexing='ij')
        K = np.sqrt(KX ** 2 + KY ** 2)
        H_2d = np.interp(K.ravel(), k_cen_H, H_radial).reshape(K.shape)
        return H_2d / (H_2d ** 2 + lam)
    return _g


for lam in lam_candidates:
    avg_rmse = eval_with_gain(gain_tikhonov(lam), val_idx, n_tune)
    if avg_rmse < best_rmse_t:
        best_rmse_t = avg_rmse; best_lam = lam
    print(f'  lambda={lam:.4f} -> RMSE={avg_rmse:.2f}', flush=True)

print(f'  Best lambda={best_lam:.4f}', flush=True)
tik_rmses = []; tik_maes = []
for vi in val_idx:
    mean_b, ny_b, nx_b = patch_meta[vi]
    F_est = F2S[vi] * gain_tikhonov(best_lam)(vi, ny_b, nx_b)
    r, m = spatial_metrics(F_est, mean_b, spatial_truth[vi], ny_b, nx_b)
    tik_rmses.append(r); tik_maes.append(m)

# ═══════════════════════════════════════════════════════════════
# 5. 谱外推：F_est = F_2S * sqrt(P_2B/P_2S)，clip gain
# ═══════════════════════════════════════════════════════════════
print('\n[Spectral Extrapolation] Estimating power spectra...', flush=True)
all_k_p = []; all_P2S = []; all_P2B = []
for idx in train_idx:
    F_in = F2S[idx]; F_tgt = F2B[idx]
    _, ny_b, nx_b = patch_meta[idx]
    ky = np.fft.fftfreq(ny_b); kx = np.fft.rfftfreq(nx_b)
    KY, KX = np.meshgrid(ky, kx, indexing='ij')
    K = np.sqrt(KX ** 2 + KY ** 2)
    all_k_p.append(K.ravel())
    all_P2S.append(np.abs(F_in).ravel() ** 2)
    all_P2B.append(np.abs(F_tgt).ravel() ** 2)

k_cen_P, P2S_radial = radial_average(all_k_p, all_P2S)
_, P2B_radial = radial_average(all_k_p, all_P2B)

gain_radial = np.sqrt(P2B_radial / (P2S_radial + 1e-30))
gain_radial = np.clip(gain_radial, 0.1, 5.0)

print('  Tuning gain_max...', flush=True)
gain_max_candidates = [2.0, 3.0, 4.0, 5.0, 7.0, 10.0, 15.0, 20.0]
best_gmax, best_rmse_e = 5.0, 1e10


def gain_extrap(g_max):
    g_use = np.clip(gain_radial, 0.1, g_max)
    def _g(vi, ny_b, nx_b):
        ky = np.fft.fftfreq(ny_b); kx = np.fft.rfftfreq(nx_b)
        KY, KX = np.meshgrid(ky, kx, indexing='ij')
        K = np.sqrt(KX ** 2 + KY ** 2)
        return np.interp(K.ravel(), k_cen_P, g_use).reshape(K.shape)
    return _g


for g_max in gain_max_candidates:
    avg_rmse = eval_with_gain(gain_extrap(g_max), val_idx, n_tune)
    if avg_rmse < best_rmse_e:
        best_rmse_e = avg_rmse; best_gmax = g_max
    print(f'  gain_max={g_max:.1f} -> RMSE={avg_rmse:.2f}', flush=True)

print(f'  Best gain_max={best_gmax:.1f}', flush=True)
ext_rmses = []; ext_maes = []
for vi in val_idx:
    mean_b, ny_b, nx_b = patch_meta[vi]
    F_est = F2S[vi] * gain_extrap(best_gmax)(vi, ny_b, nx_b)
    r, m = spatial_metrics(F_est, mean_b, spatial_truth[vi], ny_b, nx_b)
    ext_rmses.append(r); ext_maes.append(m)

# ═══════════════════════════════════════════════════════════════
# 6. 汇总
# ═══════════════════════════════════════════════════════════════
results = {
    'Bicubic': {'rmse': bic_rmse_val, 'mae': bic_mae_val},
    'Wiener': {'rmse': float(np.mean(wiener_rmses)), 'mae': float(np.mean(wiener_maes)),
               'rmse_std': float(np.std(wiener_rmses)), 'best_K': float(best_K)},
    'Tikhonov': {'rmse': float(np.mean(tik_rmses)), 'mae': float(np.mean(tik_maes)),
                 'rmse_std': float(np.std(tik_rmses)), 'best_lambda': float(best_lam)},
    'Spectral Extrapolation': {'rmse': float(np.mean(ext_rmses)), 'mae': float(np.mean(ext_maes)),
                               'rmse_std': float(np.std(ext_rmses)), 'best_gain_max': float(best_gmax)},
}

print(f'\n{"=" * 60}')
print(f'Frequency Methods (spatial split, val 0-45E, {len(val_idx)} patches, USE_GK={USE_GK})')
print(f'{"=" * 60}')
print(f'{"Method":<24s} {"RMSE":>9s} {"MAE":>9s} {"Impr":>8s}')
print('-' * 54)
for name in ['Bicubic', 'Wiener', 'Tikhonov', 'Spectral Extrapolation']:
    r = results[name]['rmse']; m = results[name]['mae']
    impr = (bic_rmse_val - r) / bic_rmse_val * 100
    print(f'{name:<24s} {r:>9.2f} {m:>9.2f} {impr:>7.1f}%')
print(f'{"=" * 60}')
print(f'(谱 U-Net Full 参考: 18.66 nT / 40.5%)', flush=True)

results['n_val'] = len(val_idx)
results['USE_GK'] = USE_GK
results['baseline_rmse'] = bic_rmse_val
results['baseline_mae'] = bic_mae_val

tag = 'gk' if USE_GK else 'nogk'
np.savez(OUT / f'freq_methods_results_{tag}.npz',
         wiener_rmse=np.array(wiener_rmses), wiener_mae=np.array(wiener_maes),
         tikhonov_rmse=np.array(tik_rmses), tikhonov_mae=np.array(tik_maes),
         extrap_rmse=np.array(ext_rmses), extrap_mae=np.array(ext_maes),
         bic_rmse=np.array([bic_rmse[k] for k in val_idx]),
         bic_mae=np.array([bic_mae[k] for k in val_idx]))
with open(OUT / f'freq_methods_summary_{tag}.json', 'w') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f'Saved to {OUT}', flush=True)
print('DONE', flush=True)
