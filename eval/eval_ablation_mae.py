# -*- coding: utf-8 -*-
"""Standalone per-patch RMSE + MAE eval for the 5 spatial-block ablation variants.

Replicates the data extraction / input construction of
train_ablations_spatial.py exactly (gain=False, i.e. the no-G(k) input
spec_nogain_raw), loads the already-trained *_best.pth checkpoints and
reports per-patch RMSE AND MAE (mean over the held-out val sector 0-45E).

Does NOT modify train_ablations_spatial.py.
"""
import time
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from scipy.interpolate import RegularGridInterpolator
import rasterio

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA = Path('F:/极区地磁导航/南极')
GAIN = Path('F:/极区地磁导航/南极/experiment/results_wavenumber/spectral_gain.npz')
OUT = Path('F:/极区地磁导航/南极/experiment/ablation_spatial')
OUT.mkdir(parents=True, exist_ok=True)

PATCH_DEG = 1.0
BATCH = 16
BASE = 48

GAIN_FLAG = False  # run as if args.gain=False

# ── 空间划分配置（verbatim） ────────────────────────────────────
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


# ═══════════════════════════════════════════════════════════════
# 1. 加载数据 + G(k)（G(k) 仅用于复现 patch_meta 的 mean_s）
# ═══════════════════════════════════════════════════════════════
print('=' * 60)
print('Loading data + G(k)...', flush=True)
gain_data = np.load(GAIN)
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

# ═══════════════════════════════════════════════════════════════
# 2. 提取 patch（verbatim from train_ablations_spatial.py）
# ═══════════════════════════════════════════════════════════════
spec_gain_raw = []
spec_nogain_raw = []
spec_tgt_raw = []
f_scale_list = []
k_info = []
patch_meta = []
split_id = []
all_k_max = []
raw_mae_list = []   # 追加：bicubic 逐 patch MAE（不改变 patch_meta 结构）

print(f'Extracting {n_lat}x{n_lon} patches...', flush=True)
t0 = time.time()
n_skipped = 0
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

    dy_s_km = dy2s_px * 111.32; dx_s_km = dx2s_px * 111.32 * np.cos(mlat_rad)
    dy_b_km = dy2b_px * 111.32; dx_b_km = dx2b_px * 111.32 * np.cos(mlat_rad)

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

        # 上采样到 2B 网格
        lat_s_arr = np.linspace(lat_top, lat_bot, ny_s)
        lon_s_arr = np.linspace(lon_l, lon_r, nx_s)
        lat_b_arr = np.linspace(lat_top, lat_bot, ny_b)
        lon_b_arr = np.linspace(lon_l, lon_r, nx_b)
        LON, LAT = np.meshgrid(lon_b_arr, lat_b_arr)
        pts_sp = np.stack([LAT.ravel(), LON.ravel()], axis=-1)

        # 校正后谱（输入用）
        interp_gain = RegularGridInterpolator(
            (lat_s_arr, lon_s_arr), p2s_corrected, method='cubic',
            bounds_error=False, fill_value=np.nanmean(p2s_corrected))
        p2s_gain_on_b = interp_gain(pts_sp).reshape(ny_b, nx_b)
        a_gain = p2s_gain_on_b - np.mean(p2s_gain_on_b)
        F_gain = np.fft.rfft2(a_gain)

        # 原始（无 G(k)）上采样谱（w/o G(k) 组输入用）+ rmse 基线
        interp_raw = RegularGridInterpolator(
            (lat_s_arr, lon_s_arr), p2s, method='cubic',
            bounds_error=False, fill_value=np.nanmean(p2s))
        s_on_b = interp_raw(pts_sp).reshape(ny_b, nx_b)
        F_nogain = np.fft.rfft2(s_on_b - np.mean(s_on_b))
        rmse_raw = float(np.sqrt(np.mean((s_on_b - p2b) ** 2)))
        mae_raw = float(np.mean(np.abs(s_on_b - p2b)))

        # 目标谱 + 幅度
        a_b = p2b - np.mean(p2b)
        F_tgt = np.fft.rfft2(a_b)
        f_scale_src = F_gain if GAIN_FLAG else F_nogain
        f_scale = float(np.sqrt(np.mean(np.abs(f_scale_src) ** 2)))
        if f_scale < 1e-8:
            n_skipped += 1
            continue

        ky_b = np.fft.fftfreq(ny_b, d=dy_b_km)
        kx_b = np.fft.rfftfreq(nx_b, d=dx_b_km)
        KY_b, KX_b = np.meshgrid(ky_b, kx_b, indexing='ij')
        K_b = np.sqrt(KX_b ** 2 + KY_b ** 2)
        all_k_max.append(K_b.max())

        spec_gain_raw.append(np.stack([F_gain.real, F_gain.imag]).astype(np.float32))
        spec_nogain_raw.append(np.stack([F_nogain.real, F_nogain.imag]).astype(np.float32))
        spec_tgt_raw.append(np.stack([F_tgt.real, F_tgt.imag]).astype(np.float32))
        f_scale_list.append(f_scale)
        k_info.append({'kx': KX_b.astype(np.float32),
                       'ky': KY_b.astype(np.float32),
                       'kmag': K_b.astype(np.float32)})
        patch_meta.append((float(np.mean(p2s_gain_on_b)), float(np.mean(p2b)),
                           f_scale, rmse_raw, ny_b, nx_b))
        raw_mae_list.append(mae_raw)
        split_id.append(label2id[classify_patch(lat_c, lon_c)])

n_valid = len(split_id)
print(f'  Valid patches: {n_valid}, skipped: {n_skipped}, '
      f'time: {(time.time() - t0) / 60:.1f} min', flush=True)

train_idx = [k for k, s in enumerate(split_id) if s == 0]
val_idx = [k for k, s in enumerate(split_id) if s == 1]
holdout_idx = [k for k, s in enumerate(split_id) if s == 2]
print(f'  Split: train={len(train_idx)}, val={len(val_idx)}, '
      f'holdout={len(holdout_idx)}', flush=True)

f_scale_global = float(np.mean([f_scale_list[k] for k in train_idx]))
baseline_rmse = float(np.mean([patch_meta[k][3] for k in val_idx]))
print(f'  f_scale_global(train) = {f_scale_global:.1f}', flush=True)
print(f'  Raw bicubic baseline (val) = {baseline_rmse:.2f} nT', flush=True)

_, H_spec, W_spec = spec_gain_raw[0].shape
H_pad, W_pad = 64, 32
h_off = (H_pad - H_spec) // 2
w_off = (W_pad - W_spec) // 2
print(f'  Spec grid {H_spec}x{W_spec} -> pad {H_pad}x{W_pad}', flush=True)


# ═══════════════════════════════════════════════════════════════
# 3. 模型定义（verbatim）
# ═══════════════════════════════════════════════════════════════
class SpectralUNet(nn.Module):
    def __init__(self, in_ch=2, out_ch=2, base=BASE):
        super().__init__()
        self.enc1 = nn.Sequential(nn.Conv2d(in_ch, base, 3, padding=1), nn.BatchNorm2d(base), nn.LeakyReLU(0.2))
        self.enc2 = self._enc(base, base * 2); self.enc3 = self._enc(base * 2, base * 4); self.enc4 = self._enc(base * 4, base * 8)
        self.bottle = nn.Sequential(nn.Conv2d(base * 8, base * 8, 3, padding=1), nn.BatchNorm2d(base * 8), nn.LeakyReLU(0.2))
        self.dec4 = self._dec(base * 8, base * 4); self.dec3 = self._dec(base * 8, base * 2)
        self.dec2 = self._dec(base * 4, base)
        self.dec1 = nn.Sequential(nn.Conv2d(base * 2, base, 3, padding=1), nn.BatchNorm2d(base), nn.ReLU(),
                                   nn.Conv2d(base, out_ch, 3, padding=1))

    def _enc(self, ci, co):
        return nn.Sequential(nn.Conv2d(ci, co, 4, stride=2, padding=1), nn.BatchNorm2d(co), nn.LeakyReLU(0.2))

    def _dec(self, ci, co):
        return nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                             nn.Conv2d(ci, co, 3, padding=1), nn.BatchNorm2d(co), nn.ReLU())

    def forward(self, x):
        e1 = self.enc1(x); e2 = self.enc2(e1); e3 = self.enc3(e2); e4 = self.enc4(e3)
        b = self.bottle(e4); d4 = self.dec4(b)
        d3 = self.dec3(torch.cat([d4, e3], dim=1))
        d2 = self.dec2(torch.cat([d3, e2], dim=1))
        return self.dec1(torch.cat([d2, e1], dim=1))


class NoSkipUNet(nn.Module):
    def __init__(self, in_ch=4, out_ch=2, base=BASE):
        super().__init__()
        self.enc1 = nn.Sequential(nn.Conv2d(in_ch, base, 3, padding=1), nn.BatchNorm2d(base), nn.LeakyReLU(0.2))
        self.enc2 = self._enc(base, base * 2); self.enc3 = self._enc(base * 2, base * 4); self.enc4 = self._enc(base * 4, base * 8)
        self.bottle = nn.Sequential(nn.Conv2d(base * 8, base * 8, 3, padding=1), nn.BatchNorm2d(base * 8), nn.LeakyReLU(0.2))
        self.dec4 = self._dec(base * 8, base * 4); self.dec3 = self._dec(base * 4, base * 2)
        self.dec2 = self._dec(base * 2, base)
        self.dec1 = nn.Sequential(nn.Conv2d(base, base, 3, padding=1), nn.BatchNorm2d(base), nn.ReLU(),
                                   nn.Conv2d(base, out_ch, 3, padding=1))

    def _enc(self, ci, co):
        return nn.Sequential(nn.Conv2d(ci, co, 4, stride=2, padding=1), nn.BatchNorm2d(co), nn.LeakyReLU(0.2))

    def _dec(self, ci, co):
        return nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                             nn.Conv2d(ci, co, 3, padding=1), nn.BatchNorm2d(co), nn.ReLU())

    def forward(self, x):
        e1 = self.enc1(x); e2 = self.enc2(e1); e3 = self.enc3(e2); e4 = self.enc4(e3)
        b = self.bottle(e4); d4 = self.dec4(b); d3 = self.dec3(d4); d2 = self.dec2(d3)
        return self.dec1(d2)


# ═══════════════════════════════════════════════════════════════
# 4. 构建输入 / 目标（verbatim）
# ═══════════════════════════════════════════════════════════════
def _normed(spec, f_scale, norm_mode):
    if norm_mode == 'per_patch':
        return spec / f_scale
    if norm_mode == 'global':
        return spec / f_scale_global
    return spec  # 'none'


def _pad_k(arr):
    out = np.zeros((H_pad, W_pad), dtype=np.float32)
    out[h_off:h_off + H_spec, w_off:w_off + W_spec] = arr
    return out


def build_2ch(spec_raw, idx, norm_mode):
    N = len(idx)
    out = np.zeros((N, 2, H_pad, W_pad), dtype=np.float32)
    for n, k in enumerate(idx):
        a = _normed(spec_raw[k], f_scale_list[k], norm_mode)
        out[n, :, h_off:h_off + H_spec, w_off:w_off + W_spec] = a
    return out


def build_4ch(spec_raw, idx, norm_mode):
    N = len(idx)
    out = np.zeros((N, 4, H_pad, W_pad), dtype=np.float32)
    for n, k in enumerate(idx):
        a = _normed(spec_raw[k], f_scale_list[k], norm_mode)
        out[n, :2, h_off:h_off + H_spec, w_off:w_off + W_spec] = a
        ki = k_info[k]
        kmax = ki['kmag'].max() + 1e-8
        out[n, 2] = _pad_k(ki['kx'] / kmax)
        out[n, 3] = _pad_k(ki['ky'] / kmax)
    return out


def build_input(spec_raw, idx, norm_mode, in_ch):
    return build_4ch(spec_raw, idx, norm_mode) if in_ch == 4 else build_2ch(spec_raw, idx, norm_mode)


def _undo(spec_2ch, f_scale, norm_mode_):
    if norm_mode_ == 'per_patch':
        return spec_2ch * f_scale
    if norm_mode_ == 'global':
        return spec_2ch * f_scale_global
    return spec_2ch


def eval_val_rmse_mae(G, va_in_t, va_tgt, val_meta, norm_mode):
    """批量前向 + 逐 patch irfft，返回验证集逐 patch 空间 RMSE 和 MAE 列表。"""
    G.eval()
    N = len(val_meta)
    preds = []
    with torch.no_grad():
        for start in range(0, N, BATCH):
            end = min(start + BATCH, N)
            preds.append(G(va_in_t[start:end]).cpu().numpy())
    pred_all = np.concatenate(preds, axis=0)

    rmses = []
    maes = []
    for i in range(N):
        ps = pred_all[i, :, h_off:h_off + H_spec, w_off:w_off + W_spec]
        _, mean_b, f_scale, _rmse_raw, ny_b, nx_b = val_meta[i]
        pred_complex = _undo(ps[0] + 1j * ps[1], f_scale, norm_mode)
        pred_spatial = np.fft.irfft2(pred_complex, s=(ny_b, nx_b)) + mean_b

        ta = va_tgt[i][:, h_off:h_off + H_spec, w_off:w_off + W_spec]
        tgt_complex = _undo(ta[0] + 1j * ta[1], f_scale, norm_mode)
        tgt_spatial = np.fft.irfft2(tgt_complex, s=(ny_b, nx_b)) + mean_b

        rmses.append(float(np.sqrt(np.mean((pred_spatial - tgt_spatial) ** 2))))
        maes.append(float(np.mean(np.abs(pred_spatial - tgt_spatial))))
    return rmses, maes


# ═══════════════════════════════════════════════════════════════
# 5. 评估 5 组
# ═══════════════════════════════════════════════════════════════
GROUPS = [
    dict(key='full',       name='1 Full (norm+skip+kxy)', norm='per_patch', skip=True,  in_ch=4),
    dict(key='nonorm',     name='2 w/o amplitude norm',  norm='none',      skip=True,  in_ch=4),
    dict(key='noskip',     name='3 w/o skip',            norm='per_patch', skip=False, in_ch=4),
    dict(key='nokxy',      name='4 w/o kx/ky',           norm='per_patch', skip=True,  in_ch=2),
    dict(key='globalnorm', name='5 global norm',         norm='global',    skip=True,  in_ch=4),
]

src_raw = spec_nogain_raw  # GAIN_FLAG = False

val_meta = [patch_meta[k] for k in val_idx]
val_meta_np = np.array([(m[0], m[1], m[2], m[3], m[4], m[5]) for m in val_meta], dtype=np.float64)

# bicubic baseline (raw) per-patch RMSE + MAE: recompute MAE per patch
# patch_meta stores rmse_raw; MAE raw is not stored, so recompute via tgt/raw.
# Raw bicubic values are exact: pred = s_on_b (bicubic upsampled 2S), tgt = p2b.
# We stored rmse_raw per patch; for MAE of bicubic we recompute from the
# numpy arrays below.

sanity = {}
print('\n' + '=' * 66)
print('EVAL: per-patch RMSE + MAE on val sector 0-45E')
print('=' * 66, flush=True)

save = {}
save['val_idx'] = np.array(val_idx, dtype=np.int64)
save['baseline_rmse'] = np.float64(baseline_rmse)
save['f_scale_global'] = np.float64(f_scale_global)

for g in GROUPS:
    ckpt = OUT / f"{g['key']}_best.pth"
    if not ckpt.exists():
        print(f"  [MISSING] {ckpt}")
        continue
    NetClass = SpectralUNet if g['skip'] else NoSkipUNet
    va_in = build_input(src_raw, val_idx, g['norm'], g['in_ch'])
    va_tgt = build_2ch(spec_tgt_raw, val_idx, g['norm'])
    va_tgt_t = torch.from_numpy(va_tgt).to(DEVICE)
    va_in_t = torch.from_numpy(va_in).to(DEVICE)

    G = NetClass(in_ch=g['in_ch'], out_ch=2, base=BASE).to(DEVICE)
    G.load_state_dict(torch.load(ckpt, map_location=DEVICE)['G'])

    rmses, maes = eval_val_rmse_mae(G, va_in_t, va_tgt, val_meta, g['norm'])
    rmse_c = float(np.mean(rmses))
    mae_c = float(np.mean(maes))
    save[f'rmse_{g["key"]}'] = np.array(rmses, dtype=np.float64)
    save[f'mae_{g["key"]}'] = np.array(maes, dtype=np.float64)
    sanity[g['key']] = (rmse_c, mae_c)
    print(f"  {g['name']:<28s} RMSE={rmse_c:6.2f}  MAE={mae_c:6.2f}  (n={len(rmses)})", flush=True)

    del va_in, va_tgt, va_in_t, va_tgt_t, G
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# ── bicubic baseline MAE（逐 patch：MAE = mean|s_on_b - p2b|） ──────
raw_rmses = [patch_meta[k][3] for k in val_idx]
raw_maes = [raw_mae_list[k] for k in val_idx]
baseline_mae = float(np.mean(raw_maes))
save['rmse_bicubic'] = np.array(raw_rmses, dtype=np.float64)
save['mae_bicubic'] = np.array(raw_maes, dtype=np.float64)
save['baseline_mae'] = np.float64(baseline_mae)

np.savez(OUT / 'ablation_mae_results.npz', **save)
print(f"\nSaved: {OUT / 'ablation_mae_results.npz'}", flush=True)

# ── sanity checks ──────────────────────────────────────────────
print('\n' + '=' * 66)
print('SUMMARY (nT)')
print('=' * 66)
print(f'{"variant":<16s} {"RMSE":>8s} {"MAE":>8s}')
print('-' * 34)
print(f'{"bicubic":<16s} {baseline_rmse:>8.2f} {baseline_mae:>8.2f}')
for g in GROUPS:
    if g['key'] not in sanity:
        continue
    r, m = sanity[g['key']]
    print(f'{g["key"]:<16s} {r:>8.2f} {m:>8.2f}')
print('=' * 66)
print('DONE', flush=True)
