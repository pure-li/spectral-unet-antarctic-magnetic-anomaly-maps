# -*- coding: utf-8 -*-
"""5 组消融实验（空间分块数据划分版，已去除 G(k) 增益校正）
================================================================
审稿意见要求：训练集内部的 train/val 切分不能随机 mix，必须空间分块。
本脚本把 1°×1° patch 按中心经纬度划分：
  - hold-out  : 4 个特殊区域（TAM/AP/EAC/WL），完全不参与训练，也不作为 val
  - val       : 经度 0–45°E 扇区（空间上连续的一块）
  - train     : 其余所有 patch

最终方法（Full）= 无 G(k) 的谱 U-Net：per-patch 幅度归一化 + skip + kx/ky。
G(k) 增益校正经测试对精度无提升（with/without = 18.8 / 18.7 nT），已从方法中
删除，其结果仅保留在给编辑的回复里（full_Gk_results.npz）。

5 组消融（Full 最佳，去掉必要组件后精度应下降）：

  #  变体              幅度归一化   skip   kx/ky
  1  Full              per-patch   ✓      ✓(4ch)
  2  w/o 幅度归一化      none        ✓      ✓(4ch)
  3  w/o skip          per-patch   ✗      ✓(4ch)
  4  w/o kx/ky         per-patch   ✓      ✗(2ch)
  5  global norm       global      ✓      ✓(4ch)

每组独立训练，best 权重 + 训练曲线 + 空间域 RMSE/improvement 存到
ablation_spatial/，5 组按顺序自动跑完（已存在的结果自动跳过）。
"""
import argparse
import gc
import json
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
RNG = np.random.RandomState(42)
BATCH = 16
LR = 2e-4
BASE = 48

ap = argparse.ArgumentParser(description='6-group spatial-block ablation')
ap.add_argument('--full', action='store_true',
                help='full run (default: DEBUG 2-epoch check)')
ap.add_argument('--epochs', type=int, default=100,
                help='epochs per group in full run (default 100)')
ap.add_argument('--gain', action='store_true',
                help='use G(k)-corrected input (spec_gain) and its f_scale')
ap.add_argument('--only', type=str, default=None,
                help='only run this group key (e.g. full, noskip)')
args = ap.parse_args()
DEBUG = not args.full
EPOCHS = 2 if DEBUG else args.epochs

if DEBUG:
    print('=' * 62)
    print('  DEBUG MODE — 2 epochs per group (bug check only)')
    print('  Use --full for the real run (default 100 epochs).')
    print('=' * 62, flush=True)

# ── 空间划分配置 ────────────────────────────────────────────────
# (name, lat_min, lat_max, lon_min, lon_max)，单位度，均为"更南<更北"
HOLDOUT_REGIONS = [
    ("TAM", -82.0, -76.0, 158.0, 172.0),
    ("AP",  -70.0, -63.0, -70.0, -55.0),
    ("EAC", -78.0, -68.0,  65.0,  85.0),
    ("WL",  -73.0, -64.0, 110.0, 135.0),
]
VAL_LON = (0.0, 45.0)   # 验证扇区（东经 0–45°）


def classify_patch(lat_c, lon_c):
    """按 patch 中心坐标划分：holdout > val > train。"""
    for _, lat_min, lat_max, lon_min, lon_max in HOLDOUT_REGIONS:
        if lat_min <= lat_c <= lat_max and lon_min <= lon_c <= lon_max:
            return 'holdout'
    if VAL_LON[0] <= lon_c <= VAL_LON[1]:
        return 'val'
    return 'train'


# ═══════════════════════════════════════════════════════════════
# 1. 加载数据 + G(k)
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
# 2. 提取 patch（存原始谱，不做幅度归一化）
# ═══════════════════════════════════════════════════════════════
spec_gain_raw = []     # F_gain   = rfft2(G(k) 校正后上采样场)
spec_nogain_raw = []   # F_nogain = rfft2(原始双三次上采样场)
spec_tgt_raw = []      # F_tgt    = rfft2(2B 真值)
f_scale_list = []      # 每 patch 的 f_scale = sqrt(mean(|F_in|^2))（输入谱，无真值泄漏）
k_info = []
patch_meta = []        # (mean_s, mean_b, f_scale, rmse_raw, ny_b, nx_b)
split_id = []          # 0=train, 1=val, 2=holdout
all_k_max = []

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

        # 目标谱 + 幅度
        a_b = p2b - np.mean(p2b)
        F_tgt = np.fft.rfft2(a_b)
        f_scale_src = F_gain if args.gain else F_nogain
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
        split_id.append(label2id[classify_patch(lat_c, lon_c)])

n_valid = len(split_id)
print(f'  Valid patches: {n_valid}, skipped: {n_skipped}, '
      f'time: {(time.time() - t0) / 60:.1f} min', flush=True)

# ── 划分索引 ───────────────────────────────────────────────────
train_idx = [k for k, s in enumerate(split_id) if s == 0]
val_idx = [k for k, s in enumerate(split_id) if s == 1]
holdout_idx = [k for k, s in enumerate(split_id) if s == 2]
print(f'  Split: train={len(train_idx)}, val={len(val_idx)}, '
      f'holdout={len(holdout_idx)}', flush=True)

if len(val_idx) == 0:
    raise RuntimeError('验证扇区 0–45°E 内没有有效 patch，检查划分配置！')

f_scale_global = float(np.mean([f_scale_list[k] for k in train_idx]))
baseline_rmse = float(np.mean([patch_meta[k][3] for k in val_idx]))
print(f'  f_scale_global(train) = {f_scale_global:.1f}', flush=True)
print(f'  Raw bicubic baseline (val) = {baseline_rmse:.2f} nT', flush=True)

# 谱网格尺寸（所有 patch 一致：50×26）
_, H_spec, W_spec = spec_gain_raw[0].shape
H_pad, W_pad = 64, 32
h_off = (H_pad - H_spec) // 2
w_off = (W_pad - W_spec) // 2
print(f'  Spec grid {H_spec}x{W_spec} -> pad {H_pad}x{W_pad}', flush=True)


# ═══════════════════════════════════════════════════════════════
# 3. 模型定义
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
# 4. 构建输入 / 目标
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


def eval_val_rmse(G, va_in_t, va_tgt, val_meta, norm_mode):
    """批量前向 + 逐 patch irfft，返回验证集逐 patch 空间 RMSE 列表。"""
    G.eval()
    N = len(val_meta)
    preds = []
    with torch.no_grad():
        for start in range(0, N, BATCH):
            end = min(start + BATCH, N)
            preds.append(G(va_in_t[start:end]).cpu().numpy())
    pred_all = np.concatenate(preds, axis=0)

    rmses = []
    for i in range(N):
        ps = pred_all[i, :, h_off:h_off + H_spec, w_off:w_off + W_spec]
        _, mean_b, f_scale, _rmse_raw, ny_b, nx_b = val_meta[i]
        pred_complex = _undo(ps[0] + 1j * ps[1], f_scale, norm_mode)
        pred_spatial = np.fft.irfft2(pred_complex, s=(ny_b, nx_b)) + mean_b

        ta = va_tgt[i][:, h_off:h_off + H_spec, w_off:w_off + W_spec]
        tgt_complex = _undo(ta[0] + 1j * ta[1], f_scale, norm_mode)
        tgt_spatial = np.fft.irfft2(tgt_complex, s=(ny_b, nx_b)) + mean_b

        rmses.append(float(np.sqrt(np.mean((pred_spatial - tgt_spatial) ** 2))))
    return rmses


# ═══════════════════════════════════════════════════════════════
# 5. 训练函数
# ═══════════════════════════════════════════════════════════════
def train_model(name, NetClass, in_ch, tr_in, va_in, tr_tgt, va_tgt,
                val_meta, norm_mode, save_key):
    print(f'\n{"=" * 60}')
    print(f'Training: {name}')
    print(f'{"=" * 60}', flush=True)

    tr_in_t = torch.from_numpy(tr_in).to(DEVICE)
    va_in_t = torch.from_numpy(va_in).to(DEVICE)
    tr_tgt_t = torch.from_numpy(tr_tgt).to(DEVICE)
    va_tgt_t = torch.from_numpy(va_tgt).to(DEVICE)

    G = NetClass(in_ch=in_ch, out_ch=2, base=BASE).to(DEVICE)
    n_params = sum(p.numel() for p in G.parameters())
    print(f'  Params: {n_params:,}', flush=True)
    opt_G = torch.optim.Adam(G.parameters(), lr=LR, betas=(0.5, 0.999))
    sch_G = torch.optim.lr_scheduler.CosineAnnealingLR(opt_G, T_max=EPOCHS)

    N_tr, N_va = len(tr_in_t), len(va_in_t)
    best_val = float('inf'); best_ep = 0
    train_l1_hist = []; val_l1_hist = []; val_rmse_hist = []; lr_hist = []; epoch_time = []
    t_start = time.time()

    for ep in range(1, EPOCHS + 1):
        G.train()
        epoch_losses = []
        idx = torch.randperm(N_tr)
        for start in range(0, N_tr, BATCH):
            end = min(start + BATCH, N_tr)
            bi = idx[start:end]
            pred = G(tr_in_t[bi])
            loss = nn.functional.l1_loss(pred, tr_tgt_t[bi])
            opt_G.zero_grad(); loss.backward(); opt_G.step()
            epoch_losses.append(loss.item())

        G.eval()
        val_loss = 0.0; val_n = 0
        with torch.no_grad():
            for start in range(0, N_va, BATCH):
                end = min(start + BATCH, N_va)
                bi_v = torch.arange(start, end, device=DEVICE)
                pv = G(va_in_t[bi_v])
                val_loss += nn.functional.l1_loss(pv, va_tgt_t[bi_v]).item() * len(bi_v)
                val_n += len(bi_v)
        val_l1 = val_loss / val_n
        val_rmse = float(np.mean(eval_val_rmse(G, va_in_t, va_tgt, val_meta, norm_mode)))

        ep_time = time.time()
        train_l1_hist.append(float(np.mean(epoch_losses)))
        val_l1_hist.append(float(val_l1))
        val_rmse_hist.append(val_rmse)
        lr_hist.append(float(sch_G.get_last_lr()[0]))
        epoch_time.append(ep_time)

        if val_l1 < best_val:
            best_val = val_l1; best_ep = ep
            torch.save({'G': G.state_dict()}, OUT / f'{save_key}_best.pth')

        if ep % 30 == 0 or ep == 1 or (DEBUG and ep <= 2):
            dt = epoch_time[-1] - (epoch_time[-2] if len(epoch_time) > 1 else t_start)
            print(f'  ep {ep:3d} | train_L1={train_l1_hist[-1]:.5f} | '
                  f'val_L1={val_l1:.5f} | val_RMSE={val_rmse:.2f} | '
                  f'best={best_val:.5f} (ep {best_ep}) | {dt:.1f}s', flush=True)
        sch_G.step()

    total_time = (time.time() - t_start) / 60
    print(f'Best: ep {best_ep}, val_L1={best_val:.5f} | Total: {total_time:.1f}min', flush=True)

    # 空间域 RMSE 评估（best checkpoint）
    print('Evaluating spatial RMSE...', flush=True)
    G.load_state_dict(torch.load(OUT / f'{save_key}_best.pth')['G'])

    N_eval = N_va
    val_rmses_corr = eval_val_rmse(G, va_in_t, va_tgt, val_meta, norm_mode)
    val_rmses_raw = [val_meta[i][3] for i in range(N_eval)]

    rmse_r = float(np.mean(val_rmses_raw))
    rmse_c = float(np.mean(val_rmses_corr))
    impr = (rmse_r - rmse_c) / rmse_r * 100
    print(f'  Raw bicubic: {rmse_r:.1f} nT')
    print(f'  {name}: {rmse_c:.1f} nT  ({impr:.1f}%)')

    np.savez(OUT / f'{save_key}_results.npz',
             raw=val_rmses_raw, corr=val_rmses_corr,
             rmse_raw_mean=rmse_r, rmse_corr_mean=rmse_c,
             train_l1_hist=np.array(train_l1_hist),
             val_l1_hist=np.array(val_l1_hist),
             val_rmse_hist=np.array(val_rmse_hist),
             lr_hist=np.array(lr_hist),
             epoch_time=np.array(epoch_time),
             total_time_min=total_time, best_epoch=best_ep, n_params=n_params)
    return rmse_c, impr, best_ep, n_params, total_time


# ═══════════════════════════════════════════════════════════════
# 6. 5 组实验（顺序自动运行，已有结果自动跳过）
# ═══════════════════════════════════════════════════════════════
GROUPS = [
    dict(key='full',       name='1 Full (norm+skip+kxy)', norm='per_patch', skip=True,  in_ch=4),
    dict(key='nonorm',     name='2 w/o amplitude norm',  norm='none',      skip=True,  in_ch=4),
    dict(key='noskip',     name='3 w/o skip',            norm='per_patch', skip=False, in_ch=4),
    dict(key='nokxy',      name='4 w/o kx/ky',           norm='per_patch', skip=True,  in_ch=2),
    dict(key='globalnorm', name='5 global norm',         norm='global',    skip=True,  in_ch=4),
]

val_meta = [patch_meta[k] for k in val_idx]
results = {}

for g in GROUPS:
    if args.only and g['key'] != args.only:
        continue

    save_key = g['key'] + ('_gk' if args.gain else '')
    res_file = OUT / f"{save_key}_results.npz"
    best_file = OUT / f"{save_key}_best.pth"
    if res_file.exists() and best_file.exists():
        d = np.load(res_file)
        rmse_c = float(d['rmse_corr_mean'])
        rmse_r = float(d['rmse_raw_mean'])
        impr = (rmse_r - rmse_c) / rmse_r * 100
        results[g['key']] = {'name': g['name'], 'rmse': rmse_c, 'impr': impr,
                             'best_ep': int(d['best_epoch']), 'n_params': int(d['n_params']),
                             'time_min': float(d['total_time_min'])}
        print(f"[skip] {g['name']} 已有结果: RMSE={rmse_c:.2f} nT ({impr:.1f}%)", flush=True)
        continue

    src_raw = spec_gain_raw if args.gain else spec_nogain_raw
    NetClass = SpectralUNet if g['skip'] else NoSkipUNet

    tr_in = build_input(src_raw, train_idx, g['norm'], g['in_ch'])
    va_in = build_input(src_raw, val_idx, g['norm'], g['in_ch'])
    tr_tgt = build_2ch(spec_tgt_raw, train_idx, g['norm'])
    va_tgt = build_2ch(spec_tgt_raw, val_idx, g['norm'])

    rmse_c, impr, bep, np_, tmin = train_model(
        g['name'], NetClass, g['in_ch'], tr_in, va_in, tr_tgt, va_tgt,
        val_meta, g['norm'], save_key)

    results[g['key']] = {'name': g['name'], 'rmse': rmse_c, 'impr': impr,
                         'best_ep': bep, 'n_params': np_, 'time_min': tmin}

    # 释放本组的大数组
    del tr_in, va_in, tr_tgt, va_tgt
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# ═══════════════════════════════════════════════════════════════
# 7. 汇总
# ═══════════════════════════════════════════════════════════════
print(f'\n{"=" * 66}')
print('ABLATION SUMMARY (spatial-block split)')
print(f'{"=" * 66}')
print(f'{"Method":<34s} {"RMSE":>8s} {"Impr":>8s} {"Params":>10s} {"BestEp":>7s} {"Time":>7s}')
print('-' * 66)
print(f'{"Raw bicubic (val)":<34s} {baseline_rmse:>8.2f} {"--":>8s} {"--":>10s} {"--":>7s} {"--":>7s}')
for g in GROUPS:
    if g['key'] not in results:
        continue
    r = results[g['key']]
    print(f"{r['name']:<34s} {r['rmse']:>8.2f} {r['impr']:>7.1f}% "
          f"{r['n_params']:>10,} {r['best_ep']:>7d} {r['time_min']:>6.1f}m")
print(f'{"=" * 66}')

summary = {
    'debug': DEBUG,
    'split': {'train': len(train_idx), 'val': len(val_idx), 'holdout': len(holdout_idx)},
    'baseline_rmse': baseline_rmse,
    'f_scale_global': f_scale_global,
    'groups': {k: {kk: (float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else v)
                   for kk, v in r.items()} for k, r in results.items()},
}
with open(OUT / 'ablation_summary.json', 'w') as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)
print(f'Summary saved to {OUT / "ablation_summary.json"}', flush=True)
print('DONE', flush=True)
