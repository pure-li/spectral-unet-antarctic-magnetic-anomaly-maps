"""
第一层：特殊区域"块边界 → 块中心"分层 RMSE 分析（globalnorm，无 G(k)）
========================================================================
把每个 hold-out 特殊区域按 1°×1° patch 网格划分，对每个 patch 独立预测并算 RMSE，
再按 patch 到区域边界的层数（depth）分组统计：
  depth 0 = 边界 patch（紧邻训练区域），depth 越大越靠块中心（纯外推）。
若 Spectral U-Net 的 RMSE 从边界到中心退化平缓，则证明"连片外推"可靠，
即真实 gap 填补可信度有代理验证支撑。

方法：Bicubic（基线）+ Spectral U-Net（核心）。
输出：分层汇总表 + RMSE-vs-depth 图（4 区域）。
"""
import numpy as np
import torch, torch.nn as nn, time, warnings
from pathlib import Path
from scipy.interpolate import RegularGridInterpolator
import rasterio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA = Path('F:/极区地磁导航/南极')
MODEL_SPECTRAL = DATA / 'experiment/ablation_spatial/globalnorm_best.pth'
F_SCALE_GLOBAL = 4437.759088306881
OUT_DIR = DATA / 'paper/4.3_special_regions/figures'
OUT_DIR.mkdir(exist_ok=True, parents=True)

PATCH_DEG = 1.0
PAD_H, PAD_W = 64, 32

REGIONS = [
    ("Transantarctic_Mountains", -82.0, -76.0, 158.0, 172.0),
    ("Antarctic_Peninsula",      -70.0, -63.0, -70.0, -55.0),
    ("East_Antarctic_Craton",    -78.0, -68.0,  65.0,  85.0),
    ("Wilkes_Land",              -73.0, -64.0, 110.0, 135.0),
]


class SpectralUNet(nn.Module):
    def __init__(self, in_ch=4, out_ch=2, base=48):
        super().__init__()
        self.enc1 = nn.Sequential(nn.Conv2d(in_ch, base, 3, padding=1), nn.BatchNorm2d(base), nn.LeakyReLU(0.2))
        self.enc2 = self._enc(base, base*2); self.enc3 = self._enc(base*2, base*4); self.enc4 = self._enc(base*4, base*8)
        self.bottle = nn.Sequential(nn.Conv2d(base*8, base*8, 3, padding=1), nn.BatchNorm2d(base*8), nn.LeakyReLU(0.2))
        self.dec4 = self._dec(base*8, base*4); self.dec3 = self._dec(base*8, base*2)
        self.dec2 = self._dec(base*4, base)
        self.dec1 = nn.Sequential(nn.Conv2d(base*2, base, 3, padding=1), nn.BatchNorm2d(base), nn.ReLU(), nn.Conv2d(base, out_ch, 3, padding=1))
    def _enc(self, ci, co): return nn.Sequential(nn.Conv2d(ci, co, 4, stride=2, padding=1), nn.BatchNorm2d(co), nn.LeakyReLU(0.2))
    def _dec(self, ci, co): return nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False), nn.Conv2d(ci, co, 3, padding=1), nn.BatchNorm2d(co), nn.ReLU())
    def forward(self, x):
        e1 = self.enc1(x); e2 = self.enc2(e1); e3 = self.enc3(e2); e4 = self.enc4(e3)
        b = self.bottle(e4); d4 = self.dec4(b)
        d3 = self.dec3(torch.cat([d4, e3], dim=1))
        d2 = self.dec2(torch.cat([d3, e2], dim=1))
        return self.dec1(torch.cat([d2, e1], dim=1))


print("Loading data...", flush=True)
with rasterio.open(DATA / 'ADMAP_2B_2017_lonlat_geod.tif') as s:
    d2b = s.read(1).astype(np.float64); d2b[d2b == s.nodata] = np.nan; b2b = s.bounds
with rasterio.open(DATA / 'ADMAP_2S_lonlat_geod.tif') as s:
    d2s = s.read(1).astype(np.float64); d2s[d2s < -1e9] = np.nan; b2s = s.bounds

H2b, W2b = d2b.shape; H2s, W2s = d2s.shape
dy2b_px = (b2b.top - b2b.bottom) / H2b; dx2b_px = (b2b.right - b2b.left) / W2b
dy2s_px = (b2s.top - b2s.bottom) / H2s; dx2s_px = (b2s.right - b2s.left) / W2s

print("Loading model...", flush=True)
spec_model = SpectralUNet(in_ch=4, out_ch=2, base=48).to(DEVICE)
spec_model.load_state_dict(torch.load(MODEL_SPECTRAL, map_location=DEVICE)['G']); spec_model.eval()


def predict_patch(p2s, p2b, lat_top, lat_bot, lon_l, lon_r, mlat_rad):
    """返回 (bicubic_pred, spectral_pred, truth) 或 None。"""
    ny_s, nx_s = p2s.shape; ny_b, nx_b = p2b.shape
    if ny_s < 4 or nx_s < 4:
        return None
    mask_s = np.isnan(p2s); mask_b = np.isnan(p2b)
    if mask_s.any(): p2s = p2s.copy(); p2s[mask_s] = np.nanmean(p2s)
    if mask_b.any(): p2b = p2b.copy(); p2b[mask_b] = np.nanmean(p2b)

    dy_s_km = dy2s_px * 111.32; dx_s_km = dx2s_px * 111.32 * np.cos(mlat_rad)
    dy_b_km = dy2b_px * 111.32; dx_b_km = dx2b_px * 111.32 * np.cos(mlat_rad)

    lat_s_arr = np.linspace(lat_top, lat_bot, ny_s)
    lon_s_arr = np.linspace(lon_l, lon_r, nx_s)
    lat_b_arr = np.linspace(lat_top, lat_bot, ny_b)
    lon_b_arr = np.linspace(lon_l, lon_r, nx_b)
    LON_b, LAT_b = np.meshgrid(lon_b_arr, lat_b_arr)
    pts = np.stack([LAT_b.ravel(), LON_b.ravel()], axis=-1)
    mean_tgt = float(np.mean(p2b))

    interp_raw = RegularGridInterpolator((lat_s_arr, lon_s_arr), p2s, method='cubic',
                                         bounds_error=False, fill_value=np.nanmean(p2s))
    bicubic_pred = interp_raw(pts).reshape(ny_b, nx_b)
    mean_bic = float(np.mean(bicubic_pred))

    a_in = bicubic_pred - mean_bic
    F_in = np.fft.rfft2(a_in)
    f_scale = F_SCALE_GLOBAL

    ky_b = np.fft.fftfreq(ny_b, d=dy_b_km); kx_b = np.fft.rfftfreq(nx_b, d=dx_b_km)
    KY_b, KX_b = np.meshgrid(ky_b, kx_b, indexing='ij')
    kmax = float(np.max(np.sqrt(KX_b**2 + KY_b**2))) + 1e-8
    H_spec, W_spec_fft = F_in.shape
    h_off = (PAD_H - H_spec) // 2; w_off = (PAD_W - W_spec_fft) // 2

    inp = np.zeros((4, PAD_H, PAD_W), dtype=np.float32)
    inp[0, h_off:h_off+H_spec, w_off:w_off+W_spec_fft] = F_in.real / f_scale
    inp[1, h_off:h_off+H_spec, w_off:w_off+W_spec_fft] = F_in.imag / f_scale
    inp[2, h_off:h_off+H_spec, w_off:w_off+W_spec_fft] = KX_b / kmax
    inp[3, h_off:h_off+H_spec, w_off:w_off+W_spec_fft] = KY_b / kmax

    x_spec = torch.from_numpy(inp[None]).float().to(DEVICE)
    with torch.no_grad():
        pred_padded = spec_model(x_spec)[0].cpu().numpy()
    pred_norm = pred_padded[:, h_off:h_off+H_spec, w_off:w_off+W_spec_fft]
    F_pred = (pred_norm[0] + 1j * pred_norm[1]) * f_scale
    spectral_pred = np.fft.irfft2(F_pred, s=(ny_b, nx_b)) + mean_tgt

    return bicubic_pred, spectral_pred, p2b


all_region_rows = []   # (region, depth, bicubic_rmse_list, spectral_rmse_list)
region_curve = {}      # region -> {depth: [bic rmse], [spec rmse]}

for region_name, lat_min, lat_max, lon_min, lon_max in REGIONS:
    print(f"\n{'='*60}\nRegion: {region_name}\n{'='*60}", flush=True)
    n_lat = int(round(lat_max - lat_min))
    n_lon = int(round(lon_max - lon_min))
    print(f"  patch grid: {n_lat} x {n_lon} = {n_lat*n_lon} patches", flush=True)

    depth_map = {}   # depth -> dict(bic=[], spec=[])
    t0 = time.time()
    n_used = 0; n_skip = 0

    for i in range(n_lat):
        lat_top = lat_max - i * PATCH_DEG
        lat_bot = lat_top - PATCH_DEG
        for j in range(n_lon):
            lon_l = lon_min + j * PATCH_DEG
            lon_r = lon_l + PATCH_DEG
            depth = min(i, n_lat - 1 - i, j, n_lon - 1 - j)
            mlat = (lat_top + lat_bot) / 2; mlat_rad = np.radians(mlat)

            i0_s = int((b2s.top - lat_top) / dy2s_px); i1_s = int((b2s.top - lat_bot) / dy2s_px)
            i0_b = int((b2b.top - lat_top) / dy2b_px); i1_b = int((b2b.top - lat_bot) / dy2b_px)
            j0_s = int((lon_l - b2s.left) / dx2s_px); j1_s = int((lon_r - b2s.left) / dx2s_px)
            j0_b = int((lon_l - b2b.left) / dx2b_px); j1_b = int((lon_r - b2b.left) / dx2b_px)
            i0_s, i1_s = max(0, i0_s), min(H2s, i1_s); j0_s, j1_s = max(0, j0_s), min(W2s, j1_s)
            i0_b, i1_b = max(0, i0_b), min(H2b, i1_b); j0_b, j1_b = max(0, j0_b), min(W2b, j1_b)
            if i1_s <= i0_s or j1_s <= j0_s or i1_b <= i0_b or j1_b <= j0_b:
                n_skip += 1; continue

            p2s = d2s[i0_s:i1_s, j0_s:j1_s].copy()
            p2b = d2b[i0_b:i1_b, j0_b:j1_b].copy()
            if np.sum(~np.isnan(p2b)) / p2b.size < 0.8:
                n_skip += 1; continue
            if np.all(np.isnan(p2s)):
                n_skip += 1; continue

            res = predict_patch(p2s, p2b, lat_top, lat_bot, lon_l, lon_r, mlat_rad)
            if res is None:
                n_skip += 1; continue
            bicubic_pred, spectral_pred, truth = res

            err_b = bicubic_pred - truth
            err_s = spectral_pred - truth
            rmse_b = float(np.sqrt(np.mean(err_b**2)))
            rmse_s = float(np.sqrt(np.mean(err_s**2)))

            depth_map.setdefault(depth, {'bic': [], 'spec': []})
            depth_map[depth]['bic'].append(rmse_b)
            depth_map[depth]['spec'].append(rmse_s)
            n_used += 1

    elapsed = time.time() - t0
    print(f"  used={n_used} skip={n_skip} time={elapsed:.1f}s", flush=True)
    region_curve[region_name] = depth_map

    print(f"  {'depth':>6s} {'n':>4s} {'Bic RMSE':>10s} {'SpecUNet RMSE':>13s}")
    for depth in sorted(depth_map):
        bic = depth_map[depth]['bic']; spec = depth_map[depth]['spec']
        print(f"  {depth:>6d} {len(bic):>4d} {np.mean(bic):>10.2f} {np.mean(spec):>13.2f}")

# ── 保存分层数据 ──
np.savez(OUT_DIR / 'boundary_depth_results.npz',
         regions=np.array([r[0] for r in REGIONS]),
         allow_pickle=True)
print(f"\nSaved depth map to {OUT_DIR / 'boundary_depth_results.npz'}", flush=True)


# ── 绘图：RMSE vs depth（4 区域，每个区域 2 条线：Bicubic / Spectral U-Net）──
plt.rcParams.update({'font.family': 'sans-serif', 'font.size': 11})
fig, axes = plt.subplots(1, 4, figsize=(20, 4.6), sharey=True)

short_names = {
    'Transantarctic_Mountains': 'TAM',
    'Antarctic_Peninsula': 'AP',
    'East_Antarctic_Craton': 'EAC',
    'Wilkes_Land': 'WL',
}

for ax, (region_name, _, _, _, _) in zip(axes, REGIONS):
    dmap = region_curve[region_name]
    depths = sorted(dmap)
    bic_mean = [np.mean(dmap[d]['bic']) for d in depths]
    bic_std = [np.std(dmap[d]['bic']) for d in depths]
    spec_mean = [np.mean(dmap[d]['spec']) for d in depths]
    spec_std = [np.std(dmap[d]['spec']) for d in depths]

    ax.errorbar(depths, bic_mean, yerr=bic_std, color='#888888', lw=2.0,
                marker='o', ms=5, capsize=3, label='Bicubic')
    ax.errorbar(depths, spec_mean, yerr=spec_std, color='#D55E00', lw=2.0,
                marker='s', ms=5, capsize=3, label='Spectral U-Net')
    ax.set_title(short_names[region_name], fontsize=26, fontweight='bold')
    ax.set_xlabel('Depth from boundary (patches)', fontsize=11)
    ax.grid(True, alpha=0.25)
    ax.set_xticks(depths)

axes[0].set_ylabel('RMSE (nT)', fontsize=12)
axes[0].legend(loc='upper left', fontsize=10)
fig.tight_layout()
out_fig = OUT_DIR / 'boundary_depth_rmse.png'
fig.savefig(out_fig, dpi=300, bbox_inches='tight', facecolor='white')
fig.savefig(OUT_DIR / 'boundary_depth_rmse.pdf', bbox_inches='tight', facecolor='white')
plt.close(fig)
print(f"Saved figure: {out_fig}", flush=True)
print("DONE", flush=True)
