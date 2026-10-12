"""
Reconstruct 4 hold-out special regions with 新协议（无 G(k)）。
2×3 layout per region:
  Row 1 = Bicubic / Spectral U-Net / Ground Truth (RdBu_r)
  Row 2 = Error(Bicubic) / Error(Spectral U-Net) / Improvement (RdYlGn)

与 reconstruct_all_methods.py 的差异：
  - 删除 G(k) 增益校正（输入 = 原始 2S 双三次）
  - 模型权重换成空间分块 + 无 G(k) 的新训练结果
  - SRCNN 结构 64→32→1（旧 32→16→1）
  - Spatial U-Net 残差（return x + out）
  - Equivalent Source 偶极子核参数（depth=20km, λ=1e-3）在空间分块训练集上调优
  - DC 恢复：输入 zero-mean 用 mean(bicubic)，输出 +mean(truth)
Output: PNG + PDF，存 experiment/figures/special_regions/。
"""
import numpy as np
import torch, torch.nn as nn, time, warnings
from pathlib import Path
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial.distance import cdist
from scipy.linalg import solve
import rasterio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA = Path('F:/极区地磁导航/南极')
SPAT_DIR = DATA / 'paper/4.2_comparison methods/4.2.3_method_comparison'
MODEL_SPECTRAL = DATA / 'experiment/ablation_spatial/globalnorm_best.pth'
F_SCALE_GLOBAL = 4437.759088306881  # 全局归一化常数（训练集 F_in RMS 均值，无 G(k)）
MODEL_SPATIAL  = SPAT_DIR / 'SpatialUNet/spatial_unet_spatial_best.pth'
MODEL_SRCNN    = SPAT_DIR / 'SRCNN/srcnn_spatial_best.pth'
OUT_DIR = DATA / 'experiment/figures/special_regions'
OUT_DIR.mkdir(exist_ok=True, parents=True)

PATCH_DEG = 1.0; STRIDE_DEG = 0.5; PAD_H, PAD_W = 64, 32
R_EARTH = 6371.0
EQ_DEPTH, EQ_LAM_REL = 20.0, 1e-3   # dipole equivalent-source params (tuned on spatial-block train)

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
        b = self.bottle(e4); d4 = self.dec4(b); d3 = self.dec3(torch.cat([d4, e3], dim=1))
        d2 = self.dec2(torch.cat([d3, e2], dim=1))
        return self.dec1(torch.cat([d2, e1], dim=1))


class SpatialUNet(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, base=48):
        super().__init__()
        self.enc1 = nn.Sequential(nn.Conv2d(in_ch, base, 3, padding=1), nn.BatchNorm2d(base), nn.LeakyReLU(0.2))
        self.enc2 = self._enc(base, base*2); self.enc3 = self._enc(base*2, base*4); self.enc4 = self._enc(base*4, base*8)
        self.bottle = nn.Sequential(nn.Conv2d(base*8, base*8, 3, padding=1), nn.BatchNorm2d(base*8), nn.LeakyReLU(0.2))
        self.dec4 = self._dec(base*8, base*4); self.dec3 = self._dec(base*8, base*2)
        self.dec2 = self._dec(base*4, base)
        self.dec1 = nn.Sequential(nn.Conv2d(base*2, base, 3, padding=1), nn.BatchNorm2d(base), nn.ReLU(), nn.Conv2d(base, out_ch, 3, padding=1))
        nn.init.zeros_(self.dec1[-1].weight); nn.init.zeros_(self.dec1[-1].bias)
    def _enc(self, ci, co): return nn.Sequential(nn.Conv2d(ci, co, 4, stride=2, padding=1), nn.BatchNorm2d(co), nn.LeakyReLU(0.2))
    def _dec(self, ci, co): return nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False), nn.Conv2d(ci, co, 3, padding=1), nn.BatchNorm2d(co), nn.ReLU())
    def forward(self, x):
        e1 = self.enc1(x); e2 = self.enc2(e1); e3 = self.enc3(e2); e4 = self.enc4(e3)
        b = self.bottle(e4); d4 = self.dec4(b)
        d4 = nn.functional.interpolate(d4, size=e3.shape[2:], mode='bilinear', align_corners=False) if d4.shape[2:] != e3.shape[2:] else d4
        d3 = self.dec3(torch.cat([d4, e3], dim=1))
        d3 = nn.functional.interpolate(d3, size=e2.shape[2:], mode='bilinear', align_corners=False) if d3.shape[2:] != e2.shape[2:] else d3
        d2 = self.dec2(torch.cat([d3, e2], dim=1))
        d2 = nn.functional.interpolate(d2, size=e1.shape[2:], mode='bilinear', align_corners=False) if d2.shape[2:] != e1.shape[2:] else d2
        out = self.dec1(torch.cat([d2, e1], dim=1))
        return x + out


class SRCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(1, 64, 9, padding=4), nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 5, padding=2), nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 5, padding=2),
        )
    def forward(self, x): return x + self.body(x)


def to_km_coords(lat, lon, lat0, lon0):
    dy = (lat - lat0) * 111.32
    dx = (lon - lon0) * 111.32 * np.cos(np.radians(lat0))
    return np.column_stack([dy, dx])


def _dipole_kernel(dx, dy, d):
    R2 = dx ** 2 + dy ** 2 + d ** 2
    return (2.0 * d ** 2 - dx ** 2 - dy ** 2) / R2 ** 2.5


def equivalent_source(src_xy_km, src_val, tgt_xy_km, depth=EQ_DEPTH, lam_rel=EQ_LAM_REL):
    mean_s = float(np.mean(src_val))
    d_obs = src_val - mean_s
    n = len(d_obs)
    G = _dipole_kernel(src_xy_km[:, 0][:, None] - src_xy_km[:, 0][None, :],
                       src_xy_km[:, 1][:, None] - src_xy_km[:, 1][None, :], depth)
    GTG = G.T @ G
    lam = lam_rel * np.mean(np.diag(GTG))
    m = np.linalg.solve(GTG + lam * np.eye(n), G.T @ d_obs)
    Gp = _dipole_kernel(tgt_xy_km[:, 0][:, None] - src_xy_km[:, 0][None, :],
                        tgt_xy_km[:, 1][:, None] - src_xy_km[:, 1][None, :], depth)
    return Gp @ m + mean_s


def to_stereo(lat, lon, lon0):
    rho = 2 * R_EARTH * np.tan(np.pi/4 - np.radians(np.abs(lat))/2)
    dlon = np.radians(lon - lon0)
    return rho * np.sin(dlon), rho * np.cos(dlon)


def hann_window_2d(ny, nx):
    wy = 0.5 * (1 - np.cos(2 * np.pi * np.arange(ny) / (ny - 1))) if ny > 1 else np.ones(1)
    wx = 0.5 * (1 - np.cos(2 * np.pi * np.arange(nx) / (nx - 1))) if nx > 1 else np.ones(1)
    return wy[:, None] * wx[None, :]


print("Loading data...", flush=True)
with rasterio.open(DATA / 'ADMAP_2B_2017_lonlat_geod.tif') as s:
    d2b = s.read(1).astype(np.float64); d2b[d2b == s.nodata] = np.nan; b2b = s.bounds
with rasterio.open(DATA / 'ADMAP_2S_lonlat_geod.tif') as s:
    d2s = s.read(1).astype(np.float64); d2s[d2s < -1e9] = np.nan; b2s = s.bounds

H2b, W2b = d2b.shape; H2s, W2s = d2s.shape
dy2b_px = (b2b.top - b2b.bottom) / H2b; dx2b_px = (b2b.right - b2b.left) / W2b
dy2s_px = (b2s.top - b2s.bottom) / H2s; dx2s_px = (b2s.right - b2s.left) / W2s

print("Loading models...", flush=True)
spec_model = SpectralUNet(in_ch=4, out_ch=2, base=48).to(DEVICE)
spec_model.load_state_dict(torch.load(MODEL_SPECTRAL, map_location=DEVICE)['G']); spec_model.eval()

spatial_model = SpatialUNet(1, 1, base=48).to(DEVICE)
spatial_model.load_state_dict(torch.load(MODEL_SPATIAL, map_location=DEVICE)['model']); spatial_model.eval()

srcnn_model = SRCNN().to(DEVICE)
srcnn_model.load_state_dict(torch.load(MODEL_SRCNN, map_location=DEVICE)['model']); srcnn_model.eval()
print("  All models loaded.", flush=True)

all_summary = []

for region_name, lat_min, lat_max, lon_min, lon_max in REGIONS:
    print(f"\n{'='*60}\nRegion: {region_name}\n{'='*60}", flush=True)
    t0 = time.time()
    lon0 = (lon_min + lon_max) / 2

    i0_reg = int((b2b.top - lat_max) / dy2b_px); i1_reg = int((b2b.top - lat_min) / dy2b_px)
    j0_reg = int((lon_min - b2b.left) / dx2b_px); j1_reg = int((lon_max - b2b.left) / dx2b_px)
    i0_reg = max(0, i0_reg); i1_reg = min(H2b, i1_reg)
    j0_reg = max(0, j0_reg); j1_reg = min(W2b, j1_reg)
    H_reg = i1_reg - i0_reg; W_reg = j1_reg - j0_reg
    print(f"  Size: {H_reg}x{W_reg} pixels ({H_reg*dy2b_px:.2f}°x{W_reg*dx2b_px:.2f}°)", flush=True)

    truth_reg = d2b[i0_reg:i1_reg, j0_reg:j1_reg].copy()
    mask_reg = np.isnan(truth_reg)
    truth_filled = truth_reg.copy()
    if mask_reg.any(): truth_filled[mask_reg] = np.nanmean(truth_reg)

    bicubic_accum = np.zeros((H_reg, W_reg), dtype=np.float64)
    spatial_accum = np.zeros((H_reg, W_reg), dtype=np.float64)
    eq_accum = np.zeros((H_reg, W_reg), dtype=np.float64)
    srcnn_accum = np.zeros((H_reg, W_reg), dtype=np.float64)
    spectral_accum = np.zeros((H_reg, W_reg), dtype=np.float64)
    weight_accum = np.zeros((H_reg, W_reg), dtype=np.float64)

    n_patches_lat = int((lat_max - lat_min) / STRIDE_DEG) + 1
    n_patches_lon = int((lon_max - lon_min) / STRIDE_DEG) + 1
    valid_p = 0

    for pi in range(n_patches_lat):
        lat_top = lat_max - pi * STRIDE_DEG; lat_bot = lat_top - PATCH_DEG
        for pj in range(n_patches_lon):
            lon_l = lon_min + pj * STRIDE_DEG; lon_r = lon_l + PATCH_DEG
            mlat = (lat_top + lat_bot) / 2; mlat_rad = np.radians(mlat)
            mlon = (lon_l + lon_r) / 2

            i0_s = int((b2s.top - lat_top) / dy2s_px); i1_s = int((b2s.top - lat_bot) / dy2s_px)
            i0_b = int((b2b.top - lat_top) / dy2b_px); i1_b = int((b2b.top - lat_bot) / dy2b_px)
            j0_s = int((lon_l - b2s.left) / dx2s_px); j1_s = int((lon_r - b2s.left) / dx2s_px)
            j0_b = int((lon_l - b2b.left) / dx2b_px); j1_b = int((lon_r - b2b.left) / dx2b_px)
            i0_s, i1_s = max(0, i0_s), min(H2s, i1_s); j0_s, j1_s = max(0, j0_s), min(W2s, j1_s)
            i0_b, i1_b = max(0, i0_b), min(H2b, i1_b); j0_b, j1_b = max(0, j0_b), min(W2b, j1_b)
            if i1_s <= i0_s or j1_s <= j0_s or i1_b <= i0_b or j1_b <= j0_b: continue

            p2s = d2s[i0_s:i1_s, j0_s:j1_s].copy()
            p2b = d2b[i0_b:i1_b, j0_b:j1_b].copy()
            if np.sum(~np.isnan(p2b)) / p2b.size < 0.8: continue
            if np.all(np.isnan(p2s)): continue
            mask_s = np.isnan(p2s); mask_b = np.isnan(p2b)
            if mask_s.any(): p2s[mask_s] = np.nanmean(p2s)
            if mask_b.any(): p2b[mask_b] = np.nanmean(p2b)
            ny_s, nx_s = p2s.shape; ny_b, nx_b = p2b.shape
            if ny_s < 4 or nx_s < 4: continue

            dy_s_km = dy2s_px * 111.32; dx_s_km = dx2s_px * 111.32 * np.cos(mlat_rad)
            dy_b_km = dy2b_px * 111.32; dx_b_km = dx2b_px * 111.32 * np.cos(mlat_rad)

            lat_s_arr = np.linspace(lat_top, lat_bot, ny_s)
            lon_s_arr = np.linspace(lon_l, lon_r, nx_s)
            lat_b_arr = np.linspace(lat_top, lat_bot, ny_b)
            lon_b_arr = np.linspace(lon_l, lon_r, nx_b)
            LON_b, LAT_b = np.meshgrid(lon_b_arr, lat_b_arr)
            pts = np.stack([LAT_b.ravel(), LON_b.ravel()], axis=-1)
            mean_tgt = float(np.mean(p2b))

            # ── Bicubic ──
            interp_raw = RegularGridInterpolator((lat_s_arr, lon_s_arr), p2s, method='cubic',
                                                  bounds_error=False, fill_value=np.nanmean(p2s))
            bicubic_pred = interp_raw(pts).reshape(ny_b, nx_b)
            mean_bic = float(np.mean(bicubic_pred))

            # ── Spatial U-Net（零均值输入 = bicubic - mean(bicubic)） ──
            inp_spatial = (bicubic_pred - mean_bic).astype(np.float32)
            x_sp = torch.from_numpy(inp_spatial[None, None, :, :]).float().to(DEVICE)
            with torch.no_grad():
                spatial_pred = spatial_model(x_sp).cpu().numpy()[0, 0] + mean_tgt

            # ── Equivalent source (vertical-dipole kernel) ──
            LON_s, LAT_s = np.meshgrid(lon_s_arr, lat_s_arr)
            valid_s = ~mask_s
            src_xy = to_km_coords(LAT_s[valid_s], LON_s[valid_s], mlat, mlon)
            tgt_xy = to_km_coords(LAT_b.ravel(), LON_b.ravel(), mlat, mlon)
            try:
                eq_pred = equivalent_source(src_xy, p2s[valid_s], tgt_xy).reshape(ny_b, nx_b)
            except Exception:
                eq_pred = bicubic_pred.copy()

            # ── SRCNN ──
            inp_srcnn = (bicubic_pred - mean_bic).astype(np.float32)
            x_sr = torch.from_numpy(inp_srcnn[None, None, :, :]).float().to(DEVICE)
            with torch.no_grad():
                srcnn_pred = srcnn_model(x_sr).cpu().numpy()[0, 0] + mean_tgt

            # ── Spectral U-Net（无 G(k)：输入 = rfft2(bicubic - mean(bicubic))） ──
            a_in = bicubic_pred - mean_bic
            F_in = np.fft.rfft2(a_in)
            a_b = p2b - mean_tgt
            F_tgt = np.fft.rfft2(a_b)
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

            i0_p = i0_b - i0_reg; i1_p = i1_b - i0_reg
            j0_p = j0_b - j0_reg; j1_p = j1_b - j0_reg
            si0, si1 = max(0, i0_p), min(H_reg, i1_p)
            sj0, sj1 = max(0, j0_p), min(W_reg, j1_p)
            pi0, pi1 = si0 - i0_p, (si1 - si0) + (si0 - i0_p)
            pj0, pj1 = sj0 - j0_p, (sj1 - sj0) + (sj0 - j0_p)
            if si1 <= si0 or sj1 <= sj0: continue

            win = hann_window_2d(pi1 - pi0, pj1 - pj0)
            bicubic_accum[si0:si1, sj0:sj1] += bicubic_pred[pi0:pi1, pj0:pj1] * win
            spatial_accum[si0:si1, sj0:sj1] += spatial_pred[pi0:pi1, pj0:pj1] * win
            eq_accum[si0:si1, sj0:sj1] += eq_pred[pi0:pi1, pj0:pj1] * win
            srcnn_accum[si0:si1, sj0:sj1] += srcnn_pred[pi0:pi1, pj0:pj1] * win
            spectral_accum[si0:si1, sj0:sj1] += spectral_pred[pi0:pi1, pj0:pj1] * win
            weight_accum[si0:si1, sj0:sj1] += win
            valid_p += 1

    mask_w = weight_accum > 0
    for arr in [bicubic_accum, spatial_accum, eq_accum, srcnn_accum, spectral_accum]:
        arr[mask_w] /= weight_accum[mask_w]
        arr[~mask_w] = np.nan

    valid_mask = ~mask_reg & mask_w
    n_valid = valid_mask.sum()

    methods = {'Bicubic': bicubic_accum, 'Spatial U-Net': spatial_accum,
               'Equivalent Source': eq_accum, 'SRCNN': srcnn_accum,
               'Spectral U-Net': spectral_accum}
    results = {}
    for mname, mdata in methods.items():
        if n_valid > 0:
            err = mdata[valid_mask] - truth_reg[valid_mask]
            rmse = np.sqrt(np.mean(err**2)); mae = np.mean(np.abs(err))
        else:
            rmse = np.nan; mae = np.nan
        results[mname] = (rmse, mae)
        print(f"  {mname:<18s} RMSE={rmse:7.2f} nT, MAE={mae:7.2f} nT", flush=True)

    elapsed = time.time() - t0
    print(f"  Valid patches: {valid_p}, time: {elapsed:.1f}s", flush=True)
    all_summary.append((region_name, results))

    # Save results txt
    txt_path = OUT_DIR / f'{region_name}_all_methods_nogk.txt'
    with open(txt_path, 'w') as f:
        f.write(f"Region: {region_name.replace('_', ' ')}\n")
        f.write(f"Bounds: lat {lat_min} to {lat_max}, lon {lon_min} to {lon_max}\n")
        f.write(f"Valid pixels: {n_valid}\n\n")
        f.write(f"{'Method':<18s} {'RMSE (nT)':>10s} {'MAE (nT)':>10s}\n")
        f.write(f"{'-'*40}\n")
        for mname in ['Bicubic', 'Spatial U-Net', 'Equivalent Source', 'SRCNN', 'Spectral U-Net']:
            rmse, mae = results[mname]
            f.write(f"{mname:<18s} {rmse:>10.2f} {mae:>10.2f}\n")
    print(f"  Saved: {txt_path.name}", flush=True)

    # Figure (2×3 layout)
    print("  Plotting...", flush=True)
    vmax_anom = np.nanpercentile(np.abs(truth_filled[~mask_reg]), 98) if (~mask_reg).sum() > 0 else 1.0

    err_bicubic = bicubic_accum - truth_filled
    err_spectral = spectral_accum - truth_filled
    improvement = np.abs(err_bicubic) - np.abs(err_spectral)
    err_mask = mask_reg | ~mask_w
    err_bicubic[err_mask] = np.nan
    err_spectral[err_mask] = np.nan
    improvement[err_mask] = np.nan
    vmax_err = np.nanpercentile(np.abs(err_spectral), 98)
    vmax_imp = np.nanpercentile(np.abs(improvement), 98)
    if not np.isfinite(vmax_err) or vmax_err <= 0: vmax_err = vmax_anom
    if not np.isfinite(vmax_imp) or vmax_imp <= 0: vmax_imp = vmax_err

    bic_plot = bicubic_accum.copy()
    spec_plot = spectral_accum.copy()
    truth_plot = truth_filled.copy(); truth_plot[mask_reg] = np.nan

    lat_corners = np.linspace(lat_max, lat_min, H_reg + 1)
    lon_corners = np.linspace(lon_min, lon_max, W_reg + 1)
    LON_c, LAT_c = np.meshgrid(lon_corners, lat_corners)
    X_c, Y_c = to_stereo(LAT_c, LON_c, lon0)

    n_pts = 100
    b_lats = np.concatenate([np.full(n_pts, lat_max), np.linspace(lat_max, lat_min, n_pts),
                              np.full(n_pts, lat_min), np.linspace(lat_min, lat_max, n_pts)])
    b_lons = np.concatenate([np.linspace(lon_min, lon_max, n_pts), np.full(n_pts, lon_max),
                              np.linspace(lon_max, lon_min, n_pts), np.full(n_pts, lon_min)])
    bX, bY = to_stereo(b_lats, b_lons, lon0)

    FONT = 12

    def decorate(ax, title):
        ax.plot(bX, bY, 'k-', linewidth=1.0)
        step_lat = 2 if (lat_max - lat_min) > 5 else 1
        step_lon = 5 if (lon_max - lon_min) > 10 else 2
        for glat in np.arange(np.ceil(lat_min/step_lat)*step_lat, lat_max, step_lat):
            if glat <= lat_min or glat >= lat_max: continue
            glon_line = np.linspace(lon_min, lon_max, 200)
            gx, gy = to_stereo(np.full_like(glon_line, glat), glon_line, lon0)
            ax.plot(gx, gy, 'gray', linewidth=0.3, linestyle='--', alpha=0.4)
        for glon in np.arange(np.ceil(lon_min/step_lon)*step_lon, lon_max, step_lon):
            if glon <= lon_min or glon >= lon_max: continue
            glat_line = np.linspace(lat_min, lat_max, 100)
            gx, gy = to_stereo(glat_line, np.full_like(glat_line, glon), lon0)
            ax.plot(gx, gy, 'gray', linewidth=0.3, linestyle='--', alpha=0.4)

        dy = (bY.max() - bY.min()) * 0.015
        dx = (bX.max() - bX.min()) * 0.015
        corners = [
            (lat_max, lon_min, 'left', 'top', -dx, dy),
            (lat_max, lon_max, 'right', 'top', dx, dy),
            (lat_min, lon_min, 'left', 'bottom', -dx, -dy),
            (lat_min, lon_max, 'right', 'bottom', dx, -dy),
        ]
        for clat, clon, ha, va, off_x, off_y in corners:
            cx, cy = to_stereo(clat, clon, lon0)
            lat_str = f'{abs(clat):.0f}°S'
            ew = 'E' if clon >= 0 else 'W'
            lon_str = f'{abs(clon):.0f}°{ew}'
            ha_lat = 'right' if ha == 'left' else 'left'
            ax.text(cx + off_x, cy, lat_str, ha=ha_lat, va='center', fontsize=FONT, color='black')
            va_lon = 'bottom' if va == 'top' else 'top'
            ax.text(cx, cy + off_y, lon_str, ha='center', va=va_lon, fontsize=FONT, color='black')

        ax.set_aspect('equal')
        margin = 0.06 * (bX.max() - bX.min())
        ax.set_xlim(bX.min() - margin, bX.max() + margin)
        ax.set_ylim(bY.min() - margin, bY.max() + margin)
        ax.axis('off')
        ax.set_title(title, fontsize=14, fontweight='bold', y=-0.06, pad=2)

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    panels = [
        (axes[0, 0], bic_plot,  'RdBu_r', -vmax_anom, vmax_anom, '(a) Bicubic'),
        (axes[0, 1], spec_plot, 'RdBu_r', -vmax_anom, vmax_anom, '(b) Spectral U-Net'),
        (axes[0, 2], truth_plot, 'RdBu_r', -vmax_anom, vmax_anom, '(c) Reference'),
        (axes[1, 0], err_bicubic,  'RdBu_r', -vmax_err, vmax_err, '(d) Error (Bicubic)'),
        (axes[1, 1], err_spectral, 'RdBu_r', -vmax_err, vmax_err, '(e) Error (Spectral U-Net)'),
        (axes[1, 2], improvement,  'RdYlGn', -vmax_imp, vmax_imp, '(f) Improvement'),
    ]
    for ax, data, cmap, vmin, vmax, title in panels:
        im = ax.pcolormesh(X_c, Y_c, data, cmap=cmap, vmin=vmin, vmax=vmax,
                           rasterized=True, shading='flat')
        decorate(ax, title)
        cb = plt.colorbar(im, ax=ax, label='nT', shrink=0.82)
        cb.ax.tick_params(labelsize=FONT)

    plt.subplots_adjust(left=0.02, right=0.98, top=0.92, bottom=0.02, wspace=0.05, hspace=0.08)
    out_png = OUT_DIR / f'{region_name}_all_methods_nogk.png'
    out_pdf = OUT_DIR / f'{region_name}_all_methods_nogk.pdf'
    fig.savefig(out_png, dpi=600, bbox_inches='tight')
    fig.savefig(out_pdf, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out_png.name}, {out_pdf.name}", flush=True)

print(f"\n{'='*60}\nAll regions complete (no-G(k)).\nOutput: {OUT_DIR}", flush=True)
print("\nSummary (RMSE nT):", flush=True)
for region_name, results in all_summary:
    line = f"  {region_name:<28s}"
    for mname in ['Bicubic', 'Spatial U-Net', 'Equivalent Source', 'SRCNN', 'Spectral U-Net']:
        line += f" | {mname.split()[0]}: {results[mname][0]:.2f}"
    print(line, flush=True)
