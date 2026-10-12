"""
第二层：val 扇区（0–45°E，训练外）连片"假 gap"块模拟（globalnorm，无 G(k)）
============================================================================
在训练外的 val 扇区里人为划出 5°×5° 连片大块，当作"假 gap"（模拟真实 gap 的
连片、大尺度形态），只用 2S 低分辨率输入重建，对比被藏起来的 2B 真值，
报告块级 RMSE + 块内"边界→中心"距离分层，并给一个代表性块的 4 宫格图。

零泄漏：globalnorm 模型训练集不含 val 扇区，故这些块是模型从未见过的区域。
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
OUT_DIR = DATA / 'paper/4.4_gap_filling/figures'
OUT_DIR.mkdir(exist_ok=True, parents=True)

PATCH_DEG = 1.0
PAD_H, PAD_W = 64, 32
VAL_LON = (0.0, 45.0)   # val 扇区经度范围
BLOCK_N = 5             # 5°×5° 连片块


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


def patch_2b_valid_frac(lat_top, lat_bot, lon_l, lon_r):
    """返回 1° patch 内 2B 有效像素比例（0~1）。"""
    i0 = int((b2b.top - lat_top) / dy2b_px); i1 = int((b2b.top - lat_bot) / dy2b_px)
    j0 = int((lon_l - b2b.left) / dx2b_px); j1 = int((lon_r - b2b.left) / dx2b_px)
    i0, i1 = max(0, i0), min(H2b, i1); j0, j1 = max(0, j0), min(W2b, j1)
    if i1 <= i0 or j1 <= j0:
        return 0.0
    sub = d2b[i0:i1, j0:j1]
    return float(np.sum(~np.isnan(sub)) / sub.size)


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


# ── 1. 在 val 扇区找 5°×5° 全覆盖连片块 ──
print("\nScanning val sector for gap-free 5°×5° blocks...", flush=True)
candidates = []
for i0_deg in range(0, 30 - BLOCK_N + 1):          # lat -60 到 -90，共 30°
    lat_top = -60.0 - i0_deg * 1.0
    for j0_deg in range(0, 45 - BLOCK_N + 1):      # lon 0 到 45，共 45°
        lon_l = 0.0 + j0_deg * 1.0
        fracs = []
        for di in range(BLOCK_N):
            for dj in range(BLOCK_N):
                lt = lat_top - di * 1.0
                lb = lt - 1.0
                ll = lon_l + dj * 1.0
                lr = ll + 1.0
                fracs.append(patch_2b_valid_frac(lt, lb, ll, lr))
        minfrac = min(fracs)
        if minfrac >= 0.95:
            lat_bot = lat_top - BLOCK_N
            lon_r = lon_l + BLOCK_N
            candidates.append((minfrac, lat_top, lat_bot, lon_l, lon_r))

candidates.sort(reverse=True)
print(f"  {len(candidates)} fully-covered candidate blocks found", flush=True)

# 选 4 个不重叠的块（贪心）
selected = []
for cand in candidates:
    minfrac, lat_top, lat_bot, lon_l, lon_r = cand
    overlap = False
    for sel in selected:
        _, st, sb, sl, sr = sel
        if not (lat_bot >= st or lat_top <= sb or lon_r <= sl or lon_l >= sr):
            overlap = True; break
    if not overlap:
        selected.append(cand)
    if len(selected) >= 4:
        break

if len(selected) < 4:
    print(f"  WARNING: only {len(selected)} non-overlapping blocks found", flush=True)

print(f"  Selected {len(selected)} blocks:", flush=True)
for minfrac, lat_top, lat_bot, lon_l, lon_r in selected:
    print(f"    lat [{lat_top:.0f},{lat_bot:.0f}] lon [{lon_l:.0f},{lon_r:.0f}] min2Bcov={minfrac:.3f}", flush=True)

# ── 2. 逐块重建 + 分层 ──
print("\nReconstructing blocks...", flush=True)
block_results = []
for bi, (minfrac, lat_top, lat_bot, lon_l, lon_r) in enumerate(selected):
    depth_map = {}
    block_truth = []; block_spec = []; block_bic = []
    for di in range(BLOCK_N):
        for dj in range(BLOCK_N):
            lt = lat_top - di * 1.0
            lb = lt - 1.0
            ll = lon_l + dj * 1.0
            lr = ll + 1.0
            depth = min(di, BLOCK_N - 1 - di, dj, BLOCK_N - 1 - dj)
            mlat = (lt + lb) / 2; mlat_rad = np.radians(mlat)

            i0_s = int((b2s.top - lt) / dy2s_px); i1_s = int((b2s.top - lb) / dy2s_px)
            i0_b = int((b2b.top - lt) / dy2b_px); i1_b = int((b2b.top - lb) / dy2b_px)
            j0_s = int((ll - b2s.left) / dx2s_px); j1_s = int((lr - b2s.left) / dx2s_px)
            j0_b = int((ll - b2b.left) / dx2b_px); j1_b = int((lr - b2b.left) / dx2b_px)
            i0_s, i1_s = max(0, i0_s), min(H2s, i1_s); j0_s, j1_s = max(0, j0_s), min(W2s, j1_s)
            i0_b, i1_b = max(0, i0_b), min(H2b, i1_b); j0_b, j1_b = max(0, j0_b), min(W2b, j1_b)
            if i1_s <= i0_s or j1_s <= j0_s or i1_b <= i0_b or j1_b <= j0_b:
                continue

            p2s = d2s[i0_s:i1_s, j0_s:j1_s].copy()
            p2b = d2b[i0_b:i1_b, j0_b:j1_b].copy()
            if np.all(np.isnan(p2s)):
                continue
            res = predict_patch(p2s, p2b, lt, lb, ll, lr, mlat_rad)
            if res is None:
                continue
            bicubic_pred, spectral_pred, truth = res
            block_truth.append(truth); block_spec.append(spectral_pred); block_bic.append(bicubic_pred)
            rmse_b = float(np.sqrt(np.mean((bicubic_pred - truth)**2)))
            rmse_s = float(np.sqrt(np.mean((spectral_pred - truth)**2)))
            depth_map.setdefault(depth, {'bic': [], 'spec': []})
            depth_map[depth]['bic'].append(rmse_b)
            depth_map[depth]['spec'].append(rmse_s)

    # 块级 pooled RMSE
    all_t = np.concatenate(block_truth); all_s = np.concatenate(block_spec); all_b = np.concatenate(block_bic)
    pooled_spec = float(np.sqrt(np.mean((all_s - all_t)**2)))
    pooled_bic = float(np.sqrt(np.mean((all_b - all_t)**2)))

    block_results.append({
        'bounds': (lat_top, lat_bot, lon_l, lon_r),
        'pooled_bic': pooled_bic, 'pooled_spec': pooled_spec,
        'depth_map': depth_map,
        'block_truth': block_truth, 'block_spec': block_spec, 'block_bic': block_bic,
    })

    print(f"\n  Block {bi+1}: lat [{lat_top:.0f},{lat_bot:.0f}] lon [{lon_l:.0f},{lon_r:.0f}]", flush=True)
    print(f"    pooled RMSE: Bicubic={pooled_bic:.2f}  Spectral U-Net={pooled_spec:.2f}", flush=True)
    print(f"    {'depth':>5s} {'n':>4s} {'Bic':>8s} {'Spec':>8s}")
    for depth in sorted(depth_map):
        b = depth_map[depth]['bic']; s = depth_map[depth]['spec']
        print(f"    {depth:>5d} {len(b):>4d} {np.mean(b):>8.2f} {np.mean(s):>8.2f}")

# ── 3. 汇总表 ──
print("\n" + "="*70)
print("SUMMARY: synthetic gap blocks (val sector, 5°×5°)")
print("="*70)
print(f"{'Block':<8s}{'lat':>10s}{'lon':>12s}{'Bic RMSE':>10s}{'Spec RMSE':>12s}")
for bi, r in enumerate(block_results):
    lt, lb, ll, lr = r['bounds']
    print(f"{bi+1:<8d}{f'[{lt:.0f},{lb:.0f}]':>10s}{f'[{ll:.0f},{lr:.0f}]':>12s}"
          f"{r['pooled_bic']:>10.2f}{r['pooled_spec']:>12.2f}")

# ── 4. 代表性块 4 宫格图（patch 拼接）──
print("\nPlotting 4-panel figure for block 1...", flush=True)
r0 = block_results[0]
lt, lb, ll, lr = r0['bounds']

# 用滑动窗口（stride 0.5°）+ Hann 融合生成平滑块级图
def build_block_image(lat_top, lat_bot, lon_l, lon_r, mode):
    ny_total = int(round((lat_top - lat_bot) * (1.0 / dy2b_px)))
    nx_total = int(round((lon_r - lon_l) * (1.0 / dx2b_px)))
    acc = np.zeros((ny_total, nx_total), dtype=np.float64)
    wacc = np.zeros((ny_total, nx_total), dtype=np.float64)
    n_pi = int((lat_top - lat_bot) / 0.5) + 1
    n_pj = int((lon_r - lon_l) / 0.5) + 1
    for pi in range(n_pi):
        ptop = lat_top - pi * 0.5; pbot = ptop - 1.0
        for pj in range(n_pj):
            pll = lon_l + pj * 0.5; plr = pll + 1.0
            mlat_rad = np.radians((ptop + pbot) / 2)
            i0_s = int((b2s.top - ptop) / dy2s_px); i1_s = int((b2s.top - pbot) / dy2s_px)
            i0_b = int((b2b.top - ptop) / dy2b_px); i1_b = int((b2b.top - pbot) / dy2b_px)
            j0_s = int((pll - b2s.left) / dx2s_px); j1_s = int((plr - b2s.left) / dx2s_px)
            j0_b = int((pll - b2b.left) / dx2b_px); j1_b = int((plr - b2b.left) / dx2b_px)
            i0_s, i1_s = max(0, i0_s), min(H2s, i1_s); j0_s, j1_s = max(0, j0_s), min(W2s, j1_s)
            i0_b, i1_b = max(0, i0_b), min(H2b, i1_b); j0_b, j1_b = max(0, j0_b), min(W2b, j1_b)
            if i1_s <= i0_s or j1_s <= j0_s or i1_b <= i0_b or j1_b <= j0_b:
                continue
            p2s = d2s[i0_s:i1_s, j0_s:j1_s].copy()
            p2b = d2b[i0_b:i1_b, j0_b:j1_b].copy()
            if np.all(np.isnan(p2s)):
                continue
            res = predict_patch(p2s, p2b, ptop, pbot, pll, plr, mlat_rad)
            if res is None:
                continue
            bic, spec, truth = res
            ny_b, nx_b = truth.shape
            img = {'bic': bic, 'spec': spec, 'truth': truth}[mode]
            # 定位到块坐标
            gi0 = int((lat_top - ptop) / dy2b_px); gj0 = int((pll - lon_l) / dx2b_px)
            gi0 = max(0, gi0); gj0 = max(0, gj0)
            h = min(ny_b, ny_total - gi0); w = min(nx_b, nx_total - gj0)
            if h <= 0 or w <= 0:
                continue
            win = np.hanning(h)[:, None] * np.hanning(w)[None, :]
            acc[gi0:gi0+h, gj0:gj0+w] += img[:h, :w] * win
            wacc[gi0:gi0+h, gj0:gj0+w] += win
    m = wacc > 0
    acc[m] /= wacc[m]
    acc[~m] = np.nan
    return acc

print("  building bicubic / spectral / truth / residual images...", flush=True)
img_bic = build_block_image(lt, lb, ll, lr, 'bic')
img_spec = build_block_image(lt, lb, ll, lr, 'spec')
img_truth = build_block_image(lt, lb, ll, lr, 'truth')
img_resid = img_spec - img_truth

vmax = np.nanpercentile(np.abs(img_truth), 98)
vmax_resid = np.nanpercentile(np.abs(img_resid), 98)

fig, axes = plt.subplots(1, 4, figsize=(20, 5))
panels = [
    (img_bic, 'RdBu_r', '2S bicubic input'),
    (img_truth, 'RdBu_r', '2B reference (hidden)'),
    (img_spec, 'RdBu_r', 'Spectral U-Net fill'),
    (img_resid, 'RdBu_r', 'Residual (U-Net $-$ reference)'),
]
for ax, (data, cmap, title) in zip(axes, panels):
    v = vmax if 'Residual' not in title else vmax_resid
    im = ax.imshow(data, cmap=cmap, vmin=-v, vmax=v, origin='upper')
    ax.set_xlabel(title, fontsize=26)
    ax.set_xticks([]); ax.set_yticks([])
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.ax.tick_params(labelsize=10)
fig.tight_layout()
out_fig = OUT_DIR / 'synthetic_gap_block.png'
fig.savefig(out_fig, dpi=300, bbox_inches='tight', facecolor='white')
fig.savefig(OUT_DIR / 'synthetic_gap_block.pdf', bbox_inches='tight', facecolor='white')
plt.close(fig)
print(f"Saved figure: {out_fig}", flush=True)
print("DONE", flush=True)
