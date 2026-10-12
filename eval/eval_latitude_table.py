# -*- coding: utf-8 -*-
"""Latitude table (Table 3): 6 representative 1x1 deg patches, 5 methods.
Methods: Bicubic / Spatial U-Net / Equivalent Source (dipole) / SRCNN / Spectral U-Net.
Uses the same spatial-block protocol, global-normalisation spectral U-Net, and
dipole equivalent source as the rest of the paper. Reports RMSE and MAE (nT).
"""
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F, time
from pathlib import Path
from scipy.interpolate import RegularGridInterpolator
import rasterio

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA = Path("F:/极区地磁导航/南极")
MODEL_DIR = DATA / "paper/4.2_comparison methods/4.2.3_method_comparison"
MODEL_SPECTRAL = DATA / "experiment/ablation_spatial/globalnorm_best.pth"
MODEL_SPATIAL = MODEL_DIR / "SpatialUNet/spatial_unet_spatial_best.pth"
MODEL_SRCNN = MODEL_DIR / "SRCNN/srcnn_spatial_best.pth"
F_SCALE_GLOBAL = 4437.759088306881
EQ_DEPTH, EQ_LAM_REL = 20.0, 1e-3
PAD_H, PAD_W = 64, 32

# the six 1x1 deg patches listed in Table 3 (lat_top, lat_bot, lon_l, lon_r)
PATCHES = [
    ("61-62S, 90-91E",  -61.0, -62.0,  90.0,  91.0),
    ("66-67S, 60-61E",  -66.0, -67.0,  60.0,  61.0),
    ("71-72S, 18-19W",  -71.0, -72.0, -19.0, -18.0),
    ("76-77S, 45-46E",  -76.0, -77.0,  45.0,  46.0),
    ("81-82S, 60-61W",  -81.0, -82.0, -61.0, -60.0),
    ("86-87S, 158-159E",-86.0, -87.0, 158.0, 159.0),
]


class SRCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(1, 64, 9, padding=4), nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 5, padding=2), nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 5, padding=2))
    def forward(self, x): return x + self.body(x)


class SpatialUNet(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, base=48):
        super().__init__()
        self.enc1 = nn.Sequential(nn.Conv2d(in_ch, base, 3, padding=1), nn.BatchNorm2d(base), nn.LeakyReLU(0.2))
        self.enc2 = self._enc(base, base*2); self.enc3 = self._enc(base*2, base*4); self.enc4 = self._enc(base*4, base*8)
        self.bottle = nn.Sequential(nn.Conv2d(base*8, base*8, 3, padding=1), nn.BatchNorm2d(base*8), nn.LeakyReLU(0.2))
        self.dec4 = self._dec(base*8, base*4); self.dec3 = self._dec(base*8, base*2); self.dec2 = self._dec(base*4, base)
        self.dec1 = nn.Sequential(nn.Conv2d(base*2, base, 3, padding=1), nn.BatchNorm2d(base), nn.ReLU(),
                                   nn.Conv2d(base, out_ch, 3, padding=1))
        nn.init.zeros_(self.dec1[-1].weight); nn.init.zeros_(self.dec1[-1].bias)
    def _enc(self, ci, co): return nn.Sequential(nn.Conv2d(ci, co, 4, stride=2, padding=1), nn.BatchNorm2d(co), nn.LeakyReLU(0.2))
    def _dec(self, ci, co): return nn.Sequential(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                                                 nn.Conv2d(ci, co, 3, padding=1), nn.BatchNorm2d(co), nn.ReLU())
    def forward(self, x):
        e1=self.enc1(x); e2=self.enc2(e1); e3=self.enc3(e2); e4=self.enc4(e3); b=self.bottle(e4); d4=self.dec4(b)
        if d4.shape[2:]!=e3.shape[2:]: d4=F.interpolate(d4,size=e3.shape[2:],mode="bilinear",align_corners=False)
        d3=self.dec3(torch.cat([d4,e3],1))
        if d3.shape[2:]!=e2.shape[2:]: d3=F.interpolate(d3,size=e2.shape[2:],mode="bilinear",align_corners=False)
        d2=self.dec2(torch.cat([d3,e2],1))
        if d2.shape[2:]!=e1.shape[2:]: d2=F.interpolate(d2,size=e1.shape[2:],mode="bilinear",align_corners=False)
        return x + self.dec1(torch.cat([d2,e1],1))


class SpectralUNet(nn.Module):
    def __init__(self, in_ch=4, out_ch=2, base=48):
        super().__init__()
        self.enc1 = nn.Sequential(nn.Conv2d(in_ch, base, 3, padding=1), nn.BatchNorm2d(base), nn.LeakyReLU(0.2))
        self.enc2 = self._enc(base, base*2); self.enc3 = self._enc(base*2, base*4); self.enc4 = self._enc(base*4, base*8)
        self.bottle = nn.Sequential(nn.Conv2d(base*8, base*8, 3, padding=1), nn.BatchNorm2d(base*8), nn.LeakyReLU(0.2))
        self.dec4 = self._dec(base*8, base*4); self.dec3 = self._dec(base*8, base*2); self.dec2 = self._dec(base*4, base)
        self.dec1 = nn.Sequential(nn.Conv2d(base*2, base, 3, padding=1), nn.BatchNorm2d(base), nn.ReLU(),
                                   nn.Conv2d(base, out_ch, 3, padding=1))
    def _enc(self, ci, co): return nn.Sequential(nn.Conv2d(ci, co, 4, stride=2, padding=1), nn.BatchNorm2d(co), nn.LeakyReLU(0.2))
    def _dec(self, ci, co): return nn.Sequential(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                                                 nn.Conv2d(ci, co, 3, padding=1), nn.BatchNorm2d(co), nn.ReLU())
    def forward(self, x):
        e1=self.enc1(x); e2=self.enc2(e1); e3=self.enc3(e2); e4=self.enc4(e3); b=self.bottle(e4); d4=self.dec4(b)
        d3=self.dec3(torch.cat([d4,e3],1)); d2=self.dec2(torch.cat([d3,e2],1))
        return self.dec1(torch.cat([d2,e1],1))


def to_km_coords(lat, lon, lat0, lon0):
    dy = (lat - lat0) * 111.32
    dx = (lon - lon0) * 111.32 * np.cos(np.radians(lat0))
    return np.column_stack([dy, dx])


def _dipole_kernel(dx, dy, d):
    R2 = dx**2 + dy**2 + d**2
    return (2.0*d**2 - dx**2 - dy**2) / R2**2.5


def equivalent_source(src_xy_km, src_val, tgt_xy_km, depth=EQ_DEPTH, lam_rel=EQ_LAM_REL):
    mean_s = float(np.mean(src_val))
    d_obs = src_val - mean_s
    n = len(d_obs)
    G = _dipole_kernel(src_xy_km[:,0][:,None]-src_xy_km[:,0][None,:],
                       src_xy_km[:,1][:,None]-src_xy_km[:,1][None,:], depth)
    GTG = G.T @ G
    lam = lam_rel * np.mean(np.diag(GTG))
    m = np.linalg.solve(GTG + lam*np.eye(n), G.T @ d_obs)
    Gp = _dipole_kernel(tgt_xy_km[:,0][:,None]-src_xy_km[:,0][None,:],
                        tgt_xy_km[:,1][:,None]-src_xy_km[:,1][None,:], depth)
    return Gp @ m + mean_s


print("Loading data + models...", flush=True)
with rasterio.open(DATA / "ADMAP_2B_2017_lonlat_geod.tif") as s:
    d2b = s.read(1).astype(np.float64); d2b[d2b == s.nodata] = np.nan; b2b = s.bounds
with rasterio.open(DATA / "ADMAP_2S_lonlat_geod.tif") as s:
    d2s = s.read(1).astype(np.float64); d2s[d2s < -1e9] = np.nan; b2s = s.bounds
H2b, W2b = d2b.shape; H2s, W2s = d2s.shape
dy2b_px = (b2b.top-b2b.bottom)/H2b; dx2b_px = (b2b.right-b2b.left)/W2b
dy2s_px = (b2s.top-b2s.bottom)/H2s; dx2s_px = (b2s.right-b2s.left)/W2s

srcnn = SRCNN().to(DEVICE); srcnn.load_state_dict(torch.load(MODEL_SRCNN, map_location=DEVICE)["model"]); srcnn.eval()
sunet = SpatialUNet(1, 1, base=48).to(DEVICE); sunet.load_state_dict(torch.load(MODEL_SPATIAL, map_location=DEVICE)["model"]); sunet.eval()
spec = SpectralUNet(4, 2, base=48).to(DEVICE); spec.load_state_dict(torch.load(MODEL_SPECTRAL, map_location=DEVICE)["G"]); spec.eval()

METHODS = ["Bicubic", "Spatial U-Net", "Equivalent Source", "SRCNN", "Spectral U-Net"]
print(f"\n{'patch':>18s} {'metric':>6s} " + " ".join(f"{m:>16s}" for m in METHODS))
print("-" * 112)

rows = []
for name, lat_top, lat_bot, lon_l, lon_r in PATCHES:
    mlat = (lat_top + lat_bot)/2; mlat_rad = np.radians(mlat); mlon = (lon_l + lon_r)/2
    i0_s = int(round((b2s.top-lat_top)/dy2s_px)); i1_s = int(round((b2s.top-lat_bot)/dy2s_px))
    i0_b = int(round((b2b.top-lat_top)/dy2b_px)); i1_b = int(round((b2b.top-lat_bot)/dy2b_px))
    j0_s = int(round((lon_l-b2s.left)/dx2s_px)); j1_s = int(round((lon_r-b2s.left)/dx2s_px))
    j0_b = int(round((lon_l-b2b.left)/dx2b_px)); j1_b = int(round((lon_r-b2b.left)/dx2b_px))
    i0_s, i1_s = max(0, i0_s), min(H2s, i1_s); j0_s, j1_s = max(0, j0_s), min(W2s, j1_s)
    i0_b, i1_b = max(0, i0_b), min(H2b, i1_b); j0_b, j1_b = max(0, j0_b), min(W2b, j1_b)
    p2s = d2s[i0_s:i1_s, j0_s:j1_s].copy(); p2b = d2b[i0_b:i1_b, j0_b:j1_b].copy()
    mask_s = np.isnan(p2s); mask_b = np.isnan(p2b)
    if mask_s.any(): p2s[mask_s] = np.nanmean(p2s)
    if mask_b.any(): p2b[mask_b] = np.nanmean(p2b)
    ny_s, nx_s = p2s.shape; ny_b, nx_b = p2b.shape

    dy_s_km = dy2s_px*111.32; dx_s_km = dx2s_px*111.32*np.cos(mlat_rad)
    dy_b_km = dy2b_px*111.32; dx_b_km = dx2b_px*111.32*np.cos(mlat_rad)
    lat_s_arr = np.linspace(lat_top, lat_bot, ny_s); lon_s_arr = np.linspace(lon_l, lon_r, nx_s)
    lat_b_arr = np.linspace(lat_top, lat_bot, ny_b); lon_b_arr = np.linspace(lon_l, lon_r, nx_b)
    LON_b, LAT_b = np.meshgrid(lon_b_arr, lat_b_arr)
    pts = np.stack([LAT_b.ravel(), LON_b.ravel()], axis=-1)
    mean_tgt = float(np.mean(p2b))
    preds = {}

    # Bicubic
    interp_raw = RegularGridInterpolator((lat_s_arr, lon_s_arr), p2s, method="cubic",
                                         bounds_error=False, fill_value=np.nanmean(p2s))
    bicubic = interp_raw(pts).reshape(ny_b, nx_b); mean_bic = float(np.mean(bicubic))
    preds["Bicubic"] = bicubic

    # Equivalent source (dipole)
    LON_s, LAT_s = np.meshgrid(lon_s_arr, lat_s_arr)
    valid_s = ~mask_s
    src_xy = to_km_coords(LAT_s[valid_s], LON_s[valid_s], mlat, mlon)
    tgt_xy = to_km_coords(LAT_b.ravel(), LON_b.ravel(), mlat, mlon)
    preds["Equivalent Source"] = equivalent_source(src_xy, p2s[valid_s], tgt_xy).reshape(ny_b, nx_b)

    # SRCNN & Spatial U-Net
    inp_zm = (bicubic - mean_bic).astype(np.float32)
    x = torch.from_numpy(inp_zm[None, None, :, :]).float().to(DEVICE)
    with torch.no_grad():
        preds["SRCNN"] = srcnn(x).cpu().numpy()[0, 0] + mean_tgt
        preds["Spatial U-Net"] = sunet(x).cpu().numpy()[0, 0] + mean_tgt

    # Spectral U-Net (global norm, no G(k))
    a_in = bicubic - mean_bic; F_in = np.fft.rfft2(a_in)
    ky_b = np.fft.fftfreq(ny_b, d=dy_b_km); kx_b = np.fft.rfftfreq(nx_b, d=dx_b_km)
    KY_b, KX_b = np.meshgrid(ky_b, kx_b, indexing="ij")
    kmax = float(np.max(np.sqrt(KX_b**2 + KY_b**2))) + 1e-8
    H_spec, W_spec = F_in.shape
    h_off = (PAD_H - H_spec)//2; w_off = (PAD_W - W_spec)//2
    inp = np.zeros((4, PAD_H, PAD_W), dtype=np.float32)
    inp[0, h_off:h_off+H_spec, w_off:w_off+W_spec] = F_in.real / F_SCALE_GLOBAL
    inp[1, h_off:h_off+H_spec, w_off:w_off+W_spec] = F_in.imag / F_SCALE_GLOBAL
    inp[2, h_off:h_off+H_spec, w_off:w_off+W_spec] = KX_b / kmax
    inp[3, h_off:h_off+H_spec, w_off:w_off+W_spec] = KY_b / kmax
    with torch.no_grad():
        ps = spec(torch.from_numpy(inp[None]).float().to(DEVICE))[0].cpu().numpy()
    ps = ps[:, h_off:h_off+H_spec, w_off:w_off+W_spec]
    pred_spatial = np.fft.irfft2((ps[0]+1j*ps[1])*F_SCALE_GLOBAL, s=(ny_b, nx_b)) + mean_tgt
    preds["Spectral U-Net"] = pred_spatial

    row = {}
    for m in METHODS:
        err = preds[m] - p2b
        row[m] = (float(np.sqrt(np.mean(err**2))), float(np.mean(np.abs(err))))
    rows.append((name, row))
    print(f"{name:>18s} {'RMSE':>6s} " + " ".join(f"{row[m][0]:>16.2f}" for m in METHODS))
    print(f"{'':>18s} {'MAE':>6s} " + " ".join(f"{row[m][1]:>16.2f}" for m in METHODS))

np.savez(MODEL_DIR / "latitude_table_results.npz",
         patches=[r[0] for r in rows], methods=METHODS,
         rmse=np.array([[r[1][m][0] for m in METHODS] for r in rows]),
         mae=np.array([[r[1][m][1] for m in METHODS] for r in rows]))
print(f"\nSaved: {MODEL_DIR/'latitude_table_results.npz'}")
print("DONE")
