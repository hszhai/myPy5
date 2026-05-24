"""Render a 3D curve as a thin line using Gaussian splatting (CPU).

Real-time 3D Gaussian Splatting rasterizers (gsplat, diff-gaussian-
rasterization) are CUDA-only, and this is an Intel Mac with no NVIDIA GPU.
So this is a compact *CPU* implementation of the same technique:

  1. the curve becomes a string of anisotropic 3D Gaussians, each stretched
     ALONG the curve and thin ACROSS it  ->  a thin line
  2. each Gaussian's 3x3 covariance is projected to a 2x2 image-space
     covariance (EWA splatting, orthographic camera)
  3. Gaussians are depth-sorted and alpha-composited back-to-front

Run:  ~/miniconda3/bin/python curve_splat.py   (or:  py5 curve_splat.py)
"""
import colorsys
import numpy as np
import matplotlib
matplotlib.use("Agg")  # offscreen, no window
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

# ---- 1. the 3D curve: a trefoil knot -------------------------------------
N = 800
t = np.linspace(0.0, 2.0 * np.pi, N, endpoint=False)
curve = np.column_stack([
    np.sin(t) + 2.0 * np.sin(2.0 * t),
    np.cos(t) - 2.0 * np.cos(2.0 * t),
    -np.sin(3.0 * t),
])

# ---- 2. one anisotropic 3D Gaussian per curve point ----------------------
# tangent of the (closed) curve
tangent = np.roll(curve, -1, axis=0) - np.roll(curve, 1, axis=0)
tangent /= np.linalg.norm(tangent, axis=1, keepdims=True)

# an orthonormal frame (u, v) perpendicular to each tangent
helper = np.tile(np.array([0.0, 0.0, 1.0]), (N, 1))
helper[np.abs(tangent[:, 2]) > 0.9] = np.array([1.0, 0.0, 0.0])
u = np.cross(tangent, helper)
u /= np.linalg.norm(u, axis=1, keepdims=True)
v = np.cross(tangent, u)

seg = np.linalg.norm(np.roll(curve, -1, axis=0) - curve, axis=1).mean()
s_long = seg * 1.6      # stretch along the curve so Gaussians merge into a line
s_thin = 0.022          # thin across the curve  <-- this is the "thin line"
var = np.array([s_long, s_thin, s_thin]) ** 2

# 3D covariance  Sigma = R diag(var) R^T,  R columns = (tangent, u, v)
R = np.stack([tangent, u, v], axis=2)                      # (N,3,3)
cov3 = R @ (np.eye(3) * var) @ np.transpose(R, (0, 2, 1))  # (N,3,3)

# per-Gaussian color (rainbow along the curve) and opacity
colors = np.array([colorsys.hsv_to_rgb(h, 0.85, 1.0)
                   for h in t / (2.0 * np.pi)])
opacity = 0.85

# ---- 3. orthographic camera ----------------------------------------------
W, H = 960, 820
elev, azim = np.radians(26.0), np.radians(40.0)
ce, se = np.cos(elev), np.sin(elev)
ca, sa = np.cos(azim), np.sin(azim)
Rz = np.array([[ca, -sa, 0.0], [sa, ca, 0.0], [0.0, 0.0, 1.0]])
Rx = np.array([[1.0, 0.0, 0.0], [0.0, ce, -se], [0.0, se, ce]])
Rcam = Rx @ Rz                                   # world -> camera

center = curve.mean(axis=0)
cam = (curve - center) @ Rcam.T                  # camera-space points
depth = cam[:, 2]

ppu = min(W, H) / (np.abs(cam[:, :2]).max() * 2.4)   # pixels per world unit
mean2d = np.column_stack([W / 2 + cam[:, 0] * ppu,
                          H / 2 - cam[:, 1] * ppu])

# project the 3D covariance to a 2D image-space covariance (orthographic)
cov_cam = Rcam @ cov3 @ Rcam.T
cov2d = cov_cam[:, :2, :2] * (ppu ** 2)          # in pixel^2

# ---- 4. splat: depth-sort, alpha-composite back-to-front -----------------
bg = np.array([0.04, 0.05, 0.09])
img = np.tile(bg, (H, W, 1)).astype(float)

for i in np.argsort(depth)[::-1]:                # farthest Gaussian first
    cx, cy = mean2d[i]
    C = cov2d[i]
    det = C[0, 0] * C[1, 1] - C[0, 1] * C[1, 0]
    if det <= 1e-9:
        continue
    inv = np.array([[C[1, 1], -C[0, 1]], [-C[1, 0], C[0, 0]]]) / det

    rx = 3.0 * np.sqrt(max(C[0, 0], 1e-6))       # 3-sigma pixel bounding box
    ry = 3.0 * np.sqrt(max(C[1, 1], 1e-6))
    x0, x1 = max(int(cx - rx), 0), min(int(cx + rx) + 1, W)
    y0, y1 = max(int(cy - ry), 0), min(int(cy + ry) + 1, H)
    if x0 >= x1 or y0 >= y1:
        continue

    dx, dy = np.meshgrid(np.arange(x0, x1) - cx, np.arange(y0, y1) - cy)
    q = inv[0, 0] * dx * dx + 2.0 * inv[0, 1] * dx * dy + inv[1, 1] * dy * dy
    a = (opacity * np.exp(-0.5 * q))[..., None]  # per-pixel Gaussian alpha
    img[y0:y1, x0:x1] = img[y0:y1, x0:x1] * (1.0 - a) + colors[i] * a

# ---- 5. bloom / glow, then save ------------------------------------------
glow = gaussian_filter(img, sigma=(6.0, 6.0, 0.0))
img = np.clip(img + 0.5 * glow, 0.0, 1.0)

out = "curve_splat.png"
plt.imsave(out, img)
print(f"splatted {N} Gaussians -> {out}  ({W}x{H})")
