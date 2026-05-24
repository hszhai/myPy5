"""Point cloud + stroke exploration for the Siyun portrait.

Reuses render_siyun.py's exact composition (same camera, crop, bias),
so any output here aligns 1:1 with images/siyun_render.png.

Outputs two images, both saved into images/:

  siyun_points.png   -- Phase 1: every Gaussian centre as a colored dot
                       on a dark canvas. Shows the raw point cloud so
                       you can see what's available for stroke design.

  siyun_strokes.png  -- Phase 2: a starting NPR. Sample a subset of
                       points (importance-weighted by opacity), connect
                       each to its k nearest 2D neighbours with short
                       line segments. Overlapping segments accumulate
                       into scribble-like strokes that follow the form.

Run:  ~/miniconda3/bin/python render_points.py
"""
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

from gsplat import (
    load_3dgs_ply, decode_3dgs, make_camera,
    project_perspective, cull,
)
import render_siyun as cfg

# ---------- knobs for Phase 2 (strokes) -----------------------------------
N_STROKES    = 2500       # number of "seed" points (importance-sampled by opacity)
N_LINKS      = 3          # k nearest neighbours per seed
MAX_LEN_PX   = 35.0       # skip links longer than this (avoids spurious far jumps)
STROKE_ALPHA = 0.35       # per-line opacity (lower = more papery feel)
INK_DARKEN   = 0.50       # multiply each stroke colour by this (lower = inkier)
PAPER_COLOR  = np.array([0.97, 0.96, 0.93])
SEED         = 7


# ========== project all Gaussian centres using cfg ========================
data = load_3dgs_ply(cfg.PLY)
G = decode_3dgs(data)
N = len(data)

ysign = +1.0 if cfg.SCENE_UP_FLIP else -1.0
center = np.median(G["xyz"], axis=0)
center[0] += cfg.HEAD_BIAS_X
center[1] += cfg.HEAD_BIAS_Y
radii = np.linalg.norm(G["xyz"] - center, axis=1)
extent = np.percentile(radii, 90) * 2.0

Rcam = make_camera(cfg.ELEV_DEG, cfg.AZIM_DEG)
cam_xyz = (G["xyz"] - center) @ Rcam.T
cov_cam = np.einsum("ij,njk,lk->nil", Rcam, G["cov3"], Rcam)

W, H = cfg.W, cfg.H
focal = W / (2.0 * np.tan(np.radians(cfg.FOV_DEG) / 2.0))
distance = extent * cfg.DISTANCE_K
mean2d, cov2d, depths, valid_z = project_perspective(
    cam_xyz, cov_cam, focal, distance, W, H, ysign)

keep = cull(mean2d, cov2d, G["opacities"], valid_z, W, H, sub_pixel=0.0)
pts = mean2d[keep]
cols = G["colors"][keep]
ops = G["opacities"][keep]
zs = depths[keep]
print(f"projected {N} centres -> {len(pts)} visible on-screen")


# ========== Phase 1: raw point cloud =======================================
img_pts = np.tile(np.array([0.05, 0.05, 0.07]), (H, W, 1)).astype(np.float64)
# back-to-front so nearer dots paint on top
order = np.argsort(-zs)
for i in order:
    x, y = int(pts[i, 0]), int(pts[i, 1])
    if 0 <= x < W and 0 <= y < H:
        a = ops[i] * 0.55                                # soften
        img_pts[y, x] = img_pts[y, x] * (1.0 - a) + cols[i] * a
plt.imsave("images/siyun_points.png", np.clip(img_pts, 0, 1))
print("Phase 1: saved -> images/siyun_points.png")


# ========== Phase 2: stroke-style NPR (KNN links) =========================
rng = np.random.default_rng(SEED)
prob = ops / ops.sum()                                   # weight by opacity
sub = rng.choice(len(pts), size=min(N_STROKES, len(pts)), p=prob, replace=False)
sub_pts = pts[sub]
sub_cols = np.clip(cols[sub] * INK_DARKEN, 0.0, 1.0)

tree = cKDTree(sub_pts)
_, nn = tree.query(sub_pts, k=N_LINKS + 1)               # +1 because self is included

canvas = np.tile(PAPER_COLOR, (H, W, 1)).astype(np.float64)


def add_line(canvas, x0, y0, x1, y1, color, alpha):
    """Cheap aliased line rasteriser (alpha-composite)."""
    n = int(max(abs(x1 - x0), abs(y1 - y0)) + 1)
    if n <= 1:
        return
    xs = np.linspace(x0, x1, n).astype(int)
    ys = np.linspace(y0, y1, n).astype(int)
    ok = (xs >= 0) & (xs < W) & (ys >= 0) & (ys < H)
    xs, ys = xs[ok], ys[ok]
    canvas[ys, xs] = canvas[ys, xs] * (1.0 - alpha) + color * alpha


t0 = time.time()
n_drawn = 0
for i in range(len(sub_pts)):
    x0, y0 = sub_pts[i]
    c = sub_cols[i]
    for k in range(1, N_LINKS + 1):                      # skip self at index 0
        j = nn[i, k]
        x1, y1 = sub_pts[j]
        if np.hypot(x1 - x0, y1 - y0) > MAX_LEN_PX:
            continue
        add_line(canvas, x0, y0, x1, y1, c, STROKE_ALPHA)
        n_drawn += 1
print(f"Phase 2: drew {n_drawn} stroke segments in {time.time() - t0:.1f}s")

plt.imsave("images/siyun_strokes.png", np.clip(canvas, 0, 1))
print("Phase 2: saved -> images/siyun_strokes.png")
