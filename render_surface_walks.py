"""Surface-walk stroke renderer.

For each of N_WALKERS iterations:
  1. pick a starting splat (weighted by `PLACEMENT`)
  2. walk STEPS steps "along the surface":
       at each step, look at neighbours within STEP_RADIUS_PX in image
       space and pick one. With FORWARD_BIAS > 0, candidates whose
       direction from the current point matches the previous direction
       are preferred (the walker carries momentum).
  3. stroke the polyline path

The walker carries STATE (its current direction) across steps -- that's
the key difference from KNN-link (static graph) or patch sampling
(random within a disk). The result reads as flowing curves that trace
the form.

FORWARD_BIAS:
   0   -> random walk
   2-3 -> gently curving paths
   5+  -> nearly straight runs that turn only when forced
   high values + few candidates --> the walker carves long sweeping arcs.

Outputs:
  images/<scene>_walks_<variant>.png
  images/<scene>_walks_grid.png

Run:  ~/miniconda3/bin/python render_surface_walks.py
"""
import os
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from PIL import Image

from gsplat import load_3dgs_ply, decode_3dgs, make_camera, project_perspective, cull
from npr_utils import (
    stamp, load_ref_image, compute_saliency,
    add_line, add_splat_stroke, walk_step, PAPER_COLOR,
)
from scene_io import load_scene

SCENE_NAME = os.environ.get("SCENE_NAME", "redhead")
cfg = load_scene(SCENE_NAME)


# ============ variants ====================================================
# STROKE_MODE: how each walk segment is drawn between two splats
#   "line"   -- a line of `STROKE_WIDTH` pixels (1.0 = the old 1-px raster;
#               >1 uses a soft-edged thick-line rasteriser).
#   "splat"  -- stamp N_STAMPS Gaussian splats along the segment, using the
#               AVERAGE of the two endpoint splats' projected covariances
#               scaled by SPLAT_SCALE. Strokes inherit the splat's anisotropy
#               and size.
VARIANTS = [
    dict(name="short",      N_WALKERS=800, STEPS=8,  STEP_RADIUS_PX=12, FORWARD_BIAS=2.0, STROKE_ALPHA=0.45, INK_DARKEN=0.35, PLACEMENT="saliency", STROKE_MODE="line",  STROKE_WIDTH=1.0),
    dict(name="medium",     N_WALKERS=400, STEPS=20, STEP_RADIUS_PX=15, FORWARD_BIAS=3.0, STROKE_ALPHA=0.40, INK_DARKEN=0.30, PLACEMENT="saliency", STROKE_MODE="line",  STROKE_WIDTH=1.0),
    dict(name="long_flow",  N_WALKERS=150, STEPS=50, STEP_RADIUS_PX=20, FORWARD_BIAS=4.0, STROKE_ALPHA=0.40, INK_DARKEN=0.30, PLACEMENT="saliency", STROKE_MODE="line",  STROKE_WIDTH=1.0),
    dict(name="meander",    N_WALKERS=600, STEPS=15, STEP_RADIUS_PX=18, FORWARD_BIAS=0.5, STROKE_ALPHA=0.35, INK_DARKEN=0.30, PLACEMENT="saliency", STROKE_MODE="line",  STROKE_WIDTH=1.0),
    dict(name="straight",   N_WALKERS=300, STEPS=30, STEP_RADIUS_PX=18, FORWARD_BIAS=8.0, STROKE_ALPHA=0.55, INK_DARKEN=0.10, PLACEMENT="saliency", STROKE_MODE="line",  STROKE_WIDTH=1.0),
    dict(name="bold_long",  N_WALKERS=200, STEPS=40, STEP_RADIUS_PX=22, FORWARD_BIAS=5.0, STROKE_ALPHA=0.70, INK_DARKEN=0.00, PLACEMENT="saliency", STROKE_MODE="line",  STROKE_WIDTH=1.0),
    # New: thick lines + splat-stamp strokes
    dict(name="bold_thick", N_WALKERS=200, STEPS=40, STEP_RADIUS_PX=22, FORWARD_BIAS=5.0, STROKE_ALPHA=0.70, INK_DARKEN=0.00, PLACEMENT="saliency", STROKE_MODE="line",  STROKE_WIDTH=2.5),
    dict(name="flow_thick", N_WALKERS=150, STEPS=50, STEP_RADIUS_PX=20, FORWARD_BIAS=4.0, STROKE_ALPHA=0.45, INK_DARKEN=0.20, PLACEMENT="saliency", STROKE_MODE="line",  STROKE_WIDTH=2.0),
    dict(name="bold_splat", N_WALKERS=200, STEPS=40, STEP_RADIUS_PX=22, FORWARD_BIAS=5.0, STROKE_ALPHA=0.60, INK_DARKEN=0.05, PLACEMENT="saliency", STROKE_MODE="splat", SPLAT_SCALE=0.35, SPLAT_ALPHA_SCALE=0.35, SPLAT_MIN_SIGMA=0.10, SPLAT_MAX_SIGMA=1.20, N_STAMPS=5),
]
SEED = 17
SCENE_BASE = os.path.splitext(os.path.basename(cfg.PLY))[0].lower()
REF_IMAGE = cfg.OUT


# ============ helpers ====================================================
def _placement_weights(pts, ops, saliency, placement):
    if placement == "saliency":
        ix = np.clip(pts[:, 0].astype(int), 0, saliency.shape[1] - 1)
        iy = np.clip(pts[:, 1].astype(int), 0, saliency.shape[0] - 1)
        s = saliency[iy, ix]
        return ops * (s + 0.04)
    return ops  # uniform


def _seg_color(p0, p1, ref_img, ink, W, H):
    mx = int(np.clip((p0[0] + p1[0]) / 2, 0, W - 1))
    my = int(np.clip((p0[1] + p1[1]) / 2, 0, H - 1))
    return ref_img[my, mx] * ink


def render_walks(pts, ops, cov2d_kept, ref_img, saliency, params, W, H, seed):
    rng = np.random.default_rng(seed)
    w = _placement_weights(pts, ops, saliency, params["PLACEMENT"])
    w = w / w.sum()
    tree = cKDTree(pts)
    canvas = np.tile(PAPER_COLOR, (H, W, 1)).astype(np.float64)

    R = params["STEP_RADIUS_PX"]
    steps = params["STEPS"]
    fb = params["FORWARD_BIAS"]
    alpha = params["STROKE_ALPHA"]
    ink = params["INK_DARKEN"]
    mode = params.get("STROKE_MODE", "line")
    width = float(params.get("STROKE_WIDTH", 1.0))
    splat_scale = float(params.get("SPLAT_SCALE", 1.0))
    splat_alpha_scale = float(params.get("SPLAT_ALPHA_SCALE", 1.0))
    splat_min_sigma = float(params.get("SPLAT_MIN_SIGMA", 0.0))
    splat_max_sigma = params.get("SPLAT_MAX_SIGMA", None)
    splat_max_sigma = None if splat_max_sigma is None else float(splat_max_sigma)
    n_stamps = int(params.get("N_STAMPS", 4))
    drawn = 0

    for _ in range(params["N_WALKERS"]):
        cur = int(rng.choice(len(pts), p=w))
        prev_dir = None
        for _ in range(steps):
            result = walk_step(cur, prev_dir, tree, pts, R, fb, rng)
            if result is None:
                break
            nxt, prev_dir = result
            p0, p1 = pts[cur], pts[nxt]
            color = _seg_color(p0, p1, ref_img, ink, W, H)
            if mode == "splat":
                cov_avg = 0.5 * (cov2d_kept[cur] + cov2d_kept[nxt])
                add_splat_stroke(canvas, p0, p1, cov_avg,
                                 color, alpha * splat_alpha_scale,
                                 W, H, n_stamps=n_stamps, scale=splat_scale,
                                 min_sigma_px=splat_min_sigma,
                                 max_sigma_px=splat_max_sigma)
            else:
                add_line(canvas, p0[0], p0[1], p1[0], p1[1],
                         color, alpha, W, H, width=width)
            cur = nxt
            drawn += 1
    return canvas, drawn


# ============ project Gaussians ==========================================
data = load_3dgs_ply(cfg.PLY)
G = decode_3dgs(data)
print(f"\n[walks] loaded {len(data)} Gaussians from {cfg.PLY}")

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
ops = G["opacities"][keep]
cov2d_kept = cov2d[keep]                           # needed for splat-stamp strokes
print(f"[walks] projected -> {len(pts)} visible")

ref_img = load_ref_image(REF_IMAGE, W, H)
saliency = compute_saliency(ref_img)


# ============ render each variant ========================================
canvases = []
for params in VARIANTS:
    t0 = time.time()
    canvas, drawn = render_walks(pts, ops, cov2d_kept, ref_img, saliency,
                                   params, W, H, SEED)
    if params.get("STROKE_MODE", "line") == "splat":
        stroke_desc = (f"splat x{params.get('SPLAT_SCALE', 1.0):.1f} "
                       f"n={params.get('N_STAMPS', 4)}")
    else:
        stroke_desc = f"line w={float(params.get('STROKE_WIDTH', 1.0)):.1f}"
    label = (
        f"walks_{params['name']}  |  N={params['N_WALKERS']}  steps={params['STEPS']}  "
        f"R={params['STEP_RADIUS_PX']}px  bias={params['FORWARD_BIAS']:.1f}  "
        f"a={params['STROKE_ALPHA']}  ink={params['INK_DARKEN']}  "
        f"{stroke_desc}  seed={SEED}"
    )
    canvas = stamp(canvas, label, W, H)
    out_path = f"images/{SCENE_BASE}_walks_{params['name']}.png"
    plt.imsave(out_path, np.clip(canvas, 0, 1))
    canvases.append(canvas)
    print(f"  {params['name']:<11} {time.time() - t0:5.1f}s  ({drawn:>6d} segs) -> {out_path}")


# ============ contact sheet ==============================================
GRID_COLS = 3
thumb_w = 360
thumb_h = int(H * thumb_w / W)
thumbs = [
    np.asarray(Image.fromarray((c * 255).astype(np.uint8)).resize((thumb_w, thumb_h))) / 255.0
    for c in canvases
]
while len(thumbs) % GRID_COLS:
    thumbs.append(np.tile(PAPER_COLOR, (thumb_h, thumb_w, 1)))
rows = [
    np.concatenate(thumbs[r * GRID_COLS:(r + 1) * GRID_COLS], axis=1)
    for r in range(len(thumbs) // GRID_COLS)
]
grid = np.concatenate(rows, axis=0)
plt.imsave(f"images/{SCENE_BASE}_walks_grid.png", np.clip(grid, 0, 1))
print(f"grid -> images/{SCENE_BASE}_walks_grid.png  ({grid.shape[1]}x{grid.shape[0]})")
