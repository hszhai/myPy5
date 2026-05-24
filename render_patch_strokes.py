"""Patch-sampling stroke renderer.

Different stroke logic from render_stroke_variants.py:

  for each of N_PATCHES iterations:
      pick a random small area (a disk of radius R in image space,
        centred on a Gaussian sampled by `PLACEMENT` weight)
      find all projected splats whose 2D centres fall inside the disk
      link some of them with line segments (pairs OR a polyline path)

Many iterations accumulate into the drawing. Different from KNN-link
because patches OVERLAP -- the same splat participates in many patches,
density builds where the subject is, and stroke direction varies with
each random sample (closer to how a hand actually draws than a graph).

Outputs:
  images/<scene>_patches_<variant>.png
  images/<scene>_patches_grid.png

Run:  ~/miniconda3/bin/python render_patch_strokes.py
"""
import os
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from PIL import Image

from gsplat import (
    load_3dgs_ply, decode_3dgs, make_camera, project_perspective, cull,
)
from npr_utils import (
    stamp, load_ref_image, compute_saliency, add_line, PAPER_COLOR,
)
from scene_io import load_scene

SCENE_NAME = os.environ.get("SCENE_NAME", "redhead")
cfg = load_scene(SCENE_NAME)


# ============ variants ====================================================
# MODE: how to link splats inside a single patch
#   "pairs"   -- LINKS_PER_PATCH independent line segments
#                between random pairs of splats in the patch.
#   "path"    -- LINKS_PER_PATCH+1 random splats connected as a polyline
#                (one continuous zigzag through the patch).
# PLACEMENT: where the patch centres are picked
#   "saliency" -- sample a splat weighted by saliency (face/lip/eye peaks)
#                 then use its 2D position
#   "uniform"  -- sample any splat uniformly (covers more of the body)
VARIANTS = [
    dict(name="fine",        N_PATCHES=3000, RADIUS_PX=15, LINKS_PER_PATCH=2, MODE="pairs", STROKE_ALPHA=0.40, INK_DARKEN=0.40, PLACEMENT="saliency"),
    dict(name="medium",      N_PATCHES=1500, RADIUS_PX=30, LINKS_PER_PATCH=3, MODE="pairs", STROKE_ALPHA=0.35, INK_DARKEN=0.40, PLACEMENT="saliency"),
    dict(name="loose_path",  N_PATCHES= 900, RADIUS_PX=45, LINKS_PER_PATCH=5, MODE="path",  STROKE_ALPHA=0.30, INK_DARKEN=0.30, PLACEMENT="saliency"),
    dict(name="dense_accum", N_PATCHES=5000, RADIUS_PX=20, LINKS_PER_PATCH=2, MODE="pairs", STROKE_ALPHA=0.20, INK_DARKEN=0.30, PLACEMENT="saliency"),
    dict(name="uniform_pl",  N_PATCHES=3000, RADIUS_PX=20, LINKS_PER_PATCH=2, MODE="pairs", STROKE_ALPHA=0.30, INK_DARKEN=0.40, PLACEMENT="uniform"),
    dict(name="bold_path",   N_PATCHES=2000, RADIUS_PX=22, LINKS_PER_PATCH=3, MODE="path",  STROKE_ALPHA=0.60, INK_DARKEN=0.20, PLACEMENT="saliency"),
]
SEED = 11
SCENE_BASE = os.path.splitext(os.path.basename(cfg.PLY))[0].lower()
REF_IMAGE = cfg.OUT


# ============ helpers =====================================================
def _placement_weights(pts, opacities, saliency, placement):
    if placement == "saliency":
        ix = np.clip(pts[:, 0].astype(int), 0, saliency.shape[1] - 1)
        iy = np.clip(pts[:, 1].astype(int), 0, saliency.shape[0] - 1)
        s = saliency[iy, ix]
        return opacities * (s + 0.04)
    if placement == "uniform":
        return opacities
    raise ValueError(f"unknown PLACEMENT: {placement!r}")


def _seg_color(p0, p1, ref_img, ink):
    """Stroke colour = the splat reference image sampled at the segment midpoint."""
    H, W = ref_img.shape[:2]
    mx = int(np.clip((p0[0] + p1[0]) / 2, 0, W - 1))
    my = int(np.clip((p0[1] + p1[1]) / 2, 0, H - 1))
    return ref_img[my, mx] * ink


def render_patches(pts, ops, ref_img, saliency, params, W, H, seed):
    rng = np.random.default_rng(seed)
    w = _placement_weights(pts, ops, saliency, params["PLACEMENT"])
    w = w / w.sum()
    tree = cKDTree(pts)

    canvas = np.tile(PAPER_COLOR, (H, W, 1)).astype(np.float64)
    R = params["RADIUS_PX"]
    n_per = params["LINKS_PER_PATCH"]
    mode = params["MODE"]
    alpha = params["STROKE_ALPHA"]
    ink = params["INK_DARKEN"]

    drawn = 0
    for _ in range(params["N_PATCHES"]):
        ci = rng.choice(len(pts), p=w)
        cx, cy = pts[ci]
        idx = tree.query_ball_point([cx, cy], R)
        if len(idx) < 2:
            continue
        patch = pts[idx]

        if mode == "pairs":
            n_pairs = min(n_per, len(idx) // 2)
            for _ in range(n_pairs):
                a, b = rng.choice(len(patch), 2, replace=False)
                p0, p1 = patch[a], patch[b]
                add_line(canvas, p0[0], p0[1], p1[0], p1[1],
                         _seg_color(p0, p1, ref_img, ink), alpha, W, H)
                drawn += 1
        elif mode == "path":
            n_pts = min(n_per + 1, len(idx))
            picks = rng.choice(len(patch), n_pts, replace=False)
            for k in range(n_pts - 1):
                p0, p1 = patch[picks[k]], patch[picks[k + 1]]
                add_line(canvas, p0[0], p0[1], p1[0], p1[1],
                         _seg_color(p0, p1, ref_img, ink), alpha, W, H)
                drawn += 1
    return canvas, drawn


# ============ project Gaussians ==========================================
data = load_3dgs_ply(cfg.PLY)
G = decode_3dgs(data)
print(f"\n[patches] loaded {len(data)} Gaussians from {cfg.PLY}")

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
pts, ops = mean2d[keep], G["opacities"][keep]
print(f"[patches] projected -> {len(pts)} visible")

ref_img = load_ref_image(REF_IMAGE, W, H)
saliency = compute_saliency(ref_img)


# ============ render each variant ========================================
canvases = []
for params in VARIANTS:
    t0 = time.time()
    canvas, drawn = render_patches(pts, ops, ref_img, saliency, params, W, H, SEED)
    label = (
        f"patch_{params['name']}  |  N={params['N_PATCHES']}  R={params['RADIUS_PX']}px  "
        f"L={params['LINKS_PER_PATCH']}  {params['MODE']}  "
        f"a={params['STROKE_ALPHA']}  ink={params['INK_DARKEN']}  "
        f"{params['PLACEMENT']}  seed={SEED}"
    )
    canvas = stamp(canvas, label, W, H)
    out_path = f"images/{SCENE_BASE}_patches_{params['name']}.png"
    plt.imsave(out_path, np.clip(canvas, 0, 1))
    canvases.append(canvas)
    print(f"  {params['name']:<13} {time.time() - t0:5.1f}s  ({drawn:>6d} segs) -> {out_path}")


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
plt.imsave(f"images/{SCENE_BASE}_patches_grid.png", np.clip(grid, 0, 1))
print(f"grid -> images/{SCENE_BASE}_patches_grid.png  ({grid.shape[1]}x{grid.shape[0]})")
