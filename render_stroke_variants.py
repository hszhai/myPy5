"""Render several KNN-stroke variants of the Siyun portrait, each with
its parameters stamped into the corner so you can keep track when you
come back later. Composition is shared with render_siyun.py.

Outputs:
  images/{SCENE_BASE}_strokes_<name>.png   for each variant in VARIANTS
  images/{SCENE_BASE}_strokes_grid.png     2x2 comparison contact sheet

Run:  ~/miniconda3/bin/python render_stroke_variants.py
"""
import os
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from PIL import Image, ImageDraw, ImageFont

from gsplat import load_3dgs_ply, decode_3dgs, make_camera, project_perspective, cull
# Scene config now comes from scenes/<name>.json -- switch with the env var:
#   SCENE_NAME=siyun python render_stroke_variants.py
from scene_io import load_scene
SCENE_NAME = os.environ.get("SCENE_NAME", "redhead")
cfg = load_scene(SCENE_NAME)

# ============ knobs to vary ===============================================
# WEIGHT_MODE controls how stroke seeds are sampled from the Gaussian set:
#   "opacity"  -- bias by per-Gaussian opacity (uniform across the silhouette)
#   "saliency" -- bias by edge strength in the splat reference image; this
#                 concentrates strokes on the features the splat renders well
#                 (eyes, mouth, finger contours, hair edges, dress folds).
VARIANTS = [
    dict(name="sparse",        N_STROKES=1200, N_LINKS=2, MAX_LEN_PX=30, STROKE_ALPHA=0.55, INK_DARKEN=0.65, WEIGHT_MODE="opacity"),
    dict(name="baseline",      N_STROKES=2500, N_LINKS=3, MAX_LEN_PX=35, STROKE_ALPHA=0.35, INK_DARKEN=0.50, WEIGHT_MODE="opacity"),
    dict(name="dense",         N_STROKES=4000, N_LINKS=5, MAX_LEN_PX=30, STROKE_ALPHA=0.22, INK_DARKEN=0.35, WEIGHT_MODE="opacity"),
    dict(name="inky",          N_STROKES=2500, N_LINKS=3, MAX_LEN_PX=35, STROKE_ALPHA=0.65, INK_DARKEN=0.00, WEIGHT_MODE="opacity"),
    dict(name="features",      N_STROKES=2200, N_LINKS=2, MAX_LEN_PX=22, STROKE_ALPHA=0.55, INK_DARKEN=0.30, WEIGHT_MODE="saliency"),
    dict(name="features_inky", N_STROKES=1800, N_LINKS=2, MAX_LEN_PX=20, STROKE_ALPHA=0.80, INK_DARKEN=0.00, WEIGHT_MODE="saliency"),
    dict(name="features_tight",N_STROKES=1500, N_LINKS=1, MAX_LEN_PX=14, STROKE_ALPHA=0.85, INK_DARKEN=0.00, WEIGHT_MODE="saliency"),
    # Finer-resolution variants: shorter strokes + more seeds, possible now that the
    # canvas is 1080x1440. K=1 keeps each mark isolated rather than meshing.
    dict(name="features_fine",   N_STROKES=5000, N_LINKS=1, MAX_LEN_PX=10, STROKE_ALPHA=0.75, INK_DARKEN=0.10, WEIGHT_MODE="saliency"),
    dict(name="features_finer",  N_STROKES=8000, N_LINKS=1, MAX_LEN_PX=7,  STROKE_ALPHA=0.85, INK_DARKEN=0.00, WEIGHT_MODE="saliency"),
    # ---- "stronger" variants ----
    # SALIENCY_POWER>1 concentrates seeds onto the brightest saliency peaks
    # (eyes, lips). COLOR_MODE="reference" samples the splat-rendered image
    # at each seed for actual shading instead of the flat DC colour.
    dict(name="features_pow2",    N_STROKES=6000, N_LINKS=1, MAX_LEN_PX=9,  STROKE_ALPHA=0.85, INK_DARKEN=0.00, WEIGHT_MODE="saliency", SALIENCY_POWER=2.0),
    dict(name="features_shaded", N_STROKES=9000, N_LINKS=1, MAX_LEN_PX=8,  STROKE_ALPHA=0.70, INK_DARKEN=0.85, WEIGHT_MODE="saliency", SALIENCY_POWER=1.5, COLOR_MODE="reference"),
    dict(name="features_strong", N_STROKES=12000,N_LINKS=1, MAX_LEN_PX=7,  STROKE_ALPHA=0.75, INK_DARKEN=0.75, WEIGHT_MODE="saliency", SALIENCY_POWER=2.5, COLOR_MODE="reference"),
]
SEED = 7
PAPER_COLOR = np.array([0.97, 0.96, 0.93])
REF_IMAGE = cfg.OUT                         # the scene's splat render
SCENE_BASE = os.path.splitext(os.path.basename(cfg.PLY))[0].lower()  # e.g. "redhead"


# ============ helpers =====================================================
def get_mono_font(size=11):
    """Find a monospace font on this macOS box; fall back to PIL default."""
    for path in [
        "/System/Library/Fonts/Menlo.ttc",
        "/System/Library/Fonts/SFNSMono.ttf",
        "/System/Library/Fonts/Supplemental/Andale Mono.ttf",
    ]:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def add_line(canvas, x0, y0, x1, y1, color, alpha, W, H):
    """Cheap aliased line rasteriser (alpha-composite)."""
    n = int(max(abs(x1 - x0), abs(y1 - y0)) + 1)
    if n <= 1:
        return
    xs = np.linspace(x0, x1, n).astype(int)
    ys = np.linspace(y0, y1, n).astype(int)
    ok = (xs >= 0) & (xs < W) & (ys >= 0) & (ys < H)
    xs, ys = xs[ok], ys[ok]
    canvas[ys, xs] = canvas[ys, xs] * (1.0 - alpha) + color * alpha


def stamp(canvas, text, W, H):
    """Write `text` near the TOP-left of the canvas with a subtle halo."""
    pil = Image.fromarray((canvas * 255).astype(np.uint8))
    draw = ImageDraw.Draw(pil)
    font = get_mono_font(11)
    pad = 12
    x, y = pad, pad
    draw.text((x + 1, y + 1), text, fill=(230, 228, 220), font=font)   # paper-tone halo
    draw.text((x, y),         text, fill=(70, 70, 70),    font=font)   # ink
    return np.asarray(pil) / 255.0


def load_ref_image(ref_path, W, H):
    """Load and resize the splat reference image to (H, W, 3) float in [0, 1]."""
    ref = plt.imread(ref_path)
    if ref.shape[-1] == 4:
        ref = ref[..., :3]
    if ref.shape[:2] != (H, W):
        ref = np.asarray(Image.fromarray((ref * 255).astype(np.uint8))
                          .resize((W, H))) / 255.0
    return ref.astype(np.float64)


def compute_saliency(ref):
    """Per-pixel saliency from an already-loaded RGB reference image.

    Strategy:
      subject mask    -- only sample on the subject (not the black bg).
      capped edges    -- fine DoG (sigma 0.5/1.5) picks up small features
                         (eyes, lips, fingers); CAPPED so the busy hair
                         texture doesn't swamp everything else.
      shadow boost    -- quadratic emphasis of dark regions on the subject
                         (eye sockets, nostrils, mouth corners, finger
                         gaps -- these are exactly the features we want).
    """
    from scipy.ndimage import gaussian_filter
    luma = ref @ np.array([0.299, 0.587, 0.114])
    subject = (luma > 0.08).astype(np.float64)
    dog_fine = np.abs(gaussian_filter(luma, 0.5) - gaussian_filter(luma, 1.5))
    edges = np.minimum(dog_fine * 10.0, 0.4) * subject                  # cap @ 0.4
    shadow = np.clip(0.45 - luma, 0.0, 1.0) * subject
    shadow_strong = (shadow ** 2) * 4.0                                 # quadratic, heavy
    saliency = (edges + shadow_strong).astype(np.float64)
    plt.imsave(f"images/_{SCENE_BASE}_saliency_debug.png",
               np.clip(saliency / max(saliency.max(), 1e-6), 0, 1), cmap="hot")
    return saliency


def make_weights(pts, opacities, mode, saliency, power=1.0):
    """Per-Gaussian sampling weight, > 0 everywhere.

    `power` raises the saliency to a power before mixing; values >1
    concentrate seeds onto the brightest peaks (eyes / mouth / lips).
    """
    if mode == "opacity":
        return opacities
    if mode == "saliency":
        ix = np.clip(pts[:, 0].astype(int), 0, saliency.shape[1] - 1)
        iy = np.clip(pts[:, 1].astype(int), 0, saliency.shape[0] - 1)
        s = saliency[iy, ix]
        baseline = 0.02 if power > 1.0 else 0.04          # tighter focus when power is on
        return opacities * (s ** power + baseline)
    raise ValueError(f"unknown WEIGHT_MODE: {mode!r}")


def sample_reference(pts, ref_img):
    """Pick up the splat reference image's RGB at each seed position."""
    H, W = ref_img.shape[:2]
    ix = np.clip(pts[:, 0].astype(int), 0, W - 1)
    iy = np.clip(pts[:, 1].astype(int), 0, H - 1)
    return ref_img[iy, ix]


def render_one(pts, cols, ops, params, W, H, seed, saliency=None, ref_img=None):
    """KNN-link stroke render for one parameter set."""
    rng = np.random.default_rng(seed)
    weights = make_weights(pts, ops, params["WEIGHT_MODE"], saliency,
                            power=params.get("SALIENCY_POWER", 1.0))
    prob = weights / weights.sum()
    n = min(params["N_STROKES"], len(pts))
    sub = rng.choice(len(pts), size=n, p=prob, replace=False)
    sub_pts = pts[sub]
    color_mode = params.get("COLOR_MODE", "gaussian")
    if color_mode == "reference" and ref_img is not None:
        base = sample_reference(sub_pts, ref_img)
    else:
        base = cols[sub]
    sub_cols = np.clip(base * params["INK_DARKEN"], 0.0, 1.0)
    tree = cKDTree(sub_pts)
    _, nn = tree.query(sub_pts, k=params["N_LINKS"] + 1)
    canvas = np.tile(PAPER_COLOR, (H, W, 1)).astype(np.float64)
    for i in range(len(sub_pts)):
        x0, y0 = sub_pts[i]
        c = sub_cols[i]
        for k in range(1, params["N_LINKS"] + 1):
            j = nn[i, k]
            x1, y1 = sub_pts[j]
            if np.hypot(x1 - x0, y1 - y0) > params["MAX_LEN_PX"]:
                continue
            add_line(canvas, x0, y0, x1, y1, c, params["STROKE_ALPHA"], W, H)
    return canvas


# ============ project Gaussians once ======================================

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
pts, cols, ops = mean2d[keep], G["colors"][keep], G["opacities"][keep]
print(f"projected {len(G['xyz'])} -> {len(pts)} visible")

# Load the reference image ONCE (used by saliency + reference-colour strokes).
ref_img = load_ref_image(REF_IMAGE, W, H)
saliency = compute_saliency(ref_img)
print(f"saliency map: min={saliency.min():.3f}  max={saliency.max():.3f}  "
      f"mean={saliency.mean():.3f}")


# ============ render each variant =========================================
canvases = []
for params in VARIANTS:
    t0 = time.time()
    canvas = render_one(pts, cols, ops, params, W, H, SEED,
                        saliency=saliency, ref_img=ref_img)
    label = (
        f"{params['name']}  |  N={params['N_STROKES']}  K={params['N_LINKS']}  "
        f"maxlen={params['MAX_LEN_PX']:.0f}px  a={params['STROKE_ALPHA']}  "
        f"ink={params['INK_DARKEN']}  seed={SEED}"
    )
    canvas = stamp(canvas, label, W, H)
    out_path = f"images/{SCENE_BASE}_strokes_{params['name']}.png"
    plt.imsave(out_path, np.clip(canvas, 0, 1))
    canvases.append(canvas)
    print(f"  {params['name']:<10} {time.time() - t0:5.1f}s -> {out_path}")


# ============ contact sheet (auto-rows of 3) ==============================
GRID_COLS = 3
thumb_w = 360
thumb_h = int(H * thumb_w / W)
thumbs = [
    np.asarray(Image.fromarray((c * 255).astype(np.uint8)).resize((thumb_w, thumb_h))) / 255.0
    for c in canvases
]
# Pad to a full last row with a paper-tone blank if needed
while len(thumbs) % GRID_COLS:
    thumbs.append(np.tile(PAPER_COLOR, (thumb_h, thumb_w, 1)))
rows = [
    np.concatenate(thumbs[r * GRID_COLS:(r + 1) * GRID_COLS], axis=1)
    for r in range(len(thumbs) // GRID_COLS)
]
grid = np.concatenate(rows, axis=0)
plt.imsave(f"images/{SCENE_BASE}_strokes_grid.png", np.clip(grid, 0, 1))
print(f"grid -> images/{SCENE_BASE}_strokes_grid.png  ({grid.shape[1]}x{grid.shape[0]})")
