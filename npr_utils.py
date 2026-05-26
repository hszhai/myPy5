"""Shared helpers for the NPR stroke pipelines (render_stroke_variants,
render_patch_strokes, tune_patch_strokes).

Kept tiny on purpose: paper colour, a 1-px line rasteriser, the splat
reference loader, the saliency map, the stamp function and a font picker.
Anything scene-specific lives in render_<scene>.py.
"""
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

try:
    from numba import njit
except ImportError:
    njit = None


PAPER_COLOR = np.array([0.97, 0.96, 0.93])


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


def add_line(canvas, x0, y0, x1, y1, color, alpha, W, H, width=1.0):
    """Alpha-composite a line segment onto `canvas` in-place.

    width <= 1.0   fast 1-px aliased rasteriser
    width >  1.0   vectorised thick line: point-to-segment distance over
                   the segment's AABB, with a soft 1-pixel anti-aliased
                   edge (1.0 inside, 0.0 outside, linear in between).
    """
    if width <= 1.0:
        n = int(max(abs(x1 - x0), abs(y1 - y0)) + 1)
        if n <= 1:
            return
        xs = np.linspace(x0, x1, n).astype(int)
        ys = np.linspace(y0, y1, n).astype(int)
        ok = (xs >= 0) & (xs < W) & (ys >= 0) & (ys < H)
        xs, ys = xs[ok], ys[ok]
        canvas[ys, xs] = canvas[ys, xs] * (1.0 - alpha) + color * alpha
        return

    r = width / 2.0
    xmin = max(int(min(x0, x1) - r), 0)
    xmax = min(int(max(x0, x1) + r) + 1, W)
    ymin = max(int(min(y0, y1) - r), 0)
    ymax = min(int(max(y0, y1) + r) + 1, H)
    if xmin >= xmax or ymin >= ymax:
        return

    xs_ = np.arange(xmin, xmax)
    ys_ = np.arange(ymin, ymax)
    px, py = np.meshgrid(xs_, ys_)

    dx = x1 - x0
    dy = y1 - y0
    seg_len2 = dx * dx + dy * dy
    if seg_len2 < 1e-9:
        d2 = (px - x0) ** 2 + (py - y0) ** 2
    else:
        t = np.clip(((px - x0) * dx + (py - y0) * dy) / seg_len2, 0.0, 1.0)
        cxp = x0 + t * dx
        cyp = y0 + t * dy
        d2 = (px - cxp) ** 2 + (py - cyp) ** 2

    mask = np.clip(r + 0.5 - np.sqrt(d2), 0.0, 1.0)        # soft edge
    a_eff = (alpha * mask)[..., None]
    sub = canvas[ymin:ymax, xmin:xmax]
    canvas[ymin:ymax, xmin:xmax] = sub * (1.0 - a_eff) + color * a_eff


def _add_splat_stroke_fallback(canvas, p0, p1, cov, color, alpha, W, H, n_stamps):
    """Pure-Python fallback (creates temporaries per-stamp)."""
    det = cov[0, 0] * cov[1, 1] - cov[0, 1] * cov[1, 0]
    if det <= 1e-9:
        return
    inv00 = cov[1, 1] / det
    inv11 = cov[0, 0] / det
    inv01 = -cov[0, 1] / det
    rx = 3.0 * np.sqrt(max(cov[0, 0], 1e-6))
    ry = 3.0 * np.sqrt(max(cov[1, 1], 1e-6))

    for t in np.linspace(0.0, 1.0, n_stamps):
        cx = p0[0] + t * (p1[0] - p0[0])
        cy = p0[1] + t * (p1[1] - p0[1])
        x0i = max(int(cx - rx), 0); x1i = min(int(cx + rx) + 1, W)
        y0i = max(int(cy - ry), 0); y1i = min(int(cy + ry) + 1, H)
        if x0i >= x1i or y0i >= y1i:
            continue
        xs = np.arange(x0i, x1i) - cx
        ys = np.arange(y0i, y1i) - cy
        q = inv00 * (xs * xs)[None, :] + inv11 * (ys * ys)[:, None] \
            + (2 * inv01) * np.outer(ys, xs)
        a = (alpha * np.exp(-0.5 * q))[..., None]
        canvas[y0i:y1i, x0i:x1i] = canvas[y0i:y1i, x0i:x1i] * (1.0 - a) + color * a


if njit is not None:
    @njit(cache=True)
    def _add_splat_stroke_numba(canvas, p0x, p0y, p1x, p1y,
                                cov00, cov01, cov11,
                                color0, color1, color2,
                                alpha, W, H, n_stamps):
        """Numba-compiled core: zero temporary allocations per stamp."""
        det = cov00 * cov11 - cov01 * cov01
        if det <= 1e-9:
            return
        inv00 = cov11 / det
        inv11 = cov00 / det
        inv01 = -cov01 / det
        rx = 3.0 * np.sqrt(max(cov00, 1e-6))
        ry = 3.0 * np.sqrt(max(cov11, 1e-6))

        for i in range(n_stamps):
            if n_stamps > 1:
                t = i / (n_stamps - 1)
            else:
                t = 0.0
            cx = p0x + t * (p1x - p0x)
            cy = p0y + t * (p1y - p0y)
            x0i = max(int(cx - rx), 0)
            x1i = min(int(cx + rx) + 1, W)
            y0i = max(int(cy - ry), 0)
            y1i = min(int(cy + ry) + 1, H)
            if x0i >= x1i or y0i >= y1i:
                continue

            for yi in range(y0i, y1i):
                dy = yi - cy
                for xi in range(x0i, x1i):
                    dx = xi - cx
                    q = inv00 * dx * dx + inv11 * dy * dy + 2.0 * inv01 * dx * dy
                    a = alpha * np.exp(-0.5 * q)
                    canvas[yi, xi, 0] = canvas[yi, xi, 0] * (1.0 - a) + color0 * a
                    canvas[yi, xi, 1] = canvas[yi, xi, 1] * (1.0 - a) + color1 * a
                    canvas[yi, xi, 2] = canvas[yi, xi, 2] * (1.0 - a) + color2 * a


def add_splat_stroke(canvas, p0, p1, cov2d, color, alpha, W, H,
                     n_stamps=5, scale=1.0, min_sigma_px=0.0,
                     max_sigma_px=None):
    """Stamp a chain of Gaussian splats along the segment p0 -> p1.

    cov2d  : 2x2 covariance (typically the average of the two endpoint
             splats' projected covariances; scaled by `scale^2`).
    n_stamps : how many overlapping splats to place along the segment.
    min/max_sigma_px : clamp the final projected Gaussian stddev in pixels.
             This keeps stroke-mode splats brush-like instead of inheriting
             oversized raw 3DGS splats.

    The stroke therefore carries the splat's actual visual character
    (anisotropy, size) instead of being a flat line.
    """
    # Pre-process covariance in Python (eigh clamping not worth JITting for 2x2)
    cov = cov2d * (scale * scale)
    if min_sigma_px or max_sigma_px is not None:
        vals, vecs = np.linalg.eigh(cov)
        sigmas = np.sqrt(np.maximum(vals, 1e-9))
        lo = max(float(min_sigma_px), 0.0)
        hi = None if max_sigma_px is None else max(float(max_sigma_px), lo + 1e-6)
        sigmas = np.maximum(sigmas, lo)
        if hi is not None:
            sigmas = np.minimum(sigmas, hi)
        cov = (vecs * (sigmas * sigmas)) @ vecs.T

    if njit is not None:
        _add_splat_stroke_numba(
            canvas, float(p0[0]), float(p0[1]), float(p1[0]), float(p1[1]),
            float(cov[0, 0]), float(cov[0, 1]), float(cov[1, 1]),
            float(color[0]), float(color[1]), float(color[2]),
            float(alpha), int(W), int(H), int(n_stamps),
        )
    else:
        _add_splat_stroke_fallback(canvas, p0, p1, cov, color, alpha, W, H, n_stamps)


def stamp(canvas, text, W, H):
    """Write `text` near the top-left of the canvas with a subtle halo."""
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


def compute_saliency(ref, debug_path=None):
    """Per-pixel saliency from an already-loaded RGB reference image.

    Strategy:
      subject mask   only sample on the subject (skip near-black bg).
      capped edges   fine DoG (sigma 0.5/1.5) picks up small features
                     (eyes, lips, fingers); CAPPED so the busy hair
                     texture doesn't swamp everything else.
      shadow boost   quadratic emphasis of dark regions on the subject
                     (eye sockets, nostrils, mouth corners, finger gaps).

    If `debug_path` is given, write a hot colormap visualisation there.
    """
    from scipy.ndimage import gaussian_filter
    luma = ref @ np.array([0.299, 0.587, 0.114])
    subject = (luma > 0.08).astype(np.float64)
    dog_fine = np.abs(gaussian_filter(luma, 0.5) - gaussian_filter(luma, 1.5))
    edges = np.minimum(dog_fine * 10.0, 0.4) * subject                  # cap @ 0.4
    shadow = np.clip(0.45 - luma, 0.0, 1.0) * subject
    shadow_strong = (shadow ** 2) * 4.0                                 # quadratic, heavy
    saliency = (edges + shadow_strong).astype(np.float64)
    if debug_path is not None:
        plt.imsave(debug_path,
                   np.clip(saliency / max(saliency.max(), 1e-6), 0, 1),
                   cmap="hot")
    return saliency


def sample_reference(pts, ref_img):
    """Look up the splat reference image's RGB at each 2D seed position."""
    H, W = ref_img.shape[:2]
    ix = np.clip(pts[:, 0].astype(int), 0, W - 1)
    iy = np.clip(pts[:, 1].astype(int), 0, H - 1)
    return ref_img[iy, ix]


def walk_step(current, prev_dir, tree, pts, step_radius, forward_bias, rng,
              direction_field=None, global_dir=None):
    """One step of a surface walker.

    Returns (next_idx, new_dir_unit), or None if there's no neighbour to step to.

    The "target direction" the walker aligns toward is picked in priority:
      direction_field  (H, W, 2) array of unit vectors per pixel  -- "noise" mode
      global_dir       (2,) unit vector applied uniformly         -- "global" mode
      prev_dir         the direction the walker came from         -- "momentum"
      else             None -> uniform random pick (first step)

    forward_bias  controls how strongly the walker aligns with the target
                  (0 = uniform; 5-8 = confident sweeps; 15+ = near-straight).
    """
    cand = tree.query_ball_point(pts[current], step_radius)
    cand = np.asarray([i for i in cand if i != current])
    if len(cand) == 0:
        return None
    deltas = pts[cand] - pts[current]
    norms = np.linalg.norm(deltas, axis=1)
    norms = np.maximum(norms, 1e-6)
    dirs = deltas / norms[:, None]

    target = None
    if direction_field is not None:
        ix = int(np.clip(pts[current, 0], 0, direction_field.shape[1] - 1))
        iy = int(np.clip(pts[current, 1], 0, direction_field.shape[0] - 1))
        target = direction_field[iy, ix]
    elif global_dir is not None:
        target = global_dir
    elif prev_dir is not None:
        target = prev_dir

    if target is None:
        weights = np.ones(len(cand))
    else:
        weights = np.exp(forward_bias * (dirs @ target))
    weights = weights / weights.sum()
    pick = rng.choice(len(cand), p=weights)
    return int(cand[pick]), dirs[pick]


def walk_step_3d(current, prev_dir, tree3d, xyz, screen_pts, max_screen_step,
                 forward_bias, rng, direction_field=None, global_dir=None,
                 k_neighbors=64):
    """One surface step using 3D neighbours but 2D direction scoring.

    The candidate set comes from nearby PLY/Gaussian positions in 3D, which
    avoids walking across unrelated surfaces that merely overlap on screen.
    The direction target is still interpreted in screen space because the
    strokes are ultimately drawn in the projected image.
    """
    k = min(int(k_neighbors) + 1, len(xyz))
    if k <= 1:
        return None
    _, idx = tree3d.query(xyz[current], k=k)
    cand = np.asarray(idx, dtype=int)
    cand = cand[cand != current]
    if len(cand) == 0:
        return None

    deltas = screen_pts[cand] - screen_pts[current]
    lengths = np.linalg.norm(deltas, axis=1)
    ok = (lengths > 1e-6) & (lengths <= max_screen_step)
    cand = cand[ok]
    deltas = deltas[ok]
    lengths = lengths[ok]
    if len(cand) == 0:
        return None

    dirs = deltas / lengths[:, None]
    target = None
    if direction_field is not None:
        ix = int(np.clip(screen_pts[current, 0], 0, direction_field.shape[1] - 1))
        iy = int(np.clip(screen_pts[current, 1], 0, direction_field.shape[0] - 1))
        target = direction_field[iy, ix]
    elif global_dir is not None:
        target = global_dir
    elif prev_dir is not None:
        target = prev_dir

    distance_weights = np.exp(-0.5 * (lengths / max(max_screen_step, 1e-6)) ** 2)
    if target is None:
        weights = distance_weights
    else:
        weights = distance_weights * np.exp(forward_bias * (dirs @ target))
    weights = weights / weights.sum()
    pick = rng.choice(len(cand), p=weights)
    return int(cand[pick]), dirs[pick]


def make_noise_direction_field(W, H, sigma=50.0, seed=0):
    """Smooth random direction field (unit vectors per pixel).

    Two independent Gaussian-filtered noise channels (vx, vy), normalised
    to unit length per pixel. Larger sigma -> smoother, larger eddies.
    Returns (H, W, 2) array.
    """
    from scipy.ndimage import gaussian_filter
    rng = np.random.default_rng(int(seed))
    vx = gaussian_filter(rng.standard_normal((H, W)).astype(np.float32), sigma)
    vy = gaussian_filter(rng.standard_normal((H, W)).astype(np.float32), sigma)
    mag = np.sqrt(vx * vx + vy * vy) + 1e-9
    return np.stack([vx / mag, vy / mag], axis=-1)


def make_envelope_field(W, H, mode="none", sigma=50.0, saliency=None, seed=0):
    """Generate a *spatial* envelope field for offset modulation.

    Returns (H, W) array with values in [0, 1] representing offset amplitude
    at each pixel. Spatial modes only -- temporal modes (ease_in, cycle, ...)
    are computed per-step in envelope_value() and do not need a field.

    Modes:
      noise      -> smooth random field
      saliency   -> peaks where image is salient (requires saliency input)
    """
    if mode == "noise":
        from scipy.ndimage import gaussian_filter
        rng = np.random.default_rng(int(seed))
        field = gaussian_filter(rng.standard_normal((H, W)).astype(np.float32), sigma)
        field = (field - field.min()) / max(field.max() - field.min(), 1e-6)
        return np.clip(field, 0, 1)
    if mode == "saliency" and saliency is not None:
        sal = saliency / (saliency.max() + 1e-6)
        return np.clip(sal, 0, 1).astype(np.float32)
    # any other mode -> no spatial field needed
    return np.ones((H, W), dtype=np.float32)


# Temporal envelope shapes -- map step progress t in [0,1] to [0,1].
_ENVELOPE_TEMPORAL_MODES = {
    "ramp",
    "ease_in", "ease_out", "ease_in_out",
    "cycle", "pulse",
}


def envelope_value(step_idx, n_steps, mode, spatial_field=None,
                   px=0.0, py=0.0, W=1, H=1):
    """Unified envelope sample: returns a scalar in [0,1] for the given step.

    Spatial modes (noise, saliency) sample the precomputed `spatial_field` at
    (px, py). Temporal modes are pure functions of t = step_idx / (n_steps-1).
    """
    if mode == "none":
        return 1.0
    if mode in ("noise", "saliency"):
        if spatial_field is None:
            return 1.0
        ix = int(np.clip(px, 0, W - 1))
        iy = int(np.clip(py, 0, H - 1))
        return float(spatial_field[iy, ix])
    if mode in _ENVELOPE_TEMPORAL_MODES:
        t = step_idx / max(n_steps - 1, 1)
        if mode == "ramp":
            return float(t)
        if mode == "ease_in":
            return float(t * t)
        if mode == "ease_out":
            return float(1.0 - (1.0 - t) ** 2)
        if mode == "ease_in_out":
            return float(t * t * (3.0 - 2.0 * t))   # smoothstep
        if mode == "cycle":
            return float(0.5 + 0.5 * np.sin(2.0 * np.pi * t))
        if mode == "pulse":
            return float(np.sin(np.pi * t))         # peaks at t=0.5
    return 1.0
