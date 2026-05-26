"""Scene-driven layered NPR compositor for 3DGS scenes.

Run:
  SCENE_FILE=scenes/siyun_walks.json python3 render_layers.py
  SCENE_NAME=siyun python3 render_layers.py

Scene files may define:
  composition: {
    out: "images/siyun_composite.png",
    background: [0.97, 0.96, 0.93],
    layers: [
      { name: "base", type: "base_splat", enabled: true, point_pct: 0.25, alpha: 0.70 },
      { name: "walks", type: "surface_walks", enabled: true, params: {...} }
    ]
  }
"""
import os
import time
import json
from collections import OrderedDict
from copy import deepcopy

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from PIL import Image

from gsplat import (
    load_3dgs_ply, decode_3dgs, make_camera, project_perspective, cull, splat,
    compute_gaussian_normals,
)
from npr_utils import (
    PAPER_COLOR, add_line, add_splat_stroke, compute_saliency, load_ref_image,
    make_noise_direction_field, stamp, walk_step_3d, make_envelope_field,
    envelope_value,
)


# ----------------------------------------------------------------------
# Debug-capture hooks for the tuner. When their flags are True, the walks
# layer records into the matching lists so the tuner can overlay on the
# preview. Defaults off; the tuner toggles and reads these directly.
#
#   DEBUG_POINTS    : surface points the walker actually stepped on -- the
#                     curve nodes. This is the "show points" overlay and is
#                     the right answer to "which points become the curves".
#   DEBUG_OFFSETS   : (p1_orig, p1_offset) pairs for normal-offset segments.
# ----------------------------------------------------------------------
DEBUG_CAPTURE_POINTS = False
DEBUG_POINTS = []
_DEBUG_POINTS_CAP = 50000

DEBUG_CAPTURE_OFFSETS = False
DEBUG_OFFSETS = []
_DEBUG_OFFSETS_CAP = 20000
from scene_io import load_scene


DEFAULT_COMPOSITION = {
    "out": None,
    "background": PAPER_COLOR.tolist(),
    "layers": [
        {
            "name": "base",
            "type": "base_splat",
            "enabled": True,
            "point_pct": 0.25,
            "alpha": 0.70,
            "seed": 17,
        },
        {
            "name": "walks",
            "type": "surface_walks",
            "enabled": True,
            "alpha": 1.0,
        },
    ],
}

DEFAULT_WALKS = {
    "N_WALKERS": 500,
    "STEPS": 35,
    "STEP_RADIUS_PX": 14.0,
    "FORWARD_BIAS": 8.0,
    "DIRECTION_MODE": "global",
    "GLOBAL_DIR_DEG": 90.0,
    "NOISE_SCALE": 90.0,
    "STROKE_ALPHA": 0.65,
    "INK_DARKEN": 0.0,
    "INK_R": 0.0,             # custom ink colour (mixed with the splat-reference
    "INK_G": 0.0,             # colour via INK_DARKEN: 0=pure ink_color, 1=pure ref)
    "INK_B": 0.0,
    "PLACEMENT": "saliency",
    "STROKE_MODE": "line",
    "STROKE_WIDTH": 1.0,
    "SPLAT_SCALE": 0.35,
    "SPLAT_ALPHA_SCALE": 0.35,
    "SPLAT_MIN_SIGMA": 0.10,
    "SPLAT_MAX_SIGMA": 1.20,
    "N_STAMPS": 5,
    "SEED": 17,
    "NORMAL_OFFSET_SCALE": 0.0,
    "OFFSET_ENVELOPE_MODE": "none",
    "OFFSET_ENVELOPE_SCALE": 50.0,
}

WALK_KEY_MAP = {
    "n_walkers": "N_WALKERS",
    "steps": "STEPS",
    "step_radius_px": "STEP_RADIUS_PX",
    "forward_bias": "FORWARD_BIAS",
    "direction_mode": "DIRECTION_MODE",
    "global_dir_deg": "GLOBAL_DIR_DEG",
    "noise_scale": "NOISE_SCALE",
    "stroke_alpha": "STROKE_ALPHA",
    "ink_darken": "INK_DARKEN",
    "placement": "PLACEMENT",
    "stroke_mode": "STROKE_MODE",
    "stroke_width": "STROKE_WIDTH",
    "splat_scale": "SPLAT_SCALE",
    "splat_alpha_scale": "SPLAT_ALPHA_SCALE",
    "splat_min_sigma": "SPLAT_MIN_SIGMA",
    "splat_max_sigma": "SPLAT_MAX_SIGMA",
    "n_stamps": "N_STAMPS",
    "seed": "SEED",
    "ink_r": "INK_R",
    "ink_g": "INK_G",
    "ink_b": "INK_B",
    "normal_offset_scale": "NORMAL_OFFSET_SCALE",
    "offset_envelope_mode": "OFFSET_ENVELOPE_MODE",
    "offset_envelope_scale": "OFFSET_ENVELOPE_SCALE",
}


# ----------------------------------------------------------------------
# Per-layer-type UI schema.
#
# tune_layers.py walks this to build editable fields. Each entry is:
#   (json_key, ui_label, type, fmt, choices_or_None)
# `params_in` says where the type-specific params live in the layer dict:
#   None        -> at the layer's top level (e.g. base_splat: layer["point_pct"])
#   "params"    -> inside layer["params"] (e.g. surface_walks: layer["params"]["n_walkers"])
# `enabled` and `alpha` are handled by the tuner separately (every layer has them)
# and are NOT listed below.
#
# MASK3D_FIELDS are appended to every type. Unlike a screen-space mask, this
# one is a 3D distance ball: per-gaussian factor = smoothstep on distance from
# an anchor xyz, applied at RENDER time (per-gaussian opacity / placement
# weight). Moves with the head when the camera rotates. The vec3 schema type
# expands "mask3d" into three sub-fields mask3d_x / mask3d_y / mask3d_z.
# All mask3d_* keys live at the layer's top level regardless of params_in.
# ----------------------------------------------------------------------
MASK3D_FIELDS = [
    ("mask3d_enabled", "mask 3d", "bool",  "",       None),
    ("mask3d",         "anchor",  "vec3",  "{:.2f}", None),
    ("mask3d_r_in",    "r inner", "float", "{:.2f}", None),
    ("mask3d_r_out",   "r outer", "float", "{:.2f}", None),
    ("mask3d_invert",  "invert",  "bool",  "",       None),
]
_MASK3D_KEYS = {"mask3d_enabled", "mask3d_x", "mask3d_y", "mask3d_z",
                "mask3d_r_in", "mask3d_r_out", "mask3d_invert"}

LAYER_PARAM_SCHEMAS = {
    "base_splat": {
        "params_in": None,
        "fields": [
            ("point_pct",  "points",     "float", "{:.2f}", None),
            ("seed",       "seed",       "int",   "{:.0f}", None),
            # "rgb" expands to <key>_r/<key>_g/<key>_b in a single grouped row
            ("bg",         "color",      "rgb",   "{:.2f}", None),
            ("saturation", "saturation", "float", "{:.2f}", None),
            # 0 = uniform/depth random sampling (current behaviour);
            # 1 = sample fully weighted by per-gaussian curvature.
            ("curvature_weight", "curvature", "float", "{:.2f}", None),
            # Depth-of-field on the splat: focus picks a depth percentile
            # (0 = closest visible, 1 = farthest), blur scales per-gaussian
            # cov2d + fades opacity by distance from that focus depth.
            ("depth_focus",      "depth focus", "float", "{:.2f}", None),
            ("depth_blur",       "depth blur",  "float", "{:.2f}", None),
        ] + MASK3D_FIELDS,
    },
    "surface_walks": {
        "params_in": "params",
        "fields": [
            ("n_walkers",         "N walkers",     "int",   "{:.0f}", None),
            ("steps",             "steps/walker",  "int",   "{:.0f}", None),
            ("step_radius_px",    "step radius",   "float", "{:.0f}", None),
            ("forward_bias",      "fwd bias",      "float", "{:.1f}", None),
            ("direction_mode",    "direction",     "str",   "{:s}",   ["momentum", "global", "noise"]),
            ("global_dir_deg",    "global deg",    "float", "{:.0f}", None),
            ("noise_scale",       "noise scale",   "float", "{:.0f}", None),
            ("stroke_alpha",      "stroke alpha",  "float", "{:.2f}", None),
            ("ink_darken",        "ref blend",     "float", "{:.2f}", None),
            ("ink",               "ink color",     "rgb",   "{:.2f}", None),
            ("placement",         "placement",     "str",   "{:s}",   ["saliency", "uniform", "curvature"]),
            ("stroke_mode",       "stroke mode",   "str",   "{:s}",   ["line", "splat"]),
            ("stroke_width",      "line width",    "float", "{:.1f}", None),
            ("splat_scale",       "splat scale",   "float", "{:.2f}", None),
            ("splat_alpha_scale", "splat a-scale", "float", "{:.2f}", None),
            ("splat_min_sigma",   "splat sig min", "float", "{:.2f}", None),
            ("splat_max_sigma",   "splat sig max", "float", "{:.2f}", None),
            ("n_stamps",          "stamps/seg",    "int",   "{:.0f}", None),
            ("normal_offset_scale", "normal offset", "float", "{:.2f}", None),
            ("offset_envelope_mode", "offset env mode", "str", "{:s}",
                ["none", "noise", "saliency",
                 "ramp", "ease_in", "ease_out", "ease_in_out", "cycle", "pulse"]),
            ("offset_envelope_scale", "env scale", "float", "{:.0f}", None),
            ("seed",              "seed",          "int",   "{:.0f}", None),
        ] + MASK3D_FIELDS,
    },
    "generative_curve": {
        "params_in": "params",
        "fields": [
            ("shape",             "shape",         "str",   "{:s}",   ["sphere", "ring", "random_walk", "lorenz"]),
            ("n_points",          "points",        "int",   "{:.0f}", None),
            ("radius",            "radius",        "float", "{:.2f}", None),
            ("center_offset",     "center",        "vec3",  "{:.2f}", None),
            ("seed",              "seed",          "int",   "{:.0f}", None),
            ("stroke_alpha",      "stroke alpha",  "float", "{:.2f}", None),
            ("stroke_mode",       "stroke mode",   "str",   "{:s}",   ["line", "splat"]),
            ("stroke_width",      "line width",    "float", "{:.1f}", None),
            ("splat_scale",       "splat scale",   "float", "{:.2f}", None),
            ("splat_alpha_scale", "splat a-scale", "float", "{:.2f}", None),
            ("splat_min_sigma",   "splat sig min", "float", "{:.2f}", None),
            ("splat_max_sigma",   "splat sig max", "float", "{:.2f}", None),
            ("n_stamps",          "stamps/seg",    "int",   "{:.0f}", None),
            ("color_mode",        "color mode",    "str",   "{:s}",   ["fixed", "ref", "depth"]),
            ("color",             "color",         "rgb",   "{:.2f}", None),
            ("depth_focus",       "depth focus",   "float", "{:.2f}", None),
            ("depth_blur",        "depth blur",    "float", "{:.2f}", None),
            ("line_jitter",       "jitter",        "float", "{:.2f}", None),
            ("connect_closest",   "closest conn",  "bool",  "",       None),
        ] + MASK3D_FIELDS,
    },
    "zline": {
        "params_in": "params",
        "fields": [
            ("p1",                "P1",            "vec3",  "{:.2f}", None),
            ("p2",                "P2",            "vec3",  "{:.2f}", None),
            ("show_endpoints",    "show ends",     "bool",  "",       None),
            ("n_lines",           "N lines",       "int",   "{:.0f}", None),
            ("recursion",         "recursion",     "int",   "{:.0f}", None),
            ("displacement",      "displacement",  "float", "{:.2f}", None),
            ("displacement_decay","decay",         "float", "{:.2f}", None),
            ("neighborhood_range","range",         "float", "{:.2f}", None),
            ("seed",              "seed",          "int",   "{:.0f}", None),
            ("stroke_alpha",      "stroke alpha",  "float", "{:.2f}", None),
            ("stroke_mode",       "stroke mode",   "str",   "{:s}",   ["line", "splat"]),
            ("stroke_width",      "line width",    "float", "{:.1f}", None),
            ("splat_scale",       "splat scale",   "float", "{:.2f}", None),
            ("splat_alpha_scale", "splat a-scale", "float", "{:.2f}", None),
            ("splat_min_sigma",   "splat sig min", "float", "{:.2f}", None),
            ("splat_max_sigma",   "splat sig max", "float", "{:.2f}", None),
            ("n_stamps",          "stamps/seg",    "int",   "{:.0f}", None),
            ("color_mode",        "color mode",    "str",   "{:s}",   ["fixed", "ref"]),
            ("color",             "color",         "rgb",   "{:.2f}", None),
            ("line_jitter",       "jitter",        "float", "{:.2f}", None),
        ] + MASK3D_FIELDS,
    },
}


def layer_param_effective(layer, key, scene_walks=None):
    """Read a layer's effective param value, walking the schema's section
    location, then scene-level defaults, then DEFAULT_WALKS."""
    schema = LAYER_PARAM_SCHEMAS.get(layer.get("type"), {})
    section = schema.get("params_in")
    if section and section in layer and key in layer[section]:
        return layer[section][key]
    if key in layer:
        return layer[key]
    if layer.get("type") == "surface_walks":
        if scene_walks and key in scene_walks:
            return scene_walks[key]
        upper = WALK_KEY_MAP.get(key, key.upper())
        if upper in DEFAULT_WALKS:
            return DEFAULT_WALKS[upper]
    return None


def layer_param_set(layer, key, value):
    """Write a value to the layer's natural location for this key."""
    schema = LAYER_PARAM_SCHEMAS.get(layer.get("type"), {})
    section = schema.get("params_in")
    # mask3d_* always lives at the layer's top level regardless of params_in
    if section and key not in _MASK3D_KEYS:
        layer.setdefault(section, {})[key] = value
    else:
        layer[key] = value


def _walk_params(*sources):
    params = DEFAULT_WALKS.copy()
    for source in sources:
        if not source:
            continue
        for raw_key, value in source.items():
            key = WALK_KEY_MAP.get(raw_key, raw_key.upper())
            if key in params:
                params[key] = value
    return params


def _project_scene(cfg, G):
    ysign = +1.0 if cfg.SCENE_UP_FLIP else -1.0
    center = np.median(G["xyz"], axis=0)
    center[0] += cfg.HEAD_BIAS_X
    center[1] += cfg.HEAD_BIAS_Y
    radii = np.linalg.norm(G["xyz"] - center, axis=1)
    extent = np.percentile(radii, 90) * 2.0
    Rcam = make_camera(cfg.ELEV_DEG, cfg.AZIM_DEG)
    cam_xyz = (G["xyz"] - center) @ Rcam.T
    cov_cam = np.einsum("ij,njk,lk->nil", Rcam, G["cov3"], Rcam)
    focal = cfg.W / (2.0 * np.tan(np.radians(cfg.FOV_DEG) / 2.0))
    distance = extent * cfg.DISTANCE_K
    mean2d, cov2d, depths, valid_z = project_perspective(
        cam_xyz, cov_cam, focal, distance, cfg.W, cfg.H, ysign)
    keep = cull(mean2d, cov2d, G["opacities"], valid_z, cfg.W, cfg.H,
                sub_pixel=0.0)
    # Camera params the tuner needs to reproject arbitrary 3D anchor points.
    camera = dict(center=center, Rcam=Rcam, focal=focal,
                  distance=distance, ysign=ysign)
    return mean2d, cov2d, depths, keep, camera


def reproject_scene(scene_data):
    """Re-run the camera-dependent projection step. Mutates scene_data.
    Call after cfg.ELEV_DEG / AZIM_DEG / FOV_DEG / DISTANCE_K change.
    Also recomputes ref_img and saliency so that surface_walks layers
    sample colours from the current viewpoint rather than a stale one.
    """
    cfg = scene_data["cfg"]
    G = scene_data["G"]
    print(f"[reproject_scene] cfg values:  "
          f"ELEV_DEG={cfg.ELEV_DEG:g}  AZIM_DEG={cfg.AZIM_DEG:g}  "
          f"FOV_DEG={cfg.FOV_DEG:g}  DISTANCE_K={cfg.DISTANCE_K:g}")
    mean2d, cov2d, depths, keep, camera = _project_scene(cfg, G)
    scene_data["mean2d"] = mean2d
    scene_data["cov2d"] = cov2d
    scene_data["depths"] = depths
    scene_data["keep"] = keep
    scene_data["camera"] = camera

    # Recompute reference image + saliency for the new viewpoint so that
    # surface_walks layers pick up correct colours after camera edits.
    order = keep[np.argsort(-depths[keep])]
    ref_img = splat(cfg.W, cfg.H, mean2d, cov2d, G["colors"], G["opacities"],
                    order, verbose=False)
    scene_data["ref_img"] = ref_img
    scene_data["saliency"] = compute_saliency(ref_img)

    import hashlib
    m_hash = hashlib.md5(mean2d.tobytes()).hexdigest()[:8]
    print(f"[reproject_scene] new mean2d: shape={mean2d.shape}  hash={m_hash}  "
          f"x∈[{float(mean2d[:,0].min()):+.1f}, {float(mean2d[:,0].max()):+.1f}]  "
          f"y∈[{float(mean2d[:,1].min()):+.1f}, {float(mean2d[:,1].max()):+.1f}]  "
          f"keep={len(keep)}")


def project_anchor(camera, cfg, xyz):
    """Project a world-space point through the same camera the scene uses.
    Returns (x_pixel, y_pixel, z_view) where z_view > 0 means in front of
    the camera. Use the inverse of `keep` mapping if you need to know if
    the projected pixel lands inside the canvas.
    """
    p = np.asarray(xyz, dtype=np.float64).reshape(3)
    cam_p = (p - camera["center"]) @ camera["Rcam"].T
    z = cam_p[2] + camera["distance"]
    if z <= 1e-6:
        return None
    x_px = cfg.W / 2.0 + camera["focal"] * cam_p[0] / z
    y_px = cfg.H / 2.0 + camera["ysign"] * camera["focal"] * cam_p[1] / z
    return float(x_px), float(y_px), float(z)


def _over(canvas, color, alpha, layer_alpha=1.0):
    a = np.clip(alpha * float(layer_alpha), 0.0, 1.0)[..., None]
    return canvas * (1.0 - a) + color * a


def _mask3d_factors(xyz_subset, spec):
    """Per-gaussian 3D-distance mask factor in [0, 1].

    Returns None when mask3d_enabled is False. Otherwise returns a (N,) array
    aligned with `xyz_subset` (typically G["xyz"][order] or G["xyz"][keep]).
    Geometry: smoothstep falloff between r_in and r_out from `(mask3d_x,
    mask3d_y, mask3d_z)`. mask3d_invert flips inside/outside.
    """
    if not spec.get("mask3d_enabled", False):
        return None
    cx = float(spec.get("mask3d_x", 0.0))
    cy = float(spec.get("mask3d_y", 0.0))
    cz = float(spec.get("mask3d_z", 0.0))
    r_in = max(float(spec.get("mask3d_r_in", 0.3)), 1e-6)
    r_out = max(float(spec.get("mask3d_r_out", 1.0)), r_in + 1e-6)
    invert = bool(spec.get("mask3d_invert", False))

    anchor = np.array([cx, cy, cz], dtype=np.float64)
    dists = np.linalg.norm(xyz_subset - anchor, axis=1)
    t = np.clip((dists - r_in) / (r_out - r_in), 0.0, 1.0)
    factor = 1.0 - t * t * (3.0 - 2.0 * t)          # smoothstep, 1 inside -> 0 outside
    if invert:
        factor = 1.0 - factor
    return factor.astype(np.float64)


def _sample_order(order, pct, seed, weights=None):
    pct = float(pct)
    if pct >= 1.0:
        return order
    if pct <= 0.0:
        return order[:0]
    rng = np.random.default_rng(int(seed))
    n = max(1, int(round(len(order) * pct)))
    if weights is not None:
        w = np.asarray(weights, dtype=np.float64) + 1e-6
        w = w / w.sum()
        pick = rng.choice(len(order), size=n, replace=False, p=w)
    else:
        pick = rng.choice(len(order), size=n, replace=False)
    pick.sort()           # keep depth ordering inside the picked subset
    return order[pick]


def _resolve_bg(spec):
    """Resolve the splat background colour from a layer spec.

    Priority:
      1. explicit bg_r/bg_g/bg_b fields (preferred, tuner-editable)
      2. legacy `bg` or `background` list
      3. black
    """
    has_individual = any(k in spec for k in ("bg_r", "bg_g", "bg_b"))
    if has_individual:
        return [float(spec.get("bg_r", 0.0)),
                float(spec.get("bg_g", 0.0)),
                float(spec.get("bg_b", 0.0))]
    bg = spec.get("bg", spec.get("background"))
    if bg is not None:
        return [float(c) for c in list(bg)[:3]]
    return [0.0, 0.0, 0.0]


def _apply_saturation(img, saturation):
    """Linear saturation around luma. 0 = grayscale, 1 = unchanged, >1 = vivid."""
    if abs(saturation - 1.0) < 1e-4:
        return img
    luma = img @ np.array([0.299, 0.587, 0.114])
    out = luma[..., None] + saturation * (img - luma[..., None])
    return np.clip(out, 0.0, 1.0)


def layer_base_splat(cfg, G, mean2d, cov2d, depths, keep, spec):
    seed = int(spec.get("seed", 17))
    point_pct = float(spec.get("point_pct", spec.get("pct", 1.0)))
    bg = _resolve_bg(spec)
    saturation = float(spec.get("saturation", 1.0))
    curvature_weight = float(spec.get("curvature_weight", 0.0))
    depth_focus = float(spec.get("depth_focus", 0.5))
    depth_blur = float(spec.get("depth_blur", 0.0))

    order = keep[np.argsort(-depths[keep])]

    weights = None
    curvature_full = G.get("curvature")
    if curvature_weight > 0.0 and curvature_full is not None:
        c = curvature_full[order]
        c_norm = c / (c.max() + 1e-9)
        weights = (1.0 - curvature_weight) + curvature_weight * c_norm

    order = _sample_order(order, point_pct, seed, weights=weights)

    # Per-gaussian opacity modulation: 3D mask (if enabled) and depth-of-field
    # fade. Done by COPYING G["opacities"] and (when blurring) cov2d, then
    # scaling only the indices we're about to render. Memory cost is bounded
    # by len(order) but easiest to copy the full arrays once.
    ops_eff = G["opacities"]
    cov2d_eff = cov2d
    tags = []

    mask3d = _mask3d_factors(G["xyz"][order], spec)
    if mask3d is not None:
        ops_eff = ops_eff.copy()
        ops_eff[order] = ops_eff[order] * mask3d
        tags.append("mask3d")

    if depth_blur > 0.0:
        vis_depths = depths[order]
        d_lo, d_hi = float(vis_depths.min()), float(vis_depths.max())
        focus_d = d_lo + np.clip(depth_focus, 0.0, 1.0) * (d_hi - d_lo)
        drange = max(d_hi - d_lo, 1e-6)
        dist = np.abs(vis_depths - focus_d) / drange     # in [0, 1]
        blur_f = depth_blur * dist                        # 0 in focus, larger off-focus
        # Variance scales by (1 + f)^2 -> sigma scales by (1 + f).
        sigma_scale = (1.0 + 3.0 * blur_f)                # 3x amplifies the look
        cov2d_eff = cov2d_eff.copy()
        cov2d_eff[order] = cov2d_eff[order] * (sigma_scale[:, None, None] ** 2)
        # And fade alpha: out-of-focus contributes less per stamp.
        alpha_fade = np.clip(1.0 - 0.7 * blur_f, 0.05, 1.0)
        if ops_eff is G["opacities"]:
            ops_eff = ops_eff.copy()
        ops_eff[order] = ops_eff[order] * alpha_fade
        tags.append(f"dof(f={depth_focus:.2f},b={depth_blur:.2f})")

    cw_tag = f"  cw={curvature_weight:.2f}" if weights is not None else ""
    extra = ("  [" + ", ".join(tags) + "]") if tags else ""
    print(f"  base_splat: {len(order)} / {len(keep)} visible points "
          f"({point_pct:.2%})  bg={[f'{c:.2f}' for c in bg]}  "
          f"sat={saturation:.2f}{cw_tag}{extra}")
    img = splat(cfg.W, cfg.H, mean2d, cov2d_eff, G["colors"], ops_eff,
                order, bg=bg, verbose=False)
    img = _apply_saturation(img, saturation)
    alpha = np.ones((cfg.H, cfg.W), dtype=np.float64)
    return img, alpha


def _placement_weights(pts, ops, saliency, placement, curvature=None):
    if placement == "saliency":
        ix = np.clip(pts[:, 0].astype(int), 0, saliency.shape[1] - 1)
        iy = np.clip(pts[:, 1].astype(int), 0, saliency.shape[0] - 1)
        return ops * (saliency[iy, ix] + 0.04)
    if placement == "curvature" and curvature is not None:
        # Small epsilon so flat regions still have some chance of being seeded.
        return ops * (curvature + 0.04)
    return ops


def _seg_color(p0, p1, ref_img, ink_color, ink_blend, W, H):
    """Stroke colour at the midpoint of a segment.

    Blends a tunable ink colour with the splat-reference colour:
      0.0 -> pure ink_color (e.g. black or a tint)
      1.0 -> pure reference colour from the splat render
    Backward compatible: with ink_color = (0,0,0) and ink_blend = 0 the
    stroke is black, matching the old `ref * ink_darken` behaviour.
    """
    mx = int(np.clip((p0[0] + p1[0]) / 2, 0, W - 1))
    my = int(np.clip((p0[1] + p1[1]) / 2, 0, H - 1))
    ref = ref_img[my, mx]
    return ink_color * (1.0 - ink_blend) + ref * ink_blend


def _apply_normal_offset(p1_screen, normal_3d, scale, envelope_value=1.0, cfg=None):
    """Apply a normal-based offset in screen space with envelope modulation.
    
    Simple approach: displace the screen position perpendicular to surface
    using a heuristic based on the normal direction. The offset is modulated
    by an envelope value (typically 0-1 from an envelope field).
    """
    if scale == 0:
        return p1_screen
    
    modulated_scale = scale * envelope_value
    normal_2d = normal_3d[:2] * modulated_scale
    return p1_screen + normal_2d


def layer_surface_walks(cfg, G, mean2d, cov2d, keep, ref_img, saliency, spec, bg=PAPER_COLOR):
    params = _walk_params(cfg.SURFACE_WALKS, spec.get("params", {}), spec)
    rng = np.random.default_rng(int(params["SEED"]))
    pts = mean2d[keep]
    xyz = G["xyz"][keep]
    ops = G["opacities"][keep]
    cov2d_kept = cov2d[keep]
    normals = G["normals"][keep]
    curvature_full = G.get("curvature")
    curvature_kept = curvature_full[keep] if curvature_full is not None else None
    tree3d = cKDTree(xyz)
    weights = _placement_weights(pts, ops, saliency, params["PLACEMENT"],
                                  curvature=curvature_kept)
    # 3D-ball mask multiplies placement weights so walkers seed (and mostly
    # stay) inside the ball. Epsilon keeps a small chance outside.
    mask3d_walks = _mask3d_factors(xyz, spec)
    if mask3d_walks is not None:
        weights = weights * (mask3d_walks + 0.04)
    weights = weights / weights.sum()

    direction_field = None
    global_dir = None
    if params["DIRECTION_MODE"] == "global":
        theta = np.radians(float(params["GLOBAL_DIR_DEG"]))
        global_dir = np.array([np.cos(theta), np.sin(theta)], dtype=np.float64)
    elif params["DIRECTION_MODE"] == "noise":
        direction_field = make_noise_direction_field(
            cfg.W, cfg.H, sigma=float(params["NOISE_SCALE"]),
            seed=int(params["SEED"]))

    normal_offset_scale = float(params.get("NORMAL_OFFSET_SCALE", 0.0))
    offset_envelope_mode = params.get("OFFSET_ENVELOPE_MODE", "none")
    offset_envelope_scale = float(params.get("OFFSET_ENVELOPE_SCALE", 50.0))
    
    # Only build a spatial field for spatial envelope modes; temporal modes
    # (ease_*, cycle, pulse, ramp) are computed per-step from step_idx.
    envelope_field = None
    if normal_offset_scale != 0.0 and offset_envelope_mode in ("noise", "saliency"):
        envelope_field = make_envelope_field(
            cfg.W, cfg.H, mode=offset_envelope_mode, sigma=offset_envelope_scale,
            saliency=saliency if offset_envelope_mode == "saliency" else None,
            seed=int(params["SEED"]))

    ink_color = np.array([
        float(params.get("INK_R", 0.0)),
        float(params.get("INK_G", 0.0)),
        float(params.get("INK_B", 0.0)),
    ], dtype=np.float64)
    ink_blend = float(params["INK_DARKEN"])
    bg = np.asarray(bg, dtype=np.float64)
    line_canvas = np.tile(bg, (cfg.H, cfg.W, 1)).astype(np.float64)
    drawn = 0
    for _ in range(int(params["N_WALKERS"])):
        cur = int(rng.choice(len(pts), p=weights))
        prev_dir = None
        # Track the *drawn* position of the previous step so each segment
        # picks up where the last one left off. Without this, consecutive
        # segments alternate surface->offset->surface->offset and the curve
        # zig-zags instead of staying on the offset path.
        prev_p = np.asarray(pts[cur], dtype=np.float64).copy()
        for step_idx in range(int(params["STEPS"])):
            result = walk_step_3d(
                cur, prev_dir, tree3d, xyz, pts,
                float(params["STEP_RADIUS_PX"]), float(params["FORWARD_BIAS"]),
                rng, direction_field=direction_field, global_dir=global_dir)
            if result is None:
                break
            nxt, prev_dir = result
            p1_raw = pts[nxt]
            p1 = np.asarray(p1_raw, dtype=np.float64).copy()

            if DEBUG_CAPTURE_POINTS and len(DEBUG_POINTS) < _DEBUG_POINTS_CAP:
                DEBUG_POINTS.append(np.asarray(p1_raw, dtype=np.float64).copy())

            if normal_offset_scale != 0.0:
                normal_3d = normals[nxt]
                env_val = envelope_value(
                    step_idx, int(params["STEPS"]), offset_envelope_mode,
                    spatial_field=envelope_field,
                    px=p1[0], py=p1[1], W=cfg.W, H=cfg.H)
                p1 = _apply_normal_offset(p1, normal_3d, normal_offset_scale,
                                         envelope_value=env_val, cfg=cfg)
                if DEBUG_CAPTURE_OFFSETS and len(DEBUG_OFFSETS) < _DEBUG_OFFSETS_CAP:
                    DEBUG_OFFSETS.append((np.asarray(p1_raw, dtype=np.float64).copy(),
                                          p1.copy()))

            # Draw from the *drawn* endpoint of the previous step (already
            # offset, when applicable) to this step's drawn endpoint.
            p0 = prev_p
            
            color = _seg_color(p0, p1, ref_img, ink_color, ink_blend,
                               cfg.W, cfg.H)
            if params["STROKE_MODE"] == "splat":
                cov_avg = 0.5 * (cov2d_kept[cur] + cov2d_kept[nxt])
                add_splat_stroke(
                    line_canvas, p0, p1, cov_avg, color,
                    float(params["STROKE_ALPHA"]) * float(params["SPLAT_ALPHA_SCALE"]),
                    cfg.W, cfg.H, n_stamps=int(params["N_STAMPS"]),
                    scale=float(params["SPLAT_SCALE"]),
                    min_sigma_px=float(params["SPLAT_MIN_SIGMA"]),
                    max_sigma_px=float(params["SPLAT_MAX_SIGMA"]))
            else:
                add_line(line_canvas, p0[0], p0[1], p1[0], p1[1], color,
                         float(params["STROKE_ALPHA"]), cfg.W, cfg.H,
                         width=float(params["STROKE_WIDTH"]))
            prev_p = p1   # carry this step's drawn endpoint into the next
            cur = nxt
            drawn += 1

    # Alpha extraction: how much did we deviate from the shared background?
    # Use a soft threshold on Euclidean distance in RGB space.
    diff = np.linalg.norm(line_canvas - bg, axis=-1)
    alpha = np.clip((diff - 0.01) / 0.08, 0.0, 1.0)
    print(f"  surface_walks: {drawn} segments")
    return line_canvas, alpha


# ----------------------------------------------------------------------
# Generative curve layer
# ----------------------------------------------------------------------

def _generate_curve_points(shape, n_points, radius, center, seed, **kwargs):
    """Generate a 3D point cloud for the generative-curve layer."""
    rng = np.random.default_rng(int(seed))
    center = np.asarray(center, dtype=np.float64)

    if shape == "sphere":
        # Uniform inside a sphere
        u = rng.random(n_points)
        r = radius * np.cbrt(u)
        theta = np.arccos(2 * rng.random(n_points) - 1)
        phi = 2 * np.pi * rng.random(n_points)
        pts = np.column_stack([
            r * np.sin(theta) * np.cos(phi),
            r * np.sin(theta) * np.sin(phi),
            r * np.cos(theta),
        ])

    elif shape == "ring":
        # Tilted ring / torus
        tilt_x = float(kwargs.get("tilt_x", 0.0))
        tilt_y = float(kwargs.get("tilt_y", 0.0))
        t = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
        # Add slight thickness
        tube_r = radius * 0.15
        tube_u = rng.random(n_points)
        tube_theta = 2 * np.pi * rng.random(n_points)
        r_eff = radius + tube_r * np.cos(tube_theta)
        z_off = tube_r * np.sin(tube_theta)
        pts = np.column_stack([
            r_eff * np.cos(t),
            r_eff * np.sin(t),
            z_off,
        ])
        # Apply tilt
        if tilt_x != 0:
            cx, sx = np.cos(np.radians(tilt_x)), np.sin(np.radians(tilt_x))
            Rxt = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
            pts = pts @ Rxt.T
        if tilt_y != 0:
            cy, sy = np.cos(np.radians(tilt_y)), np.sin(np.radians(tilt_y))
            Ryt = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
            pts = pts @ Ryt.T

    elif shape == "random_walk":
        # 3D random walk with momentum
        steps = n_points - 1
        momentum = float(kwargs.get("momentum", 0.7))
        pts = np.zeros((n_points, 3), dtype=np.float64)
        dir_vec = rng.normal(size=3)
        dir_vec /= np.linalg.norm(dir_vec) + 1e-9
        step_len = radius * 0.05
        for i in range(1, n_points):
            new_dir = rng.normal(size=3)
            new_dir /= np.linalg.norm(new_dir) + 1e-9
            dir_vec = momentum * dir_vec + (1.0 - momentum) * new_dir
            dir_vec /= np.linalg.norm(dir_vec) + 1e-9
            pts[i] = pts[i - 1] + dir_vec * step_len

    elif shape == "lorenz":
        # Lorenz attractor
        sigma, rho, beta = 10.0, 28.0, 8.0 / 3.0
        dt = 0.01
        n_steps = n_points * 10
        xyz = np.zeros((n_steps, 3), dtype=np.float64)
        xyz[0] = rng.normal(size=3) * 0.1
        for i in range(1, n_steps):
            x, y, z = xyz[i - 1]
            dx = sigma * (y - x)
            dy = x * (rho - z) - y
            dz = x * y - beta * z
            xyz[i] = xyz[i - 1] + np.array([dx, dy, dz]) * dt
        # Subsample and scale
        idx = np.linspace(0, n_steps - 1, n_points).astype(int)
        pts = xyz[idx]
        # Normalize to radius
        pts_max = np.abs(pts).max()
        if pts_max > 1e-9:
            pts = pts / pts_max * radius

    else:
        raise ValueError(f"unknown generative curve shape: {shape}")

    return pts + center


def _project_curve_points(pts_3d, camera, W, H):
    """Project world-space points to 2D using the scene camera.
    Returns (mean2d, depths, valid).
    """
    cam_xyz = (pts_3d - camera["center"]) @ camera["Rcam"].T
    cam_xyz[:, 2] += camera["distance"]
    Z = cam_xyz[:, 2]
    valid = Z > 1e-3
    safe_Z = np.where(valid, Z, 1.0)
    invZ = 1.0 / safe_Z
    mean2d = np.column_stack([
        W / 2 + camera["focal"] * cam_xyz[:, 0] * invZ,
        H / 2 + camera["ysign"] * camera["focal"] * cam_xyz[:, 1] * invZ,
    ])
    return mean2d, Z, valid


def layer_generative_curve(cfg, camera, ref_img, depths_range, spec, bg=PAPER_COLOR):
    """Render a procedurally-generated 3D curve with splats or lines.

    Generates N points in a 3D shape (sphere, ring, random walk, lorenz),
    projects them through the scene camera, and draws the polyline.
    Supports depth-of-field blur, reference-image color sampling, and
    both line and splat stroke modes.
    """
    params = spec.get("params", {})
    shape = params.get("shape", "sphere")
    n_points = int(params.get("n_points", 200))
    radius = float(params.get("radius", 0.5))
    seed = int(params.get("seed", 17))
    stroke_alpha = float(params.get("stroke_alpha", 0.6))
    stroke_mode = params.get("stroke_mode", "splat")
    stroke_width = float(params.get("stroke_width", 1.0))
    splat_scale = float(params.get("splat_scale", 0.35))
    splat_alpha_scale = float(params.get("splat_alpha_scale", 0.35))
    splat_min_sigma = float(params.get("splat_min_sigma", 0.10))
    splat_max_sigma = params.get("splat_max_sigma")
    splat_max_sigma = None if splat_max_sigma is None else float(splat_max_sigma)
    n_stamps = int(params.get("n_stamps", 5))
    color_mode = params.get("color_mode", "fixed")
    depth_focus = float(params.get("depth_focus", 0.5))
    depth_blur = float(params.get("depth_blur", 0.0))
    line_jitter = float(params.get("line_jitter", 0.0))
    connect_closest = bool(params.get("connect_closest", False))

    # Center offset from the layer spec (vec3 expands to cx/cy/cz)
    cx = float(params.get("cx", 0.0))
    cy = float(params.get("cy", 0.0))
    cz = float(params.get("cz", 0.0))
    center_offset = np.array([cx, cy, cz], dtype=np.float64)

    # Generate 3D points around the scene camera centre so the curve
    # appears in front of the rendered scene by default.
    curve_center = camera["center"] + center_offset
    pts_3d = _generate_curve_points(
        shape, n_points, radius, curve_center, seed,
        tilt_x=float(params.get("tilt_x", 0.0)),
        tilt_y=float(params.get("tilt_y", 0.0)),
        momentum=float(params.get("momentum", 0.7)),
    )

    # Project to 2D
    pts_2d, depths, valid = _project_curve_points(pts_3d, camera, cfg.W, cfg.H)

    # Build segment order
    if connect_closest:
        # Greedy nearest-neighbor path (Travelling-Salesman-ish)
        order = [0]
        remain = set(range(1, n_points))
        while remain:
            last = order[-1]
            best = min(remain, key=lambda i: np.sum((pts_2d[i] - pts_2d[last])**2))
            order.append(best)
            remain.remove(best)
    else:
        order = list(range(n_points))

    # Fixed color
    color_fixed = np.array([
        float(params.get("color_r", 0.0)),
        float(params.get("color_g", 0.0)),
        float(params.get("color_b", 0.0)),
    ], dtype=np.float64)

    # Depth range for blur
    if depths_range is not None:
        d_lo, d_hi = depths_range
    else:
        d_lo, d_hi = float(depths[valid].min()), float(depths[valid].max())
    drange = max(d_hi - d_lo, 1e-6)
    focus_d = d_lo + np.clip(depth_focus, 0.0, 1.0) * drange

    bg = np.asarray(bg, dtype=np.float64)
    canvas = np.tile(bg, (cfg.H, cfg.W, 1)).astype(np.float64)
    drawn = 0

    for i in range(len(order) - 1):
        a_idx, b_idx = order[i], order[i + 1]
        if not (valid[a_idx] and valid[b_idx]):
            continue

        p0 = np.asarray(pts_2d[a_idx], dtype=np.float64).copy()
        p1 = np.asarray(pts_2d[b_idx], dtype=np.float64).copy()

        # Line jitter
        if line_jitter > 0:
            jitter = np.random.default_rng(seed + i).normal(scale=line_jitter, size=2)
            p1 += jitter

        # Color
        if color_mode == "ref":
            mx = int(np.clip((p0[0] + p1[0]) / 2, 0, cfg.W - 1))
            my = int(np.clip((p0[1] + p1[1]) / 2, 0, cfg.H - 1))
            seg_color = ref_img[my, mx]
        elif color_mode == "depth":
            z = (depths[a_idx] + depths[b_idx]) / 2.0
            t = np.clip((z - d_lo) / drange, 0.0, 1.0)
            seg_color = np.array([t, 1.0 - t, 0.5], dtype=np.float64)
        else:
            seg_color = color_fixed

        # Depth blur: fade alpha and scale covariance by distance from focus
        seg_alpha = stroke_alpha
        seg_splat_scale = splat_scale
        if depth_blur > 0.0:
            z = (depths[a_idx] + depths[b_idx]) / 2.0
            dist = np.abs(z - focus_d) / drange
            blur_f = depth_blur * dist
            seg_alpha = stroke_alpha * np.clip(1.0 - 0.7 * blur_f, 0.05, 1.0)
            seg_splat_scale = splat_scale * (1.0 + 3.0 * blur_f)

        if stroke_mode == "splat":
            # Synthetic isotropic covariance scaled by depth blur
            seg_len = np.linalg.norm(p1 - p0)
            sigma = max(seg_len * 0.3, 2.0) * seg_splat_scale
            cov = np.array([[sigma, 0.0], [0.0, sigma]], dtype=np.float64)
            add_splat_stroke(
                canvas, p0, p1, cov, seg_color,
                seg_alpha * splat_alpha_scale,
                cfg.W, cfg.H, n_stamps=n_stamps,
                scale=1.0,
                min_sigma_px=splat_min_sigma,
                max_sigma_px=splat_max_sigma,
            )
        else:
            add_line(canvas, p0[0], p0[1], p1[0], p1[1],
                     seg_color, seg_alpha, cfg.W, cfg.H, width=stroke_width)
        drawn += 1

    # Alpha extraction: Euclidean distance from the shared background.
    # A soft threshold gives high alpha wherever the canvas deviates from bg,
    # so strokes remain visible even when this is the only layer.
    diff = np.linalg.norm(canvas - bg, axis=-1)
    alpha = np.clip((diff - 0.01) / 0.06, 0.0, 1.0)
    print(f"  generative_curve: {drawn} segments  shape={shape}")
    return canvas, alpha


def _midpoint_displace_3d(p1, p2, recursion, displacement, decay, rng):
    """Recursively displace midpoints in a random perpendicular direction.

    Returns a polyline as a list of 3-D arrays.
    """
    pts = [np.asarray(p1, dtype=np.float64), np.asarray(p2, dtype=np.float64)]
    for level in range(recursion):
        new_pts = [pts[0]]
        for i in range(len(pts) - 1):
            a = pts[i]
            b = pts[i + 1]
            mid = (a + b) * 0.5
            dx = b - a
            seg_len = np.linalg.norm(dx)
            if seg_len > 1e-9:
                # Random unit vector, then project out the axial component
                rand_dir = rng.normal(size=3)
                rand_dir /= np.linalg.norm(rand_dir) + 1e-9
                dx_unit = dx / seg_len
                perp = rand_dir - np.dot(rand_dir, dx_unit) * dx_unit
                perp_norm = np.linalg.norm(perp)
                if perp_norm > 1e-9:
                    perp /= perp_norm
                    max_disp = seg_len * displacement * (decay ** level)
                    disp = rng.uniform(-max_disp, max_disp)
                    mid += perp * disp
            new_pts.append(mid)
            new_pts.append(b)
        pts = new_pts
    return pts


def layer_zline(cfg, camera, ref_img, spec, bg=PAPER_COLOR):
    """Render a cluster of 3-D recursive midpoint-displacement zigzag lines.

    P1 and P2 are world-space endpoints.  Each line in the cluster is spawned
    in a neighborhood around P1/P2 (controlled by `neighborhood_range`), then
    midpoint-displaced in 3-D and projected through the scene camera.
    """
    params = spec.get("params", {})
    # P1 / P2 are stored as offsets from the scene camera centre so that
    # (0,0,0) always lands on the scene content regardless of world coords.
    p1 = camera["center"] + np.array([
        float(params.get("p1_x", -0.3)),
        float(params.get("p1_y", 0.0)),
        float(params.get("p1_z", 0.0)),
    ], dtype=np.float64)
    p2 = camera["center"] + np.array([
        float(params.get("p2_x", 0.3)),
        float(params.get("p2_y", 0.0)),
        float(params.get("p2_z", 0.0)),
    ], dtype=np.float64)
    show_endpoints = bool(params.get("show_endpoints", False))
    n_lines = int(params.get("n_lines", 5))
    recursion = int(params.get("recursion", 2))
    displacement = float(params.get("displacement", 0.3))
    decay = float(params.get("displacement_decay", 0.5))
    neighborhood_range = float(params.get("neighborhood_range", 0.1))
    seed = int(params.get("seed", 17))
    stroke_alpha = float(params.get("stroke_alpha", 0.7))
    stroke_mode = params.get("stroke_mode", "line")
    stroke_width = float(params.get("stroke_width", 1.0))
    color_mode = params.get("color_mode", "fixed")
    line_jitter = float(params.get("line_jitter", 0.0))

    # Splat params
    splat_scale = float(params.get("splat_scale", 0.35))
    splat_alpha_scale = float(params.get("splat_alpha_scale", 0.35))
    splat_min_sigma = float(params.get("splat_min_sigma", 0.10))
    splat_max_sigma = params.get("splat_max_sigma")
    splat_max_sigma = None if splat_max_sigma is None else float(splat_max_sigma)
    n_stamps = int(params.get("n_stamps", 5))

    # Fixed color
    color_fixed = np.array([
        float(params.get("color_r", 0.0)),
        float(params.get("color_g", 0.0)),
        float(params.get("color_b", 0.0)),
    ], dtype=np.float64)

    W, H = cfg.W, cfg.H
    bg = np.asarray(bg, dtype=np.float64)
    canvas = np.tile(bg, (H, W, 1)).astype(np.float64)

    rng = np.random.default_rng(seed)

    drawn = 0
    for li in range(n_lines):
        line_rng = np.random.default_rng(seed + li + 1)

        # Spawn endpoints in a neighborhood around P1 / P2
        offset1 = line_rng.normal(scale=neighborhood_range, size=3) if neighborhood_range > 0 else np.zeros(3)
        offset2 = line_rng.normal(scale=neighborhood_range, size=3) if neighborhood_range > 0 else np.zeros(3)
        line_p1 = p1 + offset1
        line_p2 = p2 + offset2

        # Generate 3-D polyline via midpoint displacement
        pts_3d = _midpoint_displace_3d(line_p1, line_p2, recursion, displacement, decay, line_rng)
        pts_3d = np.stack(pts_3d, axis=0)

        # Project through scene camera
        pts_2d, depths, valid = _project_curve_points(pts_3d, camera, W, H)

        # Draw projected segments
        for i in range(len(pts_2d) - 1):
            if not (valid[i] and valid[i + 1]):
                continue
            a = np.asarray(pts_2d[i], dtype=np.float64)
            b = np.asarray(pts_2d[i + 1], dtype=np.float64)

            # Line jitter in 2-D (pixels)
            if line_jitter > 0:
                jitter = line_rng.normal(scale=line_jitter, size=2)
                b = b + jitter

            # Color
            if color_mode == "ref":
                mx = int(np.clip((a[0] + b[0]) * 0.5, 0, W - 1))
                my = int(np.clip((a[1] + b[1]) * 0.5, 0, H - 1))
                seg_color = ref_img[my, mx]
            else:
                seg_color = color_fixed

            if stroke_mode == "splat":
                seg_len = np.linalg.norm(b - a)
                sigma = max(seg_len * 0.3, 2.0) * splat_scale
                cov = np.array([[sigma, 0.0], [0.0, sigma]], dtype=np.float64)
                add_splat_stroke(
                    canvas, a, b, cov, seg_color,
                    stroke_alpha * splat_alpha_scale,
                    W, H, n_stamps=n_stamps,
                    scale=1.0,
                    min_sigma_px=splat_min_sigma,
                    max_sigma_px=splat_max_sigma,
                )
            else:
                add_line(canvas, a[0], a[1], b[0], b[1],
                         seg_color, stroke_alpha, W, H, width=stroke_width)
            drawn += 1

    diff = np.linalg.norm(canvas - bg, axis=-1)
    alpha = np.clip((diff - 0.01) / 0.06, 0.0, 1.0)
    print(f"  zline: {n_lines} lines  {drawn} segments  recursion={recursion}")
    return canvas, alpha


_HASH_EXCLUDE = {"enabled", "alpha", "name",
                 "_ui_y", "_ui_h", "_ui_collapsed", "_ui_section"}
# Note: mask3d_*, depth_*, curvature_* all affect the layer's intrinsic render,
# so they DO bust the cache. No exclusion.


def _canonicalize(obj):
    if isinstance(obj, dict):
        return tuple(sorted((k, _canonicalize(v)) for k, v in obj.items()
                              if k not in _HASH_EXCLUDE))
    if isinstance(obj, list):
        return tuple(_canonicalize(x) for x in obj)
    if isinstance(obj, float):
        return round(obj, 6)
    return obj


def layer_content_hash(spec):
    """Hashable signature of a layer's content (excludes the compositor-time
    fields enabled/alpha/name and UI bookkeeping). Two layers with the same
    hash render to the same (color, alpha) buffer, so the result can be
    cached and reused when only enabled/alpha change."""
    return _canonicalize(spec)


def _build_png_metadata(cfg, composition):
    """Build a metadata dict to embed in PNG file.

    Captures everything needed to reproduce the render: scene name, the camera
    that was actually used (live cfg values at render time, NOT the on-disk
    scene JSON which may be stale), and the composition. Stored as a single
    JSON string in a `composition` tEXt chunk so show_scene.py can fetch it.
    """
    metadata = {
        "scene_name": cfg.SCENE_NAME,
        "camera": {
            "elev_deg":    float(getattr(cfg, "ELEV_DEG", 0.0)),
            "azim_deg":    float(getattr(cfg, "AZIM_DEG", 0.0)),
            "fov_deg":     float(getattr(cfg, "FOV_DEG", 28.0)),
            "distance_k":  float(getattr(cfg, "DISTANCE_K", 1.5)),
            "head_bias_x": float(getattr(cfg, "HEAD_BIAS_X", 0.0)),
            "head_bias_y": float(getattr(cfg, "HEAD_BIAS_Y", 0.0)),
        },
        "composition": composition,
    }
    return metadata


def _save_png_with_metadata(filepath, canvas, metadata):
    """Save PNG image with embedded metadata using PIL.
    
    Saves canvas to filepath with scene metadata embedded in PNG tEXt chunks.
    """
    img_array = (np.clip(canvas, 0, 1) * 255).astype(np.uint8)
    pil_img = Image.fromarray(img_array)
    
    if metadata:
        pil_img.info['composition'] = json.dumps(metadata)
    
    pil_img.save(filepath, "PNG", pnginfo=None)
    if metadata:
        from PIL import PngImagePlugin
        info = PngImagePlugin.PngInfo()
        info.add_text("composition", json.dumps(metadata))
        pil_img.save(filepath, "PNG", pnginfo=info)


def _compute_curvature(cfg, G):
    """Compute a per-gaussian curvature scalar in [0, 1] from normal disagreement.

    For each gaussian, take the mean normal of its k nearest neighbours,
    re-normalise, and use `1 - dot(self_normal, mean_neighbour_normal)`.
    Aligned neighbours -> 0 (flat); disagreeing -> high (edges/creases).

    Controlled by `curvature_k` (scene JSON) or `CURVATURE_K` (env var).
    Default 8. Set to <=1 to skip entirely (G['curvature'] is then absent).
    """
    k = int(os.environ.get("CURVATURE_K", cfg._raw.get("curvature_k", 8)))
    if k <= 1:
        return
    t0 = time.time()
    tree = cKDTree(G["xyz"])
    _, knn_idx = tree.query(G["xyz"], k=k)
    mean_n = G["normals"][knn_idx].mean(axis=1)
    mag = np.linalg.norm(mean_n, axis=1, keepdims=True) + 1e-9
    mean_n_unit = mean_n / mag
    dots = np.einsum("ij,ij->i", G["normals"], mean_n_unit)
    G["curvature"] = np.clip(1.0 - dots, 0.0, 1.0).astype(np.float32)
    c = G["curvature"]
    print(f"[layers] curvature over k={k} in {time.time() - t0:.1f}s  "
          f"(min={c.min():.3f} mean={c.mean():.3f} max={c.max():.3f})")


def _simplify_base(cfg, G):
    """Optional one-time base simplification.

    Scene-JSON keys (also overridable via env vars):
      base_density    (float, 0<x<=1)  -- random subsample to this fraction
      base_seed       (int)            -- rng seed for the subsample
      normal_smooth_k (int, >=2)       -- k-NN average gaussian normals
                                         (applied AFTER downsample)

    Both default to "off" so existing scenes are unaffected. Returns G in place.
    """
    raw_n = len(G["xyz"])

    base_density = float(os.environ.get(
        "BASE_DENSITY", cfg._raw.get("base_density", 1.0)))
    normal_smooth_k = int(os.environ.get(
        "NORMAL_SMOOTH_K", cfg._raw.get("normal_smooth_k", 0)))

    if 0.0 < base_density < 1.0:
        keep_n = max(1, int(raw_n * base_density))
        seed = int(os.environ.get("BASE_SEED", cfg._raw.get("base_seed", 17)))
        idx = np.random.default_rng(seed).choice(raw_n, size=keep_n, replace=False)
        idx.sort()
        for k in list(G.keys()):
            arr = G[k]
            if isinstance(arr, np.ndarray) and len(arr) == raw_n:
                G[k] = arr[idx]
        print(f"[layers] base downsample: {raw_n} -> {len(G['xyz'])} "
              f"({base_density:.0%}, seed={seed})")

    if normal_smooth_k > 1:
        t0 = time.time()
        tree = cKDTree(G["xyz"])
        _, knn_idx = tree.query(G["xyz"], k=int(normal_smooth_k))
        smoothed = G["normals"][knn_idx].mean(axis=1)
        mag = np.linalg.norm(smoothed, axis=1, keepdims=True) + 1e-9
        G["normals"] = (smoothed / mag).astype(G["normals"].dtype, copy=False)
        print(f"[layers] smoothed normals over k={normal_smooth_k} "
              f"neighbours in {time.time() - t0:.1f}s")

    return raw_n


def build_scene_data(scene_ref):
    """Pre-compute everything that does not depend on the layer set:
    PLY load, decode, projection, reference image, saliency, normals.
    The tuner builds this ONCE at startup and passes it into render_composition so
    each render only does the per-layer work."""
    cfg = load_scene(scene_ref)
    data = load_3dgs_ply(cfg.PLY)
    G = decode_3dgs(data)
    G["normals"] = compute_gaussian_normals(G["cov3"])
    raw_n = _simplify_base(cfg, G)
    _compute_curvature(cfg, G)
    eff_n = len(G["xyz"])
    suffix = f" (from {raw_n})" if eff_n != raw_n else ""
    bb_min = G["xyz"].min(axis=0)
    bb_max = G["xyz"].max(axis=0)
    bb_ctr = (bb_min + bb_max) / 2.0
    # Per-axis 10-90% quantile bbox -- where most points live (helps spot the
    # head vs sparse outliers).
    dens_lo = np.quantile(G["xyz"], 0.1, axis=0)
    dens_hi = np.quantile(G["xyz"], 0.9, axis=0)
    print(f"[layers] scene={cfg.SCENE_NAME} ply={cfg.PLY} "
          f"gaussians={eff_n}{suffix}")
    print(f"[layers] bbox  x=[{bb_min[0]:+.3f}, {bb_max[0]:+.3f}]  "
          f"y=[{bb_min[1]:+.3f}, {bb_max[1]:+.3f}]  "
          f"z=[{bb_min[2]:+.3f}, {bb_max[2]:+.3f}]  "
          f"centre=({bb_ctr[0]:+.3f}, {bb_ctr[1]:+.3f}, {bb_ctr[2]:+.3f})")
    print(f"[layers] dense x=[{dens_lo[0]:+.3f}, {dens_hi[0]:+.3f}]  "
          f"y=[{dens_lo[1]:+.3f}, {dens_hi[1]:+.3f}]  "
          f"z=[{dens_lo[2]:+.3f}, {dens_hi[2]:+.3f}]  (10-90% per axis)")
    t0 = time.time()
    mean2d, cov2d, depths, keep, camera = _project_scene(cfg, G)
    print(f"[layers] projected {len(keep)} visible in {time.time() - t0:.1f}s")

    ref_source = cfg.OUT if os.path.exists(cfg.OUT) else None
    if ref_source:
        ref_img = load_ref_image(ref_source, cfg.W, cfg.H)
    else:
        order = keep[np.argsort(-depths[keep])]
        ref_img = splat(cfg.W, cfg.H, mean2d, cov2d, G["colors"], G["opacities"],
                        order, verbose=False)
    saliency = compute_saliency(ref_img)
    return dict(
        cfg=cfg, G=G,
        mean2d=mean2d, cov2d=cov2d, depths=depths, keep=keep,
        ref_img=ref_img, saliency=saliency,
        camera=camera, bbox=(bb_min, bb_max),
        density_bbox=(dens_lo, dens_hi),
    )


def render_composition(scene_ref, composition=None, write=True, stamp_label=True,
                        scene_data=None, layer_cache=None, cache_max=32):
    """Render a layered composition.

    `scene_data` -- the dict from build_scene_data(); if None, this builds
                    it on the fly (one-shot mode).
    `layer_cache`-- optional OrderedDict (hash -> (color, alpha)); enables
                    fast re-compositing when only enabled/alpha change.
    """
    if scene_data is None:
        scene_data = build_scene_data(scene_ref)
    cfg = scene_data["cfg"]
    G = scene_data["G"]
    mean2d = scene_data["mean2d"]
    cov2d = scene_data["cov2d"]
    depths = scene_data["depths"]
    keep = scene_data["keep"]
    ref_img = scene_data["ref_img"]
    saliency = scene_data["saliency"]

    comp = deepcopy(DEFAULT_COMPOSITION)
    comp.update(cfg._raw.get("composition", {}))
    if composition:
        comp.update(composition)
    layers = comp.get("layers", DEFAULT_COMPOSITION["layers"])
    out_path = comp.get("out") or f"images/{cfg.SCENE_NAME}_layers.png"
    bg = np.asarray(comp.get("background", PAPER_COLOR), dtype=np.float64)

    # Pre-compute scene depth range for generative-curve depth blur
    depths_range = None
    if len(keep) > 0:
        depths_range = (float(depths[keep].min()), float(depths[keep].max()))

    canvas = np.tile(bg, (cfg.H, cfg.W, 1)).astype(np.float64)
    for spec in layers:
        if not spec.get("enabled", True):
            print(f"  {spec.get('name', spec.get('type'))}: off")
            continue
        layer_type = spec.get("type")
        layer_name = spec.get("name", layer_type)
        h = layer_content_hash(spec) if layer_cache is not None else None
        t_layer = time.time()

        cached = layer_cache.get(h) if (layer_cache is not None and h is not None) else None
        if cached is not None:
            color, alpha = cached
            print(f"  layer {layer_name}: cached")
        else:
            if layer_type == "base_splat":
                color, alpha = layer_base_splat(cfg, G, mean2d, cov2d, depths, keep, spec)
            elif layer_type == "surface_walks":
                color, alpha = layer_surface_walks(
                    cfg, G, mean2d, cov2d, keep, ref_img, saliency, spec, bg=bg)
            elif layer_type == "generative_curve":
                camera = scene_data.get("camera")
                if camera is None:
                    raise ValueError("generative_curve requires camera in scene_data")
                color, alpha = layer_generative_curve(
                    cfg, camera, ref_img, depths_range, spec, bg=bg)
            elif layer_type == "zline":
                camera = scene_data.get("camera")
                if camera is None:
                    raise ValueError("zline requires camera in scene_data")
                color, alpha = layer_zline(cfg, camera, ref_img, spec, bg=bg)
            else:
                raise ValueError(f"unknown layer type: {layer_type}")
            if layer_cache is not None:
                layer_cache[h] = (color, alpha)
                if isinstance(layer_cache, OrderedDict):
                    layer_cache.move_to_end(h)
                while len(layer_cache) > cache_max:
                    if isinstance(layer_cache, OrderedDict):
                        layer_cache.popitem(last=False)
                    else:
                        layer_cache.pop(next(iter(layer_cache)))

        canvas = _over(canvas, color, alpha,
                       layer_alpha=float(spec.get("alpha", 1.0)))
        print(f"  layer {layer_name}: on ({time.time() - t_layer:.1f}s)")

    if stamp_label:
        canvas = stamp(canvas, f"{cfg.SCENE_NAME} layers", cfg.W, cfg.H)
    if write:
        metadata = _build_png_metadata(cfg, comp)
        _save_png_with_metadata(out_path, canvas, metadata)
        print(f"\nsaved -> {out_path}")
    return canvas, out_path, comp


def main():
    scene_ref = os.environ.get("SCENE_FILE") or os.environ.get("SCENE_NAME", "siyun")
    render_composition(scene_ref)


if __name__ == "__main__":
    main()
