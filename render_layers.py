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
# FOCAL_FIELDS are appended to every type. The focal mask is a per-layer
# screen-space alpha modulator (rotated ellipse + smoothstep falloff) applied
# at COMPOSITE time, so changing focal params doesn't bust the layer cache.
# All focal_* keys live at the layer's top level regardless of params_in.
# ----------------------------------------------------------------------
FOCAL_FIELDS = [
    ("focal_enabled",   "focal on",    "bool",  "",       None),
    ("focal_cx",        "focal cx",    "float", "{:.2f}", None),
    ("focal_cy",        "focal cy",    "float", "{:.2f}", None),
    ("focal_rx",        "focal rx",    "float", "{:.2f}", None),
    ("focal_ry",        "focal ry",    "float", "{:.2f}", None),
    ("focal_angle_deg", "focal angle", "float", "{:.0f}", None),
    ("focal_falloff",   "focal soft",  "float", "{:.2f}", None),
    ("focal_invert",    "focal invert","bool",  "",       None),
]
_FOCAL_KEYS = {f[0] for f in FOCAL_FIELDS}

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
        ] + FOCAL_FIELDS,
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
        ] + FOCAL_FIELDS,
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
    # focal_* always lives at the layer's top level regardless of params_in
    if section and key not in _FOCAL_KEYS:
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
    return mean2d, cov2d, depths, keep


def _over(canvas, color, alpha, layer_alpha=1.0):
    a = np.clip(alpha * float(layer_alpha), 0.0, 1.0)[..., None]
    return canvas * (1.0 - a) + color * a


def _focal_mask(W, H, spec):
    """Per-layer focal-region alpha modulator.

    Returns an (H, W) float64 array in [0, 1], or None when focal_enabled is
    false. Geometry: rotated ellipse centred at (focal_cx*W, focal_cy*H) with
    radii (focal_rx, focal_ry) * max(W, H). Inside the inner ellipse
    (d = 1 - falloff) the mask is 1; outside the outer ellipse (d = 1 + falloff)
    the mask is 0; between them, a smoothstep transition. focal_invert flips
    the mask (useful for vignette: 0 in centre, 1 at edges)."""
    if not spec.get("focal_enabled", False):
        return None
    cx = float(spec.get("focal_cx", 0.5)) * W
    cy = float(spec.get("focal_cy", 0.5)) * H
    norm = float(max(W, H))
    rx = max(float(spec.get("focal_rx", 0.35)) * norm, 1.0)
    ry = max(float(spec.get("focal_ry", 0.45)) * norm, 1.0)
    angle = np.radians(float(spec.get("focal_angle_deg", 0.0)))
    falloff = float(spec.get("focal_falloff", 0.6))
    invert = bool(spec.get("focal_invert", False))

    ys, xs = np.mgrid[0:H, 0:W]
    dx = xs - cx
    dy = ys - cy
    cosA, sinA = np.cos(angle), np.sin(angle)
    u = (dx * cosA + dy * sinA) / rx
    v = (-dx * sinA + dy * cosA) / ry
    d = np.sqrt(u * u + v * v)

    inner = max(1.0 - falloff, 0.0)
    outer = 1.0 + max(falloff, 1e-3)
    t = np.clip((d - inner) / (outer - inner), 0.0, 1.0)
    mask = 1.0 - t * t * (3.0 - 2.0 * t)            # smoothstep, 1 inside -> 0 outside
    if invert:
        mask = 1.0 - mask
    return mask.astype(np.float64)


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

    order = keep[np.argsort(-depths[keep])]

    weights = None
    curvature_full = G.get("curvature")
    if curvature_weight > 0.0 and curvature_full is not None:
        c = curvature_full[order]
        c_norm = c / (c.max() + 1e-9)
        # Blend: uniform (1) + curvature emphasis. weight=1 -> nearly pure
        # curvature; weight=0 -> uniform (unused, falls through).
        weights = (1.0 - curvature_weight) + curvature_weight * c_norm

    order = _sample_order(order, point_pct, seed, weights=weights)
    cw_tag = f"  cw={curvature_weight:.2f}" if weights is not None else ""
    print(f"  base_splat: {len(order)} / {len(keep)} visible points "
          f"({point_pct:.2%})  bg={[f'{c:.2f}' for c in bg]}  sat={saturation:.2f}{cw_tag}")
    img = splat(cfg.W, cfg.H, mean2d, cov2d, G["colors"], G["opacities"],
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


def layer_surface_walks(cfg, G, mean2d, cov2d, keep, ref_img, saliency, spec):
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
    line_canvas = np.tile(PAPER_COLOR, (cfg.H, cfg.W, 1)).astype(np.float64)
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

    paper_luma = PAPER_COLOR @ np.array([0.299, 0.587, 0.114])
    luma = line_canvas @ np.array([0.299, 0.587, 0.114])
    alpha = np.clip((paper_luma - luma) / max(paper_luma, 1e-6), 0.0, 1.0)
    print(f"  surface_walks: {drawn} segments")
    return line_canvas, alpha


_HASH_EXCLUDE = ({"enabled", "alpha", "name",
                  "_ui_y", "_ui_h", "_ui_collapsed", "_ui_section"}
                 | _FOCAL_KEYS)   # focal is applied at composite time, not baked in


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
    
    Returns a dict with scene name, composition parameters, and render info.
    Can be serialized to JSON for tEXt chunk storage.
    """
    metadata = {
        "scene_name": cfg.SCENE_NAME,
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
    print(f"[layers] scene={cfg.SCENE_NAME} ply={cfg.PLY} "
          f"gaussians={eff_n}{suffix}")
    t0 = time.time()
    mean2d, cov2d, depths, keep = _project_scene(cfg, G)
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
                    cfg, G, mean2d, cov2d, keep, ref_img, saliency, spec)
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

        focal = _focal_mask(cfg.W, cfg.H, spec)
        composite_alpha = alpha if focal is None else alpha * focal
        canvas = _over(canvas, color, composite_alpha,
                       layer_alpha=float(spec.get("alpha", 1.0)))
        focal_tag = "  [focal]" if focal is not None else ""
        print(f"  layer {layer_name}: on ({time.time() - t_layer:.1f}s){focal_tag}")

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
