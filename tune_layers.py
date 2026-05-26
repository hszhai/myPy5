"""Interactive layer toggle/control panel for scene compositions.

The UI is driven by `LAYER_PARAM_SCHEMAS` in render_layers.py -- each
layer type advertises its editable params there, and this tool builds
fields for them automatically. Adding a new layer type = one entry in
the schema; this file does not need to change.

Run:
  SCENE_FILE=scenes/siyun_walks.json py5 tune_layers.py
  SCENE_NAME=siyun py5 tune_layers.py

Controls:
  click checkbox        toggle the layer on/off
  click numeric field   field clears; type the new value (current value
                        shown dim as placeholder; Enter to commit, Tab to
                        next, Esc to cancel)
  click str-choice      cycle through allowed values (e.g. mode, placement)
  RENDER / 'r'          run render_composition with the live state
  SAVE SCENE / 's'      write the live composition back into the scene JSON
"""
import contextlib
import io
import json
import os
import time
import traceback
from collections import OrderedDict
from copy import deepcopy

os.environ.setdefault("JAVA_HOME", "/usr/local/opt/openjdk@17")
import numpy as np
import py5

from render_layers import (
    DEFAULT_COMPOSITION, DEFAULT_WALKS, LAYER_PARAM_SCHEMAS,
    WALK_KEY_MAP, build_scene_data, layer_param_effective, layer_param_set,
    project_anchor, render_composition,
)
from scene_io import load_scene, scene_path


SCENE_REF = os.environ.get("SCENE_FILE") or os.environ.get("SCENE_NAME", "siyun")
cfg = load_scene(SCENE_REF)
SCENE_JSON = cfg.PATH if hasattr(cfg, "PATH") else scene_path(SCENE_REF)
SCENE_WALKS = getattr(cfg, "SURFACE_WALKS", {}) or {}

# ---- window layout -------------------------------------------------------
# Layout: control panel (left) | preview + camera + log (right)
W, H = 1320, 1000
PANEL_X = 20
PANEL_W = 320
PANEL_RIGHT_PAD = 18

# Preview window (right side, smaller)
PREVIEW_X = PANEL_X + PANEL_W + 20
PREVIEW_Y = 54
PREVIEW_W = 400
PREVIEW_H = 450

# Camera control panel (below preview, in right column)
CAM_PANEL_X = PREVIEW_X
CAM_PANEL_Y = PREVIEW_Y + PREVIEW_H + 10
CAM_PANEL_W = PREVIEW_W
CAM_PANEL_H = 180

# Wireframe 3D-inspection viewport (below camera panel)
WIRE_X = CAM_PANEL_X
WIRE_Y = CAM_PANEL_Y + CAM_PANEL_H + 10
WIRE_W = PREVIEW_W
WIRE_H = 150

# Log panel (below wireframe, right side)
LOG_X = PREVIEW_X
LOG_Y = WIRE_Y + WIRE_H + 10
LOG_W = PREVIEW_W
LOG_H = H - LOG_Y - 18
LOG_LINE_H = 14
LOG_KEEP = 200            # rolling buffer length

# UI font — created at UI_FONT_CREATE_SIZE so smaller text_size() calls
# downscale crisply in P2D's texture-baked PFont. We try a list of names in
# order and use the first that resolves to an installed face.
# Switched to SF Pro (Apple's native font) for better readability, with fallbacks.
UI_FONT_CANDIDATES = (".SF NS Text", "SF Pro Text", "Monaco", "Menlo", "Helvetica Neue", "Arial", "Lucida Grande")
UI_FONT_CREATE_SIZE = 18   # kept modest — larger atlases SIGILL'd on Apple Silicon JOGL
UI_TEXT_BUMP = 2           # +2pt added to every _tsz(n) call (global readability)


def _tsz(size):
    py5.text_size(size + UI_TEXT_BUMP)


HEADER_H = 28              # per-layer header row
FIELD_H = 22               # numeric field height
ROW_GAP = 4
PARAMS_PER_ROW = 2         # 2-column param grid for plain fields
LABEL_W = 105
FIELD_W = 78
COL_GAP = 14
COL_PITCH = LABEL_W + FIELD_W + COL_GAP

# RGB group layout (single full-row entry: "label  R [..]  G [..]  B [..]")
RGB_GROUP_LABEL_W = 90
RGB_SUB_LABEL_W = 16
RGB_SUB_FIELD_W = 64
RGB_SUB_GAP = 12
RGB_SUB_PITCH = RGB_SUB_LABEL_W + RGB_SUB_FIELD_W + RGB_SUB_GAP

# Two rows of buttons across the panel header — main action buttons
_BTN_W = 150
_BTN_H = 28
_BTN_GAP = 6
_BTN_ROW1_Y = 8
_BTN_ROW2_Y = _BTN_ROW1_Y + _BTN_H + _BTN_GAP
BTN_RENDER     = dict(x=PANEL_X,                         y=_BTN_ROW1_Y, w=_BTN_W, h=_BTN_H)
BTN_SAVE_IMG   = dict(x=PANEL_X + (_BTN_W + _BTN_GAP),   y=_BTN_ROW1_Y, w=_BTN_W, h=_BTN_H)
BTN_SAVE_SCENE = dict(x=PANEL_X,                         y=_BTN_ROW2_Y, w=_BTN_W, h=_BTN_H)
BTN_ADD_SPLAT  = dict(x=PANEL_X + (_BTN_W + _BTN_GAP),   y=_BTN_ROW2_Y, w=_BTN_W, h=_BTN_H)
BTN_ADD_WALKS  = dict(x=PANEL_X,                         y=_BTN_ROW2_Y + _BTN_H + _BTN_GAP, w=_BTN_W, h=_BTN_H)
BTN_ADD_CURVE  = dict(x=PANEL_X + (_BTN_W + _BTN_GAP),   y=_BTN_ROW2_Y + _BTN_H + _BTN_GAP, w=_BTN_W, h=_BTN_H)

# Layer block layout: the chevron + checkbox + name + alpha live in the header row.
TRI_W = 12
TRI_GAP = 8
CHECK_W = 18

LAYERS_START_Y = _BTN_ROW2_Y + _BTN_H * 2 + _BTN_GAP + 16

# Templates used when "+ SPLAT" / "+ WALKS" buttons add a new layer.
ADD_SPLAT_TEMPLATE = {
    "name": "splat",
    "type": "base_splat",
    "enabled": True,
    "alpha": 0.6,
    "point_pct": 0.25,
    "bg_r": 0.0, "bg_g": 0.0, "bg_b": 0.0,
    "saturation": 1.0,
    "curvature_weight": 0.0,
    "seed": 17,
}
ADD_WALKS_TEMPLATE = {
    "name": "walks",
    "type": "surface_walks",
    "enabled": True,
    "alpha": 1.0,
    "params": {
        "n_walkers": 300,
        "steps": 30,
        "step_radius_px": 14.0,
        "forward_bias": 8.0,
        "direction_mode": "global",
        "global_dir_deg": 90.0,
        "noise_scale": 90.0,
        "stroke_alpha": 0.65,
        "ink_darken": 0.0,
        "ink_r": 0.0, "ink_g": 0.0, "ink_b": 0.0,
        "placement": "saliency",
        "stroke_mode": "line",
        "stroke_width": 1.0,
        "seed": 17,
    },
}
ADD_CURVE_TEMPLATE = {
    "name": "curve",
    "type": "generative_curve",
    "enabled": True,
    "alpha": 0.9,
    "params": {
        "shape": "sphere",
        "n_points": 200,
        "radius": 0.6,
        "seed": 17,
        "stroke_mode": "splat",
        "stroke_alpha": 0.6,
        "splat_scale": 0.4,
        "splat_alpha_scale": 0.35,
        "splat_min_sigma": 0.10,
        "splat_max_sigma": 1.20,
        "n_stamps": 5,
        "color_mode": "fixed",
        "color_r": 0.9, "color_g": 0.3, "color_b": 0.1,
        "depth_focus": 0.5,
        "depth_blur": 0.0,
        "line_jitter": 0.0,
        "connect_closest": False,
    },
}


_MASK3D_DEFAULTS = dict(
    mask3d_enabled=False,
    mask3d_x=0.0, mask3d_y=0.0, mask3d_z=0.0,
    mask3d_r_in=0.3, mask3d_r_out=1.0,
    mask3d_invert=False,
)

# Strip leftover focal_* keys from legacy scenes/state so save_scene doesn't
# bloat the JSON with dead params.
_FOCAL_LEGACY_KEYS = (
    "focal_enabled", "focal_cx", "focal_cy", "focal_rx", "focal_ry",
    "focal_angle_deg", "focal_falloff", "focal_invert",
)


def _hydrate_layer(layer):
    """Expand legacy fields so the tuner can edit them as individual params."""
    # Common to every layer type: 3D mask anchor params live at top level.
    for k, v in _MASK3D_DEFAULTS.items():
        layer.setdefault(k, v)
    for k in _FOCAL_LEGACY_KEYS:
        layer.pop(k, None)
    if layer.get("type") == "base_splat":
        bg = layer.get("bg") or layer.get("background")
        if bg is not None and len(bg) >= 3:
            layer.setdefault("bg_r", float(bg[0]))
            layer.setdefault("bg_g", float(bg[1]))
            layer.setdefault("bg_b", float(bg[2]))
            # remove the legacy form so writes from the UI are the source of truth
            layer.pop("bg", None)
            layer.pop("background", None)
        else:
            layer.setdefault("bg_r", 0.0)
            layer.setdefault("bg_g", 0.0)
            layer.setdefault("bg_b", 0.0)
        layer.setdefault("saturation", 1.0)
        layer.setdefault("curvature_weight", 0.0)
        layer.setdefault("depth_focus", 0.5)
        layer.setdefault("depth_blur", 0.0)
    if layer.get("type") == "surface_walks":
        # ink_r/g/b live inside layer["params"] (params_in="params" in the schema)
        params = layer.setdefault("params", {})
        params.setdefault("ink_r", 0.0)
        params.setdefault("ink_g", 0.0)
        params.setdefault("ink_b", 0.0)
    if layer.get("type") == "generative_curve":
        params = layer.setdefault("params", {})
        params.setdefault("color_r", 0.0)
        params.setdefault("color_g", 0.0)
        params.setdefault("color_b", 0.0)
        params.setdefault("depth_focus", 0.5)
        params.setdefault("depth_blur", 0.0)


def _load_composition():
    comp = deepcopy(DEFAULT_COMPOSITION)
    comp.update(cfg._raw.get("composition", {}))
    comp["layers"] = [deepcopy(l) for l in comp.get("layers", [])]
    for layer in comp["layers"]:
        _hydrate_layer(layer)
    return comp


composition = _load_composition()
state = dict(
    render_image=None,
    msg="",
    render_pending=False,
    log_lines=[],
    # Camera parameters (used when rendering)
    cam_azimuth=0.0,
    cam_elevation=0.0,
    cam_fov=40.0,
    cam_distance_k=1.0,
    # Fields for camera controls
    cam_fields=[],
)
fields = []                  # all interactive fields, rebuilt by _rebuild_fields
chevrons = []                # collapse-arrow click rectangles, also rebuilt

# Pre-compute scene data (PLY load + projection + saliency) ONCE, reused
# across every render. layer_cache memoises per-layer (color, alpha) so
# toggling enabled/alpha doesn't trigger a re-render.
print("[tune-layers] building scene data (load + project + saliency)...")
scene_data = build_scene_data(SCENE_REF)
layer_cache = OrderedDict()
print("[tune-layers] ready")


def _log_append(text):
    """Append a string (may contain newlines) to the rolling log buffer."""
    for line in text.splitlines():
        if line.strip():
            state["log_lines"].append(line)
    if len(state["log_lines"]) > LOG_KEEP:
        state["log_lines"] = state["log_lines"][-LOG_KEEP:]


# ---- helpers -------------------------------------------------------------
def _layer(i):
    return composition["layers"][i]


def _read(field):
    layer = _layer(field["layer"])
    key = field["key"]
    if key == "enabled":
        return bool(layer.get("enabled", True))
    if key == "alpha":
        return float(layer.get("alpha", 1.0))
    return layer_param_effective(layer, key, scene_walks=SCENE_WALKS)


def _write(field, value):
    layer = _layer(field["layer"])
    key = field["key"]
    if key == "alpha":
        layer["alpha"] = float(value)
    elif key == "enabled":
        layer["enabled"] = bool(value)
    else:
        layer_param_set(layer, key, value)


def _format(value, type_, fmt):
    if value is None:
        return "?"
    if type_ == "str":
        return str(value)
    try:
        return fmt.format(value)
    except Exception:
        return str(value)


def _hit(rect):
    return (rect["x"] <= py5.mouse_x <= rect["x"] + rect["w"]
            and rect["y"] <= py5.mouse_y <= rect["y"] + rect["h"])


# ---- field layout --------------------------------------------------------
def _rebuild_fields():
    fields.clear()
    chevrons.clear()
    y = LAYERS_START_Y
    for i, layer in enumerate(composition.get("layers", [])):
        layer["_ui_y"] = y
        collapsed = bool(layer.get("_ui_collapsed", False))

        # Collapse-arrow click region (drawn separately as a chevron, not a field)
        chevrons.append(dict(layer=i, x=PANEL_X, y=y + 6, w=TRI_W, h=TRI_W))

        # Header row: chevron (PANEL_X) + checkbox + name + alpha (right)
        check_x = PANEL_X + TRI_W + TRI_GAP
        fields.append(dict(
            kind="enabled", layer=i, key="enabled",
            x=check_x, y=y + 4, w=CHECK_W, h=CHECK_W,
            type="bool", fmt="", choices=None))
        # Alpha field on the right side of the header
        fields.append(dict(
            kind="alpha", layer=i, key="alpha", label="alpha",
            x=PANEL_X + 290, y=y + 3, w=70, h=HEADER_H - 6,
            type="float", fmt="{:.2f}", choices=None,
            active=False, edit_text=""))

        if collapsed:
            # Skip per-type params -- the block stays at header height
            layer["_ui_h"] = HEADER_H + 6
            y += layer["_ui_h"]
            continue

        # Type-specific params from schema. Plain fields fall into a 2-col grid;
        # "rgb" fields take a full row with 3 sub-inputs (R/G/B).
        schema = LAYER_PARAM_SCHEMAS.get(layer.get("type"), {})
        params_y = y + HEADER_H + 6
        col, row = 0, 0
        for (key, label, type_, fmt, choices) in schema.get("fields", []):
            if type_ in ("rgb", "vec3"):
                if col != 0:
                    row += 1
                    col = 0
                row_y = params_y + row * (FIELD_H + ROW_GAP)
                suffixes = ("r", "g", "b") if type_ == "rgb" else ("x", "y", "z")
                for k, suffix in enumerate(suffixes):
                    x_label = PANEL_X + RGB_GROUP_LABEL_W + k * RGB_SUB_PITCH
                    x_field = x_label + RGB_SUB_LABEL_W
                    fields.append(dict(
                        kind="rgb_sub", layer=i,
                        key=f"{key}_{suffix}", label=suffix.upper(),
                        group_label=label if k == 0 else None,
                        x=x_field, y=row_y,
                        w=RGB_SUB_FIELD_W, h=FIELD_H,
                        type="float", fmt=fmt, choices=None,
                        active=False, edit_text=""))
                row += 1
                col = 0
                continue

            if type_ == "bool":
                # Slot a checkbox into the 2-col grid; label is drawn to the
                # left of the box by _draw_checkbox (param-kind branch).
                cx = PANEL_X + col * COL_PITCH + LABEL_W
                cy = params_y + row * (FIELD_H + ROW_GAP)
                fields.append(dict(
                    kind="param", layer=i, key=key, label=label,
                    x=cx, y=cy + 2, w=CHECK_W, h=CHECK_W,
                    type="bool", fmt="", choices=None))
                col += 1
                if col >= PARAMS_PER_ROW:
                    col = 0
                    row += 1
                continue

            cx = PANEL_X + col * COL_PITCH + LABEL_W
            cy = params_y + row * (FIELD_H + ROW_GAP)
            fields.append(dict(
                kind="param", layer=i, key=key, label=label,
                x=cx, y=cy, w=FIELD_W, h=FIELD_H,
                type=type_, fmt=fmt, choices=choices,
                active=False, edit_text=""))
            col += 1
            if col >= PARAMS_PER_ROW:
                col = 0
                row += 1

        total_rows = row + (1 if col != 0 else 0)
        block_h = HEADER_H + 6 + total_rows * (FIELD_H + ROW_GAP) + 10
        layer["_ui_h"] = block_h
        y += block_h


def _setup_complete():
    _rebuild_fields()


# ---- drawing -------------------------------------------------------------
def setup():
    py5.size(W, H, py5.P2D)
    # P2D already enables default MSAA; calling py5.smooth(N) here can SIGILL
    # on Apple Silicon JOGL drivers, and smooth() is only valid in settings()
    # anyway. Skip it.
    py5.frame_rate(60)
    # Try candidates in order; skip py5.list_fonts() (AWT enumeration has
    # been flaky on Apple Silicon). create_font on a non-installed name
    # silently falls back, so we don't gain much from prechecking anyway.
    state["ui_font"] = None
    for name in UI_FONT_CANDIDATES:
        try:
            font = py5.create_font(name, UI_FONT_CREATE_SIZE)
            state["ui_font"] = font
            state["ui_font_name"] = name
            print(f"[ui] created font {name!r} @ {UI_FONT_CREATE_SIZE}px")
            break
        except Exception as exc:
            print(f"[ui] could not load font {name!r}: {exc}")
    # Defer py5.text_font() to the first draw() call -- binding the font
    # inside setup() crashes JOGL on some Apple Silicon configurations.
    _tsz(12)
    _setup_complete()


def draw():
    # Bind the UI font lazily on the first frame. Doing this inside setup()
    # has SIGILL'd on Apple Silicon JOGL right after create_font returns.
    if state.get("ui_font") is not None and not state.get("_font_bound"):
        try:
            py5.text_font(state["ui_font"])
            state["_font_bound"] = True
            print(f"[ui] bound font {state.get('ui_font_name')!r}")
        except Exception as exc:
            print(f"[ui] text_font failed: {exc}")
            state["_font_bound"] = True   # don't keep retrying

    if state["render_pending"]:
        state["render_pending"] = False
        do_render()

    py5.background(14, 17, 23)

    # Title + scene bbox sub-line so the user can read the world-coord range
    # while typing mask3d_x/y/z and r_in/r_out values.
    py5.fill(220, 225, 235); _tsz(13)
    py5.text(f"Layers  --  {SCENE_JSON}", 20, 24)
    py5.fill(150, 160, 175); _tsz(10)
    bb_min, bb_max = scene_data["bbox"]
    py5.text(
        f"scene  x:[{bb_min[0]:+.2f}, {bb_max[0]:+.2f}]"
        f"  y:[{bb_min[1]:+.2f}, {bb_max[1]:+.2f}]"
        f"  z:[{bb_min[2]:+.2f}, {bb_max[2]:+.2f}]"
        f"  span~{float(np.linalg.norm(bb_max - bb_min)):.2f}",
        20, 42)

    # Preview box
    py5.no_fill(); py5.stroke(70, 75, 90); py5.stroke_weight(1)
    py5.rect(PREVIEW_X, PREVIEW_Y, PREVIEW_W, PREVIEW_H)
    if state["render_image"] is not None:
        py5.image(state["render_image"], PREVIEW_X, PREVIEW_Y, PREVIEW_W, PREVIEW_H)
        # Always show mask anchors and ranges in preview
        _draw_mask3d_overlay()
    else:
        py5.fill(130, 135, 150); _tsz(12)
        py5.text("(press RENDER)", PREVIEW_X + 220, PREVIEW_Y + PREVIEW_H / 2)

    _draw_button(BTN_RENDER,     "RENDER (r)",      (170, 70, 70),  (200, 110, 110))
    _draw_button(BTN_SAVE_IMG,   "SAVE IMG",        (130, 110, 60), (165, 140, 80))
    _draw_button(BTN_SAVE_SCENE, "SAVE SCENE (s)",  (70, 110, 170), (110, 150, 200))
    _draw_button(BTN_ADD_SPLAT,  "+ SPLAT",         (60, 130, 90),  (90, 165, 120))
    _draw_button(BTN_ADD_WALKS,  "+ WALKS",         (60, 130, 90),  (90, 165, 120))
    _draw_button(BTN_ADD_CURVE,  "+ CURVE",         (60, 130, 90),  (90, 165, 120))

    _draw_camera_panel()
    _draw_wireframe_view()

    # Per-layer header strip (chevron + name + type + alpha label)
    name_x = PANEL_X + TRI_W + TRI_GAP + CHECK_W + 10
    for i, layer in enumerate(composition.get("layers", [])):
        y = layer["_ui_y"]
        enabled = bool(layer.get("enabled", True))
        collapsed = bool(layer.get("_ui_collapsed", False))

        # Chevron triangle
        _draw_chevron(PANEL_X, y + 8, expanded=not collapsed)

        _tsz(13)
        py5.fill(230 if enabled else 110)
        py5.text(layer.get("name") or layer.get("type"), name_x, y + 17)
        py5.fill(140, 150, 170); _tsz(10)
        py5.text(layer.get("type", ""), name_x, y + 30)
        py5.fill(195, 205, 220); _tsz(11)
        py5.text("alpha", PANEL_X + 252, y + 17)

        bottom = y + layer["_ui_h"] - 4
        py5.stroke(48, 54, 66); py5.no_fill(); py5.stroke_weight(1)
        py5.line(PANEL_X, bottom, PANEL_X + PANEL_W, bottom)

    _draw_log()


def _draw_camera_panel():
    """Draw camera controls in the right preview column."""
    # Panel background
    py5.no_fill(); py5.stroke(70, 75, 90); py5.stroke_weight(1)
    py5.rect(CAM_PANEL_X, CAM_PANEL_Y, CAM_PANEL_W, CAM_PANEL_H)
    
    # Title
    py5.fill(195, 205, 220); _tsz(11)
    py5.text("Camera", CAM_PANEL_X + 8, CAM_PANEL_Y + 18)
    
    # Camera parameters
    row_y = CAM_PANEL_Y + 30
    row_h = 22
    row_gap = 4
    label_w = 70
    field_w = 80
    
    # Azimuth
    py5.fill(170, 180, 195); _tsz(10)
    py5.text("azimuth", CAM_PANEL_X + 8, row_y + 14)
    py5.stroke(110, 120, 140); py5.stroke_weight(1)
    py5.fill(31, 35, 45)
    py5.rect(CAM_PANEL_X + label_w, row_y, field_w, row_h)
    py5.fill(200, 210, 225); _tsz(10)
    py5.text(f"{state['cam_azimuth']:.1f}°", CAM_PANEL_X + label_w + 6, row_y + 15)
    
    # Elevation
    row_y += row_h + row_gap
    py5.fill(170, 180, 195); _tsz(10)
    py5.text("elevation", CAM_PANEL_X + 8, row_y + 14)
    py5.stroke(110, 120, 140); py5.stroke_weight(1)
    py5.fill(31, 35, 45)
    py5.rect(CAM_PANEL_X + label_w, row_y, field_w, row_h)
    py5.fill(200, 210, 225); _tsz(10)
    py5.text(f"{state['cam_elevation']:.1f}°", CAM_PANEL_X + label_w + 6, row_y + 15)
    
    # FOV
    row_y += row_h + row_gap
    py5.fill(170, 180, 195); _tsz(10)
    py5.text("fov", CAM_PANEL_X + 8, row_y + 14)
    py5.stroke(110, 120, 140); py5.stroke_weight(1)
    py5.fill(31, 35, 45)
    py5.rect(CAM_PANEL_X + label_w, row_y, field_w, row_h)
    py5.fill(200, 210, 225); _tsz(10)
    py5.text(f"{state['cam_fov']:.1f}°", CAM_PANEL_X + label_w + 6, row_y + 15)
    
    # Distance factor
    row_y += row_h + row_gap
    py5.fill(170, 180, 195); _tsz(10)
    py5.text("dist_k", CAM_PANEL_X + 8, row_y + 14)
    py5.stroke(110, 120, 140); py5.stroke_weight(1)
    py5.fill(31, 35, 45)
    py5.rect(CAM_PANEL_X + label_w, row_y, field_w, row_h)
    py5.fill(200, 210, 225); _tsz(10)
    py5.text(f"{state['cam_distance_k']:.2f}", CAM_PANEL_X + label_w + 6, row_y + 15)


def _draw_chevron(x, y, expanded):
    py5.no_stroke()
    py5.fill(170, 180, 200)
    if expanded:
        # pointing down (open)
        py5.triangle(x, y, x + TRI_W, y, x + TRI_W / 2, y + TRI_W * 0.7)
    else:
        # pointing right (collapsed)
        py5.triangle(x, y, x, y + TRI_W, x + TRI_W * 0.8, y + TRI_W / 2)

    # Fields
    for f in fields:
        if f["type"] == "bool":
            _draw_checkbox(f)
        else:
            _draw_field(f)

    _draw_log()

    # Bottom-left instruction strip (under the preview, not under the log).
    py5.fill(120, 130, 150); _tsz(10)
    py5.text("click number to edit (clears on click) -- click str field to cycle "
             "values -- Tab/Enter commit -- Esc cancel",
             20, H - 8)


def _draw_log():
    py5.no_fill(); py5.stroke(50, 56, 70); py5.stroke_weight(1)
    py5.rect(LOG_X, LOG_Y, LOG_W, LOG_H)
    py5.fill(150, 160, 180); _tsz(10)
    py5.text("log", LOG_X + 6, LOG_Y - 4)
    # Current render status, shown inline next to the "log" header
    if state["msg"]:
        py5.fill(145, 230, 155); _tsz(11)
        py5.text(state["msg"], LOG_X + 38, LOG_Y - 4)

    # Last N lines that fit
    n_visible = max(1, (LOG_H - 12) // LOG_LINE_H)
    visible = state["log_lines"][-n_visible:]
    py5.fill(190, 215, 200); _tsz(11)
    for i, line in enumerate(visible):
        # truncate too-long lines to panel width
        text = line
        max_w = LOG_W - 16
        while text and py5.text_width(text) > max_w:
            text = text[:-1]
        if text != line:
            text = text[:-2] + ".."
        py5.text(text, LOG_X + 8, LOG_Y + 14 + i * LOG_LINE_H)


def _draw_button(btn, label, base, hover_rgb):
    rgb = hover_rgb if _hit(btn) else base
    py5.stroke(*[min(c + 30, 255) for c in rgb])
    py5.fill(*rgb); py5.stroke_weight(1)
    py5.rect(btn["x"], btn["y"], btn["w"], btn["h"])
    py5.no_stroke(); py5.fill(255); _tsz(13)
    tw = py5.text_width(label)
    py5.text(label, btn["x"] + (btn["w"] - tw) / 2, btn["y"] + 20)


def _draw_toggle(btn, label, active):
    """Button that visibly latches when active."""
    base  = (60, 100, 140) if active else (45, 50, 62)
    hover = (90, 140, 190) if active else (75, 82, 98)
    rgb = hover if _hit(btn) else base
    py5.stroke(*[min(c + 30, 255) for c in rgb])
    py5.fill(*rgb); py5.stroke_weight(1)
    py5.rect(btn["x"], btn["y"], btn["w"], btn["h"])
    py5.no_stroke(); py5.fill(235 if active else 175); _tsz(13)
    prefix = "[x] " if active else "[ ] "
    tw = py5.text_width(prefix + label)
    py5.text(prefix + label, btn["x"] + (btn["w"] - tw) / 2, btn["y"] + 20)


def _draw_points_overlay():
    """Overlay the surface points the walker actually stepped on (curve nodes).
    Populated by do_render() via render_layers.DEBUG_POINTS -- shows ONLY the
    points the walks selected, not the full projected point cloud."""
    pts = state.get("debug_points") or []
    if not pts:
        return
    cfg_w = scene_data["cfg"].W
    cfg_h = scene_data["cfg"].H
    sx = PREVIEW_W / cfg_w
    sy = PREVIEW_H / cfg_h
    n_max = 8000
    if len(pts) > n_max:
        step = max(1, len(pts) // n_max)
        pts = pts[::step]
    py5.stroke(255, 220, 80, 220); py5.stroke_weight(3); py5.no_fill()
    for p in pts:
        py5.point(PREVIEW_X + p[0] * sx, PREVIEW_Y + p[1] * sy)


_BBOX_EDGES = (
    (0, 1), (0, 2), (1, 3), (2, 3),     # bottom face
    (4, 5), (4, 6), (5, 7), (6, 7),     # top face
    (0, 4), (1, 5), (2, 6), (3, 7),     # verticals
)


def _bbox_corners(lo, hi):
    return np.array([
        [lo[0], lo[1], lo[2]], [hi[0], lo[1], lo[2]],
        [lo[0], hi[1], lo[2]], [hi[0], hi[1], lo[2]],
        [lo[0], lo[1], hi[2]], [hi[0], lo[1], hi[2]],
        [lo[0], hi[1], hi[2]], [hi[0], hi[1], hi[2]],
    ], dtype=np.float64)


def _circle_3d(center, radius, axis_id, n=32):
    """n+1 points tracing a circle of `radius` around `center` in the plane
    perpendicular to one of the world axes (axis_id: 0=x, 1=y, 2=z)."""
    t = np.linspace(0.0, 2.0 * np.pi, n + 1)
    c, s = np.cos(t) * radius, np.sin(t) * radius
    pts = np.zeros((n + 1, 3), dtype=np.float64)
    if axis_id == 0:
        pts[:, 1] = c; pts[:, 2] = s
    elif axis_id == 1:
        pts[:, 0] = c; pts[:, 2] = s
    else:
        pts[:, 0] = c; pts[:, 1] = s
    return pts + np.asarray(center, dtype=np.float64)


def _wire_content_rect():
    """Inner rect inside the wireframe panel that matches the scene's render
    aspect (cfg.W : cfg.H), centred and with a small margin. Using this for
    projection scaling avoids squashing the bbox flat when WIRE_W/WIRE_H
    differs from the scene aspect."""
    cfg = scene_data["cfg"]
    aspect = cfg.W / cfg.H
    margin = 8
    av_w = WIRE_W - 2 * margin
    av_h = WIRE_H - 2 * margin
    if av_w / av_h > aspect:
        ch = av_h
        cw = int(round(ch * aspect))
    else:
        cw = av_w
        ch = int(round(cw / aspect))
    cx = WIRE_X + (WIRE_W - cw) // 2
    cy = WIRE_Y + (WIRE_H - ch) // 2
    return cx, cy, cw, ch


def _project_world_to_wire(xyz, content_rect=None):
    """Batch-project an (N, 3) world-space array into wireframe-viewport pixel
    coords using a content rect that preserves the scene aspect. Points behind
    the camera get NaN so callers can skip them."""
    cfg = scene_data["cfg"]
    cam = scene_data["camera"]
    if content_rect is None:
        content_rect = _wire_content_rect()
    cx, cy, cw, ch = content_rect
    p = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    cam_p = (p - cam["center"]) @ cam["Rcam"].T
    z = cam_p[:, 2] + cam["distance"]
    z_safe = np.where(z > 1e-6, z, np.nan)
    x_scene = cfg.W / 2.0 + cam["focal"] * cam_p[:, 0] / z_safe
    y_scene = cfg.H / 2.0 + cam["ysign"] * cam["focal"] * cam_p[:, 1] / z_safe
    sx = cw / cfg.W
    sy = ch / cfg.H
    return cx + x_scene * sx, cy + y_scene * sy


def _poly_lines(xs, ys, weight=1, rgba=(180, 180, 200, 200)):
    py5.stroke(rgba[0], rgba[1], rgba[2], rgba[3])
    py5.stroke_weight(weight)
    py5.no_fill()
    for i in range(len(xs) - 1):
        if np.isnan(xs[i]) or np.isnan(xs[i + 1]):
            continue
        py5.line(xs[i], ys[i], xs[i + 1], ys[i + 1])


def _draw_wireframe_view():
    """A small, always-live wireframe of the scene's spatial layout:
    full bbox, density bbox (10-90% per axis), world axes at the centre,
    and each mask3d_enabled layer's anchor + r_in/r_out spheres (drawn as
    three orthogonal great-circle rings each). The 3D content lives in an
    aspect-correct sub-rect so the bbox isn't squashed."""
    cfg = scene_data["cfg"]

    # Outer panel background
    py5.no_stroke(); py5.fill(16, 19, 26)
    py5.rect(WIRE_X, WIRE_Y, WIRE_W, WIRE_H)
    py5.no_fill(); py5.stroke(60, 68, 84); py5.stroke_weight(1)
    py5.rect(WIRE_X, WIRE_Y, WIRE_W, WIRE_H)

    # Header strip: title + camera params
    py5.fill(180, 188, 200); _tsz(10)
    py5.text("wireframe", WIRE_X + 6, WIRE_Y - 4)
    cam_info = (f"elev={getattr(cfg, 'ELEV_DEG', 0):.0f}°   "
                f"azim={getattr(cfg, 'AZIM_DEG', 0):.0f}°   "
                f"fov={getattr(cfg, 'FOV_DEG', 0):.0f}°   "
                f"dist×k={getattr(cfg, 'DISTANCE_K', 0):.2f}")
    py5.fill(150, 160, 175); _tsz(10)
    tw = py5.text_width(cam_info)
    py5.text(cam_info, WIRE_X + WIRE_W - tw - 6, WIRE_Y - 4)

    # Inner aspect-correct content rect + frame (subtle so it's clearly the
    # camera-aspect viewport, not the panel)
    rect = _wire_content_rect()
    cx_r, cy_r, cw_r, ch_r = rect
    py5.no_fill(); py5.stroke(40, 46, 58, 220); py5.stroke_weight(1)
    py5.rect(cx_r, cy_r, cw_r, ch_r)

    # Full bbox (dim grey)
    bb_min, bb_max = scene_data["bbox"]
    corners = _bbox_corners(bb_min, bb_max)
    xs, ys = _project_world_to_wire(corners, rect)
    py5.stroke(110, 120, 140, 180); py5.stroke_weight(1)
    for a, b in _BBOX_EDGES:
        if np.isnan(xs[a]) or np.isnan(xs[b]):
            continue
        py5.line(xs[a], ys[a], xs[b], ys[b])

    # Density bbox (warm orange)
    if "density_bbox" in scene_data:
        d_lo, d_hi = scene_data["density_bbox"]
        d_corners = _bbox_corners(d_lo, d_hi)
        dxs, dys = _project_world_to_wire(d_corners, rect)
        py5.stroke(210, 150, 70, 200); py5.stroke_weight(1)
        for a, b in _BBOX_EDGES:
            if np.isnan(dxs[a]) or np.isnan(dxs[b]):
                continue
            py5.line(dxs[a], dys[a], dxs[b], dys[b])

    # World axes triad at the scene centre, length = 15% of bbox diagonal
    bb_ctr = (bb_min + bb_max) / 2.0
    diag = float(np.linalg.norm(bb_max - bb_min))
    axis_len = diag * 0.15
    axis_specs = (
        (np.array([1.0, 0.0, 0.0]), (255,  80,  80)),
        (np.array([0.0, 1.0, 0.0]), ( 80, 220,  80)),
        (np.array([0.0, 0.0, 1.0]), ( 80, 130, 255)),
    )
    for axis, color in axis_specs:
        seg = np.stack([bb_ctr, bb_ctr + axis * axis_len])
        axs, ays = _project_world_to_wire(seg, rect)
        if np.isnan(axs[0]) or np.isnan(axs[1]):
            continue
        py5.stroke(color[0], color[1], color[2], 230); py5.stroke_weight(2)
        py5.line(axs[0], ays[0], axs[1], ays[1])

    # Mask3D layers: centre dot + three orthogonal rings for r_in and r_out
    for layer in composition.get("layers", []):
        if not layer.get("mask3d_enabled"):
            continue
        center = np.array([
            float(layer.get("mask3d_x", 0.0)),
            float(layer.get("mask3d_y", 0.0)),
            float(layer.get("mask3d_z", 0.0)),
        ], dtype=np.float64)
        r_in = float(layer.get("mask3d_r_in", 0.3))
        r_out = float(layer.get("mask3d_r_out", 1.0))
        invert = bool(layer.get("mask3d_invert", False))
        tint = (255, 110, 110) if invert else (110, 200, 255)
        dim = (max(tint[0] - 60, 0), max(tint[1] - 60, 0), max(tint[2] - 60, 0))

        # Centre marker
        mx, my = _project_world_to_wire(center[None], rect)
        if not np.isnan(mx[0]):
            py5.no_stroke(); py5.fill(tint[0], tint[1], tint[2], 230)
            py5.circle(mx[0], my[0], 5)
            py5.fill(220, 230, 245); _tsz(10)
            py5.text(layer.get("name") or layer.get("type", ""),
                     mx[0] + 6, my[0] - 4)

        for r, rgba in (
            (r_in,  (*tint, 220)),
            (r_out, (*dim,  150)),
        ):
            for axis_id in (0, 1, 2):
                pts = _circle_3d(center, r, axis_id, n=28)
                cxs, cys = _project_world_to_wire(pts, rect)
                _poly_lines(cxs, cys, weight=1, rgba=rgba)


def _draw_mask3d_overlay():
    """Project each mask3d_enabled layer's anchor + radius onto the preview.

    The 3D ball is approximated in screen space: r_pixel = r_world * focal / z.
    Inner (bright) ring is the fully-masked-in region; outer (dim) ring is the
    zero-mask boundary. Cyan tint for spotlight (mask3d_invert=false), red for
    inverted. Centre dot + layer name for identification.
    """
    cfg = scene_data["cfg"]
    cam = scene_data["camera"]
    sx = PREVIEW_W / cfg.W
    sy = PREVIEW_H / cfg.H

    n_pts = 80
    thetas = np.linspace(0.0, 2.0 * np.pi, n_pts)
    cosT = np.cos(thetas); sinT = np.sin(thetas)

    for layer in composition.get("layers", []):
        if not layer.get("mask3d_enabled"):
            continue
        anchor_xyz = np.array([
            float(layer.get("mask3d_x", 0.0)),
            float(layer.get("mask3d_y", 0.0)),
            float(layer.get("mask3d_z", 0.0)),
        ], dtype=np.float64)
        proj = project_anchor(cam, cfg, anchor_xyz)
        if proj is None:
            continue                                      # behind camera
        x_px, y_px, z_view = proj
        # Approximate world-to-screen radius scale at the anchor depth.
        pix_per_world = cam["focal"] / max(z_view, 1e-3)
        r_in_w  = float(layer.get("mask3d_r_in", 0.3))
        r_out_w = float(layer.get("mask3d_r_out", 1.0))
        invert  = bool(layer.get("mask3d_invert", False))

        tint_in  = (255, 110, 110) if invert else (110, 200, 255)
        tint_out = (190,  90,  90) if invert else ( 70, 130, 180)

        py5.no_fill()
        rings = [
            (r_in_w  * pix_per_world, tint_in,  2, 220),
            (r_out_w * pix_per_world, tint_out, 1, 150),
        ]
        for r_px, rgb, weight, alpha_ in rings:
            if r_px < 0.5:
                continue
            xpix = PREVIEW_X + (x_px + cosT * r_px) * sx
            ypix = PREVIEW_Y + (y_px + sinT * r_px) * sy
            py5.stroke(rgb[0], rgb[1], rgb[2], alpha_); py5.stroke_weight(weight)
            for i in range(n_pts - 1):
                py5.line(xpix[i], ypix[i], xpix[i + 1], ypix[i + 1])

        # Centre marker + label
        py5.no_stroke(); py5.fill(tint_in[0], tint_in[1], tint_in[2], 230)
        py5.circle(PREVIEW_X + x_px * sx, PREVIEW_Y + y_px * sy, 6)
        py5.fill(230, 235, 245); _tsz(10)
        py5.text(layer.get("name") or layer.get("type", ""),
                 PREVIEW_X + x_px * sx + 8, PREVIEW_Y + y_px * sy - 4)


def _draw_offsets_overlay():
    """Overlay walker pre/post normal-offset pairs (green dot -> red dot, line)."""
    pairs = state.get("debug_offsets") or []
    if not pairs:
        return
    cfg_w = scene_data["cfg"].W
    cfg_h = scene_data["cfg"].H
    sx = PREVIEW_W / cfg_w
    sy = PREVIEW_H / cfg_h
    n_max = 1500
    if len(pairs) > n_max:
        step = max(1, len(pairs) // n_max)
        pairs = pairs[::step]
    py5.stroke_weight(1)
    for orig, off in pairs:
        ox = PREVIEW_X + orig[0] * sx; oy = PREVIEW_Y + orig[1] * sy
        ex = PREVIEW_X + off[0]  * sx; ey = PREVIEW_Y + off[1]  * sy
        py5.stroke(120, 200, 255, 160); py5.line(ox, oy, ex, ey)
        py5.no_stroke()
        py5.fill(120, 220, 130, 220); py5.circle(ox, oy, 3)   # surface point
        py5.fill(255, 100, 120, 220); py5.circle(ex, ey, 3)   # offset point


def _draw_checkbox(field):
    # Schema-driven bool params get a label to the left of the box; the
    # layer-header "enabled" checkbox has no label (kind != "param").
    if field.get("kind") == "param" and field.get("label"):
        py5.fill(170, 180, 195); _tsz(11)
        py5.text(field["label"], field["x"] - LABEL_W + 2, field["y"] + 14)
    checked = bool(_read(field))
    py5.stroke(110, 120, 140); py5.stroke_weight(1)
    py5.fill(25, 29, 38)
    py5.rect(field["x"], field["y"], field["w"], field["h"])
    if checked:
        py5.stroke(120, 220, 150); py5.stroke_weight(2)
        py5.line(field["x"] + 4, field["y"] + 10, field["x"] + 9, field["y"] + 15)
        py5.line(field["x"] + 9, field["y"] + 15, field["x"] + 16, field["y"] + 4)
        py5.stroke_weight(1)


def _draw_field(field):
    # Group label (only on the first sub-field of an rgb row)
    if field.get("group_label"):
        py5.fill(195, 205, 220); _tsz(11)
        py5.text(field["group_label"], PANEL_X, field["y"] + 16)

    # Plain param label (left of the field box, 2-col grid)
    if field.get("kind") == "param":
        py5.fill(170, 180, 195); _tsz(11)
        py5.text(field["label"], field["x"] - LABEL_W + 2, field["y"] + 16)
    elif field.get("kind") == "rgb_sub":
        # short R/G/B letter immediately left of its field
        py5.fill(170, 180, 195); _tsz(11)
        py5.text(field["label"], field["x"] - RGB_SUB_LABEL_W + 2, field["y"] + 16)

    # Field box
    active = field.get("active", False)
    py5.stroke(220, 180, 90) if active else py5.stroke(70, 80, 98)
    py5.stroke_weight(1.5 if active else 1.0)
    py5.fill(31, 35, 45)
    py5.rect(field["x"], field["y"], field["w"], field["h"])

    # Value: typed buffer (orange) OR effective placeholder (dim)
    if active and field["edit_text"]:
        text = field["edit_text"]
        py5.fill(245, 220, 180)
    else:
        v = _read(field)
        text = _format(v, field["type"], field["fmt"])
        py5.fill(225 if not active else 120)
    _tsz(11)
    # crop overly long strings
    if py5.text_width(text) > field["w"] - 10:
        while text and py5.text_width(text + ".") > field["w"] - 10:
            text = text[:-1]
        text = text + ".."
    py5.text(text, field["x"] + 6, field["y"] + 16)

    # cycle hint for str-choices
    if field["type"] == "str" and field.get("choices") and not active:
        py5.fill(100, 110, 130); _tsz(9)
        py5.text("(cycle)", field["x"] + field["w"] + 4, field["y"] + 14)


# ---- mouse / key ---------------------------------------------------------
def mouse_pressed():
    if _hit(BTN_RENDER):
        _unfocus_all_commit()
        state["render_pending"] = True
        state["msg"] = "rendering..."
        return
    if _hit(BTN_SAVE_IMG):
        _unfocus_all_commit()
        do_save_image()
        return
    if _hit(BTN_SAVE_SCENE):
        _unfocus_all_commit()
        save_scene_file()
        return
    if _hit(BTN_ADD_SPLAT):
        _unfocus_all_commit()
        do_add_layer("splat")
        return
    if _hit(BTN_ADD_WALKS):
        _unfocus_all_commit()
        do_add_layer("walks")
        return
    if _hit(BTN_ADD_CURVE):
        _unfocus_all_commit()
        do_add_layer("curve")
        return

    # Chevron clicks -- toggle collapse for the matching layer.
    for c in chevrons:
        if _hit(c):
            _unfocus_all_commit()
            lyr = _layer(c["layer"])
            lyr["_ui_collapsed"] = not bool(lyr.get("_ui_collapsed", False))
            _rebuild_fields()
            return

    for f in fields:
        if _hit(f):
            _unfocus_all_commit()
            if f["type"] == "bool":
                _write(f, not bool(_read(f)))
                # Toggling enabled is composite-time only -- the layer cache
                # already holds the buffer, so this is a fast re-composite.
                state["render_pending"] = True
                state["msg"] = "compositing..."
            elif f["type"] == "str" and f.get("choices"):
                cur = _read(f)
                ch = f["choices"]
                try:
                    idx = ch.index(cur)
                except ValueError:
                    idx = -1
                _write(f, ch[(idx + 1) % len(ch)])
            else:
                f["active"] = True
                f["edit_text"] = ""
            return

    _unfocus_all_commit()


def key_pressed():
    active = next((f for f in fields if f.get("active")), None)
    if active:
        k = py5.key
        if k == py5.BACKSPACE:
            active["edit_text"] = active["edit_text"][:-1]
        elif k in (py5.ENTER, py5.RETURN):
            _commit(active); active["active"] = False
        elif k == py5.ESC:
            active["active"] = False; active["edit_text"] = ""
        elif k == py5.TAB:
            _commit(active); active["active"] = False
            idx = fields.index(active)
            # Tab to the next editable text field (skip checkboxes + cycle-only)
            for off in range(1, len(fields) + 1):
                nxt = fields[(idx + off) % len(fields)]
                if nxt["type"] in ("float", "int", "str") and not (
                        nxt["type"] == "str" and nxt.get("choices")):
                    nxt["active"] = True
                    nxt["edit_text"] = ""
                    break
        else:
            ch = str(k)
            if len(ch) == 1:
                if active["type"] == "str":
                    if 32 <= ord(ch) <= 126:
                        active["edit_text"] += ch
                elif ch.isdigit() or ch in ".-+e":
                    active["edit_text"] += ch
        return

    if py5.key == "r":
        state["render_pending"] = True
        state["msg"] = "rendering..."
    elif py5.key == "s":
        save_scene_file()


def _commit(field):
    txt = field.get("edit_text", "").strip()
    if not txt:
        return
    try:
        if field["type"] == "int":
            v = int(float(txt))
        elif field["type"] == "float":
            v = float(txt)
        else:
            v = txt
        if field.get("choices") and v not in field["choices"]:
            return
        _write(field, v)
    except (ValueError, TypeError):
        pass


def _unfocus_all_commit():
    for f in fields:
        if f.get("active"):
            _commit(f)
            f["active"] = False


# ---- actions -------------------------------------------------------------
def do_render():
    """Run the layered render with stdout captured into the in-window log."""
    _log_append(f"--- render @ {time.strftime('%H:%M:%S')} ---")
    t0 = time.time()
    out_path = None
    buf = io.StringIO()

    import render_layers as _rl
    _rl.DEBUG_CAPTURE_POINTS  = False
    _rl.DEBUG_POINTS          = []
    _rl.DEBUG_CAPTURE_OFFSETS = False
    _rl.DEBUG_OFFSETS         = []

    try:
        with contextlib.redirect_stdout(buf):
            _, out_path, _ = render_composition(
                SCENE_REF, composition=composition,
                write=True, stamp_label=True,
                scene_data=scene_data, layer_cache=layer_cache)
    except Exception as exc:
        _rl.DEBUG_CAPTURE_POINTS  = False
        _rl.DEBUG_CAPTURE_OFFSETS = False
        _log_append(f"ERROR: {exc}")
        _log_append(traceback.format_exc())
    captured = buf.getvalue()
    _log_append(captured)
    if captured:
        print(captured, end="")
    if out_path:
        try:
            state["render_image"] = py5.load_image(out_path)
            state["_last_out_path"] = out_path
        except Exception as exc:
            _log_append(f"(load_image failed: {exc})")
    _rl.DEBUG_CAPTURE_POINTS  = False
    _rl.DEBUG_CAPTURE_OFFSETS = False
    dt = time.time() - t0
    state["msg"] = (f"render done in {dt:.1f}s  ({out_path})"
                    if out_path else f"render FAILED ({dt:.1f}s)")
    _log_append(state["msg"])


def do_save_image():
    """Copy the current rendered image to a timestamped file in images/."""
    src = state.get("_last_out_path")
    if not src or not os.path.exists(src):
        state["msg"] = "no render yet -- press RENDER first"
        return
    import shutil
    base = os.path.splitext(os.path.basename(src))[0]
    ts = time.strftime("%Y%m%d_%H%M%S")
    dst = f"images/{base}_{ts}.png"
    shutil.copy(src, dst)
    _log_append(f"saved image -> {dst}")
    state["msg"] = f"saved image -> {dst}"


def _unique_layer_name(prefix):
    existing = {l.get("name") for l in composition.get("layers", [])}
    if prefix not in existing:
        return prefix
    i = 2
    while f"{prefix}_{i}" in existing:
        i += 1
    return f"{prefix}_{i}"


def do_add_layer(kind):
    """Append a new layer to the composition and rebuild the UI."""
    if kind == "splat":
        new_layer = deepcopy(ADD_SPLAT_TEMPLATE)
        new_layer["name"] = _unique_layer_name("splat")
    elif kind == "walks":
        new_layer = deepcopy(ADD_WALKS_TEMPLATE)
        new_layer["name"] = _unique_layer_name("walks")
    elif kind == "curve":
        new_layer = deepcopy(ADD_CURVE_TEMPLATE)
        new_layer["name"] = _unique_layer_name("curve")
    else:
        return
    composition["layers"].append(new_layer)
    _hydrate_layer(new_layer)
    _rebuild_fields()
    _log_append(f"+ added layer: {new_layer['name']} ({new_layer['type']})")
    state["msg"] = f"added {new_layer['name']}"


def save_scene_file():
    try:
        with open(SCENE_JSON) as f:
            data = json.load(f)
    except Exception:
        data = dict(name=SCENE_REF)
    data["composition"] = deepcopy(composition)
    for layer in data["composition"].get("layers", []):
        layer.pop("_ui_y", None)
        layer.pop("_ui_h", None)
    with open(SCENE_JSON, "w") as f:
        json.dump(data, f, indent=2)
        f.write("\n")
    state["msg"] = f"saved {SCENE_JSON}"


py5.run_sketch()
