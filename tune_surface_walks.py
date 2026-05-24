"""Interactive tuner for the surface-walk NPR.

Adjust walker, stroke, and camera knobs with text-input fields; press
RENDER (or 'r') to re-draw. Press RE-PROJECT when you change a camera
field -- that re-renders the splat reference image, re-projects all
the Gaussians, and refreshes the saliency map (slow: 20-40 s for RedHead).

Defaults start at the `bold_long` line-stroke variant. Camera defaults
are read from the active scene (render_redhead.py).

USAGE -- run as a STANDALONE script (macOS GL/window constraint):

  py5 tune_surface_walks.py

CONTROLS:
  click a number field   field clears; type the new value
                         (the previous value is shown dimly as a placeholder)
  Enter                  commit + unfocus
  Tab                    commit + jump to next field
  Esc                    cancel edit (revert)
  click str-choice field cycle through allowed values (mode / placement)
  RENDER  / 'r'          render the stroke pass with current state
  RE-PROJECT             re-render reference + re-project at current camera
  SAVE COPY / 's'        copy current render to timestamped filename
"""
import os
os.environ.setdefault("JAVA_HOME", "/usr/local/opt/openjdk@17")

import shutil
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from PIL import Image

import py5

from gsplat import (
    load_3dgs_ply, decode_3dgs, make_camera, project_perspective, cull, splat,
)
from npr_utils import (
    stamp, load_ref_image, compute_saliency,
    add_line, add_splat_stroke, walk_step_3d, make_noise_direction_field,
    PAPER_COLOR,
)
from scene_io import load_scene

# Switch scenes via env var (or edit this default):
#   SCENE_NAME=siyun py5 tune_surface_walks.py
#   SCENE_FILE=scenes/siyun_walks.json py5 tune_surface_walks.py
SCENE_REF = os.environ.get("SCENE_FILE") or os.environ.get("SCENE_NAME", "redhead")
cfg = load_scene(SCENE_REF)
SCENE_NAME = cfg.SCENE_NAME
print(f"[tune-walks] scene = {SCENE_NAME}  ({cfg.PLY})")


# ============ window layout ==============================================
W, H = 1240, 1000
RENDER_W, RENDER_H = cfg.W, cfg.H
THUMB_X, THUMB_Y = 10, 50
THUMB_W = 660
THUMB_H = int(RENDER_H * THUMB_W / RENDER_W)
PANEL_X = THUMB_W + 30

INPUT_H, INPUT_GAP = 26, 4
INPUT_LABEL_W, INPUT_BOX_W = 130, 200
INPUTS_Y_START = 60
BTN_RENDER = dict(x=0, y=0, w=0, h=0)
BTN_REPROJECT = dict(x=0, y=0, w=0, h=0)
BTN_SAVE = dict(x=0, y=0, w=0, h=0)


# ============ state ======================================================
state = dict(
    # walker
    N_WALKERS=200,
    STEPS=40,
    STEP_RADIUS_PX=22.0,
    FORWARD_BIAS=5.0,
    DIRECTION_MODE="momentum",
    GLOBAL_DIR_DEG=0.0,
    NOISE_SCALE=90.0,
    STROKE_ALPHA=0.70,
    INK_DARKEN=0.00,
    PLACEMENT="saliency",
    SEED=17,
    # stroke
    STROKE_MODE="line",
    STROKE_WIDTH=1.0,
    SPLAT_SCALE=0.35,
    SPLAT_ALPHA_SCALE=0.35,
    SPLAT_MIN_SIGMA=0.10,
    SPLAT_MAX_SIGMA=1.20,
    N_STAMPS=5,
    # camera (init from cfg; change + press RE-PROJECT to apply)
    ELEV_DEG=cfg.ELEV_DEG,
    AZIM_DEG=cfg.AZIM_DEG,
    DISTANCE_K=cfg.DISTANCE_K,
    FOV_DEG=cfg.FOV_DEG,
    HEAD_BIAS_X=cfg.HEAD_BIAS_X,
    HEAD_BIAS_Y=cfg.HEAD_BIAS_Y,
    # ui
    render_msg="",
    render_image=None,
    render_pending=False,
    reproject_pending=False,
)

_WALK_KEY_MAP = {
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
}
for raw_key, value in cfg.SURFACE_WALKS.items():
    key = _WALK_KEY_MAP.get(raw_key, raw_key.upper())
    if key in state:
        state[key] = value

inputs = [
    # ---- walker ----
    dict(key="N_WALKERS",      label="N walkers",    fmt="{:.0f}", type="int"),
    dict(key="STEPS",          label="steps/walker", fmt="{:.0f}", type="int"),
    dict(key="STEP_RADIUS_PX", label="step radius",  fmt="{:.0f}", type="float"),
    dict(key="FORWARD_BIAS",   label="forward bias", fmt="{:.1f}", type="float"),
    dict(key="DIRECTION_MODE",  label="direction",    fmt="{:s}",   type="str", choices=["momentum", "global", "noise"]),
    dict(key="GLOBAL_DIR_DEG",  label="dir deg",      fmt="{:.0f}", type="float"),
    dict(key="NOISE_SCALE",     label="noise scale",  fmt="{:.0f}", type="float"),
    dict(key="STROKE_ALPHA",   label="alpha",        fmt="{:.2f}", type="float"),
    dict(key="INK_DARKEN",     label="ink darken",   fmt="{:.2f}", type="float"),
    dict(key="PLACEMENT",      label="placement",    fmt="{:s}",   type="str", choices=["saliency", "uniform"]),
    # ---- stroke ----
    dict(key="STROKE_MODE",    label="stroke mode",  fmt="{:s}",   type="str", choices=["line", "splat"]),
    dict(key="STROKE_WIDTH",   label="line width",   fmt="{:.2f}", type="float"),
    dict(key="SPLAT_SCALE",    label="splat scale",  fmt="{:.2f}", type="float"),
    dict(key="SPLAT_ALPHA_SCALE", label="splat alpha x", fmt="{:.2f}", type="float"),
    dict(key="SPLAT_MIN_SIGMA", label="splat min px", fmt="{:.2f}", type="float"),
    dict(key="SPLAT_MAX_SIGMA", label="splat max px", fmt="{:.2f}", type="float"),
    dict(key="N_STAMPS",       label="stamps/seg",   fmt="{:.0f}", type="int"),
    # ---- camera (press RE-PROJECT to apply) ----
    dict(key="ELEV_DEG",       label="elev deg",     fmt="{:.1f}", type="float"),
    dict(key="AZIM_DEG",       label="azim deg",     fmt="{:.1f}", type="float"),
    dict(key="DISTANCE_K",     label="dist k",       fmt="{:.2f}", type="float"),
    dict(key="FOV_DEG",        label="fov deg",      fmt="{:.1f}", type="float"),
    dict(key="HEAD_BIAS_X",    label="bias x",       fmt="{:.2f}", type="float"),
    dict(key="HEAD_BIAS_Y",    label="bias y",       fmt="{:.2f}", type="float"),
    # ---- misc ----
    dict(key="SEED",           label="seed",         fmt="{:.0f}", type="int"),
]
CAMERA_KEYS = {"ELEV_DEG", "AZIM_DEG", "DISTANCE_K", "FOV_DEG",
               "HEAD_BIAS_X", "HEAD_BIAS_Y"}


# ============ load once + (re)project on demand =========================
print(f"[tune-walks] loading {cfg.PLY} ...")
data = load_3dgs_ply(cfg.PLY)
G = decode_3dgs(data)
SCENE_BASE = os.path.splitext(os.path.basename(cfg.PLY))[0].lower()
TUNE_REF_PATH = f"images/{SCENE_BASE}_tune_ref.png"
TUNE_OUT_PATH = f"images/{SCENE_BASE}_walks_tune.png"
print(f"[tune-walks] {len(data)} Gaussians; {len(G['xyz'])} positions")

# These are filled by reproject(); start as None
pts = xyz_kept = ops = cov2d_kept = ref_img = saliency = tree3d = None
last_reproj_camera = None     # (elev, azim, dist_k, fov, bx, by) tuple


def _camera_tuple():
    return (state["ELEV_DEG"], state["AZIM_DEG"], state["DISTANCE_K"],
            state["FOV_DEG"], state["HEAD_BIAS_X"], state["HEAD_BIAS_Y"])


def reproject(write_reference=True):
    """Project Gaussians at current camera state; optionally re-render
    the splat reference image so saliency / colour lookups stay aligned.

    `write_reference=False` skips the slow splat pass -- useful at startup
    if the cfg's own images/<scene>_render.png is up to date.
    """
    global pts, xyz_kept, ops, cov2d_kept, ref_img, saliency, tree3d, last_reproj_camera
    print(f"[reproject] camera {_camera_tuple()}  (write_ref={write_reference})")
    t0 = time.time()

    ysign = +1.0 if cfg.SCENE_UP_FLIP else -1.0
    center = np.median(G["xyz"], axis=0)
    center[0] += state["HEAD_BIAS_X"]
    center[1] += state["HEAD_BIAS_Y"]
    radii = np.linalg.norm(G["xyz"] - center, axis=1)
    extent = np.percentile(radii, 90) * 2.0
    Rcam = make_camera(state["ELEV_DEG"], state["AZIM_DEG"])
    cam_xyz = (G["xyz"] - center) @ Rcam.T
    cov_cam = np.einsum("ij,njk,lk->nil", Rcam, G["cov3"], Rcam)
    focal = RENDER_W / (2.0 * np.tan(np.radians(state["FOV_DEG"]) / 2.0))
    distance = extent * state["DISTANCE_K"]
    mean2d, cov2d, depths, valid_z = project_perspective(
        cam_xyz, cov_cam, focal, distance, RENDER_W, RENDER_H, ysign)
    keep = cull(mean2d, cov2d, G["opacities"], valid_z, RENDER_W, RENDER_H,
                sub_pixel=0.0)

    pts = mean2d[keep]
    xyz_kept = G["xyz"][keep]
    ops = G["opacities"][keep]
    cov2d_kept = cov2d[keep]
    tree3d = cKDTree(xyz_kept)

    ref_source = cfg.OUT          # default: the canonical splat render
    if write_reference:
        order = keep[np.argsort(-depths[keep])]
        print(f"[reproject] rendering reference splat ({len(order)} visible)...")
        img = splat(RENDER_W, RENDER_H, mean2d, cov2d,
                    G["colors"], G["opacities"], order, verbose=False)
        plt.imsave(TUNE_REF_PATH, np.clip(img, 0, 1))
        ref_source = TUNE_REF_PATH

    ref_img = load_ref_image(ref_source, RENDER_W, RENDER_H)
    saliency = compute_saliency(ref_img)
    last_reproj_camera = _camera_tuple()
    dt = time.time() - t0
    print(f"[reproject] {len(pts)} visible  ({dt:.1f}s)")
    state["render_msg"] = f"reprojected ({dt:.1f}s)"


# initial projection -- reuse cfg.OUT (matches cfg's camera defaults)
reproject(write_reference=False)
print("[tune-walks] ready")


# ============ render core ================================================
def _placement_weights():
    if state["PLACEMENT"] == "saliency":
        ix = np.clip(pts[:, 0].astype(int), 0, saliency.shape[1] - 1)
        iy = np.clip(pts[:, 1].astype(int), 0, saliency.shape[0] - 1)
        s = saliency[iy, ix]
        return ops * (s + 0.04)
    return ops


def _seg_color(p0, p1, ink):
    mx = int(np.clip((p0[0] + p1[0]) / 2, 0, RENDER_W - 1))
    my = int(np.clip((p0[1] + p1[1]) / 2, 0, RENDER_H - 1))
    return ref_img[my, mx] * ink


def do_render():
    n_walkers = int(state["N_WALKERS"])
    n_steps = int(state["STEPS"])
    R = float(state["STEP_RADIUS_PX"])
    fb = float(state["FORWARD_BIAS"])
    direction_mode = state["DIRECTION_MODE"]
    global_deg = float(state["GLOBAL_DIR_DEG"])
    noise_scale = float(state["NOISE_SCALE"])
    alpha = float(state["STROKE_ALPHA"])
    ink = float(state["INK_DARKEN"])
    mode = state["STROKE_MODE"]
    width = float(state["STROKE_WIDTH"])
    splat_scale = float(state["SPLAT_SCALE"])
    splat_alpha_scale = float(state["SPLAT_ALPHA_SCALE"])
    splat_min_sigma = float(state["SPLAT_MIN_SIGMA"])
    splat_max_sigma = float(state["SPLAT_MAX_SIGMA"])
    n_stamps = int(state["N_STAMPS"])

    stroke_desc = (f"splat x{splat_scale:.1f} n={n_stamps}"
                   if mode == "splat" else f"line w={width:.2f}")
    if mode == "splat":
        stroke_desc = (f"splat x{splat_scale:.2f} ax{splat_alpha_scale:.2f} "
                       f"sigma={splat_min_sigma:.2f}-{splat_max_sigma:.2f} "
                       f"n={n_stamps}")
    dir_desc = direction_mode
    if direction_mode == "global":
        dir_desc = f"global {global_deg:.0f}deg"
    elif direction_mode == "noise":
        dir_desc = f"noise scale={noise_scale:.0f}"

    print(f"[walk] N={n_walkers} steps={n_steps} R={R:.0f} bias={fb:.1f} "
          f"dir={dir_desc} a={alpha:.2f} ink={ink:.2f} {state['PLACEMENT']} "
          f"{stroke_desc} seed={state['SEED']}")
    if last_reproj_camera != _camera_tuple():
        print(f"  NOTE: camera changed since last reproject "
              f"(last={last_reproj_camera}). Press RE-PROJECT to apply.")

    t0 = time.time()
    rng = np.random.default_rng(int(state["SEED"]))
    w = _placement_weights()
    w = w / w.sum()
    canvas = np.tile(PAPER_COLOR, (RENDER_H, RENDER_W, 1)).astype(np.float64)
    drawn = 0
    direction_field = None
    global_dir = None
    if direction_mode == "global":
        theta = np.radians(global_deg)
        global_dir = np.array([np.cos(theta), np.sin(theta)], dtype=np.float64)
    elif direction_mode == "noise":
        direction_field = make_noise_direction_field(
            RENDER_W, RENDER_H, sigma=noise_scale, seed=int(state["SEED"]))

    for _ in range(n_walkers):
        cur = int(rng.choice(len(pts), p=w))
        prev_dir = None
        for _ in range(n_steps):
            result = walk_step_3d(cur, prev_dir, tree3d, xyz_kept, pts, R, fb,
                                  rng, direction_field=direction_field,
                                  global_dir=global_dir)
            if result is None:
                break
            nxt, prev_dir = result
            p0, p1 = pts[cur], pts[nxt]
            color = _seg_color(p0, p1, ink)
            if mode == "splat":
                cov_avg = 0.5 * (cov2d_kept[cur] + cov2d_kept[nxt])
                add_splat_stroke(canvas, p0, p1, cov_avg,
                                  color, alpha * splat_alpha_scale,
                                  RENDER_W, RENDER_H,
                                  n_stamps=n_stamps, scale=splat_scale,
                                  min_sigma_px=splat_min_sigma,
                                  max_sigma_px=splat_max_sigma)
            else:
                add_line(canvas, p0[0], p0[1], p1[0], p1[1],
                         color, alpha, RENDER_W, RENDER_H, width=width)
            cur = nxt
            drawn += 1

    label = (
        f"N={n_walkers}  steps={n_steps}  R={R:.0f}px  bias={fb:.1f}  "
        f"dir={dir_desc}  a={alpha:.2f}  ink={ink:.2f}  {state['PLACEMENT']}  "
        f"{stroke_desc}  seed={state['SEED']}"
    )
    canvas = stamp(canvas, label, RENDER_W, RENDER_H)
    plt.imsave(TUNE_OUT_PATH, np.clip(canvas, 0, 1))
    dt = time.time() - t0
    print(f"  -> {TUNE_OUT_PATH}  ({drawn} segs, {dt:.1f}s)")
    state["render_msg"] = f"saved {TUNE_OUT_PATH}  ({drawn} segs, {dt:.1f}s)"
    try:
        state["render_image"] = py5.load_image(TUNE_OUT_PATH)
    except Exception as e:
        print(f"  (load_image failed: {e})")


def do_save_copy():
    if not os.path.exists(TUNE_OUT_PATH):
        state["render_msg"] = "nothing to save yet -- press R first"
        return
    ts = time.strftime("%H%M%S")
    dst = f"images/{SCENE_BASE}_walks_{ts}.png"
    shutil.copy(TUNE_OUT_PATH, dst)
    print(f"saved copy -> {dst}")
    state["render_msg"] = f"saved copy {dst}"


# ============ py5 callbacks ==============================================
def setup():
    py5.size(W, H, py5.P2D)
    py5.frame_rate(60)
    py5.text_size(12)

    for i, inp in enumerate(inputs):
        inp["x"] = PANEL_X + INPUT_LABEL_W
        inp["y"] = INPUTS_Y_START + i * (INPUT_H + INPUT_GAP)
        inp["w"] = INPUT_BOX_W
        inp["h"] = INPUT_H
        inp["active"] = False
        inp["edit_text"] = ""

    last_y = INPUTS_Y_START + len(inputs) * (INPUT_H + INPUT_GAP)
    btn_w = INPUT_LABEL_W + INPUT_BOX_W

    BTN_RENDER["x"] = PANEL_X
    BTN_RENDER["y"] = last_y + 14
    BTN_RENDER["w"] = btn_w
    BTN_RENDER["h"] = 32
    BTN_REPROJECT["x"] = PANEL_X
    BTN_REPROJECT["y"] = BTN_RENDER["y"] + BTN_RENDER["h"] + 6
    BTN_REPROJECT["w"] = btn_w
    BTN_REPROJECT["h"] = 28
    BTN_SAVE["x"] = PANEL_X
    BTN_SAVE["y"] = BTN_REPROJECT["y"] + BTN_REPROJECT["h"] + 6
    BTN_SAVE["w"] = btn_w
    BTN_SAVE["h"] = 28


def draw():
    # two-pass triggers so the "rendering..." status paints first
    if state["reproject_pending"]:
        state["reproject_pending"] = False
        reproject(write_reference=True)
    if state["render_pending"]:
        state["render_pending"] = False
        do_render()

    py5.background(18, 20, 26)

    # ---- thumbnail ----
    py5.fill(220)
    py5.text_size(12)
    py5.text("tune_surface_walks.py", THUMB_X, 22)
    py5.no_fill()
    py5.stroke(70, 75, 90)
    py5.stroke_weight(1)
    py5.rect(THUMB_X, THUMB_Y, THUMB_W, THUMB_H)
    if state["render_image"] is not None:
        py5.image(state["render_image"], THUMB_X, THUMB_Y, THUMB_W, THUMB_H)
    else:
        py5.fill(140, 140, 160, 180)
        py5.text("(press RENDER / 'r')", THUMB_X + 240, THUMB_Y + THUMB_H / 2)

    py5.stroke(60, 65, 75)
    py5.line(PANEL_X - 12, 0, PANEL_X - 12, H)

    # ---- inputs ----
    py5.text_size(12)
    for inp in inputs:
        # camera fields get a slightly different label tint as a hint
        is_camera = inp["key"] in CAMERA_KEYS
        py5.fill(150, 200, 220) if is_camera else py5.fill(220)
        py5.text(inp["label"], PANEL_X, inp["y"] + 18)

        if inp["active"]:
            py5.stroke(220, 180, 90); py5.stroke_weight(1.5)
            py5.fill(35, 38, 48)
        else:
            py5.stroke(70, 75, 90); py5.stroke_weight(1.0)
            py5.fill(25, 28, 36)
        py5.rect(inp["x"], inp["y"], inp["w"], inp["h"])

        # display logic:
        # active + non-empty buffer -> show edit_text (orange)
        # active + empty buffer     -> show current state value as dim placeholder
        # not active                -> show current state value (normal)
        if inp["active"]:
            if inp["edit_text"]:
                display = inp["edit_text"]
                py5.fill(245, 220, 180)
            else:
                v = state[inp["key"]]
                display = (inp["fmt"].format(v) if inp["type"] != "str" else str(v))
                py5.fill(110, 115, 130)
        else:
            v = state[inp["key"]]
            display = (inp["fmt"].format(v) if inp["type"] != "str" else str(v))
            py5.fill(225)
        py5.text(display, inp["x"] + 8, inp["y"] + 18)

        if inp["active"] and inp["edit_text"]:
            cw = py5.text_width(display)
            py5.stroke(245, 220, 180); py5.stroke_weight(1)
            py5.line(inp["x"] + 8 + cw + 1, inp["y"] + 4,
                     inp["x"] + 8 + cw + 1, inp["y"] + inp["h"] - 4)

        if "choices" in inp and not inp["active"]:
            py5.fill(140, 145, 160)
            py5.text_size(10)
            py5.text("(click to cycle)", inp["x"] + inp["w"] + 8, inp["y"] + 18)
            py5.text_size(12)

    _draw_button(BTN_RENDER, "RENDER (r)", (170, 70, 70), (200, 110, 110))
    _draw_button(BTN_REPROJECT, "RE-PROJECT", (100, 80, 150), (130, 110, 190))
    _draw_button(BTN_SAVE, "SAVE COPY (s)", (70, 110, 170), (110, 150, 200))

    if state["render_msg"]:
        py5.fill(150, 230, 150)
        py5.text_size(11)
        py5.text(state["render_msg"], PANEL_X, BTN_SAVE["y"] + BTN_SAVE["h"] + 22)
    py5.fill(140, 145, 160)
    py5.text_size(10)
    py5.text("[r] render   [s] save   [Tab] next   [Enter] commit   [Esc] cancel    "
             "camera changes need RE-PROJECT",
             10, H - 10)


def _draw_button(btn, label, base_rgb, hover_rgb):
    is_hover = (btn["x"] <= py5.mouse_x <= btn["x"] + btn["w"]
                and btn["y"] <= py5.mouse_y <= btn["y"] + btn["h"])
    rgb = hover_rgb if is_hover else base_rgb
    py5.stroke_weight(1)
    py5.stroke(*[min(c + 30, 255) for c in rgb])
    py5.fill(*rgb)
    py5.rect(btn["x"], btn["y"], btn["w"], btn["h"])
    py5.no_stroke()
    py5.fill(255)
    py5.text_size(13)
    tw = py5.text_width(label)
    py5.text(label, btn["x"] + (btn["w"] - tw) / 2, btn["y"] + btn["h"] / 2 + 5)
    py5.text_size(12)


# ---- mouse / key ----
def _hit(rect):
    return (rect["x"] <= py5.mouse_x <= rect["x"] + rect["w"]
            and rect["y"] <= py5.mouse_y <= rect["y"] + rect["h"])


def mouse_pressed():
    if _hit(BTN_RENDER):
        _unfocus_all_commit()
        state["render_msg"] = "rendering..."
        state["render_pending"] = True
        return
    if _hit(BTN_REPROJECT):
        _unfocus_all_commit()
        state["render_msg"] = "re-projecting (this is slow -- ~30 s)..."
        state["reproject_pending"] = True
        return
    if _hit(BTN_SAVE):
        _unfocus_all_commit()
        do_save_copy()
        return

    for inp in inputs:
        if _hit(inp):
            _unfocus_all_commit()
            if "choices" in inp:
                # click cycles through allowed values
                cur = state[inp["key"]]
                try:
                    i = inp["choices"].index(cur)
                except ValueError:
                    i = -1
                state[inp["key"]] = inp["choices"][(i + 1) % len(inp["choices"])]
            else:
                # CLICK-TO-CLEAR: empty buffer so typed digits don't append
                # to the existing value. Current value still shown as dim
                # placeholder until first keystroke.
                inp["active"] = True
                inp["edit_text"] = ""
            return

    _unfocus_all_commit()


def key_pressed():
    active = next((inp for inp in inputs if inp["active"]), None)
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
            i = inputs.index(active)
            nxt = inputs[(i + 1) % len(inputs)]
            nxt["active"] = True
            nxt["edit_text"] = ""   # same click-to-clear behaviour for Tab
        else:
            try:
                ch = str(k)
                if len(ch) == 1:
                    if active["type"] == "str":
                        if 32 <= ord(ch) <= 126:
                            active["edit_text"] += ch
                    elif ch.isdigit() or ch in ".-+e":
                        active["edit_text"] += ch
            except Exception:
                pass
        return

    if py5.key == 'r':
        state["render_msg"] = "rendering..."
        state["render_pending"] = True
    elif py5.key == 's':
        do_save_copy()


def _commit(inp):
    txt = inp["edit_text"].strip()
    if not txt:
        return              # empty buffer = no change (keep previous value)
    try:
        if inp["type"] == "int":
            v = int(float(txt))
        elif inp["type"] == "float":
            v = float(txt)
        else:
            v = txt
        if "choices" in inp and v not in inp["choices"]:
            return
        state[inp["key"]] = v
    except (ValueError, TypeError):
        pass


def _unfocus_all_commit():
    for inp in inputs:
        if inp["active"]:
            _commit(inp)
            inp["active"] = False


py5.run_sketch()
