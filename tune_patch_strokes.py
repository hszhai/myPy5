"""Interactive tuner for the patch-sampling NPR.

Open a py5 window with text-input fields for every patch-stroke knob,
press RENDER (or 'r') to re-render with the current values, and the
result appears in the left panel without leaving the tool.

Run as a STANDALONE script (the macOS GL/window limitation again -- not
a notebook):

  py5 tune_patch_strokes.py

Scene config is whatever render_redhead.py / render_siyun.py / ... is
imported as `cfg` below; switch the import to retarget.

Controls:
  click a field         start editing
  Tab                   commit + move to next
  Enter                 commit + unfocus
  Esc                   cancel edit
  click str-choice fld  cycle through allowed values (mode / placement)
  RENDER / 'r'          run the patch-stroke render with current state
  SAVE COPY / 's'       copy current render to a timestamped filename
"""
import os
os.environ.setdefault("JAVA_HOME", "/usr/local/opt/openjdk@17")

import sys
import shutil
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from PIL import Image

import py5

from gsplat import load_3dgs_ply, decode_3dgs, make_camera, project_perspective, cull
from npr_utils import stamp, load_ref_image, compute_saliency, add_line, PAPER_COLOR
from scene_io import load_scene

SCENE_NAME = os.environ.get("SCENE_NAME", "redhead")
cfg = load_scene(SCENE_NAME)


# ============ window layout ==============================================
W, H = 1200, 1000
RENDER_W, RENDER_H = 1080, 1440              # render at the scene's native res
THUMB_X, THUMB_Y = 10, 50
THUMB_W = 720
THUMB_H = int(RENDER_H * THUMB_W / RENDER_W)  # preserve 3:4 -> 960
PANEL_X = THUMB_W + 30

INPUT_H, INPUT_GAP = 28, 6
INPUT_LABEL_W, INPUT_BOX_W = 120, 200
INPUTS_Y_START = 60
BTN_RENDER = dict(x=0, y=0, w=0, h=0)
BTN_SAVE = dict(x=0, y=0, w=0, h=0)


# ============ initial state ==============================================
state = dict(
    N_PATCHES=2000,
    RADIUS_PX=30.0,
    LINKS_PER_PATCH=3,
    MODE="path",                    # "pairs" or "path"
    STROKE_ALPHA=0.45,
    INK_DARKEN=0.30,
    PLACEMENT="saliency",           # "saliency" or "uniform"
    SEED=11,
    render_msg="",
    render_image=None,
    render_pending=False,           # set on render-click, picked up next frame
)

inputs = [
    dict(key="N_PATCHES",       label="N patches",   fmt="{:.0f}", type="int"),
    dict(key="RADIUS_PX",       label="radius px",   fmt="{:.0f}", type="float"),
    dict(key="LINKS_PER_PATCH", label="links/patch", fmt="{:.0f}", type="int"),
    dict(key="MODE",            label="mode",        fmt="{:s}",   type="str", choices=["pairs", "path"]),
    dict(key="STROKE_ALPHA",    label="alpha",       fmt="{:.2f}", type="float"),
    dict(key="INK_DARKEN",      label="ink darken",  fmt="{:.2f}", type="float"),
    dict(key="PLACEMENT",       label="placement",   fmt="{:s}",   type="str", choices=["saliency", "uniform"]),
    dict(key="SEED",            label="seed",        fmt="{:.0f}", type="int"),
]


# ============ load + project (once) ======================================
print(f"[tune] loading {cfg.PLY} ...")
data = load_3dgs_ply(cfg.PLY)
G = decode_3dgs(data)
ysign = +1.0 if cfg.SCENE_UP_FLIP else -1.0
center = np.median(G["xyz"], axis=0)
center[0] += cfg.HEAD_BIAS_X
center[1] += cfg.HEAD_BIAS_Y
radii = np.linalg.norm(G["xyz"] - center, axis=1)
extent = np.percentile(radii, 90) * 2.0
Rcam = make_camera(cfg.ELEV_DEG, cfg.AZIM_DEG)
cam_xyz = (G["xyz"] - center) @ Rcam.T
cov_cam = np.einsum("ij,njk,lk->nil", Rcam, G["cov3"], Rcam)
focal = RENDER_W / (2.0 * np.tan(np.radians(cfg.FOV_DEG) / 2.0))
distance = extent * cfg.DISTANCE_K
mean2d, cov2d, depths, valid_z = project_perspective(
    cam_xyz, cov_cam, focal, distance, RENDER_W, RENDER_H, ysign)
keep = cull(mean2d, cov2d, G["opacities"], valid_z, RENDER_W, RENDER_H, sub_pixel=0.0)
pts = mean2d[keep]
ops = G["opacities"][keep]
print(f"[tune] projected -> {len(pts)} visible")

ref_img = load_ref_image(cfg.OUT, RENDER_W, RENDER_H)
saliency = compute_saliency(ref_img)
print("[tune] ready")

SCENE_BASE = os.path.splitext(os.path.basename(cfg.PLY))[0].lower()
TUNE_PATH = f"images/{SCENE_BASE}_patches_tune.png"


# ============ patch-stroke render core ===================================
def _placement_weights():
    if state["PLACEMENT"] == "saliency":
        ix = np.clip(pts[:, 0].astype(int), 0, saliency.shape[1] - 1)
        iy = np.clip(pts[:, 1].astype(int), 0, saliency.shape[0] - 1)
        s = saliency[iy, ix]
        return ops * (s + 0.04)
    return ops  # uniform


def _seg_color(p0, p1, ink):
    mx = int(np.clip((p0[0] + p1[0]) / 2, 0, RENDER_W - 1))
    my = int(np.clip((p0[1] + p1[1]) / 2, 0, RENDER_H - 1))
    return ref_img[my, mx] * ink


def do_render():
    n_p = int(state["N_PATCHES"])
    R = float(state["RADIUS_PX"])
    n_per = int(state["LINKS_PER_PATCH"])
    mode = state["MODE"]
    alpha = float(state["STROKE_ALPHA"])
    ink = float(state["INK_DARKEN"])
    print(f"[tune] render N={n_p} R={R:.0f} L={n_per} {mode} a={alpha:.2f} "
          f"ink={ink:.2f} {state['PLACEMENT']} seed={state['SEED']}")
    t0 = time.time()

    rng = np.random.default_rng(int(state["SEED"]))
    w = _placement_weights()
    w = w / w.sum()
    tree = cKDTree(pts)
    canvas = np.tile(PAPER_COLOR, (RENDER_H, RENDER_W, 1)).astype(np.float64)

    for _ in range(n_p):
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
                         _seg_color(p0, p1, ink), alpha, RENDER_W, RENDER_H)
        else:  # path
            n_pts = min(n_per + 1, len(idx))
            picks = rng.choice(len(patch), n_pts, replace=False)
            for k in range(n_pts - 1):
                p0, p1 = patch[picks[k]], patch[picks[k + 1]]
                add_line(canvas, p0[0], p0[1], p1[0], p1[1],
                         _seg_color(p0, p1, ink), alpha, RENDER_W, RENDER_H)

    label = (
        f"N={n_p}  R={R:.0f}px  L={n_per}  {mode}  "
        f"a={alpha:.2f}  ink={ink:.2f}  {state['PLACEMENT']}  "
        f"seed={state['SEED']}"
    )
    canvas = stamp(canvas, label, RENDER_W, RENDER_H)
    plt.imsave(TUNE_PATH, np.clip(canvas, 0, 1))
    dt = time.time() - t0
    print(f"  -> {TUNE_PATH}  ({dt:.1f}s)")
    state["render_msg"] = f"saved {TUNE_PATH}  ({dt:.1f}s)"
    try:
        state["render_image"] = py5.load_image(TUNE_PATH)
    except Exception as e:
        print(f"  (load_image failed: {e})")


def do_save_copy():
    if not os.path.exists(TUNE_PATH):
        state["render_msg"] = "nothing to save yet -- press R first"
        return
    ts = time.strftime("%H%M%S")
    dst = f"images/{SCENE_BASE}_patches_{ts}.png"
    shutil.copy(TUNE_PATH, dst)
    print(f"saved copy -> {dst}")
    state["render_msg"] = f"saved copy {dst}"


# ============ py5 callbacks ==============================================
def setup():
    py5.size(W, H, py5.P2D)
    py5.frame_rate(60)
    py5.text_size(13)

    for i, inp in enumerate(inputs):
        inp["x"] = PANEL_X + INPUT_LABEL_W
        inp["y"] = INPUTS_Y_START + i * (INPUT_H + INPUT_GAP)
        inp["w"] = INPUT_BOX_W
        inp["h"] = INPUT_H
        inp["active"] = False
        inp["edit_text"] = ""

    last_y = INPUTS_Y_START + len(inputs) * (INPUT_H + INPUT_GAP)
    BTN_RENDER["x"] = PANEL_X
    BTN_RENDER["y"] = last_y + 18
    BTN_RENDER["w"] = INPUT_LABEL_W + INPUT_BOX_W
    BTN_RENDER["h"] = 36

    BTN_SAVE["x"] = PANEL_X
    BTN_SAVE["y"] = BTN_RENDER["y"] + BTN_RENDER["h"] + 8
    BTN_SAVE["w"] = INPUT_LABEL_W + INPUT_BOX_W
    BTN_SAVE["h"] = 30


def draw():
    # Two-pass render trigger so the "rendering..." status actually shows
    # before do_render() blocks the main thread.
    if state["render_pending"]:
        state["render_pending"] = False
        do_render()

    py5.background(18, 20, 26)

    # ---- left: thumbnail ----
    py5.fill(220)
    py5.text_size(13)
    py5.text("tune_patch_strokes.py", THUMB_X, 25)
    py5.no_fill()
    py5.stroke(70, 75, 90)
    py5.stroke_weight(1)
    py5.rect(THUMB_X, THUMB_Y, THUMB_W, THUMB_H)
    if state["render_image"] is not None:
        py5.image(state["render_image"], THUMB_X, THUMB_Y, THUMB_W, THUMB_H)
    else:
        py5.fill(140, 140, 160, 180)
        py5.text("(press RENDER / 'r' to draw)",
                 THUMB_X + 240, THUMB_Y + THUMB_H / 2)

    # separator
    py5.stroke(60, 65, 75)
    py5.line(PANEL_X - 12, 0, PANEL_X - 12, H)

    # ---- right: input fields ----
    py5.text_size(13)
    for inp in inputs:
        py5.fill(220)
        py5.text(inp["label"], PANEL_X, inp["y"] + 19)
        if inp["active"]:
            py5.stroke(220, 180, 90); py5.stroke_weight(1.5)
            py5.fill(35, 38, 48)
        else:
            py5.stroke(70, 75, 90); py5.stroke_weight(1.0)
            py5.fill(25, 28, 36)
        py5.rect(inp["x"], inp["y"], inp["w"], inp["h"])

        if inp["active"]:
            display = inp["edit_text"]
        else:
            v = state[inp["key"]]
            display = (inp["fmt"].format(v) if inp["type"] != "str" else str(v))
        py5.fill(245, 220, 180) if inp["active"] else py5.fill(225)
        py5.text(display, inp["x"] + 8, inp["y"] + 19)

        # caret
        if inp["active"]:
            cw = py5.text_width(display)
            py5.stroke(245, 220, 180); py5.stroke_weight(1)
            py5.line(inp["x"] + 8 + cw + 1, inp["y"] + 5,
                     inp["x"] + 8 + cw + 1, inp["y"] + inp["h"] - 5)

        # cycle-hint for str-choices
        if "choices" in inp and not inp["active"]:
            py5.fill(140, 145, 160)
            py5.text_size(11)
            py5.text("(click to cycle)", inp["x"] + inp["w"] + 8, inp["y"] + 19)
            py5.text_size(13)

    # buttons
    _draw_button(BTN_RENDER, "RENDER (r)", (170, 70, 70), (200, 110, 110))
    _draw_button(BTN_SAVE,   "SAVE COPY (s)", (70, 110, 170), (110, 150, 200))

    # status + hint
    if state["render_msg"]:
        py5.fill(150, 230, 150)
        py5.text_size(12)
        py5.text(state["render_msg"], PANEL_X, BTN_SAVE["y"] + BTN_SAVE["h"] + 28)
    py5.fill(140, 145, 160)
    py5.text_size(11)
    py5.text("[r] render   [s] save copy   [Tab] next   [Enter] commit   [Esc] cancel",
             10, H - 14)


def _draw_button(btn, label, base_rgb, hover_rgb):
    is_hover = (btn["x"] <= py5.mouse_x <= btn["x"] + btn["w"]
                and btn["y"] <= py5.mouse_y <= btn["y"] + btn["h"])
    rgb = hover_rgb if is_hover else base_rgb
    py5.stroke_weight(1)
    py5.stroke(*[c + 30 for c in rgb])
    py5.fill(*rgb)
    py5.rect(btn["x"], btn["y"], btn["w"], btn["h"])
    py5.no_stroke()
    py5.fill(255)
    py5.text_size(14)
    tw = py5.text_width(label)
    py5.text(label, btn["x"] + (btn["w"] - tw) / 2, btn["y"] + 23)
    py5.text_size(13)


# ---- mouse / key handling ----
def _hit(rect):
    return (rect["x"] <= py5.mouse_x <= rect["x"] + rect["w"]
            and rect["y"] <= py5.mouse_y <= rect["y"] + rect["h"])


def mouse_pressed():
    if _hit(BTN_RENDER):
        _unfocus_all_commit()
        state["render_msg"] = "rendering..."
        state["render_pending"] = True
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
                inp["active"] = True
                v = state[inp["key"]]
                inp["edit_text"] = (inp["fmt"].format(v) if inp["type"] != "str" else str(v)).strip()
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
            v = state[nxt["key"]]
            nxt["edit_text"] = (nxt["fmt"].format(v) if nxt["type"] != "str" else str(v)).strip()
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

    # global shortcuts
    if py5.key == 'r':
        state["render_msg"] = "rendering..."
        state["render_pending"] = True
    elif py5.key == 's':
        do_save_copy()


def _commit(inp):
    txt = inp["edit_text"].strip()
    try:
        if inp["type"] == "int":
            v = int(float(txt))
        elif inp["type"] == "float":
            v = float(txt)
        else:
            v = txt
        if "choices" in inp and v not in inp["choices"]:
            return  # reject invalid str choice
        state[inp["key"]] = v
    except (ValueError, TypeError):
        pass


def _unfocus_all_commit():
    for inp in inputs:
        if inp["active"]:
            _commit(inp)
            inp["active"] = False


py5.run_sketch()
