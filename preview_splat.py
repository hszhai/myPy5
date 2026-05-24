"""Interactive preview tool for a 3DGS .ply scene.

Left side  -- 3D framing helpers only (no point cloud; the blobs blocked
              the view). Bounding boxes + a small green sphere at the
              median centre + an axis gizmo so you can read orientation.

Right side -- last render thumbnail + typed input fields for every
              camera knob. The mouse can still drag/wheel in the 3D
              area for quick changes; the input fields update to match.

Controls:
  drag in 3D area        rotate (azim/elev)
  mouse wheel            change distance_k (zoom)
  click an input field   type a new value
                         Enter commits, Tab moves to next, Esc cancels
  click RENDER  /  'r'   full-res gsplat render -> images/<name>_preview.png
                         and shown in the right panel
  's'                    write camera state -> data/<name>_camera.json

Run as a standalone script (macOS GL/window limitation):
  py5 preview_splat.py                          # data/RedHead.ply
  py5 preview_splat.py data/Siyun.ply           # any 3DGS .ply
"""
import os
os.environ.setdefault("JAVA_HOME", "/usr/local/opt/openjdk@17")

import json
import sys
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import py5
from gsplat import (
    load_3dgs_ply, decode_3dgs, make_camera,
    project_perspective, cull, splat,
)
from scene_io import load_scene, save_scene, scene_exists

# ============ knobs =======================================================
# argv: either a scene name ("redhead", "siyun", "audi") or a direct .ply path.
_arg = sys.argv[1] if len(sys.argv) > 1 else "redhead"
if _arg.endswith(".ply"):
    PLY = _arg
    SCENE_NAME = os.path.splitext(os.path.basename(_arg))[0].lower()
else:
    SCENE_NAME = _arg
    if scene_exists(SCENE_NAME):
        PLY = load_scene(SCENE_NAME).PLY
    else:
        PLY = f"data/{SCENE_NAME}.ply"

# Load scene defaults if available (so the preview opens with the current
# saved camera, not always at zero).
_scn = load_scene(SCENE_NAME) if scene_exists(SCENE_NAME) else None
W, H = 1200, 900
PANEL_X = 850                       # the right panel starts here
RENDER_W = _scn.W if _scn else 1080
RENDER_H = _scn.H if _scn else 1440
SCENE_UP_FLIP = _scn.SCENE_UP_FLIP if _scn else True
print(f"[preview] scene = {SCENE_NAME}  ({PLY})  "
      f"{'loaded camera from scenes/' + SCENE_NAME + '.json' if _scn else 'using defaults'}")


# ============ load + analyse =============================================
print(f"loading {PLY}...")
data = load_3dgs_ply(PLY)
G = decode_3dgs(data)
N = len(data)
print(f"  {N} Gaussians")

xyz_all = G["xyz"].astype(np.float64)
robust_min = np.percentile(xyz_all, 5,  axis=0)
robust_max = np.percentile(xyz_all, 95, axis=0)
core_min   = np.percentile(xyz_all, 25, axis=0)
core_max   = np.percentile(xyz_all, 75, axis=0)
center0 = np.median(xyz_all, axis=0)
extent = float(np.max(robust_max - robust_min))
print(f"  centre: {center0}")
print(f"  extent: {extent:.3f}")


# ============ single source of truth for camera ==========================
# Initialise from the scene file when one exists, so the preview reflects
# the most recent saved camera.
state = dict(
    azim_deg=_scn.AZIM_DEG if _scn else 0.0,
    elev_deg=_scn.ELEV_DEG if _scn else 0.0,
    distance_k=_scn.DISTANCE_K if _scn else 1.5,
    fov_deg=_scn.FOV_DEG if _scn else 28.0,
    head_bias_x=_scn.HEAD_BIAS_X if _scn else 0.0,
    head_bias_y=_scn.HEAD_BIAS_Y if _scn else 0.0,
    render_msg="",
    render_image=None,
)

# Typed input fields. Each entry edits state[key].
inputs = [
    dict(key="azim_deg",     label="azim deg", fmt="{:.1f}"),
    dict(key="elev_deg",     label="elev deg", fmt="{:.1f}"),
    dict(key="distance_k",   label="dist k",   fmt="{:.2f}"),
    dict(key="fov_deg",      label="fov deg",  fmt="{:.1f}"),
    dict(key="head_bias_x",  label="bias x",   fmt="{:.2f}"),
    dict(key="head_bias_y",  label="bias y",   fmt="{:.2f}"),
]

# Layout (filled in setup())
PANEL_PAD = 20
THUMB_X, THUMB_Y, THUMB_W, THUMB_H = PANEL_X + PANEL_PAD, 20, 310, 414
INPUTS_START_Y = THUMB_Y + THUMB_H + 30
INPUT_H = 28
INPUT_GAP = 6
INPUT_LABEL_W = 80
INPUT_BOX_W = 200
BTN = dict(x=0, y=0, w=0, h=0)

mouse_st = dict(last_x=0, last_y=0, dragging_3d=False)


# ============ py5 callbacks ==============================================
def setup():
    py5.size(W, H, py5.P3D)
    py5.frame_rate(60)
    py5.text_size(12)

    for i, inp in enumerate(inputs):
        inp["x"] = THUMB_X + INPUT_LABEL_W
        inp["y"] = INPUTS_START_Y + i * (INPUT_H + INPUT_GAP)
        inp["w"] = INPUT_BOX_W
        inp["h"] = INPUT_H
        inp["active"] = False
        inp["edit_text"] = ""

    last_y = INPUTS_START_Y + len(inputs) * (INPUT_H + INPUT_GAP)
    BTN["x"] = THUMB_X
    BTN["y"] = last_y + 12
    BTN["w"] = INPUT_LABEL_W + INPUT_BOX_W
    BTN["h"] = 36


def draw():
    py5.background(18, 20, 26)

    # ========== 3D scene (left, no points) ================================
    py5.push_matrix()
    py5.translate(PANEL_X / 2, H / 2, 0)
    py5.rotate_x(np.radians(state["elev_deg"]))
    py5.rotate_y(np.radians(state["azim_deg"]))
    world_scale = (min(PANEL_X, H) * 0.55) / (extent * state["distance_k"])
    py5.scale(world_scale)
    if SCENE_UP_FLIP:
        py5.scale(1, -1, 1)
    cx = center0[0] + state["head_bias_x"]
    cy = center0[1] + state["head_bias_y"]
    cz = center0[2]
    py5.translate(-cx, -cy, -cz)

    # Bounding boxes
    py5.no_fill()
    inv = 1.0 / world_scale                       # for constant on-screen pixel widths
    py5.stroke_weight(1.5 * inv)
    py5.stroke(220, 200, 60, 150)
    _draw_aabb(robust_min, robust_max)
    py5.stroke(255, 90, 90, 200)
    _draw_aabb(core_min, core_max)

    # Small green sphere at the framing centre
    py5.push_matrix()
    py5.translate(cx, cy, cz)
    py5.no_stroke()
    py5.fill(120, 240, 120, 230)
    py5.sphere(extent * 0.015)
    py5.pop_matrix()

    # Axis gizmo at centre (+X red, +Y green, +Z blue)
    L = extent * 0.18
    py5.stroke_weight(2.5 * inv)
    py5.stroke(255, 90, 90);  py5.line(cx, cy, cz, cx + L, cy, cz)
    py5.stroke(120, 240, 120); py5.line(cx, cy, cz, cx, cy + L, cz)
    py5.stroke(100, 150, 255); py5.line(cx, cy, cz, cx, cy, cz + L)

    py5.pop_matrix()

    # ========== HUD on the left ==========================================
    py5.hint(py5.DISABLE_DEPTH_TEST)
    py5.camera()
    py5.no_lights()
    py5.fill(180, 200, 230)
    py5.text("yellow = robust bounds (5-95%)   red = dense core (25-75%)",
             12, 22)
    py5.text("axis gizmo at centre: +X red, +Y green, +Z blue", 12, 40)
    py5.fill(180)
    py5.text("[drag-3D] rotate   [wheel] zoom   "
             "[click field] edit   [Enter] commit   [Tab] next   [Esc] cancel",
             12, H - 12)

    # vertical separator
    py5.stroke(60, 65, 75)
    py5.stroke_weight(1)
    py5.line(PANEL_X, 0, PANEL_X, H)

    # ========== right panel ==============================================
    _draw_render_thumb()
    _draw_inputs()
    _draw_button()
    if state["render_msg"]:
        py5.fill(150, 230, 150)
        py5.text(state["render_msg"], THUMB_X, BTN["y"] + BTN["h"] + 24)

    py5.hint(py5.ENABLE_DEPTH_TEST)


def _draw_render_thumb():
    py5.no_fill()
    py5.stroke(70, 75, 90)
    py5.stroke_weight(1)
    py5.rect(THUMB_X, THUMB_Y, THUMB_W, THUMB_H)
    if state["render_image"] is not None:
        py5.image(state["render_image"], THUMB_X, THUMB_Y, THUMB_W, THUMB_H)
    else:
        py5.fill(150, 150, 170, 160)
        py5.text("(no render yet -- press R or RENDER)",
                 THUMB_X + 50, THUMB_Y + THUMB_H / 2)


def _draw_inputs():
    py5.text_size(12)
    for inp in inputs:
        py5.fill(220)
        py5.text(inp["label"], THUMB_X, inp["y"] + 19)
        if inp["active"]:
            py5.stroke(220, 180, 90); py5.stroke_weight(1.5)
            py5.fill(35, 38, 48)
        else:
            py5.stroke(70, 75, 90); py5.stroke_weight(1.0)
            py5.fill(25, 28, 36)
        py5.rect(inp["x"], inp["y"], inp["w"], inp["h"])

        display = inp["edit_text"] if inp["active"] \
                  else inp["fmt"].format(state[inp["key"]])
        py5.fill(245, 220, 180) if inp["active"] else py5.fill(225)
        py5.text(display, inp["x"] + 8, inp["y"] + 19)

        if inp["active"]:                               # cursor caret
            cw = py5.text_width(display)
            py5.stroke(245, 220, 180); py5.stroke_weight(1)
            py5.line(inp["x"] + 8 + cw + 1, inp["y"] + 5,
                     inp["x"] + 8 + cw + 1, inp["y"] + inp["h"] - 5)


def _draw_button():
    hover = (BTN["x"] <= py5.mouse_x <= BTN["x"] + BTN["w"]
             and BTN["y"] <= py5.mouse_y <= BTN["y"] + BTN["h"])
    py5.stroke_weight(1)
    if hover:
        py5.stroke(200, 110, 110); py5.fill(170, 70, 70)
    else:
        py5.stroke(140, 70, 70);   py5.fill(120, 50, 50)
    py5.rect(BTN["x"], BTN["y"], BTN["w"], BTN["h"])
    py5.no_stroke(); py5.fill(255)
    py5.text_size(14)
    label = "RENDER (r)"
    tw = py5.text_width(label)
    py5.text(label, BTN["x"] + (BTN["w"] - tw) / 2, BTN["y"] + 23)
    py5.text_size(12)


def _draw_aabb(bmin, bmax):
    c = [
        (bmin[0], bmin[1], bmin[2]), (bmax[0], bmin[1], bmin[2]),
        (bmax[0], bmax[1], bmin[2]), (bmin[0], bmax[1], bmin[2]),
        (bmin[0], bmin[1], bmax[2]), (bmax[0], bmin[1], bmax[2]),
        (bmax[0], bmax[1], bmax[2]), (bmin[0], bmax[1], bmax[2]),
    ]
    for a, b in [(0,1),(1,2),(2,3),(3,0),
                 (4,5),(5,6),(6,7),(7,4),
                 (0,4),(1,5),(2,6),(3,7)]:
        py5.line(c[a][0], c[a][1], c[a][2], c[b][0], c[b][1], c[b][2])


# ============ mouse / key handling =======================================
def mouse_pressed():
    # Render button?
    if _hit(BTN):
        _unfocus_all_commit()
        do_render()
        return
    # Input box?
    for inp in inputs:
        if _hit(inp):
            _unfocus_all_commit()
            inp["active"] = True
            inp["edit_text"] = inp["fmt"].format(state[inp["key"]]).strip()
            return
    # Otherwise -- maybe start a 3D drag (only on the left side)
    _unfocus_all_commit()
    if py5.mouse_x < PANEL_X:
        mouse_st["dragging_3d"] = True
        mouse_st["last_x"] = py5.mouse_x
        mouse_st["last_y"] = py5.mouse_y


def mouse_released():
    mouse_st["dragging_3d"] = False


def mouse_dragged():
    if not mouse_st["dragging_3d"]:
        return
    dx = py5.mouse_x - mouse_st["last_x"]
    dy = py5.mouse_y - mouse_st["last_y"]
    state["azim_deg"] += dx * 0.5
    state["elev_deg"] += dy * 0.5
    state["elev_deg"] = float(np.clip(state["elev_deg"], -89.0, 89.0))
    mouse_st["last_x"] = py5.mouse_x
    mouse_st["last_y"] = py5.mouse_y


def mouse_wheel(event):
    state["distance_k"] *= 1.1 ** event.get_count()
    state["distance_k"] = float(np.clip(state["distance_k"], 0.1, 10.0))


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
            nxt["edit_text"] = nxt["fmt"].format(state[nxt["key"]]).strip()
        else:
            try:
                ch = str(k)
                if len(ch) == 1 and (ch.isdigit() or ch in ".-+e"):
                    active["edit_text"] += ch
            except Exception:
                pass
        return

    # shortcuts when no field is focused
    if py5.key == 'r':
        do_render()
    elif py5.key == 's':
        save_state_json()


def _hit(rect):
    return (rect["x"] <= py5.mouse_x <= rect["x"] + rect["w"]
            and rect["y"] <= py5.mouse_y <= rect["y"] + rect["h"])


def _commit(inp):
    try:
        state[inp["key"]] = float(inp["edit_text"])
    except (ValueError, TypeError):
        pass


def _unfocus_all_commit():
    for inp in inputs:
        if inp["active"]:
            _commit(inp)
            inp["active"] = False


# ============ render + persist actions ===================================
def _base_name():
    return os.path.splitext(os.path.basename(PLY))[0].lower()


def do_render():
    elev_deg, azim_deg = state["elev_deg"], state["azim_deg"]
    print(f"\n[render] azim={azim_deg:.1f}  elev={elev_deg:.1f}  "
          f"dist_k={state['distance_k']:.2f}  fov={state['fov_deg']:.1f}")
    state["render_msg"] = "rendering..."
    t0 = time.time()

    ysign = +1.0 if SCENE_UP_FLIP else -1.0
    biased_center = center0.copy()
    biased_center[0] += state["head_bias_x"]
    biased_center[1] += state["head_bias_y"]
    radii = np.linalg.norm(xyz_all - biased_center, axis=1)
    ext = float(np.percentile(radii, 95) * 2.0)

    Rcam = make_camera(elev_deg, azim_deg)
    cam_xyz = (xyz_all - biased_center) @ Rcam.T
    cov_cam = np.einsum("ij,njk,lk->nil", Rcam, G["cov3"], Rcam)
    focal = RENDER_W / (2.0 * np.tan(np.radians(state["fov_deg"]) / 2.0))
    distance = ext * state["distance_k"]

    mean2d, cov2d, depths, valid_z = project_perspective(
        cam_xyz, cov_cam, focal, distance, RENDER_W, RENDER_H, ysign)
    keep = cull(mean2d, cov2d, G["opacities"], valid_z, RENDER_W, RENDER_H)
    order = keep[np.argsort(-depths[keep])]
    img = splat(RENDER_W, RENDER_H, mean2d, cov2d,
                G["colors"], G["opacities"], order, verbose=False)

    out_path = f"images/{_base_name()}_preview.png"
    plt.imsave(out_path, np.clip(img, 0, 1))
    dt = time.time() - t0
    print(f"  saved -> {out_path}  ({dt:.1f}s)")
    state["render_msg"] = f"saved {out_path}  ({dt:.1f}s)"

    try:
        state["render_image"] = py5.load_image(out_path)
    except Exception as e:
        print(f"  (load_image failed: {e})")


def save_state_json():
    """Save the current camera into the canonical scene file
    (scenes/<name>.json). Existing keys not managed here (e.g. credit)
    are preserved -- see scene_io.save_scene().
    """
    out = save_scene(
        SCENE_NAME,
        ply=PLY,
        out=f"images/{SCENE_NAME}_render.png",
        scene_up_flip=SCENE_UP_FLIP,
        w=RENDER_W,
        h=RENDER_H,
        elev_deg=float(state["elev_deg"]),
        azim_deg=float(state["azim_deg"]),
        distance_k=float(state["distance_k"]),
        fov_deg=float(state["fov_deg"]),
        head_bias_x=float(state["head_bias_x"]),
        head_bias_y=float(state["head_bias_y"]),
    )
    print(f"saved camera -> {out}")
    state["render_msg"] = f"saved {out}"


py5.run_sketch()
