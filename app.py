"""Web control surface for the layered renderer.

Replaces tune_layers.py. Python render backend (render_layers.py) is unchanged
-- this file is purely the UI glue: load a scene once, hold the live
composition in memory, serve JSON APIs for edit + render.

Run:
  SCENE_NAME=redhead python app.py
  SCENE_FILE=scenes/siyun_walks.json python app.py
"""
import contextlib
import io
import json
import os
import time
import traceback
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_from_directory

# Optional remote rendering support
try:
    import requests
except ImportError:
    requests = None

from render_layers import (
    DEFAULT_COMPOSITION, DEFAULT_WALKS, LAYER_PARAM_SCHEMAS, WALK_KEY_MAP,
    build_scene_data, render_composition, reproject_scene,
)
from scene_io import load_scene, scene_path


# DEFAULT_WALKS is keyed UPPER_CASE; the UI / layer["params"] dict uses
# lower_snake_case. Build the inverse mapping once so _hydrate_layer can
# fill in all walks-layer params with sensible defaults.
_WALKS_LOWER_DEFAULTS = {
    next((k for k, v in WALK_KEY_MAP.items() if v == upper_key),
         upper_key.lower()): value
    for upper_key, value in DEFAULT_WALKS.items()
}


# ---- scene + scene_data are loaded once at startup ----
SCENE_REF = os.environ.get("SCENE_FILE") or os.environ.get("SCENE_NAME", "siyun")
cfg = load_scene(SCENE_REF)
SCENE_JSON = cfg.PATH if hasattr(cfg, "PATH") else scene_path(SCENE_REF)

REMOTE_RENDER_URL = os.environ.get("REMOTE_RENDER_URL")
if REMOTE_RENDER_URL and requests is None:
    print("[app] WARNING: REMOTE_RENDER_URL is set but 'requests' is not installed.")
    print("[app]          Install it with: pip install requests")
    print("[app]          Remote mode disabled.")
    REMOTE_RENDER_URL = None

print(f"[app] loading scene: {SCENE_REF}")
if REMOTE_RENDER_URL:
    print(f"[app] remote render mode: {REMOTE_RENDER_URL}")
scene_data = build_scene_data(SCENE_REF)
# ROOT FIX: build_scene_data calls load_scene() internally, producing its
# own cfg namespace. Alias app.py's cfg to that one so /api/camera mutations
# and reproject_scene() reads share a single source of truth. Without this,
# mutating cfg.ELEV_DEG had zero effect on projection.
cfg = scene_data["cfg"]
assert cfg is scene_data["cfg"], "cfg aliasing failed (this should never happen)"
print(f"[init] cfg shared with scene_data: True   "
      f"initial elev={cfg.ELEV_DEG:g} azim={cfg.AZIM_DEG:g} "
      f"fov={cfg.FOV_DEG:g} dist·k={cfg.DISTANCE_K:g}")
layer_cache = OrderedDict()

# ---- per-layer defaults so every schema field has a value ----
_MASK3D_DEFAULTS = dict(
    mask3d_enabled=False,
    mask3d_x=0.0, mask3d_y=0.0, mask3d_z=0.0,
    mask3d_r_in=0.3, mask3d_r_out=1.0,
    mask3d_invert=False,
)
_BASE_SPLAT_DEFAULTS = dict(
    bg_r=0.0, bg_g=0.0, bg_b=0.0,
    saturation=1.0,
    curvature_weight=0.0,
    depth_focus=0.5, depth_blur=0.0,
)
_FOCAL_LEGACY = (
    "focal_enabled", "focal_cx", "focal_cy", "focal_rx", "focal_ry",
    "focal_angle_deg", "focal_falloff", "focal_invert",
)


def _hydrate_layer(layer):
    for k, v in _MASK3D_DEFAULTS.items():
        layer.setdefault(k, v)
    for k in _FOCAL_LEGACY:
        layer.pop(k, None)
    if layer.get("type") == "base_splat":
        bg = layer.get("bg") or layer.get("background")
        if bg is not None and len(bg) >= 3:
            for i, c in enumerate("rgb"):
                layer.setdefault(f"bg_{c}", float(bg[i]))
            layer.pop("bg", None)
            layer.pop("background", None)
        for k, v in _BASE_SPLAT_DEFAULTS.items():
            layer.setdefault(k, v)
    if layer.get("type") == "surface_walks":
        params = layer.setdefault("params", {})
        # Fill in every DEFAULT_WALKS value so the UI shows real numbers
        # instead of blanks (the python tuner used to fall back at read time).
        for k, v in _WALKS_LOWER_DEFAULTS.items():
            params.setdefault(k, v)
        for c in "rgb":
            params.setdefault(f"ink_{c}", 0.0)


def _load_composition():
    comp = deepcopy(DEFAULT_COMPOSITION)
    comp.update(cfg._raw.get("composition", {}))
    comp["layers"] = [deepcopy(l) for l in comp.get("layers", [])]
    for layer in comp["layers"]:
        _hydrate_layer(layer)
    return comp


ADD_TEMPLATES = {
    "base_splat": {
        "name": "splat", "type": "base_splat", "enabled": True, "alpha": 0.6,
        "point_pct": 0.25, "seed": 17,
        **_MASK3D_DEFAULTS, **_BASE_SPLAT_DEFAULTS,
    },
    "surface_walks": {
        "name": "walks", "type": "surface_walks", "enabled": True, "alpha": 1.0,
        "params": {
            "n_walkers": 300, "steps": 30, "step_radius_px": 14.0,
            "forward_bias": 8.0, "direction_mode": "global",
            "global_dir_deg": 90.0, "noise_scale": 90.0,
            "stroke_alpha": 0.65, "ink_darken": 0.0,
            "ink_r": 0.0, "ink_g": 0.0, "ink_b": 0.0,
            "placement": "saliency", "stroke_mode": "line", "stroke_width": 1.0,
            "seed": 17,
        },
        **_MASK3D_DEFAULTS,
    },
}


composition = _load_composition()
log_lines = []
last_render_path = None


def _camera_tuple():
    return (
        float(getattr(cfg, "ELEV_DEG", 0.0)),
        float(getattr(cfg, "AZIM_DEG", 0.0)),
        float(getattr(cfg, "FOV_DEG", 28.0)),
        float(getattr(cfg, "DISTANCE_K", 1.5)),
        float(getattr(cfg, "HEAD_BIAS_X", 0.0)),
        float(getattr(cfg, "HEAD_BIAS_Y", 0.0)),
    )


# Track the camera state that scene_data was projected for. When /api/render
# is called and this differs from cfg's current camera, we lazily reproject.
# This means orbit/typing during exploration doesn't trigger work -- only the
# render itself does, and at most once per render.
_projected_camera = _camera_tuple()


def _apply_camera_override(cam):
    """Copy camera params from a request body onto cfg. Returns True if
    any field actually changed."""
    if not cam:
        return False
    changed = False
    for key, attr in (("elev_deg", "ELEV_DEG"), ("azim_deg", "AZIM_DEG"),
                       ("fov_deg", "FOV_DEG"), ("distance_k", "DISTANCE_K"),
                       ("head_bias_x", "HEAD_BIAS_X"), ("head_bias_y", "HEAD_BIAS_Y")):
        if key in cam and cam[key] is not None:
            new_v = float(cam[key])
            old_v = float(getattr(cfg, attr, 0.0))
            if old_v != new_v:
                print(f"[_apply_camera_override] {attr}: {old_v:g} -> {new_v:g}")
                setattr(cfg, attr, new_v)
                # Verify the value was actually set
                verify_v = float(getattr(cfg, attr, 0.0))
                if verify_v != new_v:
                    print(f"[_apply_camera_override] WARNING: {attr} not actually set! {verify_v:g} != {new_v:g}")
                changed = True
    return changed


def _append_log(text):
    if not text:
        return
    for line in str(text).splitlines():
        if line.strip():
            log_lines.append(line)
    while len(log_lines) > 500:
        log_lines.pop(0)


def _camera_dict():
    return dict(
        elev_deg=float(getattr(cfg, "ELEV_DEG", 0.0)),
        azim_deg=float(getattr(cfg, "AZIM_DEG", 0.0)),
        fov_deg=float(getattr(cfg, "FOV_DEG", 40.0)),
        distance_k=float(getattr(cfg, "DISTANCE_K", 1.0)),
        head_bias_x=float(getattr(cfg, "HEAD_BIAS_X", 0.0)),
        head_bias_y=float(getattr(cfg, "HEAD_BIAS_Y", 0.0)),
    )


# ---------------------------------------------------------------- routes ----
app = Flask(__name__, template_folder="templates", static_folder="static")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/init")
def api_init():
    bb_min, bb_max = scene_data["bbox"]
    dens_lo, dens_hi = scene_data["density_bbox"]
    return jsonify({
        "scene": {
            "name": cfg.SCENE_NAME,
            "json_path": SCENE_JSON,
            "bbox_min": bb_min.tolist(),
            "bbox_max": bb_max.tolist(),
            "density_min": dens_lo.tolist(),
            "density_max": dens_hi.tolist(),
            "render_size": [int(cfg.W), int(cfg.H)],
        },
        "camera": _camera_dict(),
        "schema": LAYER_PARAM_SCHEMAS,
        "composition": composition,
        "log": log_lines[-50:],
    })


@app.route("/api/composition", methods=["POST"])
def api_composition():
    global composition
    payload = request.get_json() or {}
    composition = payload
    n = len(payload.get("layers", []))
    print(f"[composition] received update, {n} layers")
    return jsonify({"ok": True})


@app.route("/api/camera", methods=["POST"])
def api_camera():
    """Set camera params on cfg. Does NOT reproject -- that happens lazily in
    /api/render. Keeps orbiting/typing cheap; the actual projection cost is
    paid once per render when the camera differs from what was last projected.
    """
    changed = _apply_camera_override(request.get_json() or {})
    if changed:
        cur = _camera_dict()
        print(f"[camera] elev={cur['elev_deg']:g} azim={cur['azim_deg']:g} "
              f"fov={cur['fov_deg']:g} dist·k={cur['distance_k']:g} "
              f"(deferred reproject)")
    return jsonify({"ok": True, "camera": _camera_dict(), "changed": changed})


@app.route("/api/add_layer", methods=["POST"])
def api_add_layer():
    payload = request.get_json() or {}
    layer_type = payload.get("type", "base_splat")
    if layer_type not in ADD_TEMPLATES:
        return jsonify({"error": f"unknown layer type: {layer_type}"}), 400
    new_layer = deepcopy(ADD_TEMPLATES[layer_type])
    existing = {l.get("name") for l in composition.get("layers", [])}
    base = new_layer["name"]
    if base in existing:
        i = 2
        while f"{base}_{i}" in existing:
            i += 1
        new_layer["name"] = f"{base}_{i}"
    composition.setdefault("layers", []).append(new_layer)
    return jsonify({"ok": True, "layer": new_layer})


@app.route("/api/remove_layer", methods=["POST"])
def api_remove_layer():
    payload = request.get_json() or {}
    idx = int(payload.get("index", -1))
    layers = composition.get("layers", [])
    if 0 <= idx < len(layers):
        removed = layers.pop(idx)
        return jsonify({"ok": True, "removed": removed.get("name", "?")})
    return jsonify({"error": "bad index"}), 400


class RemoteRenderError(Exception):
    pass


def _remote_render(payload):
    """Forward a render request to REMOTE_RENDER_URL and save the returned
    PNG locally so the frontend can fetch it via /img/..."""
    resp = requests.post(
        f"{REMOTE_RENDER_URL}/api/render",
        json=payload,
        timeout=300,
    )
    if not resp.ok:
        try:
            detail = resp.json().get("error", resp.text)
        except Exception:
            detail = resp.text
        raise RemoteRenderError(f"remote error ({resp.status_code}): {detail}")

    ts = int(time.time())
    filename = f"{cfg.SCENE_NAME}_remote_{ts}.png"
    local_path = Path(__file__).parent / "images" / filename
    local_path.write_bytes(resp.content)

    global last_render_path
    last_render_path = str(local_path)
    _append_log(f"remote render -> {last_render_path}")

    return jsonify({
        "ok": True,
        "image": f"/img/{filename}?t={ts}",
        "time": 0,
        "log": log_lines[-50:],
    })


@app.route("/api/render", methods=["POST"])
def api_render():
    """Render with the live composition and camera.

    Accepts optional `camera` and `composition` in the request body. When
    present, they OVERRIDE whatever's in memory -- the UI is the source of
    truth for the render. After overrides land, we reproject whenever
    _apply_camera_override actually mutated cfg.

    If REMOTE_RENDER_URL is configured, the render is offloaded to the remote
    server; on failure it falls back to local rendering automatically.
    """
    global last_render_path, composition, _projected_camera
    payload = request.get_json() or {}
    cam_changed = False
    if "camera" in payload:
        cam_changed = _apply_camera_override(payload["camera"])
    if "composition" in payload:
        composition = payload["composition"]

    # Try remote render first if configured
    if REMOTE_RENDER_URL:
        try:
            return _remote_render(payload)
        except RemoteRenderError as exc:
            _append_log(f"remote render failed: {exc}")
            _append_log("falling back to local render...")
        except Exception as exc:
            _append_log(f"remote connection failed: {exc}")
            _append_log("falling back to local render...")

    # Always reproject. The camera passed in /api/render is the source of
    # truth; we deliberately do NOT trust any lazy "projection-already-up-to-
    # date" tracking, because if that tracking is ever wrong (and it has
    # been), the render silently uses stale projection. Cost is one extra
    # `_project_scene` per render -- worth the certainty.
    cur = _camera_tuple()
    print(f"[render] reprojecting for camera {cur}  (cam_changed={cam_changed})")
    print(f"[render] cfg values before reproject: ELEV_DEG={cfg.ELEV_DEG:g}  AZIM_DEG={cfg.AZIM_DEG:g}  FOV_DEG={cfg.FOV_DEG:g}  DISTANCE_K={cfg.DISTANCE_K:g}")
    reproject_scene(scene_data)
    layer_cache.clear()
    _projected_camera = cur

    t0 = time.time()
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            _, out_path, _ = render_composition(
                SCENE_REF, composition=composition,
                write=True, stamp_label=True,
                scene_data=scene_data, layer_cache=layer_cache,
            )
    except Exception as exc:
        tb = traceback.format_exc()
        _append_log(f"ERROR: {exc}")
        _append_log(tb)
        return jsonify({"error": str(exc), "trace": tb,
                        "log": log_lines[-50:]}), 500
    dt = time.time() - t0
    _append_log(buf.getvalue())
    _append_log(f"render done in {dt:.1f}s  ->  {out_path}")
    last_render_path = out_path
    return jsonify({
        "ok": True,
        "image": f"/img/{Path(out_path).name}?t={int(time.time())}",
        "time": dt,
        "log": log_lines[-50:],
    })


@app.route("/img/<path:filename>")
def serve_image(filename):
    return send_from_directory(Path(__file__).parent / "images", filename)


@app.route("/api/save_scene", methods=["POST"])
def api_save_scene():
    try:
        try:
            with open(SCENE_JSON) as f:
                data = json.load(f)
        except FileNotFoundError:
            data = {"name": SCENE_REF}
        # Strip UI-only fields so they don't pollute the scene JSON.
        cleaned = deepcopy(composition)
        for layer in cleaned.get("layers", []):
            for k in list(layer.keys()):
                if k.startswith("_ui_"):
                    del layer[k]
        data["composition"] = cleaned
        # Persist current camera params back into the nested `camera` section
        # (scene_io supports either nested or flat; we standardise on nested
        # and clear the legacy flat keys so there's one source of truth).
        cam = data.setdefault("camera", {})
        cam["elev_deg"]    = float(getattr(cfg, "ELEV_DEG", 0.0))
        cam["azim_deg"]    = float(getattr(cfg, "AZIM_DEG", 0.0))
        cam["fov_deg"]     = float(getattr(cfg, "FOV_DEG", 28.0))
        cam["distance_k"]  = float(getattr(cfg, "DISTANCE_K", 1.5))
        cam["head_bias_x"] = float(getattr(cfg, "HEAD_BIAS_X", 0.0))
        cam["head_bias_y"] = float(getattr(cfg, "HEAD_BIAS_Y", 0.0))
        for k in ("elev_deg", "azim_deg", "fov_deg", "distance_k",
                  "head_bias_x", "head_bias_y"):
            data.pop(k, None)
        with open(SCENE_JSON, "w") as f:
            json.dump(data, f, indent=2)
            f.write("\n")
        n = len(cleaned.get("layers", []))
        print(f"[save] wrote {n} layers + camera to {SCENE_JSON}")
        _append_log(f"saved scene -> {SCENE_JSON}")
        return jsonify({"ok": True, "path": SCENE_JSON, "log": log_lines[-50:]})
    except Exception as exc:
        print(f"[save] FAILED: {exc}")
        return jsonify({"error": str(exc)}), 500


@app.route("/api/reload", methods=["POST"])
def api_reload():
    """Re-read the scene JSON from disk (picks up manual edits to camera,
    composition, etc.) without re-loading the .ply -- so it's fast. Frontend
    should refresh after to pull the new state."""
    global cfg, composition, _projected_camera
    cfg = load_scene(SCENE_REF)
    scene_data["cfg"] = cfg
    reproject_scene(scene_data)
    layer_cache.clear()
    composition = _load_composition()
    _projected_camera = _camera_tuple()
    print(f"[reload] re-read {SCENE_JSON} (camera + composition)")
    _append_log(f"reloaded scene from {SCENE_JSON}")
    return jsonify({"ok": True, "log": log_lines[-50:]})


@app.route("/api/save_image", methods=["POST"])
def api_save_image():
    """Copy the most recent render to a timestamped file (metadata in the
    PNG tEXt chunk travels with the copy)."""
    if not last_render_path or not os.path.exists(last_render_path):
        return jsonify({"error": "no render yet -- click render first"}), 400
    import shutil
    base = os.path.splitext(os.path.basename(last_render_path))[0]
    ts = time.strftime("%Y%m%d_%H%M%S")
    dst = f"images/{base}_{ts}.png"
    shutil.copy(last_render_path, dst)
    print(f"[save_image] {dst}")
    _append_log(f"saved image -> {dst}")
    return jsonify({"ok": True, "path": dst, "log": log_lines[-50:]})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5050))
    print(f"[app] listening on http://127.0.0.1:{port}")
    app.run(debug=False, port=port, use_reloader=False, threaded=False)
