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
import threading
import time
import traceback
import uuid
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
    if layer.get("type") == "generative_curve":
        params = layer.setdefault("params", {})
        for c in "rgb":
            params.setdefault(f"color_{c}", 0.0)
        params.setdefault("depth_focus", 0.5)
        params.setdefault("depth_blur", 0.0)
    if layer.get("type") == "zline":
        params = layer.setdefault("params", {})
        for c in "rgb":
            params.setdefault(f"color_{c}", 0.0)
        params.setdefault("p1_x", -0.3)
        params.setdefault("p1_y", 0.0)
        params.setdefault("p1_z", 0.0)
        params.setdefault("p2_x", 0.3)
        params.setdefault("p2_y", 0.0)
        params.setdefault("p2_z", 0.0)
        params.setdefault("show_endpoints", False)
        params.setdefault("n_lines", 5)
        params.setdefault("recursion", 2)
        params.setdefault("displacement", 0.3)
        params.setdefault("displacement_decay", 0.5)
        params.setdefault("neighborhood_range", 0.1)
        params.setdefault("seed", 17)
        params.setdefault("stroke_alpha", 0.7)
        params.setdefault("stroke_mode", "line")
        params.setdefault("stroke_width", 1.0)
        params.setdefault("splat_scale", 0.35)
        params.setdefault("splat_alpha_scale", 0.35)
        params.setdefault("splat_min_sigma", 0.10)
        params.setdefault("n_stamps", 5)
        params.setdefault("color_mode", "fixed")
        params.setdefault("line_jitter", 0.0)


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
    "generative_curve": {
        "name": "curve", "type": "generative_curve", "enabled": True, "alpha": 0.9,
        "params": {
            "shape": "sphere", "n_points": 200, "radius": 0.6,
            "seed": 17, "stroke_mode": "splat", "stroke_alpha": 0.6,
            "splat_scale": 0.4, "splat_alpha_scale": 0.35,
            "splat_min_sigma": 0.10, "splat_max_sigma": 1.20,
            "n_stamps": 5, "color_mode": "fixed",
            "color_r": 0.9, "color_g": 0.3, "color_b": 0.1,
            "depth_focus": 0.5, "depth_blur": 0.0,
            "line_jitter": 0.0, "connect_closest": False,
        },
        **_MASK3D_DEFAULTS,
    },
    "zline": {
        "name": "zline", "type": "zline", "enabled": True, "alpha": 0.9,
        "params": {
            "p1_x": -0.3, "p1_y": 0.0, "p1_z": 0.0,
            "p2_x": 0.3, "p2_y": 0.0, "p2_z": 0.0,
            "show_endpoints": False,
            "n_lines": 5, "recursion": 2,
            "displacement": 0.3, "displacement_decay": 0.5,
            "neighborhood_range": 0.1,
            "seed": 17, "stroke_mode": "line", "stroke_alpha": 0.7,
            "stroke_width": 1.0,
            "splat_scale": 0.35, "splat_alpha_scale": 0.35,
            "splat_min_sigma": 0.10, "splat_max_sigma": 1.20,
            "n_stamps": 5, "color_mode": "fixed",
            "color_r": 0.0, "color_g": 0.0, "color_b": 0.0,
            "line_jitter": 0.0,
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
    cam = scene_data.get("camera", {})
    return jsonify({
        "scene": {
            "name": cfg.SCENE_NAME,
            "json_path": SCENE_JSON,
            "bbox_min": bb_min.tolist(),
            "bbox_max": bb_max.tolist(),
            "density_min": dens_lo.tolist(),
            "density_max": dens_hi.tolist(),
            "render_size": [int(cfg.W), int(cfg.H)],
            "center": cam.get("center").tolist() if hasattr(cam.get("center"), 'tolist') else (cam.get("center") or [0.0, 0.0, 0.0]),
            "Rcam": cam.get("Rcam").tolist() if hasattr(cam.get("Rcam"), 'tolist') else None,
            "focal": float(cam.get("focal", 1.0)) if cam.get("focal") is not None else None,
            "distance": float(cam.get("distance", 1.0)) if cam.get("distance") is not None else None,
            "ysign": float(cam.get("ysign", -1.0)) if cam.get("ysign") is not None else -1.0,
            "extent": float(cam.get("distance", 1.0)) / max(float(getattr(cfg, "DISTANCE_K", 1.0)), 1e-6),
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


# Async job tracking for local renders
_local_jobs = OrderedDict()
_MAX_LOCAL_JOBS = 5
_render_lock = threading.Lock()


def _prune_local_jobs():
    while len(_local_jobs) > _MAX_LOCAL_JOBS:
        _local_jobs.popitem(last=False)


def _do_local_render(job_id: str, payload: dict):
    """Run a render locally and update the job status."""
    try:
        with _render_lock:
            reproject_scene(scene_data)
            layer_cache.clear()

            t0 = time.time()
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                _, out_path, _ = render_composition(
                    SCENE_REF,
                    composition=payload.get("composition"),
                    write=True,
                    stamp_label=True,
                    scene_data=scene_data,
                    layer_cache=layer_cache,
                )
            dt = time.time() - t0
            _append_log(buf.getvalue())
            _append_log(f"render done in {dt:.1f}s  ->  {out_path}")

            global last_render_path
            last_render_path = out_path
            _local_jobs[job_id] = {
                "status": "done",
                "image": f"/img/{Path(out_path).name}?t={int(time.time())}",
                "time": dt,
            }
            _prune_local_jobs()
    except Exception as exc:
        tb = traceback.format_exc()
        _append_log(f"ERROR: {exc}")
        _append_log(tb)
        _local_jobs[job_id] = {"status": "error", "error": str(exc)}
        _prune_local_jobs()


def _do_remote_render(job_id: str, payload: dict):
    """Offload a render to REMOTE_RENDER_URL and poll for completion."""
    try:
        # 1. Create remote job
        create_resp = requests.post(
            f"{REMOTE_RENDER_URL}/api/render",
            json=payload,
            timeout=30,
        )
        if not create_resp.ok:
            raise RemoteRenderError(f"remote returned {create_resp.status_code}")
        remote_job = create_resp.json()
        remote_job_id = remote_job["job_id"]

        # 2. Poll remote status (max ~10 min)
        for _ in range(300):
            time.sleep(2)
            status_resp = requests.get(
                f"{REMOTE_RENDER_URL}/api/render/status/{remote_job_id}",
                timeout=10,
            )
            if not status_resp.ok:
                continue
            job = status_resp.json().get("job", {})

            if job.get("status") == "done":
                # 3. Fetch rendered image
                img_resp = requests.get(
                    f"{REMOTE_RENDER_URL}/img/{job['image']}",
                    timeout=60,
                )
                if not img_resp.ok:
                    raise RemoteRenderError("failed to fetch rendered image")

                ts = int(time.time())
                filename = f"{cfg.SCENE_NAME}_remote_{ts}.png"
                local_path = Path(__file__).parent / "images" / filename
                local_path.write_bytes(img_resp.content)

                global last_render_path
                last_render_path = str(local_path)
                _append_log(f"remote render -> {last_render_path}")

                _local_jobs[job_id] = {
                    "status": "done",
                    "image": f"/img/{filename}?t={ts}",
                    "time": job.get("time", 0),
                }
                _prune_local_jobs()
                return

            if job.get("status") == "error":
                raise RemoteRenderError(job.get("error", "unknown remote error"))

        raise RemoteRenderError("remote render timed out after 10 minutes")
    except Exception as exc:
        _append_log(f"remote render failed: {exc}")
        _append_log("falling back to local render...")
        # Transparent fallback to local
        _do_local_render(job_id, payload)


@app.route("/api/render", methods=["POST"])
def api_render():
    """Create an async render job and return its ID immediately.

    The frontend polls /api/render/status/<job_id> until the job is done.
    Camera and composition overrides are applied synchronously so the
    global state stays up to date.
    """
    global composition
    payload = request.get_json() or {}
    if "camera" in payload:
        _apply_camera_override(payload["camera"])
    if "composition" in payload:
        composition = payload["composition"]

    job_id = str(uuid.uuid4())[:8]
    _local_jobs[job_id] = {"status": "pending"}

    if REMOTE_RENDER_URL:
        threading.Thread(
            target=_do_remote_render, args=(job_id, payload), daemon=True
        ).start()
    else:
        threading.Thread(
            target=_do_local_render, args=(job_id, payload), daemon=True
        ).start()

    return jsonify({"ok": True, "job_id": job_id})


@app.route("/api/render/status/<job_id>", methods=["GET"])
def api_render_status(job_id):
    job = _local_jobs.get(job_id, {"status": "unknown"})
    return jsonify({"ok": True, "job": job, "log": log_lines[-50:]})



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
