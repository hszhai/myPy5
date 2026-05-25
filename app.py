"""Web-based UI for tune_layers using Flask + Tailwind CSS."""
import os
import json
import time
import io
import contextlib
import traceback
from pathlib import Path
from collections import OrderedDict

from flask import Flask, render_template, jsonify, request
import numpy as np

from render_layers import (
    DEFAULT_COMPOSITION, DEFAULT_WALKS, LAYER_PARAM_SCHEMAS,
    WALK_KEY_MAP, build_scene_data, layer_param_effective, layer_param_set,
    project_anchor, render_composition,
)
from scene_io import load_scene, scene_path


app = Flask(__name__, template_folder='templates')
app.config['JSON_SORT_KEYS'] = False

# Load scene from env or default to 'siyun'
SCENE_REF = os.environ.get("SCENE_FILE") or os.environ.get("SCENE_NAME", "siyun")
cfg = load_scene(SCENE_REF)
SCENE_JSON = cfg.PATH if hasattr(cfg, "PATH") else scene_path(SCENE_REF)
SCENE_WALKS = getattr(cfg, "SURFACE_WALKS", {}) or {}

# Global state
state = {
    "composition": None,
    "scene_data": None,
    "layer_cache": OrderedDict(),
    "log_lines": [],
    "last_render_path": None,
    "cam_azimuth": float(getattr(cfg, "AZIM_DEG", 0)),
    "cam_elevation": float(getattr(cfg, "ELEV_DEG", 0)),
    "cam_fov": float(getattr(cfg, "FOV_DEG", 40)),
    "cam_distance_k": float(getattr(cfg, "DISTANCE_K", 1.0)),
}

print("[app] loading scene data...")
state["scene_data"] = build_scene_data(SCENE_REF)
state["composition"] = DEFAULT_COMPOSITION.copy()
print("[app] ready")


# ---- API Routes ----------------------------------------------------------
@app.route("/")
def index():
    """Serve the main UI."""
    return render_template("index.html", scene_json=SCENE_JSON)


@app.route("/api/state", methods=["GET"])
def get_state():
    """Return current composition, camera params, and scene info."""
    bb_min, bb_max = state["scene_data"]["bbox"]
    return jsonify({
        "composition": state["composition"],
        "camera": {
            "azimuth": state["cam_azimuth"],
            "elevation": state["cam_elevation"],
            "fov": state["cam_fov"],
            "distance_k": state["cam_distance_k"],
        },
        "scene": {
            "bbox_min": bb_min.tolist(),
            "bbox_max": bb_max.tolist(),
            "scene_json": SCENE_JSON,
        },
        "log": state["log_lines"][-50:],  # Last 50 lines
    })


@app.route("/api/state", methods=["POST"])
def update_state():
    """Update composition or camera params."""
    data = request.get_json()
    
    if "composition" in data:
        state["composition"] = data["composition"]
    
    if "camera" in data:
        cam = data["camera"]
        state["cam_azimuth"] = float(cam.get("azimuth", state["cam_azimuth"]))
        state["cam_elevation"] = float(cam.get("elevation", state["cam_elevation"]))
        state["cam_fov"] = float(cam.get("fov", state["cam_fov"]))
        state["cam_distance_k"] = float(cam.get("distance_k", state["cam_distance_k"]))
    
    return jsonify({"status": "ok"})


def _log_append(text):
    """Append to rolling log buffer."""
    for line in text.splitlines():
        if line.strip():
            state["log_lines"].append(line)
    if len(state["log_lines"]) > 200:
        state["log_lines"] = state["log_lines"][-200:]


@app.route("/api/render", methods=["POST"])
def do_render():
    """Run the composition render and return the output image path."""
    _log_append(f"--- render @ {time.strftime('%H:%M:%S')} ---")
    t0 = time.time()
    out_path = None
    buf = io.StringIO()

    try:
        with contextlib.redirect_stdout(buf):
            _, out_path, _ = render_composition(
                SCENE_REF, 
                composition=state["composition"],
                write=True, 
                stamp_label=True,
                scene_data=state["scene_data"],
                layer_cache=state["layer_cache"],
            )
    except Exception as exc:
        _log_append(f"ERROR: {exc}")
        _log_append(traceback.format_exc())
        return jsonify({"error": str(exc)}), 500

    captured = buf.getvalue()
    _log_append(captured)
    if captured:
        print(captured, end="")
    
    if out_path:
        state["last_render_path"] = out_path

    dt = time.time() - t0
    msg = (f"render done in {dt:.1f}s" if out_path else f"render FAILED ({dt:.1f}s)")
    _log_append(msg)

    return jsonify({
        "status": "ok",
        "output": out_path,
        "time": dt,
        "log": state["log_lines"][-50:],
    })


@app.route("/api/save_scene", methods=["POST"])
def save_scene():
    """Save current composition back to the scene file."""
    try:
        with open(SCENE_JSON, "w") as f:
            json.dump(state["composition"], f, indent=2)
        _log_append(f"Saved scene to {SCENE_JSON}")
        return jsonify({"status": "ok", "path": SCENE_JSON})
    except Exception as exc:
        _log_append(f"Save failed: {exc}")
        return jsonify({"error": str(exc)}), 500


@app.route("/api/add_layer", methods=["POST"])
def add_layer():
    """Add a new layer to the composition."""
    data = request.get_json()
    layer_type = data.get("type", "base_splat")
    
    if "layers" not in state["composition"]:
        state["composition"]["layers"] = []
    
    if layer_type == "base_splat":
        new_layer = {
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
    elif layer_type == "surface_walks":
        new_layer = {
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
            },
        }
    else:
        return jsonify({"error": f"Unknown layer type: {layer_type}"}), 400
    
    state["composition"]["layers"].append(new_layer)
    return jsonify({"status": "ok", "layer": new_layer})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"[app] starting on http://localhost:{port}")
    app.run(debug=True, port=port, use_reloader=False)
