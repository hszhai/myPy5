"""Minimal headless render server for remote GPU rendering.

Run on the GPU machine (e.g. Google Colab):
  SCENE_NAME=redhead python remote_server.py

Then on your local machine:
  REMOTE_RENDER_URL=http://<tunnel-url> python app.py
"""
import os
import time
import traceback
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

from render_layers import build_scene_data, render_composition, reproject_scene
from scene_io import load_scene

# Optional CORS so a direct browser connection works without a local proxy
try:
    from flask_cors import CORS
except ImportError:
    CORS = None

SCENE_REF = os.environ.get("SCENE_FILE") or os.environ.get("SCENE_NAME", "siyun")
print(f"[remote] loading scene: {SCENE_REF}")
scene_data = build_scene_data(SCENE_REF)
cfg = scene_data["cfg"]

app = Flask(__name__, static_folder="images")
if CORS is not None:
    CORS(app)


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "ok": True,
        "scene": cfg.SCENE_NAME,
        "render_size": [int(cfg.W), int(cfg.H)],
    })


@app.route("/api/render", methods=["POST"])
def api_render():
    payload = request.get_json() or {}

    # Apply camera override from request
    if "camera" in payload:
        cam = payload["camera"]
        for key, attr in (
            ("elev_deg", "ELEV_DEG"),
            ("azim_deg", "AZIM_DEG"),
            ("fov_deg", "FOV_DEG"),
            ("distance_k", "DISTANCE_K"),
            ("head_bias_x", "HEAD_BIAS_X"),
            ("head_bias_y", "HEAD_BIAS_Y"),
        ):
            if key in cam and cam[key] is not None:
                setattr(cfg, attr, float(cam[key]))

    # Reproject for the new camera
    reproject_scene(scene_data)

    composition = payload.get("composition")

    t0 = time.time()
    try:
        _, out_path, _ = render_composition(
            SCENE_REF,
            composition=composition,
            write=True,
            stamp_label=True,
            scene_data=scene_data,
        )
    except Exception as exc:
        tb = traceback.format_exc()
        print(f"[remote render error] {exc}\n{tb}")
        return jsonify({"error": str(exc), "trace": tb}), 500

    dt = time.time() - t0
    print(f"[remote] rendered in {dt:.1f}s -> {out_path}")

    return send_from_directory(Path(__file__).parent / "images", Path(out_path).name)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5050))
    print(f"[remote] listening on http://0.0.0.0:{port}")
    app.run(debug=False, port=port, host="0.0.0.0", use_reloader=False, threaded=False)
