"""Minimal headless render server for remote GPU rendering.

Run on the GPU machine (e.g. Google Colab):
  SCENE_NAME=redhead python remote_server.py

Then on your local machine:
  REMOTE_RENDER_URL=http://<tunnel-url> python app.py
"""
import os
import threading
import time
import traceback
import uuid
from collections import OrderedDict
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

# Job queue (job_id -> status dict)
_jobs = OrderedDict()
_MAX_JOBS = 10


def _prune_jobs():
    while len(_jobs) > _MAX_JOBS:
        _jobs.popitem(last=False)


def _run_render_job(job_id: str, payload: dict):
    try:
        # Apply camera override
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

        reproject_scene(scene_data)
        composition = payload.get("composition")

        t0 = time.time()
        _, out_path, _ = render_composition(
            SCENE_REF,
            composition=composition,
            write=True,
            stamp_label=True,
            scene_data=scene_data,
        )
        dt = time.time() - t0
        print(f"[remote] job {job_id} done in {dt:.1f}s -> {out_path}")

        _jobs[job_id] = {
            "status": "done",
            "image": Path(out_path).name,
            "time": dt,
        }
        _prune_jobs()
    except Exception as exc:
        tb = traceback.format_exc()
        print(f"[remote] job {job_id} failed: {exc}\n{tb}")
        _jobs[job_id] = {"status": "error", "error": str(exc)}
        _prune_jobs()


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
    job_id = str(uuid.uuid4())[:8]
    _jobs[job_id] = {"status": "pending"}
    threading.Thread(
        target=_run_render_job, args=(job_id, payload), daemon=True
    ).start()
    return jsonify({"ok": True, "job_id": job_id})


@app.route("/api/render/status/<job_id>", methods=["GET"])
def api_render_status(job_id):
    job = _jobs.get(job_id, {"status": "unknown"})
    return jsonify({"ok": True, "job": job})


@app.route("/img/<path:filename>")
def serve_image(filename):
    return send_from_directory(Path(__file__).parent / "images", filename)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5050))
    print(f"[remote] listening on http://0.0.0.0:{port}")
    app.run(debug=False, port=port, host="0.0.0.0", use_reloader=False, threaded=False)
