"""Scene I/O: read & write scene config JSON files at scenes/<name>.json.

A scene file holds everything the tools need to know about one 3DGS
.ply asset: where it is, where its canonical splat render goes, and
the camera framing that defines the composition.

  preview_splat.py     writes (and reads) the scene file.
  render_<scene>.py    reads to render the canonical splat -> images/<name>_render.png
  render_*.py + tune_* read to know where to project from and read the
                        reference image for saliency + colour.

Legacy flat fields:
  name           short scene name (matches the json basename)
  ply            path to the .ply
  out            path the canonical splat render is written to
  credit         attribution (CC-BY etc.)
  scene_up_flip  flip image-Y for Y-down captures (default True)
  w, h           render resolution
  elev_deg       camera elevation (deg)
  azim_deg       camera azimuth (deg)
  distance_k     camera distance = scene_extent * distance_k
  fov_deg        perspective horizontal FOV (deg)
  head_bias_x    framing bias along scene X
  head_bias_y    framing bias along scene Y

Newer scene-language files may also use nested sections:

  model: { ply, credit, scene_up_flip }
  render: { out, w, h }
  camera: { elev_deg, azim_deg, distance_k, fov_deg, head_bias_x, head_bias_y }
  surface_walks: { n_walkers, steps, step_radius_px, forward_bias, ... }
"""
import json
import os
from types import SimpleNamespace

SCENES_DIR = "scenes"


def _resolve_path(name):
    if name.endswith(".json"):
        return name
    return os.path.join(SCENES_DIR, f"{name}.json")


def _section(data, key):
    value = data.get(key, {})
    return value if isinstance(value, dict) else {}


def _pick(data, section, key, default=None):
    nested = _section(data, section)
    if key in nested:
        return nested[key]
    return data.get(key, default)


def load_scene(name):
    """Load scenes/<name>.json into a namespace.

    Attribute names match the constants used by the legacy
    render_<scene>.py modules (PLY, OUT, SCENE_UP_FLIP, W, H,
    ELEV_DEG, ...), so existing imports keep working unchanged.
    """
    path = _resolve_path(name)
    with open(path) as f:
        data = json.load(f)
    model = _section(data, "model")
    render = _section(data, "render")
    camera = _section(data, "camera")
    surface_walks = _section(data, "surface_walks")
    return SimpleNamespace(
        SCENE_NAME=data.get("name", os.path.splitext(os.path.basename(path))[0]),
        PLY=_pick(data, "model", "ply"),
        OUT=_pick(data, "render", "out"),
        SCENE_UP_FLIP=bool(model.get("scene_up_flip", data.get("scene_up_flip", True))),
        W=int(render.get("w", data.get("w", 1080))),
        H=int(render.get("h", data.get("h", 1440))),
        ELEV_DEG=float(camera.get("elev_deg", data.get("elev_deg", 0.0))),
        AZIM_DEG=float(camera.get("azim_deg", data.get("azim_deg", 0.0))),
        DISTANCE_K=float(camera.get("distance_k", data.get("distance_k", 1.5))),
        FOV_DEG=float(camera.get("fov_deg", data.get("fov_deg", 28.0))),
        HEAD_BIAS_X=float(camera.get("head_bias_x", data.get("head_bias_x", 0.0))),
        HEAD_BIAS_Y=float(camera.get("head_bias_y", data.get("head_bias_y", 0.0))),
        CREDIT=model.get("credit", data.get("credit", "")),
        MODEL=model,
        RENDER=render,
        CAMERA=camera,
        SURFACE_WALKS=surface_walks,
        PATH=path,
        _raw=data,
    )


def save_scene(name, **fields):
    """Write scenes/<name>.json, merging into any existing file (so
    extra keys -- e.g., a credit line -- aren't lost when the camera
    is re-saved)."""
    path = _resolve_path(name)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    existing = {}
    if os.path.exists(path):
        try:
            with open(path) as f:
                existing = json.load(f)
        except Exception:
            existing = {}
    existing.update(fields)
    if "name" not in existing:
        existing["name"] = os.path.splitext(os.path.basename(path))[0]
    with open(path, "w") as f:
        json.dump(existing, f, indent=2)
    return path


def scene_exists(name):
    return os.path.exists(_resolve_path(name))


def scene_path(name):
    return _resolve_path(name)
