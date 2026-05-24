"""Render the canonical splat for the RedHead scene.

Scene config (PLY path, camera, resolution, credit) now lives in
scenes/redhead.json. Edit that file directly, or use preview_splat.py
to position the camera and save -- this script just reads it.

Run:  ~/miniconda3/bin/python render_redhead.py   (or:  py5 render_redhead.py)
Output: images/redhead_render.png  (path comes from the scene file)
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gsplat import (
    load_3dgs_ply, decode_3dgs, make_camera,
    project_perspective, project_ortho, cull, splat,
)
from scene_io import load_scene


# Re-export scene constants so legacy `import render_redhead as cfg` keeps working.
_cfg = load_scene("redhead")
SCENE_NAME   = _cfg.SCENE_NAME
PLY          = _cfg.PLY
OUT          = _cfg.OUT
SCENE_UP_FLIP = _cfg.SCENE_UP_FLIP
W, H         = _cfg.W, _cfg.H
ELEV_DEG     = _cfg.ELEV_DEG
AZIM_DEG     = _cfg.AZIM_DEG
DISTANCE_K   = _cfg.DISTANCE_K
FOV_DEG      = _cfg.FOV_DEG
HEAD_BIAS_X  = _cfg.HEAD_BIAS_X
HEAD_BIAS_Y  = _cfg.HEAD_BIAS_Y
PROJECTION   = "perspective"


if __name__ == "__main__":
    data = load_3dgs_ply(PLY)
    G = decode_3dgs(data)
    print(f"loaded {len(data)} Gaussians from {PLY}")

    ysign = +1.0 if SCENE_UP_FLIP else -1.0
    center = np.median(G["xyz"], axis=0)
    center[0] += HEAD_BIAS_X
    center[1] += HEAD_BIAS_Y
    radii = np.linalg.norm(G["xyz"] - center, axis=1)
    extent = np.percentile(radii, 90) * 2.0
    print(f"center: {center}   extent ~ {extent:.3f}    projection: {PROJECTION}")

    Rcam = make_camera(ELEV_DEG, AZIM_DEG)
    cam_xyz = (G["xyz"] - center) @ Rcam.T
    cov_cam = np.einsum("ij,njk,lk->nil", Rcam, G["cov3"], Rcam)

    if PROJECTION == "perspective":
        focal = W / (2.0 * np.tan(np.radians(FOV_DEG) / 2.0))
        distance = extent * DISTANCE_K
        mean2d, cov2d, depths, valid_z = project_perspective(
            cam_xyz, cov_cam, focal, distance, W, H, ysign)
        print(f"  perspective: FOV={FOV_DEG} deg, focal={focal:.1f} px, "
              f"distance={distance:.3f}")
    else:
        ppu = min(W, H) / (extent * 1.2)
        mean2d, cov2d, depths, valid_z = project_ortho(
            cam_xyz, cov_cam, ppu, W, H, ysign)
        print(f"  orthographic: ppu={ppu:.2f} px/unit")

    keep = cull(mean2d, cov2d, G["opacities"], valid_z, W, H)
    print(f"visible: {len(keep)} / {len(data)}  ({100 * len(keep) / len(data):.1f}%)")
    order = keep[np.argsort(-depths[keep])]

    img = splat(W, H, mean2d, cov2d, G["colors"], G["opacities"], order)
    plt.imsave(OUT, np.clip(img, 0, 1))
    print(f"rendered -> {OUT}")
