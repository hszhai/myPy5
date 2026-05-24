"""Render data/Female_Average_Head.obj with matplotlib (offscreen).

The .obj's mtllib line points at 'Female Average Head.mtl' (spaces); your
file on disk is 'Female_Average_Head.mtl' (underscores), so the .mtl link
is broken. The .mtl also references 'Female Average Head.bmp' which isn't
in data/. So this renders the geometry only -- shaded, untextured.

Run:  ~/miniconda3/bin/python render_head.py    (or: py5 render_head.py)
Output: head_render.png
"""
import os
import time
import numpy as np
import trimesh
import trimesh.transformations as tt
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

OBJ = "data/Female_Average_Head.obj"
assert os.path.exists(OBJ), OBJ

mesh = trimesh.load(OBJ, force="mesh", process=False)
print(f"loaded: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
print(f"extents (dx, dy, dz): {mesh.extents}")

# --- orient for matplotlib (Z-up), center, normalise to ~[-1, 1] -----------
mesh.apply_transform(tt.rotation_matrix(np.pi / 2, [1, 0, 0]))   # Y-up -> Z-up
mesh.apply_translation(-mesh.centroid)
mesh.apply_scale(2.0 / mesh.extents.max())

# --- Lambert shading from face normals -------------------------------------
triangles = mesh.triangles
normals = mesh.face_normals

light = np.array([0.35, -0.55, 0.75])           # upper-front-right
light = light / np.linalg.norm(light)
shade = np.clip(normals @ light, 0.18, 1.0)     # diffuse + ambient floor
skin = np.array([0.93, 0.78, 0.66])             # warm skin tone
face_colors = shade[:, None] * skin[None, :]

# --- render ---------------------------------------------------------------
fig = plt.figure(figsize=(7, 8))
ax = fig.add_subplot(111, projection="3d")

t0 = time.time()
ax.add_collection3d(
    Poly3DCollection(triangles, facecolors=face_colors, linewidths=0)
)
lim = 1.05
ax.set_xlim(-lim, lim)
ax.set_ylim(-lim, lim)
ax.set_zlim(-lim, lim)
ax.set_box_aspect((1, 1, 1))
ax.set_axis_off()
ax.view_init(elev=8, azim=-90)                  # face toward camera

out = "head_render.png"
fig.savefig(out, dpi=120, bbox_inches="tight", facecolor=(0.96, 0.96, 0.97))
print(f"rendered in {time.time() - t0:.1f}s -> {out}")
