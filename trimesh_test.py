"""Trimesh setup test — create a 3D object and render it to an image.

Run with:  ~/miniconda3/bin/python trimesh_test.py
(or, with the alias from ~/.zshrc:  py5 trimesh_test.py)

Renders offscreen with matplotlib's Agg backend, so it never opens a
window — it just writes trimesh_render.png next to this script.
"""
import numpy as np
import trimesh

import matplotlib
matplotlib.use("Agg")  # offscreen renderer — no window, no GL context needed
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# --- 1. create a 3D object with trimesh -----------------------------------
mesh = trimesh.creation.icosphere(subdivisions=2, radius=1.0)

print("=== trimesh mesh info ===")
print(f"trimesh version : {trimesh.__version__}")
print(f"vertices        : {len(mesh.vertices)}")
print(f"faces           : {len(mesh.faces)}")
print(f"watertight      : {mesh.is_watertight}")
print(f"volume          : {mesh.volume:.4f}")
print(f"surface area    : {mesh.area:.4f}")
print(f"bounding box    : {mesh.bounds.tolist()}")

# --- 2. render it: flat Lambert shading from a single light ---------------
triangles = mesh.triangles        # (n_faces, 3, 3) world-space triangle corners
normals = mesh.face_normals       # (n_faces, 3) unit normals

light = np.array([0.4, 0.5, 0.8])
light = light / np.linalg.norm(light)
shade = np.clip(normals @ light, 0.18, 1.0)   # diffuse term with an ambient floor

base_color = np.array([0.25, 0.55, 0.95])     # blue
face_colors = shade[:, None] * base_color[None, :]

fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, projection="3d")
ax.add_collection3d(
    Poly3DCollection(
        triangles,
        facecolors=face_colors,
        edgecolors=(0, 0, 0, 0.25),
        linewidths=0.3,
    )
)

lim = 1.05
ax.set_xlim(-lim, lim)
ax.set_ylim(-lim, lim)
ax.set_zlim(-lim, lim)
ax.set_box_aspect((1, 1, 1))
ax.set_axis_off()
ax.view_init(elev=22, azim=35)

out_path = "trimesh_render.png"
fig.savefig(out_path, dpi=130, bbox_inches="tight", facecolor="white")
print(f"\nrendered -> {out_path}")
