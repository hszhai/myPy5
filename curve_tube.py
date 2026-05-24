"""3D curve generation + render: sweep a space curve into a tube mesh.

Pipeline:
  1. numpy   -> generate a 3D curve (a trefoil knot)
  2. trimesh -> sweep a circular cross-section along the curve = a tube mesh
  3. matplotlib (Agg) -> render the tube offscreen to a PNG (no window)

Run:  ~/miniconda3/bin/python curve_tube.py     (or:  py5 curve_tube.py)
"""
import numpy as np
import trimesh
import shapely.geometry as sg

import matplotlib
matplotlib.use("Agg")  # offscreen, no window
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# --- 1. generate a 3D curve: a trefoil knot -------------------------------
n = 260
t = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
curve = np.column_stack([
    np.sin(t) + 2.0 * np.sin(2.0 * t),
    np.cos(t) - 2.0 * np.cos(2.0 * t),
    -np.sin(3.0 * t),
])

# --- 2. sweep a circular cross-section along the curve -> tube mesh --------
tube_radius = 0.4
cross_section = sg.Point(0.0, 0.0).buffer(tube_radius, quad_segs=6)  # 24-sided
tube = trimesh.creation.sweep_polygon(cross_section, curve)

print("=== curve -> tube mesh ===")
print(f"curve points : {n}")
print(f"vertices     : {len(tube.vertices)}")
print(f"faces        : {len(tube.faces)}")
print(f"watertight   : {tube.is_watertight}")
print(f"volume       : {tube.volume:.3f}")

# --- 3. render offscreen with matplotlib ----------------------------------
triangles = tube.triangles
normals = tube.face_normals

light = np.array([0.3, 0.4, 0.85])
light = light / np.linalg.norm(light)
shade = np.clip(normals @ light, 0.20, 1.0)        # diffuse + ambient floor

base_color = np.array([0.95, 0.45, 0.18])          # orange
face_colors = shade[:, None] * base_color[None, :]

fig = plt.figure(figsize=(7, 6))
ax = fig.add_subplot(111, projection="3d")
ax.add_collection3d(Poly3DCollection(triangles, facecolors=face_colors, linewidths=0))

ctr = tube.bounds.mean(axis=0)
rad = (tube.bounds[1] - tube.bounds[0]).max() / 2 * 1.05
ax.set_xlim(ctr[0] - rad, ctr[0] + rad)
ax.set_ylim(ctr[1] - rad, ctr[1] + rad)
ax.set_zlim(ctr[2] - rad, ctr[2] + rad)
ax.set_box_aspect((1, 1, 1))
ax.set_axis_off()
ax.view_init(elev=28, azim=40)

out_path = "curve_tube_render.png"
fig.savefig(out_path, dpi=130, bbox_inches="tight", facecolor="white")
print(f"\nrendered -> {out_path}")
