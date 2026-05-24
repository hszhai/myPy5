"""Render data/audi.ply (a real 3D Gaussian Splatting scene, ~200k Gaussians)
on the CPU. Supports BOTH orthographic and perspective projection.

Perspective uses EWA splatting: each Gaussian's 3x3 covariance is projected
through the *local* projection Jacobian J(X, Y, Z) so the resulting 2D
covariance shrinks correctly with depth and shears at the image edges.

  J(X,Y,Z) = [[ f/Z,    0, -f X / Z^2 ],
              [   0, -f/Z,  f Y / Z^2 ]]      (y row negated for y-down image)

  Sigma_2d = J @ Sigma_cam @ J^T            (then + 0.3 I, per the 3DGS paper)

Orthographic is the same pipeline with a constant J = [[ppu,0,0],[0,-ppu,0]],
no depth division.

The Audi capture is Y-DOWN in world space (COLMAP / OpenCV convention from
photogrammetry), so a 180-deg rotation around X is applied at load time to
make it Y-up for this renderer. Toggle SCENE_UP_FLIP if your scene differs.

Run:  ~/miniconda3/bin/python render_audi.py     (or:  py5 render_audi.py)
Output: audi_render.png
"""
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============ knobs =======================================================
PLY = "data/audi.ply"
PROJECTION = "perspective"      # "perspective" or "ortho"
SCENE_UP_FLIP = True            # If the model looks upside-down, leave True.
                                # This flips the image-Y direction in the projection
                                # (handles Y-down scenes like COLMAP captures).
W, H = 900, 700
ELEV_DEG, AZIM_DEG = 15.0, 40.0
FOV_DEG = 45.0                  # perspective field of view (horizontal)
DISTANCE_K = 1.0                # camera distance = extent * DISTANCE_K (perspective)
SH_C0 = 0.28209479177387814


# ============ helpers =====================================================
def load_3dgs_ply(path):
    """Binary-LE 3DGS PLY -> numpy structured array (one row per Gaussian)."""
    with open(path, "rb") as f:
        header = []
        while True:
            line = f.readline().decode("ascii").rstrip()
            header.append(line)
            if line == "end_header":
                break
        n = next(int(l.split()[-1]) for l in header if l.startswith("element vertex"))
        props = [l.split()[-1] for l in header if l.startswith("property")]
        dt = np.dtype([(p, "<f4") for p in props])
        return np.fromfile(f, dtype=dt, count=n)


def quat_to_rotmat(q):
    """(N,4) unit quaternions (w,x,y,z) -> (N,3,3) rotation matrices."""
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    R = np.empty(q.shape[:-1] + (3, 3))
    R[..., 0, 0] = 1 - 2 * (y*y + z*z); R[..., 0, 1] = 2 * (x*y - w*z); R[..., 0, 2] = 2 * (x*z + w*y)
    R[..., 1, 0] = 2 * (x*y + w*z);     R[..., 1, 1] = 1 - 2 * (x*x + z*z); R[..., 1, 2] = 2 * (y*z - w*x)
    R[..., 2, 0] = 2 * (x*z - w*y);     R[..., 2, 1] = 2 * (y*z + w*x); R[..., 2, 2] = 1 - 2 * (x*x + y*y)
    return R


# ============ 1. load + decode ============================================
data = load_3dgs_ply(PLY)
N = len(data)
print(f"loaded {N} Gaussians from {PLY}")

xyz = np.column_stack([data["x"], data["y"], data["z"]]).astype(np.float64)
scales = np.exp(np.column_stack([data[f"scale_{i}"] for i in range(3)]).astype(np.float64))
quats = np.column_stack([data[f"rot_{i}"] for i in range(4)]).astype(np.float64)
quats /= np.linalg.norm(quats, axis=1, keepdims=True)
fdc = np.column_stack([data[f"f_dc_{i}"] for i in range(3)]).astype(np.float64)
colors = np.clip(SH_C0 * fdc + 0.5, 0.0, 1.0)
opacities = 1.0 / (1.0 + np.exp(-data["opacity"].astype(np.float64)))

Rgauss = quat_to_rotmat(quats)
cov3 = np.einsum("nij,nj,nkj->nik", Rgauss, scales ** 2, Rgauss)

# ============ 2. flip image Y if the scene came out upside-down ==========
# A scene's Y convention may be Y-down (COLMAP/OpenCV from photogrammetry).
# Rather than rotating the scene, we just flip the sign of cam_y in the
# projection -- mathematically identical to flipping the image vertically.
ysign = +1.0 if SCENE_UP_FLIP else -1.0

# ============ 3. camera ===================================================
center = np.median(xyz, axis=0)
radii = np.linalg.norm(xyz - center, axis=1)
extent = np.percentile(radii, 95) * 2.0
print(f"scene extent ~ {extent:.3f}    projection: {PROJECTION}")

elev, azim = np.radians(ELEV_DEG), np.radians(AZIM_DEG)
ce, se = np.cos(elev), np.sin(elev)
ca, sa = np.cos(azim), np.sin(azim)
Ry = np.array([[ca, 0.0, sa], [0.0, 1.0, 0.0], [-sa, 0.0, ca]])     # azim
Rx = np.array([[1.0, 0.0, 0.0], [0.0, ce, -se], [0.0, se, ce]])     # elev
Rcam = Rx @ Ry                                                       # world -> cam (rotation)

cam_xyz = (xyz - center) @ Rcam.T
cov_cam = np.einsum("ij,njk,lk->nil", Rcam, cov3, Rcam)

# ============ 4. project to image space ===================================
if PROJECTION == "perspective":
    focal = W / (2.0 * np.tan(np.radians(FOV_DEG) / 2.0))
    distance = extent * DISTANCE_K
    cam_xyz[:, 2] += distance                     # push scene in front of cam (+Z = forward)
    Z = cam_xyz[:, 2]
    valid_z = Z > 1e-3                            # behind-camera cull
    safe_Z = np.where(valid_z, Z, 1.0)
    invZ = 1.0 / safe_Z

    mean2d = np.column_stack([
        W / 2 + focal * cam_xyz[:, 0] * invZ,
        H / 2 + ysign * focal * cam_xyz[:, 1] * invZ,
    ])
    depths = safe_Z

    # per-Gaussian Jacobian (N, 2, 3) -- the heart of EWA splatting
    fZ = focal * invZ
    J = np.zeros((N, 2, 3))
    J[:, 0, 0] = fZ
    J[:, 0, 2] = -focal * cam_xyz[:, 0] * invZ * invZ
    J[:, 1, 1] = ysign * fZ
    J[:, 1, 2] = -ysign * focal * cam_xyz[:, 1] * invZ * invZ

    cov2d = np.einsum("nij,njk,nlk->nil", J, cov_cam, J)
    cov2d[:, 0, 0] += 0.3                         # 3DGS paper: keep non-degenerate
    cov2d[:, 1, 1] += 0.3
    print(f"  perspective: FOV={FOV_DEG} deg, focal={focal:.1f} px, distance={distance:.3f}")

elif PROJECTION == "ortho":
    ppu = min(W, H) / (extent * 1.2)
    mean2d = np.column_stack([
        W / 2 + cam_xyz[:, 0] * ppu,
        H / 2 + ysign * cam_xyz[:, 1] * ppu,
    ])
    depths = cam_xyz[:, 2]
    valid_z = np.ones(N, dtype=bool)

    Jortho = np.array([[ppu, 0.0, 0.0], [0.0, ysign * ppu, 0.0]])    # 2x3, constant
    cov2d = np.einsum("ij,njk,lk->nil", Jortho, cov_cam, Jortho)
    print(f"  orthographic: ppu={ppu:.2f} px/unit")

else:
    raise ValueError(f"PROJECTION must be 'perspective' or 'ortho' (got {PROJECTION!r})")

# ============ 5. cull invisible Gaussians =================================
sxx = np.sqrt(np.maximum(cov2d[:, 0, 0], 1e-12))
syy = np.sqrt(np.maximum(cov2d[:, 1, 1], 1e-12))
det = cov2d[:, 0, 0] * cov2d[:, 1, 1] - cov2d[:, 0, 1] ** 2
m = 5.0
visible = (
    valid_z &
    (opacities > 0.02) &
    (det > 1e-6) &
    (mean2d[:, 0] + 3*sxx > -m) & (mean2d[:, 0] - 3*sxx < W + m) &
    (mean2d[:, 1] + 3*syy > -m) & (mean2d[:, 1] - 3*syy < H + m) &
    (sxx + syy > 0.7)
)
keep = np.where(visible)[0]
print(f"visible: {len(keep)} / {N}  ({100*len(keep)/N:.1f}%)")
order = keep[np.argsort(-depths[keep])]              # back-to-front

m2 = mean2d[order]; C2 = cov2d[order]
col = colors[order]; op = opacities[order]

# ============ 6. splat ====================================================
img = np.full((H, W, 3), 0.06)
t0 = time.time()
for k in range(len(order)):
    cx, cy = m2[k]
    C = C2[k]
    d = C[0, 0] * C[1, 1] - C[0, 1] * C[1, 0]
    if d <= 0:
        continue
    inv00 = C[1, 1] / d
    inv11 = C[0, 0] / d
    inv01 = -C[0, 1] / d
    rx = 3.0 * np.sqrt(max(C[0, 0], 1e-6))
    ry = 3.0 * np.sqrt(max(C[1, 1], 1e-6))
    x0 = max(int(cx - rx), 0); x1 = min(int(cx + rx) + 1, W)
    y0 = max(int(cy - ry), 0); y1 = min(int(cy + ry) + 1, H)
    if x0 >= x1 or y0 >= y1:
        continue
    xs = np.arange(x0, x1) - cx
    ys = np.arange(y0, y1) - cy
    q = inv00 * (xs * xs)[None, :] + inv11 * (ys * ys)[:, None] + (2 * inv01) * np.outer(ys, xs)
    a = (op[k] * np.exp(-0.5 * q))[..., None]
    img[y0:y1, x0:x1] = img[y0:y1, x0:x1] * (1.0 - a) + col[k] * a
    if k % 30000 == 29999:
        print(f"  splat {k+1}/{len(order)}   {time.time() - t0:.1f}s")

print(f"\nsplat time: {time.time() - t0:.1f}s")
plt.imsave("images/audi_render_2.png", np.clip(img, 0, 1))
print("rendered -> images/audi_render_2.png")
