"""Shared CPU 3D Gaussian Splatting helpers used by the per-scene
render scripts (render_audi.py, render_siyun.py, ...).

  load_3dgs_ply(path)         -> numpy structured array (one row / Gaussian)
  decode_3dgs(data)           -> dict of rendering-ready arrays incl. 3x3 covariance
  make_camera(elev, azim)     -> world->camera rotation
  project_perspective(...)    -> EWA splat with per-Gaussian Jacobian
  project_ortho(...)          -> orthographic projection with constant Jacobian
  cull(...)                   -> indices of Gaussians worth splatting
  splat(...)                  -> alpha-composite back-to-front into an image

The orthographic and perspective paths use the same downstream code; only
their projection step differs. Both flip image-y via `ysign` for scenes
whose world Y is "down" (the COLMAP / OpenCV convention from photogrammetry).
"""
import time
import numpy as np

try:
    from numba import njit
except ImportError:
    njit = None

SH_C0 = 0.28209479177387814        # DC spherical-harmonic basis value


# ---- I/O -----------------------------------------------------------------
def load_3dgs_ply(path):
    """Read a binary-LE 3DGS PLY into a numpy structured array."""
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


# ---- decoding ------------------------------------------------------------
def quat_to_rotmat(q):
    """(N,4) unit quaternions (w,x,y,z) -> (N,3,3) rotation matrices."""
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    R = np.empty(q.shape[:-1] + (3, 3))
    R[..., 0, 0] = 1 - 2*(y*y + z*z); R[..., 0, 1] = 2*(x*y - w*z); R[..., 0, 2] = 2*(x*z + w*y)
    R[..., 1, 0] = 2*(x*y + w*z);     R[..., 1, 1] = 1 - 2*(x*x + z*z); R[..., 1, 2] = 2*(y*z - w*x)
    R[..., 2, 0] = 2*(x*z - w*y);     R[..., 2, 1] = 2*(y*z + w*x); R[..., 2, 2] = 1 - 2*(x*x + y*y)
    return R


def decode_3dgs(data):
    """Decode a 3DGS PLY structured array into rendering-ready arrays.

    Returns a dict with:
      xyz       (N,3)   world positions
      scales    (N,3)   per-axis std-dev (after exp)
      colors    (N,3)   DC-only RGB color in [0,1]
      opacities (N,)    after sigmoid
      cov3      (N,3,3) 3D covariance Sigma = R diag(scale^2) R^T

    The .ply may contain higher-order SH coefficients (f_rest_*) -- we
    ignore them here, so the render is view-independent.
    """
    xyz = np.column_stack([data["x"], data["y"], data["z"]]).astype(np.float64)
    scales = np.exp(np.column_stack([data[f"scale_{i}"] for i in range(3)]).astype(np.float64))
    quats = np.column_stack([data[f"rot_{i}"] for i in range(4)]).astype(np.float64)
    quats /= np.linalg.norm(quats, axis=1, keepdims=True)
    fdc = np.column_stack([data[f"f_dc_{i}"] for i in range(3)]).astype(np.float64)
    colors = np.clip(SH_C0 * fdc + 0.5, 0.0, 1.0)
    opacities = 1.0 / (1.0 + np.exp(-data["opacity"].astype(np.float64)))

    Rgauss = quat_to_rotmat(quats)
    cov3 = np.einsum("nij,nj,nkj->nik", Rgauss, scales ** 2, Rgauss)
    return {
        "xyz": xyz, "scales": scales, "colors": colors,
        "opacities": opacities, "cov3": cov3,
    }


def compute_gaussian_normals(cov3):
    """Compute surface normals from 3D Gaussian covariances.
    
    Uses eigenvalue decomposition: the smallest eigenvector of the covariance
    matrix is the direction of least spread, which approximates the surface normal.
    
    cov3: (N,3,3) array of 3x3 covariance matrices
    Returns: (N,3) unit normals
    """
    normals = np.zeros((len(cov3), 3), dtype=np.float64)
    for i in range(len(cov3)):
        eigenvalues, eigenvectors = np.linalg.eigh(cov3[i])
        normals[i] = eigenvectors[:, 0]
    return normals


# ---- camera --------------------------------------------------------------
def make_camera(elev_deg, azim_deg):
    """Build the world->camera rotation matrix (Y-up world convention)."""
    elev, azim = np.radians(elev_deg), np.radians(azim_deg)
    ce, se = np.cos(elev), np.sin(elev)
    ca, sa = np.cos(azim), np.sin(azim)
    Ry = np.array([[ca, 0.0, sa], [0.0, 1.0, 0.0], [-sa, 0.0, ca]])
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, ce, -se], [0.0, se, ce]])
    return Rx @ Ry


# ---- projection ----------------------------------------------------------
def project_perspective(cam_xyz, cov_cam, focal, distance, W, H, ysign):
    """Perspective projection with the EWA splat Jacobian.

    cam_xyz : (N,3) camera-space positions BEFORE pushing forward by `distance`.
    cov_cam : (N,3,3) camera-space 3D covariance.
    ysign   : +1 if image-Y should map directly from cam_y (data is Y-down),
              -1 for Y-up data.

    Returns mean2d (N,2), cov2d (N,2,2), depths (N,), valid_z (N,) bool.
    """
    cam_xyz = cam_xyz.copy()
    cam_xyz[:, 2] += distance                  # push scene in front of camera
    Z = cam_xyz[:, 2]
    valid_z = Z > 1e-3
    safe_Z = np.where(valid_z, Z, 1.0)
    invZ = 1.0 / safe_Z

    mean2d = np.column_stack([
        W/2 + focal * cam_xyz[:, 0] * invZ,
        H/2 + ysign * focal * cam_xyz[:, 1] * invZ,
    ])

    fZ = focal * invZ
    N = len(cam_xyz)
    J = np.zeros((N, 2, 3))
    J[:, 0, 0] = fZ
    J[:, 0, 2] = -focal * cam_xyz[:, 0] * invZ * invZ
    J[:, 1, 1] = ysign * fZ
    J[:, 1, 2] = -ysign * focal * cam_xyz[:, 1] * invZ * invZ
    cov2d = np.einsum("nij,njk,nlk->nil", J, cov_cam, J)
    cov2d[:, 0, 0] += 0.3                      # 3DGS regulariser
    cov2d[:, 1, 1] += 0.3
    return mean2d, cov2d, safe_Z, valid_z


def project_ortho(cam_xyz, cov_cam, ppu, W, H, ysign):
    """Orthographic projection (constant 2x3 Jacobian)."""
    mean2d = np.column_stack([
        W/2 + cam_xyz[:, 0] * ppu,
        H/2 + ysign * cam_xyz[:, 1] * ppu,
    ])
    Jortho = np.array([[ppu, 0.0, 0.0], [0.0, ysign * ppu, 0.0]])
    cov2d = np.einsum("ij,njk,lk->nil", Jortho, cov_cam, Jortho)
    valid_z = np.ones(len(cam_xyz), dtype=bool)
    return mean2d, cov2d, cam_xyz[:, 2], valid_z


# ---- visibility ----------------------------------------------------------
def cull(mean2d, cov2d, opacities, valid_z, W, H, *, sub_pixel=0.7, op_thresh=0.02):
    """Indices of Gaussians worth splatting (on-screen, opaque enough, finite)."""
    sxx = np.sqrt(np.maximum(cov2d[:, 0, 0], 1e-12))
    syy = np.sqrt(np.maximum(cov2d[:, 1, 1], 1e-12))
    det = cov2d[:, 0, 0] * cov2d[:, 1, 1] - cov2d[:, 0, 1] ** 2
    m = 5.0
    visible = (
        valid_z &
        (opacities > op_thresh) &
        (det > 1e-6) &
        (mean2d[:, 0] + 3*sxx > -m) & (mean2d[:, 0] - 3*sxx < W + m) &
        (mean2d[:, 1] + 3*syy > -m) & (mean2d[:, 1] - 3*syy < H + m) &
        (sxx + syy > sub_pixel)
    )
    return np.where(visible)[0]


# ---- the splat loop ------------------------------------------------------

def _splat_fallback(W, H, mean2d, cov2d, colors, opacities, order, bg):
    """Pure-Python fallback (creates temporaries per-Gaussian)."""
    img = np.full((H, W, 3), bg, dtype=np.float64)
    m2 = mean2d[order]
    C2 = cov2d[order]
    col = colors[order]
    op = opacities[order]
    for k in range(len(order)):
        cx, cy = m2[k]
        C = C2[k]
        d = C[0, 0]*C[1, 1] - C[0, 1]*C[1, 0]
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
    return img


if njit is not None:
    @njit(cache=True)
    def _splat_numba(W, H, m2, C2, col, op, bg0, bg1, bg2):
        """Numba-compiled core: zero temporary allocations per Gaussian.
        Arrays are pre-sliced by `order` in the Python wrapper."""
        img = np.empty((H, W, 3), dtype=np.float64)
        for y in range(H):
            for x in range(W):
                img[y, x, 0] = bg0
                img[y, x, 1] = bg1
                img[y, x, 2] = bg2

        n = len(m2)
        for k in range(n):
            cx = m2[k, 0]
            cy = m2[k, 1]
            C00 = C2[k, 0, 0]
            C01 = C2[k, 0, 1]
            C11 = C2[k, 1, 1]
            d = C00 * C11 - C01 * C01
            if d <= 0:
                continue
            inv00 = C11 / d
            inv11 = C00 / d
            inv01 = -C01 / d
            rx = 3.0 * np.sqrt(max(C00, 1e-6))
            ry = 3.0 * np.sqrt(max(C11, 1e-6))
            x0 = max(int(cx - rx), 0)
            x1 = min(int(cx + rx) + 1, W)
            y0 = max(int(cy - ry), 0)
            y1 = min(int(cy + ry) + 1, H)
            if x0 >= x1 or y0 >= y1:
                continue

            op_k = op[k]
            col0 = col[k, 0]
            col1 = col[k, 1]
            col2 = col[k, 2]

            for yi in range(y0, y1):
                dy = yi - cy
                for xi in range(x0, x1):
                    dx = xi - cx
                    q = inv00 * dx * dx + inv11 * dy * dy + 2.0 * inv01 * dx * dy
                    a = op_k * np.exp(-0.5 * q)
                    img[yi, xi, 0] = img[yi, xi, 0] * (1.0 - a) + col0 * a
                    img[yi, xi, 1] = img[yi, xi, 1] * (1.0 - a) + col1 * a
                    img[yi, xi, 2] = img[yi, xi, 2] * (1.0 - a) + col2 * a

        return img


def splat(W, H, mean2d, cov2d, colors, opacities, order, *,
          bg=(0.06, 0.06, 0.06), verbose=True):
    """Alpha-composite Gaussians back-to-front into a (H,W,3) image."""
    t0 = time.time()
    if njit is not None:
        img = _splat_numba(W, H,
                           mean2d[order], cov2d[order], colors[order], opacities[order],
                           float(bg[0]), float(bg[1]), float(bg[2]))
    else:
        img = _splat_fallback(W, H, mean2d, cov2d, colors, opacities, order, bg)
    if verbose:
        print(f"  splat total: {time.time() - t0:.1f}s")
    return img
