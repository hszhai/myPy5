"""Rotating turntable: Gaussian-splat render of a 3D curve, animated.

Reuses the CPU splat rasterizer from curve_splat.py, but the camera azimuth
advances each frame. Frames are stitched into an animated GIF with Pillow.
No GPU, no plotly, no extra installs.

Run:  ~/miniconda3/bin/python curve_splat_turntable.py
Output: curve_splat_turntable.gif
"""
import colorsys
import time
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

# ===== one-time setup: curve + per-point 3D Gaussians =====================
N = 600
t = np.linspace(0.0, 2.0 * np.pi, N, endpoint=False)
curve = np.column_stack([
    np.sin(t) + 2.0 * np.sin(2.0 * t),
    np.cos(t) - 2.0 * np.cos(2.0 * t),
    -np.sin(3.0 * t),
])

tangent = np.roll(curve, -1, axis=0) - np.roll(curve, 1, axis=0)
tangent /= np.linalg.norm(tangent, axis=1, keepdims=True)
helper = np.tile(np.array([0.0, 0.0, 1.0]), (N, 1))
helper[np.abs(tangent[:, 2]) > 0.9] = np.array([1.0, 0.0, 0.0])
u = np.cross(tangent, helper); u /= np.linalg.norm(u, axis=1, keepdims=True)
v = np.cross(tangent, u)

seg = np.linalg.norm(np.roll(curve, -1, axis=0) - curve, axis=1).mean()
s_long, s_thin = seg * 1.6, 0.022
var = np.array([s_long, s_thin, s_thin]) ** 2
R = np.stack([tangent, u, v], axis=2)
cov3 = R @ (np.eye(3) * var) @ np.transpose(R, (0, 2, 1))

colors = np.array([colorsys.hsv_to_rgb(h, 0.85, 1.0)
                   for h in t / (2.0 * np.pi)])
opacity = 0.85
center = curve.mean(axis=0)

# ===== per-frame splat ====================================================
W, H = 640, 540
bg = np.array([0.04, 0.05, 0.09])


def render(azim_deg: float, elev_deg: float = 22.0) -> np.ndarray:
    elev, azim = np.radians(elev_deg), np.radians(azim_deg)
    ce, se = np.cos(elev), np.sin(elev)
    ca, sa = np.cos(azim), np.sin(azim)
    Rz = np.array([[ca, -sa, 0.0], [sa, ca, 0.0], [0.0, 0.0, 1.0]])
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, ce, -se], [0.0, se, ce]])
    Rcam = Rx @ Rz

    cam = (curve - center) @ Rcam.T
    depth = cam[:, 2]
    ppu = min(W, H) / (np.abs(cam[:, :2]).max() * 2.4)
    mean2d = np.column_stack([W / 2 + cam[:, 0] * ppu,
                              H / 2 - cam[:, 1] * ppu])
    cov_cam = Rcam @ cov3 @ Rcam.T
    cov2d = cov_cam[:, :2, :2] * (ppu ** 2)

    img = np.tile(bg, (H, W, 1)).astype(float)
    for i in np.argsort(depth)[::-1]:
        cx, cy = mean2d[i]
        C = cov2d[i]
        det = C[0, 0] * C[1, 1] - C[0, 1] * C[1, 0]
        if det <= 1e-9:
            continue
        inv = np.array([[C[1, 1], -C[0, 1]], [-C[1, 0], C[0, 0]]]) / det
        rx = 3 * np.sqrt(max(C[0, 0], 1e-6))
        ry = 3 * np.sqrt(max(C[1, 1], 1e-6))
        x0, x1 = max(int(cx - rx), 0), min(int(cx + rx) + 1, W)
        y0, y1 = max(int(cy - ry), 0), min(int(cy + ry) + 1, H)
        if x0 >= x1 or y0 >= y1:
            continue
        dx, dy = np.meshgrid(np.arange(x0, x1) - cx, np.arange(y0, y1) - cy)
        q = inv[0, 0] * dx * dx + 2 * inv[0, 1] * dx * dy + inv[1, 1] * dy * dy
        a = (opacity * np.exp(-0.5 * q))[..., None]
        img[y0:y1, x0:x1] = img[y0:y1, x0:x1] * (1.0 - a) + colors[i] * a

    glow = gaussian_filter(img, sigma=(5.0, 5.0, 0.0))
    return np.clip(img + 0.5 * glow, 0.0, 1.0)


# ===== render the turntable and save GIF ==================================
n_frames = 36
azims = np.linspace(0.0, 360.0, n_frames, endpoint=False)

print(f"rendering {n_frames} frames at {W}x{H} (azim 0 .. 360, step 10 deg)...")
t0 = time.time()
frames = []
for k, az in enumerate(azims):
    img = render(az)
    frames.append(Image.fromarray((img * 255).astype(np.uint8)))
    if (k + 1) % 6 == 0:
        print(f"  {k + 1:2d}/{n_frames}   {time.time() - t0:5.1f}s elapsed")

out = "curve_splat_turntable.gif"
frames[0].save(
    out,
    save_all=True,
    append_images=frames[1:],
    duration=70,        # ms per frame (~14 fps)
    loop=0,             # 0 = loop forever
    disposal=2,
)
print(f"\nturntable -> {out}  ({n_frames} frames, total {time.time() - t0:.1f}s)")
