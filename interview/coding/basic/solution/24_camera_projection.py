"""
24 — Pinhole camera: project / unproject, Rodrigues rotation, look-at, behind-camera check

Implement the pinhole camera model with OpenCV conventions: intrinsics K = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
extrinsics (R, t) mapping WORLD to CAMERA coordinates X_c = R X_w + t, camera looks down +z, x right, y down,
pixel (u, v) = (fx * x/z + cx, fy * y/z + cy). These are the primitives behind NeRF / 3D-GS data loaders, depth
warping in video models, and robotics camera calibration.

Signatures (numpy float64):
    def project(points_world, K, R, t) -> (pixels (N, 2), depth (N,))
    def unproject(pixels, depth, K, R, t) -> points_world (N, 3)
    def rodrigues(axis_angle) -> R (3, 3)                                  # axis-angle vector -> rotation matrix
    def look_at(eye, target, up=(0, 0, 1)) -> (R, t)                       # world->camera extrinsics
    def points_behind_camera(points_world, R, t, eps=1e-8) -> bool (N,)   # z_c <= eps

Constraints: numpy only (no cv2 / scipy). Round trip project -> unproject must be exact to 1e-6. Points with
z_c <= 0 have no valid projection: project must return NaN pixels for them (depth still returned) and
points_behind_camera flags them.

Interview budget: 20 min

Discussion follow-ups:
  * Camera-to-world vs world-to-camera: which does NeRF's "c2w" store, and how do you invert [R|t] without a
    general 3x3 inverse?
  * Convention hell: OpenCV (x right, y down, z forward) vs OpenGL/Blender (y up, z backward). Write the
    conversion. Which one does COLMAP use?
  * Rodrigues near theta = 0 and theta = pi: numerical issues, and why quaternions / 6D rotation
    representations are preferred for learning.
  * Pixel-centre convention: is pixel (0, 0) the centre or the corner of the first pixel? What changes in cx?
    How does this interact with align_corners in grid_sample?
"""
import numpy as np

__implement__ = ["project", "unproject", "rodrigues", "look_at", "points_behind_camera"]


def make_K(fx, fy, cx, cy):
    """Given helper: 3x3 intrinsics matrix."""
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)


def project(points_world, K, R, t):
    """
    points_world : (N, 3) float64 world coordinates.
    K            : (3, 3) intrinsics [[fx, 0, cx], [0, fy, cy], [0, 0, 1]].
    R, t         : (3, 3) rotation and (3,) translation, world -> camera: X_c = R X_w + t.

    Returns (pixels, depth):
        pixels (N, 2) float64: u = fx * x_c / z_c + cx, v = fy * y_c / z_c + cy   (OpenCV, y down).
        depth  (N,)   float64: z_c (the camera-frame z, NOT the Euclidean distance).
    Points with z_c <= 0 (on or behind the camera plane) get pixels = NaN (both coordinates); depth is returned
    unchanged so the caller can filter with `depth > 0`.
    """
    Xc = points_world @ R.T + t[None, :]
    z = Xc[:, 2]
    valid = z > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        u = K[0, 0] * Xc[:, 0] / z + K[0, 2]
        v = K[1, 1] * Xc[:, 1] / z + K[1, 2]
    pixels = np.stack([u, v], axis=1)
    pixels[~valid] = np.nan
    return pixels, z


def unproject(pixels, depth, K, R, t):
    """
    Inverse of project.
    pixels : (N, 2) float64 (u, v);  depth : (N,) camera-frame z;  K, R, t as in project.
    Returns (N, 3) world points: x_c = (u - cx) / fx * z, y_c = (v - cy) / fy * z, X_w = R^T (X_c - t).
    """
    z = depth
    x = (pixels[:, 0] - K[0, 2]) / K[0, 0] * z
    y = (pixels[:, 1] - K[1, 2]) / K[1, 1] * z
    Xc = np.stack([x, y, z], axis=1)
    return (Xc - t[None, :]) @ R


def rodrigues(axis_angle):
    """
    axis_angle : (3,) rotation vector r = theta * unit_axis (right-hand rule).
    Returns R (3, 3) float64:  R = I + sin(theta) K + (1 - cos(theta)) K^2, with K the skew-symmetric matrix of the
    unit axis (K v = axis x v). For theta < 1e-8 return I + [r]_x (first-order expansion, avoids 0/0). R must be
    orthonormal with det +1; small angles satisfy R ≈ I + [r]_x.
    """
    r = np.asarray(axis_angle, dtype=np.float64)
    theta = np.linalg.norm(r)
    if theta < 1e-8:
        kx, ky, kz = r
        return np.eye(3) + np.array([[0, -kz, ky], [kz, 0, -kx], [-ky, kx, 0]], dtype=np.float64)
    kx, ky, kz = r / theta
    Kx = np.array([[0, -kz, ky], [kz, 0, -kx], [-ky, kx, 0]], dtype=np.float64)
    return np.eye(3) + np.sin(theta) * Kx + (1 - np.cos(theta)) * (Kx @ Kx)


def look_at(eye, target, up=(0.0, 0.0, 1.0)):
    """
    Build world->camera extrinsics for a camera at `eye` looking at `target` (both (3,)), with the image "up"
    direction as close as possible to the world vector `up`.

    OpenCV camera axes expressed in world coordinates:
        z_cam = normalize(target - eye)        (forward)
        x_cam = normalize(cross(z_cam, up))    (right)
        y_cam = cross(z_cam, x_cam)            (down)
    R = [x_cam; y_cam; z_cam] as ROWS (i.e. R = R_c2w^T), t = -R @ eye, so that R @ target + t = (0, 0, dist)
    and the target projects to the principal point. Raise ValueError if the forward direction is parallel to up.
    """
    eye = np.asarray(eye, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    up = np.asarray(up, dtype=np.float64)
    fwd = target - eye
    fwd = fwd / np.linalg.norm(fwd)
    right = np.cross(fwd, up)
    n = np.linalg.norm(right)
    if n < 1e-12:
        raise ValueError("view direction is parallel to the up vector")
    right = right / n
    down = np.cross(fwd, right)
    R = np.stack([right, down, fwd], axis=0)
    t = -R @ eye
    return R, t


def points_behind_camera(points_world, R, t, eps=1e-8):
    """
    points_world (N, 3), R (3, 3), t (3,) world->camera. Returns bool (N,) True where the camera-frame depth
    z_c = (R X_w + t)_z <= eps, i.e. the point is on or behind the image plane and cannot be projected.
    """
    z = (points_world @ R.T + t[None, :])[:, 2]
    return z <= eps
