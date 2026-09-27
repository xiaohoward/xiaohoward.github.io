import numpy as np


def _K(fx=500.0, fy=480.0, cx=320.0, cy=240.0):
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)


def _random_pose(rng):
    R = rng_rotation(rng)
    t = rng.normal(size=3)
    return R, t


def rng_rotation(rng):
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def check_roundtrip(mod):
    rng = np.random.default_rng(0)
    K = _K()
    R, t = _random_pose(rng)
    # points in front of the camera: sample in camera frame with z > 0, then move to world
    Xc = rng.normal(size=(50, 3))
    Xc[:, 2] = rng.uniform(0.5, 10.0, size=50)
    Xw = (Xc - t) @ R
    pix, depth = mod.project(Xw, K, R, t)
    assert pix.shape == (50, 2) and depth.shape == (50,), "project must return (N,2) pixels and (N,) depth"
    assert np.allclose(depth, Xc[:, 2], atol=1e-9), "depth must be the camera-frame z"
    assert np.allclose(pix[:, 0], K[0, 0] * Xc[:, 0] / Xc[:, 2] + K[0, 2], atol=1e-6), "u formula wrong"
    assert np.allclose(pix[:, 1], K[1, 1] * Xc[:, 1] / Xc[:, 2] + K[1, 2], atol=1e-6), "v formula wrong"
    back = mod.unproject(pix, depth, K, R, t)
    assert back.shape == (50, 3)
    assert np.allclose(back, Xw, atol=1e-6), f"unproject(project(X)) must round-trip: max err {np.abs(back-Xw).max():.2e}"
    # and the other direction
    pix2, d2 = mod.project(mod.unproject(pix, depth, K, R, t), K, R, t)
    assert np.allclose(pix2, pix, atol=1e-6) and np.allclose(d2, depth, atol=1e-9)
    # identity pose: principal point comes from the optical axis
    I, z0 = np.eye(3), np.zeros(3)
    p, d = mod.project(np.array([[0.0, 0.0, 3.0]]), K, I, z0)
    assert np.allclose(p[0], [K[0, 2], K[1, 2]]) and d[0] == 3.0, "point on the optical axis must project to (cx, cy)"


def check_rodrigues(mod):
    rng = np.random.default_rng(1)
    for _ in range(5):
        r = rng.normal(size=3) * rng.uniform(0.1, 3.0)
        R = mod.rodrigues(r)
        assert R.shape == (3, 3)
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-10), "R must be orthonormal"
        assert abs(np.linalg.det(R) - 1) < 1e-10, "det R must be +1"
        # axis is invariant; rotation angle matches trace formula
        assert np.allclose(R @ r, r, atol=1e-9), "axis must be fixed by R"
        theta = np.linalg.norm(r)
        assert abs((np.trace(R) - 1) / 2 - np.cos(theta)) < 1e-9, "rotation angle wrong"
    # 90 degrees about z maps x -> y
    Rz = mod.rodrigues(np.array([0, 0, np.pi / 2]))
    assert np.allclose(Rz @ np.array([1, 0, 0]), [0, 1, 0], atol=1e-12), "right-hand rule: Rz(90) x = y"
    # small angle: R ~ I + [r]_x
    r = np.array([1e-4, -2e-4, 3e-4])
    Rs = mod.rodrigues(r)
    skew = np.array([[0, -r[2], r[1]], [r[2], 0, -r[0]], [-r[1], r[0], 0]])
    assert np.allclose(Rs, np.eye(3) + skew, atol=1e-7), "small-angle approximation must hold"
    assert np.allclose(mod.rodrigues(np.zeros(3)), np.eye(3)), "zero vector must give identity"
    # composition consistency: rodrigues(r) @ rodrigues(-r) = I
    r = rng.normal(size=3)
    assert np.allclose(mod.rodrigues(r) @ mod.rodrigues(-r), np.eye(3), atol=1e-10)


def check_look_at(mod):
    rng = np.random.default_rng(2)
    K = _K()
    for _ in range(4):
        eye = rng.normal(size=3) * 3
        target = rng.normal(size=3)
        R, t = mod.look_at(eye, target)
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-10) and abs(np.linalg.det(R) - 1) < 1e-10, "look_at R must be a rotation"
        pix, depth = mod.project(target[None], K, R, t)
        assert np.allclose(pix[0], [K[0, 2], K[1, 2]], atol=1e-6), "target must project to the principal point"
        assert abs(depth[0] - np.linalg.norm(target - eye)) < 1e-9, "target depth must be the eye-target distance"
        # camera centre maps to the origin of the camera frame
        assert np.allclose(R @ eye + t, 0, atol=1e-10), "R eye + t must be 0"
        # world 'up' must point to image 'up' (negative v direction): y_cam . up < 0
        assert R[1] @ np.array([0, 0, 1.0]) < 0, "world up must map to negative image y (y down)"
        # x_cam must be horizontal (orthogonal to up)
        assert abs(R[0] @ np.array([0, 0, 1.0])) < 1e-10, "camera x axis must be horizontal"
    # explicit sanity: camera at (-5, 0, 0) looking at the origin with z up: +y world is to the LEFT (-x_cam)
    R, t = mod.look_at(np.array([-5.0, 0, 0]), np.zeros(3))
    assert np.allclose(R @ np.array([0, 1.0, 0]), [-1, 0, 0], atol=1e-12), "handedness: looking +x with z up, +y is image left"
    try:
        mod.look_at(np.zeros(3), np.array([0, 0, 1.0])); raise AssertionError("looking along up must raise ValueError")
    except ValueError:
        pass


def check_behind_camera(mod):
    rng = np.random.default_rng(3)
    K = _K()
    R, t = _random_pose(rng)
    Xc = rng.normal(size=(20, 3))
    Xc[:10, 2] = rng.uniform(0.5, 5, size=10)
    Xc[10:, 2] = -rng.uniform(0.5, 5, size=10)
    Xw = (Xc - t) @ R
    flag = mod.points_behind_camera(Xw, R, t)
    assert flag.shape == (20,) and flag.dtype == bool
    assert not flag[:10].any() and flag[10:].all(), "behind-camera flags wrong"
    pix, depth = mod.project(Xw, K, R, t)
    assert np.isnan(pix[10:]).all() and np.isfinite(pix[:10]).all(), "behind-camera points must get NaN pixels"
    assert (depth[10:] < 0).all(), "depth of behind-camera points must be negative"
    # exactly on the image plane counts as behind
    on_plane = ((np.array([[0.3, -0.2, 0.0]]) - t) @ R)
    assert mod.points_behind_camera(on_plane, R, t)[0], "z = 0 must be flagged"


def run(mod):
    check_roundtrip(mod);      print("  ok  project / unproject round trip, principal point")
    check_rodrigues(mod);      print("  ok  Rodrigues: orthonormal, det +1, small angle, axis fixed")
    check_look_at(mod);        print("  ok  look_at: target at image centre, handedness")
    check_behind_camera(mod);  print("  ok  behind-camera flags + NaN pixels")
