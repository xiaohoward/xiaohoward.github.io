import math
import torch
torch.set_num_threads(min(torch.get_num_threads(), 8))

F64 = torch.float64


def check_dynamics(mod):
    p = {"m": 2.0, "k": 8.0, "c": 0.5}
    x = torch.tensor([[1.0, -2.0], [0.0, 1.0]], dtype=F64)
    xd = mod.spring_dynamics(x, torch.tensor(3.0, dtype=F64), p)
    ref = torch.tensor([[-2.0, (3 - 8 * 1 + 0.5 * 2) / 2], [1.0, (3 - 0 - 0.5) / 2]], dtype=F64)
    assert xd.shape == (2, 2) and torch.allclose(xd, ref), f"spring dynamics {xd.tolist()} != {ref.tolist()}"
    pp = {"m": 1.5, "L": 2.0, "g": 9.81}
    xd = mod.pendulum_dynamics(torch.tensor([math.pi / 6, 0.3], dtype=F64), torch.tensor(1.2, dtype=F64), pp)
    ref = torch.tensor([0.3, -9.81 / 2 * 0.5 + 1.2 / (1.5 * 4)], dtype=F64)
    assert torch.allclose(xd, ref, atol=1e-12), f"pendulum dynamics {xd.tolist()} != {ref.tolist()}"
    assert torch.allclose(mod.spring_energy(torch.tensor([1.0, -2.0], dtype=F64), p), torch.tensor(0.5 * 2 * 4 + 0.5 * 8, dtype=F64))


def check_rk4_analytic(mod):
    p = {"m": 1.0, "k": 4.0, "c": 0.0}                     # w = 2
    w = math.sqrt(p["k"] / p["m"])
    dt, T = 0.01, 1000
    q0, v0 = 1.0, 0.5
    x0 = torch.tensor([q0, v0], dtype=F64)
    xs = mod.rollout(mod.rk4_step, mod.spring_dynamics, x0, torch.zeros(T, dtype=F64), dt, p)
    assert xs.shape == (T + 1, 2) and xs.dtype == F64 and torch.equal(xs[0], x0)
    t = torch.arange(T + 1, dtype=F64) * dt
    q_ref = q0 * torch.cos(w * t) + v0 / w * torch.sin(w * t)
    v_ref = -q0 * w * torch.sin(w * t) + v0 * torch.cos(w * t)
    err_q, err_v = (xs[:, 0] - q_ref).abs().max().item(), (xs[:, 1] - v_ref).abs().max().item()
    assert max(err_q, err_v) < 1e-6, f"RK4 vs analytic harmonic oscillator: max err q {err_q:.2e}, v {err_v:.2e}"
    # order check: halving dt cuts the error by ~16
    xs2 = mod.rollout(mod.rk4_step, mod.spring_dynamics, x0, torch.zeros(2 * T, dtype=F64), dt / 2, p)
    t2 = torch.arange(2 * T + 1, dtype=F64) * dt / 2
    err2 = (xs2[:, 0] - (q0 * torch.cos(w * t2) + v0 / w * torch.sin(w * t2))).abs().max().item()
    assert 12 < err_q / err2 < 20, f"RK4 should be 4th order (error ratio ~16 when halving dt): got {err_q / err2:.1f}"
    # single Euler step is exactly x + dt f(x)
    x = torch.tensor([0.3, -0.7], dtype=F64)
    assert torch.allclose(mod.euler_step(mod.spring_dynamics, x, torch.tensor(0.0, dtype=F64), 0.1, p),
                          x + 0.1 * mod.spring_dynamics(x, torch.tensor(0.0, dtype=F64), p))


def check_energy(mod):
    p = {"m": 1.0, "k": 1.0, "c": 0.0}
    dt, T = 0.01, 2000
    x0 = torch.tensor([1.0, 0.0], dtype=F64)
    us = torch.zeros(T, dtype=F64)
    E0 = mod.spring_energy(x0, p).item()
    e_exp = mod.spring_energy(mod.rollout(mod.euler_step, mod.spring_dynamics, x0, us, dt, p), p)
    e_sym = mod.spring_energy(mod.rollout(mod.semi_implicit_euler_step, mod.spring_dynamics, x0, us, dt, p), p)
    e_rk4 = mod.spring_energy(mod.rollout(mod.rk4_step, mod.spring_dynamics, x0, us, dt, p), p)
    assert bool((e_exp[1:] >= e_exp[:-1] - 1e-12).all()) and e_exp[-1] > 1.15 * E0, \
        f"explicit Euler must pump energy monotonically (final E/E0 = {e_exp[-1] / E0:.3f})"
    drift = ((e_sym - E0).abs() / E0).max().item()
    assert drift < 0.01, f"symplectic Euler energy must stay within 1% (max rel drift {drift:.3e})"
    assert abs(e_sym[-1] - E0) / E0 < 0.01
    assert ((e_rk4 - E0).abs() / E0).max().item() < 1e-6
    # symplectic step: velocity uses the current acceleration, position uses the NEW velocity
    x = torch.tensor([0.5, -0.2], dtype=F64)
    u = torch.tensor(0.3, dtype=F64)
    acc = mod.spring_dynamics(x, u, p)[1]
    v_new = x[1] + dt * acc
    ref = torch.stack([x[0] + dt * v_new, v_new])
    assert torch.allclose(mod.semi_implicit_euler_step(mod.spring_dynamics, x, u, dt, p), ref, atol=1e-14)


def check_pendulum_period(mod):
    p = {"m": 1.0, "L": 1.7, "g": 9.81}
    dt = 1e-3
    T = 6000
    x0 = torch.tensor([0.005, 0.0], dtype=F64)
    xs = mod.rollout(mod.rk4_step, mod.pendulum_dynamics, x0, torch.zeros(T, dtype=F64), dt, p)
    th = xs[:, 0]
    # zero crossings of theta with linear interpolation -> half periods
    s = torch.sign(th)
    idx = torch.nonzero(s[1:] * s[:-1] < 0).flatten()
    assert len(idx) >= 3, "pendulum should cross zero at least 3 times in 6 s"
    tc = [(i + th[i] / (th[i] - th[i + 1])).item() * dt for i in idx]
    period = 2 * (tc[-1] - tc[0]) / (len(tc) - 1)
    ref = 2 * math.pi * math.sqrt(p["L"] / p["g"])
    assert abs(period - ref) / ref < 1e-4, f"small-angle period {period:.5f} vs 2 pi sqrt(L/g) = {ref:.5f}"
    # large amplitude is slower (nonlinear) - the sin must really be there
    xs = mod.rollout(mod.rk4_step, mod.pendulum_dynamics, torch.tensor([2.5, 0.0], dtype=F64), torch.zeros(T, dtype=F64), dt, p)
    s = torch.sign(xs[:, 0]); idx = torch.nonzero(s[1:] * s[:-1] < 0).flatten()
    big = 2 * (idx[-1] - idx[0]).item() * dt / (len(idx) - 1)
    assert big > 1.4 * ref, "large-angle period must be much longer than the small-angle one"


def check_linearize(mod):
    for f, p, x, u, dt in ((mod.spring_dynamics, {"m": 1.3, "k": 5.0, "c": 0.4}, torch.tensor([0.4, -1.1], dtype=F64), torch.tensor(0.7, dtype=F64), 0.05),
                           (mod.pendulum_dynamics, {"m": 0.8, "L": 1.2, "g": 9.81}, torch.tensor([2.0, 0.5], dtype=F64), torch.tensor(-0.3, dtype=F64), 0.02)):
        for step in (mod.euler_step, mod.semi_implicit_euler_step, mod.rk4_step):
            A, B = mod.linearize(step, f, x, u, dt, p)
            assert A.shape == (2, 2) and B.shape == (2, 1)
            A_ref = torch.autograd.functional.jacobian(lambda z: step(f, z, u, dt, p), x)
            B_ref = torch.autograd.functional.jacobian(lambda uu: step(f, x, uu, dt, p), u).reshape(2, 1)
            assert torch.allclose(A, A_ref, atol=1e-7), f"{step.__name__}: A mismatch vs autograd\n{A}\n{A_ref}"
            assert torch.allclose(B, B_ref, atol=1e-7), f"{step.__name__}: B mismatch vs autograd\n{B}\n{B_ref}"
    # explicit Euler on the linear spring: A = I + dt * [[0, 1], [-k/m, -c/m]], B = dt * [0, 1/m]
    p = {"m": 1.3, "k": 5.0, "c": 0.4}; dt = 0.05
    A, B = mod.linearize(mod.euler_step, mod.spring_dynamics, torch.zeros(2, dtype=F64), torch.tensor(0.0, dtype=F64), dt, p)
    assert torch.allclose(A, torch.eye(2, dtype=F64) + dt * torch.tensor([[0, 1], [-5 / 1.3, -0.4 / 1.3]], dtype=F64), atol=1e-8)
    assert torch.allclose(B, torch.tensor([[0.0], [dt / 1.3]], dtype=F64), atol=1e-8)
    # one-step LQR-style use: the linear model predicts the perturbed step to second order
    x = torch.tensor([2.0, 0.5], dtype=F64); u = torch.tensor(-0.3, dtype=F64); pp = {"m": 0.8, "L": 1.2, "g": 9.81}
    A, B = mod.linearize(mod.rk4_step, mod.pendulum_dynamics, x, u, 0.02, pp)
    dx, du = torch.tensor([1e-3, -2e-3], dtype=F64), torch.tensor(5e-3, dtype=F64)
    pred = mod.rk4_step(mod.pendulum_dynamics, x, u, 0.02, pp) + A @ dx + (B[:, 0] * du)
    true = mod.rk4_step(mod.pendulum_dynamics, x + dx, u + du, 0.02, pp)
    assert (pred - true).abs().max() < 1e-7, "linear model must predict a small perturbation to O(delta^2)"


def check_randomization(mod):
    ranges = {"m": (0.5, 2.0), "k": (5.0, 20.0), "c": (0.0, 0.0)}
    draws = [mod.sample_params(ranges, torch.Generator().manual_seed(s)) for s in range(50)]
    for d in draws:
        assert set(d) == {"m", "k", "c"}
        assert 0.5 <= d["m"] <= 2.0 and 5.0 <= d["k"] <= 20.0 and d["c"] == 0.0, f"out of range: {d}"
        assert isinstance(d["m"], float)
    assert mod.sample_params(ranges, torch.Generator().manual_seed(3)) == draws[3], "same seed must reproduce"
    assert len({round(d["m"], 6) for d in draws}) > 40, "different seeds must give different draws"
    ms = torch.tensor([d["m"] for d in draws]); assert ms.min() < 0.9 and ms.max() > 1.6, "draws should spread over the range"
    # randomized rollouts still integrate
    g = torch.Generator().manual_seed(0)
    p = mod.sample_params(ranges, g)
    xs = mod.rollout(mod.rk4_step, mod.spring_dynamics, torch.tensor([1.0, 0.0], dtype=F64), torch.zeros(100, dtype=F64), 0.01, p)
    assert torch.isfinite(xs).all()


def run(mod):
    check_dynamics(mod);         print("  ok  spring / pendulum dynamics and energy")
    check_rk4_analytic(mod);     print("  ok  RK4 matches the analytic harmonic oscillator (1e-6), 4th order")
    check_energy(mod);           print("  ok  explicit Euler pumps energy, symplectic Euler bounded (<1%), RK4 ~exact")
    check_pendulum_period(mod);  print("  ok  pendulum small-angle period = 2 pi sqrt(L/g)")
    check_linearize(mod);        print("  ok  finite-difference A, B match autograd; one-step linear prediction")
    check_randomization(mod);    print("  ok  domain randomization in bounds and reproducible")
