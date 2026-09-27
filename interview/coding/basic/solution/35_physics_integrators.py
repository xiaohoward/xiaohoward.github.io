"""
35 — Physics integrators: mass-spring-damper and pendulum, Euler / symplectic Euler / RK4, energy, linearization,
     domain randomization

The numerical core of a physics simulator used for sim-to-real. Write two continuous-time dynamics functions
x_dot = f(x, u, params), three fixed-step integrators, a rollout, the energy of the undamped oscillator, a
finite-difference linearization of the DISCRETE step (A = d x_next / d x, B = d x_next / d u, the Jacobians you hand
to LQR / iLQR), and a domain-randomization sampler for physical parameters.

Signatures:
    def spring_dynamics(x, u, params) -> x_dot      x = [q, v], m q'' = u - k q - c v
    def pendulum_dynamics(x, u, params) -> x_dot    x = [theta, omega], m L^2 theta'' = -m g L sin(theta) + u
    def euler_step(f, x, u, dt, params) -> x_next
    def semi_implicit_euler_step(f, x, u, dt, params) -> x_next   (symplectic: velocity first, then position)
    def rk4_step(f, x, u, dt, params) -> x_next
    def spring_energy(x, params) -> E             0.5 m v^2 + 0.5 k q^2
    def rollout(step, f, x0, us, dt, params) -> (T+1, 2) states
    def linearize(step, f, x, u, dt, params, eps=1e-5) -> (A (2, 2), B (2, 1))   central finite differences
    def sample_params(ranges, gen) -> dict         uniform in [lo, hi] per key, from a torch.Generator

Constraints: torch (CPU) only; every function must work in float64 (use the dtype of x) and must be differentiable
by autograd (torch.sin, not math.sin) so the finite-difference Jacobians can be checked against
torch.autograd.functional.jacobian. State layout is always x[..., 0] = position, x[..., 1] = velocity; u is a
scalar force / torque tensor (shape () or (...,)); params is a dict of Python floats.

Interview budget: 25 min

Discussion follow-ups:
  * Explicit Euler pumps energy into an oscillator (amplification factor |1 + i w dt| > 1); symplectic Euler is
    O(dt) but preserves a modified Hamiltonian, so energy stays bounded forever. Why do MuJoCo / Isaac default to
    semi-implicit Euler rather than RK4, and when does RK4 still lose (stiff contacts, dt vs stiffness)?
  * The linearization here differentiates through the integrator. When would you differentiate the continuous
    dynamics and discretize the Jacobians (A = I + dt J) instead, and what does that cost in accuracy at large dt?
  * Domain randomization: uniform vs log-uniform over mass / stiffness / friction, and how would you pick ranges
    (or learn them: automatic DR, system identification from real rollouts) for sim-to-real of a policy?
  * Differentiable simulation: what breaks the gradient (contacts, discontinuities), and where is the
    finite-difference Jacobian preferable to autograd in a real simulator?
"""
import torch

__implement__ = ["spring_dynamics", "pendulum_dynamics", "euler_step", "semi_implicit_euler_step", "rk4_step",
                 "spring_energy", "rollout", "linearize", "sample_params"]


def spring_dynamics(x, u, params):
    """
    Mass-spring-damper. x : (..., 2) = [q, v]; u : force, () or (...,); params = {"m", "k", "c"} (c may be 0).
    Returns x_dot : (..., 2) = [v, (u - k q - c v) / m].
    """
    q, v = x[..., 0], x[..., 1]
    acc = (u - params["k"] * q - params["c"] * v) / params["m"]
    return torch.stack([v, acc], dim=-1)


def pendulum_dynamics(x, u, params):
    """
    Simple pendulum with a torque input. x : (..., 2) = [theta, omega]; u : torque; params = {"m", "L", "g"}.
    Returns x_dot : (..., 2) = [omega, -(g / L) sin(theta) + u / (m L^2)]. theta = 0 is the stable bottom.
    """
    th, om = x[..., 0], x[..., 1]
    acc = -(params["g"] / params["L"]) * torch.sin(th) + u / (params["m"] * params["L"] ** 2)
    return torch.stack([om, acc], dim=-1)


def euler_step(f, x, u, dt, params):
    """Explicit (forward) Euler: x_next = x + dt * f(x, u, params). x : (..., 2). Returns (..., 2)."""
    return x + dt * f(x, u, params)


def semi_implicit_euler_step(f, x, u, dt, params):
    """
    Semi-implicit (symplectic) Euler: update velocity with the acceleration at the CURRENT state, then the position
    with the NEW velocity:
        v_next = v + dt * f(x, u)[..., 1]
        q_next = q + dt * v_next
    Returns (..., 2) = [q_next, v_next]. Same cost as explicit Euler, but energy of a conservative system stays
    bounded (it is a symplectic map).
    """
    acc = f(x, u, params)[..., 1]
    v_next = x[..., 1] + dt * acc
    q_next = x[..., 0] + dt * v_next
    return torch.stack([q_next, v_next], dim=-1)


def rk4_step(f, x, u, dt, params):
    """
    Classical 4th-order Runge-Kutta with u held constant over the step (zero-order hold):
        k1 = f(x), k2 = f(x + dt/2 k1), k3 = f(x + dt/2 k2), k4 = f(x + dt k3)
        x_next = x + dt/6 (k1 + 2 k2 + 2 k3 + k4)
    Returns (..., 2). Local error O(dt^5), global O(dt^4).
    """
    k1 = f(x, u, params)
    k2 = f(x + 0.5 * dt * k1, u, params)
    k3 = f(x + 0.5 * dt * k2, u, params)
    k4 = f(x + dt * k3, u, params)
    return x + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)


def spring_energy(x, params):
    """Total mechanical energy of the undamped spring: 0.5 m v^2 + 0.5 k q^2. x : (..., 2) -> (...)."""
    q, v = x[..., 0], x[..., 1]
    return 0.5 * params["m"] * v ** 2 + 0.5 * params["k"] * q ** 2


def rollout(step, f, x0, us, dt, params):
    """
    Integrate T steps. step : one of the *_step functions; f : dynamics; x0 : (2,); us : (T,) inputs (u[t] is
    applied over step t); dt : float. Returns states (T+1, 2) with states[0] = x0, stacked in x0's dtype.
    """
    xs = [x0]
    x = x0
    for t in range(us.shape[0]):
        x = step(f, x, us[t], dt, params)
        xs.append(x)
    return torch.stack(xs, dim=0)


def linearize(step, f, x, u, dt, params, eps=1e-5):
    """
    Central finite-difference Jacobians of the discrete map x_next = step(f, x, u, dt, params) around (x, u):
        A[i, j] = d x_next[i] / d x[j]   (2, 2),   B[i, 0] = d x_next[i] / d u   (2, 1)
    using (g(z + eps e_j) - g(z - eps e_j)) / (2 eps). x : (2,), u : scalar tensor. Returns (A, B) in x's dtype.
    Then x_next(x + dx, u + du) ~= x_next(x, u) + A dx + B du, the model an LQR step needs.
    """
    n = x.shape[0]
    A = torch.zeros(n, n, dtype=x.dtype)
    for j in range(n):
        e = torch.zeros_like(x)
        e[j] = eps
        A[:, j] = (step(f, x + e, u, dt, params) - step(f, x - e, u, dt, params)) / (2 * eps)
    B = ((step(f, x, u + eps, dt, params) - step(f, x, u - eps, dt, params)) / (2 * eps)).reshape(n, 1)
    return A, B


def sample_params(ranges, gen):
    """
    Domain randomization. ranges : dict name -> (lo, hi); gen : torch.Generator. Returns dict name -> Python float
    drawn uniformly in [lo, hi], one torch.rand((), generator=gen) draw per key in the dict's iteration order (so
    the same seed gives the same parameters). Keys with lo == hi are returned as that constant.
    """
    out = {}
    for name, (lo, hi) in ranges.items():
        r = torch.rand((), generator=gen).item()
        out[name] = lo + (hi - lo) * r
    return out
