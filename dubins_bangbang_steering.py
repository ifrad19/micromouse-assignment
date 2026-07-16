"""
Pure-Python Dubins path solver + bang-bang speed integration.

NOTE: This module provides a self-contained, pure-Python implementation 
of the six Dubins families (LSL, LSR, RSL, RSR, LRL, RRL) using the analytical
formulas from Shkel & Lumelsky (2001) and Walker (2010), followed by
bang-bang speed-profile integration from ``bangbang_steering.py``.
"""

import math
import numpy as np

from bangbang_steering import bang_bang_steer_1d, sample_trajectory_1d

EPSILON = 1e-10


# ── Pure-Python Dubins solver ── #


def _mod2pi(theta):
    """Normalize angle to [0, 2π)."""
    v = theta % (2.0 * math.pi)
    if v < 0.0:
        v += 2.0 * math.pi
    return v


def _tangents_LSL(alpha, beta, d):
    ca, sa = math.cos(alpha), math.sin(alpha)
    cb, sb = math.cos(beta), math.sin(beta)
    p_squared = 2.0 + d * d - 2.0 * (ca * cb + sa * sb - d * (sa - sb))
    if p_squared < 0.0:
        return None
    p = math.sqrt(p_squared)
    theta = math.atan2(cb - sb, d + ca - cb)
    t = _mod2pi(-alpha + theta)
    q = _mod2pi(beta - theta)
    return t, p, q


def _tangents_RSR(alpha, beta, d):
    ca, sa = math.cos(alpha), math.sin(alpha)
    cb, sb = math.cos(beta), math.sin(beta)
    p_squared = 2.0 + d * d - 2.0 * (ca * cb + sa * sb - d * (sb - sa))
    if p_squared < 0.0:
        return None
    p = math.sqrt(p_squared)
    theta = math.atan2(sb - cb, d - ca + cb)
    t = _mod2pi(alpha - theta)
    q = _mod2pi(-beta + theta)
    return t, p, q


def _tangents_LSR(alpha, beta, d):
    ca, sa = math.cos(alpha), math.sin(alpha)
    cb, sb = math.cos(beta), math.sin(beta)
    p_squared = -2.0 + d * d + 2.0 * (ca * cb + sa * sb + d * (sa + sb))
    if p_squared < 0.0:
        return None
    p = math.sqrt(p_squared)
    theta = math.atan2(-cb - sb, d + ca + cb) - math.atan2(-2.0, p)
    t = _mod2pi(-alpha + theta)
    q = _mod2pi(-beta + theta)
    return t, p, q


def _tangents_RSL(alpha, beta, d):
    ca, sa = math.cos(alpha), math.sin(alpha)
    cb, sb = math.cos(beta), math.sin(beta)
    p_squared = -2.0 + d * d + 2.0 * (ca * cb + sa * sb - d * (sa + sb))
    if p_squared < 0.0:
        return None
    p = math.sqrt(p_squared)
    theta = math.atan2(sb + cb, d - ca - cb) - math.atan2(2.0, p)
    t = _mod2pi(alpha - theta)
    q = _mod2pi(beta - theta)
    return t, p, q


def _tangents_LRL(alpha, beta, d):
    ca, sa = math.cos(alpha), math.sin(alpha)
    cb, sb = math.cos(beta), math.sin(beta)
    p_squared = 6.0 - d * d + 2.0 * (ca * cb + sa * sb) + 2.0 * d * (sa - sb)
    if p_squared < 0.0:
        return None
    p = math.sqrt(p_squared)
    theta = math.atan2(ca - cb, d + sa - sb) - math.atan2(2.0, p)
    t = _mod2pi(-alpha + theta)
    q = _mod2pi(2.0 * math.pi - _mod2pi(beta - theta))
    if t > 2.0 * math.pi + EPSILON or q > 2.0 * math.pi + EPSILON:
        return None
    return t, p, q


def _tangents_RRL(alpha, beta, d):
    ca, sa = math.cos(alpha), math.sin(alpha)
    cb, sb = math.cos(beta), math.sin(beta)
    p_squared = 6.0 - d * d + 2.0 * (ca * cb + sa * sb) - 2.0 * d * (sa - sb)
    if p_squared < 0.0:
        return None
    p = math.sqrt(p_squared)
    theta = math.atan2(-ca + cb, d - sa + sb) - math.atan2(-2.0, p)
    t = _mod2pi(alpha - theta)
    q = _mod2pi(2.0 * math.pi - _mod2pi(-beta + theta))
    if t > 2.0 * math.pi + EPSILON or q > 2.0 * math.pi + EPSILON:
        return None
    return t, p, q


_TANGENT_FNS = {
    "LSL": _tangents_LSL, "LSR": _tangents_LSR,
    "RSL": _tangents_RSL, "RSR": _tangents_RSR,
    "LRL": _tangents_LRL, "RRL": _tangents_RRL,
}


class DubinsPath:
    """Represents a Dubins path between two oriented poses."""

    def __init__(self, q0, q1, rho, length, path_type, params):
        self.q0 = q0
        self.q1 = q1
        self.rho = rho
        self._length = length
        self.path_type = path_type
        self.params = params          # (t, p, q) in units of rho

    def path_length(self):
        return self._length

    def sample_many(self, step_size):
        """Sample (x, y, θ, s) where *s* is cumulative arc distance."""
        if self._length < EPSILON:
            return [(self.q0[0], self.q0[1], self.q0[2], 0.0)]

        t, p, q = self.params
        chars = list(self.path_type)
        x, y, theta = self.q0
        out = [(x, y, theta, 0.0)]
        s = 0.0

        def _arc(angle):
            nonlocal x, y, theta, s
            arc_len = self.rho * abs(angle)
            if angle > 0.0:
                x_new = x + self.rho * (math.sin(theta + angle) - math.sin(theta))
                y_new = y - self.rho * (math.cos(theta + angle) - math.cos(theta))
            else:
                x_new = x + self.rho * (math.sin(theta) - math.sin(theta + angle))
                y_new = y - self.rho * (math.cos(theta) - math.cos(theta + angle))
            x, y, theta = x_new, y_new, theta + angle
            s += arc_len
            out.append((x, y, theta, s))

        def _seg_arc(angle_total, ch):
            if abs(angle_total) < EPSILON:
                return
            n = max(2, int(math.ceil(abs(angle_total) * self.rho / step_size)))
            sub = angle_total / n
            for _ in range(n):
                _arc(sub if ch == "L" else -sub)

        def _seg_straight(dist):
            d_cells = dist * self.rho
            if d_cells < EPSILON:
                return
            n = max(2, int(math.ceil(d_cells / step_size)))
            ds = d_cells / n
            nonlocal x, y, s
            for _ in range(n):
                x += ds * math.cos(theta)
                y += ds * math.sin(theta)
                s += ds
                out.append((x, y, theta, s))

        _seg_arc(t, chars[0])

        if chars[1] == "S":
            _seg_straight(p)
        else:
            _seg_arc(p, chars[1])

        _seg_arc(q, chars[2])
        return out


DUBINS_PATH_TYPES = ["LSL", "LSR", "RSL", "RSR", "LRL", "RRL"]


def shortest_path(q0, q1, rho):
    """Compute the shortest Dubins path between two oriented poses.

    Args:
        q0: (x, y, θ) start pose
        q1: (x, y, θ) goal pose
        rho: minimum turning radius (same units as x, y)

    Returns:
        DubinsPath or None if infeasible.
    """
    dx = q1[0] - q0[0]
    dy = q1[1] - q0[1]
    d = math.hypot(dx, dy) / rho
    if d < EPSILON:
        return DubinsPath(q0, q1, rho, 0.0, "ZZZ", (0.0, 0.0, 0.0))

    theta = math.atan2(dy, dx)
    alpha = _mod2pi(q0[2] - theta)
    beta = _mod2pi(q1[2] - theta)

    best_cost = float("inf")
    best_params = None
    best_type = None

    for name, fn in _TANGENT_FNS.items():
        result = fn(alpha, beta, d)
        if result is None:
            continue
        t, p, q = result
        cost = t + p + q
        if cost < best_cost:
            best_cost = cost
            best_params = (t, p, q)
            best_type = name

    if best_params is None:
        return None

    return DubinsPath(q0, q1, rho, best_cost * rho, best_type, best_params)


# ── NodeTheta ── #


class NodeTheta:
    """Tree node for heading-aware planner.  State is (x, y, θ, v).

    θ is in radians, range [-π, π].
    trajectory_2d is np.ndarray (N, 5): [x, y, θ, v, t].
    """

    __slots__ = (
        "x", "y", "theta", "v",
        "parent", "children",
        "cost",
        "control",
        "trajectory_2d",
    )

    def __init__(self, x, y, theta, v, parent=None, cost=0.0,
                 control=None, trajectory_2d=None):
        self.x = x
        self.y = y
        self.theta = theta
        self.v = v
        self.parent = parent
        self.children = []
        self.cost = cost
        self.control = control if control is not None else []
        self.trajectory_2d = trajectory_2d
        if parent is not None:
            parent.children.append(self)


# ── Turning-radius helper ── #


def compute_turning_radius_cells(config, num_cols):
    """Return minimum turning radius in **cells**.

    Derives meters_per_cell from the maze physical width and grid columns::

        meters_per_cell = maze_width_meters / num_cols
        turning_radius_cells = (wheelbase / 2) / meters_per_cell
    """
    try:
        wheelbase = config["diff_drive"]["wheelbase"]
    except (KeyError, TypeError):
        wheelbase = 0.06
    try:
        maze_width_m = config["physical_dimensions"]["maze_width_meters"]
    except (KeyError, TypeError):
        maze_width_m = 2.88
    meters_per_cell = maze_width_m / num_cols
    return (wheelbase / 2.0) / meters_per_cell


# ── Steering function ── #


def steer_dubins_bangbang(node_from, node_to, turning_radius, a_max, v_max,
                          samples_per_second=40.0):
    """Steer from *node_from* to *node_to* using Dubins + bang-bang speed.

    Args:
        node_from: NodeTheta with (x, y, θ, v)
        node_to:   NodeTheta with (x, y, θ, v) — only x, y, θ used as goal pose;
                   v is used as goal speed.
        turning_radius: minimum turning radius in **cells**.
        a_max: maximum acceleration in cells/s².
        v_max: maximum speed in cells/s (currently unused; kept for API symmetry).
        samples_per_second: geometry sample rate.

    Returns:
        dict with keys 'feasible', 'total_time', 'control', 'arc_length',
        'dubins_path_type', 'trajectory' (N, 5) [x, y, θ, v, t].
        Or None if infeasible.
    """
    q0 = (node_from.x, node_from.y, node_from.theta)
    q1 = (node_to.x, node_to.y, node_to.theta)

    dp = shortest_path(q0, q1, turning_radius)
    if dp is None:
        return None

    arc_length = dp.path_length()
    if arc_length < EPSILON:
        return {
            "feasible": True,
            "total_time": 0.0,
            "control": [],
            "arc_length": 0.0,
            "dubins_path_type": dp.path_type,
            "trajectory": np.array([[q0[0], q0[1], q0[2], node_from.v, 0.0]]),
        }

    # Step 2: bang-bang speed profile along arc length
    steer = bang_bang_steer_1d(
        x_init=0.0,
        v_init=node_from.v,
        x_goal=arc_length,
        v_goal=node_to.v,
        a_max=a_max,
        samples_per_second=80.0,
    )
    if steer is None or not steer["feasible"]:
        return None

    # Step 3: sample Dubins geometry
    step_size = arc_length / max(50.0, arc_length * samples_per_second)
    geo_samples = dp.sample_many(step_size)

    # Step 4: re-time geometry with bang-bang speed profile
    traj_1d = steer["trajectory"]            # (M, 3) [dist, v, t]
    s_geo = np.array([pt[3] for pt in geo_samples])   # cumulative arc distances
    v_interp = np.interp(s_geo, traj_1d[:, 0], traj_1d[:, 1])
    t_interp = np.interp(s_geo, traj_1d[:, 0], traj_1d[:, 2])

    N = len(geo_samples)
    trajectory = np.empty((N, 5), dtype=float)
    for i, pt in enumerate(geo_samples):
        trajectory[i, 0] = pt[0]    # x
        trajectory[i, 1] = pt[1]    # y
        trajectory[i, 2] = pt[2]    # θ
        trajectory[i, 3] = v_interp[i]   # v
        trajectory[i, 4] = t_interp[i]   # t

    return {
        "feasible": True,
        "total_time": steer["total_time"],
        "control": steer["control"],
        "arc_length": arc_length,
        "dubins_path_type": dp.path_type,
        "trajectory": trajectory,
    }

# ── Self-test ── #


def _selftest():
    print("=" * 60)
    print("dubins_bangbang_steering.py self-test")
    print("=" * 60)

    # 1. basic Dubins path
    q0 = (0.0, 0.0, 0.0)
    q1 = (5.0, 5.0, math.pi / 2)
    dp = shortest_path(q0, q1, 1.0)
    assert dp is not None, "Dubins path returned None"
    print(f"  q0={q0} -> q1={q1}, rho=1.0")
    print(f"  path type = {dp.path_type}, length = {dp.path_length():.4f}")
    samples = dp.sample_many(0.2)
    print(f"  samples = {len(samples)}")
    # verify endpoint is close to q1
    sx, sy, sθ, _ = samples[-1]
    err = math.hypot(sx - q1[0], sy - q1[1])
    print(f"  endpoint error = {err:.6f}")
    assert err < 0.1, f"endpoint error too large: {err}"
    print("  PASS: basic Dubins path")

    # 2. NodeTheta + steer_dubins_bangbang
    n0 = NodeTheta(0.0, 0.0, 0.0, 0.0)
    n1 = NodeTheta(5.0, 5.0, math.pi / 2, 0.0)
    result = steer_dubins_bangbang(n0, n1, turning_radius=1.0, a_max=4.0,
                                   v_max=4.0)
    assert result is not None and result["feasible"]
    print(f"\n  steer_dubins_bangbang: arc={result['arc_length']:.4f}, "
          f"t*={result['total_time']:.4f}, type={result['dubins_path_type']}")
    traj = result["trajectory"]
    print(f"  trajectory shape = {traj.shape}")
    assert traj.shape[1] == 5
    print("  PASS: steer_dubins_bangbang")

    # 3. turning radius helper
    cfg = {"diff_drive": {"wheelbase": 0.06},
           "physical_dimensions": {"maze_width_meters": 2.88}}
    r = compute_turning_radius_cells(cfg, num_cols=16)
    # 2.88/16 = 0.18 m/cell, turning_radius = 0.03/0.18 = 0.1667 cells
    print(f"\n  turning_radius = {r:.4f} cells  (expected 0.1667)")
    assert abs(r - 0.03 / (2.88 / 16)) < 1e-9
    print("  PASS: compute_turning_radius_cells")

    print("\nAll tests passed.")


if __name__ == "__main__":
    _selftest()
