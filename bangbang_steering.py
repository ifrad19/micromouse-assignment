"""
1D bang-bang steering wrapper around the LaValle double-integrator helpers.

Dynamics: x_dot = v, v_dot = a (|a| <= a_max).
Time-optimal control from (x_init, v_init) to (x_goal, v_goal) is at most
two acceleration switches. See LaValle, Sakcak, LaValle (IROS 2023).

BSD 2-Clause License
Copyright (c) 2023, Alexander J. LaValle
"""

from math import sqrt, fabs
import numpy as np

time_epsilon = 0.0000001
float_epsilon = 1.0e-200


# --- Original LaValle helpers (n=1 instances, verbatim) ---


def control_time_1d(con):
    """Total time of a control sequence [[a, t_seg], ...]."""
    t = 0.0
    for c in con:
        t += c[1]
    return t


def bang_bang_optimal_1d(ix, iv, gx, gv, umin=-1.0, umax=1.0):
    """Time-optimal bang-bang from (ix, iv) to (gx, gv). Returns [[a, t], ...]."""
    if ix == gx and iv == gv:
        return []

    invmin = 1.0 / umin
    invmax = 1.0 / umax
    c1 = ix - gx - 0.5 * (invmin * iv * iv - invmax * gv * gv)
    a1 = 0.5 * (invmin - invmax)
    s1 = -4.0 * a1 * c1

    c2 = ix - gx - 0.5 * (invmax * iv * iv - invmin * gv * gv)
    s2 = 4.0 * a1 * c2

    t1 = t1b = 1.0e20
    t2 = t2b = 1.0e20
    u1 = u1b = umin
    u2 = u2b = umax

    if s1 >= 0:
        xdot = sqrt(s1) / (2.0 * a1)
        t1 = invmin * (xdot - iv)
        t2 = invmax * (gv - xdot)
        u1 = umin
        u2 = umax

    if s2 >= 0:
        xdot = -sqrt(s2) / (2.0 * a1)
        t1b = invmax * (xdot - iv)
        t2b = invmin * (gv - xdot)
        u1b = umax
        u2b = umin

    if fabs(t1) < time_epsilon:
        t1 = 0.0
    if fabs(t2) < time_epsilon:
        t2 = 0.0
    if fabs(t1b) < time_epsilon:
        t1b = 0.0
    if fabs(t2b) < time_epsilon:
        t2b = 0.0

    if (t1b + t2b < t1 + t2 and t1b >= 0.0 and t2b >= 0.0) or t1 < 0.0 or t2 < 0.0:
        t1 = t1b
        t2 = t2b
        u1 = u1b
        u2 = u2b

    if t1 == 0.0:
        return [[u2, t2]]
    if t2 == 0.0:
        return [[u1, t1]]
    return [[u1, t1], [u2, t2]]


def bang_bang_hard_stop_1d(ix, iv, gx, gv, umin=-1.0, umax=1.0):
    """Decelerate to v=0 first, then bang-bang from rest to (gx, gv)."""
    if iv > 0:
        ux = umin
        s = -iv / ux
        sx = ix + iv * s + 0.5 * ux * s * s
    else:
        ux = umax
        s = -iv / ux
        sx = ix + iv * s + 0.5 * ux * s * s
    d = bang_bang_optimal_1d(sx, 0.0, gx, gv, umin, umax)
    c = []
    if s > 0.0:
        c.append([ux, s])
    c += d
    return c


def bang_bang_hard_stop_wait_1d(ix, iv, gx, gv, tf, umin=-1.0, umax=1.0):
    """Hard-stop trajectory padded with a wait to fit exactly duration tf."""
    c = bang_bang_hard_stop_1d(ix, iv, gx, gv, umin, umax)
    tt = control_time_1d(c)
    if tt > tf:
        return []
    if tf > tt:
        if iv == 0.0:
            c.insert(0, [0.0, tf - tt])
        else:
            c.insert(1, [0.0, tf - tt])
    return c


# --- New wrappers (trajectory sampling, 1D integration) ---


def integrate_control_1d(x0, v0, control):
    """Integrate control from (x0, v0). Returns (x_final, v_final)."""
    x = x0
    v = v0
    for a, dt in control:
        x += v * dt + 0.5 * a * dt * dt
        v += a * dt
    return x, v


def sample_trajectory_1d(x0, v0, control, n_samples=None, samples_per_second=50.0):
    """Sample (x, v, t) along a 1D bang-bang trajectory.

    Args:
        n_samples: explicit count (overrides samples_per_second)
        samples_per_second: sampling density (default 50 Hz)
    Returns:
        np.ndarray (N, 3) columns [x, v, t]
    """
    total_t = control_time_1d(control)
    if total_t <= 0.0:
        return np.array([[x0, v0, 0.0]])
    if n_samples is None:
        n_samples = max(2, int(total_t * samples_per_second) + 1)
    ts = np.linspace(0.0, total_t, n_samples)
    samples = np.empty((n_samples, 3), dtype=float)
    samples[0] = [x0, v0, 0.0]

    # Build cumulative segment boundaries
    boundaries = []
    t_acc = 0.0
    for a, dt in control:
        t_acc += dt
        boundaries.append(t_acc)

    seg_idx = 0
    t_in_seg = 0.0
    a_cur = control[0][0] if control else 0.0
    for i in range(1, n_samples):
        t = ts[i]
        # advance seg_idx to the segment containing t
        while seg_idx < len(control) - 1 and t > boundaries[seg_idx] + time_epsilon:
            seg_idx += 1
        a_cur = control[seg_idx][0]
        t_in_seg = t - (boundaries[seg_idx - 1] if seg_idx > 0 else 0.0)
        v_i = v0
        # integrate from start up to t_in_seg within current seg accounting for prior segs
        v_i = v0
        x_i = x0
        t_done = 0.0
        for j, (a_j, dt_j) in enumerate(control):
            if j < seg_idx:
                x_i += v_i * dt_j + 0.5 * a_j * dt_j * dt_j
                v_i += a_j * dt_j
                t_done += dt_j
            else:
                dt_part = t_in_seg
                x_i += v_i * dt_part + 0.5 * a_j * dt_part * dt_part
                v_i += a_j * dt_part
                t_done += dt_part
                break
        samples[i] = [x_i, v_i, t]

    return samples


def bang_bang_steer_1d(x_init, v_init, x_goal, v_goal, a_max, t_max=None,
                        samples_per_second=50.0):
    """Time-optimal 1D steering with |a| <= a_max.

    Returns dict with 'control', 'trajectory' (N,3), 'total_time', 'feasible',
    or None if no solution exists.
    """
    if a_max <= 0:
        raise ValueError("a_max must be positive")
    umin = -a_max
    umax = a_max

    if abs(x_init - x_goal) < time_epsilon and abs(v_init - v_goal) < time_epsilon:
        return {
            'control': [],
            'trajectory': np.array([[x_init, v_init, 0.0]]),
            'total_time': 0.0,
            'feasible': True,
        }

    control = bang_bang_optimal_1d(x_init, v_init, x_goal, v_goal, umin, umax)
    if not control:
        control = bang_bang_hard_stop_1d(x_init, v_init, x_goal, v_goal, umin, umax)
    if not control:
        return None

    xf, vf = integrate_control_1d(x_init, v_init, control)
    if abs(xf - x_goal) > 1e-3 or abs(vf - v_goal) > 1e-3:
        xf2, vf2 = integrate_control_1d(x_init, v_init, control)
        if abs(xf2 - x_goal) > 1e-2 or abs(vf2 - v_goal) > 1e-2:
            return None

    total_time = control_time_1d(control)
    if t_max is not None and total_time > t_max + time_epsilon:
        return None

    trajectory = sample_trajectory_1d(x_init, v_init, control,
                                      samples_per_second=samples_per_second)

    return {
        'control': control,
        'trajectory': trajectory,
        'total_time': total_time,
        'feasible': True,
    }


def steer_position_only(x_init, v_init, distance, a_max, t_max=None,
                          end_velocity=0.0, samples_per_second=50.0):
    """Steer to a position ``distance`` ahead, ending at ``end_velocity``."""
    return bang_bang_steer_1d(
        x_init=x_init,
        v_init=v_init,
        x_goal=x_init + distance,
        v_goal=end_velocity,
        a_max=a_max,
        t_max=t_max,
        samples_per_second=samples_per_second,
    )


# --- Tests ---


def _stress_test_bang_bang_steer_1d(n=1000, seed=0):
    """1000 random cases, verify integration error < 1e-3."""
    import random
    rng = random.Random(seed)
    failures = []
    for i in range(n):
        x_init = rng.uniform(-10.0, 10.0)
        v_init = rng.uniform(-5.0, 5.0)
        x_goal = rng.uniform(-10.0, 10.0)
        v_goal = rng.uniform(-5.0, 5.0)
        a_max = rng.uniform(0.5, 3.0)
        result = bang_bang_steer_1d(x_init, v_init, x_goal, v_goal, a_max)
        if result is None:
            failures.append((i, 'infeasible', x_init, v_init, x_goal, v_goal, a_max))
            continue
        xf, vf = integrate_control_1d(x_init, v_init, result['control'])
        err_x = abs(xf - x_goal)
        err_v = abs(vf - v_goal)
        if err_x > 1e-3 or err_v > 1e-3:
            failures.append((i, f'err_x={err_x:.4e} err_v={err_v:.4e}',
                             x_init, v_init, x_goal, v_goal, a_max))
    return failures


def _compare_to_time_optimal_analytic(n=200, seed=1):
    """Verify v_init=v_goal=0 cases match t* = sqrt(4d/a_max)."""
    import random
    rng = random.Random(seed)
    failures = []
    for i in range(n):
        d = rng.uniform(0.5, 20.0)
        a_max = rng.uniform(0.5, 3.0)
        t_analytic = sqrt(4.0 * d / a_max)
        result = bang_bang_steer_1d(0.0, 0.0, d, 0.0, a_max)
        if result is None:
            failures.append((i, 'infeasible', d, a_max))
            continue
        t_computed = result['total_time']
        if abs(t_computed - t_analytic) > 1e-3:
            failures.append((i, f't_analytic={t_analytic:.4f} t_computed={t_computed:.4f}',
                             d, a_max))
    return failures


if __name__ == "__main__":
    print("=" * 60)
    print("bangbang_steering.py self-test")
    print("=" * 60)

    print("\n[1] 1000 random quintuples: integration error check")
    failures = _stress_test_bang_bang_steer_1d(n=1000)
    if failures:
        print(f"  FAILED: {len(failures)} of 1000 cases did not converge")
        for f in failures[:5]:
            print(f"    {f}")
    else:
        print("  PASS: all 1000 cases integrated to within 1e-3 of goal")

    print("\n[2] 200 analytic time-optimal comparisons (v_init=v_goal=0)")
    failures = _compare_to_time_optimal_analytic(n=200)
    if failures:
        print(f"  FAILED: {len(failures)} of 200 cases did not match t* = sqrt(4d/a_max)")
        for f in failures[:5]:
            print(f"    {f}")
    else:
        print("  PASS: all 200 cases matched analytic time-optimal")

    print("\n[3] trajectory sampling density sanity")
    r = bang_bang_steer_1d(0.0, 0.0, 5.0, 0.0, 1.0)
    print(f"  d=5, a_max=1, t*={r['total_time']:.4f}, samples={len(r['trajectory'])}")
    assert r['trajectory'][0, 2] == 0.0
    assert abs(r['trajectory'][-1, 2] - r['total_time']) < 1e-9
    assert abs(r['trajectory'][-1, 0] - 5.0) < 1e-3
    assert abs(r['trajectory'][-1, 1] - 0.0) < 1e-3
    print("  PASS: trajectory endpoints correct")

    print("\n[4] infeasibility test (huge t_max under t bound)")
    r = bang_bang_steer_1d(0.0, 0.0, 100.0, 0.0, 1.0, t_max=1.0)
    if r is None:
        print("  PASS: t_max=1.0 with d=100, a_max=1 correctly rejected (inf. time)")
    else:
        print(f"  FAIL: should have rejected, got t={r['total_time']:.3f}")

    print("\nAll tests done.")
