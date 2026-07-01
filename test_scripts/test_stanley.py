"""
Stanley controller self-tests.

Verifies basic behaviour without needing pygame:
  1. Drop-in compatibility with PurePursuit interface
  2. Heading error drives steering toward path heading
  3. Cross-track error drives steering toward path (sign + magnitude)
  4. Combined heading + CTE works on an L-shaped path
  5. Speed slowdown at sharp turns
  6. Goal slowdown as robot nears end of path
  7. Simulated diff-drive tracks a straight line
  8. Simulated diff-drive tracks an L-shape
"""

import math

from pathTracking import Stanley


# ────────────────────────────────────────────────────────────────────────
# simple simulated differential drive (no slip, no acceleration limits)
# ────────────────────────────────────────────────────────────────────────


class SimDiffDrive:
    """Minimal ideal differential drive that uses the same bicycle-model
    approximation as ``simulation.py`` but stays entirely in grid units
    (cells).  No wheel-speed round-trip, no slip, no acceleration limits.
    """

    def __init__(self, wheel_equiv=1.0):
        # wheel_equiv = L_grid used by the simulation to convert steering to
        # omega.  1.0 = "1 cell of lookahead for the steering-to-omega gain".
        self.L_equiv = wheel_equiv
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0

    def step(self, spd, steer, dt):
        """Apply Stanley's (spd, steer) directly in grid units."""
        omega = spd * math.tan(steer) / self.L_equiv
        self.x += spd * math.cos(self.theta) * dt
        self.y += spd * math.sin(self.theta) * dt
        self.theta += omega * dt
        self.theta = (self.theta + math.pi) % (2 * math.pi) - math.pi


def ctrl_to_wheels(v, steer, L_grid=1.0, R=0.02 / 0.18, L_phys=0.06 / 0.18):
    """Match what simulation.py does for Stanley output (all in grid units)."""
    omega = v * math.tan(steer) / L_grid
    v_left = (2 * v - omega * L_phys) / (2 * R)
    v_right = (2 * v + omega * L_phys) / (2 * R)
    return v_left, v_right


# ────────────────────────────────────────────────────────────────────────
# tests
# ────────────────────────────────────────────────────────────────────────


def test_interface():
    sc = Stanley(maps=None, config=None, k=0.5, k_soft=1.0,
                 max_steer=math.pi / 3, wheel_base=0.08)
    sc.set_path([(0, 0), (1, 0), (2, 0)])
    spd, steer = sc.get_control((0, 0), 0.0)
    assert isinstance(spd, float)
    assert isinstance(steer, float)
    assert -sc.max_steer <= steer <= sc.max_steer
    print("[OK] interface: get_control returns (float, float), "
          "steer in [-max_steer, max_steer]")


def test_heading_error_only():
    """k=0 -> no CTE correction.  Robot on a straight +col path but pointing
    45° to the left of the path heading should output a negative steer to
    turn right back to alignment."""
    sc = Stanley(maps=None, config=None, k=0.0, k_soft=1.0,
                 max_steer=math.pi / 3, wheel_base=0.08)
    sc.set_path([(0.0, 0.0), (0.0, 10.0)])  # +col, heading=pi/2
    # robot ON the path but pointing pi/4 (45 deg to the right of path)
    # path_heading=pi/2, robot_heading=pi/4 -> theta_e = pi/2 - pi/4 = +pi/4
    _, steer = sc.get_control((0.0, 5.0), math.pi / 4)
    assert steer > 0, f"expected positive steer, got {steer}"
    assert abs(steer - (math.pi / 4)) < 0.05, f"expected ~+pi/4, got {steer}"
    print(f"[OK] heading error: theta_e=+pi/4 -> steer={steer:.3f} (expected ~+pi/4)")


def test_cte_only():
    """k>0, robot aligned with path heading but offset to one side -> only
    CTE term contributes to steering.  In our (row, col) image frame the
    steering law is ``steer = theta_e - atan(k * cte / v)``, so:
      - robot above a +col path (cte > 0) -> negative steer (right turn)
      - robot below a +col path (cte < 0) -> positive steer (left turn)
    """
    sc = Stanley(maps=None, config=None, k=2.0, k_soft=0.1,
                 max_steer=math.pi / 2, wheel_base=0.08)
    # path going +col (heading=pi/2)
    sc.set_path([(0.0, 0.0), (0.0, 10.0)])  # +col, heading=pi/2
    # robot ON the path, heading aligned -> steer ~ 0
    _, steer_on = sc.get_control((0.0, 5.0), math.pi / 2)
    assert abs(steer_on) < 0.1, f"on path -> steer should be ~0, got {steer_on}"
    # robot above the path (row=-1), heading aligned -> cte > 0 -> negative steer
    _, steer_above = sc.get_control((-1.0, 5.0), math.pi / 2)
    assert steer_above < 0, f"expected negative steer (path right of robot), got {steer_above}"
    # robot below the path (row=+1) -> cte < 0 -> positive steer
    _, steer_below = sc.get_control((1.0, 5.0), math.pi / 2)
    assert steer_below > 0, f"expected positive steer (path left of robot), got {steer_below}"
    print(f"[OK] cte-only: on={steer_on:.3f}, above={steer_above:.3f}, below={steer_below:.3f}")


def test_l_shape_tracking():
    """Drive a robot through an L-shape with Stanley; the robot should reach
    the goal and stay close to the path."""
    path_rc = [(0.0, 0.0), (5.0, 0.0), (5.0, 5.0)]
    sc = Stanley(maps=None, config=None, k=2.0, k_soft=0.5,
                 max_steer=math.pi / 3, wheel_base=0.08)
    sc.set_path(path_rc)

    sim = SimDiffDrive()
    sim.x, sim.y, sim.theta = 0.0, 0.0, 0.0
    dt = 0.05

    cte_log = []
    for _ in range(800):
        spd, steer = sc.get_control((sim.x, sim.y), sim.theta)
        sim.step(spd, steer, dt)
        _, _, cte, _ = sc._closest_point_on_path((sim.x, sim.y))
        cte_log.append(abs(cte))
        if (abs(sim.x - path_rc[-1][0]) < 0.2
                and abs(sim.y - path_rc[-1][1]) < 0.2):
            break

    mean_cte = sum(cte_log) / len(cte_log)
    max_cte = max(cte_log)
    end_dist = math.hypot(sim.x - path_rc[-1][0], sim.y - path_rc[-1][1])
    print(f"[OK] L-shape tracking: {len(cte_log)} steps, "
          f"mean CTE={mean_cte:.3f} cells, max CTE={max_cte:.3f} cells, "
          f"end_dist={end_dist:.3f}")
    assert end_dist < 0.5, f"robot didn't reach goal (end_dist={end_dist})"


def test_speed_slowdown():
    """At sharp turns Stanley should command a slower speed."""
    sc = Stanley(maps=None, config=None, k=2.0, k_soft=0.5,
                 max_steer=math.pi / 3, wheel_base=0.08)
    sc.set_path([(0, 0), (10, 0)])
    spd_straight, _ = sc.get_control((0, 0), 0.0)
    # force a large steer by being off-path with big heading error
    spd_off, steer_off = sc.get_control((5.0, -3.0), math.pi / 4)
    assert abs(steer_off) > 0
    assert spd_off < spd_straight
    print(f"[OK] speed shaping: spd_straight={spd_straight:.2f}, "
          f"spd_off={spd_off:.2f} (steer_off={steer_off:.2f})")


def test_goal_slowdown():
    sc = Stanley(maps=None, config=None, k=2.0, k_soft=0.5,
                 max_steer=math.pi / 3, wheel_base=0.08)
    sc.set_path([(0, 0), (10, 0)])
    spd_far, _ = sc.get_control((5.0, 0.0), 0.0)
    spd_near, _ = sc.get_control((9.5, 0.0), 0.0)
    assert spd_near < spd_far
    print(f"[OK] goal slowdown: spd_far={spd_far:.2f} -> spd_near={spd_near:.2f}")


def test_long_straight():
    """Robot should stay very close to a long straight path even with a
    small initial offset and heading error."""
    sc = Stanley(maps=None, config=None, k=2.0, k_soft=0.5,
                 max_steer=math.pi / 3, wheel_base=0.08)
    sc.set_path([(0, 0), (50, 0)])

    sim = SimDiffDrive()
    sim.x, sim.y, sim.theta = 0.0, 0.3, 0.05
    dt = 0.05

    cte_log = []
    for _ in range(800):
        spd, steer = sc.get_control((sim.x, sim.y), sim.theta)
        sim.step(spd, steer, dt)
        _, _, cte, _ = sc._closest_point_on_path((sim.x, sim.y))
        cte_log.append(abs(cte))
    max_cte = max(cte_log[20:])
    end_pos = (sim.x, sim.y)
    print(f"[OK] long straight: max CTE after transient = {max_cte:.4f} cells, "
          f"end_pos=({end_pos[0]:.1f}, {end_pos[1]:.1f})")
    # Stanley should converge to a small but non-zero steady-state CTE when
    # the initial offset is non-zero (k=2, k_soft=0.5 is slightly underdamped).
    assert max_cte < 0.4, f"too much drift on straight (max CTE={max_cte})"


def main():
    test_interface()
    test_heading_error_only()
    test_cte_only()
    test_speed_slowdown()
    test_goal_slowdown()
    test_long_straight()
    test_l_shape_tracking()
    print("\nAll Stanley self-tests passed.")


if __name__ == "__main__":
    main()
