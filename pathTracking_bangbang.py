"""
Open-Loop Bang-Bang Tracker

Executes a bang-bang planned trajectory by playing back the acceleration
schedule through the differential-drive robot. Ignores Pure Pursuit / Stanley
feedback — commands the planned control directly, with the diff-drive model's
wheel-acceleration limit clipping open-loop commands.

Coordinate conventions (matching simulation.py and micromouse.py):
  - pos is (row, col) = (y, x) in maze coordinates
  - heading is in radians, 0 means facing down (+y), increasing CCW
  - speeds are in maze units per second (or m/s, scaled externally)

The bang-bang trajectory has the structure produced by
bangbang_rrt_star.BangBangRRTStar.plan() and smooth_path_with_bang_bang().
Both yield a list of edges, each with a control sequence [[a, t_seg], ...]
and a heading direction (parent->child).
"""

import math
import numpy as np

from diff_drive_robot import DifferentialDriveRobot, DiffDriveController


# --- Conversion helpers ---


def _edge_heading(parent_xy, child_xy):
    """Heading of an edge in the (row, col) maze frame, radians.

    Simulation integration:
        d_row = v * cos(heading) * dt
        d_col = v * sin(heading) * dt
    so heading=0 means +row (down), heading=pi/2 means +col (right).

    parent_xy, child_xy are (x, y) = (col, row) tuples.
    d_row = child.y - parent.y
    d_col = child.x - parent.x
    heading = atan2(d_col, d_row) so that the kinematics match.
    """
    d_col = child_xy[0] - parent_xy[0]
    d_row = child_xy[1] - parent_xy[1]
    if d_col == 0 and d_row == 0:
        return 0.0
    return math.atan2(d_col, d_row)


def _integrate_control_velocity(v0, control):
    """Return the (time, v) profile of a 1D bang-bang control starting at v0.

    Returns a list of (t, v) samples at the natural control breakpoints plus
    initial state."""
    samples = [(0.0, v0)]
    t = 0.0
    v = v0
    for a, dt in control:
        v_new = v + a * dt
        t += dt
        samples.append((t, v_new))
        v = v_new
    return samples


# --- Open-loop trajectory execution ---


def execute_bangbang_trajectory(plan_result, config, meters_per_cell=1.0,
                                 render_fn=None, max_time_factor=2.0,
                                 goal_xy=None, goal_tolerance=0.6,
                                 collision_fn=None,
                                 verbose=False):
    """Execute a bang-bang planned trajectory open-loop through the diff-drive.

    Args:
        plan_result: dict returned by BangBangRRTStar.plan() or
                     smooth_path_with_bang_bang()
        config: full YAML config dict
        meters_per_cell: physical metres per maze cell
        render_fn: optional callable(pos, heading, t) for live visualisation
        max_time_factor: stop if elapsed > total_planned_time * this factor
        goal_xy: (gx, gy) goal position (maze (row, col) coords)
        goal_tolerance: distance from goal considered "reached"
        collision_fn: callable (row, col) -> bool (True = in collision)

    Returns dict with 'trajectory', 'completion_time', 'reached_goal',
    'collided', 'planned_time', 'planned_distance'.
    """
    if plan_result is None:
        return {
            'trajectory': [], 'completion_time': 0.0, 'reached_goal': False,
            'collided': False, 'planned_time': 0.0, 'planned_distance': 0.0,
        }

    controls = plan_result.get('controls', [])
    if not controls:
        return {
            'trajectory': [], 'completion_time': 0.0, 'reached_goal': False,
            'collided': False, 'planned_time': 0.0, 'planned_distance': 0.0,
        }

    if 'edges' in plan_result and plan_result['edges']:
        edges = plan_result['edges']
        waypoints = []
        for parent, child in edges:
            waypoints.append((parent.x, parent.y))
        waypoints.append((edges[-1][1].x, edges[-1][1].y))
    else:
        waypoints = plan_result.get('waypoints', [])

    if len(waypoints) < 2 or len(controls) != len(waypoints) - 1:
        if verbose:
            print(f"  [openloop] inconsistent plan: {len(waypoints)} waypoints, "
                  f"{len(controls)} controls")
        return {
            'trajectory': [], 'completion_time': 0.0, 'reached_goal': False,
            'collided': False, 'planned_time': 0.0, 'planned_distance': 0.0,
        }

    robot = DifferentialDriveRobot(config)
    dd_controller = DiffDriveController(robot.wheel_radius, robot.wheelbase)
    dt = config.get('micromouse', {}).get('dt', 0.01667)

    r, c = waypoints[0]
    heading = _edge_heading(waypoints[0], waypoints[1])
    v_init = 0.0

    edge_schedules = []
    total_time = 0.0
    for i, control in enumerate(controls):
        v_profile = _integrate_control_velocity(v_init, control)
        t_profile = [t + total_time for t, _ in v_profile]
        edge_schedules.append((t_profile, [v for _, v in v_profile]))
        if v_profile:
            total_time = t_profile[-1]
        v_init = v_profile[-1][1] if v_profile else 0.0

    edge_plans = []
    for i, control in enumerate(controls):
        start_xy = waypoints[i]
        end_xy = waypoints[i + 1]
        h = _edge_heading(start_xy, end_xy)
        t_prof, v_prof = edge_schedules[i]
        edge_plans.append({
            'start': start_xy,
            'end': end_xy,
            'heading': h,
            'v_profile': v_prof,
            't_profile': t_prof,
        })

    planned_distance = 0.0
    for i in range(len(waypoints) - 1):
        dx = waypoints[i + 1][0] - waypoints[i][0]
        dy = waypoints[i + 1][1] - waypoints[i][1]
        planned_distance += math.hypot(dx, dy)

    traj_log = []
    sim_t = 0.0
    current_edge = 0
    collision = False
    reached_goal = False
    max_sim_t = total_time * max_time_factor

    robot.wheel_vel_left = 0.0
    robot.wheel_vel_right = 0.0

    n_steps = int(math.ceil(max_sim_t / dt)) + 1
    if verbose:
        print(f"  [openloop] starting execution: {n_steps} steps, {len(edge_plans)} edges")
        for i, ep in enumerate(edge_plans):
            print(f"    edge {i}: from {ep['start']} to {ep['end']} "
                  f"heading={math.degrees(ep['heading']):.1f}° "
                  f"v_profile={ep['v_profile']} t_profile={ep['t_profile']}")
    for step in range(n_steps):
        if current_edge >= len(edge_plans):
            if goal_xy is not None:
                gr, gc = goal_xy
                if math.hypot(r - gr, c - gc) < goal_tolerance:
                    reached_goal = True
            break

        plan = edge_plans[current_edge]
        if sim_t >= plan['t_profile'][-1] - 1e-9:
            current_edge += 1
            if verbose:
                if current_edge < len(edge_plans):
                    nh = math.degrees(edge_plans[current_edge]['heading'])
                else:
                    nh = float('nan')
                print(f"  [openloop] t={sim_t:.3f}: edge {current_edge-1} -> {current_edge}, "
                      f"new heading={nh:.1f}°")
            if current_edge >= len(edge_plans):
                if goal_xy is not None:
                    gr, gc = goal_xy
                    if math.hypot(r - gr, c - gc) < goal_tolerance:
                        reached_goal = True
                break
            plan = edge_plans[current_edge]
            heading = plan['heading']

        t_prof = plan['t_profile']
        v_prof = plan['v_profile']
        idx = max(0, min(len(t_prof) - 2,
                         int(np.searchsorted(t_prof, sim_t, side='right')) - 1))
        if idx + 1 < len(t_prof) and t_prof[idx + 1] > t_prof[idx]:
            frac = (sim_t - t_prof[idx]) / (t_prof[idx + 1] - t_prof[idx])
            v_target = v_prof[idx] * (1 - frac) + v_prof[idx + 1] * frac
        else:
            v_target = v_prof[min(idx, len(v_prof) - 1)]

        v_linear_mps = v_target * meters_per_cell
        v_wheel_target = v_linear_mps / robot.wheel_radius
        robot.set_wheel_velocities(v_wheel_target, v_wheel_target, dt)

        actual_v_m_per_s, actual_omega = robot.update_kinematics(dt)
        actual_v_cells = actual_v_m_per_s / meters_per_cell

        r += actual_v_cells * math.cos(heading) * dt
        c += actual_v_cells * math.sin(heading) * dt

        sim_t += dt
        traj_log.append((r, c, heading, sim_t, actual_v_cells))

        if render_fn is not None:
            render_fn((r, c), heading, sim_t)

        if not (math.isfinite(r) and math.isfinite(c)):
            collision = True
            if verbose:
                print(f"  [openloop] NaN at step {step}, t={sim_t:.3f}")
            break
        if collision_fn is not None and collision_fn(r, c):
            collision = True
            if verbose:
                print(f"  [openloop] collision at ({r:.2f},{c:.2f}), t={sim_t:.3f}")
            break

    return {
        'trajectory': traj_log,
        'completion_time': sim_t,
        'reached_goal': reached_goal,
        'collided': collision,
        'planned_time': total_time,
        'planned_distance': planned_distance,
    }


# --- Smoke test ---


def _smoke_test_tracker():
    """Plan a trajectory on a small maze, play it back, verify goal reached."""
    import sys
    import os
    sys.path.insert(0, os.path.dirname(__file__))
    from bangbang_rrt_star import BangBangRRTStar, make_grid_collision_fn
    from micromouse import load_maze, Map

    small_csv = "/tmp/small_maze.csv"
    if not os.path.exists(small_csv):
        print("Small maze CSV not found.")
        return

    maze_array = load_maze(small_csv)
    m = Map(maze_array)
    grid = m.get_grid_representation()
    grid_walls = (grid == 1).astype(int)
    sy, sx = m.start
    gy, gx = m.goal
    n_rows, n_cols = grid.shape
    collision_fn = make_grid_collision_fn(grid_walls, robot_radius=0.0)
    goal_fn = lambda x, y: (abs(x - gx) <= 0.6 and abs(y - gy) <= 0.6)

    print("=" * 60)
    print("open-loop tracker smoke test")
    print("=" * 60)

    planner = BangBangRRTStar(
        x_min=0.0, x_max=float(n_cols),
        y_min=0.0, y_max=float(n_rows),
        v_max=4.0, a_max=8.0,
        collision_fn=collision_fn, goal_fn=goal_fn,
        max_iter=3000, goal_bias=0.2, max_steer_dist=1.0,
        rewire_gamma=2.0, rewire_k=20, seed=42, verbose=False,
    )
    result = planner.plan(
        start_xy=(sx, sy), goal_xy=(gx, gy),
        start_v=0.0, goal_tolerance=0.6, goal_v=0.0,
    )
    if result is None:
        print("  planner failed, cannot test tracker")
        return
    print(f"  Planner: {len(result['nodes'])} nodes, "
          f"best_cost={result['best_cost']:.3f}s, "
          f"{len(result['waypoints'])} waypoints")
    print(f"  Waypoints (col, row):")
    for i, (x, y) in enumerate(result['waypoints']):
        print(f"    {i}: ({x:.2f}, {y:.2f})")

    config = {
        'diff_drive': {
            'wheel_radius': 0.02,
            'wheelbase': 0.06,
            'max_wheel_speed': 250.0,
            'max_acceleration': 400.0,
            'use_lateral_accel_model': False,
            'base_slip_factor': 0.9999,
            'velocity_slip_factor': 0.0,
            'turn_rate_slip_factor': 0.0,
            'combined_slip_factor': 0.0,
            'max_slip': 0.0,
            'velocity_noise': 0.0,
        },
        'micromouse': {'dt': 0.02},
        'physical_dimensions': {'use_metric_speeds': False},
    }
    meters_per_cell = 1.0

    collision_fn_exec = lambda r, c: collision_fn(c, r)

    exec_result = execute_bangbang_trajectory(
        plan_result=result, config=config,
        meters_per_cell=meters_per_cell,
        max_time_factor=2.0, verbose=True,
        goal_xy=(gy, gx), goal_tolerance=0.8,
        collision_fn=collision_fn_exec,
    )
    print(f"  completion_time={exec_result['completion_time']:.3f}s, "
          f"reached_goal={exec_result['reached_goal']}, collided={exec_result['collided']}")
    if exec_result['trajectory']:
        last_r, last_c, last_h, last_t, last_v = exec_result['trajectory'][-1]
        d_to_goal = math.hypot(last_r - gy, last_c - gx)
        print(f"  Final pos: row={last_r:.2f}, col={last_c:.2f}, "
              f"v={last_v:.3f}, dist to goal = {d_to_goal:.2f}")
    print(f"  Planned time: {exec_result['planned_time']:.3f}s, "
          f"planned distance: {exec_result['planned_distance']:.2f} cells")

    if exec_result['trajectory']:
        print("\n  Executed trajectory (every 10th step):")
        for i, (r, c, h, t, v) in enumerate(exec_result['trajectory']):
            if i % 10 == 0 or i == len(exec_result['trajectory']) - 1:
                print(f"    step={i:4d}  pos=({r:5.2f},{c:5.2f})  v={v:6.3f}  t={t:6.3f}")

    if exec_result['reached_goal'] and not exec_result['collided']:
        print("PASS")
    else:
        print("FAIL: robot did not reach goal or collided")


if __name__ == "__main__":
    _smoke_test_tracker()
