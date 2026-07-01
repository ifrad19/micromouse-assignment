"""
Stanley vs Pure Pursuit headless benchmark.

For each maze and each controller, runs the maze end-to-end and reports
the completion time, traversed distance, and number of collisions.

Reuses the same maze set as experiments/run_comparison.py.
"""

import argparse
import math
import os
import sys
import time
import contextlib
import io
import csv
import yaml

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from micromouse import load_maze, Map
from pathPlanning import AStar
from pathTracking import PurePursuit, Stanley
from diff_drive_robot import DifferentialDriveRobot, DiffDriveController


# 16x16 benchmark mazes from experiments/make_bench_mazes.py
DEFAULT_MAZES = [
    '/tmp/bench/corridor.csv',
    '/tmp/bench/open_field.csv',
    '/tmp/bench/spiral.csv',
    '/tmp/bench/narrow.csv',
]


def make_controller(tracker, config):
    if tracker == "stanley":
        return Stanley(maps=None, config=config)
    cfg = {**config, "pure_pursuit": {**config.get("pure_pursuit", {}), "debug": False}}
    return PurePursuit(None, cfg)


def plan(maze, config):
    start = maze.map[maze.start[0]][maze.start[1]]
    goal = maze.map[maze.goal[0]][maze.goal[1]]
    with contextlib.redirect_stdout(io.StringIO()):
        planner = AStar(maze, config)
        px, py = planner.plan_path(start, goal)
    if not px or not py:
        return None
    return list(zip(px, py)), planner.calculate_path_distance(px, py)


def run_trial(maze_file, config, tracker, max_steps=20000):
    """Run one trial; return dict of metrics."""
    maze_array = load_maze(maze_file)
    maze = Map(maze_array)

    # metric <-> grid conversion
    use_metric = config.get('physical_dimensions', {}).get('use_metric_speeds', False)
    mpc = 1.0
    if use_metric:
        pd = config['physical_dimensions']
        mpc = (pd['maze_width_meters'] / maze.col + pd['maze_height_meters'] / maze.row) / 2

    cfg = yaml.safe_load(yaml.safe_dump(config))
    for sect in ('pure_pursuit', 'stanley'):
        if sect in cfg:
            for k in ('max_speed', 'min_speed'):
                if k in cfg[sect]:
                    cfg[sect][k] /= mpc
    if 'min_speed_threshold' in cfg.get('micromouse', {}):
        cfg['micromouse']['min_speed_threshold'] /= mpc

    res = plan(maze, cfg)
    if res is None:
        return {'status': 'NO_PATH', 'maze': maze_file, 'tracker': tracker}
    path, planned_dist = res

    ctrl = make_controller(tracker, cfg)
    ctrl.set_path(path)

    robot = DifferentialDriveRobot(cfg)
    ddc = DiffDriveController(robot.wheel_radius, robot.wheelbase)
    dt = cfg.get('micromouse', {}).get('dt', 0.0167)
    init_h = cfg.get('micromouse', {}).get('initial_heading', math.pi / 4)
    goal_thr = cfg.get('micromouse', {}).get('goal_threshold', 0.5)

    pos = [maze.start[0] + 0.5, maze.start[1] + 0.5]
    heading = init_h
    sim_time = 0.0
    traversed = 0.0
    collision = False
    goal_reached = False
    max_cte = 0.0
    mean_cte_sum = 0.0
    mean_cte_n = 0
    final_step = 0
    radius = robot.wheelbase / mpc / 2.0

    for step in range(max_steps):
        spd, steer = ctrl.get_control(tuple(pos), heading)
        spd_m = spd * mpc
        omega = ddc.steering_to_omega(spd_m, steer, wheelbase_equiv=1.0 * mpc)
        v_left, v_right = ddc.velocity_to_wheels(spd_m, omega)
        robot.set_wheel_velocities(v_left, v_right, dt)
        actual_v_m, actual_omega = robot.update_kinematics(dt)
        actual_v = actual_v_m / mpc
        prev = tuple(pos)
        pos = [pos[0] + actual_v * math.cos(heading) * dt,
               pos[1] + actual_v * math.sin(heading) * dt]
        heading = (heading + actual_omega * dt + math.pi) % (2 * math.pi) - math.pi
        sim_time += dt
        traversed += math.hypot(pos[0] - prev[0], pos[1] - prev[1])

        # track cross-track error
        try:
            _, _, cte, _ = ctrl._closest_point_on_path(tuple(pos))
            cte_abs = abs(cte)
            if cte_abs > max_cte:
                max_cte = cte_abs
            mean_cte_sum += cte_abs
            mean_cte_n += 1
        except Exception:
            pass

        # collision check
        r, c = pos
        if not (0 <= r < maze.row and 0 <= c < maze.col):
            collision = True
            break
        collided = False
        for i in range(8):
            a = 2 * math.pi * i / 8
            cr = int(r + radius * math.cos(a))
            cc = int(c + radius * math.sin(a))
            if 0 <= cr < maze.row and 0 <= cc < maze.col:
                if maze.map[cr][cc].state == '#':
                    collided = True
                    break
        if collided:
            collision = True
            break

        # goal check
        gr, gc = path[-1]
        if math.hypot(r - gr, c - gc) < goal_thr:
            goal_reached = True
            final_step = step
            break

    return {
        'maze': os.path.basename(maze_file),
        'tracker': tracker,
        'status': 'GOAL' if goal_reached else ('COLLISION' if collision else 'TIMEOUT'),
        'time_s': sim_time,
        'traversed': traversed,
        'planned_dist': planned_dist,
        'steps': final_step + 1 if goal_reached else (step + 1),
        'max_cte': max_cte,
        'mean_cte': (mean_cte_sum / mean_cte_n) if mean_cte_n else 0.0,
        'efficiency': (planned_dist / max(traversed, 0.01)) if goal_reached else 0.0,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mazes', nargs='+', default=DEFAULT_MAZES)
    p.add_argument('--out', default='experiments/results/controller_comparison.csv')
    p.add_argument('--max-steps', type=int, default=20000)
    p.add_argument('--speed', type=float, default=1.0,
                   help='Override max_speed (m/s if metric) for both controllers')
    args = p.parse_args()

    with open('config.yaml') as f:
        base = yaml.safe_load(f)
    # override initial heading to face south for fair comparison
    base.setdefault('micromouse', {})['initial_heading'] = 0.0

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    rows = []
    print(f"{'Maze':12s} {'Tracker':12s} {'Status':10s} {'Time(s)':>9s} "
          f"{'Traversed':>10s} {'Planned':>8s} {'Steps':>6s} "
          f"{'MaxCTE':>7s} {'MeanCTE':>8s} {'Eff%':>6s}")
    for maze_file in args.mazes:
        for tracker in ('pure_pursuit', 'stanley'):
            cfg = yaml.safe_load(yaml.safe_dump(base))
            cfg['path_tracker'] = tracker
            cfg['pure_pursuit']['max_speed'] = args.speed
            cfg['pure_pursuit']['min_speed'] = args.speed * 0.1
            cfg['stanley']['max_speed'] = args.speed
            cfg['stanley']['min_speed'] = args.speed * 0.1
            cfg['stanley']['k'] = 0.6
            cfg['stanley']['k_soft'] = 0.4
            r = run_trial(maze_file, cfg, tracker, max_steps=args.max_steps)
            rows.append(r)
            print(f"{r['maze']:12s} {r['tracker']:12s} {r['status']:10s} "
                  f"{r['time_s']:9.3f} {r['traversed']:10.3f} {r['planned_dist']:8.1f} "
                  f"{r['steps']:6d} {r['max_cte']:7.3f} {r['mean_cte']:8.3f} "
                  f"{r['efficiency']*100:6.1f}")

    with open(args.out, 'w', newline='') as f:
        if rows:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
