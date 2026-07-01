"""
End-to-end test: load a maze, run both PurePursuit and Stanley in headless
mode, compare completion times.
"""

import sys
import math
import yaml
sys.path.insert(0, '/Users/ifrad/Documents/Fourth Term/Summer internship/micromouse-assignment-main')

import numpy as np

import contextlib
import io
from micromouse import Map, load_maze
from pathPlanning import AStar
from pathTracking import PurePursuit, Stanley
from diff_drive_robot import DifferentialDriveRobot, DiffDriveController


def make_controller(tracker, config):
    if tracker == "stanley":
        return Stanley(maps=None, config=config)
    cfg = {**config, "pure_pursuit": {**config.get("pure_pursuit", {}), "debug": False}}
    return PurePursuit(None, cfg)


def run_one(maze_file, config, tracker, max_steps=20000):
    maze_array = load_maze(maze_file)
    maze = Map(maze_array)

    use_metric = config.get('physical_dimensions', {}).get('use_metric_speeds', False)
    mpc = 1.0
    if use_metric:
        pd = config['physical_dimensions']
        mpc = (pd['maze_width_meters'] / maze.col + pd['maze_height_meters'] / maze.row) / 2
    meters_per_cell = mpc

    # convert controller speeds
    if 'pure_pursuit' in config:
        for key in ('max_speed', 'min_speed'):
            if key in config['pure_pursuit']:
                config['pure_pursuit'][key] /= meters_per_cell
    if 'stanley' in config:
        for key in ('max_speed', 'min_speed'):
            if key in config['stanley']:
                config['stanley'][key] /= meters_per_cell
    if 'micromouse' in config and 'min_speed_threshold' in config['micromouse']:
        config['micromouse']['min_speed_threshold'] /= meters_per_cell

    # plan
    start = maze.map[maze.start[0]][maze.start[1]]
    goal = maze.map[maze.goal[0]][maze.goal[1]]

    with contextlib.redirect_stdout(io.StringIO()):
        planner = AStar(maze, config)
        px, py = planner.plan_path(start, goal)
    if not px or not py:
        return {"status": "NO_PATH"}
    path = list(zip(px, py))

    # controller
    ctrl = make_controller(tracker, config)
    ctrl.set_path(path)

    # robot
    robot = DifferentialDriveRobot(config)
    ddc = DiffDriveController(robot.wheel_radius, robot.wheelbase)

    mc = config.get("micromouse", {})
    dt = mc.get("dt", 0.0167)
    goal_thr = mc.get("goal_threshold", 0.5)
    init_heading = mc.get("initial_heading", math.pi / 4)

    pos = list(maze.start)
    heading = init_heading
    sim_time = 0.0
    traversed = 0.0
    collision = False
    goal_reached = False
    initial_heading_for_log = init_heading

    # PygameSimulation convention: heading=0=+row, cos=h, sin=h
    # d_row = v*cos(h)*dt, d_col = v*sin(h)*dt
    for step in range(max_steps):
        spd, steer = ctrl.get_control(tuple(pos), heading)
        spd_m = spd * meters_per_cell
        omega = ddc.steering_to_omega(spd_m, steer,
                                      wheelbase_equiv=1.0 * meters_per_cell)
        v_left, v_right = ddc.velocity_to_wheels(spd_m, omega)
        robot.set_wheel_velocities(v_left, v_right, dt)
        actual_v_m, actual_omega = robot.update_kinematics(dt)
        actual_v = actual_v_m / meters_per_cell

        prev = tuple(pos)
        pos = [pos[0] + actual_v * math.cos(heading) * dt,
               pos[1] + actual_v * math.sin(heading) * dt]
        heading = (heading + actual_omega * dt + math.pi) % (2 * math.pi) - math.pi
        sim_time += dt
        traversed += math.hypot(pos[0] - prev[0], pos[1] - prev[1])

        # collision check (circular approx)
        wheelbase_m = robot.wheelbase
        radius = wheelbase_m / meters_per_cell / 2.0
        r, c = pos
        collision = False
        if not (0 <= r < maze.row and 0 <= c < maze.col):
            collision = True
            break
        n_check = 8
        for i in range(n_check):
            a = 2 * math.pi * i / n_check
            if (int(r + radius * math.cos(a)) >= 0
                and int(r + radius * math.cos(a)) < maze.row
                and int(c + radius * math.sin(a)) >= 0
                and int(c + radius * math.sin(a)) < maze.col
                and maze.map[int(r + radius * math.cos(a))][int(c + radius * math.sin(a))].state == "#"):
                collision = True
                break
        if collision:
            break

        # goal check
        gr, gc = path[-1]
        if math.hypot(r - gr, c - gc) < goal_thr:
            goal_reached = True
            break

    return {
        "status": "GOAL" if goal_reached else ("COLLISION" if collision else "TIMEOUT"),
        "time": sim_time,
        "traversed": traversed,
        "steps": step + 1,
        "n_path": len(path),
    }


def main():
    with open('config.yaml') as f:
        base = yaml.safe_load(f)

    for maze_file in ('/tmp/bench/corridor.csv',):
        print(f"\nMaze: {maze_file}")
        print(f"{'Tracker':12s} {'Status':10s} {'Time (s)':>10s} {'Traversed':>12s} {'Steps':>8s}")
        for tracker in ('pure_pursuit', 'stanley'):
            cfg = yaml.safe_load(yaml.safe_dump(base))
            cfg['path_tracker'] = tracker
            cfg.setdefault('pure_pursuit', {})['max_speed'] = 2.0
            cfg.setdefault('pure_pursuit', {})['min_speed'] = 0.1
            cfg.setdefault('stanley', {})['max_speed'] = 2.0
            cfg.setdefault('stanley', {})['min_speed'] = 0.1
            cfg.setdefault('micromouse', {})['initial_heading'] = 0.0
            r = run_one(maze_file, cfg, tracker, max_steps=20000)
            print(f"{tracker:12s} {r['status']:10s} {r['time']:10.3f} {r['traversed']:12.3f} {r['steps']:8d}")


if __name__ == "__main__":
    main()
