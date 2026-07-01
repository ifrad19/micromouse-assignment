"""
Planner path comparison: Same path, different planners.

Generates paths from all planners, saves them, then runs both Stanley and
PurePursuit controllers on each saved path to compare how path quality
affects controller performance.

Usage:
    python experiments/run_planner_path_comparison.py [--maze bench_mazes/corridor.csv]
"""

import argparse
import csv
import json
import math
import os
import sys
import time
import contextlib
import io

import yaml
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from micromouse import load_maze, Map
from pathPlanning import AStar, ThetaStar
from pathTracking import PurePursuit, Stanley
from diff_drive_robot import DifferentialDriveRobot, DiffDriveController
from experiments.planner_wrappers import (
    PLANNERS, run_one, path_length_xy, count_turns,
    center_path_away_from_walls,
)
from bangbang_rrt_star import smooth_path_with_bang_bang, make_grid_collision_fn


# --- Configuration ---

DEFAULT_MAZES = [
    'bench_mazes/corridor.csv',
    'bench_mazes/open_field.csv',
    'bench_mazes/spiral.csv',
    'bench_mazes/narrow.csv',
]

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
CHART_DIR = os.path.join(RESULTS_DIR, "charts")
os.makedirs(CHART_DIR, exist_ok=True)


# --- Path generation and saving ---

def generate_all_paths(maze_file, config):
    """Generate paths from all planners and return as dict."""
    maze_array = load_maze(maze_file)
    maze = Map(maze_array)
    start_xy = (maze.start[1], maze.start[0])
    goal_xy = (maze.goal[1], maze.goal[0])

    grid = maze.get_grid_representation()
    grid_walls = (grid == 1).astype(int)

    paths = {}
    for planner_name in PLANNERS:
        try:
            result = run_one(planner_name, maze, start_xy, goal_xy, config)
            if result['success'] and result['n_waypoints'] >= 2:
                wrapper_result = PLANNERS[planner_name](maze, start_xy, goal_xy, config)
                waypoints_xy = wrapper_result['waypoints']

                waypoints_xy = center_path_away_from_walls(waypoints_xy, grid_walls)

                waypoints_rc = [(wp[1], wp[0]) for wp in waypoints_xy]
                paths[planner_name] = {
                    'waypoints_xy': waypoints_xy,
                    'waypoints_rc': waypoints_rc,
                    'plan_time_s': result['plan_time_s'],
                    'path_length': result['path_length'],
                    'n_waypoints': result['n_waypoints'],
                    'n_turns': result['n_turns'],
                }
                print(f"  {planner_name:25s} -> {len(waypoints_rc):3d} waypoints, "
                      f"length={result['path_length']:.2f}, "
                      f"time={result['plan_time_s']:.4f}s")
            else:
                print(f"  {planner_name:25s} -> FAILED")
        except Exception as e:
            print(f"  {planner_name:25s} -> ERROR: {e}")

    return paths, maze


def save_paths(paths, maze_file, output_dir):
    """Save all paths to JSON files."""
    os.makedirs(output_dir, exist_ok=True)
    maze_name = os.path.splitext(os.path.basename(maze_file))[0]

    for planner_name, data in paths.items():
        filename = f"{maze_name}_{planner_name}_path.json"
        filepath = os.path.join(output_dir, filename)
        with open(filepath, 'w') as f:
            json.dump({
                'planner': planner_name,
                'maze': maze_file,
                'waypoints_rc': data['waypoints_rc'],
                'waypoints_xy': data['waypoints_xy'],
                'plan_time_s': data['plan_time_s'],
                'path_length': data['path_length'],
                'n_waypoints': data['n_waypoints'],
                'n_turns': data['n_turns'],
            }, f, indent=2)
    print(f"Saved {len(paths)} paths to {output_dir}")


# --- Controller simulation ---

def make_controller(tracker, config):
    """Create a path tracking controller."""
    if tracker == "stanley":
        return Stanley(maps=None, config=config)
    cfg = {**config, "pure_pursuit": {**config.get("pure_pursuit", {}), "debug": False}}
    return PurePursuit(None, cfg)


def compute_bb_rrt_time(waypoints_xy, maze, config):
    """Compute theoretical minimum traversal time using bang-bang speed profile."""
    if len(waypoints_xy) < 2:
        return {'total_time': float('inf'), 'feasible': False, 'path_length_cells': 0.0}

    bb_cfg = config.get('bb_rrt_star', {})
    grid = maze.get_grid_representation()
    grid_walls = (grid == 1).astype(int)
    collision_fn = make_grid_collision_fn(grid_walls, robot_radius=bb_cfg.get('robot_radius', 0.0))

    result = smooth_path_with_bang_bang(
        waypoints=waypoints_xy,
        a_max=bb_cfg.get('a_max', 8.0),
        v_max=bb_cfg.get('v_max', 4.0),
        collision_fn=collision_fn,
        traj_sample_step=0.05,
        grid=grid_walls,
    )

    return {
        'total_time': result.get('total_time', float('inf')),
        'feasible': result.get('feasible', False),
        'path_length_cells': path_length_xy(waypoints_xy),
    }


def run_controller_on_path(path_rc, maze, config, tracker, max_steps=20000):
    """Run a controller on a pre-computed (row, col) path. Return metrics."""
    ctrl = make_controller(tracker, config)
    ctrl.set_path(path_rc)

    robot = DifferentialDriveRobot(config)
    ddc = DiffDriveController(robot.wheel_radius, robot.wheelbase)

    use_metric = config.get('physical_dimensions', {}).get('use_metric_speeds', False)
    mpc = 1.0
    if use_metric:
        pd = config['physical_dimensions']
        mpc = (pd['maze_width_meters'] / maze.col + pd['maze_height_meters'] / maze.row) / 2

    dt = config.get('micromouse', {}).get('dt', 0.01667)
    init_h = config.get('micromouse', {}).get('initial_heading', 0.0)
    goal_thr = config.get('micromouse', {}).get('goal_threshold', 1.0)

    cfg = yaml.safe_load(yaml.safe_dump(config))
    for sect in ('pure_pursuit', 'stanley'):
        if sect in cfg:
            for k in ('max_speed', 'min_speed'):
                if k in cfg[sect]:
                    cfg[sect][k] /= mpc

    pos = [maze.start[0], maze.start[1]]
    heading = init_h
    sim_time = 0.0
    traversed = 0.0
    collision = False
    goal_reached = False
    max_cte = 0.0
    mean_cte_sum = 0.0
    mean_cte_n = 0
    final_step = 0

    for step in range(max_steps):
        spd, steer = ctrl.get_control(tuple(pos), heading)

        spd_m = spd * mpc
        omega = ddc.steering_to_omega(spd_m, steer, wheelbase_equiv=1.0 * mpc)
        v_left, v_right = ddc.velocity_to_wheels(spd_m, omega)

        robot.set_wheel_velocities(v_left, v_right, dt)
        actual_v_m, actual_omega = robot.update_kinematics(dt)
        actual_v = actual_v_m / mpc

        prev = list(pos)
        pos[0] += actual_v * math.cos(heading) * dt
        pos[1] += actual_v * math.sin(heading) * dt
        heading = (heading + actual_omega * dt + math.pi) % (2 * math.pi) - math.pi
        sim_time += dt
        traversed += math.hypot(pos[0] - prev[0], pos[1] - prev[1])

        try:
            _, _, cte, _ = ctrl._closest_point_on_path(tuple(pos))
            cte_abs = abs(cte)
            if cte_abs > max_cte:
                max_cte = cte_abs
            mean_cte_sum += cte_abs
            mean_cte_n += 1
        except Exception:
            pass

        r, c = pos
        if not (0 <= r < maze.row and 0 <= c < maze.col):
            collision = True
            break
        if maze.map[int(r)][int(c)].state == '#':
            collision = True
            break

        gr, gc = path_rc[-1]
        if math.hypot(r - gr, c - gc) < goal_thr:
            goal_reached = True
            final_step = step
            break

    return {
        'status': 'GOAL' if goal_reached else ('COLLISION' if collision else 'TIMEOUT'),
        'time_s': sim_time,
        'traversed': traversed,
        'steps': final_step + 1 if goal_reached else (step + 1),
        'max_cte': max_cte,
        'mean_cte': (mean_cte_sum / mean_cte_n) if mean_cte_n else 0.0,
    }


# --- Main comparison ---

def run_comparison(maze_file, config, output_dir):
    """Run full planner path comparison."""
    print(f"\n{'='*70}")
    print(f"PLANNER PATH COMPARISON: {os.path.basename(maze_file)}")
    print(f"{'='*70}")

    print("\n[1] Generating paths from all planners...")
    paths, maze = generate_all_paths(maze_file, config)

    if not paths:
        print("No paths generated. Exiting.")
        return

    print(f"\n[2] Saving paths to {output_dir}...")
    save_paths(paths, maze_file, output_dir)

    print("\n[3] Computing theoretical minimum time (bang-bang speed profile)...")
    results = []

    for planner_name, path_data in paths.items():
        waypoints_xy = path_data['waypoints_xy']

        bb_result = compute_bb_rrt_time(waypoints_xy, maze, config)
        print(f"  {planner_name:25s} -> theoretical_time={bb_result['total_time']:.3f}s "
              f"feasible={bb_result['feasible']}")

        results.append({
            'maze': os.path.basename(maze_file),
            'planner': planner_name,
            'tracker': 'theoretical',
            'planner_time_s': path_data['plan_time_s'],
            'path_length': path_data['path_length'],
            'n_waypoints': path_data['n_waypoints'],
            'n_turns': path_data['n_turns'],
            'status': 'FEASIBLE' if bb_result['feasible'] else 'INFEASIBLE',
            'time_s': bb_result['total_time'],
            'traversed': path_data['path_length'],
            'steps': 0,
            'max_cte': 0.0,
            'mean_cte': 0.0,
        })

    print("\n[4] Running controllers on each path...")

    for planner_name, path_data in paths.items():
        path_rc = path_data['waypoints_rc']
        if len(path_rc) < 2:
            print(f"  {planner_name:25s} + {'pure_pursuit':15s}... SKIP (path too short)")
            print(f"  {planner_name:25s} + {'stanley':15s}... SKIP (path too short)")
            continue

        for tracker in ['pure_pursuit', 'stanley']:
            print(f"  {planner_name:25s} + {tracker:15s}...", end=" ", flush=True)
            metrics = run_controller_on_path(path_rc, maze, config, tracker)
            result = {
                'maze': os.path.basename(maze_file),
                'planner': planner_name,
                'tracker': tracker,
                'planner_time_s': path_data['plan_time_s'],
                'path_length': path_data['path_length'],
                'n_waypoints': path_data['n_waypoints'],
                'n_turns': path_data['n_turns'],
                **metrics,
            }
            results.append(result)
            print(f"{metrics['status']:10s} time={metrics['time_s']:.3f}s "
                  f"maxCTE={metrics['max_cte']:.3f}")

    return results


def print_summary(results):
    """Print a summary table."""
    print(f"\n{'='*110}")
    print("SUMMARY")
    print(f"{'='*110}")

    theoretical = [r for r in results if r['tracker'] == 'theoretical']
    if theoretical:
        print(f"\n--- Theoretical Minimum Time (bang-bang speed profile, v_max=4.0, a_max=8.0) ---")
        print(f"{'Planner':25s} {'Length':>8s} {'Theor.Time':>12s} {'Feasible':>10s} {'PlanTime':>10s}")
        print("-" * 70)
        for r in sorted(theoretical, key=lambda x: x['time_s']):
            print(f"{r['planner']:25s} {r['path_length']:8.2f} {r['time_s']:12.3f} "
                  f"{r['status']:10s} {r['planner_time_s']:10.4f}")

    controller = [r for r in results if r['tracker'] != 'theoretical']
    if controller:
        print(f"\n--- Controller Results ---")
        print(f"{'Planner':25s} {'Controller':15s} {'Status':10s} {'Time(s)':>9s} "
              f"{'MaxCTE':>7s}")
        print("-" * 80)
        for r in controller:
            print(f"{r['planner']:25s} {r['tracker']:15s} {r['status']:10s} "
                  f"{r['time_s']:9.3f} {r['max_cte']:7.3f}")


def save_results(results, output_file):
    """Save results to CSV."""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', newline='') as f:
        if results:
            w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            w.writeheader()
            w.writerows(results)
    print(f"\nWrote {output_file}")


# --- Chart generation ---

def generate_charts(results, chart_dir):
    """Generate comparison charts."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(chart_dir, exist_ok=True)

    theoretical = [r for r in results if r['tracker'] == 'theoretical']
    controller = [r for r in results if r['tracker'] != 'theoretical']

    if theoretical:
        fig, ax = plt.subplots(figsize=(12, 6))
        planners = sorted(set(r['planner'] for r in theoretical))
        x = np.arange(len(planners))

        times = [next(r['time_s'] for r in theoretical if r['planner'] == p) for p in planners]
        feasible = [next(r['status'] == 'FEASIBLE' for r in theoretical if r['planner'] == p) for p in planners]
        colors = ['#2ca02c' if f else '#d62728' for f in feasible]

        bars = ax.bar(x, times, 0.6, color=colors)
        ax.set_xticks(x)
        ax.set_xticklabels(planners, rotation=45, ha='right')
        ax.set_ylabel('Theoretical minimum time (s)')
        ax.set_title('Theoretical minimum traversal time per planner\n(Green=feasible, Red=infeasible for bang-bang)')
        ax.grid(axis='y', alpha=0.3)

        for bar, t, f in zip(bars, times, feasible):
            if t < float('inf'):
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                       f'{t:.1f}s', ha='center', va='bottom', fontsize=9)

        fig.tight_layout()
        fig.savefig(os.path.join(chart_dir, 'theoretical_time.png'), dpi=120)
        plt.close(fig)
        print(f"Wrote {os.path.join(chart_dir, 'theoretical_time.png')}")

    if theoretical:
        fig, ax = plt.subplots(figsize=(12, 6))
        planners = sorted(set(r['planner'] for r in theoretical))
        x = np.arange(len(planners))

        lengths = [next(r['path_length'] for r in theoretical if r['planner'] == p) for p in planners]
        ax.bar(x, lengths, 0.6, color='#ff7f0e')
        ax.set_xticks(x)
        ax.set_xticklabels(planners, rotation=45, ha='right')
        ax.set_ylabel('Path length (cells)')
        ax.set_title('Geometric path length per planner')
        ax.grid(axis='y', alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(chart_dir, 'path_length.png'), dpi=120)
        plt.close(fig)
        print(f"Wrote {os.path.join(chart_dir, 'path_length.png')}")

    if theoretical:
        fig, ax = plt.subplots(figsize=(12, 6))
        planners = sorted(set(r['planner'] for r in theoretical))
        x = np.arange(len(planners))

        plan_times = [next(r['planner_time_s'] for r in theoretical if r['planner'] == p) for p in planners]
        ax.bar(x, plan_times, 0.6, color='#9467bd')
        ax.set_xticks(x)
        ax.set_xticklabels(planners, rotation=45, ha='right')
        ax.set_ylabel('Planning time (s)')
        ax.set_title('Planner computational cost')
        ax.grid(axis='y', alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(chart_dir, 'planning_time.png'), dpi=120)
        plt.close(fig)
        print(f"Wrote {os.path.join(chart_dir, 'planning_time.png')}")

    if controller:
        stanley_results = [r for r in controller if r['tracker'] == 'stanley' and r['status'] == 'GOAL']
        if stanley_results:
            fig, ax = plt.subplots(figsize=(12, 6))
            planners = [r['planner'] for r in stanley_results]
            x = np.arange(len(planners))
            times = [r['time_s'] for r in stanley_results]

            ax.bar(x, times, 0.6, color='#1f77b4')
            ax.set_xticks(x)
            ax.set_xticklabels(planners, rotation=45, ha='right')
            ax.set_ylabel('Traversal time (s)')
            ax.set_title('Stanley controller: time to goal (successful runs only)')
            ax.grid(axis='y', alpha=0.3)
            fig.tight_layout()
            fig.savefig(os.path.join(chart_dir, 'stanley_time.png'), dpi=120)
            plt.close(fig)
            print(f"Wrote {os.path.join(chart_dir, 'stanley_time.png')}")


# --- Entry point ---

def main():
    parser = argparse.ArgumentParser(
        description="Compare planner paths with different controllers"
    )
    parser.add_argument('--maze', nargs='+', default=DEFAULT_MAZES,
                        help='Maze file(s) to test')
    parser.add_argument('--out-dir', default=os.path.join(RESULTS_DIR, 'planner_paths'),
                        help='Directory to save generated paths')
    parser.add_argument('--csv', default=os.path.join(RESULTS_DIR, 'planner_path_comparison.csv'),
                        help='Output CSV file')
    parser.add_argument('--charts', action='store_true', default=True,
                        help='Generate comparison charts')
    args = parser.parse_args()

    with open('config.yaml') as f:
        config = yaml.safe_load(f)

    config.setdefault('micromouse', {})['initial_heading'] = 0.0

    all_results = []
    for maze_file in args.maze:
        if not os.path.exists(maze_file):
            print(f"Warning: {maze_file} not found, skipping")
            continue
        results = run_comparison(maze_file, config, args.out_dir)
        if results:
            all_results.extend(results)

    if all_results:
        print_summary(all_results)
        save_results(all_results, args.csv)

        if args.charts:
            generate_charts(all_results, CHART_DIR)

    print("\nDone!")


if __name__ == "__main__":
    main()
