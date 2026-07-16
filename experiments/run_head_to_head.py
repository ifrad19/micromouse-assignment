"""
Head-to-head planner comparison on a single maze.

Runs A*, Theta*, RRT, RRT-Smooth, and BB-RRT* on the same maze under identical
conditions. Reports planning time, path length, number of waypoints/turns,
and theoretical minimum traversal time (bang-bang speed profile per segment).

Saves a path JSON per planner for visualisation with:
    python3 simulation.py --load-path experiments/results/planner_paths/<file>.json

Usage:
    python experiments/run_head_to_head.py [--maze bench_mazes/AAMC15Maze.csv]
"""

import argparse
import json
import math
import os
import sys
import time

import yaml
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from micromouse import load_maze, Map
from experiments.planner_wrappers import (
    PLANNERS, run_one, path_length_xy, count_turns,
    center_path_away_from_walls, validate_planned_path_cspace,
)
from bangbang_rrt_star import smooth_path_with_bang_bang, make_grid_collision_fn, bang_bang_steer_1d


# --- Planners to compare ---

TARGET_PLANNERS = ['astar', 'theta_star', 'rrt', 'rrt_grid_smooth', 'bb_rrt_star', 'bb_rrt_star_theta']

NICE_NAMES = {
    'astar':           'A*',
    'theta_star':      'Theta*',
    'rrt':             'RRT',
    'rrt_grid_smooth': 'RRT-Smooth',
    'bb_rrt_star':     'BB-RRT*',
    'bb_rrt_star_theta': 'BB-RRT*_Theta',
}


# --- Theoretical minimum time ---


def compute_theoretical_time(waypoints_xy, maze, config):
    """Compute theoretical minimum traversal time for a (col, row) path
    using bang-bang speed profile, WITHOUT collision checking.

    The question is: "how fast could any controller drive this geometric path?"
    """
    if len(waypoints_xy) < 2:
        return float('inf'), False

    bb_cfg = config.get('bb_rrt_star', {})
    v_max = bb_cfg.get('v_max', 4.0)
    a_max = bb_cfg.get('a_max', 8.0)

    total_time = 0.0
    for i in range(len(waypoints_xy) - 1):
        x0, y0 = waypoints_xy[i]
        x1, y1 = waypoints_xy[i + 1]
        dist = math.hypot(x1 - x0, y1 - y0)
        if dist < 1e-9:
            continue
        steer = bang_bang_steer_1d(
            x_init=0.0, v_init=0.0,
            x_goal=dist, v_goal=0.0,
            a_max=a_max, samples_per_second=80.0,
        )
        if steer is None:
            return float('inf'), False
        total_time += steer["total_time"]

    return total_time, True


# --- Path JSON saving ---


def save_path_json(waypoints_xy, planner_name, maze_file, plan_time_s,
                   path_length, output_dir):
    """Save a path as JSON compatible with ``simulation.py --load-path``."""
    os.makedirs(output_dir, exist_ok=True)
    maze_name = os.path.splitext(os.path.basename(maze_file))[0]
    filename = f"{maze_name}_{planner_name}_path.json"
    filepath = os.path.join(output_dir, filename)

    waypoints_rc = [(wp[1], wp[0]) for wp in waypoints_xy]  # (col,row) -> (row,col)

    with open(filepath, 'w') as f:
        json.dump({
            'planner': planner_name,
            'maze': maze_file,
            'waypoints_rc': waypoints_rc,
            'waypoints_xy': list(waypoints_xy),
            'plan_time_s': plan_time_s,
            'path_length': path_length,
            'n_waypoints': len(waypoints_xy),
        }, f, indent=2)
    return filepath


# --- Chart generation ---


def generate_chart(results, chart_path):
    """Bar charts for planning time, path length, and theoretical time."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(chart_path), exist_ok=True)

    names = [NICE_NAMES[r['planner']] for r in results]
    x = np.arange(len(names))

    planning_times = [r['plan_time_s'] for r in results]
    path_lengths = [r['path_length_cells'] for r in results]
    theo_times = [r['theo_time'] for r in results]
    feasible = [r['feasible'] for r in results]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    ax = axes[0]
    ax.bar(x, planning_times, 0.6, color='#9467bd')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha='right')
    ax.set_ylabel('Planning time (s)')
    ax.set_title('Computational cost')
    ax.grid(axis='y', alpha=0.3)

    ax = axes[1]
    ax.bar(x, path_lengths, 0.6, color='#ff7f0e')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha='right')
    ax.set_ylabel('Path length (cells)')
    ax.set_title('Geometric path length')
    ax.grid(axis='y', alpha=0.3)

    ax = axes[2]
    colors = ['#2ca02c' if f else '#d62728' for f in feasible]
    ax.bar(x, theo_times, 0.6, color=colors)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha='right')
    ax.set_ylabel('Theoretical time (s)')
    ax.set_title('Min traversal time (bang-bang)')
    ax.grid(axis='y', alpha=0.3)
    for i, (t, f) in enumerate(zip(theo_times, feasible)):
        if t < float('inf'):
            ax.text(i, t + 0.3, f'{t:.1f}s', ha='center', va='bottom', fontsize=9)

    fig.tight_layout()
    fig.savefig(chart_path, dpi=120)
    plt.close(fig)
    print(f"Chart saved: {chart_path}")


# --- Main ---


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--maze', default='bench_mazes/AAMC15Maze.csv',
                        help='Maze CSV to test (default: AAMC15Maze)')
    parser.add_argument('--out-dir',
                        default=os.path.join(os.path.dirname(__file__),
                                             'results', 'planner_paths'),
                        help='Directory for saved path JSONs')
    parser.add_argument('--chart',
                        default=os.path.join(os.path.dirname(__file__),
                                             'results', 'charts',
                                             'head_to_head.png'),
                        help='Output chart path')
    parser.add_argument('--quick', action='store_true',
                        help='Reduce BB-RRT* time budget for fast runs')
    args = parser.parse_args()

    config_path = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), 'config.yaml')
    with open(config_path) as f:
        config = yaml.safe_load(f)

    config.setdefault('rrt', {})['max_iter'] = 20000
    config['rrt'].setdefault('step_size', 0.5)
    config['rrt'].setdefault('goal_bias', 0.15)

    if args.quick:
        config.setdefault('bb_rrt_star', {})['time_budget_s'] = 10

    maze_array = load_maze(args.maze)
    maze = Map(maze_array)
    maze_name = os.path.splitext(os.path.basename(args.maze))[0]
    start_xy = (maze.start[1], maze.start[0])
    goal_xy = (maze.goal[1], maze.goal[0])

    grid = maze.get_grid_representation()
    grid_walls = (grid == 1).astype(int)
    robot_radius = config.get('bb_rrt_star', {}).get('robot_radius', 0.28)

    print(f"{'='*70}")
    print(f"HEAD-TO-HEAD PLANNER COMPARISON")
    print(f"Maze: {maze_name} ({maze.row}x{maze.col})")
    print(f"Start: ({maze.start[0]},{maze.start[1]})  "
          f"Goal: ({maze.goal[0]},{maze.goal[1]})")
    print(f"{'='*70}")

    results = []

    for planner_key in TARGET_PLANNERS:
        nice = NICE_NAMES[planner_key]
        print(f"\n[{nice}] Running {planner_key}...", end=" ", flush=True)

        try:
            row = run_one(planner_key, maze, start_xy, goal_xy, config)
        except Exception as e:
            print(f"ERROR: {e}")
            results.append({
                'planner': planner_key,
                'success': False,
                'plan_time_s': 0.0,
                'path_length_cells': float('inf'),
                'n_waypoints': 0,
                'n_turns': 0,
                'theo_time': float('inf'),
                'feasible': False,
            })
            continue

        if not row['success']:
            print("NO PATH FOUND")
            results.append({
                'planner': planner_key,
                'success': False,
                'plan_time_s': row['plan_time_s'],
                'path_length_cells': float('inf'),
                'n_waypoints': row['n_waypoints'],
                'n_turns': row['n_turns'],
                'theo_time': float('inf'),
                'feasible': False,
            })
            continue

        wrapper_fn = PLANNERS[planner_key]
        wrapper_out = wrapper_fn(maze, start_xy, goal_xy, config)
        waypoints_xy = wrapper_out['waypoints']

        theo_time, feasible = compute_theoretical_time(waypoints_xy, maze, config)

        cspace_result = validate_planned_path_cspace(
            waypoints_xy, grid_walls, robot_radius)

        path_len = row['path_length_cells']
        print(f"OK  path_len={path_len:.2f}  "
              f"plan_time={row['plan_time_s']:.3f}s  "
              f"wps={row['n_waypoints']}  turns={row['n_turns']}  "
              f"theo_time={theo_time:.3f}s  feasible={feasible}  "
              f"planned_coll={cspace_result['n_collisions']}")

        waypoints_centered = center_path_away_from_walls(waypoints_xy, grid_walls)
        json_path = save_path_json(
            waypoints_centered, planner_key, args.maze,
            row['plan_time_s'], path_len, args.out_dir)

        results.append({
            'planner': planner_key,
            'success': True,
            'plan_time_s': row['plan_time_s'],
            'path_length_cells': path_len,
            'n_waypoints': row['n_waypoints'],
            'n_turns': row['n_turns'],
            'theo_time': theo_time,
            'feasible': feasible,
            'json_path': json_path,
            'planned_collisions': cspace_result['n_collisions'],
            'planned_free': cspace_result['collision_free'],
        })

    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]

    print(f"\n{'='*110}")
    print("RESULTS SUMMARY")
    print(f"{'='*110}")
    print(f"{'Planner':<12} {'Path Len':>10} {'Plan Time':>11} {'Waypoints':>10} "
          f"{'Turns':>7} {'Theo Time':>11} {'Collisions':>11} {'Free':>6} {'Feasible':>9}")
    print("-" * 110)

    for r in sorted(successful, key=lambda x: x['theo_time']):
        print(f"{NICE_NAMES[r['planner']]:<12} "
              f"{r['path_length_cells']:10.2f} "
              f"{r['plan_time_s']:11.3f} "
              f"{r['n_waypoints']:10d} "
              f"{r['n_turns']:7d} "
              f"{r['theo_time']:11.3f} "
              f"{r.get('planned_collisions', 0):11d} "
              f"{'yes' if r.get('planned_free', True) else 'NO':>6} "
              f"{'yes' if r['feasible'] else 'NO':>9}")

    for r in failed:
        print(f"{NICE_NAMES[r['planner']]:<12} {'FAILED':>10}")

    print()

    feasible_results = [r for r in successful if r['feasible']]
    if feasible_results:
        ranked = sorted(feasible_results, key=lambda x: x['theo_time'])
        print("RANKING (by theoretical minimum traversal time):")
        for i, r in enumerate(ranked, 1):
            print(f"  {i}. {NICE_NAMES[r['planner']]:<10}  "
                  f"{r['theo_time']:.3f}s  "
                  f"(path length: {r['path_length_cells']:.2f} cells)")

    print(f"\n{'='*70}")
    print("VISUALISE IN PYGAME:")
    print(f"{'='*70}")
    for r in successful:
        print(f"  python3 simulation.py --load-path {r['json_path']}")
    print()

    if successful:
        generate_chart(successful, args.chart)

    return results


if __name__ == "__main__":
    main()
