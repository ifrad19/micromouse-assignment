"""
Headless benchmark harness: compare A*, Theta*, basic RRT, RRT+smoothing,
and bang-bang RRT* on a set of mazes.

Outputs:
    experiments/results/<maze_name>__<planner>__<timestamp>.json  (per-trial)
    experiments/results/summary_<timestamp>.csv                    (flat table)
    experiments/results/summary_<timestamp>.md                     (markdown report)

Usage:
    python experiments/run_comparison.py [--maze PATH] [--planners a,b,c]
                                          [--out DIR] [--quick]
"""

import argparse
import csv
import json
import math
import os
import sys
import time
import yaml
from datetime import datetime

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from micromouse import Map, load_maze  # noqa: E402
from experiments.planner_wrappers import PLANNERS, run_one  # noqa: E402


# --- Default config ---

DEFAULT_CONFIG = {
    'astar': {
        'debug': False,
        'heuristic_weight': 1.0,
        'turn_cost_enabled': False,
    },
    'wall_cost': {
        'enabled': False,
    },
    'theta_star': {
        'resolution': 1.0,
        'robot_radius': 0.0,
        'debug': False,
    },
    'rrt': {
        'max_iter': 5000,
        'step_size': 0.5,
        'goal_bias': 0.1,
        'robot_radius': 0.0,
        'smooth_iters': 500,
        'seed': 42,
    },
    'bb_rrt_star': {
        'v_max': 4.0,
        'a_max': 8.0,
        'max_iter': 3000,
        'goal_bias': 0.2,
        'max_steer_dist': 1.0,
        'rewire_gamma': 2.0,
        'rewire_k': 20,
        'seed': 42,
        'goal_tolerance': 0.6,
        'robot_radius': 0.0,
    },
}


# --- Default maze suite ---

DEFAULT_MAZE_DIR = os.path.join(REPO_ROOT, 'mazefiles')

DEFAULT_MAZES = [
    os.path.join('/tmp', 'small_maze.csv'),
]


# --- Helpers ---

def maze_summary(maze):
    return {
        'name':     getattr(maze, 'name', '?'),
        'rows':     maze.row,
        'cols':     maze.col,
        'start_rc': list(maze.start),
        'goal_rc':  list(maze.goal),
    }


def run_trial(maze, maze_name, planners, config, out_dir, verbose=True):
    sr, sc = maze.start
    gr, gc = maze.goal
    start_xy = (sc, sr)
    goal_xy = (gc, gr)
    results = []
    print(f"\n=== {maze_name} ({maze.row}x{maze.col})  "
          f"start=({sr},{sc}) goal=({gr},{gc}) ===")
    for pname in planners:
        print(f"  [{pname}] ...", end=' ', flush=True)
        try:
            row = run_one(pname, maze, start_xy, goal_xy, config)
        except Exception as e:
            print(f"FAILED: {e}")
            row = {
                'planner': pname, 'n_waypoints': 0, 'path_length': float('inf'),
                'plan_time_s': 0.0, 'n_turns': 0, 'total_turn_deg': 0.0,
                'success': False, 'failure': f'{type(e).__name__}: {e}',
            }
        else:
            tag = 'OK' if row['success'] else 'NO PATH'
            print(f"{tag:>8}  cost={row['path_length']:.3f}  "
                  f"wps={row['n_waypoints']:>3}  "
                  f"plan_t={row['plan_time_s']:.2f}s  "
                  f"turns={row['n_turns']}")
        row['maze'] = maze_name
        results.append(row)
        trial_path = os.path.join(out_dir, f"{maze_name}__{pname}.json")
        with open(trial_path, 'w') as f:
            json.dump({
                **row,
                'maze_summary': maze_summary(maze),
            }, f, indent=2, default=str)
    return results


def make_markdown_summary(all_results, out_path):
    """Group by maze, then planner; show length (cells), time (s), plan time, turns."""
    by_maze = {}
    for r in all_results:
        by_maze.setdefault(r['maze'], []).append(r)

    lines = ['# Planner Comparison Summary', '']
    lines.append(f'Generated: {datetime.now().isoformat()}')
    lines.append('')
    lines.append('`path_length_cells` = geometric length of planned waypoint path.')
    lines.append('`arrival_time_s` = planner cost (only meaningful for bang-bang RRT*, where '
                 'it is the time-optimal arrival time given a_max and v_max).')
    lines.append('')

    for maze_name, rows in by_maze.items():
        lines.append(f'## {maze_name}')
        lines.append('')
        rows_sorted = sorted(
            rows,
            key=lambda r: r.get('path_length_cells', float('inf'))
                          if r['success'] else float('inf'),
        )
        lines.append('| Planner | Status | Length (cells) | Arrival (s) | Waypoints | Plan time (s) | Turns | Turn angle (deg) |')
        lines.append('|---------|--------|----------------|-------------|-----------|---------------|-------|------------------|')
        for r in rows_sorted:
            status = '✓' if r['success'] else '✗'
            length = f"{r.get('path_length_cells', float('inf')):.3f}" if r['success'] else '—'
            arrival = f"{r['path_length']:.3f}" if r['success'] else '—'
            lines.append(
                f"| {r['planner']} | {status} | {length} | {arrival} | "
                f"{r['n_waypoints']} | {r['plan_time_s']:.3f} | "
                f"{r['n_turns']} | {r['total_turn_deg']:.1f} |"
            )
        lines.append('')
    with open(out_path, 'w') as f:
        f.write('\n'.join(lines))


# --- Main ---

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--maze', action='append', default=[],
                        help='Path to a maze file (.csv or .txt). May be given multiple times.')
    parser.add_argument('--planner', action='append', default=None,
                        help='Planner name(s). Default: all.')
    parser.add_argument('--out', default=os.path.join(os.path.dirname(__file__), 'results'),
                        help='Output directory for results.')
    parser.add_argument('--max-grid', type=int, default=64,
                        help='Skip mazes with grid larger than this (RRT scaling).')
    parser.add_argument('--quick', action='store_true',
                        help='Reduce iteration counts for fast smoke runs.')
    parser.add_argument('--meters-per-cell', type=float, default=None,
                        help='Override maze meters_per_cell for bang-bang conversion.')
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    maze_paths = list(args.maze) if args.maze else DEFAULT_MAZES
    if not maze_paths:
        parser.error('No mazes specified and no default mazes found.')

    planners = args.planner if args.planner else list(PLANNERS.keys())
    for p in planners:
        if p not in PLANNERS:
            parser.error(f'Unknown planner: {p}. Available: {list(PLANNERS.keys())}')

    config = yaml.safe_load(yaml.safe_dump(DEFAULT_CONFIG))
    if args.quick:
        config['bb_rrt_star']['max_iter'] = 500
        config['rrt']['max_iter'] = 1000
        config['rrt']['smooth_iters'] = 100

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    all_results = []
    for mp in maze_paths:
        if not os.path.exists(mp):
            print(f"[skip] {mp} not found")
            continue
        maze_array = load_maze(mp)
        maze = Map(maze_array)
        maze.name = os.path.splitext(os.path.basename(mp))[0]
        if maze.row > args.max_grid or maze.col > args.max_grid:
            print(f"[skip] {maze.name} too large ({maze.row}x{maze.col} > {args.max_grid})")
            continue
        rows = run_trial(maze, maze.name, planners, config, args.out)
        all_results.extend(rows)

    csv_path = os.path.join(args.out, f'summary_{timestamp}.csv')
    if all_results:
        keys = sorted({k for r in all_results for k in r.keys()})
        keys.remove('maze')
        keys = ['maze', 'planner'] + [k for k in keys if k not in ('maze', 'planner')]
        with open(csv_path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction='ignore')
            w.writeheader()
            for r in all_results:
                w.writerow(r)
        print(f'\nCSV summary -> {csv_path}')

    md_path = os.path.join(args.out, f'summary_{timestamp}.md')
    make_markdown_summary(all_results, md_path)
    print(f'Markdown report -> {md_path}')


if __name__ == '__main__':
    main()
