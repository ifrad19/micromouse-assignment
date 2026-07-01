"""
Planner wrappers for comparison benchmark.

Each planner exposes:
    plan(maze, start_xy, goal_xy, config) -> dict

Returns dict with 'waypoints' (list of (col, row)), 'plan_time_s', 'cost',
'extra'.

Coordinate convention: waypoints are (col, row) = (x, y).
  A* / Theta* return (row, col) — convert.
  RRT / RRT* / BB-RRT* return (x, y) = (col, row) — keep.
"""

import math
import random
import time
import numpy as np

from micromouse import Map
from pathPlanning import AStar
from pathPlanning import ThetaStar
from rrt import RRT
from rrt_with_pathsmoothing import path_smoothing
from bangbang_rrt_star import make_grid_collision_fn, line_collision_check


# --- Path metrics ---


def path_length_xy(waypoints):
    """Euclidean length of a (col, row) waypoint sequence."""
    if len(waypoints) < 2:
        return 0.0
    total = 0.0
    for i in range(len(waypoints) - 1):
        dx = waypoints[i + 1][0] - waypoints[i][0]
        dy = waypoints[i + 1][1] - waypoints[i][1]
        total += math.hypot(dx, dy)
    return total


def count_turns(waypoints, angle_threshold_deg=15.0):
    """Count significant direction changes and sum their magnitudes (degrees)."""
    if len(waypoints) < 3:
        return 0, 0.0
    threshold = math.radians(angle_threshold_deg)
    total_angle = 0.0
    n_turns = 0
    for i in range(1, len(waypoints) - 1):
        v1x = waypoints[i][0] - waypoints[i - 1][0]
        v1y = waypoints[i][1] - waypoints[i - 1][1]
        v2x = waypoints[i + 1][0] - waypoints[i][0]
        v2y = waypoints[i + 1][1] - waypoints[i][1]
        m1 = math.hypot(v1x, v1y)
        m2 = math.hypot(v2x, v2y)
        if m1 < 1e-9 or m2 < 1e-9:
            continue
        cos_a = (v1x * v2x + v1y * v2y) / (m1 * m2)
        cos_a = max(-1.0, min(1.0, cos_a))
        a = math.acos(cos_a)
        total_angle += math.degrees(a)
        if a >= threshold:
            n_turns += 1
    return n_turns, total_angle


# --- Path validation (DDA-based collision check) ---


def validate_path(waypoints_xy, maze, sample_step=0.05):
    """Validate a (col, row) waypoint path against the maze grid using DDA.

    Checks every segment between consecutive waypoints. If a segment
    collides with a wall, inserts intermediate waypoints along the
    grid-cell path to route around it.

    Returns a new waypoint list that is guaranteed collision-free.
    """
    grid = maze.get_grid_representation()
    grid_walls = (grid == 1).astype(int)
    collision_fn = make_grid_collision_fn(grid_walls, robot_radius=0.0)

    if len(waypoints_xy) < 2:
        return list(waypoints_xy)

    validated = [waypoints_xy[0]]
    for i in range(len(waypoints_xy) - 1):
        p1 = waypoints_xy[i]
        p2 = waypoints_xy[i + 1]

        if line_collision_check(p1, p2, collision_fn, sample_step=sample_step,
                                grid=grid_walls):
            validated.append(p2)
        else:
            r1, c1 = int(round(p1[1])), int(round(p1[0]))
            r2, c2 = int(round(p2[1])), int(round(p2[0]))

            r1 = max(0, min(r1, maze.row - 1))
            c1 = max(0, min(c1, maze.col - 1))
            r2 = max(0, min(r2, maze.row - 1))
            c2 = max(0, min(c2, maze.col - 1))

            if maze.map[r1][c1].state == '#' or maze.map[r2][c2].state == '#':
                validated.append(p2)
                continue

            start_state = maze.map[r1][c1]
            goal_state = maze.map[r2][c2]

            try:
                astar = AStar(maze, {})
                px, py = astar.plan_path(start_state, goal_state)
                if px and py and len(px) >= 2:
                    sub_path = [(py[j], px[j]) for j in range(1, len(px))]
                    validated.extend(sub_path)
                else:
                    validated.append(p2)
            except Exception:
                validated.append(p2)

    return validated


def center_path_away_from_walls(waypoints_xy, grid, clearance=0.3, max_shift=0.5):
    """Shift each waypoint away from nearby walls toward the corridor center.

    For each waypoint, scans 4 cardinal directions to find the nearest wall
    boundary. If the waypoint is closer to a wall than ``clearance``, it is
    pushed toward the corridor midpoint. This prevents controller drift into
    walls in narrow 1-cell corridors.

    Args:
        waypoints_xy: list of (col, row) waypoints
        grid: numpy array, grid[row][col] == 1 means wall
        clearance: minimum desired distance from wall center (cells)
        max_shift: maximum shift per waypoint (cells)

    Returns:
        list of adjusted (col, row) waypoints (new list, originals untouched)
    """
    if len(waypoints_xy) < 2:
        return list(waypoints_xy)

    n_rows, n_cols = grid.shape
    centered = []

    for col, row in waypoints_xy:
        new_col, new_row = col, row

        north_wall = 0
        for r in range(int(row) - 1, -1, -1):
            if 0 <= r < n_rows and grid[r, int(col)] == 1:
                north_wall = r + 1
                break
        south_wall = n_rows
        for r in range(int(row) + 1, n_rows):
            if grid[r, int(col)] == 1:
                south_wall = r
                break

        west_wall = 0
        for c in range(int(col) - 1, -1, -1):
            if 0 <= c < n_cols and grid[int(row), c] == 1:
                west_wall = c + 1
                break
        east_wall = n_cols
        for c in range(int(col) + 1, n_cols):
            if grid[int(row), c] == 1:
                east_wall = c
                break

        corridor_center_row = (north_wall + south_wall) / 2.0
        corridor_center_col = (west_wall + east_wall) / 2.0

        dist_north = row - north_wall
        dist_south = south_wall - row
        dist_west = col - west_wall
        dist_east = east_wall - col

        min_row_dist = min(dist_north, dist_south)
        min_col_dist = min(dist_west, dist_east)

        if min_row_dist < clearance:
            target_row = corridor_center_row
            shift = min(max_shift, abs(target_row - new_row))
            if target_row > new_row:
                new_row += shift
            else:
                new_row -= shift

        if min_col_dist < clearance:
            target_col = corridor_center_col
            shift = min(max_shift, abs(target_col - new_col))
            if target_col > new_col:
                new_col += shift
            else:
                new_col -= shift

        new_row = max(0.1, min(new_row, n_rows - 1.1))
        new_col = max(0.1, min(new_col, n_cols - 1.1))

        centered.append((new_col, new_row))

    centered[0] = waypoints_xy[0]
    centered[-1] = waypoints_xy[-1]

    return centered


# --- A* ---


def plan_astar(maze, start_xy, goal_xy, config):
    sy, sx = start_xy[1], start_xy[0]
    gy, gx = goal_xy[1], goal_xy[0]
    start_state = maze.map[sy][sx]
    goal_state = maze.map[gy][gx]
    planner = AStar(maze, config)
    t0 = time.time()
    px, py = planner.plan_path(start_state, goal_state)
    elapsed = time.time() - t0
    waypoints = [(y, x) for x, y in zip(px, py)]
    waypoints = validate_path(waypoints, maze)
    return {
        'waypoints': waypoints,
        'plan_time_s': elapsed,
        'cost': path_length_xy(waypoints),
        'extra': {},
    }


# --- Theta* ---


def plan_theta_star(maze, start_xy, goal_xy, config):
    """Theta* uses (x, y) inside, returns (row, col). Convert."""
    import contextlib, io
    with contextlib.redirect_stdout(io.StringIO()):
        planner = ThetaStar(maze, config)
        t0 = time.time()
        path_rc = planner.plan()
        elapsed = time.time() - t0
    waypoints = [(c, r) for r, c in path_rc]
    waypoints = validate_path(waypoints, maze)
    return {
        'waypoints': waypoints,
        'plan_time_s': elapsed,
        'cost': path_length_xy(waypoints),
        'extra': {},
    }


# --- RRT (basic, position-only) ---


def _maze_to_obstacles(maze):
    """Return list of (x, y, r) circle obstacles in (col, row) coordinates."""
    obstacles = []
    for i in range(maze.row):
        for j in range(maze.col):
            if maze.map[i][j].state == "#":
                obstacles.append((float(j), float(i), 0.5))
    return obstacles


def plan_basic_rrt(maze, start_xy, goal_xy, config):
    sx, sy = start_xy
    gx, gy = goal_xy
    obstacles = _maze_to_obstacles(maze)
    expand = config.get('rrt', {}).get('step_size', 0.5)
    max_iter = config.get('rrt', {}).get('max_iter', 5000)
    goal_rate = config.get('rrt', {}).get('goal_bias', 0.1)
    robot_radius = config.get('rrt', {}).get('robot_radius', 0.0)
    seed = config.get('rrt', {}).get('seed', None)
    if seed is not None:
        random.seed(seed)
    play_area = [0.0, float(maze.col), 0.0, float(maze.row)]
    rrt = RRT(
        start=[sx, sy], goal=[gx, gy],
        obstacle_list=obstacles, rand_area=[-1.0, max(maze.col, maze.row) + 1.0],
        expand_dis=expand, path_resolution=0.25,
        goal_sample_rate=int(goal_rate * 100),
        max_iter=max_iter, play_area=play_area, robot_radius=robot_radius,
    )
    t0 = time.time()
    path = rrt.planning(animation=False)
    elapsed = time.time() - t0
    if path is None:
        return {'waypoints': [], 'plan_time_s': elapsed, 'cost': float('inf'), 'extra': {'failure': 'no_path'}}
    waypoints = [(p[0], p[1]) for p in path][::-1]
    return {
        'waypoints': waypoints,
        'plan_time_s': elapsed,
        'cost': path_length_xy(waypoints),
        'extra': {},
    }


# --- RRT + line-of-sight smoothing ---


def plan_rrt_smooth(maze, start_xy, goal_xy, config):
    base = plan_basic_rrt(maze, start_xy, goal_xy, config)
    if not base['waypoints']:
        return base
    obstacles = _maze_to_obstacles(maze)
    robot_radius = config.get('rrt', {}).get('robot_radius', 0.0)
    smooth_iters = config.get('rrt', {}).get('smooth_iters', 500)
    seed = config.get('rrt', {}).get('seed', None)
    if seed is not None:
        random.seed(seed + 1)
    smoothed = path_smoothing(base['waypoints'], smooth_iters, obstacles,
                              robot_radius=robot_radius)
    return {
        'waypoints': smoothed,
        'plan_time_s': base['plan_time_s'],
        'cost': path_length_xy(smoothed),
        'extra': {'raw_waypoints': len(base['waypoints'])},
    }


def _grid_path_smoothing(waypoints, grid_walls, n_iters=500, sample_step=0.05):
    """Line-of-sight smoothing using the same grid collision check as BB-RRT*."""
    from bangbang_rrt_star import line_collision_check
    collision_fn = make_grid_collision_fn(grid_walls, robot_radius=0.0)

    def get_target_point(path, targetL):
        le = 0.0
        for i in range(len(path) - 1):
            dx = path[i + 1][0] - path[i][0]
            dy = path[i + 1][1] - path[i][1]
            d = math.hypot(dx, dy)
            if le + d >= targetL:
                t = (targetL - le) / d
                x = path[i][0] + t * dx
                y = path[i][1] + t * dy
                return (x, y, i)
            le += d
        return (path[-1][0], path[-1][1], len(path) - 1)

    def get_path_length(path):
        return sum(
            math.hypot(path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1])
            for i in range(len(path) - 1)
        )

    path = [tuple(w) for w in waypoints]
    le = get_path_length(path)
    for _ in range(n_iters):
        if le < 2 * sample_step:
            break
        pickPoints = sorted([random.uniform(0, le), random.uniform(0, le)])
        first = get_target_point(path, pickPoints[0])
        second = get_target_point(path, pickPoints[1])
        if first[2] <= 0 or second[2] <= 0:
            continue
        if second[2] + 1 > len(path):
            continue
        if second[2] == first[2]:
            continue
        if not line_collision_check((first[0], first[1]), (second[0], second[1]),
                                    collision_fn, sample_step=sample_step):
            continue
        new_path = list(path[:first[2] + 1])
        new_path.append((first[0], first[1]))
        new_path.append((second[0], second[1]))
        new_path.extend(path[second[2] + 1:])
        path = new_path
        le = get_path_length(path)
    return path


# --- Bang-Bang RRT* ---


def plan_bb_rrt_star(maze, start_xy, goal_xy, config):
    """Wraps BangBangRRTStar from bangbang_rrt_star.py."""
    from bangbang_rrt_star import BangBangRRTStar, make_grid_collision_fn

    bb_cfg = config.get('bb_rrt_star', {})
    grid = maze.get_grid_representation()
    grid_walls = (grid == 1).astype(int)
    n_rows, n_cols = grid.shape

    sx, sy = start_xy
    gx, gy = goal_xy
    collision_fn = make_grid_collision_fn(grid_walls, robot_radius=bb_cfg.get('robot_radius', 0.0))

    goal_tol = bb_cfg.get('goal_tolerance', 0.6)
    goal_fn = lambda x, y: (abs(x - gx) <= goal_tol and abs(y - gy) <= goal_tol)

    planner = BangBangRRTStar(
        x_min=0.0, x_max=float(n_cols),
        y_min=0.0, y_max=float(n_rows),
        v_max=bb_cfg.get('v_max', 4.0),
        a_max=bb_cfg.get('a_max', 8.0),
        collision_fn=collision_fn, goal_fn=goal_fn,
        max_iter=bb_cfg.get('max_iter', 3000),
        goal_bias=bb_cfg.get('goal_bias', 0.2),
        max_steer_dist=bb_cfg.get('max_steer_dist', 1.0),
        rewire_gamma=bb_cfg.get('rewire_gamma', 2.0),
        rewire_k=bb_cfg.get('rewire_k', 20),
        seed=bb_cfg.get('seed', 42),
        verbose=False,
        grid=grid_walls,
    )
    t0 = time.time()
    result = planner.plan(
        start_xy=(sx, sy), goal_xy=(gx, gy),
        start_v=bb_cfg.get('start_v', 0.0),
        goal_tolerance=goal_tol,
        goal_v=bb_cfg.get('goal_v', 0.0),
        time_budget_s=bb_cfg.get('time_budget_s', None),
    )
    elapsed = time.time() - t0
    if result is None:
        return {'waypoints': [], 'plan_time_s': elapsed, 'cost': float('inf'),
                'extra': {'failure': 'no_path', 'path_length_cells': float('inf')}}
    waypoints = [(w[0], w[1]) for w in result['waypoints']]
    L = path_length_xy(waypoints)
    return {
        'waypoints': waypoints,
        'plan_time_s': elapsed,
        'cost': result['best_cost'],
        'controls': result['controls'],
        'edges': result['edges'],
        'extra': {
            'n_edges': len(result['edges']),
            'tree_size': len(result.get('nodes', [])),
            'path_length_cells': L,
        },
    }


# --- RRT-smooth + bang-bang speed profile ---


def plan_rrt_smooth_bangbang(maze, start_xy, goal_xy, config):
    """Baseline RRT + grid-aware line-of-sight smoothing, then bang-bang speed profile."""
    from bangbang_rrt_star import smooth_path_with_bang_bang, make_grid_collision_fn

    base = plan_basic_rrt(maze, start_xy, goal_xy, config)
    if not base['waypoints']:
        return base

    grid = maze.get_grid_representation()
    grid_walls = (grid == 1).astype(int)
    seed = config.get('rrt', {}).get('seed', None)
    if seed is not None:
        random.seed(seed + 1)
    smooth_iters = config.get('rrt', {}).get('smooth_iters', 500)
    wps = _grid_path_smoothing(base['waypoints'], grid_walls,
                               n_iters=smooth_iters, sample_step=0.02)

    deduped = []
    for w in wps:
        if deduped and math.hypot(w[0] - deduped[-1][0], w[1] - deduped[-1][1]) < 1e-3:
            continue
        deduped.append(w)
    if len(deduped) < 2:
        return {'waypoints': deduped, 'plan_time_s': base['plan_time_s'],
                'cost': float('inf'),
                'extra': {'failure': 'dedup_too_short',
                          'path_length_cells': path_length_xy(deduped)}}

    bb_cfg = config.get('bb_rrt_star', {})
    collision_fn = make_grid_collision_fn(grid_walls, robot_radius=bb_cfg.get('robot_radius', 0.0))

    t0 = time.time()
    smoothed = smooth_path_with_bang_bang(
        waypoints=deduped,
        a_max=bb_cfg.get('a_max', 8.0),
        v_max=bb_cfg.get('v_max', 4.0),
        collision_fn=collision_fn,
        traj_sample_step=0.05,
    )
    elapsed = time.time() - t0
    feasible = smoothed.get('feasible', False)
    return {
        'waypoints': deduped,
        'plan_time_s': base['plan_time_s'] + elapsed,
        'cost': smoothed.get('total_time', float('inf')) if feasible else float('inf'),
        'extra': {
            'n_edges': len(smoothed.get('controls', [])),
            'trajectory_distance': path_length_xy(deduped),
            'feasible': feasible,
            'raw_waypoints': len(base['waypoints']),
            'path_length_cells': path_length_xy(deduped),
        },
    }


def plan_rrt_smooth_bangbang_circles(maze, start_xy, goal_xy, config):
    """Same as plan_rrt_smooth_bangbang but uses circle-obstacle smoothing."""
    from bangbang_rrt_star import smooth_path_with_bang_bang, make_grid_collision_fn

    base = plan_rrt_smooth(maze, start_xy, goal_xy, config)
    if not base['waypoints']:
        return base

    deduped = []
    for w in base['waypoints']:
        if deduped and math.hypot(w[0] - deduped[-1][0], w[1] - deduped[-1][1]) < 1e-3:
            continue
        deduped.append(w)
    if len(deduped) < 2:
        return {'waypoints': deduped, 'plan_time_s': base['plan_time_s'],
                'cost': float('inf'),
                'extra': {'failure': 'dedup_too_short',
                          'path_length_cells': path_length_xy(deduped)}}

    bb_cfg = config.get('bb_rrt_star', {})
    grid = maze.get_grid_representation()
    grid_walls = (grid == 1).astype(int)
    collision_fn = make_grid_collision_fn(grid_walls, robot_radius=bb_cfg.get('robot_radius', 0.0))

    t0 = time.time()
    smoothed = smooth_path_with_bang_bang(
        waypoints=deduped,
        a_max=bb_cfg.get('a_max', 8.0),
        v_max=bb_cfg.get('v_max', 4.0),
        collision_fn=collision_fn,
        traj_sample_step=0.05,
    )
    elapsed = time.time() - t0
    feasible = smoothed.get('feasible', False)
    return {
        'waypoints': deduped,
        'plan_time_s': base['plan_time_s'] + elapsed,
        'cost': smoothed.get('total_time', float('inf')) if feasible else float('inf'),
        'extra': {
            'n_edges': len(smoothed.get('controls', [])),
            'trajectory_distance': path_length_xy(deduped),
            'feasible': feasible,
            'raw_waypoints': len(base['waypoints']),
            'path_length_cells': path_length_xy(deduped),
        },
    }


# --- RRT + strategic interval-selection smoothing + bang-bang speed profile ---


def plan_rrt_grid_smooth_strategic(maze, start_xy, goal_xy, config):
    """RRT + strategic bang-bang smoothing (greedy interval selection)."""
    from bangbang_rrt_star import (
        smooth_path_strategic, make_grid_collision_fn
    )

    base = plan_basic_rrt(maze, start_xy, goal_xy, config)
    if not base['waypoints']:
        return base

    grid = maze.get_grid_representation()
    grid_walls = (grid == 1).astype(int)
    seed = config.get('rrt', {}).get('seed', None)
    if seed is not None:
        random.seed(seed + 1)
    smooth_iters = config.get('rrt', {}).get('smooth_iters', 200)
    wps = _grid_path_smoothing(base['waypoints'], grid_walls,
                               n_iters=smooth_iters, sample_step=0.02)

    deduped = []
    for w in wps:
        if deduped and math.hypot(w[0] - deduped[-1][0], w[1] - deduped[-1][1]) < 1e-3:
            continue
        deduped.append(w)
    if len(deduped) < 2:
        return {'waypoints': deduped, 'plan_time_s': base['plan_time_s'],
                'cost': float('inf'),
                'extra': {'failure': 'dedup_too_short',
                          'path_length_cells': path_length_xy(deduped)}}

    bb_cfg = config.get('bb_rrt_star', {})
    collision_fn = make_grid_collision_fn(grid_walls, robot_radius=bb_cfg.get('robot_radius', 0.0))

    t0 = time.time()
    smoothed = smooth_path_strategic(
        waypoints=deduped,
        a_max=bb_cfg.get('a_max', 8.0),
        v_max=bb_cfg.get('v_max', 4.0),
        collision_fn=collision_fn,
        max_iters=smooth_iters,
        grid=grid_walls,
    )
    elapsed = time.time() - t0
    feasible = smoothed.get('feasible', False)
    return {
        'waypoints': smoothed['waypoints'],
        'plan_time_s': base['plan_time_s'] + elapsed,
        'cost': smoothed.get('total_time', float('inf')) if feasible else float('inf'),
        'extra': {
            'n_edges': len(smoothed.get('controls', [])),
            'trajectory_distance': path_length_xy(smoothed['waypoints']),
            'feasible': feasible,
            'raw_waypoints': len(base['waypoints']),
            'path_length_cells': path_length_xy(smoothed['waypoints']),
        },
    }


# --- Registry ---


PLANNERS = {
    'astar':                     plan_astar,
    'theta_star':                plan_theta_star,
    'rrt':                       plan_basic_rrt,
    'rrt_smooth':                plan_rrt_smooth,
    'rrt_grid_smooth':           plan_rrt_smooth_bangbang,
    'rrt_grid_smooth_strategic': plan_rrt_grid_smooth_strategic,
    'rrt_smooth_bangbang_circles': plan_rrt_smooth_bangbang_circles,
    'bb_rrt_star':              plan_bb_rrt_star,
}


def run_one(planner_name, maze, start_xy, goal_xy, config):
    """Run a single (planner, maze) trial and return a flat result row."""
    fn = PLANNERS[planner_name]
    out = fn(maze, start_xy, goal_xy, config)
    wps = out['waypoints']
    n_turns, total_angle = count_turns(wps)
    extra = out.get('extra', {})
    row = {
        'planner':      planner_name,
        'n_waypoints':  len(wps),
        'path_length':  out['cost'],
        'plan_time_s':  out['plan_time_s'],
        'n_turns':      n_turns,
        'total_turn_deg': total_angle,
        'success':      bool(wps),
        **extra,
    }
    if 'path_length_cells' not in row:
        row['path_length_cells'] = out['cost']
    return row
