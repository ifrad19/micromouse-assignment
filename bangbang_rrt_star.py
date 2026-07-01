"""
Bang-Bang RRT* for Micromouse

RRT* planner on 3D state (x, y, v) where v is scalar speed along the edge
heading. Tree edges are time-optimal bang-bang trajectories on the 1D speed
channel; 2D position is integrated along the constant parent->child heading.

3D-state specialisation of LaValle, Sakcak, LaValle (IROS 2023).
Cost is arrival time (seconds).
"""

import math
import random
import numpy as np

from bangbang_steering import (
    bang_bang_steer_1d,
    control_time_1d,
    integrate_control_1d,
    sample_trajectory_1d,
    time_epsilon,
)


# --- Node ---


class Node:
    """A tree node. State is (x, y, v). Edge from parent is a bang-bang
    control + a 2D trajectory for collision checking."""

    __slots__ = (
        "x", "y", "v", "parent", "children",
        "cost",           # arrival time from start
        "control",        # [[a, t_seg], ...] along the parent->self edge
        "trajectory_2d",  # np.ndarray (N, 4) [x, y, v, t]
    )

    def __init__(self, x, y, v, parent=None, cost=0.0,
                 control=None, trajectory_2d=None):
        self.x = x
        self.y = y
        self.v = v
        self.parent = parent
        self.children = []
        self.cost = cost
        self.control = control if control is not None else []
        self.trajectory_2d = trajectory_2d
        if parent is not None:
            parent.children.append(self)


# --- Collision-function helpers ---


def make_grid_collision_fn(grid, robot_radius=0.0):
    """Build a point-in-grid collision function.

    Args:
        grid: 2D numpy array, 1=wall, 0=free
        robot_radius: inflate walls by this many cells (Manhattan)
    Returns:
        collision_fn(x, y) -> bool  (True = in collision)
    """
    rows, cols = grid.shape
    if robot_radius > 0:
        from scipy.ndimage import distance_transform_edt
        free = (grid == 0).astype(float)
        dist_to_wall = distance_transform_edt(free)
        threshold = robot_radius
    else:
        dist_to_wall = None
        threshold = 0.0

    def collision_fn(x, y):
        col_f = float(x)
        row_f = float(y)
        col = int(math.floor(col_f))
        row = int(math.floor(row_f))
        if col < 0 or col >= cols or row < 0 or row >= rows:
            return True
        if grid[row, col] == 1:
            return True
        if dist_to_wall is not None:
            if col_f - col < 0.5 and col > 0 and grid[row, col - 1] == 1:
                d = (col_f - col) + 0.5
            elif col_f - col > 0.5 and col < cols - 1 and grid[row, col + 1] == 1:
                d = (col + 1.5) - col_f
            elif row_f - row < 0.5 and row > 0 and grid[row - 1, col] == 1:
                d = (row_f - row) + 0.5
            elif row_f - row > 0.5 and row < rows - 1 and grid[row + 1, col] == 1:
                d = (row + 1.5) - row_f
            else:
                d = dist_to_wall[row, col] - 0.5
            if d < threshold:
                return True
        return False

    return collision_fn


def line_collision_check(p1, p2, collision_fn, sample_step=0.2, grid=None):
    """Check whether the straight segment from p1 to p2 is collision-free.

    If *grid* is provided, uses DDA grid traversal to check every cell the
    line crosses — this catches diagonal lines slipping between wall cells.
    Also samples a few intermediate points and checks them with the
    collision function to enforce clearance from nearby walls.
    Falls back to point sampling when *grid* is None.
    """
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    length = math.hypot(dx, dy)
    if length == 0.0:
        return not collision_fn(x1, y1)

    if grid is not None:
        rows_g, cols_g = grid.shape
        cx1 = int(math.floor(x1))
        cy1 = int(math.floor(y1))
        cx2 = int(math.floor(x2))
        cy2 = int(math.floor(y2))

        step_x = 1 if dx >= 0 else -1
        step_y = 1 if dy >= 0 else -1

        if abs(dx) > 1e-12:
            t_max_x = ((cx1 + (1 if dx >= 0 else 0)) - x1) / dx
            t_delta_x = abs(1.0 / dx)
        else:
            t_max_x = float('inf')
            t_delta_x = float('inf')

        if abs(dy) > 1e-12:
            t_max_y = ((cy1 + (1 if dy >= 0 else 0)) - y1) / dy
            t_delta_y = abs(1.0 / dy)
        else:
            t_max_y = float('inf')
            t_delta_y = float('inf')

        cx, cy = cx1, cy1
        if 0 <= cy < rows_g and 0 <= cx < cols_g and grid[cy, cx] == 1:
            return False

        max_steps = int(abs(cx2 - cx1) + abs(cy2 - cy1)) + 3
        for _ in range(max_steps):
            if t_max_x < t_max_y:
                cx += step_x
                t_max_x += t_delta_x
            else:
                cy += step_y
                t_max_y += t_delta_y

            if cx < 0 or cx >= cols_g or cy < 0 or cy >= rows_g:
                return False
            if grid[cy, cx] == 1:
                return False

            if (step_x > 0 and cx >= cx2) or (step_x < 0 and cx <= cx2):
                break
            if (step_y > 0 and cy >= cy2) or (step_y < 0 and cy <= cy2):
                break

        n_extra = max(2, int(length / 0.5))
        for i in range(1, n_extra):
            t = i / n_extra
            sx = x1 + t * dx
            sy = y1 + t * dy
            if collision_fn(sx, sy):
                return False

        return True

    steps = max(1, int(length / sample_step) + 1)
    for i in range(steps + 1):
        t = i / steps
        x = x1 + t * dx
        y = y1 + t * dy
        if collision_fn(x, y):
            return False
    return True


# --- RRT* planner ---


class BangBangRRTStar:
    """RRT* on (x, y, v) state with bang-bang steering on the speed channel.

    Args:
        x_min, x_max, y_min, y_max: sampling bounds (continuous)
        v_max: maximum speed (cells/s)
        a_max: maximum acceleration magnitude
        collision_fn: callable (x, y) -> bool   (True = in collision)
        goal_fn:      callable (x, y) -> bool   (True = at goal)
        max_iter:     maximum RRT* iterations
        goal_bias:    probability of sampling the goal
        max_steer_dist: cap on edge length (cells)
        rewire_gamma: coefficient in the rewire radius formula
        rewire_k:     cap on the number of neighbours in X_near
        seed:         RNG seed
    """

    def __init__(self, x_min, x_max, y_min, y_max, v_max, a_max,
                 collision_fn, goal_fn,
                 max_iter=2000, goal_bias=0.1, max_steer_dist=4.0,
                 v_sample_low=None, v_sample_high=None,
                 rewire_gamma=2.0, rewire_k=30,
                 seed=0, verbose=False, traj_sample_step=0.2, grid=None):
        self.x_min, self.x_max = x_min, x_max
        self.y_min, self.y_max = y_min, y_max
        self.v_max = v_max
        self.a_max = a_max
        self.collision_fn = collision_fn
        self.goal_fn = goal_fn
        self.max_iter = max_iter
        self.goal_bias = goal_bias
        self.max_steer_dist = max_steer_dist
        self.v_sample_low = v_sample_low if v_sample_low is not None else 0.0
        self.v_sample_high = v_sample_high if v_sample_high is not None else v_max
        self.rewire_gamma = rewire_gamma
        self.rewire_k = rewire_k
        self.rng = random.Random(seed)
        self.verbose = verbose
        self.traj_sample_step = traj_sample_step
        self.grid = grid  # raw wall grid for DDA traversal

    # ------------------------------------------------------------------ I/O

    def plan(self, start_xy, goal_xy, start_v=0.0, goal_tolerance=0.5,
             goal_v=0.0, time_budget_s=None):
        """Plan a path from start_xy to within goal_tolerance of goal_xy.

        Returns dict with 'nodes', 'goal_node', 'best_cost', 'waypoints',
        'controls', 'edges'.  Or None if no path found.
        """
        import time as _time
        t_start = _time.time()

        sx, sy = start_xy
        gx, gy = goal_xy
        start_node = Node(sx, sy, start_v, cost=0.0)
        tree = [start_node]

        best_goal_node = None
        best_goal_cost = float("inf")

        goal_x_min = gx - goal_tolerance
        goal_x_max = gx + goal_tolerance
        goal_y_min = gy - goal_tolerance
        goal_y_max = gy + goal_tolerance

        for it in range(self.max_iter):
            if time_budget_s is not None and (_time.time() - t_start) > time_budget_s:
                if self.verbose:
                    print(f"  [bangbang_rrt_star] time budget exhausted at iter {it}")
                break

            if self.rng.random() < self.goal_bias:
                x_rand = self.rng.uniform(goal_x_min, goal_x_max)
                y_rand = self.rng.uniform(goal_y_min, goal_y_max)
                v_rand = goal_v
            else:
                x_rand = self.rng.uniform(self.x_min, self.x_max)
                y_rand = self.rng.uniform(self.y_min, self.y_max)
                v_rand = self.rng.uniform(self.v_sample_low, self.v_sample_high)

            x_near = self._nearest(tree, x_rand, y_rand, v_rand)

            dx = x_rand - x_near.x
            dy = y_rand - x_near.y
            dist = math.hypot(dx, dy)
            if dist < 1e-9:
                continue
            cos_h = dx / dist
            sin_h = dy / dist
            if dist > self.max_steer_dist:
                dist = self.max_steer_dist
            x_target = x_near.x + dist * cos_h
            y_target = x_near.y + dist * sin_h

            steer = bang_bang_steer_1d(
                x_init=0.0,
                v_init=x_near.v,
                x_goal=dist,
                v_goal=v_rand,
                a_max=self.a_max,
                samples_per_second=80.0,
            )
            if steer is None or not steer["feasible"]:
                continue

            traj_1d = steer["trajectory"]  # (N, 3) [x_dist, v, t]
            N = traj_1d.shape[0]
            traj_2d = np.empty((N, 4), dtype=float)
            traj_2d[:, 0] = x_near.x + traj_1d[:, 0] * cos_h
            traj_2d[:, 1] = x_near.y + traj_1d[:, 0] * sin_h
            traj_2d[:, 2] = traj_1d[:, 1]
            traj_2d[:, 3] = traj_1d[:, 2]

            if not self._trajectory_collision_free(traj_2d):
                continue

            x_new = Node(
                x=x_target, y=y_target, v=v_rand,
                parent=x_near, cost=x_near.cost + steer["total_time"],
                control=steer["control"],
                trajectory_2d=traj_2d,
            )
            tree.append(x_new)

            X_near = self._near(tree, x_new, self._rewire_radius(len(tree)))
            best_parent = x_near
            best_cost = x_new.cost
            best_control = steer["control"]
            best_traj = traj_2d
            for x_near_nb in X_near:
                if x_near_nb is x_near:
                    continue
                cand = self._try_connect(x_near_nb, x_new)
                if cand is None:
                    continue
                c_cost, c_control, c_traj = cand
                if c_cost + 1e-9 < best_cost:
                    best_parent = x_near_nb
                    best_cost = c_cost
                    best_control = c_control
                    best_traj = c_traj
            if best_parent is not x_near:
                x_near.children.remove(x_new)
                x_new.parent = None
                x_new.cost = best_cost
                x_new.control = best_control
                x_new.trajectory_2d = best_traj
                best_parent.children.append(x_new)
                x_new.parent = best_parent

            for x_near_nb in X_near:
                if x_near_nb is x_new:
                    continue
                cand = self._try_connect(x_new, x_near_nb)
                if cand is None:
                    continue
                c_cost, c_control, c_traj = cand
                if c_cost + 1e-9 < x_near_nb.cost:
                    if x_near_nb.parent is not None:
                        x_near_nb.parent.children.remove(x_near_nb)
                    x_near_nb.parent = x_new
                    x_near_nb.cost = c_cost
                    x_near_nb.control = c_control
                    x_near_nb.trajectory_2d = c_traj
                    x_new.children.append(x_near_nb)
                    self._update_descendant_costs(x_near_nb)

            if (abs(x_new.x - gx) <= goal_tolerance
                    and abs(x_new.y - gy) <= goal_tolerance):
                if x_new.cost < best_goal_cost:
                    best_goal_node = x_new
                    best_goal_cost = x_new.cost
                    if self.verbose:
                        print(f"  [bangbang_rrt_star] iter {it}: "
                              f"new best goal cost = {best_goal_cost:.3f} s "
                              f"({len(tree)} nodes)")

        if best_goal_node is None:
            if self.verbose:
                print(f"  [bangbang_rrt_star] FAILED: no goal-reaching node in {len(tree)} nodes")
            return None

        waypoints, controls, edges = self._reconstruct(best_goal_node)
        return {
            "nodes": tree,
            "goal_node": best_goal_node,
            "best_cost": best_goal_cost,
            "waypoints": waypoints,
            "controls": controls,
            "edges": edges,
        }

    # ------------------------------------------------------ internal helpers

    def _rewire_radius(self, n):
        if n <= 1:
            return self.max_steer_dist
        d = 3.0
        log_n = math.log(n + 1.0)
        r = self.rewire_gamma * (log_n / n) ** (1.0 / d)
        return max(self.max_steer_dist, r)

    def _bb_time_metric(self, n, x_q, y_q, v_q):
        """Estimate bang-bang travel time from node n to query (x_q, y_q, v_q).

        O(1) — better than Euclidean for nearest-neighbor because two points
        close in space but with very different velocities are far apart in time.
        """
        d = math.hypot(n.x - x_q, n.y - y_q)
        if d < 1e-9:
            return abs(n.v - v_q) / self.a_max
        avg_v = max((n.v + v_q) / 2.0, 0.1)
        t_travel = d / avg_v
        t_accel = abs(n.v - v_q) / self.a_max
        return t_travel + t_accel

    def _nearest(self, tree, x_q, y_q, v_q=0.0):
        best = None
        best_d = float("inf")
        for n in tree:
            d = self._bb_time_metric(n, x_q, y_q, v_q)
            if d < best_d:
                best_d = d
                best = n
        return best

    def _near(self, tree, q_node, radius):
        """Find nodes within a time-based radius of q_node."""
        time_radius = radius / max(0.1, q_node.v) + 2.0 * radius / self.a_max
        out = []
        for n in tree:
            t = self._bb_time_metric(n, q_node.x, q_node.y, q_node.v)
            if t <= time_radius:
                out.append((t, n))
        out.sort(key=lambda pair: pair[0])
        return [n for _, n in out[: self.rewire_k]]

    def _trajectory_collision_free(self, traj_2d):
        """Check each consecutive segment of a (N,4) trajectory for collisions."""
        prev = (traj_2d[0, 0], traj_2d[0, 1])
        if self.collision_fn(prev[0], prev[1]):
            return False
        for i in range(1, traj_2d.shape[0]):
            cur = (traj_2d[i, 0], traj_2d[i, 1])
            if not line_collision_check(prev, cur, self.collision_fn,
                                        sample_step=self.traj_sample_step,
                                        grid=self.grid):
                return False
            prev = cur
        return True

    def _try_connect(self, parent_node, child_node):
        """Try a 1D bang-bang edge from parent_node to child_node.
        Returns (new_cost, control, traj_2d) or None."""
        dx = child_node.x - parent_node.x
        dy = child_node.y - parent_node.y
        dist = math.hypot(dx, dy)
        if dist < 1e-9:
            return None
        cos_h = dx / dist
        sin_h = dy / dist

        steer = bang_bang_steer_1d(
            x_init=0.0,
            v_init=parent_node.v,
            x_goal=dist,
            v_goal=child_node.v,
            a_max=self.a_max,
            samples_per_second=80.0,
        )
        if steer is None or not steer["feasible"]:
            return None

        traj_1d = steer["trajectory"]
        N = traj_1d.shape[0]
        traj_2d = np.empty((N, 4), dtype=float)
        traj_2d[:, 0] = parent_node.x + traj_1d[:, 0] * cos_h
        traj_2d[:, 1] = parent_node.y + traj_1d[:, 0] * sin_h
        traj_2d[:, 2] = traj_1d[:, 1]
        traj_2d[:, 3] = traj_1d[:, 2]

        if not self._trajectory_collision_free(traj_2d):
            return None

        new_cost = parent_node.cost + steer["total_time"]
        return new_cost, steer["control"], traj_2d

    def _update_descendant_costs(self, node):
        for child in node.children:
            edge = self._try_connect(node, child)
            if edge is None:
                continue
            child.cost = edge[0]
            child.control = edge[1]
            child.trajectory_2d = edge[2]
            self._update_descendant_costs(child)

    def _reconstruct(self, goal_node):
        nodes = []
        n = goal_node
        while n is not None:
            nodes.append(n)
            n = n.parent
        nodes.reverse()
        waypoints = [(nn.x, nn.y) for nn in nodes]
        controls = [nn.control for nn in nodes[1:]]
        edges = list(zip(nodes[:-1], nodes[1:]))
        return waypoints, controls, edges


# --- Smoothing variant ---


def smooth_path_with_bang_bang(waypoints, a_max, v_max, collision_fn,
                                traj_sample_step=0.05, verbose=False,
                                grid=None):
    """Apply bang-bang speed profile between consecutive waypoints.

    The mouse comes to rest (v=0) at each waypoint (brake-turn-go).

    Args:
        waypoints: list of (x, y)
        a_max, v_max: physical limits
        collision_fn: callable (x, y) -> bool
        traj_sample_step: collision check step along each edge

    Returns dict with 'waypoints', 'controls', 'trajectory', 'total_time',
    'feasible'.  Feasible is False if any segment collides.
    """
    if len(waypoints) < 2:
        return {
            "waypoints": waypoints,
            "controls": [],
            "trajectory": np.zeros((0, 4)),
            "total_time": 0.0,
            "feasible": True,
        }

    controls = []
    segments = []
    cumulative_t = 0.0
    feasible = True
    for i in range(len(waypoints) - 1):
        x0, y0 = waypoints[i]
        x1, y1 = waypoints[i + 1]
        dx = x1 - x0
        dy = y1 - y0
        dist = math.hypot(dx, dy)
        if dist < 1e-9:
            controls.append([])
            continue
        v_init = 0.0
        v_goal = 0.0
        steer = bang_bang_steer_1d(
            x_init=0.0, v_init=v_init,
            x_goal=dist, v_goal=v_goal,
            a_max=a_max, samples_per_second=80.0,
        )
        if steer is None:
            feasible = False
            if verbose:
                print(f"  [smooth] infeasible segment {i}->{i+1}: dist={dist:.3f}")
            break
        controls.append(steer["control"])
        segments.append((i, (x0, y0), (x1, y1), dist, steer))
        cumulative_t += steer["total_time"]

    if not feasible or not segments:
        return {
            "waypoints": waypoints,
            "controls": controls,
            "trajectory": np.zeros((0, 4)),
            "total_time": 0.0,
            "feasible": False,
        }

    for i, (x0, y0), (x1, y1), dist, steer in segments:
        if not line_collision_check((x0, y0), (x1, y1), collision_fn,
                                    sample_step=traj_sample_step, grid=grid):
            feasible = False
            if verbose:
                print(f"  [smooth] segment {i}->{i+1} collides")
            break

    if not feasible:
        return {
            "waypoints": waypoints,
            "controls": controls,
            "trajectory": np.zeros((0, 4)),
            "total_time": 0.0,
            "feasible": False,
        }

    pieces = []
    t_offset = 0.0
    for i, (x0, y0), (x1, y1), dist, steer in segments:
        traj_1d = steer["trajectory"]
        cos_h = (x1 - x0) / dist
        sin_h = (y1 - y0) / dist
        piece = np.empty((traj_1d.shape[0], 4), dtype=float)
        piece[:, 0] = x0 + traj_1d[:, 0] * cos_h
        piece[:, 1] = y0 + traj_1d[:, 0] * sin_h
        piece[:, 2] = traj_1d[:, 1]
        piece[:, 3] = traj_1d[:, 2] + t_offset
        if i < len(segments) - 1:
            piece = piece[:-1]
        pieces.append(piece)
        t_offset += steer["total_time"]

    trajectory = np.concatenate(pieces, axis=0)
    return {
        "waypoints": waypoints,
        "controls": controls,
        "trajectory": trajectory,
        "total_time": t_offset,
        "feasible": True,
    }


# --- Strategic interval-selection smoothing ---


def smooth_path_strategic(waypoints, a_max, v_max, collision_fn,
                          max_iters=200, verbose=False, grid=None):
    """Strategic interval-selection smoothing for bang-bang paths.

    Enumerates all possible waypoint shortcuts, ranks them by estimated
    time savings, and greedily applies the most promising ones first.

    Returns dict with same keys as smooth_path_with_bang_bang.
    """
    if len(waypoints) < 3:
        return smooth_path_with_bang_bang(waypoints, a_max, v_max,
                                          collision_fn, grid=grid)

    path = [tuple(w) for w in waypoints]

    def _seg_time(p1, p2):
        d = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
        if d < 1e-9:
            return 0.0
        s = bang_bang_steer_1d(0.0, 0.0, d, 0.0, a_max, samples_per_second=40.0)
        return s["total_time"] if s and s["feasible"] else float("inf")

    def _total_time(path):
        return sum(_seg_time(path[i], path[i + 1])
                   for i in range(len(path) - 1))

    def _shortcut_collision_free(p1, p2):
        return line_collision_check(p1, p2, collision_fn,
                                   sample_step=0.05, grid=grid)

    n = len(path)
    savings = {}
    for i in range(n - 2):
        for j in range(i + 2, n):
            current_t = sum(_seg_time(path[k], path[k + 1])
                           for k in range(i, j))
            direct_d = math.hypot(path[j][0] - path[i][0],
                                  path[j][1] - path[i][1])
            direct_s = bang_bang_steer_1d(0.0, 0.0, direct_d, 0.0, a_max,
                                          samples_per_second=40.0)
            if direct_s is None or not direct_s["feasible"]:
                savings[(i, j)] = -1.0
            else:
                savings[(i, j)] = current_t - direct_s["total_time"]

    applied = 0
    for it in range(max_iters):
        best_key = None
        best_saving = 0.01
        for (i, j), saving in savings.items():
            if saving > best_saving:
                best_saving = saving
                best_key = (i, j)

        if best_key is None:
            break

        i, j = best_key

        if i >= len(path) - 1 or j >= len(path):
            del savings[best_key]
            continue

        if not _shortcut_collision_free(path[i], path[j]):
            savings[best_key] = -1.0
            continue

        new_path = path[:i + 1] + [path[j]] + path[j + 1:]
        path = new_path
        applied += 1

        if verbose:
            print(f"  [strategic] iter {it}: shortcut {i}->{j} "
                  f"(saved {best_saving:.3f}s), path now {len(path)} waypoints")

        to_delete = []
        for (si, sj) in list(savings.keys()):
            if si >= i or sj >= i:
                to_delete.append((si, sj))
        for k in to_delete:
            del savings[k]

        for si in range(max(0, i - 1), len(path) - 1):
            for sj in range(si + 2, len(path)):
                if (si, sj) not in savings:
                    current_t = sum(_seg_time(path[kk], path[kk + 1])
                                   for kk in range(si, sj))
                    direct_d = math.hypot(path[sj][0] - path[si][0],
                                          path[sj][1] - path[si][1])
                    direct_s = bang_bang_steer_1d(0.0, 0.0, direct_d, 0.0,
                                                  a_max, samples_per_second=40.0)
                    if direct_s is None or not direct_s["feasible"]:
                        savings[(si, sj)] = -1.0
                    else:
                        savings[(si, sj)] = current_t - direct_s["total_time"]

    controls = []
    segments = []
    cumulative_t = 0.0
    feasible = True
    for i in range(len(path) - 1):
        x0, y0 = path[i]
        x1, y1 = path[i + 1]
        dx = x1 - x0
        dy = y1 - y0
        dist = math.hypot(dx, dy)
        if dist < 1e-9:
            controls.append([])
            continue
        steer = bang_bang_steer_1d(0.0, 0.0, dist, 0.0, a_max,
                                   samples_per_second=80.0)
        if steer is None:
            feasible = False
            break
        controls.append(steer["control"])
        segments.append((i, (x0, y0), (x1, y1), dist, steer))
        cumulative_t += steer["total_time"]

    if not feasible or not segments:
        return {
            "waypoints": path,
            "controls": controls,
            "trajectory": np.zeros((0, 4)),
            "total_time": 0.0,
            "feasible": False,
        }

    for i, (x0, y0), (x1, y1), dist, steer in segments:
        if not line_collision_check((x0, y0), (x1, y1), collision_fn,
                                    sample_step=0.05, grid=grid):
            feasible = False
            break

    if not feasible:
        return {
            "waypoints": path,
            "controls": controls,
            "trajectory": np.zeros((0, 4)),
            "total_time": 0.0,
            "feasible": False,
        }

    pieces = []
    t_offset = 0.0
    for i, (x0, y0), (x1, y1), dist, steer in segments:
        traj_1d = steer["trajectory"]
        cos_h = (x1 - x0) / dist
        sin_h = (y1 - y0) / dist
        piece = np.empty((traj_1d.shape[0], 4), dtype=float)
        piece[:, 0] = x0 + traj_1d[:, 0] * cos_h
        piece[:, 1] = y0 + traj_1d[:, 0] * sin_h
        piece[:, 2] = traj_1d[:, 1]
        piece[:, 3] = traj_1d[:, 2] + t_offset
        if i < len(segments) - 1:
            piece = piece[:-1]
        pieces.append(piece)
        t_offset += steer["total_time"]

    trajectory = np.concatenate(pieces, axis=0)
    if verbose:
        print(f"  [strategic] done: {applied} shortcuts, "
              f"{len(path)} waypoints, {t_offset:.3f}s total")
    return {
        "waypoints": path,
        "controls": controls,
        "trajectory": trajectory,
        "total_time": t_offset,
        "feasible": True,
    }


# --- Self-test ---


def _selftest():
    """Small maze smoke test: verify zero collisions and goal reached."""
    import os
    import time as _time
    from micromouse import load_maze, Map

    small_csv = "/tmp/small_maze.csv"
    if not os.path.exists(small_csv):
        rows = [
            "1,1,1,1,1,1,1,1,1,1,1",
            "1,2,0,0,0,0,0,0,0,0,1",
            "1,0,1,1,1,1,1,1,1,0,1",
            "1,0,1,0,0,0,0,0,1,0,1",
            "1,0,1,0,1,1,1,0,1,0,1",
            "1,0,1,0,1,4,1,0,1,0,1",
            "1,0,1,0,1,1,1,0,1,0,1",
            "1,0,1,0,0,0,0,0,1,0,1",
            "1,0,1,1,1,1,1,1,1,0,1",
            "1,0,0,0,0,0,0,0,0,0,1",
            "1,1,1,1,1,1,1,1,1,1,1",
        ]
        with open(small_csv, "w") as f:
            f.write("\n".join(rows) + "\n")

    maze_array = load_maze(small_csv)
    m = Map(maze_array)
    grid = m.get_grid_representation()
    grid_walls = (grid == 1).astype(int)
    sy, sx = m.start
    gy, gx = m.goal
    n_rows, n_cols = grid.shape

    collision_fn = make_grid_collision_fn(grid_walls, robot_radius=0.0)
    goal_fn = lambda x, y: (abs(x - gx) <= 0.5 and abs(y - gy) <= 0.5)

    print("=" * 60)
    print("BangBangRRTStar selftest (small numeric maze)")
    print("=" * 60)
    print(f"  grid shape: {grid.shape}, start=({sx:.1f},{sy:.1f}), "
          f"goal=({gx:.1f},{gy:.1f})")

    planner = BangBangRRTStar(
        x_min=0.0, x_max=float(n_cols),
        y_min=0.0, y_max=float(n_rows),
        v_max=4.0, a_max=8.0,
        collision_fn=collision_fn, goal_fn=goal_fn,
        max_iter=3000, goal_bias=0.2, max_steer_dist=1.0,
        rewire_gamma=2.0, rewire_k=20, seed=42, verbose=True,
    )

    t0 = _time.time()
    result = planner.plan(
        start_xy=(sx, sy), goal_xy=(gx, gy),
        start_v=0.0, goal_tolerance=0.6, goal_v=0.0,
    )
    dt = _time.time() - t0
    if result is None:
        print("FAIL: no path found on small maze")
        return
    print(f"OK: plan() returned in {dt:.2f}s, {len(result['nodes'])} nodes, "
          f"best_cost={result['best_cost']:.3f}s, "
          f"{len(result['waypoints'])} waypoints")

    n_collisions = 0
    for (parent, child) in result["edges"]:
        if not line_collision_check((parent.x, parent.y),
                                    (child.x, child.y),
                                    collision_fn, sample_step=0.05):
            n_collisions += 1
    print(f"  collisions along final path: {n_collisions}")
    assert n_collisions == 0, "Path has collisions!"

    final = result["waypoints"][-1]
    d_to_goal = math.hypot(final[0] - gx, final[1] - gy)
    print(f"  final waypoint = ({final[0]:.2f},{final[1]:.2f}), "
          f"distance to goal = {d_to_goal:.3f}")
    assert d_to_goal < 0.6, "Path does not reach goal"
    print("PASS (small maze)")

    maze_path = "mazefiles/classic/alljapan-045-2024-exp-fin.txt"
    if not os.path.exists(maze_path):
        print("Skipping AAMC smoke test (file not found).")
        return
    print()
    print("AAMC smoke test SKIPPED (see benchmarks for full run).")


if __name__ == "__main__":
    _selftest()
