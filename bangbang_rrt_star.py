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
        free = (grid != 1).astype(float)
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
            fx = col_f - col
            fy = row_f - row
            c0 = max(0, min(col, cols - 2))
            r0 = max(0, min(row, rows - 2))
            d00 = dist_to_wall[r0,     c0]
            d01 = dist_to_wall[r0,     c0 + 1]
            d10 = dist_to_wall[r0 + 1, c0]
            d11 = dist_to_wall[r0 + 1, c0 + 1]
            d = (d00 * (1 - fx) * (1 - fy) +
                 d01 * fx * (1 - fy) +
                 d10 * (1 - fx) * fy +
                 d11 * fx * fy)
            if d < threshold:
                return True
        return False

    return collision_fn


def build_cspace_obstacle_map(grid, robot_radius):
    """Build an explicit configuration-space obstacle map by inflating every
    wall cell outward by robot_radius (Minkowski sum with a disc of radius R).

    For a circular robot this is exact and independent of heading.

    Args:
        grid: 2D numpy array, 1 = wall, 0 = free
        robot_radius: robot radius in cell units (float)

    Returns:
        cspace_blocked: 2D numpy boolean array, same shape as grid.
                        True  = center of robot NOT allowed here (C_obs)
                        False = free configuration for the robot center (C_free)
    """
    import numpy as _np
    from scipy.ndimage import distance_transform_edt as _dte

    grid = _np.asarray(grid)
    free = (grid != 1).astype(float)
    dist_to_wall = _dte(free)
    cspace_blocked = (grid == 1) | (dist_to_wall <= robot_radius)
    return cspace_blocked


def make_cspace_collision_fn(cspace_blocked):
    """Build a point-collision function from a precomputed C-space obstacle map.
    The robot is treated as a POINT because the map already encodes its radius.

    Returns collision_fn(x, y) -> bool  (True = in collision).
    """
    rows, cols = cspace_blocked.shape

    def collision_fn(x, y):
        col = int(math.floor(float(x)))
        row = int(math.floor(float(y)))
        if col < 0 or col >= cols or row < 0 or row >= rows:
            return True
        return bool(cspace_blocked[row, col])

    return collision_fn


def make_continuous_cspace_collision_fn(grid, robot_radius):
    """Continuous, sub-cell-accurate collision check for a circular robot.

    Returns collision_fn(x, y) -> True if the robot body (radius robot_radius,
    centred at continuous position (x, y)) overlaps any wall.

    Uses bilinear interpolation of the Euclidean distance transform so that
    fractional positions between cell centres are handled correctly.  This is
    the Minkowski / C-space test done in continuous space rather than by
    discretising the obstacle map (which rounds away any radius < 1 cell).
    """
    import numpy as _np
    from scipy.ndimage import distance_transform_edt as _dte

    grid = _np.asarray(grid)
    rows, cols = grid.shape
    free = (grid != 1).astype(float)
    dist = _dte(free)

    def collision_fn(x, y):
        if x < 0 or y < 0 or x >= cols or y >= rows:
            return True

        col = int(math.floor(float(x)))
        row = int(math.floor(float(y)))
        if grid[min(row, rows - 1), min(col, cols - 1)] == 1:
            return True

        c0 = max(0, min(col, cols - 2))
        r0 = max(0, min(row, rows - 2))
        fx = min(max(float(x) - c0, 0.0), 1.0)
        fy = min(max(float(y) - r0, 0.0), 1.0)

        d00 = dist[r0,     c0]
        d01 = dist[r0,     c0 + 1]
        d10 = dist[r0 + 1, c0]
        d11 = dist[r0 + 1, c0 + 1]

        d = (d00 * (1 - fx) * (1 - fy) +
             d01 * fx       * (1 - fy) +
             d10 * (1 - fx) * fy +
             d11 * fx       * fy)

        d_surface = d - 0.5
        return d_surface < robot_radius

    return collision_fn


def make_rect_collision_fn(grid, half_length, half_width):
    """Collision check for a RECTANGULAR robot with heading.

    Returns collision_fn(x, y, theta) -> True if ANY of the robot's four
    corners (or any point on its edges) lies inside a wall cell.

    Walls are cell value 1 only (0, 2, 3, 4 are free).

    The heading convention matches simulation.py:
      heading=0  -> robot faces +row (downward)
      heading=pi/2 -> robot faces +col (rightward)
    """
    import numpy as _np

    grid = _np.asarray(grid)
    rows, cols = grid.shape

    def collision_fn(x, y, theta):
        sin_t = math.sin(theta)
        cos_t = math.cos(theta)

        fwd_x, fwd_y = sin_t, cos_t
        rgt_x, rgt_y = cos_t, -sin_t

        hl, hw = half_length, half_width
        corners = [
            (x + hl * fwd_x + hw * rgt_x, y + hl * fwd_y + hw * rgt_y),
            (x + hl * fwd_x - hw * rgt_x, y + hl * fwd_y - hw * rgt_y),
            (x - hl * fwd_x + hw * rgt_x, y - hl * fwd_y + hw * rgt_y),
            (x - hl * fwd_x - hw * rgt_x, y - hl * fwd_y - hw * rgt_y),
        ]

        for cx, cy in corners:
            if cx < 0 or cy < 0 or cx >= cols or cy >= rows:
                return True
            if grid[int(math.floor(cy)), int(math.floor(cx))] == 1:
                return True

        step = 0.1
        for i in range(4):
            x1, y1 = corners[i]
            x2, y2 = corners[(i + 1) % 4]
            d = math.hypot(x2 - x1, y2 - y1)
            n = max(1, int(math.ceil(d / step)))
            for j in range(1, n):
                t = j / n
                sx = x1 + t * (x2 - x1)
                sy = y1 + t * (y2 - y1)
                if sx < 0 or sy < 0 or sx >= cols or sy >= rows:
                    return True
                if grid[int(math.floor(sy)), int(math.floor(sx))] == 1:
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
            "1,0,0,0,1,1,1,1,1,0,1",
            "1,0,1,0,0,0,0,0,1,0,1",
            "1,0,1,1,1,1,1,0,1,0,1",
            "1,0,1,0,1,4,0,0,1,0,1",
            "1,0,1,1,1,1,1,0,1,0,1",
            "1,0,1,0,0,0,0,0,1,0,1",
            "1,0,0,0,1,1,1,1,1,0,1",
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


# --- Heading-aware RRT* (Dubins + bang-bang) ---


class BangBangRRTStarTheta:
    """RRT* on (x, y, θ, v) state with Dubins steering + bang-bang speed.

    Mirrors ``BangBangRRTStar`` but adds heading θ to the state and uses
    Dubins curves for geometric paths.  All existing classes are untouched.

    Args:
        x_min, x_max, y_min, y_max: sampling bounds (continuous)
        v_max: maximum speed (cells/s)
        a_max: maximum acceleration magnitude (cells/s²)
        turning_radius: minimum turning radius in cells
        collision_fn: callable (x, y) -> bool (True = in collision)
        goal_fn: callable (x, y) -> bool (True = at goal)
        max_iter: maximum RRT* iterations
        goal_bias: probability of sampling the goal
        max_steer_dist: cap on edge length (cells)
        rewire_gamma: coefficient in the rewire radius formula
        rewire_k: cap on the number of neighbours in X_near
        seed: RNG seed
        grid: raw wall grid for DDA traversal (optional)
    """

    def __init__(self, x_min, x_max, y_min, y_max, v_max, a_max,
                 turning_radius, collision_fn, goal_fn,
                 max_iter=2000, goal_bias=0.1, max_steer_dist=4.0,
                 v_sample_low=None, v_sample_high=None,
                 rewire_gamma=2.0, rewire_k=30,
                 seed=0, verbose=False, traj_sample_step=0.05, grid=None):
        self.x_min, self.x_max = x_min, x_max
        self.y_min, self.y_max = y_min, y_max
        self.v_max = v_max
        self.a_max = a_max
        self.turning_radius = turning_radius
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
        self.grid = grid

    # ------------------------------------------------------------------ I/O

    def plan(self, start_xy, goal_xy, start_v=0.0, goal_tolerance=0.5,
             goal_v=0.0, time_budget_s=None):
        """Plan a path from start_xy to within goal_tolerance of goal_xy.

        Returns dict with 'nodes', 'goal_node', 'best_cost', 'waypoints',
        'controls', 'edges', 'trajectory'.  Or None if no path found.
        """
        from dubins_bangbang_steering import NodeTheta, steer_dubins_bangbang

        import time as _time
        t_start = _time.time()

        sx, sy = start_xy
        gx, gy = goal_xy
        start_node = NodeTheta(sx, sy, 0.0, start_v, cost=0.0)
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
                    print(f"  [bb_rrt_star_theta] time budget exhausted at iter {it}")
                break

            # --- sampling ---
            if self.rng.random() < self.goal_bias:
                x_rand = self.rng.uniform(goal_x_min, goal_x_max)
                y_rand = self.rng.uniform(goal_y_min, goal_y_max)
                theta_rand = 0.0
                v_rand = goal_v
            else:
                x_rand = self.rng.uniform(self.x_min, self.x_max)
                y_rand = self.rng.uniform(self.y_min, self.y_max)
                theta_rand = self.rng.uniform(-math.pi, math.pi)
                v_rand = self.rng.uniform(self.v_sample_low, self.v_sample_high)

            # --- nearest ---
            x_near = self._nearest(tree, x_rand, y_rand, theta_rand, v_rand)

            # --- steer ---
            dx = x_rand - x_near.x
            dy = y_rand - x_near.y
            dist = math.hypot(dx, dy)
            if dist < 1e-9:
                continue
            if dist > self.max_steer_dist:
                cos_h = dx / dist
                sin_h = dy / dist
                x_rand = x_near.x + self.max_steer_dist * cos_h
                y_rand = x_near.y + self.max_steer_dist * sin_h
                dist = self.max_steer_dist

            node_to = NodeTheta(x_rand, y_rand, theta_rand, v_rand)
            steer = steer_dubins_bangbang(
                x_near, node_to, self.turning_radius, self.a_max, self.v_max,
                samples_per_second=40.0,
            )
            if steer is None or not steer["feasible"]:
                continue

            traj_2d = steer["trajectory"]       # (N, 5) [x, y, θ, v, t]
            if not self._trajectory_collision_free(traj_2d):
                continue

            x_new = NodeTheta(
                x=x_rand, y=y_rand, theta=theta_rand, v=v_rand,
                parent=x_near, cost=x_near.cost + steer["total_time"],
                control=steer["control"],
                trajectory_2d=traj_2d,
            )
            tree.append(x_new)

            # --- rewire (choose best parent) ---
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

            # --- rewire neighbours ---
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

            # --- goal check ---
            if (abs(x_new.x - gx) <= goal_tolerance
                    and abs(x_new.y - gy) <= goal_tolerance):
                if x_new.cost < best_goal_cost:
                    best_goal_node = x_new
                    best_goal_cost = x_new.cost
                    if self.verbose:
                        print(f"  [bb_rrt_star_theta] iter {it}: "
                              f"new best goal cost = {best_goal_cost:.3f} s "
                              f"({len(tree)} nodes)")

        if best_goal_node is None:
            if self.verbose:
                print(f"  [bb_rrt_star_theta] FAILED in {len(tree)} nodes")
            return None

        waypoints, controls, edges, trajectory = self._reconstruct(best_goal_node)
        return {
            "nodes": tree,
            "goal_node": best_goal_node,
            "best_cost": best_goal_cost,
            "waypoints": waypoints,
            "controls": controls,
            "edges": edges,
            "trajectory": trajectory,
        }

    # ------------------------------------------------------ internal helpers

    def _rewire_radius(self, n):
        if n <= 1:
            return self.max_steer_dist
        d = 3.0
        log_n = math.log(n + 1.0)
        r = self.rewire_gamma * (log_n / n) ** (1.0 / d)
        return max(self.max_steer_dist, r)

    def _bb_time_metric_theta(self, n, x_q, y_q, theta_q, v_q):
        d = math.hypot(n.x - x_q, n.y - y_q)
        if d < 1e-9:
            return abs(n.v - v_q) / self.a_max
        avg_v = max((n.v + v_q) / 2.0, 0.1)
        t_travel = d / avg_v
        t_accel = abs(n.v - v_q) / self.a_max
        delta_theta = abs(math.atan2(math.sin(n.theta - theta_q),
                                     math.cos(n.theta - theta_q)))
        t_turn = self.turning_radius * delta_theta / max(avg_v, 0.1)
        return t_travel + t_accel + 0.5 * t_turn

    def _nearest(self, tree, x_q, y_q, theta_q, v_q):
        best = None
        best_d = float("inf")
        for n in tree:
            d = self._bb_time_metric_theta(n, x_q, y_q, theta_q, v_q)
            if d < best_d:
                best_d = d
                best = n
        return best

    def _near(self, tree, q_node, radius):
        time_radius = radius / max(0.1, q_node.v) + 2.0 * radius / self.a_max
        out = []
        for n in tree:
            t = self._bb_time_metric_theta(n, q_node.x, q_node.y,
                                           q_node.theta, q_node.v)
            if t <= time_radius:
                out.append((t, n))
        out.sort(key=lambda pair: pair[0])
        return [n for _, n in out[:self.rewire_k]]

    def _trajectory_collision_free(self, traj_2d):
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
        from dubins_bangbang_steering import steer_dubins_bangbang, NodeTheta
        node_to = NodeTheta(child_node.x, child_node.y, child_node.theta,
                            child_node.v)
        steer = steer_dubins_bangbang(
            parent_node, node_to, self.turning_radius, self.a_max, self.v_max,
            samples_per_second=40.0,
        )
        if steer is None or not steer["feasible"]:
            return None
        traj_2d = steer["trajectory"]
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
        # concatenate trajectories with offset
        pieces = []
        t_offset = 0.0
        for i, nn in enumerate(nodes):
            if nn.trajectory_2d is not None and nn.trajectory_2d.shape[0] > 0:
                piece = nn.trajectory_2d.copy()
                piece[:, 4] += t_offset
                if i < len(nodes) - 1 and piece.shape[0] > 1:
                    piece = piece[:-1]
                pieces.append(piece)
                t_offset += piece[-1, 4] if piece.shape[0] > 0 else 0.0
        trajectory = np.concatenate(pieces, axis=0) if pieces else np.zeros((0, 5))
        return waypoints, controls, edges, trajectory


# --- Self-test for heading-aware planner ---


def _selftest_theta():
    """Smoke test for BangBangRRTStarTheta on a small maze."""
    import os
    import time as _time
    from micromouse import load_maze, Map

    small_csv = "/tmp/small_maze_theta.csv"
    if not os.path.exists(small_csv):
        rows = [
            "1,1,1,1,1,1,1,1,1,1,1",
            "1,2,0,0,0,0,0,0,0,0,1",
            "1,0,0,0,0,0,0,0,0,0,1",
            "1,0,0,0,0,0,0,0,0,0,1",
            "1,0,0,0,0,0,0,0,0,0,1",
            "1,0,0,0,0,4,0,0,0,0,1",
            "1,0,0,0,0,0,0,0,0,0,1",
            "1,0,0,0,0,0,0,0,0,0,1",
            "1,0,0,0,0,0,0,0,0,0,1",
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

    collision_fn = make_grid_collision_fn(grid_walls, robot_radius=0.3)
    goal_fn = lambda x, y: (abs(x - gx) <= 0.5 and abs(y - gy) <= 0.5)

    # turning radius in cells: wheelbase/2 / meters_per_cell
    # For open test maze, use small turning radius
    turning_radius = 0.15  # cells

    print("=" * 60)
    print("BangBangRRTStarTheta selftest (small numeric maze)")
    print("=" * 60)
    print(f"  grid shape: {grid.shape}, start=({sx:.1f},{sy:.1f}), "
          f"goal=({gx:.1f},{gy:.1f}), turning_radius={turning_radius:.2f}")

    planner = BangBangRRTStarTheta(
        x_min=0.0, x_max=float(n_cols),
        y_min=0.0, y_max=float(n_rows),
        v_max=4.0, a_max=8.0,
        turning_radius=turning_radius,
        collision_fn=collision_fn, goal_fn=goal_fn,
        max_iter=10000, goal_bias=0.3, max_steer_dist=2.0,
        rewire_gamma=2.0, rewire_k=20, seed=42, verbose=True,
    )

    t0 = _time.time()
    result = planner.plan(
        start_xy=(sx, sy), goal_xy=(gx, gy),
        start_v=0.0, goal_tolerance=1.0, goal_v=0.0,
    )
    dt = _time.time() - t0
    if result is None:
        print("FAIL: no path found on small maze")
        return False
    print(f"OK: plan() returned in {dt:.2f}s, {len(result['nodes'])} nodes, "
          f"best_cost={result['best_cost']:.3f}s, "
          f"{len(result['waypoints'])} waypoints")

    # 1. zero collisions along final trajectory
    traj = result["trajectory"]
    n_collisions = 0
    for i in range(traj.shape[0]):
        if collision_fn(traj[i, 0], traj[i, 1]):
            n_collisions += 1
    print(f"  collisions along final trajectory: {n_collisions}/{traj.shape[0]}")
    assert n_collisions == 0, "Path has collisions!"

    # 2. goal reached
    final = result["waypoints"][-1]
    d_to_goal = math.hypot(final[0] - gx, final[1] - gy)
    print(f"  final waypoint = ({final[0]:.2f},{final[1]:.2f}), "
          f"distance to goal = {d_to_goal:.3f}")
    assert d_to_goal < 2.0, "Path does not reach goal"

    # 3. smooth heading: no jump larger than turning_radius * pi between samples
    traj = result["trajectory"]
    if traj.shape[0] > 1:
        max_jump = 0.0
        for i in range(1, traj.shape[0]):
            dtheta = abs(math.atan2(math.sin(traj[i, 2] - traj[i - 1, 2]),
                                    math.cos(traj[i, 2] - traj[i - 1, 2])))
            if dtheta > max_jump:
                max_jump = dtheta
        threshold = turning_radius * math.pi
        print(f"  max heading jump between samples: {max_jump:.4f} "
              f"(threshold: {threshold:.4f})")
        if max_jump > threshold:
            print(f"  WARNING: heading jump exceeds threshold (may be OK for large turning_radius)")

    print("PASS (selftest_theta)")
    return True


def _selftest_cspace():
    """Self-test for build_cspace_obstacle_map and make_cspace_collision_fn."""
    print("\n" + "=" * 60)
    print("selftest_cspace (C-space obstacle inflation)")
    print("=" * 60)

    import numpy as _np

    # 1. small grid with a single wall block in the centre
    grid = _np.zeros((11, 11), dtype=int)
    grid[5, 5] = 1  # single wall cell
    robot_radius = 1.0

    cspace = build_cspace_obstacle_map(grid, robot_radius)
    collision_fn = make_cspace_collision_fn(cspace)

    # 2. original wall cell is still blocked
    assert cspace[5, 5], "FAIL: original wall cell not blocked"
    print("  [1] original wall cell blocked: OK")

    # 3. cells within radius 1.0 of (5,5) are blocked
    expected_blocked = set()
    for r in range(11):
        for c in range(11):
            if grid[r, c] == 1 or ((r - 5)**2 + (c - 5)**2)**0.5 <= robot_radius:
                expected_blocked.add((r, c))

    actual_blocked = set()
    for r in range(11):
        for c in range(11):
            if cspace[r, c]:
                actual_blocked.add((r, c))

    missed = expected_blocked - actual_blocked
    extra = actual_blocked - expected_blocked
    assert not missed, f"FAIL: cells near wall not blocked: {missed}"
    assert not extra, f"FAIL: cells far from wall incorrectly blocked: {extra}"
    print(f"  [2] all {len(expected_blocked)} cells within radius {robot_radius} blocked: OK")

    # 4. a cell far from the wall is free
    assert not cspace[0, 0], "FAIL: cell (0,0) should be free"
    assert not collision_fn(0.0, 0.0), "FAIL: collision_fn(0,0) should be False"
    print("  [3] distant cell (0,0) is free: OK")

    # 5. a cell within radius is blocked
    assert cspace[5, 6], "FAIL: cell (5,6) should be blocked"
    assert collision_fn(6.0, 5.0), "FAIL: collision_fn(6,5) should be True"
    print("  [4] nearby cell (5,6) is blocked: OK")

    # 6. out-of-bounds returns True (collision)
    assert collision_fn(-1.0, 5.0), "FAIL: out-of-bounds should collide"
    assert collision_fn(5.0, 11.0), "FAIL: out-of-bounds should collide"
    print("  [5] out-of-bounds collision: OK")

    # 7. larger radius test
    grid2 = _np.zeros((21, 21), dtype=int)
    grid2[10, 10] = 1
    cspace2 = build_cspace_obstacle_map(grid2, robot_radius=3.0)
    # cell (10,13) is exactly at distance 3.0 — should be blocked (<= not <)
    assert cspace2[10, 13], "FAIL: cell at exact radius should be blocked"
    # cell (10,14) is at distance 4.0 — should be free
    assert not cspace2[10, 14], "FAIL: cell beyond radius should be free"
    print("  [6] larger radius (3.0) boundary: OK")

    print("\n[selftest_cspace] PASS")
    return True


def _selftest_continuous_collision():
    """Self-test for make_continuous_cspace_collision_fn."""
    print("\n" + "=" * 60)
    print("selftest_continuous_collision (sub-cell grazing detection)")
    print("=" * 60)

    import micromouse as _mm

    m = _mm.Map(_mm.load_maze("bench_mazes/narrow.csv"))
    grid = m.get_grid_representation()
    grid_walls = (grid == 1).astype(int)
    R = 0.28

    coll = make_continuous_cspace_collision_fn(grid_walls, R)

    # 1. grazing point: centre is in a free cell but body overlaps wall
    assert coll(2.45, 3.0), "FAIL: (2.45,3.0) should be COLLISION (0.05 from wall)"
    print("  [1] (2.45, 3.0) -> True  (grazing caught): OK")

    # 2. another grazing point
    assert coll(1.05, 4.4), "FAIL: (1.05,4.4) should be COLLISION"
    print("  [2] (1.05, 4.4) -> True  (grazing caught): OK")

    # 3. far from walls -> free
    assert not coll(7.5, 7.5), "FAIL: (7.5,7.5) should be free"
    print("  [3] (7.5, 7.5) -> False (far from walls): OK")

    # 4. inside a wall -> collision
    assert coll(0.5, 0.5), "FAIL: (0.5,0.5) is inside a wall"
    print("  [4] (0.5, 0.5) -> True  (inside wall): OK")

    # 5. point that is genuinely safe
    assert not coll(1.9, 3.8), "FAIL: (1.9,3.8) should be free (0.30 from wall)"
    print("  [5] (1.9, 3.8) -> False (correctly safe at 0.30): OK")

    # 6. out of bounds -> collision
    assert coll(-1.0, 5.0), "FAIL: out-of-bounds should collide"
    assert coll(5.0, 20.0), "FAIL: out-of-bounds should collide"
    print("  [6] out-of-bounds -> True: OK")

    # 7. verify old cell-floor check misses (2.45, 3.0) but continuous catches it
    old_fn = make_cspace_collision_fn(
        build_cspace_obstacle_map(grid_walls, R))
    assert not old_fn(2.45, 3.0), "precondition: old checker must miss this"
    assert coll(2.45, 3.0), "continuous checker must catch this"
    print("  [7] old checker misses, continuous catches: OK")

    # 8. verify d_surface value at a known point
    import numpy as _np
    from scipy.ndimage import distance_transform_edt as _dte
    free = (grid_walls == 0).astype(float)
    dist = _dte(free)
    x, y = 2.45, 3.0
    c0, r0 = 2, 3
    fx, fy = 0.45, 0.0
    d = (dist[r0, c0] * (1 - fx) * (1 - fy) +
         dist[r0, c0 + 1] * fx * (1 - fy) +
         dist[r0 + 1, c0] * (1 - fx) * fy +
         dist[r0 + 1, c0 + 1] * fx * fy)
    d_surface = d - 0.5
    print(f"  [8] d_surface at (2.45, 3.0) = {d_surface:.4f}  (< {R} -> collision): OK")
    assert d_surface < R

    print("\n[selftest_continuous_collision] PASS")
    return True


if __name__ == "__main__":
    _selftest()
    _selftest_theta()
    _selftest_cspace()
    _selftest_continuous_collision()
