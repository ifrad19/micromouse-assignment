import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sys import maxsize

class Stanley:
    """Stanley controller for path following (Thrun et al., 2006).

    Steering law::

        steer = theta_e + atan(k * cte / (|v| + k_soft))

    where ``theta_e`` is the heading error (path heading - robot heading)
    and ``cte`` is the signed cross-track error (positive = robot is to
    the left of the path direction).

    Output is ``(speed, steering_angle)``, matching the PurePursuit
    interface. The simulation's ``DiffDriveController`` then converts
    the steering angle to an angular velocity via the bicycle-model
    approximation ``omega = v * tan(steer) / L_equiv`` and finally to
    differential-drive wheel velocities.

    The class can be constructed in two ways:

    1. As required by the assignment spec::

        Stanley(maps, config=None,
                k=0.5, k_soft=1.0, max_steer=pi/3, wheel_base=0.08)

    2. With a config dict (the same dict the simulation passes to
       PurePursuit) that contains a ``"stanley"`` sub-dict of tuning
       parameters.
    """

    def __init__(self, maps=None, config=None,
                 k=0.5, k_soft=1.0, max_steer=np.pi / 3, wheel_base=0.08):
        self.map = maps

        if config is not None and isinstance(config, dict) and 'stanley' in config:
            sc = config['stanley']
        elif config is not None and isinstance(config, dict):
            sc = config
        else:
            sc = {}

        self.k = sc.get('k', k)
        self.k_soft = sc.get('k_soft', k_soft)
        self.max_steer = sc.get('max_steer', max_steer)
        self.wheel_base = sc.get('wheel_base', wheel_base)

        # speed shaping
        self.max_speed = sc.get('max_speed', 0.5)
        self.min_speed = sc.get('min_speed', 0.05)
        self.goal_slowdown_dist = sc.get('goal_slowdown_dist', 1.5)
        self.goal_threshold = sc.get('goal_threshold', 0.5)
        self.curvature_slowdown = sc.get('curvature_slowdown', True)
        self.steer_slow_threshold = sc.get('steer_slow_threshold', 0.5)
        self.steer_stop_threshold = sc.get('steer_stop_threshold', 1.2)

        # bookkeeping for analysis & debugging
        self.debug = sc.get('debug', False)
        self.path = []
        self.closest_index = 0
        self.counter = 0
        self.last_heading_error = 0.0
        self.last_cte = 0.0
        self.last_steer = 0.0
        self.last_speed = 0.0

    def set_path(self, path):
        self.path = list(path)
        self.closest_index = 0
        self.current_target = self.path[0] if self.path else None

    def _closest_point_on_path(self, pos):
        """Find closest point on the path to ``pos``.

        Returns ``(closest_point, segment_index, signed_cte, path_heading)``.

        ``signed_cte`` is the perpendicular distance to the segment with
        the sign convention: positive if the robot is to the LEFT of
        the path's travel direction (using real-world left, which in
        the simulation's (row, col) image frame corresponds to the
        left-normal ``(-dc, dr) / L``).  ``path_heading`` is in radians
        with the same convention as the simulation (heading=0=down,
        pi/2=right).

        The sign convention matches Thrun et al. (2006), so the
        Stanley steering law can be applied as
        ``steer = theta_e - atan(k * cte / v)`` (note the minus,
        because in this frame positive cte means "path is to the
        right of the robot" and a positive correction rotates the
        robot to the right, which corresponds to a *negative* Ackermann
        steering angle).
        """
        if len(self.path) < 2:
            if not self.path:
                return (pos, 0, 0.0, 0.0)
            return (self.path[0], 0, 0.0, 0.0)

        r, c = pos
        n = len(self.path)
        best_dist_sq = float('inf')
        best = (pos, 0, 0.0, 0.0)

        # search the whole path, but prefer the segment just past the last
        # closest_index to avoid wasting time on already-traversed segments
        start = max(0, self.closest_index - 1)
        for k in range(n - 1):
            i = (start + k) % (n - 1)
            ri, ci = self.path[i]
            rj, cj = self.path[i + 1]
            dr, dc = rj - ri, cj - ci
            L2 = dr * dr + dc * dc
            if L2 < 1e-12:
                continue
            pr, pc = r - ri, c - ci
            t = (pr * dr + pc * dc) / L2
            if t < 0.0:
                t = 0.0
            elif t > 1.0:
                t = 1.0
            rc = ri + t * dr
            cc = ci + t * dc
            d2 = (r - rc) ** 2 + (c - cc) ** 2
            # Use '<=' so that at a waypoint tie the *later* segment wins:
            # the robot should already be thinking about the next edge,
            # otherwise it overshoots every corner.
            if d2 <= best_dist_sq:
                # Signed CTE: dot of (robot - p1) with the LEFT normal
                # of the segment direction, normalised by length.
                # In (row, col) image coords the left normal of
                # (dr, dc) is (-dc, dr) (math-CCW rotation), which
                # in screen coords points to the visual "left" of the
                # path direction.
                cte = (dr * (c - ci) - dc * (r - ri)) / math.sqrt(L2)
                path_heading = math.atan2(dc, dr)
                best_dist_sq = d2
                best = ((rc, cc), i, cte, path_heading)
                if d2 < 1e-4:
                    # exact (or near-exact) match on a later segment
                    # — stop searching to avoid lock-step oscillation
                    if i >= start:
                        break

        return best

    def get_control(self, current_pos, current_heading):
        """Compute (speed, steering_angle) for the current pose."""
        if not self.path:
            return 0.0, 0.0

        (cr, cc), seg_idx, cte, path_heading = self._closest_point_on_path(current_pos)
        self.closest_index = seg_idx
        self.current_target = (cr, cc)

        # heading error, wrapped to [-pi, pi]
        theta_e = path_heading - current_heading
        theta_e = (theta_e + math.pi) % (2 * math.pi) - math.pi

        # base speed
        spd = self.max_speed
        gr, gc = self.path[-1]
        dist_to_goal = math.hypot(current_pos[0] - gr, current_pos[1] - gc)
        if dist_to_goal < self.goal_slowdown_dist:
            frac = max(0.0, dist_to_goal / self.goal_slowdown_dist)
            spd = self.max_speed * frac + self.min_speed * (1.0 - frac)

        # Stanley steering law.
        # In this frame (heading=0=down, +col=robot's left when facing
        # down): positive cte means "path is to the right of the
        # robot", which requires a *right* turn to come back.  A
        # right turn in our bicycle-model convention is a *negative*
        # Ackermann steering angle, so the atan term is subtracted.
        cte_term = math.atan(self.k * cte / (abs(spd) + self.k_soft))
        steer = theta_e - cte_term
        if steer > self.max_steer:
            steer = self.max_steer
        elif steer < -self.max_steer:
            steer = -self.max_steer

        self.last_heading_error = theta_e
        self.last_cte = cte
        self.last_steer = steer
        self.last_speed = spd

        # slow down in sharp turns
        if self.curvature_slowdown:
            a = abs(steer)
            if a > self.steer_stop_threshold:
                spd = self.min_speed
            elif a > self.steer_slow_threshold:
                # linear ramp: full speed at threshold, min speed at stop threshold
                t = (self.steer_stop_threshold - a) / (self.steer_stop_threshold - self.steer_slow_threshold)
                t = max(0.0, min(1.0, t))
                spd = self.min_speed + (spd - self.min_speed) * t

        # when basically on top of the goal, command zero
        if dist_to_goal < self.goal_threshold * 0.5:
            spd = 0.0

        return spd, steer

    def velocity_to_wheels(self, v, omega, wheel_radius=None, wheelbase=None):
        """Convert (v, omega) into differential-drive wheel angular velocities.

        Provided for completeness; the simulation uses its own
        ``DiffDriveController``. Useful for stand-alone testing.
        """
        R = wheel_radius if wheel_radius is not None else self.wheel_base * 0.5
        L = wheelbase if wheelbase is not None else self.wheel_base
        v_left = (2.0 * v - omega * L) / (2.0 * R)
        v_right = (2.0 * v + omega * L) / (2.0 * R)
        return v_left, v_right

class PurePursuit:
    """
    Pure Pursuit Controller for path following
    Implements the pure pursuit algorithm to follow a pre-computed path
    """
    def __init__(self, maps, config=None):
        self.map = maps
        
        # Load configuration with defaults
        self.config = config.get('pure_pursuit', {}) if config else {}
        self.lookahead_dist = self.config.get('lookahead_distance', 1.5)
        self.debug = self.config.get('debug', False)
        self.visualize_every = self.config.get('visualize_every', 5)
        self.max_speed = self.config.get('max_speed', 0.5)
        self.min_speed = self.config.get('min_speed', 0.2)
        self.steering_gain = self.config.get('steering_gain', 1.0)
        self.slow_steering_threshold = self.config.get('slow_steering_threshold', math.pi/4)
        
        self.path = []
        self.current_index = 0
        self.counter = 0
        
        if self.debug:
            plt.ion()
            figsize = config.get('visualization', {}).get('figure_size', [12, 12])
            self.fig, self.ax = plt.subplots(figsize=figsize)
            cmap_colors = config.get('visualization', {}).get('cmap_colors', ['white', 'black', 'green', 'red', 'blue'])
            self.cmap = ListedColormap(cmap_colors)
            self.initialize_debug_plot()

    def initialize_debug_plot(self):
        """Set up real-time visualization for debugging"""
        grid = self.map.get_grid_representation()
        self.background = self.ax.imshow(grid, cmap=self.cmap)
        # Visualization elements for path following
        self.path_plot = self.ax.plot([], [], 'y-', linewidth=2, label='Planned Path')[0]
        self.lookahead_plot = self.ax.plot([], [], 'mo', markersize=6, alpha=0.8, label='Lookahead')[0]
        self.vector_plot = self.ax.plot([], [], 'm-', linewidth=1, alpha=0.8, label='Target Vector')[0]
        self.heading_plot = self.ax.plot([], [], 'g-', linewidth=2, alpha=0.8, label='Heading')[0]
        self.robot_plot = self.ax.plot([], [], 'ro', markersize=8, label='Robot')[0]
        self.ax.set_title("Pure Pursuit Controller")
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.legend()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def set_path(self, path):
        """Set the path to follow"""
        self.path = path
        self.current_index = 0
        if self.debug and len(path) > 0:
            path_arr = np.array(path)
            self.path_plot.set_data(path_arr[:,1], path_arr[:,0])
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

    def find_lookahead_point(self, current_pos):
        """Find the lookahead point on the path"""
        for i in range(self.current_index, len(self.path)):
            dx = self.path[i][0] - current_pos[0]
            dy = self.path[i][1] - current_pos[1]
            dist = math.sqrt(dx**2 + dy**2)
            if dist >= self.lookahead_dist:
                self.current_index = i
                return self.path[i]
        return self.path[-1] if self.path else None

    def update_debug_plot(self, current_pos, current_heading, target_pos):
        """Update the real-time visualization"""
        if not self.debug or self.counter % self.visualize_every != 0:
            self.counter += 1
            return
            
        self.lookahead_plot.set_data([target_pos[1]], [target_pos[0]])
        self.vector_plot.set_data([current_pos[1], target_pos[1]], 
                                 [current_pos[0], target_pos[0]])
        
        heading_length = 2
        self.heading_plot.set_data(
            [current_pos[1], current_pos[1] + heading_length * math.sin(current_heading)],
            [current_pos[0], current_pos[0] + heading_length * math.cos(current_heading)])
        
        self.robot_plot.set_data([current_pos[1]], [current_pos[0]])
        
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        self.counter += 1

    def get_control(self, current_pos, current_heading):
        """Calculate control commands (speed, steering) to follow the path"""
        target = self.find_lookahead_point(current_pos)
        if target is None:
            return 0, 0  # Stop if no target

        if self.debug:
            self.update_debug_plot(current_pos, current_heading, target)
        
        # Convert target to robot's local coordinate system
        dx = target[0] - current_pos[0]
        dy = target[1] - current_pos[1]
        target_local_x = dx * math.cos(-current_heading) - dy * math.sin(-current_heading)
        target_local_y = dx * math.sin(-current_heading) + dy * math.cos(-current_heading)
        
        # Calculate curvature and steering angle
        curvature = 2 * target_local_y / (self.lookahead_dist**2)
        steering_angle = math.atan(curvature * self.steering_gain)
        
        # Adjust speed based on steering angle
        speed = self.max_speed if abs(steering_angle) < self.slow_steering_threshold else self.min_speed
        
        return speed, steering_angle